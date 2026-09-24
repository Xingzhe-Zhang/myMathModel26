"""Task 3: mixture proportions to validation-loss modelling.

The module keeps the experimental roles separate: A4+A5 is the only model
selection set, A6--A11 are held-out checks, and A12--A15 are reported as
extrapolation/subset diagnostics.  A16--A18 inform quality-bridge diagnostics,
but no fixed domain quality score is inserted into the primary response model:
the corpora are independent and p @ q is determined by the mixture itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .quality_evidence import audit_regmix_text


@dataclass
class MixturePair:
    name: str
    p: np.ndarray
    y: np.ndarray
    indexes: np.ndarray
    domains: list[str]
    loss_domains: list[str]
    audit: dict


def _domain_from_mixture_column(column: str) -> str:
    return column.removeprefix("train_the_pile_")


def _domain_from_loss_column(column: str) -> str:
    return column.removeprefix("metric/the_pile_").removesuffix("_val_loss")


def load_pair(root: Path, mixture_file: str, loss_file: str, name: str,
              sum_tolerance=(0.995, 1.005)) -> MixturePair:
    """Read and explicitly join one mixture/loss pair by index."""
    mixture = pd.read_csv(root / mixture_file)
    loss = pd.read_csv(root / loss_file)
    if "index" not in mixture.columns or "index" not in loss.columns:
        raise ValueError(f"{name}: both tables require an index column")
    if mixture["index"].duplicated().any() or loss["index"].duplicated().any():
        raise ValueError(f"{name}: index is not unique")
    if set(mixture["index"]) != set(loss["index"]):
        raise ValueError(f"{name}: mixture and loss index sets do not match")
    mix_cols = [c for c in mixture.columns if c != "index"]
    loss_cols = [c for c in loss.columns if c != "index"]
    domains = [_domain_from_mixture_column(c) for c in mix_cols]
    loss_domains = [_domain_from_loss_column(c) for c in loss_cols]
    # Four of the 17 training domains have no separate validation-loss column;
    # the 13 loss domains must nevertheless be a subset of the mixture domains.
    if not set(loss_domains).issubset(set(domains)) or len(domains) != 17 or len(loss_cols) != 13:
        raise ValueError(f"{name}: expected 17 mixture and 13 validation-loss columns")
    mixture = mixture.set_index("index").sort_index()
    loss = loss.set_index("index").loc[mixture.index]
    raw_p = mixture[mix_cols].to_numpy(float)
    # Preserve the file's 13 validation-domain order.
    y = loss[loss_cols].to_numpy(float)
    row_sum = raw_p.sum(axis=1)
    finite = np.isfinite(raw_p).all(axis=1) & np.isfinite(y).all(axis=1)
    nonnegative = (raw_p >= 0).all(axis=1)
    positive_loss = (y > 0).all(axis=1)
    in_tolerance = (row_sum >= sum_tolerance[0]) & (row_sum <= sum_tolerance[1])
    if not finite.all() or not nonnegative.all() or not positive_loss.all() or not in_tolerance.all():
        bad = np.flatnonzero(~(finite & nonnegative & positive_loss & in_tolerance))[:10]
        raise ValueError(f"{name}: invalid rows at positions {bad.tolist()}")
    p = raw_p / row_sum[:, None]
    audit = {
        "name": name, "mixture_file": mixture_file, "loss_file": loss_file,
        "rows": len(p), "mixture_columns": len(mix_cols), "loss_columns": len(loss_cols),
        "unique_indexes": int(mixture.index.nunique()),
        "raw_sum_min": float(row_sum.min()), "raw_sum_max": float(row_sum.max()),
        "max_normalization_adjustment": float(np.max(np.abs(p - raw_p))),
        "negative_rows": int((~nonnegative).sum()), "nonfinite_rows": int((~finite).sum()),
        "nonpositive_loss_rows": int((~positive_loss).sum()),
        "out_of_sum_tolerance_rows": int((~in_tolerance).sum()),
        "exact_duplicate_proportions": int(pd.DataFrame(p).duplicated().sum()),
        "normalization": "row-wise divide by raw sum; sums are rounded in source",
    }
    return MixturePair(name, p, y, mixture.index.to_numpy(), domains, loss_domains, audit)


def mixture_features(p: np.ndarray, quadratic: bool) -> np.ndarray:
    p = np.asarray(p, float)
    if not quadratic:
        return p.copy()
    n, d = p.shape
    pairs = [p[:, i] * p[:, j] for i in range(d) for j in range(i + 1, d)]
    return np.column_stack([p] + pairs)


def feature_names(d=17):
    names = [f"p_{i}" for i in range(d)]
    names.extend(f"p_{i}__p_{j}" for i in range(d) for j in range(i + 1, d))
    return names


def _ridge_fit(x, y, lam, interaction_penalty=1.0, quadratic=False):
    x = mixture_features(x, quadratic)
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-12] = 1.0
    xs = x / scale
    penalty = np.ones(x.shape[1])
    if quadratic:
        penalty[17:] = float(interaction_penalty)
    lhs = xs.T @ xs + float(lam) * len(x) * np.diag(penalty)
    try:
        beta = np.linalg.solve(lhs, xs.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ xs.T @ y
    return {"family": "quadratic_ridge" if quadratic else "linear_ridge",
            "lambda": float(lam), "interaction_penalty": float(interaction_penalty),
            "quadratic": bool(quadratic), "scale": scale.tolist(),
            "beta": beta.tolist(), "intercept": False}


def _ridge_predict(model, x):
    f = mixture_features(x, model["quadratic"])
    return (f / np.asarray(model["scale"])) @ np.asarray(model["beta"])


def _rbf_fit(x, y, gamma, lam):
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    upper = d[np.triu_indices(len(x), 1)]
    bandwidth = float(np.median(upper)) if len(upper) else 1.0
    bandwidth = max(bandwidth, 1e-12)
    k = np.exp(-float(gamma) * d / bandwidth)
    mean = y.mean(axis=0)
    try:
        alpha = np.linalg.solve(k + float(lam) * len(x) * np.eye(len(x)), y - mean)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(k + float(lam) * len(x) * np.eye(len(x))) @ (y - mean)
    return {"family": "rbf_kernel_ridge", "gamma": float(gamma), "lambda": float(lam),
            "bandwidth": bandwidth, "x_train": x.tolist(), "alpha": alpha.tolist(),
            "mean": mean.tolist(), "intercept": "training_mean_baseline"}


def _rbf_predict(model, x):
    train = np.asarray(model["x_train"])
    d = ((x[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-model["gamma"] * d / max(model["bandwidth"], 1e-12))
    return np.asarray(model["mean"]) + k @ np.asarray(model["alpha"])


def fit_model(family, x, y, params):
    if family == "linear_ridge":
        return _ridge_fit(x, y, params["lambda"], quadratic=False)
    if family == "quadratic_ridge":
        return _ridge_fit(x, y, params["lambda"], params["interaction_penalty"], quadratic=True)
    if family == "rbf_kernel_ridge":
        return _rbf_fit(x, y, params["gamma"], params["lambda"])
    raise ValueError(f"Unknown model family: {family}")


def predict_model(model, x):
    return _rbf_predict(model, x) if model["family"] == "rbf_kernel_ridge" else _ridge_predict(model, x)


def grouped_folds(p, threshold, folds, seed):
    n = len(p)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.abs(p[:, None, :] - p[None, :, :]).sum(axis=2)
    for i, j in zip(*np.where(np.triu(d <= threshold, 1))):
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            parent[ri] = rj
    groups = np.asarray([root(i) for i in range(n)])
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_for = {int(g): i % folds for i, g in enumerate(unique)}
    fold = np.asarray([fold_for[int(g)] for g in groups])
    return fold, groups


def _mse(actual, pred):
    return float(np.mean((actual - pred) ** 2))


def cross_validate(p, y, config):
    folds, groups = grouped_folds(p, config["train"]["group_l1_threshold"],
                                   config["train"]["folds"], config["seed"])
    records = []
    families = config["models"]["families"]
    for family in families:
        if family == "linear_ridge":
            grid = [{"lambda": v} for v in config["models"]["lambda_grid"]]
        elif family == "quadratic_ridge":
            grid = [{"lambda": lam, "interaction_penalty": ratio}
                    for ratio in config["models"]["quadratic_interaction_penalty_grid"]
                    for lam in config["models"]["lambda_grid"]]
        elif family == "rbf_kernel_ridge":
            grid = [{"gamma": gamma, "lambda": lam}
                    for gamma in config["models"]["rbf_gamma_grid"]
                    for lam in config["models"]["lambda_grid"]]
        else:
            raise ValueError(f"Unsupported family {family}")
        for params in grid:
            fold_scores = []
            for fold_id in range(config["train"]["folds"]):
                tr, va = folds != fold_id, folds == fold_id
                model = fit_model(family, p[tr], y[tr], params)
                fold_scores.append(_mse(y[va], predict_model(model, p[va])))
            records.append({"family": family, **params, "cv_mse": float(np.mean(fold_scores)),
                            "cv_rmse": float(np.sqrt(np.mean(fold_scores))),
                            "fold_mse": json.dumps(fold_scores)})
    table = pd.DataFrame(records).sort_values(["family", "cv_mse"]).reset_index(drop=True)
    best = table.loc[table.groupby("family")["cv_mse"].idxmin()].copy()
    chosen = best.sort_values("cv_mse").iloc[0].to_dict()
    return table, best, chosen, folds, groups


def _rank_corr(a, b):
    a = pd.Series(a).rank(method="average").to_numpy(float)
    b = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(actual, predicted, weights=None):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    weights = np.ones(actual.shape[1]) / actual.shape[1] if weights is None else np.asarray(weights)
    ma = actual @ weights
    mp = predicted @ weights
    selected = int(np.argmin(mp))
    oracle = int(np.argmin(ma))
    return {
        "n": len(actual), "output_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "output_mae": float(np.mean(np.abs(actual - predicted))),
        "macro_loss_rmse": float(np.sqrt(np.mean((ma - mp) ** 2))),
        "macro_loss_mae": float(np.mean(np.abs(ma - mp))),
        "macro_loss_spearman": _rank_corr(ma, mp),
        "mean_output_spearman": float(np.nanmean([_rank_corr(actual[:, k], predicted[:, k]) for k in range(actual.shape[1])])),
        "selected_regret": float(ma[selected] - ma[oracle]),
        "selected_index": selected,
        "oracle_index": oracle,
        "top10_overlap": float(len(set(np.argsort(ma)[:10]) & set(np.argsort(mp)[:10])) / 10),
    }


def per_output_metrics(actual, predicted, dataset, family, output_names=None):
    output_names = output_names or list(range(actual.shape[1]))
    rows = []
    for k in range(actual.shape[1]):
        err = actual[:, k] - predicted[:, k]
        denom = np.sum((actual[:, k] - actual[:, k].mean()) ** 2)
        rows.append({"dataset": dataset, "family": family, "output": output_names[k],
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "mae": float(np.mean(np.abs(err))),
                     "r2": float(1 - np.sum(err ** 2) / denom) if denom > 1e-12 else np.nan,
                     "spearman": _rank_corr(actual[:, k], predicted[:, k])})
    return rows


def replacement_effects(model, p, domains, delta, bootstrap_repeats, rng, weights, output_names=None):
    output_names = output_names or list(range(p.shape[1]))
    rows = []
    for a in range(p.shape[1]):
        for b in range(p.shape[1]):
            if a == b:
                continue
            keep = p[:, b] >= delta
            if not keep.any():
                continue
            shifted = p[keep].copy()
            shifted[:, a] += delta
            shifted[:, b] -= delta
            # Feasibility on the simplex does not imply empirical support.
            # A target share beyond its observed maximum is an extrapolation.
            within_target_range = shifted[:, a] <= p[:, a].max() + 1e-12
            effects = predict_model(model, shifted) - predict_model(model, p[keep])
            agg = effects @ weights
            boots = []
            for _ in range(bootstrap_repeats):
                ids = rng.integers(0, len(effects), len(effects))
                boots.append(effects[ids].mean(axis=0) @ weights)
            for k in range(effects.shape[1]):
                rows.append({"from_domain": domains[b], "to_domain": domains[a], "from_index": b,
                             "to_index": a, "output": output_names[k], "delta": delta, "n_support": int(keep.sum()),
                             "n_within_target_observed_range": int(within_target_range.sum()),
                             "fraction_within_target_observed_range": float(within_target_range.mean()),
                             "all_target_shares_extrapolated": bool(not within_target_range.any()),
                             "mean_delta_loss": float(effects[:, k].mean()),
                             "aggregate_mean_delta_loss": float(agg.mean()),
                             "aggregate_ci_low": float(np.quantile(boots, .025)),
                             "aggregate_ci_high": float(np.quantile(boots, .975)),
                             "interpretation": "negative lowers predicted loss"})
    return pd.DataFrame(rows)


def combination_effects(model, p, indexes, weights, sample_size, rng, output_names=None):
    output_names = output_names or list(range(p.shape[1]))
    n = len(p)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > sample_size:
        pairs = [pairs[i] for i in rng.choice(len(pairs), sample_size, replace=False)]
    i = np.asarray([x[0] for x in pairs], int)
    j = np.asarray([x[1] for x in pairs], int)
    mid = .5 * p[i] + .5 * p[j]
    h = predict_model(model, mid) - .5 * (predict_model(model, p[i]) + predict_model(model, p[j]))
    agg = h @ weights
    summary = []
    for k in range(h.shape[1]):
        summary.append({"output": output_names[k], "n_pairs": len(h), "mean_H": float(h[:, k].mean()),
                        "p10_H": float(np.quantile(h[:, k], .1)), "median_H": float(np.median(h[:, k])),
                        "p90_H": float(np.quantile(h[:, k], .9)), "min_H": float(h[:, k].min()),
                        "max_H": float(h[:, k].max())})
    summary.append({"output": "macro_mean", "n_pairs": len(h), "mean_H": float(agg.mean()),
                    "p10_H": float(np.quantile(agg, .1)), "median_H": float(np.median(agg)),
                    "p90_H": float(np.quantile(agg, .9)), "min_H": float(agg.min()), "max_H": float(agg.max())})
    order = np.argsort(agg)
    top = []
    for q in list(order[:10]) + list(order[-10:]):
        top.append({"index_a": int(indexes[i[q]]), "index_b": int(indexes[j[q]]),
                    "row_a": int(i[q]), "row_b": int(j[q]), "macro_H": float(agg[q]),
                    "interpretation": "negative indicates midpoint below endpoint interpolation"})
    return pd.DataFrame(summary), pd.DataFrame(top)


def quality_bridge(q1_summary_path: Path, mapping_path: Path, a17_path: Path,
                   a18_path: Path, p, domains, indexes):
    """Build a transparent A1--A3/A16 bridge and audit A17--A18 separately.

    A17/A18 do not contain the 22 quality signals, so no quality score is
    inferred from text length or other structural diagnostics.
    """
    mapping = pd.read_csv(mapping_path)
    summary = pd.read_csv(q1_summary_path)
    if not {"dataset", "domain", "Q_primary_mean"}.issubset(summary):
        raise ValueError("Q1 domain summary lacks the required quality columns")
    a1 = summary[summary["dataset"] == "A1"]
    q_by_domain = dict(zip(a1["domain"], a1["Q_primary_mean"]))
    q_ci_low = dict(zip(a1["domain"], a1["ci_low_primary"])) if "ci_low_primary" in a1 else q_by_domain
    q_ci_high = dict(zip(a1["domain"], a1["ci_high_primary"])) if "ci_high_primary" in a1 else q_by_domain
    if not q_by_domain:
        raise ValueError("Q1 domain summary has no A1 reference scores")
    extension = summary[summary["dataset"].isin(["A2", "A3"])]
    q_extension = dict(zip(extension["domain"], extension["Q_primary_mean"]))
    mapping = mapping.rename(columns={"mixture_domain": "domain"})
    mapping["q_mapped"] = mapping["quality_domain"].map(q_by_domain)
    mapping["q_mapped_ci_low"] = mapping["quality_domain"].map(q_ci_low)
    mapping["q_mapped_ci_high"] = mapping["quality_domain"].map(q_ci_high)
    mapping["q_extension"] = mapping["quality_domain"].map(q_extension)
    mapping["q_extension_difference"] = mapping["q_extension"] - mapping["q_mapped"]
    mapped = mapping["q_mapped"].dropna()
    global_q = float(mapped.mean()) if len(mapped) else np.nan
    if not np.isfinite(global_q):
        raise ValueError("A16 has no quality domain present in the Q1 A1 summary")
    mapping["q_sensitivity_fill"] = mapping["q_mapped"].fillna(global_q)
    evidence, evidence_meta = audit_regmix_text(mapping_path, a17_path, a18_path, domains)
    rows = []
    domain_to_row = mapping.set_index("domain").to_dict("index")
    q_mapped = []
    q_low = []
    q_high = []
    q_fill = []
    q_extension_sensitivity = []
    for d in domains:
        item = domain_to_row[d]
        q_mapped.append(item["q_mapped"])
        q_low.append(item["q_mapped_ci_low"])
        q_high.append(item["q_mapped_ci_high"])
        q_fill.append(item["q_sensitivity_fill"])
        q_extension_sensitivity.append(item["q_extension"] if np.isfinite(item["q_extension"])
                                       else item["q_sensitivity_fill"])
        rows.append({"domain": d, **item})
    q_mapped = np.asarray(q_mapped, float); q_fill = np.asarray(q_fill, float)
    q_low = np.asarray(q_low, float); q_high = np.asarray(q_high, float)
    mapped_mass = p @ np.isfinite(q_mapped).astype(float)
    contribution = p @ np.nan_to_num(q_mapped, nan=0.0)
    recipe = pd.DataFrame({"index": indexes, "q_mix_known_contribution": contribution,
                           "q_mix_on_mapped_mass": np.divide(contribution, mapped_mass,
                                                              out=np.full(len(p), np.nan), where=mapped_mass > 0),
                           "q_mix_sensitivity_fill": p @ q_fill,
                           "q_mix_extension_sensitivity": p @ np.asarray(q_extension_sensitivity, float),
                           "q_mix_lower_if_unmapped_zero": contribution,
                           "q_mix_upper_if_unmapped_one": contribution + 1 - mapped_mass,
                           "q_mix_lower_with_a1_ci": p @ np.nan_to_num(q_low, nan=0.0),
                           "q_mix_upper_with_a1_ci": p @ np.nan_to_num(q_high, nan=0.0) + 1 - mapped_mass,
                           "mapped_mass": mapped_mass})
    domain = pd.DataFrame(rows).merge(evidence.drop(columns=["mapping_type", "quality_domain"]),
                                      on="domain", validate="one_to_one")
    meta = {"mapped_domains": int(np.isfinite(q_mapped).sum()),
            "mapping_types": {str(k): int(v) for k, v in mapping["mapping_type"].value_counts().items()},
            "total_domains": len(domains), "global_q_sensitivity": global_q,
            "mean_mapped_mass_in_train": float(mapped_mass.mean()),
            "a17_a18": evidence_meta,
            "note": "A16 quality transfer is hypothetical; A17/A18 are text diagnostics, not Q. "
                    "The full-mixture Q values are sensitivity scenarios only; Q is excluded from the primary loss model."}
    return domain, recipe, meta, evidence


def extrapolation_diagnostics(train, pair, model, weights):
    d = np.abs(pair.p[:, None, :] - train.p[None, :, :]).sum(axis=2)
    match = d.argmin(axis=1)
    exact = d[np.arange(len(pair.p)), match] < 1e-12
    source = train.y[match]
    ratio = np.divide(pair.y, source, out=np.full_like(pair.y, np.nan), where=np.abs(source) > 1e-12)
    recipe_scale = np.nanmedian(ratio, axis=1)
    pred_domain_scaled = source * np.nanmedian(ratio, axis=0)
    pred_recipe_scaled = source * recipe_scale[:, None]
    rows = []
    for k in range(pair.y.shape[1]):
        rows.append({"dataset": pair.name, "output": k, "exact_recipe_matches": int(exact.sum()),
                     "rows": len(pair.p), "median_ratio": float(np.nanmedian(ratio[:, k])),
                     "ratio_p10": float(np.nanquantile(ratio[:, k], .1)),
                     "ratio_p90": float(np.nanquantile(ratio[:, k], .9)),
                     "domain_scaled_rmse": float(np.sqrt(np.mean((pair.y[:, k] - pred_domain_scaled[:, k]) ** 2))),
                     "recipe_scaled_rmse": float(np.sqrt(np.mean((pair.y[:, k] - pred_recipe_scaled[:, k]) ** 2))),
                     "interpretation": "subset/extrapolation consistency, not independent validation"})
    return pd.DataFrame(rows)
