"""Controlled predictive ablation of fixed domain-quality priors in RegMix.

Q and text proxies are domain constants. Their interactions with p change the
RBF distance metric, but contain no information independent of p. Results are
predictive inductive-bias checks, never causal quality coefficients.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import json
import lzma
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .mixture import _rank_corr, _rbf_fit, _rbf_predict, grouped_folds, load_pair

TEXT_FEATURES = ("log_token_length", "repetition_ratio", "vocabulary_richness",
                 "normalized_entropy", "special_character_ratio")


def text_features(value: str, cap: int = 2048) -> np.ndarray:
    """Deterministic, whitespace-token proxies; cap n-gram work per document."""
    words = value.casefold().split()
    token_count = len(words)
    words = words[:cap]
    n = len(words)
    triples = max(0, n - 2)
    repetition = 1 - len(set(zip(words, words[1:], words[2:]))) / triples if triples else 0.0
    counts = np.fromiter(Counter(words).values(), float) if n else np.empty(0)
    probabilities = counts / n if n else counts
    entropy = -float(np.sum(probabilities * np.log(probabilities))) / math.log(len(counts)) if len(counts) > 1 else 0.0
    special = sum(not ch.isalnum() and not ch.isspace() for ch in value) / len(value) if value else 0.0
    return np.array([math.log1p(token_count), repetition, len(counts) / n if n else 0.0,
                     entropy, special], float)


def load_text_domain_features(path: Path, domains: list[str], cap: int = 2048) -> pd.DataFrame:
    expected = set(domains)
    accum = defaultdict(lambda: np.zeros(len(TEXT_FEATURES), float))
    accum2 = defaultdict(lambda: np.zeros(len(TEXT_FEATURES), float))
    counts = defaultdict(int)
    invalid_paths = defaultdict(bool)
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            domain = record.get("_source_domain")
            value = record.get("text")
            source_path = record.get("_source_path")
            if domain not in expected or not isinstance(value, str) or not isinstance(source_path, str):
                raise ValueError(f"A18 invalid record on line {line_number}")
            invalid_paths[domain] = invalid_paths[domain] or not source_path.startswith("valid/")
            vector = text_features(value, cap)
            accum[domain] += vector
            accum2[domain] += vector * vector
            counts[domain] += 1
    rows = []
    for domain in domains:
        n = counts[domain]
        if not n:
            raise ValueError(f"A18 missing domain {domain}")
        mean = accum[domain] / n
        sd = np.sqrt(np.maximum(0, accum2[domain] / n - mean * mean))
        row = {"domain": domain, "a18_rows": n, "all_valid_shards": not invalid_paths[domain],
               "sampling_note": "A18 validation shards; not training-corpus measurements"}
        row.update({name: float(mean[j]) for j, name in enumerate(TEXT_FEATURES)})
        row.update({name + "_sd": float(sd[j]) for j, name in enumerate(TEXT_FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_domain_quality(summary_path: Path, mapping_path: Path, domains: list[str]) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    mapping = pd.read_csv(mapping_path)
    if set(mapping.mixture_domain) != set(domains) or len(mapping) != len(domains):
        raise ValueError("A16 must uniquely cover all mixture domains")
    if mapping.mixture_domain.duplicated().any():
        raise ValueError("A16 has duplicate domains")
    # A2/A3 provide larger domain-specific extensions; A1 supplies the other four.
    source = {"arxiv": "A2", "github": "A3"}
    a1_domains = set(summary.loc[summary.dataset.eq("A1"), "domain"])
    if len(a1_domains) != 7:
        raise ValueError("Expected seven A1 quality domains")
    lookup = summary.set_index(["dataset", "domain"])["Q_primary_mean"]
    observed = {}
    for qdomain in a1_domains:
        dataset = source.get(qdomain, "A1")
        observed[qdomain] = float(lookup.loc[(dataset, qdomain)])
    rows = []
    for domain in domains:
        entry = mapping.set_index("mixture_domain").loc[domain]
        qdomain = entry.quality_domain
        mapped = qdomain in observed and entry.mapping_type in {"direct", "near_direct"}
        rows.append({"domain": domain, "quality_domain": qdomain,
                     "mapping_type": entry.mapping_type, "q_observed": mapped,
                     "q_source": source.get(qdomain, "A1") if mapped else "neutral_imputation",
                     "q_official_22": observed[qdomain] if mapped else np.nan})
    frame = pd.DataFrame(rows)
    if int(frame.q_observed.sum()) != 6:
        raise ValueError("Expected six mapped quality domains; inspect A16")
    mean = float(frame.q_official_22.mean())
    sd = float(frame.q_official_22.std(ddof=0))
    if sd <= 0:
        raise ValueError("Mapped Q has no variation")
    frame["q_model_input"] = frame.q_official_22.fillna(mean)
    frame["q_centered_scaled"] = (frame.q_model_input - mean) / sd
    frame["q_imputed"] = ~frame.q_observed
    return frame


def standardized_text_matrix(frame: pd.DataFrame) -> np.ndarray:
    raw = frame.loc[:, TEXT_FEATURES].to_numpy(float)
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (raw - center) / scale


def geometry(p: np.ndarray, model: int, q: np.ndarray, text: np.ndarray,
             alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    blocks = [p]
    if model >= 2:
        blocks += [alpha * p * q, alpha * (p @ q)[:, None]]
    if model >= 3:
        blocks += [beta * (p[:, :, None] * text[None, :, :]).reshape(len(p), -1) / math.sqrt(text.shape[1]),
                   beta * (p @ text) / math.sqrt(text.shape[1])]
    return np.column_stack(blocks)


def candidates(model: int, config: dict):
    grid = config["grid"]
    for gamma, lam, alpha, beta in product(grid["gamma"], grid["lambda"],
                                            grid["alpha"] if model >= 2 else [0.0],
                                            grid["beta"] if model >= 3 else [0.0]):
        yield {"gamma": gamma, "lambda": lam, "alpha": alpha, "beta": beta}


def cv_select(p, y, q, text, config):
    folds, groups = grouped_folds(p, config["train"]["group_l1_threshold"],
                                   config["train"]["folds"], config["seed"])
    rows = []
    for model in (1, 2, 3):
        for params in candidates(model, config):
            z = geometry(p, model, q, text, params["alpha"], params["beta"])
            errors = []
            for fold in range(config["train"]["folds"]):
                tr, va = folds != fold, folds == fold
                fit = _rbf_fit(z[tr], y[tr], params["gamma"], params["lambda"])
                predicted = _rbf_predict(fit, z[va])
                errors.append(float(np.mean((y[va] - predicted) ** 2)))
            rows.append({"model": f"Model-{model}", **params,
                         "cv_mse": float(np.mean(errors)), "cv_rmse": float(np.sqrt(np.mean(errors))),
                         "fold_mse": json.dumps(errors)})
    table = pd.DataFrame(rows)
    best = table.loc[table.groupby("model").cv_mse.idxmin()].sort_values("model").reset_index(drop=True)
    return table, best, folds, groups


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = actual - predicted
    ma, mp = actual.mean(axis=1), predicted.mean(axis=1)
    denom = np.sum((actual - actual.mean(axis=0)) ** 2)
    macro_denom = np.sum((ma - ma.mean()) ** 2)
    top = min(10, len(ma))
    return {"n": len(ma), "rmse": float(np.sqrt(np.mean(err * err))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err * err) / denom) if denom else np.nan,
            "macro_rmse": float(np.sqrt(np.mean((ma - mp) ** 2))),
            "macro_mae": float(np.mean(np.abs(ma - mp))),
            "macro_r2": float(1 - np.sum((ma - mp) ** 2) / macro_denom) if macro_denom else np.nan,
            "macro_spearman": _rank_corr(ma, mp),
            "mean_output_spearman": float(np.nanmean([_rank_corr(actual[:, j], predicted[:, j])
                                                       for j in range(actual.shape[1])])),
            "top10_overlap": len(set(np.argsort(ma)[:top]) & set(np.argsort(mp)[:top])) / top,
            "selected_regret": float(ma[np.argmin(mp)] - np.min(ma))}


def effect_table(models: dict, train, q: np.ndarray, text: np.ndarray,
                 quality_frame: pd.DataFrame, delta: float = 0.005) -> pd.DataFrame:
    """Supported local replacement sensitivity, averaged across training recipes."""
    rows = []
    bounds = train.p.max(axis=0)
    base_preds = {name: predict_bundle(bundle, train.p, q, text).mean(axis=1)
                  for name, bundle in models.items()}
    for target, domain in enumerate(train.domains):
        changes, origin = [], []
        for k, p in enumerate(train.p):
            donors = np.argsort(-p)
            donor = next((int(d) for d in donors if d != target and p[d] >= delta), None)
            if donor is None or p[target] + delta > bounds[target] + 1e-12:
                continue
            changed = p.copy()
            changed[donor] -= delta
            changed[target] += delta
            changes.append(changed)
            origin.append(k)
        row = {"domain": domain, "delta_share": delta,
               "supported_recipes": len(changes),
               "official_q_observed": bool(quality_frame.loc[target, "q_observed"])}
        for name, bundle in models.items():
            if changes:
                pred = predict_bundle(bundle, np.stack(changes), q, text).mean(axis=1)
                row[name + "_delta_macro_loss"] = float(np.mean(pred - base_preds[name][origin]))
            else:
                row[name + "_delta_macro_loss"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def predict_bundle(bundle, p, q, text):
    params = bundle["params"]
    z = geometry(p, bundle["model"], q, text, params["alpha"], params["beta"])
    return _rbf_predict(bundle["fit"], z)


def run(config_path: Path, data_root: Path, summary_path: Path, out_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    root = data_root / "A_data_value"
    tables = root / "regmix_tables"
    trspec = config["train"]
    train = load_pair(tables, trspec["mixture"], trspec["loss"], "train_1m")
    external = {name: load_pair(tables, spec["mixture"], spec["loss"], name)
                for name, spec in config["external_sets"].items()}
    for pair in external.values():
        if pair.domains != train.domains or pair.loss_domains != train.loss_domains:
            raise ValueError(f"{pair.name}: domain/order mismatch")
    quality = load_domain_quality(summary_path, root / "domain_mapping_guide.csv", train.domains)
    quality.to_csv(out_dir / "quality_domain_mapping.csv", index=False, encoding="utf-8-sig")
    q = quality.q_centered_scaled.to_numpy(float)
    text_frame = load_text_domain_features(root / "regmix_domain_sample.jsonl.xz", train.domains,
                                            config["a18_token_cap"])
    text_frame.to_csv(out_dir / "a18_domain_features.csv", index=False, encoding="utf-8-sig")
    text_matrix = standardized_text_matrix(text_frame)
    cv, best, folds, groups = cv_select(train.p, train.y, q, text_matrix, config)
    cv.to_csv(out_dir / "cv_grid.csv", index=False, encoding="utf-8-sig")
    best.to_csv(out_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
    models = {}
    for _, row in best.iterrows():
        name = row["model"]
        model = int(name[-1])
        params = {key: float(row[key]) for key in ("gamma", "lambda", "alpha", "beta")}
        z = geometry(train.p, model, q, text_matrix, params["alpha"], params["beta"])
        models[name] = {"model": model, "params": params,
                        "fit": _rbf_fit(z, train.y, params["gamma"], params["lambda"])}
    metrics, per_output, predictions, rank_comparisons = [], [], [], []
    for name, pair in external.items():
        macro_predictions = {}
        for model_name, bundle in models.items():
            predicted = predict_bundle(bundle, pair.p, q, text_matrix)
            metrics.append({"dataset": name, "kind": config["external_sets"][name]["kind"],
                            "model": model_name, **score(pair.y, predicted)})
            macro_predictions[model_name] = predicted.mean(axis=1)
            for j, output in enumerate(pair.loss_domains):
                err = pair.y[:, j] - predicted[:, j]
                denom = np.sum((pair.y[:, j] - pair.y[:, j].mean()) ** 2)
                per_output.append({"dataset": name, "model": model_name, "output": output,
                                   "rmse": float(np.sqrt(np.mean(err * err))),
                                   "mae": float(np.mean(np.abs(err))),
                                   "r2": float(1 - np.sum(err * err) / denom) if denom else np.nan,
                                   "spearman": _rank_corr(pair.y[:, j], predicted[:, j])})
            for k, index in enumerate(pair.indexes):
                predictions.append({"dataset": name, "model": model_name, "index": index,
                                    "actual_macro_loss": float(pair.y[k].mean()),
                                    "predicted_macro_loss": float(predicted[k].mean())})
        baseline = macro_predictions["Model-1"]
        top = min(10, len(pair.p))
        for model_name in ("Model-2", "Model-3"):
            other = macro_predictions[model_name]
            rank_comparisons.append({"dataset": name, "comparison": "Model-1 vs " + model_name,
                                     "prediction_rank_spearman": _rank_corr(baseline, other),
                                     "prediction_top10_overlap": len(set(np.argsort(baseline)[:top]) &
                                                                     set(np.argsort(other)[:top])) / top})
    pd.DataFrame(metrics).to_csv(out_dir / "external_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_output).to_csv(out_dir / "external_output_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(out_dir / "macro_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rank_comparisons).to_csv(out_dir / "ranking_stability.csv", index=False, encoding="utf-8-sig")
    effects = effect_table(models, train, q, text_matrix, quality, config["effect_delta"])
    effects.to_csv(out_dir / "domain_effects.csv", index=False, encoding="utf-8-sig")
    metadata = {"version": config["version"], "train_rows": len(train.p),
                "validation_rows": {name: len(pair.p) for name, pair in external.items()},
                "quality_mapped_domains": int(quality.q_observed.sum()),
                "quality_neutral_imputed_domains": int(quality.q_imputed.sum()),
                "q_source": "A1-A3 official 22-signal Q_primary domain means (A2 arxiv, A3 github)",
                "a18_rows": int(text_frame.a18_rows.sum()),
                "a18_all_valid_shards": bool(text_frame.all_valid_shards.all()),
                "cv_fold_counts": np.bincount(folds).tolist(), "cv_near_duplicate_groups": int(len(np.unique(groups))),
                "model_selection": "A4-A5 only; held-out and extrapolated targets never tune models",
                "interpretation": "Fixed Q and A18 domain statistics are deterministic functions of p in feature space. Any gain is predictive regularization/geometry, not identified independent or causal quality impact.",
                "cross_scale": "No absolute-loss scale calibration; cross-scale RMSE is descriptive.",
                "extrapolation": "A12-A15 are estimated subset tables, not independent validation.",
                "text_proxy": "Whitespace tokens; first cap tokens for lexical ratios; all characters for special ratio; A18 valid-shard sample, not training corpus."}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cv": best, "metrics": pd.DataFrame(metrics), "quality": quality,
            "text": text_frame, "effects": effects, "metadata": metadata}
