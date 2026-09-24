"""Three-model comparison, conditional on validated reconstructed Q.

Both Q vectors are fixed domain profiles derived without A4-A15 labels. Their
interaction with p changes RBF geometry; no independent causal quality effect
is identified because p alone determines every p_i Q_i term.
"""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mixture import _rank_corr, _rbf_fit, _rbf_predict, grouped_folds, load_pair
from .quality_reconstruction import reconstruct
from .quality_transfer import score_transfer, similarity_transfer


def centered_quality(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    if not np.isfinite(q).all():
        raise ValueError("Quality vector contains non-finite values")
    scale = q.std()
    return (q - q.mean()) / scale if scale > 1e-10 else np.zeros_like(q)


def geometry(p: np.ndarray, model: int, q: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Use the same quality geometry for both transfer approaches."""
    p = np.asarray(p, float)
    if model == 1:
        return p.copy()
    if model not in (2, 3):
        raise ValueError("Only Model-1/2/3 are defined")
    q = np.asarray(q, float)
    if q.shape != (p.shape[1],):
        raise ValueError("Q vector must match the 17 training domains")
    return np.column_stack([p, alpha * p * q, alpha * (p @ q)[:, None]])


def candidates(model: int, config: dict):
    grid = config["grid"]
    for gamma, lam, alpha in product(grid["gamma"], grid["lambda"],
                                      [0.0] if model == 1 else grid["alpha"]):
        yield {"gamma": gamma, "lambda": lam, "alpha": alpha}


def cv_select(p, y, quality_vectors, config):
    folds, groups = grouped_folds(p, config["train"]["group_l1_threshold"],
                                   config["train"]["folds"], config["seed"])
    rows = []
    for model in (1, 2, 3):
        q = quality_vectors.get(model, np.zeros(p.shape[1]))
        for params in candidates(model, config):
            z = geometry(p, model, q, params["alpha"])
            fold_errors = []
            for fold in np.unique(folds):
                tr, va = folds != fold, folds == fold
                fit = _rbf_fit(z[tr], y[tr], params["gamma"], params["lambda"])
                fold_errors.append(float(np.mean((y[va] - _rbf_predict(fit, z[va])) ** 2)))
            rows.append({"model": f"Model-{model}", **params,
                         "cv_mse": float(np.mean(fold_errors)),
                         "cv_rmse": float(np.sqrt(np.mean(fold_errors))),
                         "fold_mse": json.dumps(fold_errors)})
    table = pd.DataFrame(rows)
    best = table.loc[table.groupby("model").cv_mse.idxmin()].sort_values("model").reset_index(drop=True)
    return table, best, folds, groups


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Loss arrays must have matching (recipe, output) shapes")
    err = actual - predicted
    ma, mp = actual.mean(axis=1), predicted.mean(axis=1)
    denominator = np.sum((actual - actual.mean(axis=0)) ** 2)
    macro_denominator = np.sum((ma - ma.mean()) ** 2)
    oracle = float(ma.min())
    result = {"n": len(ma), "rmse": float(np.sqrt(np.mean(err * err))),
              "mae": float(np.mean(np.abs(err))),
              "r2": float(1 - np.sum(err * err) / denominator) if denominator else np.nan,
              "macro_rmse": float(np.sqrt(np.mean((ma - mp) ** 2))),
              "macro_mae": float(np.mean(np.abs(ma - mp))),
              "macro_r2": float(1 - np.sum((ma - mp) ** 2) / macro_denominator) if macro_denominator else np.nan,
              "macro_spearman": _rank_corr(ma, mp),
              "mean_output_spearman": float(np.nanmean([_rank_corr(actual[:, j], predicted[:, j])
                                                         for j in range(actual.shape[1])]))}
    for k in (1, 5, 10):
        k_actual = min(k, len(ma))
        selected = np.argsort(mp)[:k_actual]
        true_top = np.argsort(ma)[:k_actual]
        result[f"top{k}_overlap"] = len(set(selected) & set(true_top)) / k_actual
        result[f"regret_at_{k}"] = float(ma[selected].min() - oracle)
    return result


def predict_bundle(bundle: dict, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    z = geometry(p, bundle["model"], q, bundle["params"]["alpha"])
    return _rbf_predict(bundle["fit"], z)


def _paired_intervals(row_mse: dict, name: str, seed: int) -> list[dict]:
    n = len(row_mse["Model-1"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(2000, n))
    rows = []
    for candidate in ("Model-2", "Model-3"):
        reference = np.sqrt(row_mse["Model-1"][draws].mean(axis=1))
        comparison = reference - np.sqrt(row_mse[candidate][draws].mean(axis=1))
        rows.append({"dataset": name, "comparison": candidate + " vs Model-1",
                     "rmse_improvement": float(np.sqrt(row_mse["Model-1"].mean()) -
                                               np.sqrt(row_mse[candidate].mean())),
                     "bootstrap_ci_low": float(np.quantile(comparison, 0.025)),
                     "bootstrap_ci_high": float(np.quantile(comparison, 0.975)),
                     "bootstrap_repeats": 2000})
    return rows


def _save(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run(config_path: Path, data_root: Path, summary_path: Path, out_dir: Path,
        sample_per_domain: int | None = None, allow_loss: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    root = data_root / "A_data_value"
    tables = root / "regmix_tables"
    spec = config["train"]
    train = load_pair(tables, spec["mixture"], spec["loss"], "train_1m")
    external = {name: load_pair(tables, row["mixture"], row["loss"], name)
                for name, row in config["external_sets"].items()}
    for pair in external.values():
        if pair.domains != train.domains or pair.loss_domains != train.loss_domains:
            raise ValueError(f"{pair.name}: domain/output order mismatch")
    sample_per_domain = sample_per_domain or config["signal_sample_per_domain"]
    try:
        features = reconstruct(root / "slimpajama_quality_signal_sample.jsonl.xz",
                               root / "regmix_domain_sample.jsonl.xz",
                               summary_path.parent / "sample_scores.csv", out_dir,
                               a1_domains=sorted(pd.read_csv(summary_path).query("dataset == 'A1'").domain.unique()),
                               regmix_domains=train.domains,
                               per_domain=sample_per_domain,
                               batch_size=config["public_models"]["batch_size"],
                               max_length=config["public_models"]["max_length"],
                               model_text_chars=config["public_models"]["max_text_chars"],
                               device=config["public_models"].get("device"))
    except (RuntimeError, OSError, ValueError) as exc:
        gate = {"passed": False, "stage": "public_signal_inference", "reason": str(exc),
                "loss_evaluation_run": False}
        (out_dir / "quality_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": gate}
    sim, sim_sensitivity, a1_profile, a18_profile, sim_validation = similarity_transfer(
        features["a1_x"], features["a1_domain"], features["a18_x"], features["a18_domain"],
        features["accepted_fields"], summary_path, root / "domain_mapping_guide.csv", train.domains,
        temperatures=tuple(config["similarity_temperatures"]))
    direct, lodo, direct_validation = score_transfer(
        features["a1_official_raw"], features["a1_rebuilt_raw"],
        features["a1_full_rebuilt_raw"], features["a1_domain"], features["a1_q"],
        features["a18_rebuilt_raw"], features["a18_domain"], train.domains,
        config_path.parent / "q1_quality.json", summary_path.parent / "fitted_scoring_model.json")
    _save(a1_profile, out_dir / "a1_feature_profiles.csv")
    _save(a18_profile, out_dir / "a18_feature_profiles.csv")
    _save(sim, out_dir / "similarity_mapping.csv")
    _save(sim_sensitivity, out_dir / "similarity_sensitivity.csv")
    _save(lodo, out_dir / "task12_q_lodo.csv")
    q_domain = sim.merge(direct, on=["domain", "a18_rows"], validate="one_to_one")
    q_domain["q_model2_standardized"] = centered_quality(q_domain.q_similarity.to_numpy(float))
    q_domain["q_model3_standardized"] = centered_quality(q_domain.q_direct.to_numpy(float))
    _save(q_domain, out_dir / "quality_domain_estimates.csv")
    gate = {"passed": bool(sim_validation["passes_gate"] and direct_validation["passes_gate"]),
            "stage": "Q_cross_domain_validation", "model2_mapping": sim_validation,
            "model3_task12_scoring": direct_validation,
            "accepted_shared_fields": features["accepted_fields"],
            "loss_requires_explicit_flag": True,
            "loss_evaluation_run": bool(sim_validation["passes_gate"] and direct_validation["passes_gate"] and allow_loss)}
    (out_dir / "quality_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    if not gate["passed"] or not allow_loss:
        return {"status": gate, "mapping": sim, "lodo": lodo}
    quality_vectors = {1: np.zeros(17),
                       2: q_domain.q_model2_standardized.to_numpy(float),
                       3: q_domain.q_model3_standardized.to_numpy(float)}
    cv, best, folds, groups = cv_select(train.p, train.y, quality_vectors, config)
    _save(cv, out_dir / "cv_grid.csv")
    _save(best, out_dir / "cv_summary.csv")
    models = {}
    for _, row in best.iterrows():
        model = int(row.model[-1])
        params = {key: float(row[key]) for key in ("gamma", "lambda", "alpha")}
        z = geometry(train.p, model, quality_vectors[model], params["alpha"])
        models[row.model] = {"model": model, "params": params,
                             "fit": _rbf_fit(z, train.y, params["gamma"], params["lambda"])}
    metric_rows, output_rows, prediction_rows, rank_rows, interval_rows = [], [], [], [], []
    for name, pair in external.items():
        macro_predictions, row_mse = {}, {}
        for model_name, bundle in models.items():
            predicted = predict_bundle(bundle, pair.p, quality_vectors[bundle["model"]])
            metric_rows.append({"dataset": name, "kind": config["external_sets"][name]["kind"],
                                "model": model_name, **score(pair.y, predicted)})
            macro_predictions[model_name] = predicted.mean(axis=1)
            row_mse[model_name] = np.mean((pair.y - predicted) ** 2, axis=1)
            for j, output in enumerate(pair.loss_domains):
                err = pair.y[:, j] - predicted[:, j]
                denom = np.sum((pair.y[:, j] - pair.y[:, j].mean()) ** 2)
                output_rows.append({"dataset": name, "model": model_name, "output": output,
                                    "rmse": float(np.sqrt(np.mean(err * err))),
                                    "mae": float(np.mean(np.abs(err))),
                                    "r2": float(1 - np.sum(err * err) / denom) if denom else np.nan,
                                    "spearman": _rank_corr(pair.y[:, j], predicted[:, j])})
            prediction_rows.extend({"dataset": name, "model": model_name, "index": index,
                                    "actual_macro_loss": float(pair.y[k].mean()),
                                    "predicted_macro_loss": float(predicted[k].mean())}
                                   for k, index in enumerate(pair.indexes))
        baseline = macro_predictions["Model-1"]
        for model_name in ("Model-2", "Model-3"):
            other = macro_predictions[model_name]
            row = {"dataset": name, "comparison": "Model-1 vs " + model_name,
                   "prediction_rank_spearman": _rank_corr(baseline, other)}
            for k in (1, 5, 10):
                actual_k = min(k, len(pair.p))
                row[f"prediction_top{k}_overlap"] = len(set(np.argsort(baseline)[:actual_k]) &
                                                        set(np.argsort(other)[:actual_k])) / actual_k
            rank_rows.append(row)
        if config["external_sets"][name]["kind"] != "extrapolated_subset":
            interval_rows.extend(_paired_intervals(row_mse, name, config["seed"] + len(interval_rows)))
    metrics = pd.DataFrame(metric_rows)
    intervals = pd.DataFrame(interval_rows)
    _save(metrics, out_dir / "external_metrics.csv")
    _save(pd.DataFrame(output_rows), out_dir / "external_output_metrics.csv")
    _save(pd.DataFrame(prediction_rows), out_dir / "macro_predictions.csv")
    _save(pd.DataFrame(rank_rows), out_dir / "ranking_stability.csv")
    _save(intervals, out_dir / "paired_rmse_intervals.csv")
    decisions = []
    for model_name, gate in (("Model-2", sim_validation["passes_gate"]),
                             ("Model-3", direct_validation["passes_gate"])):
        same_scale = metrics[(metrics.dataset == "test_1m") & (metrics.model == model_name)].iloc[0]
        baseline = metrics[(metrics.dataset == "test_1m") & (metrics.model == "Model-1")].iloc[0]
        ci = intervals[(intervals.dataset == "test_1m") &
                       (intervals.comparison == model_name + " vs Model-1")].iloc[0]
        rank_ok = all(
            metrics[(metrics.dataset == dataset) & (metrics.model == model_name)].iloc[0].macro_spearman >=
            metrics[(metrics.dataset == dataset) & (metrics.model == "Model-1")].iloc[0].macro_spearman - 0.02
            for dataset in ("test_60m", "test_1B"))
        decisions.append({"model": model_name, "q_transfer_validation_passed": bool(gate),
                          "test_1m_rmse_improved": bool(same_scale.rmse < baseline.rmse),
                          "test_1m_paired_ci_positive": bool(ci.bootstrap_ci_low > 0),
                          "cross_scale_rank_not_materially_worse": bool(rank_ok),
                          "stable_gain": bool(gate and ci.bootstrap_ci_low > 0 and rank_ok),
                          "decision_scope": "A6-A11 only; A12-A15 are estimated subset diagnostics"})
    _save(pd.DataFrame(decisions), out_dir / "gain_decision.csv")
    metadata = {"version": config["version"], "train_rows": len(train.p),
                "a1_text_rows": len(features["a1_x"]), "a18_text_rows": len(features["a18_x"]),
                "shared_features": features["accepted_fields"],
                "signal_sample_per_domain": sample_per_domain,
                "similarity_validation": sim_validation, "direct_q_validation": direct_validation,
                "cv_fold_counts": np.bincount(folds).tolist(),
                "cv_near_duplicate_groups": int(len(np.unique(groups))),
                "quality_identification": "Fixed Q_i and p_i Q_i are functions of p; only predictive inductive-bias effects are tested.",
                "data_roles": "A1 learns Q transfer; A16 validates similarity; A18 supplies valid-shard texts; A4+A5 select Loss models; A6-A11 validate; A12-A15 estimated subset only.",
                "cross_scale": "No absolute-Loss scale calibration is applied."}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cv": best, "metrics": metrics, "decisions": pd.DataFrame(decisions),
            "mapping": sim, "lodo": lodo, "metadata": metadata}
