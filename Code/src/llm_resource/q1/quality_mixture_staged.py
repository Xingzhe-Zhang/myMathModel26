"""Nine-signal quality transfer and nested Q1 task-3 mixture comparisons.

The six teacher-provided A16 mappings are immutable. Nine validated common
signals infer the remaining eleven mappings, with per-domain uncertainty.
All Loss models select parameters on A4+A5 alone. A12-A15 are diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mixture import _rbf_fit, _rbf_predict, _rank_corr, grouped_folds, load_pair, mixture_features
from .quality_mixture_experiment import centered_quality, score
from .quality_reconstruction import reconstruct
from .quality_transfer import score_transfer, similarity_transfer


def _save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def complete_a16_mapping(sim: pd.DataFrame, sensitivity: pd.DataFrame,
                         summary_path: Path) -> pd.DataFrame:
    """Keep provided links and fill only the eleven inferred links by top-1 similarity."""
    summary = pd.read_csv(summary_path)
    a1 = summary.loc[summary.dataset.eq("A1"), ["domain", "Q_primary_mean"]]
    q_lookup = dict(zip(a1.domain, a1.Q_primary_mean))
    rows = []
    for record in sim.itertuples(index=False):
        provided = bool(record.a16_known)
        assigned = record.a16_quality_domain if provided else record.top1_quality_domain
        if assigned not in q_lookup or record.top2_quality_domain not in q_lookup:
            raise ValueError(f"No A1 Q for inferred mapping of {record.domain}")
        if provided and (record.a16_mapping_type not in {"direct", "near_direct"} or
                         not np.isclose(q_lookup[assigned], record.a16_reference_q)):
            raise ValueError(f"Teacher-provided A16 mapping changed: {record.domain}")
        variants = sensitivity.loc[(sensitivity.domain.eq(record.domain)) &
                                   (sensitivity.omitted_field.ne("none")) &
                                   (np.isclose(sensitivity.temperature, 1.0))]
        stability = float(variants.top1_quality_domain.eq(record.top1_quality_domain).mean())
        confidence = "teacher_provided" if provided else (
            "limited" if record.top1_probability < .35 or record.distance_margin < .25 or stability < .75
            else "moderate")
        rows.append({"domain": record.domain,
                     "a16_original_type": record.a16_mapping_type,
                     "a16_original_quality_domain": record.a16_quality_domain,
                     "assigned_quality_domain": assigned,
                     "assignment_source": "teacher_provided" if provided else "nine_signal_top1_inference",
                     "q_assigned": float(q_lookup[assigned]),
                     "top1_quality_domain": record.top1_quality_domain,
                     "top2_quality_domain": record.top2_quality_domain,
                     "top1_probability": float(record.top1_probability),
                     "distance_margin": float(record.distance_margin),
                     "leave_one_signal_top1_stability": stability,
                     "q_second_choice": float(q_lookup[record.top2_quality_domain]),
                     "q_soft_similarity": float(record.q_similarity),
                     "q_soft_sensitivity_min": float(record.q_sensitivity_min),
                     "q_soft_sensitivity_max": float(record.q_sensitivity_max),
                     "confidence_note": confidence,
                     "a18_sample_rows": int(record.a18_rows)})
    result = pd.DataFrame(rows)
    if len(result) != 17 or result.domain.duplicated().any() or result.assignment_source.eq("teacher_provided").sum() != 6:
        raise ValueError("Completed A16 mapping must have six fixed and eleven inferred rows")
    return result


def run_quality_phase(config: dict, data_root: Path, summary_path: Path,
                      out_dir: Path, domains: list[str]) -> dict:
    """Rebuild only the A1-audited RedPajama statistics; never infer public models."""
    root = data_root / "A_data_value"
    features = reconstruct(
        root / "slimpajama_quality_signal_sample.jsonl.xz",
        root / "regmix_domain_sample.jsonl.xz",
        summary_path.parent / "sample_scores.csv", out_dir,
        a1_domains=sorted(pd.read_csv(summary_path).query("dataset == 'A1'").domain.unique()),
        regmix_domains=domains, per_domain=int(config["signal_sample_per_domain"]),
        batch_size=1, max_length=1, model_text_chars=1,
        include_public_models=False,
    )
    expected = set(config["accepted_nine_fields"])
    accepted = set(features["accepted_fields"])
    if accepted != expected:
        raise ValueError(f"Nine-signal gate failed: expected {sorted(expected)}, got {sorted(accepted)}")
    sim, sensitivity, a1_profile, a18_profile, sim_validation = similarity_transfer(
        features["a1_x"], features["a1_domain"],
        features["a18_x"], features["a18_domain"],
        features["accepted_fields"], summary_path,
        root / "domain_mapping_guide.csv", domains,
        temperatures=tuple(config["similarity_temperatures"]),
    )
    direct, lodo, q_validation = score_transfer(
        features["a1_official_raw"], features["a1_rebuilt_raw"],
        features["a1_full_rebuilt_raw"], features["a1_domain"],
        features["a1_q"], features["a18_rebuilt_raw"],
        features["a18_domain"], domains,
        out_dir.parent.parent / "configs/q1_quality.json",
        summary_path.parent / "fitted_scoring_model.json",
    )
    estimates = sim.merge(direct, on=["domain", "a18_rows"], validate="one_to_one")
    completed = complete_a16_mapping(sim, sensitivity, summary_path)
    estimates = estimates.merge(completed[["domain", "assigned_quality_domain", "assignment_source",
                                           "q_assigned", "confidence_note"]], on="domain", validate="one_to_one")
    estimates["q_a16_completed_centered"] = centered_quality(estimates.q_assigned.to_numpy(float))
    estimates["q_similarity_centered"] = centered_quality(estimates.q_similarity.to_numpy(float))
    estimates["q_direct_centered"] = centered_quality(estimates.q_direct.to_numpy(float))
    _save(a1_profile, out_dir / "a1_feature_profiles.csv")
    _save(a18_profile, out_dir / "a18_feature_profiles.csv")
    _save(sim, out_dir / "a16_similarity_audit.csv")
    _save(sensitivity, out_dir / "a16_similarity_sensitivity.csv")
    _save(completed, out_dir / "completed_a16_mapping.csv")
    _save(lodo, out_dir / "task12_q_lodo.csv")
    _save(estimates, out_dir / "quality_domain_estimates.csv")
    gate = {
        "stage": "nine_statistics_only",
        "sample_per_domain": int(config["signal_sample_per_domain"]),
        "accepted_fields": features["accepted_fields"],
        "a1_signal_passed": True,
        "a16_similarity_diagnostic": sim_validation,
        "a18_task12_q_lodo": q_validation,
        "a18_similarity_q_passed": bool(sim_validation["passes_gate"]),
        "a18_direct_task12_q_passed": bool(q_validation["passes_gate"]),
        "a16_known_domains": 6,
        "a16_inferred_domains": 11,
        "a16_completed_mapping": True,
        "a16_inferred_limited_confidence": int(completed.confidence_note.eq("limited").sum()),
        "public_model_outputs_used": False,
    }
    (out_dir / "quality_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate


def quality_geometry(p: np.ndarray, model: str, q: np.ndarray,
                     alpha: float) -> np.ndarray:
    """Nested RBF representations; fixed Q only changes the model's geometry."""
    p = np.asarray(p, float)
    if model == "M1-p":
        return p.copy()
    if model.endswith(("A16", "A18")):
        weighted = p * q
        if model.startswith("M2"):
            return np.column_stack((p, alpha * weighted.sum(axis=1)))
        if model.startswith("M3"):
            return np.column_stack((p, alpha * weighted))
    raise ValueError(f"Unknown staged model: {model}")


def _quality_vectors(estimates: pd.DataFrame,
                     domains: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimates = estimates.set_index("domain").loc[domains].reset_index()
    known = estimates.a16_known.to_numpy(bool).astype(float)
    if int(known.sum()) != 6:
        raise ValueError("A16 must contain exactly six direct/near-direct anchors")
    if estimates.assignment_source.eq("teacher_provided").sum() != 6:
        raise ValueError("Six A16 teacher mappings must remain fixed")
    if estimates.assignment_source.eq("nine_signal_top1_inference").sum() != 11:
        raise ValueError("Eleven A16 mappings must be inferred from nine signals")
    q_map = estimates.q_a16_completed_centered.to_numpy(float)
    q_sim = estimates.q_similarity_centered.to_numpy(float)
    if not np.isfinite(q_sim).all() or not np.isfinite(q_map).all():
        raise ValueError("Both Q approaches must cover all 17 domains")
    return q_map, q_sim, known


def _paired_delta_ci(baseline_error: np.ndarray, candidate_error: np.ndarray,
                     rng: np.random.Generator, repeats: int = 1000) -> dict:
    n = len(baseline_error)
    draws = rng.integers(0, n, size=(repeats, n))
    delta = np.sqrt(baseline_error[draws].mean(axis=1)) - np.sqrt(candidate_error[draws].mean(axis=1))
    return {"rmse_improvement": float(np.sqrt(baseline_error.mean()) - np.sqrt(candidate_error.mean())),
            "ci_low": float(np.quantile(delta, .025)), "ci_high": float(np.quantile(delta, .975)),
            "bootstrap_repeats": repeats}


def _prior_ridge_fit(p: np.ndarray, y: np.ndarray, q: np.ndarray, known: np.ndarray,
                     lam: float, interaction_penalty: float, prior_strength: float) -> dict:
    """Quadratic Scheffe ridge with linear-domain effects shrunk toward a+bQ.

    The projection penalty constrains effects only where Q is supported. With
    prior_strength=0 this is the identical quality-free quadratic baseline.
    """
    x = mixture_features(p, quadratic=True)
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-12] = 1.0
    xs = x / scale
    penalty = np.ones(x.shape[1])
    penalty[17:] = interaction_penalty
    matrix = xs.T @ xs + lam * len(p) * np.diag(penalty)
    if prior_strength > 0:
        chosen = np.flatnonzero(known)
        if len(chosen) < 3 or np.std(q[chosen]) < 1e-10:
            raise ValueError("Quality prior needs at least three variable anchor scores")
        basis = np.column_stack([np.ones(len(chosen)), q[chosen]])
        residual_projection = np.eye(len(chosen)) - basis @ np.linalg.pinv(basis)
        inverse_scale = 1 / scale[chosen]
        matrix[np.ix_(chosen, chosen)] += (prior_strength * len(p) *
                                           inverse_scale[:, None] * residual_projection * inverse_scale[None, :])
    beta = np.linalg.solve(matrix, xs.T @ y)
    return {"scale": scale, "beta": beta}


def _prior_ridge_predict(model: dict, p: np.ndarray) -> np.ndarray:
    return (mixture_features(p, quadratic=True) / model["scale"]) @ model["beta"]


def run_loss_phase(config: dict, data_root: Path, summary_path: Path,
                   out_dir: Path, *, require_quality_gate: bool = True) -> dict:
    """Fit only on A4+A5; confirm on A6-A11 and diagnose A12-A15."""
    gate = json.loads((out_dir / "quality_gate.json").read_text(encoding="utf-8"))
    if int(gate.get("sample_per_domain", config["signal_sample_per_domain"])) != int(config["signal_sample_per_domain"]):
        raise ValueError("Quality and Loss phases use different text sample sizes")
    if require_quality_gate and not (gate["a18_similarity_q_passed"] and gate["a16_completed_mapping"]):
        raise ValueError("Nine-signal mapping gate failed; Loss phase is gated")
    root = data_root / "A_data_value/regmix_tables"
    spec = config["train"]
    train = load_pair(root, spec["mixture"], spec["loss"], "train_1m")
    external = {name: load_pair(root, item["mixture"], item["loss"], name)
                for name, item in config["external_sets"].items()}
    for pair in external.values():
        if pair.domains != train.domains or pair.loss_domains != train.loss_domains:
            raise ValueError(f"{pair.name}: input or output domain order changed")
    estimates = pd.read_csv(out_dir / "quality_domain_estimates.csv")
    q_map, q_sim, known = _quality_vectors(estimates, train.domains)
    variants = ["M1-p", "M1-quadratic", "M2-A16", "M2-A18",
                "M3-A16", "M3-A18", "M3prior-A16", "M3prior-A18"]
    folds, groups = grouped_folds(train.p, spec["group_l1_threshold"], spec["folds"], config["seed"])
    grid = config["loss_grid"]
    cv_rows = []
    for variant in variants:
        q = q_map if variant.endswith("A16") else q_sim
        if variant == "M1-quadratic" or variant.startswith("M3prior"):
            strengths = [0.0] if variant == "M1-quadratic" else grid["prior_strength"]
            for lam in grid["ridge_lambda"]:
                for interaction_penalty in grid["interaction_penalty"]:
                    for prior_strength in strengths:
                        fold_mse = []
                        for fold in range(spec["folds"]):
                            tr, va = folds != fold, folds == fold
                            model = _prior_ridge_fit(train.p[tr], train.y[tr], q,
                                                     np.ones(17),
                                                     lam, interaction_penalty, prior_strength)
                            prediction = _prior_ridge_predict(model, train.p[va])
                            fold_mse.append(float(np.mean((train.y[va] - prediction) ** 2)))
                        cv_rows.append({"model": variant, "alpha": np.nan, "gamma": np.nan,
                                        "lambda": lam, "interaction_penalty": interaction_penalty,
                                        "prior_strength": prior_strength,
                                        "cv_mse": float(np.mean(fold_mse)),
                                        "cv_rmse": float(np.sqrt(np.mean(fold_mse))),
                                        "fold_mse": json.dumps(fold_mse)})
            continue
        alphas = [0.0] if variant == "M1-p" else grid["alpha"]
        for alpha in alphas:
            z = quality_geometry(train.p, variant, q, alpha)
            for gamma in grid["gamma"]:
                for lam in grid["lambda"]:
                    fold_mse = []
                    for fold in range(spec["folds"]):
                        tr, va = folds != fold, folds == fold
                        model = _rbf_fit(z[tr], train.y[tr], gamma, lam)
                        prediction = _rbf_predict(model, z[va])
                        fold_mse.append(float(np.mean((train.y[va] - prediction) ** 2)))
                    cv_rows.append({"model": variant, "alpha": alpha, "gamma": gamma,
                                    "lambda": lam, "cv_mse": float(np.mean(fold_mse)),
                                    "cv_rmse": float(np.sqrt(np.mean(fold_mse))),
                                    "fold_mse": json.dumps(fold_mse)})
    cv = pd.DataFrame(cv_rows)
    best = cv.loc[cv.groupby("model").cv_mse.idxmin()].sort_values("model").reset_index(drop=True)
    _save(cv, out_dir / "loss_cv_grid.csv")
    _save(best, out_dir / "loss_cv_summary.csv")
    eligible = best
    chosen = eligible.sort_values("cv_mse").iloc[0]
    fitted = {}
    for _, row in best.iterrows():
        q = q_map if row["model"].endswith("A16") else q_sim
        if row["model"] == "M1-quadratic" or row["model"].startswith("M3prior"):
            fit = _prior_ridge_fit(train.p, train.y, q,
                                   np.ones(17),
                                   float(row["lambda"]), float(row["interaction_penalty"]),
                                   float(row["prior_strength"]))
            fitted[row["model"]] = (row, fit)
            continue
        z = quality_geometry(train.p, row["model"], q, float(row["alpha"]))
        fitted[row["model"]] = (row, _rbf_fit(z, train.y, float(row["gamma"]), float(row["lambda"])))
    metric_rows, output_rows, prediction_rows, ranking_rows, interval_rows, coverage_rows = [], [], [], [], [], []
    rng = np.random.default_rng(config["seed"])
    for name, pair in external.items():
        predictions = {}
        row_errors = {}
        coverage = pair.p @ known
        coverage_rows.append({"dataset": name, "mean_mapped_mass": float(coverage.mean()),
                              "min_mapped_mass": float(coverage.min()),
                              "max_mapped_mass": float(coverage.max()),
                              "coverage_meaning": "share of recipe in six teacher-provided domains; all 17 have assigned Q"})
        for variant, (row, fitted_model) in fitted.items():
            q = q_map if variant.endswith("A16") else q_sim
            if variant == "M1-quadratic" or variant.startswith("M3prior"):
                pred = _prior_ridge_predict(fitted_model, pair.p)
            else:
                z = quality_geometry(pair.p, variant, q, float(row["alpha"]))
                pred = _rbf_predict(fitted_model, z)
            predictions[variant] = pred
            row_errors[variant] = np.mean((pair.y - pred) ** 2, axis=1)
            metric_rows.append({"dataset": name, "kind": config["external_sets"][name]["kind"],
                                "model": variant, "selected_by_train_cv": variant == chosen.model,
                                **score(pair.y, pred)})
            for j, output in enumerate(pair.loss_domains):
                actual = pair.y[:, j]
                error = actual - pred[:, j]
                denom = np.sum((actual - actual.mean()) ** 2)
                output_rows.append({"dataset": name, "model": variant, "output": output,
                                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                                    "mae": float(np.mean(np.abs(error))),
                                    "r2": float(1 - np.sum(error ** 2) / denom) if denom else np.nan,
                                    "spearman": _rank_corr(actual, pred[:, j])})
            prediction_rows.extend({"dataset": name, "model": variant, "index": index,
                                    "actual_macro_loss": float(pair.y[i].mean()),
                                    "predicted_macro_loss": float(pred[i].mean())}
                                   for i, index in enumerate(pair.indexes))
        base_rank = np.argsort(predictions["M1-p"].mean(axis=1))
        for variant in variants[1:]:
            other_rank = np.argsort(predictions[variant].mean(axis=1))
            row = {"dataset": name, "model": variant,
                   "prediction_rank_spearman": _rank_corr(predictions["M1-p"].mean(axis=1),
                                                           predictions[variant].mean(axis=1))}
            for k in (1, 5, 10):
                row[f"prediction_top{k}_overlap"] = len(set(base_rank[:k]) & set(other_rank[:k])) / k
            ranking_rows.append(row)
            if config["external_sets"][name]["kind"] == "validation":
                interval_rows.append({"dataset": name, "model": variant,
                                      **_paired_delta_ci(row_errors["M1-p"], row_errors[variant], rng)})
    metrics = pd.DataFrame(metric_rows)
    intervals = pd.DataFrame(interval_rows)
    _save(metrics, out_dir / "external_metrics.csv")
    _save(pd.DataFrame(output_rows), out_dir / "external_output_metrics.csv")
    _save(pd.DataFrame(prediction_rows), out_dir / "macro_predictions.csv")
    _save(pd.DataFrame(ranking_rows), out_dir / "ranking_stability.csv")
    _save(intervals, out_dir / "paired_rmse_intervals.csv")
    _save(pd.DataFrame(coverage_rows), out_dir / "a16_recipe_coverage.csv")
    completed = pd.read_csv(out_dir / "completed_a16_mapping.csv").set_index("domain").loc[train.domains]
    inferred = completed.assignment_source.eq("nine_signal_top1_inference").to_numpy()
    limited = completed.confidence_note.eq("limited").to_numpy()
    sensitivity_rows = []
    for scenario, changed in (("top2_limited_only", inferred & limited),
                              ("top2_all_inferred", inferred)):
        q_alt_raw = completed.q_assigned.to_numpy(float).copy()
        q_alt_raw[changed] = completed.q_second_choice.to_numpy(float)[changed]
        q_alt = centered_quality(q_alt_raw)
        for variant in ("M2-A16", "M3-A16", "M3prior-A16"):
            row = fitted[variant][0]
            if variant.startswith("M3prior"):
                alt_fit = _prior_ridge_fit(train.p, train.y, q_alt, np.ones(17),
                                           float(row["lambda"]), float(row["interaction_penalty"]),
                                           float(row["prior_strength"]))
            else:
                alt_z = quality_geometry(train.p, variant, q_alt, float(row["alpha"]))
                alt_fit = _rbf_fit(alt_z, train.y, float(row["gamma"]), float(row["lambda"]))
            for name, pair in external.items():
                if variant.startswith("M3prior"):
                    prediction = _prior_ridge_predict(alt_fit, pair.p)
                else:
                    alt_z = quality_geometry(pair.p, variant, q_alt, float(row["alpha"]))
                    prediction = _rbf_predict(alt_fit, alt_z)
                sensitivity_rows.append({"scenario": scenario, "changed_inferred_domains": int(changed.sum()),
                                         "model": variant, "dataset": name,
                                         "kind": config["external_sets"][name]["kind"],
                                         "hyperparameters": "frozen_from_main_A4_A5_CV",
                                         **score(pair.y, prediction)})
    _save(pd.DataFrame(sensitivity_rows), out_dir / "a16_mapping_sensitivity_metrics.csv")
    metadata = {"version": config["version"], "train_rows": len(train.p),
                "signal_sample_per_domain": int(config["signal_sample_per_domain"]),
                "cv_groups": int(len(np.unique(groups))),
                "cv_fold_sizes": np.bincount(folds).tolist(),
                "chosen_by_A4_A5_cv": str(chosen.model),
                "chosen_cv_rmse": float(chosen.cv_rmse),
                "a16_route_scope": "Six teacher mappings fixed; eleven mappings inferred by nine-signal top-1 similarity.",
                "a16_limited_confidence_inferred_domains": int((inferred & limited).sum()),
                "a16_sensitivity": "Top-2 alternatives refit with main CV hyperparameters frozen; no validation labels select a mapping.",
                "a18_route_scope": "Similarity profile Q from nine shared signals; task1/2 direct Q fails LODO and is excluded.",
                "interpretation": "Fixed domain Q changes feature geometry; independent quality effects are not identified.",
                "A6_A11": "held-out validation, not parameter selection",
                "A12_A15": "estimated subset/extrapolation diagnostics, not independent validation"}
    (out_dir / "loss_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cv": best, "metrics": metrics, "intervals": intervals, "metadata": metadata}
