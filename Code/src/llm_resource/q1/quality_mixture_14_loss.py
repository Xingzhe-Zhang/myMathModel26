"""Gated fourteen-signal task-3 Loss experiment on fixed RegMix splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mixture import _rank_corr, _rbf_fit, _rbf_predict, grouped_folds, load_pair
from .quality_mixture_experiment import centered_quality, score
from .quality_mixture_staged import _paired_delta_ci, _prior_ridge_fit, _prior_ridge_predict


def _save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def geometry(p: np.ndarray, model: str, q: np.ndarray | None, alpha: float) -> np.ndarray:
    if model == "M1-RBF":
        return p.copy()
    if q is None or q.shape != (p.shape[1],):
        raise ValueError("A quality model needs one fixed Q for every training domain")
    if model.startswith("M2-"):
        return np.column_stack((p, alpha * (p @ q)))
    if model.startswith("M3-") and not model.startswith("M3prior-"):
        return np.column_stack((p, alpha * p * q))
    raise ValueError(f"Unknown RBF model {model}")


def fit_variant(model: str, p: np.ndarray, y: np.ndarray,
                q: np.ndarray | None, known: np.ndarray, params: dict) -> dict:
    if model == "M1-Scheffe" or model.startswith("M3prior-"):
        return _prior_ridge_fit(p, y, np.zeros(p.shape[1]) if q is None else q, known,
                                float(params["lambda"]), float(params["interaction_penalty"]),
                                float(params["prior_strength"]))
    z = geometry(p, model, q, float(params["alpha"]))
    return _rbf_fit(z, y, float(params["gamma"]), float(params["lambda"]))


def predict_variant(model: str, fitted: dict, p: np.ndarray,
                    q: np.ndarray | None, params: dict) -> np.ndarray:
    if model == "M1-Scheffe" or model.startswith("M3prior-"):
        return _prior_ridge_predict(fitted, p)
    return _rbf_predict(fitted, geometry(p, model, q, float(params["alpha"])))


def parameter_grid(model: str, grid: dict) -> list[dict]:
    if model == "M1-Scheffe" or model.startswith("M3prior-"):
        strengths = [0.0] if model == "M1-Scheffe" else grid["prior_strength"]
        return [{"lambda": lam, "interaction_penalty": penalty, "prior_strength": strength}
                for lam in grid["ridge_lambda"] for penalty in grid["interaction_penalty"]
                for strength in strengths]
    alphas = [0.0] if model == "M1-RBF" else grid["alpha"]
    return [{"alpha": alpha, "gamma": gamma, "lambda": lam}
            for alpha in alphas for gamma in grid["gamma"] for lam in grid["lambda"]]


def source_vectors(out_dir: Path, domains: list[str], gate: dict) -> tuple[dict, np.ndarray]:
    bridge = pd.read_csv(out_dir / "q_bridge_17_domains.csv").set_index("domain").loc[domains]
    known = bridge.a16_fixed.to_numpy(bool).astype(float)
    if int(known.sum()) != 6:
        raise ValueError("Six A16 links must remain fixed")
    vectors = {}
    if gate["bridge_loss_eligible"]:
        vectors["bridge"] = centered_quality(bridge.q_final.to_numpy(float))
    if gate["direct_loss_eligible"]:
        direct = pd.read_csv(out_dir / "q_direct_17_domains.csv")
        direct = direct[direct.scenario.eq("rebuilt14_calibrated")].set_index("domain").loc[domains]
        vectors["direct"] = centered_quality(direct.q_mean.to_numpy(float))
    return vectors, known


def _model_source(model: str) -> str | None:
    if model.endswith("-bridge"):
        return "bridge"
    if model.endswith("-direct"):
        return "direct"
    return None


def run_loss_phase(config: dict, data_root: Path, out_dir: Path) -> dict:
    gate = json.loads((out_dir / "quality_gate.json").read_text(encoding="utf-8"))
    if gate["version"] != config["version"] or not gate["a1_signal_passed"]:
        raise ValueError("Quality phase/version must finish before Loss phase")
    tables = data_root / "A_data_value/regmix_tables"
    train_spec = config["train"]
    train = load_pair(tables, train_spec["mixture"], train_spec["loss"], "train_1m")
    external = {name: load_pair(tables, spec["mixture"], spec["loss"], name)
                for name, spec in config["external_sets"].items()}
    for pair in external.values():
        if pair.domains != train.domains or pair.loss_domains != train.loss_domains:
            raise ValueError(f"{pair.name}: training and validation columns differ")
    q_vectors, known = source_vectors(out_dir, train.domains, gate)
    variants = ["M1-RBF", "M1-Scheffe"]
    for source in ("bridge", "direct"):
        if source in q_vectors:
            variants.extend((f"M2-{source}", f"M3-{source}", f"M3prior-{source}"))
    folds, groups = grouped_folds(train.p, train_spec["group_l1_threshold"],
                                  train_spec["folds"], config["seed"])
    _save(pd.DataFrame({"index": train.indexes, "fold": folds, "near_duplicate_group": groups}),
          out_dir / "a4_a5_cv_folds.csv")
    grid = config["loss_grid"]
    cv_rows = []
    for model in variants:
        q = q_vectors.get(_model_source(model))
        print(f"A4+A5 CV: {model}", flush=True)
        for params in parameter_grid(model, grid):
            fold_mse = []
            for fold in range(train_spec["folds"]):
                tr, va = folds != fold, folds == fold
                fitted = fit_variant(model, train.p[tr], train.y[tr], q, known, params)
                predicted = predict_variant(model, fitted, train.p[va], q, params)
                fold_mse.append(float(np.mean((train.y[va] - predicted) ** 2)))
            cv_rows.append({"model": model, "source": _model_source(model) or "none",
                            "alpha": params.get("alpha", np.nan), "gamma": params.get("gamma", np.nan),
                            "lambda": params["lambda"],
                            "interaction_penalty": params.get("interaction_penalty", np.nan),
                            "prior_strength": params.get("prior_strength", np.nan),
                            "cv_mse": float(np.mean(fold_mse)),
                            "cv_rmse": float(np.sqrt(np.mean(fold_mse))),
                            "fold_mse": json.dumps(fold_mse)})
    cv = pd.DataFrame(cv_rows)
    best = cv.loc[cv.groupby("model").cv_mse.idxmin()].sort_values("model").reset_index(drop=True)
    _save(cv, out_dir / "loss_cv_grid.csv")
    _save(best, out_dir / "loss_cv_summary.csv")
    rbf_best = best[~best.model.str.contains("Scheffe|prior")]
    chosen = rbf_best.sort_values("cv_mse").iloc[0]
    fitted_models = {}
    for _, row in best.iterrows():
        model = row["model"]
        params = {key: float(row[key]) for key in ("alpha", "gamma", "lambda", "interaction_penalty", "prior_strength")
                  if pd.notna(row[key])}
        fitted_models[model] = (params, fit_variant(model, train.p, train.y,
                                                   q_vectors.get(_model_source(model)), known, params))

    metric_rows, output_rows, prediction_rows, ranking_rows, interval_rows = [], [], [], [], []
    rng = np.random.default_rng(config["seed"])
    for dataset, pair in external.items():
        print(f"External check: {dataset}", flush=True)
        predictions = {}
        row_mse = {}
        for model, (params, fitted) in fitted_models.items():
            pred = predict_variant(model, fitted, pair.p, q_vectors.get(_model_source(model)), params)
            predictions[model] = pred
            row_mse[model] = np.mean((pair.y - pred) ** 2, axis=1)
            metrics = score(pair.y, pred)
            metric_rows.append({"dataset": dataset, "kind": config["external_sets"][dataset]["kind"],
                                "model": model, "source": _model_source(model) or "none",
                                "selected_by_A4_A5_CV": model == chosen.model, **metrics})
            ranking_rows.append({"dataset": dataset, "model": model,
                                 "macro_spearman": metrics["macro_spearman"],
                                 **{f"top{k}_overlap": metrics[f"top{k}_overlap"] for k in (1, 5, 10)},
                                 **{f"regret_at_{k}": metrics[f"regret_at_{k}"] for k in (1, 5, 10)}})
            for j, output in enumerate(pair.loss_domains):
                actual = pair.y[:, j]
                error = actual - pred[:, j]
                denom = np.sum((actual - actual.mean()) ** 2)
                output_rows.append({"dataset": dataset, "model": model, "output": output,
                                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                                    "mae": float(np.mean(np.abs(error))),
                                    "r2": float(1 - np.sum(error ** 2) / denom) if denom else np.nan,
                                    "spearman": _rank_corr(actual, pred[:, j])})
            prediction_rows.extend({"dataset": dataset, "model": model, "index": int(index),
                                    "actual_macro_loss": float(pair.y[i].mean()),
                                    "predicted_macro_loss": float(pred[i].mean())}
                                   for i, index in enumerate(pair.indexes))
            if config["external_sets"][dataset]["kind"] == "validation" and model not in ("M1-RBF", "M1-Scheffe"):
                baseline = "M1-Scheffe" if model.startswith("M3prior-") else "M1-RBF"
                # Both baseline fits are already in fitted_models, independent of iteration order.
                base_params, base_fit = fitted_models[baseline]
                base_pred = predict_variant(baseline, base_fit, pair.p, None, base_params)
                base_error = np.mean((pair.y - base_pred) ** 2, axis=1)
                interval_rows.append({"dataset": dataset, "model": model, "baseline": baseline,
                                      **_paired_delta_ci(base_error, row_mse[model], rng,
                                                        repeats=1000)})
        base_rank = np.argsort(predictions["M1-RBF"].mean(axis=1))
        for model, pred in predictions.items():
            if model == "M1-RBF":
                continue
            current_rank = np.argsort(pred.mean(axis=1))
            row = next(row for row in ranking_rows if row["dataset"] == dataset and row["model"] == model)
            row["prediction_rank_spearman_vs_M1"] = _rank_corr(predictions["M1-RBF"].mean(axis=1), pred.mean(axis=1))
            for k in (1, 5, 10):
                take = min(k, len(pair.p))
                row[f"prediction_top{k}_overlap_vs_M1"] = len(set(base_rank[:take]) & set(current_rank[:take])) / take

    _save(pd.DataFrame(metric_rows), out_dir / "external_metrics.csv")
    _save(pd.DataFrame(output_rows), out_dir / "external_output_metrics.csv")
    _save(pd.DataFrame(prediction_rows), out_dir / "macro_predictions.csv")
    _save(pd.DataFrame(ranking_rows), out_dir / "ranking_regret.csv")
    _save(pd.DataFrame(interval_rows), out_dir / "paired_gain_intervals.csv")
    _save(pd.DataFrame([{"dataset": name, "source": source,
                         "mean_recipe_mass_in_six_A16_anchors": float((pair.p @ known).mean()),
                         "min_recipe_mass_in_six_A16_anchors": float((pair.p @ known).min())}
                        for name, pair in external.items() for source in q_vectors]),
          out_dir / "a16_recipe_coverage.csv")

    sensitivity_rows = []
    bridge_table = pd.read_csv(out_dir / "q_bridge_17_domains.csv").set_index("domain").loc[train.domains]
    source_scenarios = {}
    quality_source_columns = {
        "q_sensitivity_A22_original_union",
        "q_sensitivity_B23_original_union",
        "q_sensitivity_B20_rank_union",
        "q_sensitivity_B20_original_A1",
    }
    if "bridge" in q_vectors:
        for label in ("hard_top1", "hard_top2"):
            source_scenarios[label] = centered_quality(bridge_table[f"q_{label}"].to_numpy(float))
        for column in bridge_table.columns:
            if column in quality_source_columns:
                source_scenarios[column.removeprefix("q_sensitivity_")] = centered_quality(bridge_table[column].to_numpy(float))
        mapping_sensitivity = pd.read_csv(out_dir / "mapping_and_sample_sensitivity.csv")
        for tau in config["temperature_sensitivity"]:
            selected = mapping_sensitivity[(mapping_sensitivity.scenario == "main") &
                                           (np.isclose(mapping_sensitivity.temperature, tau))]
            source_scenarios[f"temperature_{tau}"] = centered_quality(
                selected.set_index("domain").loc[train.domains, "q_bridge"].to_numpy(float))
    for source, scenarios in (("bridge", source_scenarios),
                              ("direct", {"uncalibrated_public_models": centered_quality(
                                  pd.read_csv(out_dir / "q_direct_17_domains.csv").query("scenario == 'rebuilt14_raw'")
                                  .set_index("domain").loc[train.domains, "q_mean"].to_numpy(float))}
                               if "direct" in q_vectors else {})):
        model = f"M3-{source}"
        if model not in fitted_models:
            continue
        params = fitted_models[model][0]
        for scenario, q_alt in scenarios.items():
            alt_fit = fit_variant(model, train.p, train.y, q_alt, known, params)
            for dataset, pair in external.items():
                pred = predict_variant(model, alt_fit, pair.p, q_alt, params)
                sensitivity_rows.append({"scenario": scenario, "quality_route": source,
                                         "model": model, "dataset": dataset,
                                         "kind": config["external_sets"][dataset]["kind"],
                                         "hyperparameters": "frozen_from_main_A4_A5_CV",
                                         **score(pair.y, pred)})
    _save(pd.DataFrame(sensitivity_rows), out_dir / "quality_source_loss_sensitivity.csv")
    extrapolation = pd.DataFrame(metric_rows)
    extrapolation = extrapolation[extrapolation.kind.eq("extrapolated_subset")].copy()
    train_indexes = set(train.indexes)
    extrapolation["is_A4_recipe_subset"] = extrapolation.dataset.map(
        {name: bool(set(pair.indexes).issubset(train_indexes)) for name, pair in external.items()})
    extrapolation["loss_status"] = "estimated_power_law_not_independent_large_model_measurement"
    _save(extrapolation, out_dir / "extrapolation_diagnostics.csv")
    metadata = {"version": config["version"], "train_rows": len(train.p),
                "cv_groups": int(len(np.unique(groups))), "cv_fold_sizes": np.bincount(folds).tolist(),
                "models_run": variants, "chosen_RBF_by_A4_A5_CV": str(chosen.model),
                "chosen_RBF_cv_rmse": float(chosen.cv_rmse), "quality_gate": gate,
                "quality_effect_identification": "Fixed 17-domain Q is a deterministic prior on p; no independent quality treatment effect is identified.",
                "A6_A11": "held-out actual Loss evaluation only",
                "A12_A15": "estimated subset/extrapolation diagnosis only",
                "sensitivity_hyperparameters": "frozen from main A4+A5 CV; no external Loss used for selection"}
    (out_dir / "loss_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chosen_RBF_by_A4_A5_CV": str(chosen.model),
            "models_run": variants, "external_rows": len(metric_rows)}
