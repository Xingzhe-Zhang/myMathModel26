"""Post-hoc source, geometry, placebo, and top-recipe diagnostics for task 3.

These analyses explain the completed experiment; they do not reselect a model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from llm_resource.q1.mixture import load_pair
from llm_resource.q1.quality_mixture_14_loss import (
    fit_variant, predict_variant, source_vectors,
)
from llm_resource.q1.quality_mixture_experiment import score


def selected_params(table: pd.DataFrame, model: str) -> dict:
    row = table.set_index("model").loc[model]
    return {name: float(row[name]) for name in ("alpha", "gamma", "lambda")}


def main() -> None:
    root = CODE.parent
    out = CODE / "outputs/q1_quality_mixture_14_stage"
    config = json.loads((CODE / "configs/q1_quality_mixture_14_stage.json").read_text(encoding="utf-8"))
    gate = json.loads((out / "quality_gate.json").read_text(encoding="utf-8"))
    if not gate["bridge_loss_eligible"] or not gate["direct_loss_eligible"]:
        raise RuntimeError("Both quality routes must pass their gate")
    tables = root / "real_attachments/A_data_value/regmix_tables"
    spec = config["train"]
    train = load_pair(tables, spec["mixture"], spec["loss"], "train_1m")
    external = {
        name: load_pair(tables, item["mixture"], item["loss"], name)
        for name, item in config["external_sets"].items()
        if item["kind"] == "validation"
    }
    q_vectors, known = source_vectors(out, train.domains, gate)
    selected = pd.read_csv(out / "loss_cv_summary.csv")
    params = {source: selected_params(selected, f"M3-{source}") for source in q_vectors}
    split = pd.read_csv(out / "a4_a5_cv_folds.csv").set_index("index").loc[train.indexes]
    folds = split.fold.to_numpy(int)

    rows = []
    for source, q in q_vectors.items():
        for parameter_source, setting in params.items():
            fold_mse = []
            for fold in range(spec["folds"]):
                tr, va = folds != fold, folds == fold
                cv_fit = fit_variant(f"M3-{source}", train.p[tr], train.y[tr], q, known, setting)
                cv_pred = predict_variant(f"M3-{source}", cv_fit, train.p[va], q, setting)
                fold_mse.append(float(np.mean((train.y[va] - cv_pred) ** 2)))
            cv_rmse = float(np.sqrt(np.mean(fold_mse)))
            fitted = fit_variant(f"M3-{source}", train.p, train.y, q, known, setting)
            for dataset, pair in external.items():
                predicted = predict_variant(f"M3-{source}", fitted, pair.p, q, setting)
                metrics = score(pair.y, predicted)
                rows.append({
                    "q_source": source,
                    "parameter_source": parameter_source,
                    "dataset": dataset,
                    "parameter_policy": "frozen_from_source_specific_A4_A5_CV",
                    "cv_rmse": cv_rmse,
                    **metrics,
                })
    pd.DataFrame(rows).to_csv(out / "quality_source_geometry_control.csv", index=False,
                              encoding="utf-8-sig")

    # Match the exact training split and fixed parameters used in the main
    # M3-direct result. Permutations remove Q-to-domain alignment but preserve
    # the Q distribution. External metrics are descriptive, not used to choose.
    model = "M3-direct"
    setting = params["direct"]
    permutations = []
    rng = np.random.default_rng(config["seed"] + 30)
    for repeat in range(100):
        q = rng.permutation(q_vectors["direct"])
        fold_mse = []
        for fold in range(spec["folds"]):
            tr, va = folds != fold, folds == fold
            fitted = fit_variant(model, train.p[tr], train.y[tr], q, known, setting)
            predicted = predict_variant(model, fitted, train.p[va], q, setting)
            fold_mse.append(float(np.mean((train.y[va] - predicted) ** 2)))
        fitted = fit_variant(model, train.p, train.y, q, known, setting)
        for dataset, pair in external.items():
            predicted = predict_variant(model, fitted, pair.p, q, setting)
            metrics = score(pair.y, predicted)
            permutations.append({
                "repeat": repeat,
                "dataset": dataset,
                "cv_rmse": float(np.sqrt(np.mean(fold_mse))),
                "rmse": metrics["rmse"],
                "macro_spearman": metrics["macro_spearman"],
                "regret_at_10": metrics["regret_at_10"],
            })
    placebo = pd.DataFrame(permutations)
    placebo.to_csv(out / "quality_permutation_control.csv", index=False,
                   encoding="utf-8-sig")

    predictions = pd.read_csv(out / "macro_predictions.csv")
    top_rows = []
    for dataset in external:
        subset = predictions[predictions.dataset.eq(dataset)]
        wide = subset.pivot(index="index", columns="model", values="predicted_macro_loss")
        actual = subset.drop_duplicates("index").set_index("index").loc[wide.index, "actual_macro_loss"]
        actual_values = actual.to_numpy(float)
        oracle = int(np.argmin(actual_values))
        true_top10 = set(np.argsort(actual_values)[:10])
        true_top26 = np.argsort(actual_values)[:min(26, len(actual_values))]
        for model_name in ("M1-RBF", "M2-direct", "M3-bridge", "M3-direct"):
            predicted = wide[model_name].to_numpy(float)
            selection = np.argsort(predicted)[:10]
            best_selected = selection[np.argmin(actual_values[selection])]
            error = predicted - actual_values
            top_rows.append({
                "dataset": dataset,
                "model": model_name,
                "n": len(actual_values),
                "macro_rmse_all": float(np.sqrt(np.mean(error ** 2))),
                "macro_rmse_true_top10": float(np.sqrt(np.mean(error[list(true_top10)] ** 2))),
                "macro_rmse_true_top26": float(np.sqrt(np.mean(error[true_top26] ** 2))),
                "macro_bias_true_top10_pred_minus_actual": float(np.mean(error[list(true_top10)])),
                "predicted_rank_of_true_oracle": int(np.where(np.argsort(predicted) == oracle)[0][0] + 1),
                "best_actual_rank_among_selected10": int(np.where(np.argsort(actual_values) == best_selected)[0][0] + 1),
                "top10_overlap": len(set(selection) & true_top10) / 10,
                "regret_at_10": float(actual_values[best_selected] - actual_values[oracle]),
            })
    pd.DataFrame(top_rows).to_csv(out / "quality_top_region_diagnostic.csv", index=False,
                                  encoding="utf-8-sig")

    direct, bridge = q_vectors["direct"], q_vectors["bridge"]
    one_m, sixty_m = external["test_1m"], external["test_60m"]
    main = pd.read_csv(out / "external_metrics.csv")
    observed_cv = float(selected.set_index("model").loc["M3-direct", "cv_rmse"])
    observed_1m = float(main.query("dataset == 'test_1m' and model == 'M3-direct'").rmse.iloc[0])
    summary = {
        "status": "post_hoc_explanation_not_model_selection",
        "q_bridge_direct_pearson": float(np.corrcoef(bridge, direct)[0, 1]),
        "q_bridge_sd_original": float(pd.read_csv(out / "q_bridge_17_domains.csv").q_final.std(ddof=0)),
        "q_direct_sd_original": float(
            pd.read_csv(out / "q_direct_17_domains.csv")
            .query("scenario == 'rebuilt14_calibrated'").q_mean.std(ddof=0)
        ),
        "test_1m_and_60m_same_recipe_indexes": bool(np.array_equal(one_m.indexes, sixty_m.indexes)),
        "test_1m_and_60m_same_mixture": bool(np.allclose(one_m.p, sixty_m.p)),
        "observed_M3_direct_cv_rmse": observed_cv,
        "permutation_cv_rmse_median": float(placebo.groupby("repeat").cv_rmse.first().median()),
        "permutation_cv_fraction_better_than_observed": float(
            (placebo.groupby("repeat").cv_rmse.first() <= observed_cv).mean()
        ),
        "observed_M3_direct_test_1m_rmse": observed_1m,
        "permutation_test_1m_rmse_median": float(placebo.query("dataset == 'test_1m'").rmse.median()),
        "permutation_test_1m_fraction_better_than_observed": float(
            (placebo.query("dataset == 'test_1m'").rmse <= observed_1m).mean()
        ),
    }
    (out / "quality_explanation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
