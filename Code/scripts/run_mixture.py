"""Run Q1 task 3: 17-domain mixture proportions and 13 validation losses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from llm_resource.q1.mixture import (  # noqa: E402
    combination_effects, cross_validate, evaluate, extrapolation_diagnostics,
    feature_names, fit_model, load_pair, per_output_metrics, predict_model,
    quality_bridge, replacement_effects,
)


def _args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Run Q1 task 3 mixture modelling")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def run(config_path: Path, data_root: Path, out_dir: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        import lightgbm  # noqa: F401
        lightgbm_available = True
    except ImportError:
        lightgbm_available = False
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = data_root / "A_data_value" / "regmix_tables"
    train = load_pair(table_dir, config["train"]["mixture"], config["train"]["loss"], "train_1m")
    external = {}
    for name, spec in config["external_sets"].items():
        external[name] = load_pair(table_dir, spec["mixture"], spec["loss"], name)

    audits = [train.audit] + [x.audit for x in external.values()]
    pd.DataFrame(audits).to_csv(out_dir / "mixture_audit.csv", index=False, encoding="utf-8-sig")
    domains = train.domains
    weights = np.ones(train.y.shape[1]) / train.y.shape[1]

    cv_table, best_by_family, chosen, folds, groups = cross_validate(train.p, train.y, config)
    cv_table.to_csv(out_dir / "model_cv_grid.csv", index=False, encoding="utf-8-sig")
    best_by_family["selected_for_prediction"] = best_by_family["family"].eq(chosen["family"])
    best_by_family.to_csv(out_dir / "model_cv_summary.csv", index=False, encoding="utf-8-sig")

    models = {}
    final_params = {}
    for _, row in best_by_family.iterrows():
        family = row["family"]
        if family == "linear_ridge":
            params = {"lambda": float(row["lambda"])}
        elif family == "quadratic_ridge":
            params = {"lambda": float(row["lambda"]), "interaction_penalty": float(row["interaction_penalty"])}
        else:
            params = {"gamma": float(row["gamma"]), "lambda": float(row["lambda"])}
        final_params[family] = params
        models[family] = fit_model(family, train.p, train.y, params)

    # Held-out and extrapolation metrics for every candidate, with no target-scale calibration.
    metric_rows, output_rows = [], []
    all_predictions = []
    for dataset, pair in external.items():
        for family, model in models.items():
            pred = predict_model(model, pair.p)
            metrics = evaluate(pair.y, pred, weights)
            metric_rows.append({"dataset": dataset, "kind": config["external_sets"][dataset]["kind"],
                                "family": family, "selected_for_prediction": family == chosen["family"], **metrics})
            output_rows.extend(per_output_metrics(pair.y, pred, dataset, family, pair.loss_domains))
            if family == chosen["family"]:
                frame = pd.DataFrame({"dataset": dataset, "index": pair.indexes,
                                      "actual_macro_loss": pair.y @ weights, "predicted_macro_loss": pred @ weights,
                                      "selected_for_prediction": True})
                for k in range(pair.y.shape[1]):
                    frame[f"actual_loss_{k}"] = pair.y[:, k]
                    frame[f"predicted_loss_{k}"] = pred[:, k]
                all_predictions.append(frame)
    pd.DataFrame(metric_rows).to_csv(out_dir / "external_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(output_rows).to_csv(out_dir / "external_output_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(out_dir / "selected_model_predictions.csv", index=False, encoding="utf-8-sig")

    # Stable quadratic Scheffe model is retained as the explanation model even if
    # a different family wins the prediction CV.
    explanation_family = config["models"].get("explanation_model", "quadratic_ridge")
    explanation_model = models[explanation_family]
    rng = np.random.default_rng(config["seed"] + 17)
    replacement = replacement_effects(explanation_model, train.p, domains,
                                      config["effects"]["transfer_delta"],
                                      config["effects"]["bootstrap_repeats"], rng, weights, train.loss_domains)
    replacement.to_csv(out_dir / "replacement_effects.csv", index=False, encoding="utf-8-sig")
    combo_summary, combo_top = combination_effects(
        explanation_model, train.p, train.indexes, weights,
        config["effects"]["combination_pair_sample"], rng, train.loss_domains)
    combo_summary.to_csv(out_dir / "combination_effect_summary.csv", index=False, encoding="utf-8-sig")
    combo_top.to_csv(out_dir / "combination_effect_top.csv", index=False, encoding="utf-8-sig")

    extrap_rows = []
    for name in ["est_10B", "est_70B"]:
        extrap_rows.append(extrapolation_diagnostics(train, external[name], explanation_model, weights))
    pd.concat(extrap_rows, ignore_index=True).to_csv(out_dir / "extrapolation_diagnostics.csv", index=False, encoding="utf-8-sig")

    # A16 maps domains; A17/A18 audit the RegMix text sample without treating
    # structural text diagnostics as 22-signal quality scores.
    bridge_config = config["quality_bridge"]
    bridge_domain, bridge_recipe, bridge_meta, text_evidence = quality_bridge(
        CODE.parent / bridge_config["q1_summary"],
        CODE.parent / bridge_config["mapping"],
        CODE.parent / bridge_config["a17_summary"],
        CODE.parent / bridge_config["a18_sample"], train.p, domains, train.indexes)
    bridge_domain.to_csv(out_dir / "quality_bridge_domain.csv", index=False, encoding="utf-8-sig")
    bridge_recipe.to_csv(out_dir / "quality_bridge_recipe.csv", index=False, encoding="utf-8-sig")
    text_evidence.to_csv(out_dir / "regmix_text_evidence.csv", index=False, encoding="utf-8-sig")

    support = []
    for j, domain in enumerate(domains):
        support.append({"domain": domain, "positive_rows": int((train.p[:, j] > 0).sum()),
                        "zero_rows": int((train.p[:, j] == 0).sum()), "max_share": float(train.p[:, j].max()),
                        "mean_share": float(train.p[:, j].mean())})
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            support.append({"domain": f"{domains[i]}__{domains[j]}",
                            "positive_rows": int(((train.p[:, i] > 0) & (train.p[:, j] > 0)).sum()),
                            "zero_rows": np.nan, "max_share": np.nan, "mean_share": np.nan})
    pd.DataFrame(support).to_csv(out_dir / "domain_support.csv", index=False, encoding="utf-8-sig")

    _write_json(out_dir / "mixture_model.json", {
        "config": config, "domains": domains, "loss_outputs": train.loss_domains,
        "feature_names_quadratic": feature_names(), "fold_counts": np.bincount(folds).tolist(),
        "near_duplicate_groups": int(len(np.unique(groups))), "chosen_model": chosen,
        "final_parameters": final_params, "explanation_model": explanation_family,
        "quality_bridge": bridge_meta,
        "model_limitations": [
            "A4+A5 is the only model-selection source; A6-A11 are not used for tuning.",
            "No absolute-loss calibration is applied across scales.",
            "A12-A15 are extrapolated/subset tables and are not independent observations.",
            "A1-A3 and RegMix are independent corpora; A17/A18 text diagnostics are not 22-signal Q scores.",
            "Fixed domain Q_mix is a deterministic function of p and cannot identify an independent quality effect.",
            "RBF kernel ridge is the nonlinear baseline when LightGBM is unavailable in the runtime."
        ],
        "optional_baselines": {"lightgbm": {"available": lightgbm_available,
                                               "status": "not fitted" if not lightgbm_available else "not selected in configured families"}}
    })
    _write_json(out_dir / "task3_run_metadata.json", {
        "status": "completed", "version": config["version"], "seed": config["seed"],
        "train_rows": len(train.p), "external_rows": {k: len(v.p) for k, v in external.items()},
        "chosen_model": chosen["family"], "explanation_model": explanation_family,
        "source_type_policy": {k: v["kind"] for k, v in config["external_sets"].items()},
        "quality_bridge_excluded_from_primary": True,
        "lightgbm_available": lightgbm_available,
    })
    return {"chosen": chosen["family"], "train_rows": len(train.p), "out": str(out_dir),
            "cv": cv_table, "metrics": pd.DataFrame(metric_rows)}


def main(argv=None):
    args = _args(argv)
    result = run(args.config, args.data_root, args.out)
    print(f"Completed Q1 task 3: {result['train_rows']} training mixtures; selected {result['chosen']}; outputs: {result['out']}")


if __name__ == "__main__":
    main()
