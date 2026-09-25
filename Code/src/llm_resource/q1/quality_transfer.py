"""A16-checked profile mapping and task1/2-scored cross-domain Q transfer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import balanced_weights, fit_preprocessor, transform
from .schema import FIELDS, RAW_COLUMNS
from .quality_reconstruction import SHARED_FIELDS, _audit
from .public_quality_models import MODEL_REGISTRY
from .scoring import critic_weights, quality, pair_indices, interaction_weights


def profiles(x: np.ndarray, domain: np.ndarray, expected: list[str], fields: list[str]) -> pd.DataFrame:
    if set(np.unique(domain)) != set(expected):
        raise ValueError("Text sample does not cover expected domains")
    rows = []
    for name in expected:
        values = x[domain == name]
        row = {"domain": name, "n": len(values)}
        for j, field in enumerate(fields):
            row[field + "_mean"] = float(values[:, j].mean())
            row[field + "_sd"] = float(values[:, j].std())
            row[field + "_p10"] = float(np.quantile(values[:, j], .1))
            row[field + "_median"] = float(np.median(values[:, j]))
            row[field + "_p90"] = float(np.quantile(values[:, j], .9))
        rows.append(row)
    return pd.DataFrame(rows)


def similarity_transfer(a1_x: np.ndarray, a1_domain: np.ndarray, a18_x: np.ndarray,
                        a18_domain: np.ndarray, fields: list[str], summary_path: Path,
                        mapping_path: Path, train_domains: list[str],
                        temperatures=(.5, 1.0, 2.0)):
    """Fit feature scale on A1 only; use A16 direct labels only for validation."""
    summary = pd.read_csv(summary_path)
    source_domains = sorted(summary.loc[summary.dataset.eq("A1"), "domain"].unique())
    if len(source_domains) != 7:
        raise ValueError("Expected seven A1 quality domains")
    a1_profile = profiles(a1_x, a1_domain, source_domains, fields)
    a18_profile = profiles(a18_x, a18_domain, train_domains, fields)
    q_lookup = dict(zip(summary.loc[summary.dataset.eq("A1"), "domain"],
                        summary.loc[summary.dataset.eq("A1"), "Q_primary_mean"]))
    q_values = np.asarray([q_lookup[d] for d in source_domains], float)
    scale = a1_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    a1_mu = a1_profile[[f + "_mean" for f in fields]].to_numpy() / scale
    a18_mu = a18_profile[[f + "_mean" for f in fields]].to_numpy() / scale
    a1_sd = a1_profile[[f + "_sd" for f in fields]].to_numpy() / scale
    a18_sd = a18_profile[[f + "_sd" for f in fields]].to_numpy() / scale
    component = (a18_mu[:, None, :] - a1_mu[None, :, :]) ** 2 + .25 * (a18_sd[:, None, :] - a1_sd[None, :, :]) ** 2
    distance_matrix = np.sqrt(component.sum(axis=2))
    mapping = pd.read_csv(mapping_path)
    if len(mapping) != len(train_domains) or mapping.mixture_domain.duplicated().any() or set(mapping.mixture_domain) != set(train_domains):
        raise ValueError("A16 mapping must uniquely cover all 17 domains")
    mapping = mapping.set_index("mixture_domain")
    rows, sensitivity = [], []
    for j, name in enumerate(train_domains):
        distances = distance_matrix[j]
        order = np.argsort(distances)
        entry = mapping.loc[name]
        known = entry.mapping_type in {"direct", "near_direct"} and entry.quality_domain in source_domains
        true_rank = int(np.where(np.asarray(source_domains)[order] == entry.quality_domain)[0][0] + 1) if known else np.nan
        variants = []
        for omit in [None, *range(len(fields))]:
            d = distances if omit is None else np.sqrt(np.maximum(0, component[j].sum(axis=1) - component[j, :, omit]))
            variant_order = np.argsort(d)
            for temperature in temperatures:
                logits = -d / temperature
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                estimated = float(probabilities @ q_values)
                if omit is None and temperature == 1.0:
                    main_p, main_q = probabilities, estimated
                variants.append(estimated)
                sensitivity.append({"domain": name, "omitted_field": "none" if omit is None else fields[omit],
                                    "temperature": temperature, "q_estimate": estimated,
                                    "top1_quality_domain": source_domains[variant_order[0]],
                                    "top1_probability": float(probabilities[variant_order[0]])})
        rows.append({"domain": name, "a18_rows": int(a18_profile.loc[j, "n"]),
                     "a16_mapping_type": entry.mapping_type,
                     "a16_quality_domain": entry.quality_domain if known else "(none)",
                     "a16_known": bool(known), "a16_true_rank": true_rank,
                     "a16_reference_q": float(q_lookup[entry.quality_domain]) if known else np.nan,
                     "top1_quality_domain": source_domains[order[0]],
                     "top2_quality_domain": source_domains[order[1]],
                     "top1_distance": float(distances[order[0]]),
                     "top2_distance": float(distances[order[1]]),
                     "distance_margin": float(distances[order[1]] - distances[order[0]]),
                     "top1_probability": float(main_p[order[0]]),
                     "q_similarity": main_q,
                     "q_sensitivity_min": min(variants), "q_sensitivity_max": max(variants),
                     "q_sensitivity_sd": float(np.std(variants))})
    result = pd.DataFrame(rows)
    known = result[result.a16_known]
    near = known[known.a16_mapping_type.eq("near_direct")]
    baseline_q = float(q_values.mean())
    mae = float(np.mean(np.abs(known.q_similarity - known.a16_reference_q)))
    baseline_mae = float(np.mean(np.abs(baseline_q - known.a16_reference_q)))
    validation = {"n_known": int(len(known)), "top1_accuracy": float((known.a16_true_rank == 1).mean()),
                  "top3_accuracy": float((known.a16_true_rank <= 3).mean()),
                  "near_direct_top1_accuracy": float((near.a16_true_rank == 1).mean()) if len(near) else np.nan,
                  "known_mapping_q_mae": mae, "global_mean_q_mae": baseline_mae,
                  "gate": "top1 >= 5/6, all top3, near-direct top1 >= 2/3, Q MAE < global-mean baseline",
                  "passes_gate": bool(len(known) == 6 and (known.a16_true_rank == 1).sum() >= 5 and
                                      (known.a16_true_rank <= 3).all() and
                                      (near.a16_true_rank == 1).sum() >= 2 and mae < baseline_mae)}
    return result, pd.DataFrame(sensitivity), a1_profile, a18_profile, validation


def _score_with_fitted(raw: np.ndarray, domains: np.ndarray, fitted: dict,
                       config: dict, weights: np.ndarray) -> np.ndarray:
    z = transform(raw, domains, fitted)[0]
    edges = pair_indices(config["conflict"]["pairs"])
    theta = interaction_weights(weights, edges, config["conflict"]["eta"])
    base, rule, _ = quality(z, weights, edges, theta)
    return base if config.get("primary_score", "Q_base") == "Q_base" else rule


def score_transfer(a1_official_raw: np.ndarray, a1_rebuilt_raw: np.ndarray,
                   a1_full_rebuilt_raw: np.ndarray,
                   a1_domain: np.ndarray, a1_q_reference: np.ndarray,
                   a18_rebuilt_raw: np.ndarray,
                   a18_domain: np.ndarray, train_domains: list[str],
                   quality_config_path: Path, frozen_model_path: Path):
    """LODO refits task1/2 preprocessing; final A18 uses its frozen fit."""
    config = json.loads(quality_config_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_model_path.read_text(encoding="utf-8"))
    frozen_weights = np.asarray(frozen["score_model"]["weights"], float)
    reproduced = _score_with_fitted(a1_official_raw, a1_domain, frozen["preprocessor"], config, frozen_weights)
    reproduction_mae = float(np.mean(np.abs(reproduced - a1_q_reference)))
    if reproduction_mae > 1e-8:
        raise ValueError(f"Task1/2 frozen scorer does not reproduce joined A1 Q: MAE={reproduction_mae:g}")
    domain_rows = []
    for held in sorted(np.unique(a1_domain)):
        train, test = a1_domain != held, a1_domain == held
        fold_raw = a1_full_rebuilt_raw[test].copy()
        accepted_fold = []
        for field in SHARED_FIELDS:
            col = RAW_COLUMNS.index(field)
            status = _audit(field, a1_official_raw[train, col],
                            a1_full_rebuilt_raw[train, col], model=field in MODEL_REGISTRY)
            if status["passed"]:
                accepted_fold.append(field)
            else:
                fold_raw[:, col] = np.nan
        fitted = fit_preprocessor(a1_official_raw[train], a1_domain[train], config)
        z_train = transform(a1_official_raw[train], a1_domain[train], fitted)[0]
        critic = critic_weights(z_train, balanced_weights(a1_domain[train]))
        shrinkage = config["weights"]["critic_shrinkage"]
        weights = (1 - shrinkage) * np.full(len(FIELDS), 1 / len(FIELDS)) + shrinkage * critic
        actual = _score_with_fitted(a1_official_raw[test], a1_domain[test], fitted, config, weights)
        pred = _score_with_fitted(fold_raw, a1_domain[test], fitted, config, weights)
        training_q = _score_with_fitted(a1_official_raw[train], a1_domain[train], fitted, config, weights)
        baseline = float(np.mean([training_q[a1_domain[train] == d].mean() for d in np.unique(a1_domain[train])]))
        domain_rows.append({"domain": held, "n": int(test.sum()),
                            "fold_accepted_fields": json.dumps(accepted_fold),
                            "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
                            "mae": float(np.mean(np.abs(actual - pred))),
                            "baseline_rmse": float(np.sqrt(np.mean((actual - baseline) ** 2))),
                            "baseline_mae": float(np.mean(np.abs(actual - baseline))),
                            "actual_mean": float(actual.mean()), "predicted_mean": float(pred.mean()),
                            "baseline_mean": baseline})
    lodo = pd.DataFrame(domain_rows)
    fitted = frozen["preprocessor"]
    weights = frozen_weights
    predictions = _score_with_fitted(a18_rebuilt_raw, a18_domain, fitted, config, weights)
    direct_rows = []
    for name in train_domains:
        values = predictions[a18_domain == name]
        direct_rows.append({"domain": name, "a18_rows": len(values), "q_direct": float(values.mean()),
                            "q_direct_sd": float(values.std()), "q_direct_se": float(values.std() / np.sqrt(len(values))),
                            "q_direct_p10": float(np.quantile(values, .1)), "q_direct_p90": float(np.quantile(values, .9))})
    domain_mae = float(np.mean(np.abs(lodo.actual_mean - lodo.predicted_mean)))
    baseline_domain_mae = float(np.mean(np.abs(lodo.actual_mean - lodo.baseline_mean)))
    max_domain_error = float(np.max(np.abs(lodo.actual_mean - lodo.predicted_mean)))
    validation = {"lodo_macro_rmse": float(lodo.rmse.mean()), "baseline_macro_rmse": float(lodo.baseline_rmse.mean()),
                  "lodo_macro_mae": float(lodo.mae.mean()), "baseline_macro_mae": float(lodo.baseline_mae.mean()),
                  "domain_mean_mae": domain_mae, "baseline_domain_mean_mae": baseline_domain_mae,
                  "max_domain_mean_abs_error": max_domain_error,
                  "frozen_task12_q_reproduction_mae": reproduction_mae,
                  "gate": "LODO macro RMSE < baseline, domain-mean MAE <= 0.8 baseline, max domain error <= 0.08",
                  "passes_gate": bool(lodo.rmse.mean() < lodo.baseline_rmse.mean() and
                                      domain_mae <= .8 * baseline_domain_mae and max_domain_error <= .08),
                  "scorer": "task1/2 fit_preprocessor -> transform -> quality; frozen final weights/preprocessor"}
    return pd.DataFrame(direct_rows), lodo, validation
