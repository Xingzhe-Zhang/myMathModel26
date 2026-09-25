"""Fourteen audited signals: B20 quality reference, A16 bridge and direct-Q gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ab_experiment import ATOMIC_FIELDS, Q_FIELDS, build_mappings
from .preprocess import balanced_weights, fit_preprocessor, transform
from .schema import RAW_COLUMNS
from .scoring import critic_weights


def _save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_signal_config(fields: list[str], config: dict) -> np.ndarray:
    if fields != config["accepted_fields_required"] or len(fields) != 14:
        raise ValueError("Audited field list does not match the frozen fourteen-signal protocol")
    families = config["distance_families"]
    if len(families) != 5 or sorted(sum(families.values(), [])) != sorted(fields):
        raise ValueError("Distance families must partition the fourteen accepted fields")
    weights = np.zeros(len(fields))
    for members in families.values():
        for member in members:
            weights[fields.index(member)] = 1 / (len(families) * len(members))
    if not np.isclose(weights.sum(), 1):
        raise AssertionError("Family distance weights do not sum to one")
    return weights


def prepared_profiles(x: np.ndarray, fields: list[str], log_fields: list[str]) -> np.ndarray:
    z = np.asarray(x, float).copy()
    if not np.isfinite(z).all():
        raise ValueError("Profile matrix contains non-finite values")
    for field in log_fields:
        j = fields.index(field)
        z[:, j] = np.log1p(np.maximum(z[:, j], 0))
    return z


def domain_profiles(x: np.ndarray, domains: np.ndarray, expected: list[str],
                    fields: list[str], source: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if set(np.unique(domains)) != set(expected):
        raise ValueError(f"{source} domain coverage differs from the protocol")
    means, sds, rows = [], [], []
    for domain in expected:
        values = x[domains == domain]
        if len(values) == 0:
            raise ValueError(f"No profile rows for {source}/{domain}")
        means.append(values.mean(axis=0))
        sds.append(values.std(axis=0))
        row = {"source": source, "domain": domain, "n": len(values)}
        for j, field in enumerate(fields):
            row[f"{field}_mean"] = float(values[:, j].mean())
            row[f"{field}_sd"] = float(values[:, j].std())
            row[f"{field}_p10"] = float(np.quantile(values[:, j], .1))
            row[f"{field}_p90"] = float(np.quantile(values[:, j], .9))
        rows.append(row)
    return np.asarray(means), np.asarray(sds), pd.DataFrame(rows)


def distance_components(mu_a1: np.ndarray, sd_a1: np.ndarray,
                        mu_a18: np.ndarray, sd_a18: np.ndarray,
                        a1_scale: np.ndarray) -> np.ndarray:
    scale = np.maximum(a1_scale, 1e-8)
    delta_mu = (mu_a18[:, None, :] - mu_a1[None, :, :]) / scale
    delta_sd = (sd_a18[:, None, :] - sd_a1[None, :, :]) / scale
    return delta_mu ** 2 + .25 * delta_sd ** 2


def similarity_weights(components: np.ndarray, field_weights: np.ndarray,
                       temperature: float) -> tuple[np.ndarray, np.ndarray]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    distance = np.sqrt(np.maximum(components @ field_weights, 0))
    logits = -distance / temperature
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return distance, exp / exp.sum(axis=1, keepdims=True)


def b20_fold_model(raw: np.ndarray, domains: np.ndarray,
                   quality_config: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    fit = fit_preprocessor(raw, domains, quality_config)
    atom = transform(raw, domains, fit)[3]
    q_idx = [ATOMIC_FIELDS.index(name) for name in Q_FIELDS]
    weights_rows = balanced_weights(domains)
    q_inner = critic_weights(atom[:, q_idx], weights_rows)
    mapping = build_mappings(q_inner, quality_config["qurater_weights"])["B20_hierarchical"][1]
    concept = atom @ mapping
    outer = critic_weights(concept, weights_rows)
    return fit, mapping, outer


def score_b20(raw: np.ndarray, domains: np.ndarray, fit: dict,
              mapping: np.ndarray, weights: np.ndarray,
              global_missing: bool = False) -> np.ndarray:
    if global_missing:
        # RegMix is a separate corpus: do not borrow A1 domain medians merely
        # because names match. Keep the real domain for utility allowances.
        raw = np.where(np.isfinite(raw), raw, np.asarray(fit["global_medians"], float))
    atomic = transform(raw, domains, fit)[3]
    return atomic @ mapping @ weights


def accepted_only(raw: np.ndarray, accepted: list[str]) -> np.ndarray:
    result = np.full_like(raw, np.nan)
    for field in accepted:
        j = RAW_COLUMNS.index(field)
        result[:, j] = raw[:, j]
    return result


def fit_proxy_calibration(official: np.ndarray, rebuilt: np.ndarray,
                          accepted: list[str]) -> dict:
    result = {}
    for field in accepted:
        if field != "fineweb_edu" and not field.startswith("modernbert_"):
            continue
        j = RAW_COLUMNS.index(field)
        x, y = rebuilt[:, j], official[:, j]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 20:
            raise ValueError(f"Too few A1 calibration pairs for {field}")
        var = float(np.var(x[good]))
        slope = max(0., float(np.cov(x[good], y[good], bias=True)[0, 1] / var)) if var > 1e-12 else 0.
        intercept = float(y[good].mean() - slope * x[good].mean())
        result[field] = {"slope": slope, "intercept": intercept,
                         "clip": [0., 5.] if field == "fineweb_edu" else [0., 1.],
                         "n_fit": int(good.sum())}
    return result


def apply_proxy_calibration(raw: np.ndarray, calibration: dict) -> np.ndarray:
    result = raw.copy()
    for field, spec in calibration.items():
        j = RAW_COLUMNS.index(field)
        result[:, j] = np.clip(spec["slope"] * result[:, j] + spec["intercept"], *spec["clip"])
    return result


def direct_lodo(a1_official: np.ndarray, a1_rebuilt: np.ndarray,
                a1_domains: np.ndarray, accepted: list[str],
                quality_config: dict, gate_spec: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for held in sorted(np.unique(a1_domains)):
        tr, te = a1_domains != held, a1_domains == held
        fit, mapping, weights = b20_fold_model(a1_official[tr], a1_domains[tr], quality_config)
        q_official = score_b20(a1_official[te], a1_domains[te], fit, mapping, weights)
        q_training = score_b20(a1_official[tr], a1_domains[tr], fit, mapping, weights)
        constant = float(np.mean([q_training[a1_domains[tr] == d].mean() for d in np.unique(a1_domains[tr])]))
        partial = accepted_only(a1_official[te], accepted)
        calibration = fit_proxy_calibration(a1_official[tr], a1_rebuilt[tr], accepted)
        scenarios = {"official14_missing8": score_b20(partial, a1_domains[te], fit, mapping, weights),
                     "rebuilt14_raw": score_b20(a1_rebuilt[te], a1_domains[te], fit, mapping, weights),
                     "rebuilt14_calibrated": score_b20(apply_proxy_calibration(a1_rebuilt[te], calibration),
                                                       a1_domains[te], fit, mapping, weights),
                     "constant_train_domain_mean": np.full(int(te.sum()), constant)}
        for scenario, predicted in scenarios.items():
            rows.append({"domain": held, "scenario": scenario, "n": int(te.sum()),
                         "rmse": float(np.sqrt(np.mean((q_official - predicted) ** 2))),
                         "mae": float(np.mean(np.abs(q_official - predicted))),
                         "official_mean": float(q_official.mean()), "predicted_mean": float(predicted.mean()),
                         "domain_mean_abs_error": float(abs(q_official.mean() - predicted.mean()))})
    table = pd.DataFrame(rows)
    primary = table[table.scenario.eq("rebuilt14_calibrated")]
    baseline = table[table.scenario.eq("constant_train_domain_mean")]
    rmse, base_rmse = float(primary.rmse.mean()), float(baseline.rmse.mean())
    mae, base_mae = float(primary.domain_mean_abs_error.mean()), float(baseline.domain_mean_abs_error.mean())
    max_error = float(primary.domain_mean_abs_error.max())
    rank_spearman = float(primary.official_mean.rank().corr(primary.predicted_mean.rank()))
    gate = {"lodo_macro_rmse": rmse, "constant_macro_rmse": base_rmse,
            "domain_mean_mae": mae, "constant_domain_mean_mae": base_mae,
            "max_domain_mean_abs_error": max_error,
            "domain_mean_spearman": rank_spearman,
            "passes_gate": bool(rmse < base_rmse and
                                mae <= gate_spec["domain_mae_fraction_of_constant"] * base_mae and
                                max_error <= gate_spec["max_domain_mean_abs_error"]),
            "gate": gate_spec, "main_scenario": "rebuilt14_calibrated"}
    return table, gate


def run_quality_phase(config: dict, data_root: Path, signal_dir: Path,
                      ab_dir: Path, out_dir: Path, train_domains: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = signal_dir / "accepted_fields.json"
    npz_path = signal_dir / "reconstructed_signals.npz"
    fields = json.loads(accepted_path.read_text(encoding="utf-8"))
    field_weights = validate_signal_config(fields, config)
    signals = np.load(npz_path, allow_pickle=False)
    a1_domains, a18_domains = signals["a1_domain"], signals["a18_domain"]
    a1_domains_expected = sorted(np.unique(a1_domains).tolist())
    if len(a1_domains_expected) != 7 or len(train_domains) != 17:
        raise ValueError("Expected seven quality and seventeen RegMix domains")
    a1 = prepared_profiles(signals["a1_x"], fields, config["length_log1p_fields"])
    a18 = prepared_profiles(signals["a18_x"], fields, config["length_log1p_fields"])
    mu1, sd1, profile1 = domain_profiles(a1, a1_domains, a1_domains_expected, fields, "A1")
    mu18, sd18, profile18 = domain_profiles(a18, a18_domains, train_domains, fields, "A18")
    _save(pd.concat((profile1, profile18), ignore_index=True), out_dir / "a1_a18_domain_profiles.csv")
    components = distance_components(mu1, sd1, mu18, sd18, a1.std(axis=0))
    temperature = float(config["temperature_main"])
    distance, soft = similarity_weights(components, field_weights, temperature)
    summary = pd.read_csv(ab_dir / "domain_summary.csv")
    q_spec = config["quality_source"]
    if (q_spec["scheme"], q_spec["profile"], q_spec["score"]) != (
        "B20_hierarchical", "frozen_original", "Q_base"
    ):
        raise ValueError("The direct fourteen-signal scorer requires frozen B20 Q_base")

    def reference(scheme: str, profile: str, source: str) -> np.ndarray:
        selected = summary[(summary.scheme == scheme) & (summary.profile == profile) & (summary.source == source)]
        if set(selected.domain) != set(a1_domains_expected) or len(selected) != 7:
            raise ValueError(f"Missing seven quality-domain summaries: {scheme}/{profile}/{source}")
        return selected.set_index("domain").loc[a1_domains_expected, "mean"].to_numpy(float)

    q_reference = reference(q_spec["scheme"], q_spec["profile"], q_spec["reference_source"])
    mapping = pd.read_csv(data_root / "A_data_value/domain_mapping_guide.csv").set_index("mixture_domain")
    if set(mapping.index) != set(train_domains) or mapping.index.duplicated().any():
        raise ValueError("A16 must uniquely cover the seventeen training domains")
    final_weights = soft.copy()
    map_rows, weight_rows, anchor_rows = [], [], []
    known_count = 0
    for i, domain in enumerate(train_domains):
        anchor = mapping.loc[domain]
        known = anchor.mapping_type in {"direct", "near_direct"}
        if known:
            known_count += 1
            target = a1_domains_expected.index(anchor.quality_domain)
            final_weights[i] = 0
            final_weights[i, target] = 1
        order = np.argsort(distance[i])
        top1, top2 = a1_domains_expected[order[0]], a1_domains_expected[order[1]]
        hard_first = anchor.quality_domain if known else top1
        hard_second = anchor.quality_domain if known else top2
        row = {"domain": domain, "mapping_type": anchor.mapping_type,
               "a16_quality_domain": anchor.quality_domain,
               "a16_fixed": known, "a18_rows": int((a18_domains == domain).sum()),
               "top1_quality_domain": top1, "top2_quality_domain": top2,
               "top1_soft_weight": float(soft[i, order[0]]),
               "distance_margin": float(distance[i, order[1]] - distance[i, order[0]]),
               "weight_entropy": float(-np.sum(soft[i] * np.log(np.maximum(soft[i], 1e-12)))),
               "q_soft_unfixed": float(soft[i] @ q_reference),
               "q_final": float(final_weights[i] @ q_reference),
               "q_hard_top1": float(q_reference[a1_domains_expected.index(hard_first)]),
               "q_hard_top2": float(q_reference[a1_domains_expected.index(hard_second)])}
        map_rows.append(row)
        for j, quality_domain in enumerate(a1_domains_expected):
            weight_rows.append({"domain": domain, "quality_domain": quality_domain,
                                "distance": float(distance[i, j]), "soft_weight": float(soft[i, j]),
                                "final_weight": float(final_weights[i, j]), "a16_fixed": known})
        if known:
            rank = int(np.where(order == target)[0][0] + 1)
            anchor_rows.append({"domain": domain, "mapping_type": anchor.mapping_type,
                                "reference_quality_domain": anchor.quality_domain,
                                "predicted_top1": top1, "predicted_top2": top2,
                                "reference_rank": rank, "q_reference": float(q_reference[target]),
                                "q_soft_unfixed": float(soft[i] @ q_reference),
                                "q_abs_error": float(abs(soft[i] @ q_reference - q_reference[target]))})
    if known_count != 6 or not np.allclose(final_weights.sum(axis=1), 1):
        raise ValueError("A16 fixed mappings or final weight rows are invalid")
    map_table, anchors = pd.DataFrame(map_rows), pd.DataFrame(anchor_rows)
    _save(pd.DataFrame(weight_rows), out_dir / "a16_mapping_weights.csv")
    _save(anchors, out_dir / "a16_anchor_validation.csv")
    top1_hits = int((anchors.reference_rank == 1).sum())
    top3_hits = int((anchors.reference_rank <= 3).sum())
    near = anchors[anchors.mapping_type.eq("near_direct")]
    near_hits = int((near.reference_rank == 1).sum())
    q_mae = float(anchors.q_abs_error.mean())
    constant_mae = float(np.mean(np.abs(anchors.q_reference - q_reference.mean())))
    anchor_rng = np.random.default_rng(config["seed"] + 1)
    anchor_resamples = anchor_rng.integers(0, len(anchors), size=(10000, len(anchors)))
    anchor_mae_draws = anchors.q_abs_error.to_numpy(float)[anchor_resamples].mean(axis=1)
    anchor_accuracy_draws = (anchors.reference_rank.to_numpy(int)[anchor_resamples] == 1).mean(axis=1)
    spec = config["a16_gate"]
    bridge_gate = {"n_known": len(anchors), "top1_hits": top1_hits, "top3_hits": top3_hits,
                   "near_direct_top1_hits": near_hits, "q_mae": q_mae,
                   "constant_q_mae": constant_mae,
                   "six_anchor_mae_bootstrap_95": np.quantile(anchor_mae_draws, [.025, .975]).tolist(),
                   "six_anchor_top1_bootstrap_95": np.quantile(anchor_accuracy_draws, [.025, .975]).tolist(),
                   "passes_gate": bool(top1_hits >= spec["min_top1_of_6"] and
                                       top3_hits >= spec["min_top3_of_6"] and
                                       near_hits >= spec["min_near_top1_of_3"] and q_mae < constant_mae),
                   "gate": spec}

    sensitivity_rows = []
    families = config["distance_families"]
    variants = [("main", field_weights)]
    variants.extend((f"omit_field:{field}", np.where(np.arange(len(fields)) == j, 0, field_weights))
                    for j, field in enumerate(fields))
    variants.extend((f"omit_family:{family}", np.asarray([0 if field in members else field_weights[j]
                                                            for j, field in enumerate(fields)]))
                    for family, members in families.items())
    for scenario, w in variants:
        w = w / w.sum()
        for tau in [temperature, *config["temperature_sensitivity"]]:
            d, candidate = similarity_weights(components, w, float(tau))
            for i, domain in enumerate(train_domains):
                applied = final_weights[i] if map_table.loc[i, "a16_fixed"] else candidate[i]
                sensitivity_rows.append({"domain": domain, "scenario": scenario, "temperature": tau,
                                         "top1_quality_domain": a1_domains_expected[int(np.argmin(d[i]))],
                                         "q_bridge": float(applied @ q_reference),
                                         "a16_fixed": bool(map_table.loc[i, "a16_fixed"])})
    sensitivity = pd.DataFrame(sensitivity_rows)
    _save(sensitivity, out_dir / "mapping_and_sample_sensitivity.csv")
    for i, domain in enumerate(train_domains):
        subset = sensitivity[sensitivity.domain.eq(domain)]
        main_top1 = map_table.loc[i, "top1_quality_domain"]
        field_variants = subset[(subset.scenario.str.startswith("omit_field:")) &
                                (np.isclose(subset.temperature, temperature))]
        family_variants = subset[(subset.scenario.str.startswith("omit_family:")) &
                                 (np.isclose(subset.temperature, temperature))]
        map_table.loc[i, "leave_one_field_top1_stability"] = float(field_variants.top1_quality_domain.eq(main_top1).mean())
        map_table.loc[i, "leave_one_family_top1_stability"] = float(family_variants.top1_quality_domain.eq(main_top1).mean())
        map_table.loc[i, "q_sensitivity_min"] = float(subset.q_bridge.min())
        map_table.loc[i, "q_sensitivity_max"] = float(subset.q_bridge.max())

    rng = np.random.default_rng(config["seed"])
    repeats = int(config["bootstrap_repeats"])
    ref_summary = summary[(summary.scheme == q_spec["scheme"]) & (summary.profile == q_spec["profile"]) &
                          (summary.source == q_spec["reference_source"])].set_index("domain").loc[a1_domains_expected]
    ref_se = ref_summary["std"].to_numpy(float) / np.sqrt(ref_summary["n_unique"].to_numpy(float))
    boot_q = np.empty((repeats, len(train_domains)))
    boot_top1 = np.empty((repeats, len(train_domains)), int)
    for b in range(repeats):
        resampled_a1 = np.vstack([a1[rng.choice(np.flatnonzero(a1_domains == d), (a1_domains == d).sum(), replace=True)]
                                  for d in a1_domains_expected])
        resampled_a18 = np.vstack([a18[rng.choice(np.flatnonzero(a18_domains == d), (a18_domains == d).sum(), replace=True)]
                                   for d in train_domains])
        repeated_a1_domain = np.concatenate([np.full((a1_domains == d).sum(), d) for d in a1_domains_expected])
        repeated_a18_domain = np.concatenate([np.full((a18_domains == d).sum(), d) for d in train_domains])
        boot_mu1, boot_sd1, _ = domain_profiles(resampled_a1, repeated_a1_domain, a1_domains_expected, fields, "A1_boot")
        boot_mu18, boot_sd18, _ = domain_profiles(resampled_a18, repeated_a18_domain, train_domains, fields, "A18_boot")
        boot_component = distance_components(boot_mu1, boot_sd1, boot_mu18, boot_sd18, a1.std(axis=0))
        boot_distance, boot_soft = similarity_weights(boot_component, field_weights, temperature)
        boot_top1[b] = np.argmin(boot_distance, axis=1)
        boot_soft[map_table.a16_fixed.to_numpy(bool)] = final_weights[map_table.a16_fixed.to_numpy(bool)]
        boot_ref = np.clip(q_reference + rng.normal(0, ref_se), 0, 1)
        boot_q[b] = boot_soft @ boot_ref
    map_table["q_bootstrap_p025"] = np.quantile(boot_q, .025, axis=0)
    map_table["q_bootstrap_p975"] = np.quantile(boot_q, .975, axis=0)
    map_table["bootstrap_top1_stability"] = (boot_top1 == np.argmin(distance, axis=1)).mean(axis=0)
    q_variants = {"A22_original_union": reference("A22", "frozen_original", "union"),
                  "B23_original_union": reference("B23_flat", "frozen_original", "union"),
                  "B20_rank_union": reference("B20_hierarchical", "dsir_reference_rank", "union"),
                  "B20_original_A1": reference("B20_hierarchical", "frozen_original", "A1")}
    for label, q in q_variants.items():
        map_table[f"q_sensitivity_{label}"] = final_weights @ q
    _save(map_table, out_dir / "q_bridge_17_domains.csv")
    _save(map_table[["domain", *[f"q_sensitivity_{label}" for label in q_variants]]],
          out_dir / "q_source_sensitivity.csv")

    quality_config = json.loads((Path("Code/configs/q1_quality.json")).read_text(encoding="utf-8"))
    official, rebuilt = signals["a1_official_raw"], signals["a1_rebuilt_raw"]
    lodo, direct_gate = direct_lodo(official, rebuilt, a1_domains, fields, quality_config, config["direct_gate"])
    _save(lodo, out_dir / "q_direct_lodo.csv")
    frozen = json.loads((ab_dir / "fitted_experiment.json").read_text(encoding="utf-8"))
    model = frozen["normalization_profiles"][q_spec["profile"]]["schemes"][q_spec["scheme"]]
    fit = frozen["quality_preprocessor"]
    m = np.asarray(model["atomic_to_concept"], float)
    w = np.asarray(model["final_concept_weights"], float)
    calibration = fit_proxy_calibration(official, rebuilt, fields)
    a18_raw = signals["a18_rebuilt_raw"]
    direct_rows = []
    for scenario, raw in (("rebuilt14_raw", a18_raw),
                          ("rebuilt14_calibrated", apply_proxy_calibration(a18_raw, calibration))):
        q_text = score_b20(raw, a18_domains, fit, m, w, global_missing=True)
        for domain in train_domains:
            values = q_text[a18_domains == domain]
            draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
            direct_rows.append({"domain": domain, "scenario": scenario, "n": len(values),
                                "q_mean": float(values.mean()), "q_sd": float(values.std()),
                                "q_p10": float(np.quantile(values, .1)),
                                "q_p90": float(np.quantile(values, .9)),
                                "q_bootstrap_p025": float(np.quantile(draws, .025)),
                                "q_bootstrap_p975": float(np.quantile(draws, .975)),
                                "quality_gate_passed": bool(direct_gate["passes_gate"])})
    _save(pd.DataFrame(direct_rows), out_dir / "q_direct_17_domains.csv")

    manifest = {"version": config["version"], "accepted_fields": fields,
                "signal_npz_sha256": _digest(npz_path), "accepted_fields_sha256": _digest(accepted_path),
                "a1_rows": len(a1), "a18_rows": len(a18),
                "a1_domains": a1_domains_expected, "a18_domains": train_domains,
                "distance_field_weights": dict(zip(fields, field_weights.tolist())),
                "calibration_fitted_on_A1_sample_only": calibration,
                "a18_missing_policy": "A1 global medians; retain real domains for frozen utility allowances",
                "failed_signals_excluded": ["rps_lines_uppercase_letter_fraction",
                                             "rps_lines_ending_with_terminal_punctution_mark", "fluency_en"]}
    (out_dir / "signal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    version = {"scheme": q_spec["scheme"], "profile": q_spec["profile"],
               "score": q_spec["score"], "reference_source": q_spec["reference_source"],
               "frozen_experiment_sha256": _digest(ab_dir / "fitted_experiment.json"),
               "domain_summary_sha256": _digest(ab_dir / "domain_summary.csv"),
               "selection_reason": "B20 controls QuRating and DSIR family budgets; original frozen scale avoids the larger DSIR-rank shift. This is a structural prior, not validated training utility.",
               "direct_missing_official_fields": 8,
               "direct_missing_atomic_signals": len(ATOMIC_FIELDS) - len(fields)}
    (out_dir / "q_source_version.json").write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = {"version": config["version"], "a1_signal_passed": True,
            "bridge_a16": bridge_gate, "direct_b20": direct_gate,
            "bridge_loss_eligible": bool(bridge_gate["passes_gate"]),
            "direct_loss_eligible": bool(direct_gate["passes_gate"]),
            "nine_signal_result_not_substituted": True,
            "interpretation": "Quality transfer is an audited cross-corpus prior, not an independent causal covariate."}
    (out_dir / "quality_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate
