"""Frozen A1-fitted A22, B23, and B20 quality-dimension experiment.

All three schemes use the same atomic utility matrix within each normalization
profile. CRITIC estimates contrast and nonredundancy, not quality validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import source_masks
from .data import read_corpus
from .preprocess import balanced_weights, fit_preprocessor, transform, utility_dictionary, validate_config, weighted_quantile
from .schema import FIELDS, RAW_COLUMNS
from .scoring import critic_weights, diagnostic_scores, interaction_weights, quality

Q_FIELDS = (
    "qurater_writing_style", "qurater_required_expertise",
    "qurater_facts_trivia", "qurater_educational_value",
)
DSIR_FIELDS = ("dsir_books", "dsir_wiki", "dsir_math")
ATOMIC_FIELDS = tuple(
    Q_FIELDS[int(c.rsplit("_", 1)[1])] if c.startswith("qurater_") else c
    for c in RAW_COLUMNS
)
DSIR_GROUP = "dsir_target_relevance_equal_prior"
Q_GROUP = "qurater_family"


def validate_experiment_config(config: dict) -> None:
    if config["normalization_profiles"] != ["frozen_original", "dsir_reference_rank"]:
        raise ValueError("Expected the two preregistered normalization profiles")
    if tuple(config["dsir_group"]["members"]) != DSIR_FIELDS:
        raise ValueError("DSIR group membership changed")
    if tuple(config["qurater_group"]["members"]) != Q_FIELDS:
        raise ValueError("QuRating group membership changed")
    if not 0 <= config["critic_shrinkage"] <= 1:
        raise ValueError("critic_shrinkage must be in [0,1]")
    grid = config["critic_shrinkage_sensitivity"]
    if not grid or any(not 0 <= float(x) <= 1 for x in grid):
        raise ValueError("Invalid CRITIC sensitivity values")
    if not 0 < config["selection_fraction_per_domain"] < 1:
        raise ValueError("Invalid selection fraction")
    if not 0 <= config["bootstrap_repeats"] <= 1000:
        raise ValueError("Invalid bootstrap repetitions")
    if not 11 <= config["dsir_reference_rank_knots"] <= 2001:
        raise ValueError("Invalid reference rank knot count")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def fit_dsir_rank_reference(raw_a1: np.ndarray, domains_a1: np.ndarray, knots: int) -> dict:
    """Monotone empirical percentile map fitted on A1, with equal domain mass."""
    weights = balanced_weights(domains_a1)
    grid = np.linspace(0, 1, knots)
    fitted = {}
    for name in DSIR_FIELDS:
        j = RAW_COLUMNS.index(name)
        values = weighted_quantile(raw_a1[:, j], grid, weights)
        unique, first, counts = np.unique(values, return_index=True, return_counts=True)
        percentiles = np.asarray([grid[k:k+n].mean() for k, n in zip(first, counts)])
        if len(unique) < 2:
            raise ValueError(f"Cannot fit DSIR percentile map: {name} is constant")
        fitted[name] = {"raw_knots": unique.tolist(), "reference_percentiles": percentiles.tolist(),
                        "direction": "higher raw importance means more target relevance"}
    return fitted


def transform_dsir_rank(raw: np.ndarray, domains: np.ndarray, fitted_preprocessor: dict,
                        reference: dict, atomic_original: np.ndarray) -> np.ndarray:
    result = atomic_original.copy()
    for name in DSIR_FIELDS:
        j = RAW_COLUMNS.index(name)
        values = raw[:, j].copy()
        for domain in np.unique(domains):
            selected = (domains == domain) & ~np.isfinite(values)
            if selected.any():
                median = fitted_preprocessor["domain_medians"].get(str(domain), fitted_preprocessor["global_medians"])[j]
                values[selected] = median
        if not np.isfinite(values).all():
            raise ValueError(f"Missing DSIR values remained after A1-frozen imputation: {name}")
        spec = reference[name]
        result[:, j] = np.interp(values, spec["raw_knots"], spec["reference_percentiles"], left=0, right=1)
    return result


def build_mappings(q_inner: np.ndarray, q_parent: np.ndarray) -> dict:
    """Return atomic-to-concept linear maps, each atomic assigned exactly once."""
    q_inner = np.asarray(q_inner, float)
    q_parent = np.asarray(q_parent, float)
    if len(q_inner) != 4 or len(q_parent) != 4 or (q_inner < 0).any() or (q_parent < 0).any():
        raise ValueError("Invalid QuRating family weights")
    if not np.isclose(q_inner.sum(), 1) or not np.isclose(q_parent.sum(), 1):
        raise ValueError("QuRating family weights must sum to one")
    n = len(ATOMIC_FIELDS)
    index = {name: i for i, name in enumerate(ATOMIC_FIELDS)}

    a_names = list(FIELDS)
    a = np.zeros((n, len(a_names)))
    for col, name in enumerate(a_names):
        if name == "qurater":
            for q_name, value in zip(Q_FIELDS, q_parent):
                a[index[q_name], col] = value
        else:
            a[index[name], col] = 1

    b23_names = [name for name in ATOMIC_FIELDS if name not in DSIR_FIELDS] + [DSIR_GROUP]
    b23 = np.zeros((n, len(b23_names)))
    for col, name in enumerate(b23_names):
        if name == DSIR_GROUP:
            for member in DSIR_FIELDS:
                b23[index[member], col] = 1 / 3
        else:
            b23[index[name], col] = 1

    b20_names = [name for name in ATOMIC_FIELDS if name not in (*Q_FIELDS, *DSIR_FIELDS)] + [Q_GROUP, DSIR_GROUP]
    b20 = np.zeros((n, len(b20_names)))
    for col, name in enumerate(b20_names):
        if name == Q_GROUP:
            for member, value in zip(Q_FIELDS, q_inner):
                b20[index[member], col] = value
        elif name == DSIR_GROUP:
            for member in DSIR_FIELDS:
                b20[index[member], col] = 1 / 3
        else:
            b20[index[name], col] = 1
    result = {"A22": (a_names, a), "B23_flat": (b23_names, b23), "B20_hierarchical": (b20_names, b20)}
    for name, (names, matrix) in result.items():
        if len(names) != {"A22": 22, "B23_flat": 23, "B20_hierarchical": 20}[name]:
            raise AssertionError(f"Wrong concept count for {name}")
        if not np.all((matrix > 0).sum(axis=1) == 1):
            raise AssertionError(f"Atomic coverage is incomplete in {name}")
    return result


def top_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(values) * fraction)))
    selected = np.zeros(len(values), bool)
    selected[np.argsort(-values, kind="stable")[:count]] = True
    return selected


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.sort(np.concatenate((a, b)))
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def bootstrap_critic(matrix: np.ndarray, domains: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    if repeats == 0:
        return np.empty((0, matrix.shape[1]))
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(domains == name) for name in np.unique(domains)]
    weight = balanced_weights(domains)
    result = np.empty((repeats, matrix.shape[1]))
    for b in range(repeats):
        take = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in strata])
        result[b] = critic_weights(matrix[take], weight[take])
    return result


def fitted_scheme(atomic: np.ndarray, a1: np.ndarray, domains: np.ndarray,
                  quality_config: dict, xi: float, bootstrap_repeats: int, seed: int) -> dict:
    reference_weight = balanced_weights(domains[a1])
    q_idx = [ATOMIC_FIELDS.index(name) for name in Q_FIELDS]
    q_inner = critic_weights(atomic[a1][:, q_idx], reference_weight)
    mappings = build_mappings(q_inner, quality_config["qurater_weights"])
    models = {}
    for k, (name, (concept_names, mapping)) in enumerate(mappings.items()):
        concepts = atomic @ mapping
        full_critic = critic_weights(concepts[a1], reference_weight)
        weights = (1 - xi) / len(concept_names) + xi * full_critic
        boots = bootstrap_critic(concepts[a1], domains[a1], bootstrap_repeats, seed + k)
        models[name] = {"concept_names": concept_names, "mapping": mapping,
                        "full_critic": full_critic, "weights": weights,
                        "atomic_weights": mapping @ weights, "concepts": concepts,
                        "bootstrap": boots}
    return {"q_inner": q_inner, "models": models}


def candidate_conflict_edges(names: list[str], source_pairs: list[list[str]]) -> list[tuple[int, int]]:
    """Keep the original semantic pairs; expand QuRating only in flat B23."""
    index = {name: j for j, name in enumerate(names)}
    edges = set()
    for left, right in source_pairs:
        lhs = Q_FIELDS if left == "qurater" and Q_GROUP not in index and "qurater" not in index else (Q_GROUP if left == "qurater" and Q_GROUP in index else left,)
        rhs = Q_FIELDS if right == "qurater" and Q_GROUP not in index and "qurater" not in index else (Q_GROUP if right == "qurater" and Q_GROUP in index else right,)
        for a in lhs:
            for b in rhs:
                if a not in index or b not in index or a == b:
                    raise ValueError(f"Cannot map conflict pair {left}, {right}")
                edges.add(tuple(sorted((index[a], index[b]))))
    return sorted(edges)


def utility_direction(kind: str) -> str:
    return {"identity": "higher", "fixed_positive": "higher",
            "reference_positive": "higher", "reference_negative": "lower",
            "reference_interval": "typical_interval", "upper_tolerance": "below_tolerance"}[kind]


def _source_labels(corpus) -> list[str]:
    members = corpus.members.groupby("row").source.apply(lambda x: ";".join(sorted(set(x)))).to_dict()
    return [members.get(i, "") for i in range(len(corpus.records))]


def run(data_root: Path, quality_config_path: Path, experiment_config_path: Path,
        out_dir: Path, bootstrap_override: int | None = None) -> dict:
    quality_config = json.loads(quality_config_path.read_text(encoding="utf-8"))
    config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    validate_config(quality_config)
    validate_experiment_config(config)
    if bootstrap_override is not None:
        if not 0 <= bootstrap_override <= 1000:
            raise ValueError("Invalid bootstrap override")
        config["bootstrap_repeats"] = bootstrap_override
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = read_corpus(data_root)
    print(f"Loaded {len(corpus.records):,} unique records; fitting A1 reference", flush=True)
    masks = source_masks(corpus)
    domains = corpus.records.domain.to_numpy()
    a1 = masks["A1"]
    fitted = fit_preprocessor(corpus.raw[a1], domains[a1], quality_config)
    z22, valid22, _, original_atomic = transform(corpus.raw, domains, fitted,
                                                  domain_allowance=bool(quality_config["reference"].get("domain_allowance", True)))
    dsir_reference = fit_dsir_rank_reference(corpus.raw[a1], domains[a1], config["dsir_reference_rank_knots"])
    profiles = {"frozen_original": original_atomic,
                "dsir_reference_rank": transform_dsir_rank(corpus.raw, domains, fitted, dsir_reference, original_atomic)}
    source_labels = _source_labels(corpus)
    scores = pd.DataFrame({"uid": corpus.records.uid.to_numpy(), "id": corpus.records.id.to_numpy(),
                           "domain": domains, "source_membership": source_labels,
                           "valid_22_field_fraction": valid22.mean(axis=1)})
    all_fitted = {}
    weight_rows, utility_rows, summary_rows, comparison_rows = [], [], [], []
    shift_rows, sensitivity_rows, normalization_rows = [], [], []
    inner_sensitivity_rows, rule_comparison_rows, redundancy_rows = [], [], []
    fraction = config["selection_fraction_per_domain"]
    for label in ("A1", "A2_new", "A3_new"):
        selected = masks[label]
        for family, members in (("dsir", DSIR_FIELDS), ("qurater", Q_FIELDS)):
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    a = corpus.raw[selected, RAW_COLUMNS.index("qurater_" + str(Q_FIELDS.index(left)) if left in Q_FIELDS else left)]
                    b = corpus.raw[selected, RAW_COLUMNS.index("qurater_" + str(Q_FIELDS.index(right)) if right in Q_FIELDS else right)]
                    valid = np.isfinite(a) & np.isfinite(b)
                    redundancy_rows.append({"source": label, "family": family, "left": left, "right": right,
                                            "n_pair": int(valid.sum()),
                                            "spearman_raw": rank_correlation(a[valid], b[valid])})
    for profile_index, profile in enumerate(config["normalization_profiles"]):
        print(f"Scoring profile {profile} ({profile_index + 1}/2)", flush=True)
        atomic = profiles[profile]
        if not np.isfinite(atomic).all() or atomic.min() < -1e-12 or atomic.max() > 1 + 1e-12:
            raise ValueError(f"Non-finite or out-of-range utilities in {profile}")
        fitted_models = fitted_scheme(atomic, a1, domains, quality_config, config["critic_shrinkage"],
                                      config["bootstrap_repeats"], config["seed"] + 100 * profile_index)
        print(f"Fitted three schemes and {config['bootstrap_repeats']} bootstrap replicates per scheme", flush=True)
        models = fitted_models["models"]
        model_json = {"qurater_within_family_critic_weights": fitted_models["q_inner"], "schemes": {}}
        for scheme, model in models.items():
            concept_names = model["concept_names"]
            edges = candidate_conflict_edges(concept_names, quality_config["conflict"]["pairs"])
            theta = interaction_weights(model["weights"], edges, quality_config["conflict"]["eta"])
            model_json["schemes"][scheme] = {"concept_names": concept_names,
                                             "atomic_to_concept": model["mapping"],
                                             "full_critic_weights": model["full_critic"],
                                             "final_concept_weights": model["weights"],
                                             "final_atomic_weights": model["atomic_weights"],
                                             "conflict_edges": [[concept_names[a], concept_names[b]] for a, b in edges],
                                             "conflict_theta": theta}
            q, q_rule, penalty = quality(model["concepts"], model["weights"], edges, theta)
            scores[f"Q_{scheme}_{profile}"] = q
            scores[f"Q_rule_{scheme}_{profile}"] = q_rule
            scores[f"rule_penalty_{scheme}_{profile}"] = penalty
            boots = model["bootstrap"]
            for j, name in enumerate(concept_names):
                weight_rows.append({"profile": profile, "scheme": scheme, "level": "concept", "name": name,
                                    "weight": model["weights"][j], "full_critic": model["full_critic"][j],
                                    "bootstrap_p025": np.quantile(boots[:, j], .025) if len(boots) else np.nan,
                                    "bootstrap_p975": np.quantile(boots[:, j], .975) if len(boots) else np.nan})
            for j, name in enumerate(ATOMIC_FIELDS):
                weight_rows.append({"profile": profile, "scheme": scheme, "level": "atomic", "name": name,
                                    "weight": model["atomic_weights"][j], "full_critic": np.nan,
                                    "bootstrap_p025": np.nan, "bootstrap_p975": np.nan})
            for xi in config["critic_shrinkage_sensitivity"]:
                w = (1 - xi) / len(concept_names) + xi * model["full_critic"]
                candidate = model["concepts"][a1] @ w
                full = q[a1]
                for domain in np.unique(domains[a1]):
                    selected = domains[a1] == domain
                    a, b = full[selected], candidate[selected]
                    sensitivity_rows.append({"profile": profile, "scheme": scheme, "xi": xi, "domain": domain,
                                             "mean": b.mean(), "spearman_vs_primary_xi": rank_correlation(a, b),
                                             "top_fraction_overlap_vs_primary_xi":
                                             (top_mask(a, fraction) & top_mask(b, fraction)).sum() / top_mask(a, fraction).sum()})
        equal_map = build_mappings(np.full(4, .25), quality_config["qurater_weights"])["B20_hierarchical"][1]
        equal_concepts = atomic @ equal_map
        equal_outer = critic_weights(equal_concepts[a1], balanced_weights(domains[a1]))
        equal_score = equal_concepts @ equal_outer
        primary_score = scores[f"Q_B20_hierarchical_{profile}"].to_numpy()
        for label, selected in masks.items():
            for domain in np.unique(domains[selected]):
                mask = selected & (domains == domain)
                a, b = primary_score[mask], equal_score[mask]
                inner_sensitivity_rows.append({"profile": profile, "source": label, "domain": domain,
                                               "n": len(a), "spearman": rank_correlation(a, b),
                                               "top_fraction_overlap":
                                               (top_mask(a, fraction) & top_mask(b, fraction)).sum() / top_mask(a, fraction).sum(),
                                               "mean_score_change": (b-a).mean()})
        all_fitted[profile] = model_json
        uniform = np.full(len(ATOMIC_FIELDS), 1 / len(ATOMIC_FIELDS))
        _, intensity, high_low = diagnostic_scores(atomic, uniform, .75, .25)
        scores[f"atomic_conflict_intensity_{profile}"] = intensity
        scores[f"atomic_has_high_low_difference_{profile}"] = high_low
        q_idx = [ATOMIC_FIELDS.index(name) for name in Q_FIELDS]
        d_idx = [ATOMIC_FIELDS.index(name) for name in DSIR_FIELDS]
        scores[f"qurater_spread_{profile}"] = np.ptp(atomic[:, q_idx], axis=1)
        scores[f"dsir_spread_{profile}"] = np.ptp(atomic[:, d_idx], axis=1)
        for j, name in enumerate(ATOMIC_FIELDS):
            ref = atomic[a1, j]
            utility_rows.append({"profile": profile, "atomic_field": name, "mean_A1": ref.mean(),
                                 "std_A1": ref.std(), "fraction_at_zero_A1": (ref == 0).mean(),
                                 "fraction_at_one_A1": (ref == 1).mean(),
                                 "direction": dsir_reference[name]["direction"] if name in DSIR_FIELDS
                                              else utility_direction(fitted["specs"][RAW_COLUMNS[j]]["kind"]),
                                 "normalization_kind": "A1_domain_balanced_empirical_rank" if name in DSIR_FIELDS and profile == "dsir_reference_rank" else fitted["specs"][RAW_COLUMNS[j]]["kind"],
                                 "reference_basis": fitted["specs"][RAW_COLUMNS[j]]["basis"]})

        for label, selected in masks.items():
            for domain in np.unique(domains[selected]):
                mask = selected & (domains == domain)
                for scheme in models:
                    q = scores.loc[mask, f"Q_{scheme}_{profile}"].to_numpy()
                    summary_rows.append({"profile": profile, "source": label, "domain": domain, "scheme": scheme,
                                         "n_unique": len(q), "mean": q.mean(), "std": q.std(),
                                         "p10": np.quantile(q, .1), "median": np.median(q), "p90": np.quantile(q, .9),
                                         "Q_rule_mean": scores.loc[mask, f"Q_rule_{scheme}_{profile}"].mean(),
                                         "rule_penalty_mean": scores.loc[mask, f"rule_penalty_{scheme}_{profile}"].mean(),
                                         "atomic_high_low_rate": high_low[mask].mean(),
                                         "atomic_conflict_intensity": intensity[mask].mean(),
                                         "qurater_spread_mean": scores.loc[mask, f"qurater_spread_{profile}"].mean(),
                                         "dsir_spread_mean": scores.loc[mask, f"dsir_spread_{profile}"].mean()})
                    if scheme != "A22":
                        a = scores.loc[mask, f"Q_A22_{profile}"].to_numpy()
                        comparison_rows.append({"profile": profile, "source": label, "domain": domain,
                                                "comparison": f"{scheme}-A22", "n": len(q),
                                                "spearman": rank_correlation(a, q),
                                                "top_fraction_overlap":
                                                (top_mask(a, fraction) & top_mask(q, fraction)).sum() / top_mask(a, fraction).sum(),
                                                "mean_score_difference": (q - a).mean()})
                        rule_a = scores.loc[mask, f"Q_rule_A22_{profile}"].to_numpy()
                        rule_q = scores.loc[mask, f"Q_rule_{scheme}_{profile}"].to_numpy()
                        rule_comparison_rows.append({"profile": profile, "source": label, "domain": domain,
                                                     "comparison": f"{scheme}-A22", "n": len(rule_a),
                                                     "spearman": rank_correlation(rule_a, rule_q),
                                                     "top_fraction_overlap":
                                                     (top_mask(rule_a, fraction) & top_mask(rule_q, fraction)).sum() / top_mask(rule_a, fraction).sum(),
                                                     "mean_score_difference": (rule_q-rule_a).mean()})
        for domain, extension in (("arxiv", "A2_new"), ("github", "A3_new")):
            ref_mask = masks["A1"] & (domains == domain)
            ext_mask = masks[extension] & (domains == domain)
            for scheme in models:
                a = scores.loc[ref_mask, f"Q_{scheme}_{profile}"].to_numpy()
                b = scores.loc[ext_mask, f"Q_{scheme}_{profile}"].to_numpy()
                shift_rows.append({"profile": profile, "scheme": scheme, "domain": domain,
                                   "new_source": extension, "n_A1": len(a), "n_new": len(b),
                                   "mean_shift": b.mean() - a.mean(), "KS_distance": ks_distance(a, b),
                                   "note": "Same-source extension without independent quality labels"})

    for scheme in all_fitted["frozen_original"]["schemes"]:
        for label in ("A1", "A2_new", "A3_new"):
            mask = masks[label]
            for domain in np.unique(domains[mask]):
                selected = mask & (domains == domain)
                a = scores.loc[selected, f"Q_{scheme}_frozen_original"].to_numpy()
                b = scores.loc[selected, f"Q_{scheme}_dsir_reference_rank"].to_numpy()
                normalization_rows.append({"scheme": scheme, "source": label, "domain": domain, "n": len(a),
                                           "spearman": rank_correlation(a, b),
                                           "top_fraction_overlap":
                                           (top_mask(a, fraction) & top_mask(b, fraction)).sum() / top_mask(a, fraction).sum(),
                                           "mean_score_change": (b - a).mean()})

    print("Writing comparison tables", flush=True)
    scores.to_csv(out_dir / "sample_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weight_rows).to_csv(out_dir / "weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(utility_rows).to_csv(out_dir / "atomic_utility_audit_A1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_dir / "domain_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison_rows).to_csv(out_dir / "scheme_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shift_rows).to_csv(out_dir / "extension_stability.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(out_dir / "shrinkage_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(normalization_rows).to_csv(out_dir / "normalization_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_sensitivity_rows).to_csv(out_dir / "qurater_inner_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rule_comparison_rows).to_csv(out_dir / "rule_scheme_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(redundancy_rows).to_csv(out_dir / "group_redundancy.csv", index=False, encoding="utf-8-sig")
    corpus.audit.to_csv(out_dir / "source_audit.csv", index=False, encoding="utf-8-sig")
    corpus.issues.to_csv(out_dir / "parse_issues.csv", index=False, encoding="utf-8-sig")
    utility_dictionary(fitted).to_csv(out_dir / "utility_dictionary.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "fitted_experiment.json", {"version": config["version"], "quality_preprocessor": fitted,
                                                      "dsir_rank_reference": dsir_reference,
                                                      "normalization_profiles": all_fitted})
    write_json(out_dir / "run_metadata.json", {"experiment_config": config, "quality_config": quality_config,
                                                 "n_unique": len(scores), "n_A1": int(a1.sum()),
                                                 "source_counts": corpus.audit.to_dict(orient="records"),
                                                 "interpretation": "Unlabeled sensitivity experiment; no winning quality model inferred"})
    return {"n_unique": len(scores), "output": str(out_dir)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare A22, B23 and B20 on all A1/A2/A3 quality signals")
    parser.add_argument("--data-root", type=Path, default=Path("real_attachments"))
    parser.add_argument("--quality-config", type=Path, default=Path("Code/configs/q1_quality.json"))
    parser.add_argument("--config", type=Path, default=Path("Code/configs/q1_ab_experiment.json"))
    parser.add_argument("--out", type=Path, default=Path("Code/outputs/q1_ab_experiment"))
    parser.add_argument("--bootstrap", type=int, default=None, help="Override weight bootstrap count for a smoke run")
    args = parser.parse_args(argv)
    result = run(args.data_root, args.quality_config, args.config, args.out, args.bootstrap)
    print(f"Completed A/B23/B20 on {result['n_unique']:,} unique records: {result['output']}")


if __name__ == "__main__":
    main()
