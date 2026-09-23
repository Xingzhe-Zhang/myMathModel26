"""Command line pipeline for Q1 tasks 1 and 2.

The pipeline is deliberately two-stage:
1. fit empirical reference scales on A1 only;
2. freeze those rules and score the de-duplicated A1/A2/A3 union.

The resulting Q is a transparent quality preference proxy. It is not a
ground-truth label and is not claimed to be a causal training-utility score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import source_masks, summaries
from .data import read_corpus
from .diagnostics import exploratory_factors, length_strata, pair_table, permutation_pairs
from .preprocess import fit_preprocessor, transform, utility_dictionary, validate_config
from .scoring import critic_weights, score_all
from .schema import FIELDS, RAW_COLUMNS


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Q1 quality and conflict analysis")
    parser.add_argument("--data-root", type=Path, required=True, help="real_attachments directory")
    parser.add_argument("--config", type=Path, required=True, help="quality configuration JSON")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--bootstrap", type=int, default=None, help="override domain mean bootstrap repetitions")
    parser.add_argument("--permutations", type=int, default=None, help="override conflict permutation repetitions")
    return parser.parse_args(argv)


def run(config_path: Path, data_root: Path, out_dir: Path, bootstrap=None, permutations=None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bootstrap is not None:
        config["bootstrap_repeats"] = int(bootstrap)
    if permutations is not None:
        config["conflict"]["permutation_repeats"] = int(permutations)
    validate_config(config)
    rng = np.random.default_rng(int(config["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = read_corpus(data_root)
    masks = source_masks(corpus)
    a1 = masks["A1"]
    if int(a1.sum()) == 0:
        raise RuntimeError("A1 reference set is empty")

    fitted = fit_preprocessor(corpus.raw[a1], corpus.records.loc[a1, "domain"].to_numpy(), config)
    z, valid, applicable, _ = transform(
        corpus.raw, corpus.records["domain"].to_numpy(), fitted,
        domain_allowance=bool(config["reference"].get("domain_allowance", True)),
    )
    reference_z = z[a1]
    row_weights = np.zeros(int(a1.sum()), dtype=float)
    ref_domains = corpus.records.loc[a1, "domain"].to_numpy()
    for domain in np.unique(ref_domains):
        row_weights[ref_domains == domain] = 1 / (len(np.unique(ref_domains)) * (ref_domains == domain).sum())
    critic = critic_weights(reference_z, row_weights)
    score_arrays, score_model = score_all(
        z, corpus.records["domain"].to_numpy(), a1, corpus.ordinal, config, critic
    )

    scores = pd.DataFrame(score_arrays)
    scores.insert(0, "uid", corpus.records["uid"].to_numpy())
    scores.insert(1, "id", corpus.records["id"].to_numpy())
    scores.insert(2, "domain", corpus.records["domain"].to_numpy())
    scores.insert(3, "first_source", corpus.records["first_source"].to_numpy())
    scores.insert(4, "source_shards", corpus.records["source_shard"].to_numpy())
    scores["valid_coverage"] = valid.mean(axis=1)
    scores["applicable_coverage"] = applicable.mean(axis=1)
    scores["is_A1_reference"] = a1
    member_labels = corpus.members.groupby("row")["source"].apply(lambda x: ";".join(sorted(set(x)))).to_dict()
    scores["source_membership"] = [member_labels.get(i, "") for i in range(len(scores))]
    scores.to_csv(out_dir / "sample_scores.csv", index=False, encoding="utf-8-sig")

    # The issue table is intentionally separate from scores: malformed or
    # conflicting source records must remain auditable.
    corpus.audit.to_csv(out_dir / "source_audit.csv", index=False, encoding="utf-8-sig")
    corpus.issues.to_csv(out_dir / "parse_issues.csv", index=False, encoding="utf-8-sig")

    lengths = corpus.raw[:, RAW_COLUMNS.index("rps_doc_word_count")]
    summary, scenarios, comparisons = summaries(
        scores, scores["domain"].to_numpy(), lengths, masks,
        int(config.get("bootstrap_repeats", 200)), rng,
    )
    summary.to_csv(out_dir / "domain_summary.csv", index=False, encoding="utf-8-sig")
    scenarios.to_csv(out_dir / "score_sensitivity_by_domain.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(out_dir / "extension_comparison.csv", index=False, encoding="utf-8-sig")
    utility_dictionary(fitted).to_csv(out_dir / "utility_dictionary.csv", index=False, encoding="utf-8-sig")

    # Conflict diagnostics use strata learned from A1 only. A2/A3 scores are
    # then evaluated with exactly the same bins and thresholds.
    a1_lengths = np.log1p(np.maximum(lengths[a1], 0))
    finite = np.isfinite(a1_lengths)
    boundaries = np.quantile(a1_lengths[finite], np.linspace(.2, .8, 4)) if finite.any() else np.zeros(3)
    boundaries = np.unique(boundaries).tolist()
    strata = length_strata(scores["domain"].to_numpy(), lengths, boundaries)
    for label, selected in (("A1", masks["A1"]), ("A2", masks["A2"]), ("A3", masks["A3"]), ("union", masks["union"])):
        table = pair_table(z[selected], strata[selected], config["conflict"]["high"], config["conflict"]["low"], label)
        table.to_csv(out_dir / f"conflict_pairs_{label}.csv", index=False, encoding="utf-8-sig")
    permutation = permutation_pairs(z[a1], scores.loc[a1, "domain"].to_numpy(), strata[a1], config["conflict"], rng)
    permutation.to_csv(out_dir / "conflict_permutation_A1.csv", index=False, encoding="utf-8-sig")

    loadings, phi, parallel, factor_meta = exploratory_factors(z[a1], scores.loc[a1, "domain"].to_numpy(), config["factor_analysis"], rng)
    loadings.to_csv(out_dir / "factor_loadings_A1.csv", encoding="utf-8-sig")
    phi.to_csv(out_dir / "factor_correlation_A1.csv", encoding="utf-8-sig")
    parallel.to_csv(out_dir / "factor_parallel_analysis_A1.csv", index=False, encoding="utf-8-sig")

    _write_json(out_dir / "fitted_scoring_model.json", {"preprocessor": fitted, "score_model": score_model, "factor_meta": factor_meta, "length_strata_boundaries_log1p_word_count": boundaries})
    _write_json(out_dir / "run_metadata.json", {
        "config": config, "data_root": str(data_root), "n_unique_rows": len(scores),
        "n_A1_reference": int(a1.sum()), "n_A2_rows": int(masks["A2"].sum()),
        "n_A3_rows": int(masks["A3"].sum()), "fields": FIELDS,
        "raw_columns": RAW_COLUMNS, "status": "completed; Q is a rule-based candidate proxy",
    })
    return {"n_unique_rows": len(scores), "n_A1": int(a1.sum()), "out": str(out_dir), "summary": summary}


def main(argv=None):
    args = _parse_args(argv)
    result = run(args.config, args.data_root, args.out, args.bootstrap, args.permutations)
    print(f"Completed Q1 tasks 1-2: {result['n_unique_rows']:,} unique rows; outputs: {result['out']}")


if __name__ == "__main__":
    main()
