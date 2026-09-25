"""Compare preserved equal-weight Q1 records with the A22/B23/B20 experiment.

Run from the repository root. Neither input CSV is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_resource.q1.ab_experiment import rank_correlation, top_mask  # noqa: E402

SCHEMES = ("A22", "B23_flat", "B20_hierarchical")
PROFILES = ("frozen_original", "dsir_reference_rank")
SOURCES = ("A1", "A2", "A3", "A2_new", "A3_new", "union")


def source_mask(membership: pd.Series, label: str) -> np.ndarray:
    if label == "union":
        return np.ones(len(membership), bool)
    members = membership.str.split(";")
    if label.endswith("_new"):
        source = label.removesuffix("_new")
        return (members.map(lambda values: source in values and "A1" not in values)).to_numpy()
    return members.map(lambda values: label in values).to_numpy()


def compare(equal_csv: Path, experiment_csv: Path, output_csv: Path, fraction: float = .2) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("Selection fraction must be in (0,1)")
    equal = pd.read_csv(equal_csv, usecols=["uid", "domain", "source_membership",
                                            "Q_equal", "Q_primary", "Q_critic", "Q_rule"])
    columns = ["uid", "domain", "source_membership"]
    columns += [f"{prefix}{scheme}_{profile}" for profile in PROFILES for scheme in SCHEMES
                for prefix in ("Q_", "Q_rule_")]
    experiment = pd.read_csv(experiment_csv, usecols=columns)
    if equal.uid.duplicated().any() or experiment.uid.duplicated().any():
        raise ValueError("Duplicate UID in an input score table")
    merged = equal.merge(experiment, on="uid", how="inner", validate="one_to_one", suffixes=("_equal", "_experiment"))
    if len(merged) != len(equal) or len(merged) != len(experiment):
        raise ValueError("Equal-weight and A/B outputs do not cover the same unique texts")
    for column in ("domain", "source_membership"):
        if not merged[f"{column}_equal"].equals(merged[f"{column}_experiment"]):
            raise ValueError(f"Source alignment differs: {column}")
    primary_error = float(np.max(np.abs(merged.Q_equal - merged.Q_primary)))
    critic_error = float(np.max(np.abs(merged.Q_critic - merged.Q_A22_frozen_original)))
    if primary_error > 1e-12 or critic_error > 1e-12:
        raise ValueError("Preserved baseline does not match the current reference scales")

    domains = merged.domain_equal.to_numpy()
    membership = merged.source_membership_equal
    rows = []
    for source in SOURCES:
        source_selected = source_mask(membership, source)
        for domain in np.unique(domains[source_selected]):
            selected = source_selected & (domains == domain)
            for kind, baseline_column, prefix in (("base", "Q_equal", "Q_"),
                                                  ("rule", "Q_rule", "Q_rule_")):
                baseline = merged.loc[selected, baseline_column].to_numpy()
                baseline_top = top_mask(baseline, fraction)
                for profile in PROFILES:
                    for scheme in SCHEMES:
                        candidate = merged.loc[selected, f"{prefix}{scheme}_{profile}"].to_numpy()
                        rows.append({"source": source, "domain": domain, "score_kind": kind,
                                     "profile": profile, "scheme": scheme, "n": len(baseline),
                                     "equal_mean": baseline.mean(), "candidate_mean": candidate.mean(),
                                     "mean_difference": (candidate - baseline).mean(),
                                     "mean_absolute_difference": np.abs(candidate - baseline).mean(),
                                     "spearman_vs_equal": rank_correlation(baseline, candidate),
                                     "top_fraction_overlap_vs_equal":
                                     (baseline_top & top_mask(candidate, fraction)).sum() / baseline_top.sum()})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    metadata = {"n_unique": len(merged), "equal_csv": str(equal_csv),
                "experiment_csv": str(experiment_csv), "selection_fraction_per_domain": fraction,
                "max_equal_vs_primary_error": primary_error,
                "max_critic_vs_A22_error": critic_error,
                "interpretation": "Unlabeled ranking and score sensitivity, not training-utility validity"}
    output_csv.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare preserved 22-field equal baseline with A22/B23/B20")
    parser.add_argument("--equal", type=Path, default=Path("Code/outputs/q1/sample_scores.csv"))
    parser.add_argument("--experiment", type=Path, default=Path("Code/outputs/q1_ab_experiment/sample_scores.csv"))
    parser.add_argument("--out", type=Path, default=Path("Code/outputs/q1_ab_experiment/equal_baseline_comparison.csv"))
    args = parser.parse_args()
    result = compare(args.equal, args.experiment, args.out)
    print(f"Compared {result['n_unique']:,} aligned records; wrote {args.out}")


if __name__ == "__main__":
    main()
