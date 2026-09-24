"""Unlabeled sensitivity audit for zero-weight groups in Q_semantic.

Run after compare_dimensions.py. This tests ranking sensitivity, not quality.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = {
    "structure": "G_structure_diagnostic",
    "complexity": "G_complexity_diagnostic",
    "target_relevance": "G_target_relevance_diagnostic",
}
FRACTION = 0.2


def top_mask(values: np.ndarray) -> np.ndarray:
    chosen = np.zeros(len(values), dtype=bool)
    count = max(1, int(np.ceil(FRACTION * len(values))))
    chosen[np.argsort(-values, kind="stable")[:count]] = True
    return chosen


def audit(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"domain", "Q_semantic", "is_A1_reference", "source_membership", *GROUPS.values()}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    membership = frame.source_membership.fillna("")
    sources = {
        "A1": frame.is_A1_reference.astype(bool),
        "A2_new": membership.str.contains(r"(?:^|;)A2(?:;|$)", regex=True) & ~frame.is_A1_reference,
        "A3_new": membership.str.contains(r"(?:^|;)A3(?:;|$)", regex=True) & ~frame.is_A1_reference,
    }
    rows = []
    for source, selected in sources.items():
        for domain, part in frame.loc[selected].groupby("domain", sort=True):
            base = part.Q_semantic.to_numpy(float)
            base_top = top_mask(base)
            base_rank = pd.Series(base).rank().to_numpy()
            for group, column in GROUPS.items():
                group_values = part[column].to_numpy(float)
                group_rank = pd.Series(group_values).rank().to_numpy()
                group_corr = float(np.corrcoef(base_rank, group_rank)[0, 1])
                for budget in (0.05, 0.10, 0.20):
                    candidate = (1-budget) * base + budget * group_values
                    candidate_rank = pd.Series(candidate).rank().to_numpy()
                    rows.append({
                        "source": source, "domain": domain, "n": len(part), "group": group,
                        "budget_added": budget, "group_std": float(group_values.std()),
                        "group_spearman_with_B": group_corr,
                        "rank_spearman_B_vs_variant": float(np.corrcoef(base_rank, candidate_rank)[0, 1]),
                        "top20_overlap": float((base_top & top_mask(candidate)).sum() / base_top.sum()),
                    })
            candidate = 0.7*base + 0.1*sum(part[column].to_numpy(float) for column in GROUPS.values())
            candidate_rank = pd.Series(candidate).rank().to_numpy()
            rows.append({
                "source": source, "domain": domain, "n": len(part), "group": "all_three",
                "budget_added": 0.30, "group_std": np.nan,
                "group_spearman_with_B": np.nan,
                "rank_spearman_B_vs_variant": float(np.corrcoef(base_rank, candidate_rank)[0, 1]),
                "top20_overlap": float((base_top & top_mask(candidate)).sum() / base_top.sum()),
            })
    return pd.DataFrame(rows)


def core_influence(frame: pd.DataFrame) -> pd.DataFrame:
    """Decompose B's within-domain score variance into its three core groups."""
    c = 21 / 220  # non-QuRating weight per core concept in dimension config v1
    q = 1 / 66    # QuRating weight per included subscale
    expression = c*(3*frame.G_expression-frame.qurater_writing_style) + q*frame.qurater_writing_style
    knowledge = c*(4*frame.G_knowledge-frame.qurater_facts_trivia-frame.qurater_educational_value)
    knowledge += q*(frame.qurater_facts_trivia+frame.qurater_educational_value)
    cleanliness = c*6*frame.G_cleanliness
    components = {"expression": expression, "knowledge": knowledge, "cleanliness": cleanliness}
    if not np.allclose(sum(components.values()), frame.Q_semantic, atol=1e-10):
        raise ValueError("Core decomposition no longer matches configured Q_semantic")
    rows = []
    for domain, part in frame.loc[frame.is_A1_reference].groupby("domain", sort=True):
        score = part.Q_semantic.to_numpy(float)
        variance = float(np.var(score))
        for group, values in components.items():
            x = values.loc[part.index].to_numpy(float)
            rows.append({"domain": domain, "n_A1": len(part), "core_group": group,
                         "nominal_weight": {"expression": 0.20606060606060606,
                                            "knowledge": 0.2212121212121212,
                                            "cleanliness": 0.5727272727272728}[group],
                         "component_std": float(x.std()),
                         "covariance_share_of_B_variance": float(np.cov(x, score, bias=True)[0, 1] / variance)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("Code/outputs/q1_dimension_schemes/sample_scores_both.csv"))
    parser.add_argument("--out", type=Path, default=Path("Code/outputs/q1_dimension_reassessment"))
    args = parser.parse_args()
    cols = ["domain", "Q_semantic", "is_A1_reference", "source_membership",
            "G_expression", "G_knowledge", "G_cleanliness", "qurater_writing_style",
            "qurater_facts_trivia", "qurater_educational_value", *GROUPS.values()]
    frame = pd.read_csv(args.scores, usecols=cols)
    detail = audit(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "zero_group_sensitivity_by_domain.csv", index=False, encoding="utf-8-sig")
    macro = detail.groupby(["source", "group", "budget_added"], as_index=False).agg(
        domains=("domain", "count"), mean_group_std=("group_std", "mean"),
        mean_group_spearman_with_B=("group_spearman_with_B", "mean"),
        macro_rank_spearman=("rank_spearman_B_vs_variant", "mean"),
        macro_top20_overlap=("top20_overlap", "mean"),
    )
    macro.to_csv(args.out / "zero_group_sensitivity_macro.csv", index=False, encoding="utf-8-sig")
    core_influence(frame).to_csv(args.out / "core_variance_influence_A1.csv", index=False, encoding="utf-8-sig")
    print(f"Audited {len(frame):,} records; {len(detail)} domain-variant rows; outputs: {args.out}")


if __name__ == "__main__":
    main()
