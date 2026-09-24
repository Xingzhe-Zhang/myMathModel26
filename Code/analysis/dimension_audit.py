"""Reproduce dimension and weighting diagnostics on the A1 reference set.

This script compares representations; it does not fit a quality ground truth.
Run from the repository root with ``python Code/analysis/dimension_audit.py``.
"""
from __future__ import annotations

import json
import lzma
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from llm_resource.q1.diagnostics import adjusted_correlation, spearman_matrix  # noqa: E402
from llm_resource.q1.preprocess import fit_preprocessor, transform  # noqa: E402
from llm_resource.q1.schema import FIELDS, RAW_COLUMNS, decode_signals  # noqa: E402
from llm_resource.q1.scoring import critic_weights  # noqa: E402


def load_a1() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz"
    raw, domains = [], []
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            values, _, _ = decode_signals(row)
            raw.append(values)
            domains.append(row.get("_source_domain", ""))
    return np.asarray(raw, float), np.asarray(domains)


def group_score(z: np.ndarray) -> np.ndarray:
    col = {name: z[:, j] for j, name in enumerate(FIELDS)}
    content = np.mean(
        [col[name] for name in (
            "fineweb_edu", "fluency_en", "modernbert_readability",
            "modernbert_reasoning", "modernbert_professionalism", "qurater",
        )], axis=0,
    )
    hygiene = np.mean(
        [col[name] for name in (
            "modernbert_cleanliness", "ad_en", "rps_doc_frac_no_alph_words",
            "rps_lines_uppercase_letter_fraction", "rps_lines_ending_with_terminal_punctution_mark",
            "rps_lines_numerical_chars_fraction",
        )] + [0.5 * (col["rps_doc_frac_chars_top_2gram"] + col["rps_doc_frac_chars_top_3gram"])], axis=0,
    )
    structure = np.mean(
        [0.5 * (col["rps_doc_word_count"] + col["rps_doc_num_sentences"]),
         col["rps_doc_unigram_entropy"], col["rps_doc_frac_unique_words"],
         col["rps_doc_mean_word_length"]], axis=0,
    )
    return (content + hygiene + structure) / 3


def main() -> None:
    config = json.loads((CODE_ROOT / "configs/q1_quality.json").read_text(encoding="utf-8"))
    raw, domains = load_a1()
    fitted = fit_preprocessor(raw, domains, config)
    z22, _, _, z25 = transform(raw, domains, fitted)
    names25 = [c.replace("qurater_0", "qurater_writing_style")
               .replace("qurater_1", "qurater_required_expertise")
               .replace("qurater_2", "qurater_facts_trivia")
               .replace("qurater_3", "qurater_educational_value") for c in RAW_COLUMNS]
    w = np.zeros(len(domains))
    for domain in np.unique(domains):
        mask = domains == domain
        w[mask] = 1 / (len(np.unique(domains)) * mask.sum())
    critic = critic_weights(z22, w)

    a = {f: z22[:, i] for i, f in enumerate(FIELDS)}
    scores = pd.DataFrame({
        "domain": domains,
        "Q22_equal": z22.mean(axis=1),
        "Q25_naive_equal": z25.mean(axis=1),
        "Q25_parent_preserved": (z25.sum(axis=1) - z25[:, 6:10].sum(axis=1)) / 22
                                + z25[:, 6:10].mean(axis=1) / 22,
        "Q22_CRITIC": z22 @ critic,
        "Q19_without_DSIR": np.mean([a[f] for f in FIELDS if not f.startswith("dsir_")], axis=0),
        "Q_grouped_core_candidate": group_score(z22),
    })
    out = CODE_ROOT / "outputs/q1_dimensions"
    out.mkdir(parents=True, exist_ok=True)
    by_domain = scores.groupby("domain", sort=True).agg(["mean", "std"])
    by_domain.columns = [f"{name}_{stat}" for name, stat in by_domain.columns]
    by_domain.to_csv(out / "domain_score_comparison.csv")

    q22 = scores["Q22_equal"]
    rows = []
    for name in scores.columns[1:]:
        other = scores[name]
        top_overlaps = []
        for domain in np.unique(domains):
            mask = scores.domain == domain
            q_cut = q22[mask].quantile(.9)
            o_cut = other[mask].quantile(.9)
            a_top = set(np.flatnonzero(mask & (q22 >= q_cut)))
            b_top = set(np.flatnonzero(mask & (other >= o_cut)))
            top_overlaps.append(len(a_top & b_top) / max(len(a_top), 1))
        rows.append({"scenario": name, "spearman_vs_Q22": q22.rank().corr(other.rank()),
                     "pearson_vs_Q22": q22.corr(other),
                     "mean_A1_domain_top_decile_overlap": np.mean(top_overlaps),
                     "mean_absolute_score_change": np.mean(np.abs(q22 - other)),
                     "max_absolute_score_change": np.max(np.abs(q22 - other))})
    pd.DataFrame(rows).to_csv(out / "score_sensitivity.csv", index=False)

    pooled = spearman_matrix(z25)
    adjusted = adjusted_correlation(z25, domains, raw[:, RAW_COLUMNS.index("rps_doc_word_count")])
    selected = [
        ("rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"),
        ("rps_doc_word_count", "rps_doc_num_sentences"),
        ("dsir_books", "dsir_wiki"), ("dsir_books", "dsir_math"), ("dsir_wiki", "dsir_math"),
        ("fineweb_edu", "qurater_educational_value"),
        ("modernbert_readability", "fluency_en"),
        ("modernbert_cleanliness", "ad_en"),
        ("qurater_writing_style", "qurater_required_expertise"),
        ("qurater_writing_style", "qurater_facts_trivia"),
        ("qurater_writing_style", "qurater_educational_value"),
        ("qurater_required_expertise", "qurater_facts_trivia"),
        ("qurater_required_expertise", "qurater_educational_value"),
        ("qurater_facts_trivia", "qurater_educational_value"),
    ]
    pair_rows = []
    for first, second in selected:
        i, j = names25.index(first), names25.index(second)
        pair_rows.append({"first": first, "second": second,
                          "spearman_pooled": pooled[i, j],
                          "rank_correlation_adjusted_domain_loglength": adjusted[i, j]})
    pd.DataFrame(pair_rows).to_csv(out / "selected_correlations.csv", index=False)
    field_stats = pd.DataFrame({"field": names25, "mean": z25.mean(axis=0), "std": z25.std(axis=0),
                                "at_zero": (z25 == 0).mean(axis=0), "at_one": (z25 == 1).mean(axis=0)})
    field_stats.to_csv(out / "field_utility_stats.csv", index=False)
    q = z22.mean(axis=1)
    q_var = np.var(q)
    shares = [np.cov(z22[:, j], q, bias=True)[0, 1] / (22 * q_var) for j in range(len(FIELDS))]
    pd.DataFrame({"field": FIELDS, "nominal_weight": np.full(len(FIELDS), 1 / 22),
                  "score_variance_share": shares}).to_csv(out / "effective_influence.csv", index=False)
    pd.DataFrame({"field": FIELDS, "primary_weight": np.full(len(FIELDS), 1 / 22),
                  "CRITIC_comparison_weight": critic}).to_csv(out / "weights.csv", index=False)
    print(f"A1 records: {len(domains):,}; output: {out}")
    print(pd.DataFrame(pair_rows).round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
