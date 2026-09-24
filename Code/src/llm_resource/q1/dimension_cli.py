"""Run and compare the 22-field and semantic dimension schemes on A1/A2/A3."""
from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import source_masks
from .data import read_corpus
from .diagnostics import adjusted_correlation, spearman_matrix
from .dimension_evaluation import evaluate_independent_labels, summarize_unlabeled
from .dimensions import ATOMIC_FIELDS, score_dimension_views, validate_dimension_config
from .preprocess import fit_preprocessor, transform, validate_config
from .schema import RAW_COLUMNS


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_template(corpus, data_root: Path, out: Path, per_domain: int, seed: int) -> int:
    a1 = corpus.records[corpus.records.first_source == "A1"]
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in a1.groupby("domain", sort=True):
        indexes = rng.choice(group.index.to_numpy(), min(per_domain, len(group)), replace=False)
        picks.extend(indexes.tolist())
    chosen = corpus.records.loc[picks, ["uid", "id", "domain"]]
    identities = {(row.domain, str(row.id)): row.uid for row in chosen.itertuples(index=False)}
    content = {}
    path = data_root / "A_data_value/slimpajama_quality_signal_sample.jsonl.xz"
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            uid = identities.get((row.get("_source_domain"), str(row.get("id"))))
            if uid is not None:
                content[uid] = row.get("content", "")
                if len(content) == len(identities):
                    break
    template = chosen.copy()
    template["content_preview"] = [content.get(uid, "")[:8000] for uid in template.uid]
    template["content_length"] = [len(content.get(uid, "")) for uid in template.uid]
    template["preview_truncated"] = template.content_length > 8000
    template["quality"] = ""
    template["defect"] = ""
    template["reviewer_id"] = ""
    template.to_csv(out, index=False, encoding="utf-8-sig")
    with out.with_name("blind_review_fulltext.jsonl").open("w", encoding="utf-8") as stream:
        for uid in template.uid:
            stream.write(json.dumps({"uid": uid, "content": content.get(uid, "")}, ensure_ascii=False) + "\n")
    return len(template)


def run(data_root: Path, quality_config_path: Path, dimension_config_path: Path,
        out_dir: Path, labels_path: Path | None = None, review_per_domain: int = 60) -> dict:
    quality_config = json.loads(quality_config_path.read_text(encoding="utf-8"))
    dimension_config = json.loads(dimension_config_path.read_text(encoding="utf-8"))
    validate_config(quality_config)
    validate_dimension_config(dimension_config)
    if not 0 <= review_per_domain <= 1000:
        raise ValueError("review_per_domain must be in [0,1000]")
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = read_corpus(data_root)
    masks = source_masks(corpus)
    a1 = masks["A1"]
    domains = corpus.records.domain.to_numpy()
    fitted = fit_preprocessor(corpus.raw[a1], domains[a1], quality_config)
    z22, valid22, _, z25 = transform(corpus.raw, domains, fitted,
                                    domain_allowance=bool(quality_config["reference"].get("domain_allowance", True)))
    values, weights = score_dimension_views(z22, z25, dimension_config)
    values.insert(0, "uid", corpus.records.uid.to_numpy())
    values.insert(1, "domain", domains)
    values.insert(2, "id", corpus.records.id.to_numpy())
    values["is_A1_reference"] = a1
    values["valid_22_field_fraction"] = valid22.mean(axis=1)
    members = corpus.members.groupby("row").source.apply(lambda x: ";".join(sorted(set(x)))).to_dict()
    values["source_membership"] = [members.get(i, "") for i in range(len(values))]
    values.to_csv(out_dir / "sample_scores_both.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(out_dir / "dimension_weights.csv", index=False, encoding="utf-8-sig")

    fraction = dimension_config["human_evaluation"]["selection_fraction_per_domain"]
    summary, ranking, extension = summarize_unlabeled(values, masks, fraction)
    summary.to_csv(out_dir / "domain_scores_both.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(out_dir / "A1_ranking_comparison.csv", index=False, encoding="utf-8-sig")
    extension.to_csv(out_dir / "extension_stability.csv", index=False, encoding="utf-8-sig")

    ref_atomic = z25[a1]
    stats = pd.DataFrame({"atomic_field": ATOMIC_FIELDS, "mean_A1": ref_atomic.mean(axis=0),
                          "std_A1": ref_atomic.std(axis=0), "fraction_at_one_A1": (ref_atomic == 1).mean(axis=0),
                          "fraction_at_zero_A1": (ref_atomic == 0).mean(axis=0)})
    stats.to_csv(out_dir / "atomic_utility_stats_A1.csv", index=False, encoding="utf-8-sig")
    pooled = spearman_matrix(ref_atomic)
    adjusted = adjusted_correlation(ref_atomic, domains[a1], corpus.raw[a1, RAW_COLUMNS.index("rps_doc_word_count")])
    correlations = [{"field_a": ATOMIC_FIELDS[i], "field_b": ATOMIC_FIELDS[j],
                     "spearman_A1": pooled[i, j], "rank_corr_adjusted_domain_length_A1": adjusted[i, j]}
                    for i in range(len(ATOMIC_FIELDS)) for j in range(i+1, len(ATOMIC_FIELDS))]
    pd.DataFrame(correlations).to_csv(out_dir / "atomic_correlations_A1.csv", index=False, encoding="utf-8-sig")
    selected_pairs = [
        ("dsir_books", "dsir_wiki"), ("dsir_books", "dsir_math"), ("dsir_wiki", "dsir_math"),
        ("rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"),
        ("fineweb_edu", "qurater_educational_value"),
        ("modernbert_readability", "fluency_en"),
        ("qurater_writing_style", "qurater_required_expertise"),
    ]
    extension_pairs = []
    for source in ("A1", "A2_new", "A3_new"):
        subset = z25[masks[source]]
        for first, second in selected_pairs:
            x = pd.Series(subset[:, ATOMIC_FIELDS.index(first)]).rank().to_numpy()
            y = pd.Series(subset[:, ATOMIC_FIELDS.index(second)]).rank().to_numpy()
            extension_pairs.append({"source": source, "n": len(subset), "field_a": first, "field_b": second,
                                    "spearman": float(np.corrcoef(x, y)[0, 1])})
    pd.DataFrame(extension_pairs).to_csv(out_dir / "selected_correlations_extensions.csv", index=False, encoding="utf-8-sig")

    review_count = 0
    review_path = out_dir / "blind_review_template.csv"
    if review_path.exists():
        review_count = len(pd.read_csv(review_path, usecols=["uid"]))
    elif review_per_domain:
        review_count = _review_template(corpus, data_root, review_path,
                                        review_per_domain, quality_config["seed"])
    if labels_path:
        result, detail = evaluate_independent_labels(values, pd.read_csv(labels_path),
                                                     dimension_config, quality_config["seed"])
        detail.to_csv(out_dir / "independent_label_domain_results.csv", index=False, encoding="utf-8-sig")
    else:
        result = {"status": "no_independent_labels", "decision": "undetermined",
                  "reason": "A1/A2/A3 contain quality signals but no independent sample quality or training-utility labels"}
    _write_json(out_dir / "selection_result.json", result)
    _write_json(out_dir / "run_metadata.json", {"quality_config": quality_config, "dimension_config": dimension_config,
                "n_unique": len(values), "n_A1": int(a1.sum()), "n_review_template": review_count,
                "decision_scope": "human labels or controlled proxy training required for quality-model selection"})
    return {"n_unique": len(values), "n_A1": int(a1.sum()), "decision": result["decision"], "out": str(out_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare 22-field and semantic dimension schemes")
    parser.add_argument("--data-root", type=Path, default=Path("real_attachments"))
    parser.add_argument("--quality-config", type=Path, default=Path("Code/configs/q1_quality.json"))
    parser.add_argument("--dimension-config", type=Path, default=Path("Code/configs/q1_dimension_schemes.json"))
    parser.add_argument("--out", type=Path, default=Path("Code/outputs/q1_dimension_schemes"))
    parser.add_argument("--labels", type=Path, help="Adjudicated, blinded A1 labels: uid,quality,defect")
    parser.add_argument("--review-per-domain", type=int, default=60)
    args = parser.parse_args(argv)
    result = run(args.data_root, args.quality_config, args.dimension_config,
                 args.out, args.labels, args.review_per_domain)
    print(f"Compared both schemes on {result['n_unique']:,} unique records; decision: {result['decision']}; outputs: {result['out']}")


if __name__ == "__main__":
    main()
