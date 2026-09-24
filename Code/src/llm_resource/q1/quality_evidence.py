"""Audit A16--A18 as evidence for (but not measurements of) RegMix quality.

A18 contains raw validation-shard text without the 22 Meta-rater signals.  The
statistics here are coverage and text-integrity diagnostics, not a new Q score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd


def audit_regmix_text(mapping_path: Path, summary_path: Path, sample_path: Path,
                      domains: list[str]) -> tuple[pd.DataFrame, dict]:
    """Stream all A18 records and reconcile them with A16 and A17 by domain."""
    mapping = pd.read_csv(mapping_path)
    summary = pd.read_csv(summary_path)
    required_map = {"mixture_domain", "quality_domain", "mapping_type"}
    required_summary = {"domain", "source_path", "sample_rows", "sample_bytes_requested", "avg_text_chars"}
    if not required_map.issubset(mapping) or not required_summary.issubset(summary):
        raise ValueError("A16 or A17 is missing required columns")
    expected = set(domains)
    if (len(domains) != 17 or len(mapping) != 17 or len(summary) != 17
            or mapping["mixture_domain"].duplicated().any() or summary["domain"].duplicated().any()
            or set(mapping["mixture_domain"]) != expected or set(summary["domain"]) != expected):
        raise ValueError("A16 and A17 must each cover the 17 unique mixture domains")

    counts = Counter()
    characters = Counter()
    empty = Counter()
    replacement = Counter()
    nul = Counter()
    duplicates = Counter()
    paths = defaultdict(set)
    lengths = defaultdict(list)
    seen = defaultdict(set)
    all_keys = set()
    with lzma.open(sample_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"A18 invalid JSON at line {line_number}") from exc
            if not isinstance(row, dict) or not {"text", "_source_domain", "_source_path"}.issubset(row):
                raise ValueError(f"A18 missing required fields at line {line_number}")
            domain, path, value = row["_source_domain"], row["_source_path"], row["text"]
            if domain not in expected or not isinstance(path, str) or not isinstance(value, str):
                raise ValueError(f"A18 invalid domain/path/text at line {line_number}")
            all_keys.update(row)
            counts[domain] += 1
            paths[domain].add(path)
            size = len(value)
            lengths[domain].append(size)
            characters[domain] += size
            empty[domain] += size == 0
            replacement[domain] += value.count("\ufffd")
            nul[domain] += value.count("\x00")
            digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
            duplicates[domain] += digest in seen[domain]
            seen[domain].add(digest)

    rows = []
    by_summary = summary.set_index("domain")
    by_mapping = mapping.set_index("mixture_domain")
    for domain in domains:
        a17 = by_summary.loc[domain]
        a16 = by_mapping.loc[domain]
        n = counts[domain]
        if n == 0:
            raise ValueError(f"A18 has no text for {domain}")
        observed_mean = characters[domain] / n
        row_match = n == int(a17["sample_rows"])
        path_match = paths[domain] == {str(a17["source_path"])}
        mean_match = abs(observed_mean - float(a17["avg_text_chars"])) <= 0.011
        rows.append({
            "domain": domain, "mapping_type": a16["mapping_type"],
            "quality_domain": a16["quality_domain"],
            "a17_sample_rows": int(a17["sample_rows"]),
            "a17_sample_bytes_requested": int(a17["sample_bytes_requested"]),
            "a17_avg_text_chars": float(a17["avg_text_chars"]),
            "a17_source_path": a17["source_path"],
            "a18_rows": n, "a18_mean_text_chars": observed_mean,
            "a18_median_text_chars": float(np.median(lengths[domain])),
            "a18_empty_rows": empty[domain], "a18_exact_duplicate_rows": duplicates[domain],
            "a18_replacement_chars": replacement[domain], "a18_nul_chars": nul[domain],
            "a18_distinct_source_paths": len(paths[domain]),
            "a18_rows_match_a17": row_match, "a18_path_matches_a17": path_match,
            "a18_mean_chars_matches_a17": mean_match,
            "a18_source_is_valid_shard": all(path.startswith("valid/") for path in paths[domain]),
        })
    result = pd.DataFrame(rows)
    meta = {
        "a18_total_rows": int(sum(counts.values())),
        "a18_domains": int(len(counts)),
        "a18_keys": sorted(all_keys),
        "a18_precomputed_quality_signals": False,
        "a17_a18_row_path_mean_reconciled": bool(result[["a18_rows_match_a17", "a18_path_matches_a17",
                                                     "a18_mean_chars_matches_a17"]].all().all()),
        "a18_all_from_valid_shards": bool(result["a18_source_is_valid_shard"].all()),
        "role": "text coverage and integrity evidence; not a 17-domain Q measurement",
    }
    return result, meta
