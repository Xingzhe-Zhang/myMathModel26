"""Streaming source reader and provenance-aware duplicate audit."""
from dataclasses import dataclass
from pathlib import Path
import collections
import hashlib
import json
import lzma
import numpy as np
import pandas as pd
from .schema import FIELDS, decode_signals


@dataclass
class Corpus:
    raw: np.ndarray
    ordinal: np.ndarray
    records: pd.DataFrame
    members: pd.DataFrame
    audit: pd.DataFrame
    issues: pd.DataFrame
    source_paths: list


def stable_fraction(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2**64


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_corpus(data_root):
    a = Path(data_root) / "A_data_value"
    sources = [("A1", a / "slimpajama_quality_signal_sample.jsonl.xz")]
    sources += [("A2", p) for p in sorted((a / "slimpajama_quality_extended").glob("arxiv_*.xz"))]
    sources += [("A3", p) for p in sorted((a / "slimpajama_quality_extended").glob("github_*.xz"))]
    if len(sources) < 3 or not all(p.exists() for _, p in sources):
        raise FileNotFoundError("A1 and both arxiv/github extension sources are required")
    raw, ordinal, records, members, audits, issues = [], [], [], [], [], []
    known, identities = {}, collections.defaultdict(set)
    for label, path in sources:
        count, overlap, internal, mismatches, invalid = 0, 0, 0, 0, 0
        local = set()
        with lzma.open(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                count += 1
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Cannot silently discard invalid JSON: {path}:{line_no}") from exc
                domain = row.get("_source_domain") or path.name.split("_", 1)[0]
                identity = row.get("id")
                if not identity:
                    raise ValueError(f"Missing identity: {path}:{line_no}")
                source = Path(row.get("_source_path", "")).name if label == "A1" else path.name.removeprefix(domain + "_")
                source = source.removesuffix(".xz")
                identity = str(identity)
                signature = hashlib.sha256(json.dumps({f: row.get(f) for f in FIELDS}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                text_hash = hashlib.sha256(row["content"].encode()).hexdigest() if isinstance(row.get("content"), str) else ""
                key = (domain, identity, source, signature)
                base = (domain, identity)
                if base in identities and key not in identities[base]:
                    mismatches += 1
                    issues.append({"source": label, "line": line_no, "id": identity, "issue": "identity_conflict_retained"})
                identities[base].add(key)
                if key in known and text_hash and records[known[key]]["content_sha256"] and text_hash != records[known[key]]["content_sha256"]:
                    raise ValueError(f"Content differs for an otherwise identical identity: {identity}")
                if key in known:
                    pos = known[key]
                    if key in local:
                        internal += 1
                    else:
                        overlap += 1
                else:
                    vals, levels, errors = decode_signals(row)
                    invalid += bool(errors)
                    issues.extend({"source": label, "line": line_no, "id": identity, "issue": e} for e in errors)
                    uid = hashlib.sha256("|".join(key).encode()).hexdigest()[:24]
                    pos = len(raw)
                    known[key] = pos
                    raw.append(vals)
                    ordinal.append(levels)
                    records.append({"uid": uid, "domain": domain, "id": identity, "source_shard": source,
                                    "first_source": label, "content_sha256": text_hash, "has_content": bool(text_hash)})
                local.add(key)
                members.append({"row": pos, "source": label, "source_file": path.name, "line": line_no})
        audits.append({"source": label, "file": path.name, "records": count, "parsed": count,
                       "internal_duplicates": internal, "cross_file_overlap": overlap,
                       "identity_conflicts": mismatches, "new_records_with_invalid_fields": invalid})
        print(f"Read {label}: {count:,} records, {overlap:,} cross-file duplicates", flush=True)
    return Corpus(np.asarray(raw, float), np.asarray(ordinal, float), pd.DataFrame(records),
                  pd.DataFrame(members), pd.DataFrame(audits),
                  pd.DataFrame(issues, columns=["source", "line", "id", "issue"]), sources)
