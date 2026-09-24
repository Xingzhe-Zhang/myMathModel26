"""Audit common A1/A18 signals before any transfer into the Loss experiment."""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from .public_quality_models import MODEL_REGISTRY, infer_logits
from .reconstructed_signals import RPS_FIELDS, rps_signals
from .schema import FIELDS, RAW_COLUMNS, decode_signals, softmax


SHARED_FIELDS = tuple(RPS_FIELDS) + tuple(MODEL_REGISTRY)


def _sample(path: Path, content_key: str, per_domain: int, expected: set[str], *,
            allow_shortfall: bool = False, coverage: dict | None = None) -> list[dict]:
    """Stable hash sample; optionally retain all valid rows in undersized domains."""
    import heapq

    if per_domain < 1:
        raise ValueError("per_domain must be positive")
    heaps: dict[str, list[tuple[int, int, dict]]] = {d: [] for d in expected}
    available = {d: 0 for d in expected}
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            record = json.loads(line)
            domain = record.get("_source_domain")
            if domain not in heaps:
                continue
            text = record.get(content_key)
            if not isinstance(text, str):
                raise ValueError(f"{path.name}:{line_number + 1} missing text")
            if content_key == "text" and not str(record.get("_source_path", "")).startswith("valid/"):
                raise ValueError(f"{path.name}:{line_number + 1} is not a RegMix validation-shard text")
            available[domain] += 1
            identity = (str(record.get("id", "")) if content_key == "content" else
                        str(record.get("_source_path", "")) + "\0" + text)
            key = int.from_bytes(hashlib.sha256((domain + "\0" + identity).encode("utf-8")).digest()[:8], "big")
            item = (-key, -line_number, record)
            heap = heaps[domain]
            if len(heap) < per_domain:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    missing = {d: len(heap) for d, heap in heaps.items()
               if not heap or (not allow_shortfall and len(heap) < per_domain)}
    if missing:
        raise ValueError(f"Insufficient text rows per domain: {missing}")
    if coverage is not None:
        coverage.update({d: {"available": available[d], "selected": len(heaps[d]),
                             "target": per_domain} for d in sorted(expected)})
    return [item[2] for domain in sorted(heaps) for item in sorted(heaps[domain], reverse=True)]


def _model_scalar(field: str, logits: np.ndarray) -> np.ndarray:
    if field == "fineweb_edu":
        return logits[:, 0]
    if field == "fluency_en":
        return softmax(logits)[:, 1]
    return softmax(logits) @ (np.arange(6) / 5)


def _official_scalar(field: str, records: list[dict]) -> np.ndarray:
    result = []
    for record in records:
        value = record.get(field)
        if field in MODEL_REGISTRY:
            result.append(float(_model_scalar(field, np.asarray(value, float).reshape(1, -1))[0]))
        else:
            result.append(float(value))
    return np.asarray(result)


def _audit(field: str, actual: np.ndarray, rebuilt: np.ndarray, *, model: bool) -> dict:
    valid = np.isfinite(actual) & np.isfinite(rebuilt)
    if valid.sum() < 10:
        return {"field": field, "n": int(valid.sum()), "passed": False, "reason": "insufficient_finite_pairs"}
    actual, rebuilt = actual[valid], rebuilt[valid]
    iqr = max(float(np.quantile(actual, .75) - np.quantile(actual, .25)), 1e-8)
    mae = float(np.mean(np.abs(actual - rebuilt)))
    corr = float(np.corrcoef(actual, rebuilt)[0, 1]) if np.std(actual) > 0 and np.std(rebuilt) > 0 else np.nan
    domain = "public_model_proxy" if model else "text_statistic"
    # Different thresholds reflect exact-statistic replication versus model
    # approximation. Both are fixed before looking at A16 or Loss values.
    passed = bool(np.isfinite(corr) and corr >= (.70 if model else .95) and
                  (model or mae / iqr <= .10))
    return {"field": field, "kind": domain, "n": len(actual), "pearson": corr,
            "official_mean": float(actual.mean()), "recomputed_mean": float(rebuilt.mean()),
            "mae": mae, "mae_over_official_iqr": mae / iqr,
            "exact_rate_1e-6": float(np.mean(np.abs(actual - rebuilt) <= 1e-6)),
            "passed": passed,
            "reason": "A1 held-out signal audit" if passed else "A1 signal agreement below preset threshold"}


def reconstruct(a1_path: Path, a18_path: Path, scores_path: Path, out: Path,
                a1_domains: list[str], regmix_domains: list[str], *, per_domain: int,
                batch_size: int, max_length: int, model_text_chars: int, device: str | None = None) -> dict:
    """Sample both corpora, audit 11 rules + six model fields, join Q targets."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_registry.json").write_text(json.dumps({
        "models": {field: {"repository": model, "revision": revision, "output_width": width,
                            "identity": kind}
                   for field, (model, revision, width, kind) in MODEL_REGISTRY.items()},
        "max_length_tokens": max_length, "max_text_chars": model_text_chars,
        "sample_per_domain": per_domain,
        "caution": "fluency_en uses a CoLA proxy; A1 agreement is required before A18 application"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    a1_coverage: dict = {}
    a18_coverage: dict = {}
    a1 = _sample(a1_path, "content", per_domain, set(a1_domains), coverage=a1_coverage)
    a18 = _sample(a18_path, "text", per_domain, set(regmix_domains),
                  allow_shortfall=True, coverage=a18_coverage)
    coverage_rows = [{"dataset": dataset, "domain": domain, **counts,
                      "shortfall": max(0, per_domain - counts["selected"]),
                      "sampling": "all_available" if counts["selected"] < per_domain else "stable_hash"}
                     for dataset, domain_counts in (("A1", a1_coverage), ("A18", a18_coverage))
                     for domain, counts in domain_counts.items()]
    pd.DataFrame(coverage_rows).to_csv(out / "sample_coverage.csv", index=False)
    a1_domain = np.asarray([r["_source_domain"] for r in a1])
    a18_domain = np.asarray([r["_source_domain"] for r in a18])
    scores = pd.read_csv(scores_path, usecols=["id", "domain", "first_source", "Q_primary"])
    scores = scores[scores.first_source.eq("A1")]
    if scores.duplicated(["domain", "id"]).any():
        raise ValueError("Ambiguous A1 scored record join")
    lookup = {(r.domain, str(r.id)): float(r.Q_primary) for r in scores.itertuples()}
    a1_q = np.asarray([lookup[(r["_source_domain"], str(r["id"]))] for r in a1])
    a1_rps = [rps_signals(r["content"]) for r in a1]
    a18_rps = [rps_signals(r["text"]) for r in a18]
    a1_pred: dict[str, np.ndarray] = {field: np.asarray([r[field] for r in a1_rps]) for field in RPS_FIELDS}
    a18_pred: dict[str, np.ndarray] = {field: np.asarray([r[field] for r in a18_rps]) for field in RPS_FIELDS}
    audit_rows = [_audit(field, _official_scalar(field, a1), a1_pred[field], model=False) for field in RPS_FIELDS]
    pd.DataFrame(audit_rows).to_csv(out / "rps_a1_audit.csv", index=False)
    np.savez_compressed(out / "rps_sample_cache.npz", a1_domain=a1_domain, a18_domain=a18_domain,
                        a1_q=a1_q, **{"a1_" + f: v for f, v in a1_pred.items()},
                        **{"a18_" + f: v for f, v in a18_pred.items()})
    a1_texts = [r["content"][:model_text_chars] for r in a1]
    a18_texts = [r["text"][:model_text_chars] for r in a18]
    logits_by_field = {}
    for field, (_, _, width, _) in MODEL_REGISTRY.items():
        a1_logits = infer_logits(a1_texts, field, out / "model_cache",
                                 batch_size=batch_size, max_length=max_length, device=device)
        a1_pred[field] = _model_scalar(field, a1_logits)
        status = _audit(field, _official_scalar(field, a1), a1_pred[field], model=True)
        if status["passed"]:
            a18_logits = infer_logits(a18_texts, field, out / "model_cache",
                                      batch_size=batch_size, max_length=max_length, device=device)
            a18_pred[field] = _model_scalar(field, a18_logits)
        else:
            a18_logits = np.full((len(a18), width), np.nan)
            a18_pred[field] = np.full(len(a18), np.nan)
        logits_by_field[field] = {"a1": a1_logits, "a18": a18_logits}
        audit_rows.append(status)
        pd.DataFrame(audit_rows).to_csv(out / "signal_a1_audit.csv", index=False)
        np.savez_compressed(out / (field + "_logits.npz"), a1=a1_logits, a18=a18_logits)
    audit = pd.DataFrame(audit_rows)
    by_domain_rows = []
    for field in SHARED_FIELDS:
        actual = _official_scalar(field, a1)
        for name in a1_domains:
            selected = a1_domain == name
            by_domain_rows.append({"domain": name, **_audit(field, actual[selected], a1_pred[field][selected],
                                                               model=field in MODEL_REGISTRY)})
    pd.DataFrame(by_domain_rows).to_csv(out / "signal_a1_audit_by_domain.csv", index=False)
    accepted = [field for field in audit.loc[audit.passed, "field"]
                if np.isfinite(a1_pred[field]).all() and np.isfinite(a18_pred[field]).all()]
    audit["accepted_for_transfer"] = audit.field.isin(accepted)
    audit.to_csv(out / "signal_a1_audit.csv", index=False)
    if len(accepted) < 3:
        raise ValueError("Fewer than three shared A1-validated signals; Q transfer is not identifiable")
    a1_x = np.column_stack([a1_pred[f] for f in accepted])
    a18_x = np.column_stack([a18_pred[f] for f in accepted])
    # Reconstruct exactly the field shape expected by task1/2. Unvalidated and
    # non-reproducible fields remain NaN; frozen task1/2 medians handle them.
    a1_raw, a18_raw, a1_full_raw = [], [], []
    for records, rps, domain in ((a1, a1_rps, "a1"), (a18, a18_rps, "a18")):
        rebuilt = []
        for i, record in enumerate(records):
            full_row = {field: rps[i][field] for field in RPS_FIELDS}
            for field in MODEL_REGISTRY:
                full_row[field] = logits_by_field[field][domain][i].tolist()
            if domain == "a1":
                a1_full_raw.append(decode_signals(full_row)[0])
            row = {field: value for field, value in full_row.items() if field in accepted}
            rebuilt.append(decode_signals(row)[0])
        (a1_raw if domain == "a1" else a18_raw).extend(rebuilt)
    official_raw = np.asarray([decode_signals(r)[0] for r in a1], float)
    result = {"a1_x": a1_x, "a18_x": a18_x, "a1_domain": a1_domain, "a18_domain": a18_domain,
              "a1_q": a1_q, "a1_rebuilt_raw": np.asarray(a1_raw, float),
              "a1_full_rebuilt_raw": np.asarray(a1_full_raw, float),
              "a18_rebuilt_raw": np.asarray(a18_raw, float), "a1_official_raw": official_raw,
              "accepted_fields": accepted, "audit": audit}
    np.savez_compressed(out / "reconstructed_signals.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    (out / "accepted_fields.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    return result
