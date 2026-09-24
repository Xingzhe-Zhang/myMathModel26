"""Check whether the supplied A1 text actually reproduces its 11 RPS fields."""
import argparse
import json
import lzma
from itertools import islice
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from llm_resource.q1.reconstructed_signals import RPS_FIELDS, rps_signals  # noqa: E402
from llm_resource.q1.quality_reconstruction import _audit, _sample  # noqa: E402


def run(input_path: Path, out: Path, limit: int | None, per_domain: int | None = None):
    rows = []
    if per_domain:
        domains = set(pd.read_csv(CODE / "outputs/q1/domain_summary.csv").query("dataset == 'A1'").domain)
        records = _sample(input_path, "content", per_domain, domains)
    else:
        stream = lzma.open(input_path, "rt", encoding="utf-8")
        records = (json.loads(line) for line in islice(stream, limit))
    for record in records:
        computed = rps_signals(record["content"])
        row = {"domain": record["_source_domain"], "id": record["id"], "text_length": len(record["content"])}
        for field in RPS_FIELDS:
            row[field + "_official"] = record.get(field)
            row[field + "_recomputed"] = computed[field]
        rows.append(row)
    if not per_domain:
        stream.close()
    frame = pd.DataFrame(rows)
    summary = []
    for field in RPS_FIELDS:
        a = frame[field + "_official"].to_numpy(float)
        b = frame[field + "_recomputed"].to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        if not valid.any():
            summary.append({"field": field, "n": 0})
            continue
        difference = a[valid] - b[valid]
        gate = _audit(field, a, b, model=False)
        summary.append({"field": field, "n": int(valid.sum()),
                        "official_mean": float(np.mean(a[valid])),
                        "recomputed_mean": float(np.mean(b[valid])),
                        "mae": float(np.mean(np.abs(difference))),
                        "rmse": float(np.sqrt(np.mean(difference**2))),
                        "exact_rate_1e-6": float(np.mean(np.abs(difference) <= 1e-6)),
                        "correlation": float(np.corrcoef(a[valid], b[valid])[0, 1]),
                        "mae_over_official_iqr": gate["mae_over_official_iqr"],
                        "passed": gate["passed"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out, index=False)
    frame.to_csv(out.with_name(out.stem + "_rows.csv"), index=False)
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CODE.parent / "real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz")
    parser.add_argument("--out", type=Path, default=CODE / "outputs/q1_quality_transfer_experiment/rps_a1_audit.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-domain", type=int)
    args = parser.parse_args()
    print(run(args.input, args.out, args.limit, args.per_domain).to_string(index=False))
