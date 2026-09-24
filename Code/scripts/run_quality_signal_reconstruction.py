"""Reconstruct and audit A1/A18 quality signals only; never train Loss models."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(CODE / ".cache-hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(CODE / "src"))
from llm_resource.q1.quality_reconstruction import reconstruct  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "real_attachments")
    parser.add_argument("--quality-summary", type=Path, default=CODE / "outputs/q1/domain_summary.csv")
    parser.add_argument("--config", type=Path, default=CODE / "configs/q1_quality_mixture_experiment.json")
    parser.add_argument("--out", type=Path, default=CODE / "outputs/q1_quality_reconstruction_experiment")
    parser.add_argument("--per-domain", type=int, help="Override 64 per domain; small values are exploratory only")
    parser.add_argument("--batch-size", type=int, help="Override classifier batch size")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.data_root / "A_data_value"
    a1_domains = sorted(pd.read_csv(args.quality_summary).query("dataset == 'A1'").domain.unique())
    regmix_domains = pd.read_csv(root / "domain_mapping_guide.csv").mixture_domain.tolist()
    result = reconstruct(root / "slimpajama_quality_signal_sample.jsonl.xz",
                         root / "regmix_domain_sample.jsonl.xz",
                         args.quality_summary.parent / "sample_scores.csv", args.out,
                         a1_domains, regmix_domains,
                         per_domain=args.per_domain or config["signal_sample_per_domain"],
                         batch_size=args.batch_size or config["public_models"]["batch_size"],
                         max_length=config["public_models"]["max_length"],
                         model_text_chars=config["public_models"]["max_text_chars"],
                         device=None if args.device == "auto" else args.device)
    columns = ["field", "n", "official_mean", "recomputed_mean", "mae", "pearson",
               "passed", "accepted_for_transfer"]
    print(result["audit"][columns].to_string(index=False))
    print("\nSignal audit only. No A4-A15 Loss model was trained.")
    print("Outputs:", args.out.resolve())


if __name__ == "__main__":
    main()
