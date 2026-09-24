"""Run gated 11+6 signal reconstruction and three-model Loss comparison."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
CODE = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(CODE / ".cache-hf"))
sys.path.insert(0, str(CODE / "src"))
from llm_resource.q1.quality_mixture_experiment import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "real_attachments")
    parser.add_argument("--quality-summary", type=Path, default=CODE / "outputs/q1/domain_summary.csv")
    parser.add_argument("--config", type=Path, default=CODE / "configs/q1_quality_mixture_experiment.json")
    parser.add_argument("--out", type=Path, default=CODE / "outputs/q1_quality_reconstruction_experiment")
    parser.add_argument("--sample-per-domain", type=int, help="Pilot sample size; formal config uses 64 per domain")
    parser.add_argument("--allow-loss-after-q-gate", action="store_true",
                        help="Explicitly allow A4-A15 Loss comparison only if both Q cross-domain gates pass")
    args = parser.parse_args()
    result = run(args.config, args.data_root, args.quality_summary, args.out,
                 sample_per_domain=args.sample_per_domain,
                 allow_loss=args.allow_loss_after_q_gate)
    if "status" in result:
        print("Quality gate:", result["status"])
    else:
        print(result["cv"].to_string(index=False))
        print(result["decisions"].to_string(index=False))
    print("\nResults:", args.out)


if __name__ == "__main__":
    main()
