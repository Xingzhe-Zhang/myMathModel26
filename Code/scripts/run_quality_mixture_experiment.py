"""Compare p-only, official-Q-informed, and A18-text-informed RegMix models."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from llm_resource.q1.quality_mixture_experiment import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "real_attachments")
    parser.add_argument("--quality-summary", type=Path, default=CODE / "outputs/q1/domain_summary.csv")
    parser.add_argument("--config", type=Path, default=CODE / "configs/q1_quality_mixture_experiment.json")
    parser.add_argument("--out", type=Path, default=CODE / "outputs/q1_quality_experiment")
    args = parser.parse_args()
    result = run(args.config, args.data_root, args.quality_summary, args.out)
    print(result["cv"].to_string(index=False))
    print("\nResults:", args.out)


if __name__ == "__main__":
    main()
