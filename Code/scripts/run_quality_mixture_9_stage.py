"""Nine-signal quality transfer and the staged Q1 task-3 experiment."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from llm_resource.q1.mixture import load_pair  # noqa: E402
from llm_resource.q1.quality_mixture_staged import run_loss_phase, run_quality_phase  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "real_attachments")
    parser.add_argument("--quality-summary", type=Path, default=CODE / "outputs/q1/domain_summary.csv")
    parser.add_argument("--config", type=Path, default=CODE / "configs/q1_quality_mixture_9_stage.json")
    parser.add_argument("--out", type=Path, default=CODE / "outputs/q1_quality_mixture_9_stage_complete_a16")
    parser.add_argument("--phase", choices=("quality", "loss", "all"), default="all")
    parser.add_argument("--sample-per-domain", type=int,
                        help="Fixed sample-size sensitivity run; use a separate --out directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.sample_per_domain is not None:
        if args.sample_per_domain < 1:
            parser.error("--sample-per-domain must be positive")
        config["signal_sample_per_domain"] = args.sample_per_domain
    args.out.mkdir(parents=True, exist_ok=True)
    tables = args.data_root / "A_data_value/regmix_tables"
    train = load_pair(tables, config["train"]["mixture"], config["train"]["loss"], "train_1m")
    if args.phase in ("quality", "all"):
        gate = run_quality_phase(config, args.data_root, args.quality_summary, args.out, train.domains)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
    if args.phase in ("loss", "all"):
        result = run_loss_phase(config, args.data_root, args.quality_summary, args.out)
        print(result["cv"].to_string(index=False))
        print("Chosen by A4+A5 CV:", result["metadata"]["chosen_by_A4_A5_cv"])


if __name__ == "__main__":
    main()
