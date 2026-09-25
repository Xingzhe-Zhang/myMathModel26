"""Run the fourteen-signal Q1 task-3 quality gate and eligible Loss models."""
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
from llm_resource.q1.quality_mixture_14 import run_quality_phase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "real_attachments")
    parser.add_argument("--config", type=Path, default=CODE / "configs/q1_quality_mixture_14_stage.json")
    parser.add_argument("--signal-dir", type=Path, default=CODE / "outputs/q1_quality_reconstruction_experiment")
    parser.add_argument("--ab-dir", type=Path, default=CODE / "outputs/q1_ab_experiment")
    parser.add_argument("--out", type=Path, default=CODE / "outputs/q1_quality_mixture_14_stage")
    parser.add_argument("--phase", choices=("quality", "loss", "all"), default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    tables = args.data_root / "A_data_value/regmix_tables"
    train = load_pair(tables, config["train"]["mixture"], config["train"]["loss"], "train_1m")
    if args.phase in ("quality", "all"):
        gate = run_quality_phase(config, args.data_root, args.signal_dir, args.ab_dir,
                                 args.out, train.domains)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
    if args.phase in ("loss", "all"):
        from llm_resource.q1.quality_mixture_14_loss import run_loss_phase
        result = run_loss_phase(config, args.data_root, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
