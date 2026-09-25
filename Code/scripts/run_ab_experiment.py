"""Run the full A/B23/B20 quality-dimension comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_resource.q1.ab_experiment import main


if __name__ == "__main__":
    main()
