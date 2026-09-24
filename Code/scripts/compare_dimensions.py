"""Command-line entry point for the two Q1 quality-dimension schemes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_resource.q1.dimension_cli import main


if __name__ == "__main__":
    main()
