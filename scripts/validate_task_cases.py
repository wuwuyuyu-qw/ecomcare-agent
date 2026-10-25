"""Check annotation completeness, group isolation and the frozen dataset digest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_cases import load_dataset, validate_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/agent_eval")
    args = parser.parse_args()
    cases, manifest = load_dataset(args.dataset)
    print(json.dumps({**validate_cases(cases), "dataset_version": manifest["dataset_version"],
                      "dataset_sha256": manifest["dataset_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
