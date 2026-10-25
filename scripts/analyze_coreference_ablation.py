"""Analyze four recorded development reports without running the Agent."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.coreference_ablation import analyze_reports, write_analysis
from app.evaluation.task_report import assert_new_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=4, required=True, metavar=("A1", "B1", "B2", "A2"),
                        help="four actual report paths in chronological A B B A order")
    parser.add_argument("--output", type=Path, required=True, help="new .json path; a .md sibling is also created")
    args = parser.parse_args()
    assert_new_output(args.output, markdown=True)
    result = analyze_reports(args.reports)
    write_analysis(args.output, result)
    print(f"Saved read-only ABBA analysis to {args.output}; paired cases: {result['paired_latency']['paired_case_count']}.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"ablation analysis input rejected: {exc}") from None
