"""Normalize two actual recording sources, then reuse the human blind exporter."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.common_visible import export_common_reports
from app.evaluation.task_review import export_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=2, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    reports = export_common_reports(args.reports, args.output_dir, args.case_ids)
    summary = export_packet(reports, args.output_dir / "packet.json", args.output_dir / "private-bindings.json")
    print(json.dumps({"mode": "retrospective_common_visible_review", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError) as exc:
        raise SystemExit("common review input rejected: " + type(exc).__name__) from None
