"""Export an offline blind packet or import independent human answer ratings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_review import export_packet, import_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    export = commands.add_parser("export", help="read supplied online reports; create a public packet and private binding")
    export.add_argument("--reports", nargs="+", type=Path, required=True)
    export.add_argument("--packet", type=Path, required=True)
    export.add_argument("--bindings", type=Path, required=True)
    load = commands.add_parser("import", help="validate ratings; create a new independently labelled human report")
    load.add_argument("--packet", type=Path, required=True)
    load.add_argument("--bindings", type=Path, required=True)
    load.add_argument("--reviews", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "export":
        result = export_packet(args.reports, args.packet, args.bindings)
    else:
        result = import_reviews(args.packet, args.bindings, args.reviews, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"human review input rejected: {exc}") from None
