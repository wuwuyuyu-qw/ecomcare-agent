"""Run six deterministic read-only recovery adapters with offline guards."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def probes(output):
    from app.evaluation.readonly_recovery import load_fixture, probe_case, VERSION
    data, digest = load_fixture()
    rows = []
    for case in data["cases"]:
        row = await probe_case(case["id"], checkpoint_path=output / (case["id"] + "-checkpoint.sqlite"))
        with (output / (case["id"] + ".json")).open("x", encoding="utf8") as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2, default=str)
        rows.append({k: v for k, v in row.items() if k != "observed"})
    result = {"adapter_version": VERSION, "fixture_sha256": digest, "requested": 6, "executed": len(rows),
              "pass": sum(r["success"] for r in rows), "rows": rows, "model_calls": 0,
              "measurement": "component/lifecycle fixture adapters, excluded from Agent success rate"}
    with (output / "result.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("adapter_version", "pass", "executed", "model_calls")}))
    return 0 if result["pass"] == 6 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    from scripts.check_agent_core import _environment, _install_offline_guards
    if args.child:
        _install_offline_guards()
        from sqlalchemy.engine import Engine
        from sqlalchemy.ext.asyncio import AsyncEngine
        def deny(*args, **kwargs):
            raise AssertionError("recovery adapters forbid business database connections")
        Engine.connect = AsyncEngine.connect = deny
        return asyncio.run(probes(output))
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="mewhelp-readonly-") as working:
        return subprocess.call([sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--output", str(output)],
                               cwd=working, env=_environment())


if __name__ == "__main__":
    raise SystemExit(main())
