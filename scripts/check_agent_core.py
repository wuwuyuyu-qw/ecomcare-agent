"""Run only offline core regressions and frozen dev-definition checks.

Children run from an empty temporary directory with synthetic settings. Neither
this command nor its policy probe loads .env, held-out labels or application
startup. Definition validation is not an Agent task-success measurement.
``--compatible-course`` adds only the explicit file allowlist below. Original
routing assertions expecting ``stop`` at the step cap are excluded because the
current contract returns ``limit`` and the new safety test covers that behavior.
Context/node tests importing the vector runtime are outside this offline scope.
"""
import argparse
import builtins
import importlib.abc
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMPATIBLE_COURSE_TESTS = (
    "tests/test_chunking_split.py", "tests/test_chunking_overlap.py", "tests/test_chunking_table.py",
    "tests/test_prompts.py", "tests/test_memory.py", "tests/core/test_layers.py",
    "tests/core/test_budget.py", "tests/core/test_llm_usage.py", "tests/test_llm_usage.py",
    "tests/core/test_llm_slot.py", "tests/core/test_observability.py", "tests/test_model_guard.py",
    "tests/core/test_confidence.py", "tests/graph/test_state.py",
)


def _environment():
    # Do not inherit provider credentials or unrelated pytest plugins.
    keep = {"PATH", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT",
            "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA",
            "LANG", "LC_ALL", "CI"}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CHAT_MODEL="synthetic-offline",
               CHAT_BASE_URL="http://127.0.0.1:9/v1", CHAT_API_KEY="offline-placeholder",
               EMBED_BASE_URL="http://127.0.0.1:9/v1", EMBED_API_KEY="offline-placeholder",
               RERANK_BASE_URL="http://127.0.0.1:9/v1", RERANK_API_KEY="offline-placeholder",
               DATABASE_URL="mysql+asyncmy://offline:offline@127.0.0.1:9/mewhelp_agent_core_never_connect",
               TEST_DATABASE_URL="mysql+asyncmy://offline:offline@127.0.0.1:9/mewhelp_agent_core_test_never_connect",
               LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="", LANGFUSE_BASE_URL="",
               POLICY_CATALOG_PATH="", MEWHELP_FIXTURE_PROFILE="demo",
               MCP_LOGISTICS_URL="http://127.0.0.1:9/mcp", MCP_AFTERSALES_URL="http://127.0.0.1:9/mcp")
    return env


def _install_offline_guards():
    def no_network(*_args, **_kwargs):
        raise RuntimeError("agent-core checks forbid network access")

    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    socket.getaddrinfo = no_network

    def guarded_open(original):
        def opening(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                name = Path(os.fsdecode(file)).name.lower()
                if name == ".env" or name.startswith(".env.") or name == "holdout.jsonl":
                    raise RuntimeError("agent-core checks forbid environment files and held-out data")
            return original(file, *args, **kwargs)
        return opening

    builtins.open = guarded_open(builtins.open)
    io.open = guarded_open(io.open)

    class NoStartup(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if (fullname in {"app.main", "pymilvus", "app.kb.milvus_client"}
                    or fullname.startswith("pymilvus.")):
                raise RuntimeError("agent-core checks forbid application startup and vector services")
            return None

    sys.meta_path.insert(0, NoStartup())


def _child(phase, output, compatible_course=False):
    sys.path.insert(0, str(ROOT))
    sys.dont_write_bytecode = True
    _install_offline_guards()
    if phase == "tests":
        import pytest
        from sqlalchemy.engine import Engine
        from sqlalchemy.ext.asyncio import AsyncEngine

        def no_database(*_args, **_kwargs):
            raise RuntimeError("agent-core checks forbid database connections")

        Engine.connect = no_database
        AsyncEngine.connect = no_database

        class OfflineFixtures:
            def pytest_fixture_setup(self, fixturedef, request):
                if fixturedef.argname in {"_test_engine", "db_session_factory", "db_clean", "client"}:
                    raise RuntimeError("agent-core checks forbid database and application fixtures")

        selected = [str(ROOT / "tests/agent_core")]
        if compatible_course:
            selected.extend(str(ROOT / path) for path in COMPATIBLE_COURSE_TESTS)
            print(json.dumps({"compatible_course_file_allowlist": list(COMPATIBLE_COURSE_TESTS),
                              "scope": "selected offline regressions; not the full original pytest suite",
                              "intentional_contract_change": "step cap returns limit; original routing stop assertion excluded",
                              "excluded_dependency_paths": ["tests/test_context_assembly.py", "tests/test_prompt_cache_prefix.py"]}, indent=2))
        return pytest.main([*selected, "-q", "-p", "pytest_asyncio.plugin",
                            "-p", "no:cacheprovider"], plugins=[OfflineFixtures()])
    if phase == "dev":
        from app.evaluation.task_cases import load_dataset, validate_cases
        cases, manifest = load_dataset(ROOT / "data/agent_eval", split="dev")
        inventory = validate_cases(cases, expected_splits={"dev": 80, "holdout": 0})
        print(json.dumps({**inventory, "dataset_version": manifest["dataset_version"],
                          "dataset_sha256": manifest["dataset_sha256"],
                          "validation_scope": manifest["validation_scope"],
                          "measurement": "definition_validation_only"}, ensure_ascii=False, indent=2))
        return 0
    if phase == "policy":
        import runpy
        sys.argv = [str(ROOT / "scripts/probe_policy_evidence.py"), "--output", str(output)]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            return exc.code or 0
        return 0
    raise ValueError("unknown offline phase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--tests-only", action="store_true")
    scope.add_argument("--dev-only", action="store_true")
    parser.add_argument("--compatible-course", action="store_true",
                        help="add the explicit compatible offline course test file allowlist; not the full suite")
    parser.add_argument("--policy-probe-output", type=Path,
                        help="optional new JSON and Markdown report paths; existing reports are never overwritten")
    parser.add_argument("--_offline-phase", choices=("tests", "dev", "policy"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.policy_probe_output.resolve() if args.policy_probe_output else None
    if args._offline_phase:
        return _child(args._offline_phase, output, args.compatible_course)
    phases = ["tests"] if args.tests_only else ["dev"] if args.dev_only else ["tests", "dev"]
    if output:
        phases.append("policy")
    for phase in phases:
        with tempfile.TemporaryDirectory(prefix="mewhelp-agent-core-") as working:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_offline-phase", phase]
            if args.compatible_course:
                command.append("--compatible-course")
            if output:
                command.extend(["--policy-probe-output", str(output)])
            result = subprocess.run(command, cwd=working, env=_environment(), check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
