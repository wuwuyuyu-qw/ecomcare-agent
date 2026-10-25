"""Reproducible owned-service diagnostics for the local MewHelp candidate.

Examples (credentials remain in the explicitly selected environment file):
  python scripts/run_agent_diagnostics.py --env-file /project/.env \
      --output-dir /project/output/eval/auth-final --mode auth
  python scripts/run_agent_diagnostics.py --env-file /project/.env \
      --output-dir /project/output/eval/candidate-final --mode candidate \
      --split dev --eligible

No containers, volumes, prior reports or knowledge collections are removed.
Only a selected dataset split is parsed; holdout annotations are never archived.
Dedicated auth/fault/memory probes are synthetic dev diagnostics, not dataset
scores. Deadline uses the public lifecycle probe and an independent private
checkpoint observations. Policy uses a dedicated synthetic index; disconnect
restarts only the owned services, reusing the private checkpoint.
Importing this file or requesting --help does not load settings or call services.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("candidate", "baseline", "auth", "fault", "memory", "coref-a", "coref-b", "policy", "deadline", "disconnect", "compound-fixed", "compound-agent", "compound-routing-a", "compound-routing-b")
COMPOUND_MODES = {"compound-fixed", "compound-agent", "compound-routing-a", "compound-routing-b"}
ROUTING_MODES = {"compound-routing-a", "compound-routing-b"}
FAULT_MODES = ("healthy", "transient_timeout", "persistent_timeout", "invalid_hours", "wrong_tracking", "injected_trace")
TASK_MODES = {"candidate", "baseline", "coref-a", "coref-b"}
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SECRET_KEY = re.compile(r"api.?key|password|passwd|secret|access.?token|authorization|cookie|credential|private.?key", re.I)
_URL_AUTH = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@")
_SECRET_TEXT = re.compile(r"(?i)(?:bearer\s+[^\s,;\"']+|sk-[A-Za-z0-9_-]{8,}|(?:password|api_key|access_token|secret)\s*[:=]\s*[^\s,;]+)")


class DiagnosticError(RuntimeError):
    """Fixed local diagnostic codes; external exception text is never retained."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Redactor:
    def __init__(self, environment: dict[str, str]):
        self.values = {value for key, value in environment.items() if value and _SECRET_KEY.search(key)}

    def add(self, *values: str) -> None:
        self.values.update(value for value in values if value)

    def text(self, value: str) -> str:
        value = _URL_AUTH.sub(r"\1[redacted]@", value)
        for secret in sorted(self.values, key=len, reverse=True):
            value = value.replace(secret, "[redacted]")
        return _SECRET_TEXT.sub("[redacted]", value)

    def value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self.value(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(child) for child in value]
        return self.text(value) if isinstance(value, str) else value


def write_new(directory: Path, name: str, value: dict, redactor: Redactor | None = None) -> None:
    if redactor is not None:
        value = redactor.value(value)
    with (directory / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def emit(event: str, directory: Path, **fields: Any) -> None:
    # Callers pass only fixed codes, IDs, paths and aggregate observations.
    print(json.dumps({"event": event, "directory": str(directory), **fields}, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--env-file", type=Path, required=True, help="Explicit existing .env; never printed, hashed or archived")
    result.add_argument("--output-dir", type=Path, required=True, help="NEW private artifact directory; existing directories are refused")
    result.add_argument("--mode", choices=MODES, default="candidate")
    result.add_argument("--split", choices=("dev", "holdout"), default="dev", help="Candidate reads only this split; tuning/dedicated probes require dev")
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--case-ids", nargs="+", help="Exact whitelist within the selected split")
    selection.add_argument("--eligible", action="store_true", help="All task-v1 eligible cases from the selected split")
    result.add_argument("--fault-mode", choices=FAULT_MODES)
    result.add_argument("--task-key", help="Safe run grouping label; default is mode-split")
    result.add_argument("--api-port", type=int, default=18000)
    result.add_argument("--logistics-port", type=int, default=18101)
    result.add_argument("--aftersales-port", type=int, default=18102)
    result.add_argument("--startup-timeout", type=float, default=100)
    result.add_argument("--runner-timeout", type=float, default=800)
    result.add_argument("--plan-only", action="store_true", help="Freeze sources and selection only; do not open .env or contact any service")
    result.add_argument("--skip-provider-probe", action="store_true", help="Omit only the separate model readiness request; diagnostics may still call models")
    result.add_argument("--guide-corpus", action="store_true", help="Routing dev only: create an owned self-written operations guide collection")
    result.add_argument("--selected-evidence", action="store_true", help="Guide diagnostic only: opt in to source selection")
    result.add_argument("--frozen-experiment-plan", type=Path, help="Required before any holdout run; existing preregistration JSON")
    return result


def validate_args(args: argparse.Namespace, argument_parser: argparse.ArgumentParser) -> None:
    if getattr(args, "guide_corpus", False) and args.mode not in ROUTING_MODES:
        argument_parser.error("guide corpus requires a routing development mode")
    if getattr(args, "selected_evidence", False) and not getattr(args, "guide_corpus", False):
        argument_parser.error("selected evidence diagnostic requires guide corpus")
    if args.mode not in {"candidate", "baseline"} and args.split != "dev":
        argument_parser.error("dedicated probes and coreference tuning require --split dev")
    if args.split == "holdout" and not args.frozen_experiment_plan:
        argument_parser.error("holdout requires a pre-existing --frozen-experiment-plan")
    if args.mode not in TASK_MODES | COMPOUND_MODES and (args.case_ids or args.eligible):
        argument_parser.error("case selection applies only to candidate/coreference task modes")
    if (args.mode == "fault") != (args.fault_mode is not None):
        argument_parser.error("--fault-mode is required exactly when --mode fault is selected")
    args.task_key = args.task_key or f"{args.mode}-{args.split}"
    if not _KEY.fullmatch(args.task_key):
        argument_parser.error("--task-key must be a safe identifier of at most 64 characters")
    if args.case_ids and (len(args.case_ids) != len(set(args.case_ids)) or any(not _KEY.fullmatch(v) for v in args.case_ids)):
        argument_parser.error("--case-ids must be a unique list of safe identifiers")
    ports = (args.api_port, args.logistics_port, args.aftersales_port)
    if len(set(ports)) != 3 or any(not 1024 <= port <= 65535 for port in ports):
        argument_parser.error("three different loopback ports from 1024 to 65535 are required")
    if any(not math.isfinite(value) or value <= 0 for value in (args.startup_timeout, args.runner_timeout)):
        argument_parser.error("timeouts must be positive finite values")


def select_cases(args: argparse.Namespace) -> tuple[list[str], dict]:
    if args.mode in COMPOUND_MODES:
        from app.evaluation.compound_cases import load_cases
        routing = args.mode in ROUTING_MODES
        cases, _ = load_cases(ROOT / ("data/compound_routing_eval" if routing else "data/compound_eval"), "dev",
                             "compound-routing-v1.0" if routing else "compound-v1.1")
        ids = [case["id"] for case in cases if not args.case_ids or case["id"] in args.case_ids]
        if not ids or args.case_ids and set(ids) != set(args.case_ids):
            raise DiagnosticError("compound_dev_selection_differs")
        return ids, {"parsed_splits": ["compound-routing-dev" if routing else "compound-dev"], "selection_mode": "explicit_compound_dev",
                     "holdout_opened": False}
    if args.mode not in TASK_MODES:
        return [], {"parsed_splits": [], "selection_mode": "dedicated_synthetic_probe"}
    from app.evaluation.task_cases import load_dataset
    cases, manifest = load_dataset(ROOT / "data/agent_eval", args.split)
    available = {case["id"] for case in cases}
    if args.case_ids:
        if not set(args.case_ids) <= available:
            raise DiagnosticError("case_id_outside_selected_split")
        requested = set(args.case_ids)
        ids = [case["id"] for case in cases if case["id"] in requested]
        selection_mode = "explicit_case_ids"
    elif args.eligible:
        ids = [case["id"] for case in cases if "task-v1" in case["online_profiles"]]
        selection_mode = "all_eligible_selected_split"
    elif args.split == "holdout":
        # A frozen holdout run requests the whole chosen split, retaining skips.
        ids = [case["id"] for case in cases]
        selection_mode = "all_selected_holdout"
    else:
        requested = {"task-01-01", "task-02-01", "task-03-01", "task-04-01"}
        ids = [case["id"] for case in cases if case["id"] in requested]
        if set(ids) != requested:
            raise DiagnosticError("default_dev_case_inventory_changed")
        selection_mode = "small_declared_dev_diagnostic"
    if not ids:
        raise DiagnosticError("empty_case_selection")
    return ids, {"parsed_splits": [args.split], "selection_mode": selection_mode,
                 "dataset_validation_scope": manifest["validation_scope"]}


def source_paths(args: argparse.Namespace) -> list[Path]:
    paths = [path for folder in ("app", "scripts", "mcp_servers", "tests")
             for path in (ROOT / folder).rglob("*")
             if path.is_file() and path.suffix in {".py", ".js", ".html", ".sh"}]
    paths.extend(ROOT / name for name in (
        "pyproject.toml", "uv.lock", "data/agent_fixtures/task-v1.json",
        "data/agent_eval/manifest.json", "data/policies/example/catalog.json",
        "data/policies/example/chunks.json", "data/policies/example/world.json",
        ".github/workflows/agent-core.yml", "Makefile"))
    if args.mode in TASK_MODES and args.split == "dev":
        paths.append(ROOT / "data/agent_eval/dev.jsonl")
    if args.mode in COMPOUND_MODES:
        paths.extend(ROOT / "data/compound_eval" / name for name in ("manifest.json", "dev.jsonl"))
    if args.mode in ROUTING_MODES:
        paths.extend(ROOT / "data/compound_routing_eval" / name for name in ("manifest.json", "dev.jsonl"))
    if getattr(args, "guide_corpus", False):
        paths.extend((ROOT / "data/rag_answerability").glob("*"))
    return sorted(set(paths))


def source_snapshot(directory: Path, args: argparse.Namespace, selection: dict) -> tuple[str, dict, list[Path]]:
    paths = source_paths(args)
    manifest = json.loads((ROOT / "data/agent_eval/manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    # The opposite split's identity comes from its frozen manifest, without
    # opening its file. Never put holdout labels into the source archive.
    split_hashes = {split: manifest["files"][split + ".jsonl"]["sha256"] for split in ("dev", "holdout")}
    with zipfile.ZipFile(directory / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            raw = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            hashes[name] = sha256(raw)
            archive.writestr(name, raw)
    hashes.setdefault("data/agent_eval/dev.jsonl", split_hashes["dev"])
    if args.mode in TASK_MODES and args.split == "holdout":
        # This split was explicitly selected and already validated by selection.
        hashes["data/agent_eval/holdout.jsonl"] = sha256((ROOT / "data/agent_eval/holdout.jsonl").read_bytes())
    else:
        hashes["data/agent_eval/holdout.jsonl"] = split_hashes["holdout"]
    identity = sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    write_new(directory, "source-manifest.json", {"source": str(ROOT), "source_sha256": identity,
        "files": hashes, "holdout_annotations_included": False,
        "holdout_hash_source": "selected_split_file_bytes" if args.mode in TASK_MODES and args.split == "holdout" else "frozen_manifest_not_file_opened",
        "parsed_splits": selection["parsed_splits"], "credentials_or_env_files_included": False})
    return identity, hashes, paths


def source_end_check(directory: Path, args: argparse.Namespace, hashes: dict, paths: list[Path]) -> None:
    current = source_paths(args)
    changed = []
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        if not path.is_file() or sha256(path.read_bytes()) != hashes[name]:
            changed.append(name)
    added = sorted(path.relative_to(ROOT).as_posix() for path in set(current) - set(paths))
    write_new(directory, "source-stability.json", {"unchanged": not changed and not added,
        "changed_or_missing": changed, "added": added,
        "scope": "executed source, configuration schema, fixture, policy examples and archived selected dev labels; unselected holdout never reopened"})


def environment(args: argparse.Namespace, directory: Path) -> tuple[Any, Redactor]:
    # This is the sole .env loading operation. A new empty working directory
    # prevents Settings(env_file='.env') in imported code from opening another.
    env_file = args.env_file.expanduser().resolve(strict=True)
    if not env_file.is_file() or not env_file.name.startswith(".env"):
        raise DiagnosticError("explicit_env_file_required")
    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)
    for key in ("POLICY_CATALOG_PATH", "MILVUS_URI"):
        value = os.environ.get(key, "")
        if value and "://" not in value and not Path(value).is_absolute():
            os.environ[key] = str((ROOT / value).resolve())
    os.environ.update({
        "MEWHELP_FIXTURE_PROFILE": "task-v1",
        "AUTH_DB_PATH": str(directory / "private-auth.sqlite"),
        "CHECKPOINTER_DB_PATH": str(directory / "private-checkpoints.sqlite"),
        "MCP_LOGISTICS_URL": f"http://127.0.0.1:{args.logistics_port}/mcp",
        "MCP_AFTERSALES_URL": f"http://127.0.0.1:{args.aftersales_port}/mcp",
        "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LANGFUSE_BASE_URL": "",
        "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
    })
    if args.mode == "memory":
        os.environ["CONTEXT_WINDOW_MAX_TOKENS"] = "700"
    if args.mode == "fault":
        os.environ.update(MCP_TOOL_TIMEOUT="3", TOOL_MAX_RETRIES="2")
    if args.mode == "deadline":
        os.environ["TURN_TIMEOUT_SECONDS"] = "0.4"
    if args.mode == "disconnect":
        os.environ["TURN_TIMEOUT_SECONDS"] = "120"
    if args.mode == "policy":
        collection = "diagnostic_policy_" + hashlib.sha256(str(directory).encode()).hexdigest()[:20]
        os.environ.update(RETRIEVAL_COLLECTION=collection,
                          POLICY_CATALOG_PATH=str(ROOT / "data/policies/example/catalog.json"))
    if args.mode in {"coref-a", "coref-b"}:
        os.environ["SKIP_COREF_WITHOUT_HISTORY"] = "true" if args.mode == "coref-b" else "false"
    # This experiment factor is explicit, never inherited accidentally.
    os.environ["COMPOUND_AGENT_ENABLED"] = "true" if args.mode in {"compound-agent", *ROUTING_MODES} else "false"
    os.environ["COMPOUND_JOINT_TRIAGE_ENABLED"] = "true" if args.mode == "compound-routing-b" else "false"
    os.environ["EVIDENCE_SELECTION_ENABLED"] = "true" if getattr(args, "selected_evidence", False) else "false"
    if getattr(args, "guide_corpus", False):
        collection = "diagnostic_guides_" + hashlib.sha256(str(directory).encode()).hexdigest()[:20]
        os.environ.update(RETRIEVAL_COLLECTION=collection, POLICY_CATALOG_PATH="")
    os.chdir(directory)
    if (directory / ".env").exists():
        raise DiagnosticError("unexpected_implicit_env_file")
    redactor = Redactor(dict(os.environ))
    from app.config import settings
    return settings, redactor


async def connect_database(settings):
    import asyncmy
    from sqlalchemy.engine import make_url
    url = make_url(settings.database_url)
    return await asyncmy.connect(host=url.host, port=url.port or 3306, user=url.username,
                                 password=url.password, db=url.database, charset="utf8mb4")


async def knowledge_digest(settings) -> dict:
    conn = await connect_database(settings)
    try:
        async with conn.cursor() as cursor:
            await cursor.execute("SHOW COLUMNS FROM knowledge_chunks")
            available = {row[0] for row in await cursor.fetchall()}
            columns = [name for name in ("id", "category", "questions", "answer", "section_path", "vector_id", "vectorize_status") if name in available]
            if not columns:
                raise DiagnosticError("knowledge_columns_unavailable")
            await cursor.execute("SELECT " + ",".join("`" + name + "`" for name in columns) + " FROM knowledge_chunks ORDER BY id")
            rows = await cursor.fetchall()
        raw = json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
        return {"mysql_rows": len(rows), "mysql_content_sha256": sha256(raw),
                "milvus_index_version": "existing_collection_not_independently_versioned"}
    finally:
        conn.close()


async def provider_probe(settings) -> dict:
    from openai import AsyncOpenAI
    async with AsyncOpenAI(base_url=settings.chat_base_url, api_key=settings.chat_api_key,
                           timeout=25, max_retries=0) as client:
        response = await client.chat.completions.create(model=settings.chat_model,
            messages=[{"role": "user", "content": "请只回复 OK。"}], max_tokens=8,
            temperature=0, extra_body={"thinking": {"type": "disabled"}})
    return {"requested_model": settings.chat_model, "returned_model": response.model,
        "usage": response.usage.model_dump() if response.usage else None,
        "scope": "separate readiness request; excluded from foreground task metrics", "observed_at": utc_now()}


def run_config(settings, knowledge: dict, world: dict, args: argparse.Namespace) -> dict:
    from app.tools.aftersales_fixtures import VERSION as aftersales_version, REFERENCE_TIME
    result = {"selected_evidence_enabled": settings.evidence_selection_enabled,
        "model_config": {"chat_model": settings.chat_model,
        "intent_model": settings.intent_model or settings.chat_model,
        "embedding_model": settings.embed_model, "rerank_model": settings.rerank_model,
        "stage_temperatures": {"coreference": 0.3, "legacy_intent_classification": 0.3,
            "joint_triage": 0, "compound_checklist": 0,
            "logistics_extraction": 0, "logistics_draft": 0, "main_agent": 0.3},
        "thinking": settings.chat_thinking, "reasoning_effort": settings.chat_reasoning_effort,
        "structured_output_thinking": "disabled", "max_output_tokens": settings.max_output_tokens},
        "knowledge_version": "mysql-sha256:" + knowledge["mysql_content_sha256"] + ";milvus-existing-unversioned",
        **world,
        "aftersales_world": {"version": aftersales_version, "reference_time": REFERENCE_TIME.isoformat(),
                             "scope": "new diagnostic orders only; course mocks unchanged"},
        "pricing": {"currency": settings.model_price_currency, "rates_as_of": None, "source": None,
            "configured_on_server": bool(settings.model_prices), "model_rates": settings.model_prices},
        "execution": {"turn_timeout_seconds": settings.turn_timeout_seconds,
            "max_agent_steps": settings.max_agent_steps, "max_task_tool_calls": settings.max_task_tool_calls,
            "max_identical_tool_calls": settings.max_identical_tool_calls, "mcp_timeout_seconds": settings.mcp_tool_timeout,
            "skip_coref_without_history": settings.skip_coref_without_history}}
    if args.mode == "memory":
        result["diagnostic_memory_window_tokens"] = settings.context_window_max_tokens
    if args.mode == "fault":
        result["fault_injection"] = {"service": "diagnostic_logistics_server", "mode": args.fault_mode,
                                     "delay_seconds": 8.0, "tracking_no": "SFDEMO3003"}
    return result


def port_preflight(args: argparse.Namespace) -> None:
    for port in (args.api_port, args.logistics_port, args.aftersales_port):
        with socket.socket() as sock:
            # Match uvicorn's POSIX bind behavior: a closed prior connection in
            # TIME_WAIT is not a live service. Never enable SO_REUSEPORT. On
            # Windows use exclusive binding rather than address sharing.
            if os.name == "nt":
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                sock.listen(1)
            except OSError:
                raise DiagnosticError("loopback_port_occupied") from None


def windows_job(process: subprocess.Popen):
    """Assign only an owned process to a private kill-on-close Windows job."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Io(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
            "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise DiagnosticError("owned_windows_job_unavailable")
    limits = Extended()
    limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle)))):
        kernel.CloseHandle(job)
        raise DiagnosticError("owned_windows_job_assignment_failed")
    return kernel, job


class OwnedProcesses:
    def __init__(self, directory: Path, redactor: Redactor):
        self.directory, self.redactor = directory, redactor
        self.children = []

    def start(self, name: str, command: list[str], environment: dict[str, str]) -> subprocess.Popen:
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        process = subprocess.Popen(command, cwd=self.directory, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", **options)
        entry = {"name": name, "process": process, "job": None, "thread": None}
        self.children.append(entry)
        if os.name == "nt":
            try:
                entry["job"] = windows_job(process)
            except Exception:
                process.terminate()
                process.wait(timeout=5)
                raise

        def consume():
            # Raw service/runner output is never written to disk or echoed.
            with (self.directory / (name + ".log")).open("x", encoding="utf-8") as stream:
                if process.stdout is not None:
                    for line in process.stdout:
                        stream.write(self.redactor.text(line))
                        stream.flush()
        thread = threading.Thread(target=consume, name="diagnostic-log-" + name, daemon=True)
        entry["thread"] = thread
        thread.start()
        return process

    def alive(self) -> bool:
        return all(entry["process"].poll() is None for entry in self.children)

    def close(self) -> dict:
        observations = []
        for entry in reversed(self.children):
            process = entry["process"]
            if entry["job"] is not None:
                kernel, handle = entry["job"]
                kernel.CloseHandle(handle)  # Only descendants assigned to this private job.
                entry["job"] = None
            elif os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGTERM)  # Session created by this coordinator.
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()  # Assignment failure fallback: exact owned process only.
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=3)
            if entry["thread"] is not None:
                entry["thread"].join(timeout=3)
            if process.stdout is not None:
                process.stdout.close()
            observations.append({"service": entry["name"], "pid": process.pid,
                "exit_code": process.poll(), "owned_group_stopped": process.poll() is not None})
        return {"scope": "only process groups/jobs started by this coordinator",
                "processes": observations, "containers_volumes_knowledge_removed": False}


async def seed_stale_checkpoints() -> list[dict]:
    from langchain_core.messages import HumanMessage
    from app.db import repository, base
    from app.graph import runtime
    seeds = []
    await runtime.init_graph()
    try:
        for kind in ("owner_mismatch", "conversation_mismatch"):
            cid = await repository.create_conversation("customer_demo")
            old_owner = "customer_other" if kind == "owner_mismatch" else "customer_demo"
            old_cid = cid if kind == "owner_mismatch" else cid + 1_000_000
            await runtime.get_graph().aupdate_state({"configurable": {"thread_id": str(cid)}},
                {"user_id": old_owner, "conversation_id": old_cid,
                 "messages": [HumanMessage("synthetic stale checkpoint marker", id="seed-" + kind)]}, as_node="log")
            seeds.append({"kind": kind, "conversation_id": cid, "sql_owner": "customer_demo",
                          "checkpoint_owner": old_owner, "checkpoint_conversation_id": old_cid})
    finally:
        await runtime.close_graph()
        await base.engine.dispose()
    return seeds


async def database_observation(settings, report: dict) -> dict:
    ids = report.get("fresh_conversation_ids")
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 160
            or any(type(cid) is not int or cid <= 0 for cid in ids)):
        raise DiagnosticError("probe_fresh_conversation_ids_unavailable")
    ids = sorted(set(ids))
    conn = await connect_database(settings)
    try:
        async with conn.cursor() as cursor:
            markers = ",".join(["%s"] * len(ids))
            await cursor.execute("SELECT id,user_id FROM conversations WHERE id IN (" + markers + ") ORDER BY id", ids)
            owners = await cursor.fetchall()
            await cursor.execute("SELECT conversation_id,COUNT(*) FROM tickets WHERE conversation_id IN (" + markers + ") GROUP BY conversation_id", ids)
            counts = dict(await cursor.fetchall())
        return {"scope": "read-only SQL on this run's synthetic conversations only",
            "conversations": [{"conversation_id": cid, "owner": owner, "ticket_count": counts.get(cid, 0)} for cid, owner in owners],
            "all_ids_observed": len(owners) == len(ids), "ticket_count": sum(counts.values())}
    finally:
        conn.close()


async def memory_observation(report: dict) -> dict:
    from app.graph import runtime
    from app.db import base
    from sqlalchemy import text
    await runtime.init_graph()
    rows = []
    try:
        for cid in report["fresh_conversation_ids"]:
            snap = await runtime.get_graph().aget_state({"configurable": {"thread_id": str(cid)}})
            values = snap.values or {}
            context = values.get("task_context") or {}
            async with base.engine.connect() as conn:
                row = (await conn.execute(text("SELECT summary_upto_msg_id,layer1_from_msg_id FROM conversations WHERE id=:cid"), {"cid": cid})).first()
                summaries = (await conn.execute(text("SELECT COUNT(*) FROM conversation_summaries WHERE conversation_id=:cid"), {"cid": cid})).scalar()
            if row is None:
                raise DiagnosticError("memory_conversation_row_unavailable")
            rows.append({"conversation_id": cid, "checkpoint_user_id": values.get("user_id"),
                "checkpoint_conversation_id": values.get("conversation_id"), "active_order_id": context.get("active_order_id"),
                "termination_reason": context.get("termination_reason"), "persisted_message_count": len(values.get("messages") or []),
                "summary_upto_msg_id": row[0], "layer1_from_msg_id": row[1], "summary_segments": summaries})
    finally:
        await runtime.close_graph()
        await base.engine.dispose()
    return {"scope": "read-only reopened private checkpoint and this run's SQL conversation rows", "conversations": rows}


async def deadline_observation(report: dict) -> dict:
    """Observe actual saved state only after stopping all owned services."""
    cid = report.get("deadline_cid")
    if type(cid) is not int or cid <= 0:
        return {"observed": False, "reason": "deadline_conversation_id_unavailable"}
    from app.graph import runtime
    from app.db import base
    await runtime.init_graph()
    try:
        snap = await runtime.get_graph().aget_state({"configurable": {"thread_id": str(cid)}})
        values = snap.values or {}
        context = values.get("task_context") or {}
        scope = {"user_id": "customer_demo", "conversation_id": cid}
        return {"scope": "independent private checkpoint after owned service stop", "observed": bool(values),
            "conversation_id": cid, "identity_matches": all(values.get(key) == value for key, value in scope.items()),
            "task_scope_matches": all(context.get(key) == value for key, value in scope.items()),
            "termination_reason": context.get("termination_reason"),
            "next_nodes": list(snap.next), "task_id": context.get("task_id"), "request_id": context.get("request_id"),
            "checks": {"persisted_deadline_termination": context.get("termination_reason") == "deadline_exceeded",
                       "no_queued_graph_nodes": not snap.next}}
    finally:
        await runtime.close_graph()
        await base.engine.dispose()


async def prepare_policy_index(settings) -> dict:
    from app.core import embeddings
    from app.kb import milvus_client as mv
    chunks = json.loads((ROOT / "data/policies/example/chunks.json").read_text(encoding="utf-8"))
    chunk = next(c for c in chunks if c["id"] == 41003)
    question = "物流无更新时按最新物流异常政策能怎么处理"
    text = "物流异常\n" + question + "\n" + chunk["answer"]
    client = await mv.acall(mv.get_client)
    name = settings.retrieval_collection
    if not re.fullmatch(r"diagnostic_policy_[0-9a-f]{20}", name) or await mv.acall(client.has_collection, name):
        raise DiagnosticError("synthetic_policy_collection_must_be_new_and_owned")
    try:
        vectors = await embeddings.embed_texts([text])
        await mv.acall(mv.ensure_collection, client, name)
        await mv.acall(mv.upsert_vectors, client, [{"id": 41003, "dense": vectors[0], "text": text,
            "question": question, "answer": chunk["answer"], "section_path": "合成政策/物流异常",
            "content_type": "policy", "category": "物流异常"}], name)
        await mv.acall(mv.flush, client, name)
        count = await mv.acall(mv.count, client, name)
        return {"collection": name, "indexed_document_ids": [41003], "observed_rows": count,
            "body_sha256": sha256(chunk["answer"].encode("utf-8")),
            "catalog_sha256": sha256((ROOT / "data/policies/example/catalog.json").read_bytes()),
            "scope": "one owned self-authored document; original knowledge/MySQL corpus untouched",
            "setup_usage": "one setup embedding excluded from foreground task ledger"}
    finally:
        await embeddings.close_client()


async def prepare_guide_index(settings) -> dict:
    from app.evaluation.rag_answerability import load_bundle
    from app.core import embeddings
    from app.kb import milvus_client as mv
    docs, _, manifest = load_bundle(ROOT / "data/rag_answerability")
    client = await mv.acall(mv.get_client)
    name = settings.retrieval_collection
    if not re.fullmatch(r"diagnostic_guides_[0-9a-f]{20}", name) or await mv.acall(client.has_collection, name):
        raise DiagnosticError("guide_collection_must_be_new_and_owned")
    try:
        texts = [d["category"] + "\n" + d["question"] + "\n" + d["answer"] for d in docs]
        vectors = await embeddings.embed_texts(texts)
        await mv.acall(mv.ensure_collection, client, name)
        await mv.acall(mv.upsert_vectors, client, [{**d, "dense": v, "text": t}
            for d, v, t in zip(docs, vectors, texts, strict=True)], name)
        await mv.acall(mv.flush, client, name)
        rows = await mv.acall(client.query, name, filter="id >= 0", limit=len(docs), output_fields=["question", "answer"])
        if {str(d["id"]): (d["question"], d["answer"]) for d in rows} != {str(d["id"]): (d["question"], d["answer"]) for d in docs}:
            raise DiagnosticError("guide_body_binding_differs")
        return {"collection": name, "documents": len(rows), "exact_match": True,
            "dataset_sha256": manifest["dataset_sha256"], "selected_evidence_enabled": settings.evidence_selection_enabled,
            "scope": "owned synthetic guides; no original knowledge changes; index retained",
            "setup_usage": "one setup embedding excluded from foreground ledger"}
    finally:
        await embeddings.close_client()


async def lifecycle_checkpoint_observation(report: dict) -> dict:
    from app.graph import runtime
    from app.db import base
    await runtime.init_graph()
    rows = []
    try:
        for cid in (report["disconnect_cid"], report["interrupt_cid"]):
            snap = await runtime.get_graph().aget_state({"configurable": {"thread_id": str(cid)}})
            state = snap.values or {}
            context = state.get("task_context") or {}
            rows.append({"conversation_id": cid, "checkpoint_user_id": state.get("user_id"),
                "checkpoint_conversation_id": state.get("conversation_id"), "next": list(snap.next),
                "termination_reason": context.get("termination_reason"), "active_order_id": context.get("active_order_id"),
                "verified_fact_names": sorted((context.get("verified_facts") or {}).keys()),
                "pending_actions": context.get("pending_actions"),
                "pending_interrupt_kinds": [item.value.get("type") for task in snap.tasks for item in task.interrupts
                    if isinstance(item.value, dict)]})
    finally:
        await runtime.close_graph()
        await base.engine.dispose()
    return {"scope": "parent independently reads this run's private checkpoint", "conversations": rows}


def observe_readiness(base: str) -> dict:
    out = {}
    for path in ("/healthz", "/readyz"):
        try:
            try:
                response = urlopen(base + path, timeout=4)
            except HTTPError as exc:
                response = exc
            with response:
                raw = response.read(4097)
                if len(raw) > 4096:
                    raise DiagnosticError("readiness_response_oversized")
                data = json.loads(raw)
                out[path] = {"http_status": response.status,
                    "projection": {k: data[k] for k in ("status", "ready", "checks", "scope") if k in data}}
        except Exception as exc:
            out[path] = {"error_class": type(exc).__name__}
    return {"observed_at": utc_now(), "endpoints": out,
            "scope": "actual HTTP; startup knowledge snapshot, no provider/MCP health claim"}


def wait_ready(owned: OwnedProcesses, base_url: str, allowance: float) -> None:
    deadline = time.monotonic() + allowance
    while time.monotonic() < deadline:
        if not owned.alive():
            raise DiagnosticError("owned_service_exited")
        try:
            with urlopen(base_url + "/openapi.json", timeout=2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(1)
    raise DiagnosticError("api_startup_timeout")


def runner_command(args: argparse.Namespace, directory: Path, code_version: str, case_ids: list[str]) -> tuple[list[str], Path]:
    base = f"http://127.0.0.1:{args.api_port}"
    common = ["--config", str(directory / "run-config.json"), "--base-url", base, "--code-version", code_version]
    probe_names = {"memory": "memory", "fault": "faults", "auth": "boundaries"}
    if args.mode in COMPOUND_MODES:
        report_path = directory / "compound.json"
        command = [sys.executable, str(ROOT / "scripts/eval_compound_agent.py"), *common,
                   "--strategy", "joint-frontier" if args.mode == "compound-routing-b" else
                       "frontier" if args.mode in {"compound-agent", "compound-routing-a"} else "fixed",
                   "--dataset", "routing" if args.mode in ROUTING_MODES else "compound", "--case-ids", *case_ids]
    elif args.mode == "deadline":
        report_path = directory / "deadline.json"
        command = [sys.executable, str(ROOT / "scripts/probe_agent_lifecycle.py"),
            "--phase", "deadline", "--base-url", base, "--code-version", code_version,
            "--declared-turn-timeout-seconds", "0.4"]
    elif args.mode == "disconnect":
        report_path = directory / "before.json"
        command = [sys.executable, str(ROOT / "scripts/probe_agent_lifecycle.py"), "--phase", "before",
            "--base-url", base, "--code-version", code_version, "--declared-turn-timeout-seconds", "120",
            "--state", str(directory / "lifecycle-state.json")]
    elif args.mode == "policy":
        report_path = directory / "policy.json"
        command = [sys.executable, str(ROOT / "scripts/probe_agent_policy.py"), *common]
    elif args.mode == "baseline":
        report_path = directory / "baseline.json"
        command = [sys.executable, str(ROOT / "scripts/eval_course_baseline.py"), "--split", args.split,
            "--case-ids", *case_ids, *common, "--implementation-label", "course-inspired-safe-react"]
        if args.frozen_experiment_plan:
            command += ["--frozen-experiment-sha256", sha256(args.frozen_experiment_plan.read_bytes())]
    elif args.mode in probe_names:
        name = probe_names[args.mode]
        report_path = directory / ("boundaries.json" if args.mode == "auth" else args.mode + ".json")
        command = [sys.executable, str(ROOT / "scripts" / ("probe_agent_" + name + ".py")), *common]
        if args.mode == "fault":
            command += ["--mode", args.fault_mode]
    else:
        report_path = directory / "candidate.json"
        label = f"candidate-coref-ablation-{args.mode[-1].upper()}" if args.mode.startswith("coref-") else f"candidate-{args.split}-diagnostic"
        command = [sys.executable, str(ROOT / "scripts/eval_task_agent.py"), "online", "--split", args.split,
            "--case-ids", *case_ids, *common, "--implementation-label", label]
    return [*command, "--output", str(report_path)], report_path


def run(args: argparse.Namespace, directory: Path) -> int:
    stage = "selection"
    owned = None
    redactor = None
    hashes, paths = {}, []
    try:
        case_ids, selection = select_cases(args)
        fingerprint, hashes, paths = source_snapshot(directory, args, selection)
        experiment = None
        if args.frozen_experiment_plan:
            raw = args.frozen_experiment_plan.read_bytes()
            if len(raw) > 65536:
                raise DiagnosticError("experiment_plan_too_large")
            experiment = json.loads(raw)
            if experiment.get("source_sha256") != fingerprint or experiment.get("label_usage") != "holdout_final_no_tuning":
                raise DiagnosticError("experiment_source_or_label_usage_differs")
        request = {"task_key": args.task_key, "mode": args.mode, "fault_mode": args.fault_mode,
            "guide_corpus": getattr(args, "guide_corpus", False), "selected_evidence": getattr(args, "selected_evidence", False),
            "split": args.split if args.mode in TASK_MODES else None, "case_ids": case_ids, "selection": selection,
            "ports": {"api": args.api_port, "logistics": args.logistics_port, "aftersales": args.aftersales_port},
            "startup_timeout_seconds": args.startup_timeout, "runner_timeout_seconds": args.runner_timeout,
            "plan_only": args.plan_only, "prepared_at": utc_now(), "code_version": "dirty-source-sha256:" + fingerprint}
        write_new(directory, "diagnostic-plan.json", request)
        if args.plan_only:
            emit("plan_prepared_no_env_or_service_access", directory, case_count=len(case_ids), split=request["split"])
            return 0
        stage = "environment"
        settings, redactor = environment(args, directory)
        stage = "port_preflight"
        port_preflight(args)
        stage = "knowledge_preflight"
        knowledge = asyncio.run(knowledge_digest(settings))
        stage = "provider_preflight"
        probe = {"status": "explicitly_skipped", "usage": None} if args.skip_provider_probe else asyncio.run(provider_probe(settings))
        from app.tools.synthetic_fixtures import profile_info
        world = profile_info()
        config = run_config(settings, knowledge, world, args)
        if args.mode == "policy":
            stage = "owned_synthetic_policy_index_setup"
            config["policy_diagnostic"] = asyncio.run(prepare_policy_index(settings))
        if getattr(args, "guide_corpus", False):
            stage = "owned_operations_guide_index_setup"
            config["guide_diagnostic"] = asyncio.run(prepare_guide_index(settings))
        write_new(directory, "run-config.json", config, redactor)
        if experiment and experiment.get("run_config_sha256") != sha256((directory / "run-config.json").read_bytes()):
            raise DiagnosticError("frozen_experiment_configuration_differs")
        stage = "private_account_setup"
        from app.core.auth import create_demo_account
        password = secrets.token_urlsafe(24)
        other_password, admin_password = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        redactor.add(password, other_password, admin_password)
        create_demo_account("customer_demo", password, path=settings.auth_db_path)
        if args.mode == "auth":
            create_demo_account("customer_other", other_password, path=settings.auth_db_path)
            create_demo_account("admin_probe", admin_password, role="admin", path=settings.auth_db_path)
        manifest = {**request, "run_id": directory.name, "stage": args.mode + "_diagnostic",
            "source": str(ROOT), "python_executable": sys.executable, "knowledge": knowledge,
            "fixture": world, "provider_preflight": probe,
            "isolation": "new auth/checkpoint files; existing course MySQL synthetic conversations appended; owned child groups only",
            "credential_handling": "explicit env file plus inherited environment; no env file archived, hashed or printed; stdout/stderr scrubbed",
            "experiment_factor": ({"name": "compound_joint_triage_enabled", "value": args.mode == "compound-routing-b"}
                                  if args.mode in ROUTING_MODES else
                                  {"name": "compound_agent_enabled", "value": args.mode == "compound-agent"}
                                  if args.mode in COMPOUND_MODES else
                                  {"name": "skip_coref_without_history", "value": args.mode == "coref-b"} if args.mode.startswith("coref-") else None),
            "limitations": ["candidate diagnostic, not a course baseline or improvement estimate",
                "only the explicitly selected split's annotations opened", "holdout labels excluded from source archive",
                "model/knowledge config is operator-supplied metadata, not server attestation",
                "unknown prices stay unknown; readiness usage excluded from foreground metrics"], "started_at": utc_now()}
        write_new(directory, "run-manifest.json", manifest, redactor)
        if args.mode == "auth":
            write_new(directory, "boundary-plan.json", {"kind": "real_http_identity_boundaries", "code_version": request["code_version"],
                "requested_probes": 30, "scope": ["server-selected identity", "conversation/admin roles", "readonly write/resume gates",
                    "two mismatched private checkpoints", "non-owned order", "token revocation"],
                "rule": "retain every requested probe; credentials excluded; not dataset accuracy", "prepared_at": utc_now()})
            stage = "private_checkpoint_seed"
            seeds = asyncio.run(seed_stale_checkpoints())
            write_new(directory, "seeded-checkpoint-cases.json", {"scope": "new synthetic SQL conversations and this run's private checkpoint", "cases": seeds})
            os.environ.update(MEWHELP_PROBE_STALE_OWNER_CID=str(seeds[0]["conversation_id"]),
                              MEWHELP_PROBE_STALE_SCOPE_CID=str(seeds[1]["conversation_id"]))
        if args.mode == "fault":
            write_new(directory, "fault-plan.json", {"configuration": config["fault_injection"], "code_version": request["code_version"],
                "mcp_timeout_seconds": 3.0, "readonly_max_retries": 2,
                "rule": "independent fixed demo facts, actual HTTP/MCP, retain failures; not dataset accuracy", "prepared_at": utc_now()})
        emit("prepared", directory, mode=args.mode, case_ids=case_ids)
        stage = "owned_service_startup"
        owned = OwnedProcesses(directory, redactor)
        for name, port in (("logistics", args.logistics_port), ("aftersales", args.aftersales_port)):
            # Mock services need neither provider credentials nor database URLs.
            service_env = {key: value for key, value in os.environ.items()
                if not _SECRET_KEY.search(key) and not key.endswith("DATABASE_URL")}
            service_env.update(PORT=str(port), MOCK_DELAY_SECONDS="0")
            server_file = name + "_server.py"
            if name == "logistics" and args.mode == "fault":
                service_env["MEWHELP_DIAGNOSTIC_FAULT"] = args.fault_mode
                server_file = "diagnostic_logistics_server.py"
            owned.start(name, [sys.executable, str(ROOT / "mcp_servers" / server_file)], service_env)
        owned.start("api", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                           "--port", str(args.api_port), "--workers", "1"], dict(os.environ))
        wait_ready(owned, f"http://127.0.0.1:{args.api_port}", args.startup_timeout)
        write_new(directory, "startup-readiness.json", observe_readiness(f"http://127.0.0.1:{args.api_port}"))
        emit("services_ready", directory)
        stage = "diagnostic_runner"
        eval_env = dict(os.environ)
        eval_env.update(MEWHELP_EVAL_USERNAME="customer_demo", MEWHELP_EVAL_PASSWORD=password,
                        MEWHELP_PROBE_OWNER_PASSWORD=password)
        if args.mode == "auth":
            eval_env.update(MEWHELP_PROBE_OTHER_PASSWORD=other_password, MEWHELP_PROBE_ADMIN_PASSWORD=admin_password)
        command, report_path = runner_command(args, directory, request["code_version"], case_ids)
        evaluated = owned.start("runner", command, eval_env)
        returncode = evaluated.wait(timeout=args.runner_timeout)
        if not report_path.is_file():
            raise DiagnosticError("diagnostic_report_unavailable")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        write_new(directory, "runner-result.json", {"exit_code": returncode, "report": report_path.name,
            "completed_at": utc_now(), "all_requested_failures_retained": True})
        if args.mode in {"auth", "fault", "memory", "policy"}:
            stage = "owned_database_observation"
            if report.get("fresh_conversation_ids"):
                write_new(directory, "database-observation.json", asyncio.run(database_observation(settings, report)), redactor)
            else:
                write_new(directory, "database-observation.json", {"scope": "this run's synthetic conversations only",
                    "observed": False, "reason": "fresh_conversation_ids_unavailable", "ticket_count": None})
        if args.mode in TASK_MODES | COMPOUND_MODES:
            if args.mode == "baseline":
                ids = (report.get("environment_verification") or {}).get("fresh_conversation_ids") or []
            else:
                ids = sorted({turn["response_recording"]["projection"]["conversation_id"]
                    for row in report.get("rows", []) for turn in row.get("turns", [])
                    if (turn.get("response_recording") or {}).get("complete") is True})
            if ids:
                write_new(directory, "database-observation.json", asyncio.run(database_observation(settings, {"fresh_conversation_ids": ids})))
        if args.mode == "memory" and report.get("fresh_conversation_ids"):
            stage = "private_memory_observation"
            write_new(directory, "memory-observation.json", asyncio.run(memory_observation(report)), redactor)
        if args.mode == "fault":
            attempts = []
            for line in (directory / "logistics.log").read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("event") == "diagnostic_tool_attempt":
                    attempts.append(event)
            write_new(directory, "mcp-attempt-observation.json", {"scope": "owned MCP service's scrubbed log", "attempts": attempts}, redactor)
        if args.mode == "deadline":
            stage = "owned_service_stop_for_checkpoint_observation"
            write_new(directory, "owned-process-cleanup.json", owned.close())
            owned = None
            stage = "private_deadline_observation"
            observation = asyncio.run(deadline_observation(report))
            write_new(directory, "deadline-checkpoint-observation.json", observation, redactor)
            cid = report.get("deadline_cid")
            if type(cid) is int and cid > 0:
                write_new(directory, "database-observation.json", asyncio.run(database_observation(
                    settings, {"fresh_conversation_ids": [cid]})), redactor)
            if observation.get("observed") is not True or any(value is not True for value in observation.get("checks", {}).values()):
                returncode = returncode or 1
        if args.mode == "disconnect":
            stage = "independent_checkpoint_before_restart"
            before = asyncio.run(lifecycle_checkpoint_observation(report))
            write_new(directory, "checkpoint-before-restart.json", before)
            checks = {"persisted_client_cancellation": before["conversations"][0]["termination_reason"] == "client_cancelled",
                      "cancelled_has_no_queued_nodes": not before["conversations"][0]["next"],
                      "actual_waiting_selector": before["conversations"][1]["pending_interrupt_kinds"] == ["select_order"]}
            old_pid = next(e["process"].pid for e in owned.children if e["name"] == "api")
            stage = "owned_service_restart"
            cleanup = owned.close()
            write_new(directory, "owned-before-restart-cleanup.json", cleanup)
            owned = OwnedProcesses(directory, redactor)
            for name, port in (("logistics", args.logistics_port), ("aftersales", args.aftersales_port)):
                service_env = {k: v for k, v in os.environ.items() if not _SECRET_KEY.search(k) and not k.endswith("DATABASE_URL")}
                service_env.update(PORT=str(port), MOCK_DELAY_SECONDS="0")
                owned.start(name + "-restarted", [sys.executable, str(ROOT / "mcp_servers" / (name + "_server.py"))], service_env)
            restarted = owned.start("api-restarted", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                "--port", str(args.api_port), "--workers", "1"], dict(os.environ))
            wait_ready(owned, f"http://127.0.0.1:{args.api_port}", args.startup_timeout)
            write_new(directory, "restart-observation.json", {"old_pid": old_pid, "new_pid": restarted.pid,
                "same_auth_file": settings.auth_db_path, "same_checkpoint_file": settings.checkpointer_db_path,
                "old_owned_processes_stopped": all(e["owned_group_stopped"] for e in cleanup["processes"]),
                "scope": "actual owned process stop and relaunch; no containers stopped"})
            command = [sys.executable, str(ROOT / "scripts/probe_agent_lifecycle.py"), "--phase", "after",
                "--base-url", f"http://127.0.0.1:{args.api_port}", "--code-version", request["code_version"],
                "--declared-turn-timeout-seconds", "120", "--state", str(directory / "lifecycle-state.json"),
                "--output", str(directory / "after.json")]
            after_runner = owned.start("after-runner", command, eval_env)
            after_code = after_runner.wait(timeout=args.runner_timeout)
            after_report = json.loads((directory / "after.json").read_text(encoding="utf-8"))
            write_new(directory, "checkpoint-after-restart.json", asyncio.run(lifecycle_checkpoint_observation(report)))
            ids = [report["disconnect_cid"], report["interrupt_cid"]]
            write_new(directory, "database-observation.json", asyncio.run(database_observation(settings, {"fresh_conversation_ids": ids})))
            checks["after_public_checks_complete"] = not after_report["summary"]["failed_checks"] and not after_report["summary"]["unknown_checks"]
            checks["different_api_pid"] = old_pid != restarted.pid
            write_new(directory, "lifecycle-parent-checks.json", {"checks": checks, "passed": all(checks.values())})
            returncode = returncode or after_code or int(not all(checks.values()))
        emit("evaluated", directory, summary=redactor.value(report.get("summary")), runner_exit_code=returncode)
        # A recorded failure/unknown remains a nonzero result with its report.
        return returncode
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"stage": stage, "error_class": type(exc).__name__, "not_an_agent_score": True,
                   "code": str(exc) if isinstance(exc, DiagnosticError) else "diagnostic_stage_unavailable"}
        write_new(directory, "environment-failure.json", failure, redactor)
        emit("environment_failure", directory, stage=stage, error_class=type(exc).__name__)
        return 1
    finally:
        if owned is not None:
            try:
                write_new(directory, "owned-process-cleanup.json", owned.close())
            except Exception as exc:
                write_new(directory, "owned-process-cleanup-failure.json", {"error_class": type(exc).__name__,
                    "scope": "owned groups only", "requires_operator_review": True})
        if paths:
            try:
                source_end_check(directory, args, hashes, paths)
            except Exception as exc:
                write_new(directory, "source-stability-failure.json", {"error_class": type(exc).__name__})


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    validate_args(args, argument_parser)
    directory = args.output_dir.expanduser().resolve()
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError:
        print(json.dumps({"event": "new_output_directory_required"}), flush=True)
        return 1
    result = run(args, directory)
    # Preserve the report but never label a changed source or failed owned
    # cleanup as a reproducible successful coordinator run.
    instability = directory / "source-stability.json"
    if (directory / "owned-process-cleanup-failure.json").exists() or (directory / "source-stability-failure.json").exists():
        return result or 1
    if instability.is_file() and json.loads(instability.read_text(encoding="utf-8")).get("unchanged") is not True:
        emit("source_changed_during_run", directory)
        return result or 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
