"""Foreground local launcher; Ctrl+C stops only its own API/MCP children."""
import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import sys
import time
from types import SimpleNamespace

from run_agent_diagnostics import (
    ROOT, OwnedProcesses, Redactor, _SECRET_KEY, port_preflight,
    wait_ready, observe_readiness, write_new,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    envfile = args.env_file.resolve(strict=True)
    from dotenv import load_dotenv
    load_dotenv(envfile, override=False)
    for key, default in (("AUTH_DB_PATH", "data/demo_auth.sqlite"),
                         ("CHECKPOINTER_DB_PATH", "data/ch05_checkpoints.sqlite"),
                         ("POLICY_CATALOG_PATH", "")):
        value = os.environ.get(key, default)
        if value and not Path(value).is_absolute():
            value = str((ROOT / value).resolve())
        os.environ[key] = value
    os.environ.update(PYTHONPATH=str(ROOT), PYTHONUTF8="1", PYTHONUNBUFFERED="1",
        MCP_LOGISTICS_URL="http://127.0.0.1:8101/mcp", MCP_AFTERSALES_URL="http://127.0.0.1:8102/mcp")
    runid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    directory = ROOT / "log/local-agent" / runid
    directory.mkdir(parents=True, exist_ok=False)
    os.chdir(directory)  # No implicit second .env file.
    redactor = Redactor(dict(os.environ))
    owned = OwnedProcesses(directory, redactor)
    try:
        port_preflight(SimpleNamespace(api_port=8000, logistics_port=8101, aftersales_port=8102))
        for name, port in (("logistics", 8101), ("aftersales", 8102)):
            env = {k: v for k, v in os.environ.items() if not _SECRET_KEY.search(k) and not k.endswith("DATABASE_URL")}
            env["PORT"] = str(port)
            owned.start(name, [sys.executable, str(ROOT / "mcp_servers" / (name + "_server.py"))], env)
        owned.start("api", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                            "--port", "8000", "--workers", "1"], dict(os.environ))
        wait_ready(owned, "http://127.0.0.1:8000", 100)
        readiness = observe_readiness("http://127.0.0.1:8000")
        write_new(directory, "startup-readiness.json", readiness)
        if readiness["endpoints"].get("/readyz", {}).get("http_status") != 200:
            raise RuntimeError("Agent dependencies not ready")
        from app.tools import registry
        specs = asyncio.run(registry.get_all_specs())
        names = {s.name for s in specs}
        if not {"query_logistics", "query_warranty", "query_return_status"} <= names:
            raise RuntimeError("MCP handshake incomplete")
        write_new(directory, "owned-services.json", {"processes": [{"service": e["name"], "pid": e["process"].pid} for e in owned.children],
                   "mcp_tools": sorted(names), "scope": "owned children only; one API worker"})
        print("本地 Agent 已启动：http://127.0.0.1:8000；Ctrl+C 关闭本次应用与 MCP。", flush=True)
        print("运行记录：" + str(directory), flush=True)
        while owned.alive():
            time.sleep(0.5)
        raise RuntimeError("owned service exited")
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print("启动或运行失败：" + type(exc).__name__ + "；查看本次运行记录。", flush=True)
        return 1
    finally:
        write_new(directory, "owned-process-cleanup.json", owned.close())


if __name__ == "__main__":
    raise SystemExit(main())
