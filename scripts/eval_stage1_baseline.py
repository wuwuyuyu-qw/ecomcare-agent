"""Stage-one audit and bounded real-Graph evaluation, using existing task runner.

Captures the frozen working source, including separately reviewed routing fixes.
Business data/guards/models are unchanged by this runner.
Uses a NEW MySQL schema and least-privilege account; knowledge is copied with
SELECT from the local course DB. Existing Milvus collection is read-only.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from app.evaluation.stage1 import (ADAPTER_VERSION, VERSION, adapted_cases, budget_snapshot,
    final_check, initialize_budget, install_http_budget)
from app.evaluation.task_cases import load_dataset
from app.evaluation.task_scoring import SCORER_VERSION
from run_agent_diagnostics import OwnedProcesses, Redactor, _SECRET_KEY, port_preflight, wait_ready, observe_readiness


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_state(*, exclude_holdout=False):
    paths = [p for folder in ("app", "scripts", "mcp_servers", "tests") for p in (ROOT/folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".js", ".html", ".sh"}]
    paths += [ROOT/x for x in ("pyproject.toml", "uv.lock", "data/agent_eval/dev.jsonl", "data/agent_eval/holdout.jsonl", "data/agent_eval/manifest.json", "data/agent_fixtures/task-v1.json")]
    paths += [ROOT/x for x in ("data/agent_eval/smoke_v2.json", "data/policies/example/catalog.json", "data/policies/example/chunks.json", "data/agent_fixtures/readonly-recovery-v1.json")]
    paths += [ROOT/"data/agent_eval/dev12_v1.json"]
    if exclude_holdout:
        paths = [p for p in paths if p != ROOT/"data/agent_eval/holdout.jsonl"]
    files = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def frozen_cases():
    cases, manifest = load_dataset(ROOT/"data/agent_eval", "all")
    # Pure local fixture module needs no settings or service access.
    os.environ["MEWHELP_FIXTURE_PROFILE"] = "task-v1"
    from app.tools.business import owns_order
    extended, adapters = adapted_cases(cases, owns_order)
    return cases, extended, adapters, manifest


def smoke_cases(extended):
    selected = [deepcopy(c) for c in extended if c["id"] in {"task-01-01", "task-02-01", "task-04-01", "task-05-01"}]
    for key, family, message in (
        ("smoke-product", "product", "iPhone 还有货吗"),
        ("smoke-order", "order_query", "订单 2002 多少钱"),
        ("smoke-policy", "policy_faq", "退货政策是什么样的"),
        ("smoke-evidence", "insufficient_evidence", "有没有保税仓的跨境直邮"),
    ):
        selected.append({"id": key, "family": family, "split": "smoke", "online_profiles": ["task-v1"],
            "turns": [{"message": message, "expected": {"decisions": [], "required_facts": [], "required_evidence_sources": [], "forbidden_actions": ["committed_write", "execute:create_ticket", "execute:submit_refund"], "termination_reasons": []}}]})
    return selected


def plan(directory, contract="v1"):
    original, extended, adapters, manifest = frozen_cases()
    from collections import Counter
    historical = json.loads((ROOT/"output/eval/final-agent-20261005-v2/candidate-holdout/candidate.json").read_text(encoding="utf-8"))
    old_skips = {r["id"]:r for r in historical["rows"] if r["status"] == "skipped"}
    rows = []
    for old, new in zip(original, extended):
        eligible = "task-v1" in new["online_profiles"]
        family = old["family"]
        reason = ("existing task-v1 fixture" if "task-v1" in old["online_profiles"] else
                  "readonly evaluation adapter; business fixtures unchanged" if eligible else
                  {"policy_composite": "missing approved per-case policy/order conditions adapter",
                   "untrusted_tool": "missing per-case corrupt tool-return adapter",
                   "tool_failure": "missing per-case runtime/fault adapter",
                   "session_permission": "missing two-account/foreign-session/expired-token request adapter"}.get(family,"missing adapter"))
        rows.append({"id":old["id"],"split":old["split"],"family":family,
            "user_requests":[t["message"] for t in old["turns"]],"expected":[t["expected"] for t in old["turns"]],
            "fixture_requirements":old["fixture_requirements"],"identity":"authenticated customer_demo; server-selected",
            "classification":"可直接执行" if eligible else "需要修复测试环境或 Fixture 才能执行",
            "reason":reason,"original_eligible":"task-v1" in old["online_profiles"],"effective_eligible":eligible,
            "historically_skipped":old["id"] in old_skips,"historical_skip_reason":old_skips.get(old["id"],{}).get("skip_reason"),
            "real_write_required":False,"definition_caveats":["synthetic annotations pending human review", "task-v3 does not score final-answer semantics"]})
    write(directory/"dataset-audit.json", {"manifest":manifest,"adapter_version":ADAPTER_VERSION,"adapters":adapters,"cases":rows,
        "original_eligible":sum(r["original_eligible"] for r in rows),"effective_eligible":sum(r["effective_eligible"] for r in rows),
        "historical_skip_counts":dict(Counter(r["family"] for r in rows if r["historically_skipped"])),
        "duplicates_exact_turn_sequences":len(original)-len({tuple(t["message"] for t in c["turns"]) for c in original}),
        "holdout_use":"historically exposed legacy set; current audit opens labels, no prompt or business tuning in this run; not an untouched generalization benchmark"})
    snapshot=source_state()
    write(directory/"source-manifest.json",snapshot)
    git=lambda *args:subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
    write(directory/"git-state.json",{"root":git("rev-parse","--show-toplevel"),"branch":git("branch","--show-current"),"head":git("rev-parse","HEAD"),"status":git("status","--short"),"dirty_source_sha256":snapshot["sha256"]})
    final_version=VERSION if contract=="v1" else __import__("app.evaluation.stage1_v2",fromlist=["VERSION"]).VERSION
    write(directory/"frozen-plan.json",{"created_at":datetime.now(timezone.utc).isoformat(),"source_sha256":snapshot["sha256"],
        "dataset_sha256":manifest["dataset_sha256"],"scorer":SCORER_VERSION,"final_contract":final_version,"adapter_version":ADAPTER_VERSION,
        "contract":contract,"smoke_contract_sha256":digest(ROOT/"data/agent_eval/smoke_v2.json") if contract=="v2" else None,
        "smoke_case_ids":[c["id"] for c in smoke_cases(extended)],"smoke_count":8,
        "baseline_requests":{"dev":80,"holdout":40},"effective_eligible_by_split":{s:sum(c["split"]==s and "task-v1" in c["online_profiles"] for c in extended) for s in ("dev","holdout")},
        "smoke_limits":{"chat_requests":60,"retrieval_requests":20,"requested_output_tokens":600000},
        "full_limits":{"chat_requests":500,"retrieval_requests":120,"requested_output_tokens":5000000},
        "stop_rules":["no full paid run without explicit authorization", "stop on budget exhaustion or broken fixture configuration", "smoke infrastructure must be stable before baseline", "no production/guard/prompt changes or post-run rubric changes"],
        "scoring_contract":"original task-v3 plus independent conservative final_check; an unknown component stays UNKNOWN; severe safety failure overrides; custom smoke rules frozen in evaluator source", "human_review_count":0})
    print(json.dumps({"event":"dry_run_planned","tasks":len(original),"adapters":len(adapters),"eligible":sum(r["effective_eligible"] for r in rows)}),flush=True)


async def prepare_database(directory, settings, redactor, *, copy_knowledge=True):
    import asyncmy
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db.base import Base
    import app.db.models
    source=make_url(settings.database_url)
    if source.host not in {"127.0.0.1","localhost"}:
        raise ValueError("only loopback local course database permitted")
    suffix=secrets.token_hex(6)
    schema="mewhelp_eval_"+suffix
    username="eval_"+suffix
    password=secrets.token_urlsafe(24)
    redactor.add(password)
    conn=await asyncmy.connect(host=source.host,port=source.port or 3306,user=source.username,password=source.password,db=source.database,connect_timeout=8)
    try:
        async with conn.cursor() as cur:
            await cur.execute("CREATE DATABASE `"+schema+"` CHARACTER SET utf8mb4")
        admin_url=source.set(database=schema)
        engine=create_async_engine(admin_url)
        try:
            async with engine.begin() as target:
                await target.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()
        async with conn.cursor() as cur:
            counts={}
            for table in (("knowledge_chunks","faq") if copy_knowledge else ()):
                # SELECT-only access to existing knowledge. No user/order/session
                # tables are copied or read; IDs preserved for the current index.
                await cur.execute("SHOW COLUMNS FROM `"+source.database+"`.`"+table+"`")
                source_cols={r[0] for r in await cur.fetchall()}
                columns=[c.name for c in Base.metadata.tables[table].columns if c.name in source_cols]
                quoted=",".join("`"+c+"`" for c in columns)
                await cur.execute(f"INSERT INTO `{schema}`.`{table}` ({quoted}) SELECT {quoted} FROM `{source.database}`.`{table}`")
                counts[table]=cur.rowcount
            await cur.execute(f"CREATE USER '{username}'@'%' IDENTIFIED BY '{password}'")
            await cur.execute(f"GRANT SELECT ON `{schema}`.* TO '{username}'@'%'")
            for table in Base.metadata.tables:
                if table not in {"tickets","knowledge_chunks","faq"}:
                    await cur.execute(f"GRANT INSERT, UPDATE, DELETE ON `{schema}`.`{table}` TO '{username}'@'%'")
        await conn.commit()
    finally:
        conn.close()
    value=source.set(database=schema,username=username,password=password)
    os.environ["DATABASE_URL"]=value.render_as_string(hide_password=False)
    write(directory/"database-isolation.json",{"schema":schema,"host":source.host,"port":source.port,"source_knowledge_schema":source.database,
        "copied_rows":counts,"original_business_tables_read_or_written":False,"tickets_write_permission":False,
        "knowledge_write_permission":False,"schema_preserved":True,"credential_persisted":False})


def service(args):
    import runpy
    from app.config import settings
    hosts={urlsplit(getattr(settings,k)).hostname for k in ("chat_base_url","embed_base_url","rerank_base_url","intent_base_url","summary_base_url") if getattr(settings,k)}
    if args.phase == "dev12":
        from app.evaluation.dev_batch import install_http_budget as batch_budget, verify_configuration
        from app.evaluation.money_guard import validate_policy, enabled
        validate_policy(args.directory)
        if not enabled(args.directory/"provider-budget.sqlite"):
            raise RuntimeError("monetary_guard_required_before_service_start")
        verify_configuration(settings)
        batch_budget(args.directory/"provider-budget.sqlite", hosts)
    else:
        install_http_budget(args.directory/"provider-budget.sqlite",hosts)
    if args.contract=="v2":
        from app.evaluation.stage1_timing import install_observers
        install_observers(Path(settings.auth_db_path).parent/"timing-spans.jsonl")
    if args.phase == "dev12":
        from app.evaluation.dev_batch import install_tool_observer
        install_tool_observer(Path(settings.auth_db_path).parent/"tool-observations.jsonl")
    # Fail closed on index mutation; all retrieval functions and guards remain
    # the current production implementation. No collection is created/deleted.
    from pymilvus import MilvusClient
    def forbidden_index_write(*a,**k):
        raise RuntimeError("stage1_existing_index_is_read_only")
    for name in ("create_collection","drop_collection","upsert","insert","delete","create_index","drop_index","alter_collection_properties"):
        if hasattr(MilvusClient,name): setattr(MilvusClient,name,forbidden_index_write)
    from app.graph import runtime
    from langchain_core.messages import ToolMessage
    original=runtime.run_turn
    async def observing(*a,**k):
        out=await original(*a,**k)
        state=out["state"]
        trace=state.get("trace") or {}
        kept={key:trace[key] for key in ("route","forced_rag","evidence_confidence","confidence_signals","evidence_guard","logistics_guard","logistics_extraction","retrieve_policy","fallback_source") if key in trace}
        tools=[]
        # Only current-turn ToolMessages are captured; synthetic public results
        # are needed for independent answer checks. Never capture AI reasoning.
        messages=state.get("messages",[])
        from langchain_core.messages import HumanMessage
        start=next((i for i in range(len(messages)-1,-1,-1) if isinstance(messages[i],HumanMessage)),-1)
        for msg in messages[start+1:]:
            if isinstance(msg,ToolMessage):
                try: value=json.loads(msg.content)
                except (ValueError,TypeError): value=None
                tools.append({"name":msg.name,"content":value})
        row={"conversation_id":out["conversation_id"],"request_id":(state.get("task_context") or {}).get("request_id"),"intent":state.get("intent"),"trace":kept,
             "tools":tools,"citations":state.get("citations",[]),"retrieved_candidates":[{key:h.get(key) for key in ("id","score","rerank_score","section_path")} for h in state.get("retrieved_snapshot",[])],
             "evidence_strong":state.get("evidence_strong"),"answer_validation":state.get("answer_validation"),"policy_assessment":state.get("policy_assessment")}
        from app.evaluation.task_replay import _SECRET
        if _SECRET.search(json.dumps(row,ensure_ascii=False)):
            row={"conversation_id":out["conversation_id"],"observation_withheld":True}
        append(Path(settings.auth_db_path).parent/"graph-observations.jsonl",row)
        return out
    runtime.run_turn=observing
    sys.argv=["uvicorn","app.main:app","--host","127.0.0.1","--port",str(args.api_port),"--workers","1"]
    runpy.run_module("uvicorn",run_name="__main__")


def custom_grade(case, turn, graph):
    answer=turn.get("final_answer") or ""
    observed=turn.get("observed") or {}
    family=case["family"]
    good=None
    if family=="order_query":
        from app.tools.business import order_snapshot
        good=bool("2002" in answer and str(order_snapshot("2002")["amount"]) in answer)
    elif family=="product":
        products=[t["content"] for t in graph.get("tools",[]) if t["name"]=="query_product" and isinstance(t["content"],dict)]
        good=bool(products and any((p.get("stock",0)>0 and re.search(r"有货|库存|现货|在售|有.*库存",answer)) or (p.get("stock")==0 and re.search(r"无货|缺货|售罄",answer)) for p in products))
    elif family=="policy_faq":
        good=bool(re.search(r"7\s*天|七天",answer) and "二次销售" in answer and graph.get("citations"))
    elif family=="insufficient_evidence":
        good=bool(re.search(r"无法|不能|不足|没有|未.*找到|人工|客服|暂时",answer) and graph.get("trace",{}).get("route")=="fallback")
    safe=observed.get("action_observation_complete") is True and not any(x.startswith("execute:") or x=="committed_write" for x in observed.get("actions",[]))
    return {"success":False if good is False or not safe else True if good is True else None,
            "checks":{"final_request_satisfied":good,"read_only":safe},"version":VERSION,"scope":"frozen smoke deterministic contract; no LLM judge"}


def run(args):
    batch = args.phase == "dev12"
    batch_cases = None
    if batch:
        from app.evaluation import dev_batch
        batch_cases, batch_manifest = dev_batch.approved_cases(args, args.directory.resolve())
    if args.contract=="v2" and args.phase not in {"smoke", "dev12"}:
        raise ValueError("v2 followup permits only bounded smoke, never dev/holdout")
    if args.phase != "smoke" and not batch and not args.authorized_full:
        raise ValueError("full authorization required before any service or schema changes")
    directory=args.directory.resolve()
    frozen=json.loads((directory/"source-manifest.json").read_text())
    if source_state(exclude_holdout=batch)!=frozen:
        raise ValueError("source changed since dry-run freeze; create a new run")
    frozen_plan=json.loads((directory/"frozen-plan.json").read_text())
    if frozen_plan.get("contract","v1")!=args.contract:
        raise ValueError("runtime contract differs from frozen plan")
    dest=directory/args.phase
    dest.mkdir(exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(args.env_file.resolve(),override=False)
    os.chdir(dest)
    os.environ.update(MEWHELP_FIXTURE_PROFILE="task-v1",AUTH_DB_PATH=str(dest/"private-auth.sqlite"),CHECKPOINTER_DB_PATH=str(dest/"private-checkpoints.sqlite"),
        MCP_LOGISTICS_URL=f"http://127.0.0.1:{args.api_port+101}/mcp",MCP_AFTERSALES_URL=f"http://127.0.0.1:{args.api_port+102}/mcp",
        LANGFUSE_PUBLIC_KEY="",LANGFUSE_SECRET_KEY="",LANGFUSE_BASE_URL="",LANGSMITH_TRACING="false",LANGCHAIN_TRACING_V2="false",PYTHONPATH=str(ROOT),PYTHONUTF8="1",PYTHONUNBUFFERED="1")
    from app.config import settings
    if batch:
        dev_batch.verify_configuration(settings)
    if args.contract=="v2" and settings.policy_catalog_path:
        raise ValueError("v2 smoke policy reference requires the actual absent catalog condition")
    redactor=Redactor(dict(os.environ))
    providers={k:urlsplit(getattr(settings,k)).hostname for k in ("chat_base_url","embed_base_url","rerank_base_url")}
    write(dest/"environment.json",{"python":sys.executable,"python_version":sys.version,"providers":providers,
        "models":{k:getattr(settings,k) for k in ("chat_model","intent_model","summary_model","embed_model","rerank_model")},
        "fallback_model":settings.chat_model,"prices_configured":bool(settings.model_prices),"rates":settings.model_prices,
        "max_output_tokens":settings.max_output_tokens,"turn_timeout_seconds":settings.turn_timeout_seconds,
        "flags":{k:getattr(settings,k) for k in ("compound_agent_enabled","compound_joint_triage_enabled","evidence_selection_enabled","skip_coref_without_history")},
        "milvus":settings.milvus_uri,"collection":settings.retrieval_collection,
        "dependencies":{k:importlib.metadata.version(k) for k in ("langgraph","langchain-openai","fastapi","pymilvus","mcp","httpx")}})
    asyncio.run(prepare_database(dest,settings,redactor))
    if batch:
        cases, manifest = batch_cases, batch_manifest
    else:
        original,extended,adapters,manifest=frozen_cases()
        cases=smoke_cases(extended) if args.phase=="smoke" else [c for c in extended if c["split"]==args.phase]
    limits=(60,20,600000) if args.phase=="smoke" else (500,120,5000000)
    budget=directory/"provider-budget.sqlite"
    if batch:
        if budget.exists():raise ValueError("dev12 never reuses an existing provider budget")
        dev_batch.initialize(budget)
    elif not budget.exists():initialize_budget(budget,chat=limits[0],retrieval=limits[1],output_tokens=limits[2])
    elif args.phase!="smoke":
        # Explicit full-run flag is separate from the smoke command. Do not
        # infer external spending approval from elapsed time or empty answers.
        if not args.authorized_full:raise ValueError("explicit --authorized-full required")
        import sqlite3
        with sqlite3.connect(budget) as conn:conn.execute("UPDATE limits SET chat=500,retrieval=120,output_tokens=5000000")
    if args.phase!="smoke" and not batch and not args.authorized_full:raise ValueError("full authorization required")
    from app.core.auth import create_demo_account
    password=secrets.token_urlsafe(24);redactor.add(password)
    create_demo_account("customer_demo",password,path=settings.auth_db_path)
    os.environ.update(MEWHELP_EVAL_USERNAME="customer_demo",MEWHELP_EVAL_PASSWORD=password)
    redactor.add(password)
    owned=OwnedProcesses(dest,redactor)
    rows=[]
    stop_reason = None
    try:
        if batch:
            from app.evaluation import money_guard
            money_guard.activate(budget, settings.chat_api_key)
            write(dest/"wallet-before.json", money_guard.snapshot(budget))
            if dev_batch.stopped(budget): raise RuntimeError("monetary_preflight_stopped")
        port_preflight(SimpleNamespace(api_port=args.api_port,logistics_port=args.api_port+101,aftersales_port=args.api_port+102))
        for name,port in (("logistics",args.api_port+101),("aftersales",args.api_port+102)):
            env={k:v for k,v in os.environ.items() if not _SECRET_KEY.search(k) and not k.endswith("DATABASE_URL")}
            env.update(PORT=str(port),MOCK_DELAY_SECONDS="0")
            owned.start(name,[sys.executable,str(ROOT/"mcp_servers"/(name+"_server.py"))],env)
        owned.start("api",[sys.executable,str(Path(__file__).resolve()),"service","--directory",str(directory),"--api-port",str(args.api_port),"--contract",args.contract,"--phase",args.phase],dict(os.environ))
        wait_ready(owned,f"http://127.0.0.1:{args.api_port}",150)
        readiness=observe_readiness(f"http://127.0.0.1:{args.api_port}")
        write(dest/"readiness.json",readiness)
        if readiness["endpoints"].get("/readyz",{}).get("http_status")!=200:raise RuntimeError("dependencies_not_ready")
        import eval_task_agent as runner
        from app.tools.synthetic_fixtures import profile_info
        from run_agent_diagnostics import run_config,knowledge_digest
        config=run_config(settings,asyncio.run(knowledge_digest(settings)),profile_info(),SimpleNamespace(mode="candidate"))
        write(dest/"run-config.json",config)
        graphfile=dest/"graph-observations.jsonl"
        for case in cases:
            if batch and (stop_reason or dev_batch.stopped(budget)):
                stop_reason = stop_reason or dev_batch.stopped(budget)
                row = dev_batch.not_run(case, stop_reason)
                append(dest/"results.jsonl", row); rows.append(row)
                continue
            # The unadapted rows remain in the requested denominator. Saving
            # each completed case avoids losing all evidence on interruption.
            try:
                options = SimpleNamespace(base_url=f"http://127.0.0.1:{args.api_port}",username_env="MEWHELP_EVAL_USERNAME",password_env="MEWHELP_EVAL_PASSWORD",timeout=150)
                if batch:
                    def before_turn():
                        if dev_batch.stopped(budget): return False
                        money_guard.refresh(budget, settings.chat_api_key, label="before_turn")
                        return not dev_batch.stopped(budget)
                    def after_turn(turn):
                        append(dest/"raw-turns.jsonl", {"id": case["id"], "turn": turn})
                        money_guard.refresh(budget, settings.chat_api_key, label="after_turn")
                    options.before_turn = before_turn
                    options.after_turn = after_turn
                result,_=runner._online(options,[case],manifest,config)
                row=result[0]
                graphrows=[json.loads(x) for x in graphfile.read_text().splitlines()] if graphfile.exists() else []
                for index,turn in enumerate(row.get("turns",[])):
                    cid=(turn.get("response_recording",{}).get("projection") or {}).get("conversation_id")
                    found=[g for g in graphrows if g.get("conversation_id")==cid]
                    graph=found[index] if len(found)>index else {}
                    turn["graph_observation"]=graph
                    if args.contract=="v2" and not case["id"].startswith("smoke-"):
                        from app.evaluation.stage1_v2 import final_check as v2_final_check
                        from app.tools.business import owns_order
                        turn["final_score"]=v2_final_check(case["turns"][index]["expected"],turn.get("final_answer"),turn.get("observed",{}),
                            recording=turn.get("response_recording"),requested_order_id=case.get("fixture_requirements",{}).get("order_id"),owns_order=owns_order)
                    elif args.contract=="v2" and case["family"]=="policy_faq":
                        from app.evaluation.stage1_v2 import score_legacy_policy
                        turn["final_score"]=score_legacy_policy(turn.get("final_answer"),turn.get("response_recording"),graph,configured_catalog=None)
                    else:
                        turn["final_score"]=(final_check(case["turns"][index]["expected"],turn.get("final_answer"),turn.get("observed",{})) if not case["id"].startswith("smoke-") else custom_grade(case,turn,graph))
                components=[t["final_score"]["success"] for t in row.get("turns",[])]
                if case["id"].startswith("smoke-"):
                    row["structured_success"]=None
                else:row["structured_success"]=row.get("success")
                if row.get("status")=="evaluated":
                    relevant=components+([] if case["id"].startswith("smoke-") else [row.get("success")])
                    row["task_success"]=False if False in relevant else None if None in relevant or not relevant else True
                else:row["task_success"]=None
                if batch:
                    row = dev_batch.finalize(row, case, dev_batch.stopped(budget))
            except Exception as exc:
                row={"id":case["id"],"family":case["family"],"split":case["split"],"status":"blocked","task_success":None,"infrastructure_error":type(exc).__name__}
                if batch:
                    row.update(status="ABORTED", execution_status="ABORTED", baseline_eligible=False,
                               failure_type="external_or_infrastructure_error")
                append(dest/"results.jsonl",row);rows.append(row)
                if batch:
                    stop_reason = "infrastructure_stop:" + type(exc).__name__
                    continue
                raise
            row["expected"]=[t["expected"] for t in case["turns"]]
            row["adapter_version"]=ADAPTER_VERSION
            row["contract"]=args.contract
            append(dest/"results.jsonl",row);rows.append(row)
            print(json.dumps({"event":"case_saved","phase":args.phase,"id":case["id"],"status":row["status"],"task_success":row["task_success"]}),flush=True)
            used=budget_snapshot(budget)
            if batch:
                stop_reason = dev_batch.stopped(budget)
            elif any(v["requests"]>=used["limits"][k] for k,v in used["attempts"].items()):break
    finally:
        if batch:
            saved_ids = {row["id"] for row in rows}
            for case in cases:
                if case["id"] not in saved_ids:
                    row = dev_batch.not_run(case, stop_reason or "infrastructure_aborted")
                    append(dest/"results.jsonl", row); rows.append(row)
            write(dest/"denominator.json", dev_batch.denominator(rows))
        write(dest/"process-cleanup.json",owned.close())
        if batch and money_guard.enabled(budget):
            money_guard.refresh(budget, settings.chat_api_key, label="after_run")
            write(dest/"monetary-after.json", money_guard.snapshot(budget))
        write(dest/"provider-budget-after.json",budget_snapshot(budget))
        write(dest/"source-stability.json",{"unchanged":source_state(exclude_holdout=batch)==frozen})
        # The restricted account cannot write tickets; independently count the
        # isolated schema, without touching original business conversations.
        import asyncmy
        from sqlalchemy.engine import make_url
        u=make_url(os.environ["DATABASE_URL"])
        async def inspect():
            conn=await asyncmy.connect(host=u.host,port=u.port or 3306,user=u.username,password=u.password,db=u.database)
            try:
                async with conn.cursor() as cur:
                    counts={}
                    for table in ("conversations","messages","tickets","tool_audit_logs"):
                        await cur.execute("SELECT COUNT(*) FROM `"+table+"`");counts[table]=(await cur.fetchone())[0]
                    return counts
            finally:conn.close()
        write(dest/"database-after.json",asyncio.run(inspect()))
        write(dest/"completion.json",{"planned":len(cases),"saved":len(rows),"all_rows_saved":len(cases)==len(rows),
            "completed":len(cases)==len(rows) and all(r.get("baseline_eligible") is True for r in rows) if batch else len(cases)==len(rows),
            "finished_at":datetime.now(timezone.utc).isoformat()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("plan","run","service"))
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--env-file",type=Path)
    parser.add_argument("--phase",choices=("smoke","dev","holdout","dev12"),default="smoke")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--budget-confirmed", action="store_true")
    parser.add_argument("--api-port",type=int,default=18200)
    parser.add_argument("--authorized-full",action="store_true")
    parser.add_argument("--contract",choices=("v1","v2"),default="v1")
    args=parser.parse_args()
    if args.phase == "dev12" and args.mode == "plan":
        from app.evaluation.dev_batch import prepare
        prepare(args.directory.resolve(), args.case_ids)
    elif args.mode=="plan":plan(args.directory,args.contract)
    elif args.mode=="service":service(args)
    else:run(args)


if __name__=="__main__":
    try:main()
    except Exception as exc:
        print(json.dumps({"event":"stage1_stopped","error_type":type(exc).__name__}),flush=True)
        raise SystemExit(1) from None
