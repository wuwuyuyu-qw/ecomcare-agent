"""Deterministic, model-free adapters for six previously unavailable tasks.

Only an explicit evaluation context binds a private read-only catalog and local
tool transport. Ownership, Guard, contracts, investigator and runtime are real.
This is a component/lifecycle measurement, NOT a real-model E2E score. No original
labels, production profile registration, network or business tables are changed.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

VERSION = "readonly-recovery-v1"
FIXTURE = Path(__file__).resolve().parents[2] / "data/agent_fixtures/readonly-recovery-v1.json"


def load_fixture():
    raw = FIXTURE.read_bytes()
    data = json.loads(raw)
    if data["version"] != VERSION or len(data["cases"]) != 6:
        raise ValueError("invalid recovery fixture")
    return data, hashlib.sha256(raw).hexdigest()


@contextmanager
def readonly_world(data, digest):
    from app.tools import contracts, synthetic_fixtures as fixtures
    orders = {c["order_id"]: {**data["order"], "order_id": c["order_id"], "tracking_no": c["tracking_no"]}
              for c in data["cases"]}
    logistics = {c["tracking_no"]: {**data["logistics"], "tracking_no": c["tracking_no"]} for c in data["cases"]}
    for oid, order in orders.items():
        contracts.validate_result("query_order", order, {"order_id": oid})
    for tracking, value in logistics.items():
        contracts.validate_result("query_logistics", value, {"tracking_no": tracking})
    # The underlying catalog dependency is substituted, never owns_order or Guard.
    # Transport uses demo-shaped payloads; this is a distinct evaluation world,
    # explicitly NOT a task-v1 bilateral MCP profile identity experiment.
    catalog = fixtures._Catalog("demo", VERSION, digest, fixtures._freeze(orders),
                               fixtures._freeze(logistics), fixtures._freeze({oid: data["owner"] for oid in orders}))
    with patch.object(fixtures, "_catalog", return_value=catalog):
        yield


class ReadTransport:
    def __init__(self, call):
        self.call, self.calls, self.cancelled = call, [], False

    async def ainvoke(self, args):
        self.calls.append(deepcopy(args))
        try:
            return await self.call(args)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def transports(data, case):
    from app.tools import business, registry, synthetic_fixtures
    from app.tools.builtin.orders import query_order
    real_order = registry.spec_from_langchain_tool(query_order, source="builtin", inject_user_id=True,
                                                  max_retries=0, timeout=2)

    async def order(args):
        if case["fault"] == "deadline":
            await asyncio.sleep(2)
        value = await query_order.ainvoke(args)
        return value

    async def logistics(args):
        value = synthetic_fixtures.logistics_snapshot(args["tracking_no"])
        # MCP transport's external content blocks, through real engine parser.
        if case["fault"] == "missing_tracking":
            value.pop("tracking_no")
        elif case["fault"] == "trace_not_list":
            value["trace"] = "malformed-list"
        elif case["fault"] == "non_object_result":
            value = ["non-object-result"]
        elif case["fault"] == "partial_evidence":
            raise ValueError("local synthetic logistics failure")
        return [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]

    order_transport, logistics_transport = ReadTransport(order), ReadTransport(logistics)
    real_order.tool = order_transport
    remote = registry.ToolSpec("query_logistics", "evaluation-only local MCP-shaped read fixture",
        {"type": "object", "properties": {"tracking_no": {"type": "string"}}, "required": ["tracking_no"], "additionalProperties": False},
        logistics_transport, "read", "mcp", mcp_server="logistics", max_retries=0, timeout=2)
    specs = [real_order, remote]
    assert all(s.permission == "read" for s in specs)
    assert business.owns_order(data["owner"], case["order_id"])
    assert not business.owns_order("customer_other", case["order_id"])
    return specs, order_transport, logistics_transport


async def probe_case(case_id, *, checkpoint_path=None):
    from langchain_core.messages import HumanMessage
    from app.config import settings
    from app.core.execution import utc_now
    from app.graph import logistics, task_context
    from app.tools import engine, registry
    data, digest = load_fixture()
    case = next(c for c in data["cases"] if c["id"] == case_id)
    cid, uid = 71001 + data["cases"].index(case), data["owner"]
    message = f"订单{case['order_id']}物流卡住了，查询失败时不要猜结果。"
    with readonly_world(data, digest), ExitStack() as patches:
        specs, order_transport, logistics_transport = transports(data, case)
        patches.enter_context(patch.object(registry, "get_all_specs", AsyncMock(return_value=specs)))
        patches.enter_context(patch.object(engine, "_audit", AsyncMock()))
        # If a failure unexpectedly continues to summary generation it fails closed.
        patches.enter_context(patch.object(logistics, "_draft_summary", AsyncMock(side_effect=AssertionError("unexpected model summary"))))
        ctx = task_context.begin_turn(None, uid, cid, message,
            {"request_id": "offline-" + case_id, "task_id": "offline-task-" + case_id, "now": utc_now()})
        if case["fault"] == "tool_budget":
            ctx["budgets"]["max_tool_calls"] = 1
        state = {"user_id": uid, "conversation_id": cid, "task_context": ctx,
                 "messages": [HumanMessage(message)], "logistics_order_id": case["order_id"],
                 "logistics_needs_policy": False, "logistics_claim_type": "stalled", "logistics_claim": message}
        http_status, pending, checkpoint_scope, cancelled = None, None, None, None
        if case["fault"] == "deadline":
            if checkpoint_path is None:
                raise ValueError("deadline adapter requires private checkpoint path")
            from langgraph.graph import StateGraph, START, END
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            from app.graph import runtime
            from app.graph.state import ConversationState
            from app.db import repository
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            import httpx
            from app.api.agent import router
            from app.core.auth import Principal, require_principal

            async def investigate(value):
                return await logistics.investigate_logistics({**value, "logistics_order_id": case["order_id"],
                    "logistics_claim_type": "stalled", "logistics_needs_policy": False})
            builder = StateGraph(ConversationState)
            builder.add_node("slow_read", investigate)
            builder.add_node("log", lambda value: {})
            builder.add_edge(START, "slow_read"); builder.add_edge("slow_read", "log"); builder.add_edge("log", END)
            conv = SimpleNamespace(user_id=uid, summary="", summary_upto_msg_id=0, layer1_from_msg_id=0)
            patches.enter_context(patch.object(repository, "get_conversation", AsyncMock(return_value=conv)))
            saved_messages = []
            async def append_message(cid, role, **kwargs):
                saved_messages.append({"conversation_id": cid, "role": role, **kwargs})
                return len(saved_messages)
            patches.enter_context(patch.object(repository, "append_message", append_message))
            patches.enter_context(patch.object(settings, "turn_timeout_seconds", .08))
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                await saver.setup()
                graph = builder.compile(checkpointer=saver)
                patches.enter_context(patch.object(runtime, "_graph", graph))
                app = FastAPI(); app.include_router(router)
                app.dependency_overrides[require_principal] = lambda: Principal(user_id=uid, role="customer")
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
                    response = await client.post("/api/agent", json={"message": message, "conversation_id": cid})
                http_status = response.status_code
                snapshot = await graph.aget_state({"configurable": {"thread_id": str(cid)}})
                out = dict(snapshot.values)
                pending = list(snapshot.next)
                checkpoint_scope = {k: out.get(k) for k in ("user_id", "conversation_id")}
                cancelled = order_transport.cancelled
        else:
            out = await logistics.investigate_logistics(state)
        context = out["task_context"]
        facts = task_context.current_facts(context)
        evidence = out.get("logistics_decision", {}).get("evidence", [])
        expected_termination = "deadline_exceeded" if case["fault"] == "deadline" else "tool_budget_exhausted" if case["fault"] == "tool_budget" else "tool_evidence_unavailable"
        checks = {"termination": context.get("termination_reason") == expected_termination,
                  "no_logistics_facts": not any(k.startswith("logistics_") for k in facts),
                  "no_logistics_evidence": not any(e.get("source") == "logistics" for e in evidence),
                  "no_write_or_ticket_draft": not any(a.get("type") == "create_ticket" for a in out.get("suggested_actions", [])),
                  "scope_identity": context.get("user_id") == uid and context.get("conversation_id") == cid,
                  "only_local_read_calls": len(order_transport.calls) == 1 and len(logistics_transport.calls) <= 1}
        if case["fault"] == "deadline":
            checks.update(actual_504=http_status == 504, no_pending_nodes=pending == [], tool_cancelled=cancelled is True,
                          checkpoint_scope=checkpoint_scope == {"user_id": uid, "conversation_id": cid})
        else:
            checks["verified_order_retained"] = facts.get("order_id", {}).get("value") == case["order_id"]
        if case["fault"] == "tool_budget":
            checks["second_call_not_dispatched"] = not logistics_transport.calls
        if case["fault"] != "deadline":
            trace = out.get("trace", {}).get("logistics_tool_trace", [])
            expected_error = ("tool_budget_exhausted" if case["fault"] == "tool_budget" else
                              "execution_error" if case["fault"] == "partial_evidence" else "invalid_tool_result")
            checks["intended_failure_reached"] = bool(trace) and trace[-1].get("error_code") == expected_error
            checks["expected_physical_dispatches"] = len(logistics_transport.calls) == (
                0 if case["fault"] == "tool_budget" else 1)
        return {"id": case_id, "adapter_version": VERSION, "fixture_sha256": digest,
                "measurement": "offline real contracts/guards/investigator/runtime with local read transport; no model E2E",
                "checks": checks, "success": all(checks.values()), "observed": out,
                "physical_read_calls": {"query_order": len(order_transport.calls), "query_logistics": len(logistics_transport.calls)},
                "http_status": http_status, "pending_nodes": pending, "tool_cancelled": cancelled,
                "model_calls": 0, "external_requests": 0}
