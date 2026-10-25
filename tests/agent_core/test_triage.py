"""Joint request triage contracts; synthetic stubs and no external services."""
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.config import settings
from app.graph import compound, nodes, routing, task_context, triage
from app.graph.runtime import _graph_input
from app.graph.answer_guard import guard_answer


def payload(mode="lookup", checks=None, intent="其他"):
    return {"protocol": triage.PROTOCOL, "status": "parsed", "intent": intent,
            "confidence": 0.8, "mode": mode,
            "checks": ["logistics", "warranty"] if checks is None else checks}


def state(value=None):
    text = "订单3003物流和在保查一下"
    ctx = task_context.begin_turn(None, "synthetic_customer", 7, text, {
        "now": "2026-10-05T00:00:00Z", "task_id": "triage-task", "request_id": "triage-request"})
    return {"request_triage": value or payload(), "task_context": ctx,
            "messages": [HumanMessage(text)], "route": "compound", "steps": 0,
            "user_id": "synthetic_customer", "conversation_id": 7}


@pytest.mark.parametrize("mode,checks,intent", [
    ("lookup", [], "物流"), ("lookup", ["policy"], "其他"),
    ("lookup", ["logistics", "logistics"], "物流"),
    ("knowledge", ["warranty"], "售后"), ("knowledge", ["create_ticket"], "其他"),
    ("conversation", [], "订单"), ("handoff", [], "订单"),
])
def test_schema_rejects_inconsistent_or_unrecognized_authority(mode, checks, intent):
    with pytest.raises(ValidationError):
        triage.validated(payload(mode, checks, intent))


def test_joint_lookup_does_not_depend_on_single_intent(monkeypatch):
    monkeypatch.setattr(settings, "compound_agent_enabled", True)
    monkeypatch.setattr(settings, "compound_joint_triage_enabled", True)
    s = {"intent": "其他", "request_triage": payload()}
    assert routing.route_by_intent(s) == "compound"
    monkeypatch.setattr(settings, "compound_joint_triage_enabled", False)
    assert routing.route_by_intent(s) == "fallback_script"
    monkeypatch.setattr(settings, "compound_joint_triage_enabled", True)
    monkeypatch.setattr(settings, "compound_agent_enabled", False)
    assert routing.route_by_intent(s) == "fallback_script"


@pytest.mark.parametrize("mode,intent,route,allowed", [
    ("knowledge", "商品咨询", "knowledge", {"query_faq", "query_product"}),
    ("conversation", "其他", "fallback_script", set()),
    ("handoff", "人工", "escalate", set()),
])
def test_non_lookup_modes_never_allow_owned_order_reads(mode, intent, route, allowed):
    s = {"request_triage": payload(mode, [], intent)}
    assert triage.route(s) == route and triage.allowed_tools(s) == allowed
    assert triage.guard_call(s, "query_order") == "order_lookup_not_requested"
    assert triage.guard_call({}, "query_order") is None


def test_invalid_and_unavailable_payloads_fail_closed():
    for value in ({"protocol": triage.PROTOCOL}, payload("knowledge", ["order"]),
                  {**payload(), "status": "unavailable"}):
        s = {"request_triage": value}
        assert triage.route(s) == "fallback_script"
        assert triage.guard_call(s, "query_order") == "order_lookup_not_requested"


@pytest.mark.asyncio
async def test_classification_receives_original_words_and_reuses_validated_checks(monkeypatch):
    monkeypatch.setattr(settings, "compound_agent_enabled", True)
    monkeypatch.setattr(settings, "compound_joint_triage_enabled", True)
    s = state()
    s["resolved_query"] = "rewritten query must not authorize new checks"
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=triage.validated(payload())))
    monkeypatch.setattr(triage.llm, "structured", lambda *a, **k: model)
    result = await nodes.classify_intent(s)
    assert s["messages"][0].content in model.ainvoke.call_args.args[0][-1].content
    assert s["resolved_query"] not in model.ainvoke.call_args.args[0][-1].content
    assert result["route"] == "compound" and result["intent"] == "其他"
    s.update(result)
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: deepcopy(s["task_context"]))
    monkeypatch.setattr(compound.llm, "structured", lambda *a, **k: pytest.fail("second planning call"))
    prepared = await compound.prepare(s)
    assert prepared["compound_plan"]["checks"] == ["logistics", "warranty"]
    assert prepared["trace"]["compound_prepare"] == "joint_triage_reused"
    assert prepared["steps"] == prepared["task_context"]["budgets"]["steps"] == 1


@pytest.mark.asyncio
async def test_provider_failure_no_permissions_and_cancel_propagates(monkeypatch):
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("synthetic parser failure")))
    monkeypatch.setattr(triage.llm, "structured", lambda *a, **k: model)
    result = await triage.classify("只讲如何查询，不要读取订单3003")
    assert result["status"] == "unavailable" and result["checks"] == []
    assert "synthetic parser failure" not in str(result)
    assert triage.allowed_tools({"request_triage": result}) == set()
    model.ainvoke.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await triage.classify("查订单3003")


@pytest.mark.asyncio
async def test_execution_rechecks_non_lookup_restriction(monkeypatch):
    s = state(payload("knowledge", [], "其他"))
    s["compound_plan"] = {}
    s["messages"] = [AIMessage(content="", tool_calls=[{
        "id": "t", "name": "query_order", "args": {"order_id": "3003"}}])]
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: s["task_context"])
    monkeypatch.setattr(nodes.registry, "get_all_specs", AsyncMock(return_value=[]))
    monkeypatch.setattr(nodes.engine, "execute_tool_call", AsyncMock(side_effect=AssertionError("cannot execute")))
    result = await nodes.agent_tools(s)
    assert "order_lookup_not_requested" in result["messages"][0].content
    nodes.engine.execute_tool_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_tool_binding_is_filtered_as_well_as_executor(monkeypatch):
    s = state(payload("knowledge", [], "其他"))
    s["route"] = "knowledge"
    seen = []
    class LocalModel:
        def bind_tools(self, tools, **kwargs):
            seen.extend(tools)
            return self
        async def ainvoke(self, *args):
            return AIMessage(content="synthetic knowledge explanation")
    monkeypatch.setattr(nodes, "get_chat_model", lambda **k: LocalModel())
    monkeypatch.setattr(nodes, "_agent_messages", lambda _s: [HumanMessage("synthetic local")])
    monkeypatch.setattr(nodes.registry, "get_all_specs", AsyncMock(return_value=[
        SimpleNamespace(name=name, permission="read", tool=name)
        for name in ("query_order", "query_logistics", "query_product", "query_faq")]))
    await nodes.main_agent(s)
    assert set(seen) == {"query_product", "query_faq"}


@pytest.mark.parametrize("message", [
    "订单3003不用查在保状态，只查询物流轨迹。", "不用查询保修，只核对物流。",
    "停止查询物流，仅查询在保。", "取消本次核查保修，保留物流查询。",
])
def test_negated_named_subtask_does_not_cancel_entire_request(message):
    assert task_context._cancel_requested(message) is False


def test_fresh_input_resets_triage_and_dev_loader_is_explicit():
    from app.evaluation.compound_cases import load_cases
    fresh = _graph_input("synthetic_customer", "新请求", 7, 1, "", 0)
    assert fresh["request_triage"] == {}
    directory = Path(__file__).resolve().parents[2] / "data/compound_routing_eval"
    cases, manifest = load_cases(directory, expected_version="compound-routing-v1.0")
    assert len(cases) == 12 and sum(len(c["turns"]) for c in cases) == 14
    assert manifest["files"]["dev.jsonl"]["sha256"]
    with pytest.raises(ValueError):
        load_cases(directory)


def test_negated_logistics_keyword_does_not_hide_verified_warranty_answer():
    s = state(payload(checks=["warranty"], intent="物流"))
    s["intent"] = "物流"
    s["compound_plan"] = {"protocol": compound.PROTOCOL, "checks": ["warranty"]}
    ctx = task_context.record_tool_result(s["task_context"], "query_order", {"order_id": "3003"},
        {"order_id": "3003", "status": "已发货", "tracking_no": "SFDEMO3003"}, "ok")
    s["task_context"] = task_context.record_tool_result(ctx, "query_warranty", {"order_id": "3003"},
        {"order_id": "3003", "warranty": "在保", "warranty_until": "2027-09-25"}, "ok")
    result = guard_answer(s, "查询记录显示订单3003在保，截止日期为2027-09-25。")
    assert result["allowed"] is True and "在保" in result["answer"]
    warranty_id = s["task_context"]["verified_facts"]["warranty"]["evidence_id"]
    assert warranty_id in result["evidence_ids"]
    missing = deepcopy(s)
    missing["task_context"]["verified_facts"].pop("warranty")
    assert guard_answer(missing, "查询记录显示在保")["reason"] == "requested_check_evidence_required"
    missing["task_context"]["verified_facts"].pop("warranty_until")
    assert "查询记录显示在保" not in guard_answer(missing, "查询记录显示在保")["answer"]
    s["compound_plan"]["checks"] = ["warranty", "logistics"]
    assert guard_answer(s, "所有项目已核验")["reason"] == "current_logistics_evidence_required"


@pytest.mark.asyncio
async def test_explicit_handoff_discloses_unavailable_channel_without_requesting_order():
    result = await nodes.complaint_reply(state(payload("handoff", [], "人工")))
    assert "无法直接接通" in result["answer"] and "没有查询订单" in result["answer"]
    assert "请补充" not in result["answer"] and "已转人工" not in result["answer"]
    assert result["trace"]["handoff"] == "unavailable_no_backend_action"


@pytest.mark.parametrize("mode,strategy", [("compound-routing-a", "frontier"),
                                          ("compound-routing-b", "joint-frontier")])
def test_coordinator_routes_same_development_dataset(mode, strategy, tmp_path):
    from scripts.run_agent_diagnostics import runner_command
    command, _ = runner_command(SimpleNamespace(mode=mode, api_port=18000), tmp_path, "local", ["routing-dev-01"])
    assert command[command.index("--strategy") + 1] == strategy
    assert command[command.index("--dataset") + 1] == "routing"
