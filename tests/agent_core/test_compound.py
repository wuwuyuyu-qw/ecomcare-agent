"""Local synthetic graph/evidence checks; no provider, MCP or database calls."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.graph import answer_guard, compound, nodes, routing, task_context as tasks
from app.graph.build import build_graph
from app.graph.runtime import _graph_input
from app.config import settings
from scripts.eval_compound_agent import coverage_checks


def context():
    return tasks.begin_turn(None, "customer_demo", 1, "订单3003物流和保修查一下", {
        "now": "2026-10-05T00:00:00+00:00", "task_id": "compound-local-task",
        "request_id": "compound-local-request", "max_steps": 6, "max_tool_calls": 8})


def state(checks=("logistics", "policy", "warranty")):
    return {"compound_plan": {"protocol": compound.PROTOCOL, "checks": list(checks)},
            "task_context": context(), "messages": [HumanMessage("订单3003物流和保修查一下")],
            "user_id": "customer_demo", "conversation_id": 1, "route": "compound", "intent": "物流"}


def order(ctx):
    return tasks.record_tool_result(ctx, "query_order", {"order_id": "3003"},
        {"order_id": "3003", "tracking_no": "SFDEMO3003", "status": "已发货"}, "ok")


def logistics(ctx):
    return tasks.record_tool_result(ctx, "query_logistics", {"tracking_no": "SFDEMO3003"},
        {"tracking_no": "SFDEMO3003", "status": "运输中", "stalled_hours": 6}, "ok")


def test_upstream_router_and_opt_in_preserve_default(monkeypatch):
    monkeypatch.setattr(settings, "compound_agent_enabled", False)
    assert routing.route_by_intent({"intent": "物流"}) == "logistics"
    monkeypatch.setattr(settings, "compound_agent_enabled", True)
    assert routing.route_by_intent({"intent": "物流"}) == "compound"
    assert routing.route_by_intent({"intent": "商品咨询"}) == "knowledge"
    assert routing.route_by_intent({"intent": "投诉"}) == "escalate"
    s = state()
    s["messages"] = [AIMessage(content="", tool_calls=[{"id": "t", "name": "query_order", "args": {}}])]
    assert routing.should_continue(s) == "continue"
    s["messages"] = [AIMessage(content="核查完毕")]
    assert routing.should_continue(s) == "compound_stop"


def test_frontier_exposes_dependency_ready_tools_only():
    s = state()
    assert compound.available_tools(s, s["task_context"]) == {"query_order"}
    ctx = order(s["task_context"])
    assert compound.available_tools(s, ctx) == {"query_logistics", "query_warranty"}
    ctx = logistics(ctx)
    assert compound.available_tools(s, ctx) == {"query_faq", "query_warranty"}
    assert compound.guard_call(s, ctx, "create_ticket") == "compound_tool_not_ready"
    assert compound.guard_call(s, ctx, "query_product") == "compound_tool_not_ready"


def test_failed_lookup_is_unavailable_and_not_retried_by_model():
    s = state(("logistics", "warranty"))
    ctx = tasks.record_tool_result(order(s["task_context"]), "query_logistics", {}, None, "error")
    assert compound.statuses(s, ctx)["logistics"] == "unavailable"
    assert compound.available_tools(s, ctx) == {"query_warranty"}


def test_failed_ownership_and_cancellation_stop_all_tools():
    s = state()
    denied = tasks.record_tool_result(s["task_context"], "query_order", {"order_id": "3003"}, None, "forbidden")
    assert compound.available_tools(s, denied) == set()
    assert compound.available_tools(s, tasks.finish(s["task_context"], "cancelled")) == set()


def test_switch_and_ttl_invalidate_frontier():
    s = state()
    ctx = logistics(order(s["task_context"]))
    ctx["updated_at"] = "2026-10-05T00:16:00+00:00"
    assert compound.available_tools(s, ctx) == {"query_order"}
    changed = tasks.begin_turn(order(s["task_context"]), "customer_demo", 1, "更正为订单3001", {
        "now": "2026-10-05T00:01:00+00:00"})
    assert compound.available_tools(s, changed) == {"query_order"}


@pytest.mark.parametrize("checks", [[], ["logistics", "logistics"], ["create_ticket"], ["product"]])
def test_plan_rejects_invalid_or_duplicate_checks(checks):
    with pytest.raises(ValidationError):
        compound.CheckPlan(checks=checks)


@pytest.mark.asyncio
async def test_prepare_uses_structured_model_checks_and_counts_step(monkeypatch):
    s = state()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value={"checks": ["logistics", "warranty"]}))
    monkeypatch.setattr(compound.llm, "structured", lambda *a, **k: model)
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: deepcopy(s["task_context"]))
    result = await compound.prepare(s)
    assert result["compound_plan"]["checks"] == ["logistics", "warranty"]
    assert result["steps"] == 1 and result["task_context"]["budgets"]["steps"] == 1


@pytest.mark.asyncio
async def test_missing_order_does_not_call_model(monkeypatch):
    s = state()
    s["task_context"]["active_order_id"] = ""
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: s["task_context"])
    monkeypatch.setattr(compound.llm, "structured", lambda *a, **k: pytest.fail("unexpected model call"))
    result = await compound.prepare(s)
    assert result["task_context"]["decision"] == "clarify"
    assert compound.route_after_prepare(result) == "done"


@pytest.mark.asyncio
async def test_failed_plan_cannot_grant_tools_and_cancellation_propagates(monkeypatch):
    s = state()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=None))
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: deepcopy(s["task_context"]))
    monkeypatch.setattr(compound.llm, "structured", lambda *a, **k: model)
    result = await compound.prepare(s)
    assert not result["compound_plan"] and result["task_context"]["termination_reason"] == "execution_failed"
    model.ainvoke.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await compound.prepare(s)


@pytest.mark.asyncio
async def test_premature_final_answer_is_replaced_in_state_and_history(monkeypatch):
    s = state()
    s["task_context"] = order(s["task_context"])
    s["messages"].append(AIMessage(content="都已经核查完成", id="answer-1"))
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: s["task_context"])
    result = await compound.finalize(s)
    assert "尚不完整" in result["answer"]
    assert result["messages"][0].id == "answer-1"
    assert result["task_context"]["decision"] == "human_review"
    assert result["compound_completion"]["checks"]["order"] == "observed"


def test_runtime_resets_plan_and_completion_and_graph_compiles():
    fresh = _graph_input("customer_demo", "订单3003", 1, 1, "", 0)
    assert fresh["compound_plan"] == fresh["compound_completion"] == {}
    graph = build_graph()
    assert "compound_prepare" in graph.nodes and "compound_finalize" in graph.nodes


def test_coverage_does_not_count_blocked_tool_as_executed():
    expected = {"required_tools": ["query_logistics"], "required_facts": ["logistics_status"]}
    observed = {"facts": {}}
    response = {"execution_summary": {"tool_events": [{"tool": "query_logistics", "ok": False,
                                                      "error_code": "compound_tool_not_ready"}]}}
    checks = coverage_checks(expected, observed, response)
    assert checks["requested_tools_observed"] is False and checks["required_facts_observed"] is False
    response["execution_summary"]["tool_events"][0]["error_code"] = "timeout"
    assert coverage_checks(expected, observed, response)["requested_tools_observed"] is True
    assert coverage_checks(expected, observed, response)["required_facts_observed"] is False


def test_dataset_loader_reads_development_only(monkeypatch):
    from pathlib import Path
    from app.evaluation.compound_cases import load_cases
    original = Path.read_bytes

    def read(path):
        assert path.name != "holdout.jsonl"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    cases, manifest = load_cases(Path(__file__).resolve().parents[2] / "data/compound_eval", "dev")
    assert len(cases) == 8 and manifest["files"]["holdout.jsonl"]["count"] == 4


def test_coordinator_output_argument_matches_runner(monkeypatch, tmp_path):
    import sys
    from scripts import eval_compound_agent as runner
    from scripts.run_agent_diagnostics import runner_command
    args = SimpleNamespace(mode="compound-agent", api_port=18000)
    command, output = runner_command(args, tmp_path, "synthetic-version", ["compound-dev-01"])
    assert "--output" in command
    monkeypatch.setattr(sys, "argv", command[1:])
    monkeypatch.setattr(runner, "run", lambda parsed: {"mode": "synthetic-local-cli-check", "out": str(parsed.out)})
    runner.main()
    assert output.exists()


@pytest.mark.parametrize("message", ["取消这次查询，不要再查，也不要提交申请。", "请停止本次核查", "不用再查了"])
def test_explicit_query_cancellation_clears_facts(message):
    ctx = tasks.begin_turn(order(context()), "customer_demo", 1, message,
                          {"now": "2026-10-05T00:01:00+00:00"})
    assert ctx["cancelled"] is True and not ctx["verified_facts"] and not ctx["active_order_id"]


@pytest.mark.parametrize("message", ["不要取消查询，请继续", "  不要取消查询，请继续  ", "怎么取消查询？", "如果取消查询会怎样？",
                                     "取消订单需要什么条件？", "查询取消订单的政策", "我提到‘取消查询’是在举例"])
def test_policy_negation_hypothetical_and_quote_do_not_cancel(message):
    assert tasks._cancel_requested(message) is False


@pytest.mark.asyncio
async def test_pending_checks_require_tool_choice_not_premature_final(monkeypatch):
    s = state(("logistics", "warranty"))
    seen = {}

    class LocalModel:
        def bind_tools(self, tools, **kwargs):
            seen.update(kwargs)
            return self

        async def ainvoke(self, *args):
            return AIMessage(content="", tool_calls=[{"id": "t", "name": "query_order", "args": {"order_id": "3003"}}])

    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: s["task_context"])
    monkeypatch.setattr(nodes, "_agent_messages", lambda _s: [HumanMessage("synthetic-local-input")])
    def local_model(**kwargs):
        seen["model_settings"] = kwargs
        return LocalModel()

    monkeypatch.setattr(nodes, "get_chat_model", local_model)
    monkeypatch.setattr(nodes.registry, "get_all_specs", AsyncMock(return_value=[
        SimpleNamespace(name="query_order", permission="read", tool="local-stub")]))
    result = await nodes.main_agent(s)
    assert seen["tool_choice"] == "required" and result["messages"][0].tool_calls
    assert seen["model_settings"]["thinking"] is False
    assert result["messages"][0].response_metadata["mewhelp_tool_thinking_disabled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("compound_final", [False, True])
async def test_default_route_and_compound_final_use_consistent_thinking(monkeypatch, compound_final):
    s = state(()) if compound_final else {"messages": [], "steps": 0}
    seen = {}

    class LocalModel:
        def bind_tools(self, *args, **kwargs):
            return self

        async def ainvoke(self, *args):
            return AIMessage(content="synthetic local final")

    def local_model(**kwargs):
        seen.update(kwargs)
        return LocalModel()

    monkeypatch.setattr(nodes, "_agent_messages", lambda _s: [HumanMessage("synthetic-local-input")])
    monkeypatch.setattr(nodes, "get_chat_model", local_model)
    monkeypatch.setattr(nodes.registry, "get_all_specs", AsyncMock(return_value=[]))
    await nodes.main_agent(s)
    assert seen["thinking"] is (not compound_final)


def test_mode_is_preserved_only_while_actual_window_contains_our_tool_exchange():
    marked = AIMessage(content="", tool_calls=[{"id": "t", "name": "query_order", "args": {}}],
                       response_metadata={"mewhelp_tool_thinking_disabled": True})
    unmarked = AIMessage(content="", tool_calls=[{"id": "u", "name": "query_order", "args": {}}])
    assert nodes._agent_thinking_enabled({}, [marked]) is False
    assert nodes._agent_thinking_enabled({}, [unmarked]) is True
    assert nodes._agent_thinking_enabled({}, [HumanMessage("after old tools trimmed")]) is True
    assert nodes._agent_thinking_enabled(state(), []) is False


def test_layer2_and_checkpoint_roundtrip_preserve_only_our_transport_marker():
    from app.core import memory
    from langchain_core.messages import message_to_dict, messages_from_dict
    marked = AIMessage(content="synthetic earlier answer", id="prior-ai", tool_calls=[
        {"id": "t", "name": "query_order", "args": {"order_id": "3003"}}],
        response_metadata={"mewhelp_tool_thinking_disabled": True, "unused_debug": "drop"})
    compact = memory.to_layer2([marked])[0]
    assert compact.response_metadata == {"mewhelp_tool_thinking_disabled": True}
    assert marked.response_metadata["unused_debug"] == "drop"
    restored = messages_from_dict([message_to_dict(compact)])[0]
    assert restored.id == marked.id and restored.tool_calls == marked.tool_calls
    assert nodes._agent_thinking_enabled({}, [restored]) is False


def policy_state():
    s = state(("logistics", "warranty", "policy"))
    ctx = tasks.record_tool_result(logistics(order(s["task_context"])), "query_warranty",
        {"order_id": "3003"}, {"order_id": "3003", "warranty": "在保", "warranty_until": "2026-11-01"}, "ok")
    return {**s, "task_context": ctx, "policy_application_required": True,
            "policy_assessment": {"outcome": "unknown", "reason": "missing_approval"}}


def test_policy_unknown_keeps_only_independent_requested_verified_observations():
    s = policy_state()
    result = answer_guard.guard_answer(s, "订单3003商品在保。可以获得900元赔偿。")
    assert result["reason"] == "policy_scope_unverified" and result["allowed"] is False
    assert "物流状态：运输中" in result["answer"] and "在保状态：在保" in result["answer"]
    assert "2026-11-01" in result["answer"] and "900" not in result["answer"]
    assert len(result["evidence_ids"]) >= 3


def test_policy_unknown_fact_renderer_still_rejects_wrong_scope_stale_and_unrequested_data():
    s = policy_state()
    s["compound_plan"]["checks"] = ["logistics", "policy"]
    text, _ = answer_guard._compound_observed_text(s, s["task_context"])
    assert "在保状态" not in text
    s["compound_plan"]["checks"] = ["warranty", "policy"]
    for name in ("warranty", "warranty_until"):
        s["task_context"]["verified_facts"][name]["scope"]["active_order_id"] = "3001"
    text, _ = answer_guard._compound_observed_text(s, s["task_context"])
    assert not text
    s = policy_state()
    s["task_context"]["updated_at"] = "2026-10-05T00:20:00+00:00"
    assert answer_guard._compound_observed_text(s, s["task_context"]) == ("", [])
    s = policy_state()
    s["task_context"]["verified_facts"].pop("order_owned")
    assert answer_guard._compound_observed_text(s, s["task_context"]) == ("", [])


def test_policy_fallback_never_preserves_model_write_claim_or_changes_default_behavior():
    s = policy_state()
    assert "在保状态" not in answer_guard.guard_answer(s, "已创建工单")["answer"]
    s["route"] = "business"
    result = answer_guard.guard_answer(s, "订单3003商品在保")
    assert "在保状态" not in result["answer"] and result["reason"] == "policy_scope_unverified"


@pytest.mark.asyncio
async def test_executor_rechecks_frontier_and_never_calls_disallowed_tool(monkeypatch):
    s = state()
    s["messages"] = [AIMessage(content="", tool_calls=[{"id": "t", "name": "query_logistics",
                                                       "args": {"tracking_no": "SFDEMO3003"}}])]
    monkeypatch.setattr(nodes, "_task_for_state", lambda _s: s["task_context"])
    monkeypatch.setattr(nodes.registry, "get_all_specs", AsyncMock(return_value=[]))
    monkeypatch.setattr(nodes.engine, "execute_tool_call", AsyncMock(side_effect=AssertionError("cannot execute")))
    result = await nodes.agent_tools(s)
    assert "compound_tool_not_ready" in result["messages"][0].content
    nodes.engine.execute_tool_call.assert_not_awaited()
