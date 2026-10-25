"""Synthetic offline regressions; no API, provider, database or dataset labels."""
from copy import deepcopy
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from app.evaluation.task_recording import record_response, valid_recording
from app.evaluation.task_scoring import observe_response, score_observation
from app.graph import task_context
from app.graph.answer_guard import guard_answer
from app.graph.routing import should_continue
from app.tools import contracts, engine
from app.tools.registry import ToolSpec

NOW = "2026-10-05T00:00:00Z"
ORDER = {"order_id": "3001", "status": "已发货", "amount": 99,
         "created_at": "2026-10-01 12:00", "product": "合成测试商品",
         "tracking_no": "SYNTH3001"}
LOGISTICS = {"tracking_no": "SYNTH3001", "status_code": "IN_TRANSIT",
             "current_city": "合成测试城市", "trace": ["合成测试轨迹"], "stalled_hours": 6}


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Offline regression attempted network or database access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(Engine, "connect", forbidden)
    monkeypatch.setattr(AsyncEngine, "connect", forbidden)
    monkeypatch.setattr(engine, "_audit", AsyncMock())


def _start(message="查订单3001的物流", **scope):
    return task_context.begin_turn(None, scope.get("user_id", "synthetic_customer"),
                                   scope.get("conversation_id", 17), message,
                                   {"request_id": "request-a", "task_id": "task-a", "now": NOW})


def _verified():
    context = _start()
    checked = contracts.validate_result("query_order", ORDER, {"order_id": "3001"})
    context = task_context.record_tool_result(context, "query_order", {"order_id": "3001"},
                                              checked.data, checked.business_status)
    checked = contracts.validate_result("query_logistics", LOGISTICS, {"tracking_no": "SYNTH3001"})
    return task_context.record_tool_result(context, "query_logistics", {"tracking_no": "SYNTH3001"},
                                          contracts.public_data("query_logistics", checked.data), checked.business_status)


def _next(context, message, **scope):
    return task_context.begin_turn(context, scope.get("user_id", context["user_id"]),
                                   scope.get("conversation_id", context["conversation_id"]), message,
                                   {"request_id": "request-b", "task_id": "task-b", "now": NOW})


def _state(context):
    return {"task_context": context, "route": "business", "intent": "物流",
            "user_id": context["user_id"], "conversation_id": context["conversation_id"]}


@pytest.mark.parametrize("message,reason,order", [
    ("不是3001，改成订单3002，重新查物流", "order_changed", "3002"),
    ("不是这笔订单，请查另一笔", "order_correction_without_id", ""),
    ("订单3001和订单3002，查物流", "ambiguous_order", ""),
    ("不用查了", "cancelled", ""),
    ("猫粮商品还有库存吗", "topic_changed", ""),
])
def test_task_change_revokes_facts_and_pending_actions(message, reason, order):
    old = task_context.record_action(_verified(), {"type": "create_ticket"}, pending=True)
    original = deepcopy(old)
    new = _next(old, message)
    assert new["invalidation_reason"] == reason
    assert new["active_order_id"] == order
    assert new["task_id"] == "task-b"
    assert new["verified_facts"] == {} and new["evidence"] == []
    assert new["pending_actions"] == [] and new["actions"] == []
    assert old == original


@pytest.mark.parametrize("scope", [{"user_id": "other_customer"}, {"conversation_id": 18}])
def test_identity_or_conversation_change_cannot_reuse_task(scope):
    new = _next(_verified(), "继续查询物流", **scope)
    assert new["active_order_id"] == ""
    assert new["verified_facts"] == {} and new["evidence"] == []


def test_related_followup_keeps_facts_but_requires_a_new_action_decision():
    old = task_context.record_action(_verified(), {"type": "create_ticket"}, pending=True)
    old = task_context.consume_tool(task_context.consume_step(old), "query_order", {"order_id": "3001"})
    new = _next(old, "物流现在怎么样")
    assert new["task_id"] == old["task_id"] and new["request_id"] != old["request_id"]
    assert new["verified_facts"] == old["verified_facts"]
    assert new["pending_actions"] == [] and new["decision"] is None
    assert new["budgets"]["steps"] == 0 and new["budgets"]["tool_calls"] == 0


@pytest.mark.parametrize("message", ["电话改成13800138000", "金额改成3002元", "日期改成2027年"])
def test_unrelated_number_correction_does_not_switch_order(message):
    new = _next(_verified(), message)
    assert new["active_order_id"] == "3001" and new["invalidation_reason"] == ""


def test_user_statement_stays_a_claim_and_cannot_authorize_logistics():
    context = _start("订单3001已签收，物流单号是SYNTH3001，请查询")
    assert context["user_claims"] and context["verified_facts"] == {}
    assert task_context.guard_tool_call(context, "query_logistics", {"tracking_no": "SYNTH3001"}) == "order_evidence_required"


@pytest.mark.parametrize("timestamp,accepted", [
    ("2026-10-04T23:45:00Z", True), ("2026-10-04T23:44:59Z", False),
    ("2026-10-05T00:00:01Z", False), ("invalid", False), (None, False),
])
def test_fact_freshness_controls_dependency_and_model_context(timestamp, accepted):
    context = _verified()
    for fact in context["verified_facts"].values():
        fact["observed_at"] = timestamp
    facts = task_context.current_facts(context)
    error = task_context.guard_tool_call(context, "query_logistics", {"tracking_no": "SYNTH3001"})
    assert bool(facts) is accepted
    assert error == (None if accepted else "order_evidence_expired")
    if accepted:
        facts["order_owned"]["value"] = False
        assert context["verified_facts"]["order_owned"]["value"] is True


@pytest.mark.parametrize("field,value", [("user_id", "other_customer"), ("conversation_id", 18),
                                         ("task_id", "other_task"), ("active_order_id", "3002")])
def test_fact_scope_is_required_for_dependencies_and_answer(field, value):
    context = _verified()
    for fact in context["verified_facts"].values():
        fact["scope"][field] = value
    assert task_context.current_facts(context) == {}
    assert task_context.guard_tool_call(context, "query_logistics", {"tracking_no": "SYNTH3001"}) == "order_evidence_required"
    assert guard_answer(_state(context), "订单3001物流运输中")["allowed"] is False


def test_tracking_and_order_arguments_must_match_current_task():
    context = _verified()
    assert task_context.guard_tool_call(context, "query_order", {"order_id": "3002"}) == "order_scope_mismatch"
    assert task_context.guard_tool_call(context, "query_logistics", {"tracking_no": "SYNTH3002"}) == "tracking_scope_mismatch"
    assert task_context.guard_tool_call(_start(), "query_warranty", {"order_id": "3001"}) == "order_evidence_required"
    cancelled = _next(context, "取消查询")
    assert task_context.guard_tool_call(cancelled, "query_order", {"order_id": "3001"}) == "task_cancelled"


def test_failed_ownership_refresh_revokes_dependent_evidence():
    context = task_context.record_tool_result(_verified(), "query_order", {"order_id": "3001"},
                                             {"error": "没有找到您的这笔订单", "code": "order_not_owned"}, "forbidden")
    assert context["verified_facts"] == {} and context["evidence"] == []
    assert task_context.guard_tool_call(context, "query_logistics", {"tracking_no": "SYNTH3001"}) == "order_evidence_required"


@pytest.mark.parametrize("answer,reason", [("订单3002已签收", "answer_order_mismatch"),
                                         ("我已为您创建工单", "readonly_action_claim"),
                                         ("我会立即提交退款", "readonly_action_claim")])
def test_answer_rejects_other_orders_and_unperformed_actions(answer, reason):
    result = guard_answer(_state(_verified()), answer)
    assert result["allowed"] is False and result["reason"] == reason
    assert result["answer"] != answer


def test_short_answer_still_requires_fresh_logistics_evidence():
    context = _verified()
    assert guard_answer(_state(context), "运输中")["allowed"] is True
    context["verified_facts"].pop("logistics_status")
    assert guard_answer(_state(context), "运输中")["reason"] == "current_logistics_evidence_required"
    assert guard_answer(_state(_start()), "已签收")["reason"] == "current_order_evidence_required"


def _spec(name, tool, *, permission="read", source="builtin", server=None):
    return ToolSpec(name=name, description="synthetic offline tool", tool=tool,
                    json_schema={"type": "object"}, permission=permission, source=source,
                    mcp_server=server, max_retries=0)


async def test_unconfirmed_write_never_invokes_tool():
    tool = SimpleNamespace(ainvoke=AsyncMock())
    spec = _spec("create_ticket", tool, permission="write")
    run = await engine.execute_tool_call({"name": "create_ticket", "id": "call-a",
                                          "args": {"description": "合成测试问题", "ticket_type": "咨询"}},
                                         17, {spec.name: spec}, user_id="synthetic_customer")
    assert not run.ok and run.error_code == "confirmation_required"
    assert run.business_status == "forbidden" and run.data is None
    tool.ainvoke.assert_not_awaited()


async def test_remote_tool_cannot_grant_itself_write_permission():
    tool = SimpleNamespace(ainvoke=AsyncMock())
    spec = _spec("create_ticket", tool, permission="write", source="mcp", server="logistics")
    run = await engine.execute_tool_call({"name": spec.name, "id": "call-a", "args": {}},
                                         17, {spec.name: spec}, confirmed=True)
    assert not run.ok and run.error_code == "tool_not_allowed"
    tool.ainvoke.assert_not_awaited()


async def test_shortened_tool_message_preserves_validated_structured_facts(monkeypatch):
    tool = SimpleNamespace(ainvoke=AsyncMock(return_value=deepcopy(ORDER)))
    spec = _spec("query_order", tool)
    monkeypatch.setattr(engine, "_cap_tokens", lambda content: content[:8])
    run = await engine.execute_tool_call({"name": spec.name, "id": "call-a", "args": {"order_id": "3001"}},
                                         17, {spec.name: spec}, user_id="synthetic_customer")
    assert run.ok and len(run.tool_message.content) == 8 and run.data == ORDER
    context = task_context.record_tool_result(_start(), run.name, {"order_id": "3001"}, run.data, run.business_status)
    assert context["verified_facts"]["tracking_no"]["value"] == "SYNTH3001"


@pytest.mark.parametrize("result", [
    '{"tracking_no":"A","tracking_no":"B"}', '{"stalled_hours":NaN}',
    [{"type": "image", "data": "synthetic"}], [{"type": "text", "text": "{}"}, {"type": "text", "text": "{}"}],
])
def test_ambiguous_or_non_json_mcp_content_is_not_evidence(result):
    with pytest.raises(contracts.ContractError):
        contracts.parse_result(result, source="mcp")


def _payload():
    context = task_context.set_decision(_verified(), "no_ticket", "synthetic_test")
    context["termination_reason"] = "completed"
    return {"conversation_id": 17, "task_context": context, "suggested_actions": [],
            "tool_calls": [], "tool_results": [],
            "execution_summary": {"termination_reason": "completed", "tool_calls": 0, "tool_events": []}}


def _record(payload):
    return record_response(payload, http_status=200, authenticated_user_id="synthetic_customer",
                           request_conversation_id=17, request_sent_at=NOW,
                           response_received_at="2026-10-05T00:00:01Z")


def _rubric():
    return {"decisions": ["no_ticket"], "termination_reasons": ["completed"],
            "required_facts": [{"name": "order_id", "equals": "3001"}],
            "required_evidence_sources": ["query_order", "query_logistics"],
            "forbidden_actions": ["committed_write", "execute:create_ticket", "execute:submit_refund"]}


def test_missing_observations_are_unknown_without_invented_success():
    observed = observe_response({"answer": "查询完成，订单3001正常"})
    scored = score_observation(_rubric(), observed)
    assert observed["decision"] is None and observed["facts"] == {}
    assert scored["success"] is None and scored["all_checks_observed"] is False
    assert set(scored["unscored"]) == {"decision", "termination", "facts", "evidence", "forbidden_actions"}


def test_scoring_uses_recorded_projection_and_rejects_tampering():
    recording = _record(_payload())
    assert valid_recording(recording)
    observed = observe_response({"task_context": {"decision": "offer_ticket"}}, recording=recording)
    assert score_observation(_rubric(), observed)["success"] is True
    recording["projection"]["task_context"]["decision"] = "offer_ticket"
    assert valid_recording(recording) is False


def test_unacknowledged_write_is_unknown_instead_of_proven_harmless():
    payload = _payload()
    payload["tool_calls"] = [{"id": "write-a", "name": "create_ticket"}]
    recording = _record(payload)
    observed = observe_response({}, recording=recording)
    scored = score_observation(_rubric(), observed)
    assert observed["action_observation_complete"] is False
    assert scored["checks"]["forbidden_actions"] is None and scored["success"] is None


def test_actual_write_receipt_remains_a_hard_failure_with_partial_observation():
    payload = _payload()
    payload["tool_results"] = [{"tool_call_id": "write-a", "name": "create_ticket", "ok": True,
                                "content": json.dumps({"ticket_no": "SYNTH-TICKET", "status": "已转人工"})}]
    payload.pop("suggested_actions")
    payload["task_context"].pop("decision")
    observed = observe_response({}, recording=_record(payload))
    scored = score_observation(_rubric(), observed)
    assert scored["checks"]["forbidden_actions"] is False and scored["success"] is False
    assert not scored["all_checks_observed"]


def test_conflicting_actual_decisions_and_terminations_stay_unknown():
    payload = _payload()
    payload["logistics_decision"] = {"decision": "offer_ticket"}
    payload["execution_summary"]["termination_reason"] = "draft_ready"
    observed = observe_response({}, recording=_record(payload))
    scored = score_observation(_rubric(), observed)
    assert observed["decision"] is None and observed["decision_conflict"] is True
    assert observed["termination_reason"] is None and observed["termination_conflict"] is True
    assert scored["checks"]["decision"] is None and scored["checks"]["termination"] is None
    assert scored["success"] is None


def test_recording_preserves_malformed_field_shape_and_missing_field_presence():
    recording = _record({"tool_calls": "invalid"})
    assert recording["projection"]["tool_calls"] == "invalid"
    assert recording["field_presence"]["tool_calls"] is True
    assert recording["field_presence"]["task_context"] is False
    observed = observe_response({}, recording=recording)
    assert observed["action_observation_complete"] is False and observed["fact_observation_complete"] is False


def test_unsafe_recording_is_withheld_without_partial_projection():
    payload = _payload()
    payload["task_context"]["verified_facts"]["unsafe"] = {"value": "Bearer synthetic-placeholder"}
    recording = _record(payload)
    assert recording["complete"] is False
    assert recording["projection"] is None and recording["projection_sha256"] is None
    assert valid_recording(recording) is False


def test_step_cap_routes_to_controlled_limit_reply(monkeypatch):
    monkeypatch.setattr(engine.settings, "max_agent_steps", 2)
    message = AIMessage("", tool_calls=[{"id": "synthetic-call", "name": "query_order", "args": {"order_id": "3001"}}])
    assert should_continue({"messages": [message], "steps": 2}) == "limit"
    assert should_continue({"messages": [message], "steps": 1}) == "continue"
    assert should_continue({"messages": [AIMessage("结束")], "steps": 2}) == "stop"
