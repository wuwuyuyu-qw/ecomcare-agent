"""物流 Agent 离线测试：模型与工具均模拟，不接远端和真实数据库。"""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph import logistics, routing, runtime
from app.logistics_agent.decision import decide_logistics
from app.tools import business


def test_pure_guard_normal_stalled_signed_and_mismatched_evidence():
    order = business.order_snapshot("3001")
    base = {"tracking_no": order["tracking_no"], "status": "运输中"}
    assert decide_logistics(claim_type="status", user_text="3001 到哪了", order=order,
                            logistics={**base, "stalled_hours": 6}).verdict == "no_ticket"
    stalled = decide_logistics(claim_type="stalled", user_text="三天没更新", order=order,
                               logistics={**base, "stalled_hours": 72})
    assert (stalled.verdict, stalled.issue_type) == ("offer_ticket", "stalled_48h")
    assert decide_logistics(claim_type="stalled", user_text="三天没更新", order=order,
                            logistics={**base, "stalled_hours": 47.9}).verdict == "no_ticket"
    signed = decide_logistics(claim_type="signed_not_received", user_text="显示签收但我没收到", order=order,
                              logistics={"tracking_no": order["tracking_no"], "status": "已签收"})
    assert (signed.verdict, signed.issue_type) == ("offer_ticket", "signed_not_received")
    mismatch = decide_logistics(claim_type="signed_not_received", user_text="显示签收但我没收到", order=order,
                                logistics={"tracking_no": "SFOTHER", "status": "已签收"})
    assert mismatch.verdict == "human_review"
    missing = decide_logistics(claim_type="status", user_text="到哪了", order=order,
                               logistics={"tracking_no": order["tracking_no"]})
    assert missing.verdict == "human_review"
    ungrounded = decide_logistics(claim_type="signed_not_received", user_text="3002 到哪了",
                                  order=order,
                                  logistics={"tracking_no": order["tracking_no"], "status": "已签收"})
    assert ungrounded.verdict == "human_review"
    no_time = decide_logistics(claim_type="status", user_text="3001 到哪了", order=order,
                               logistics={**base})
    assert no_time.verdict == "human_review"


def test_fixture_orders_preserve_original_demo_orders():
    orders = {o["order_id"] for o in business.list_user_orders("u-logistics")}
    assert {"1001", "2002", "3001", "3002", "3003"} <= orders
    assert not business.owns_order("u-logistics", "9999")
    assert business.order_snapshot("3001")["tracking_no"] == "SFDEMO3001"
    assert business.order_snapshot("3002")["tracking_no"] == "SFDEMO3002"
    assert business.order_snapshot("3003")["tracking_no"] == "SFDEMO3003"


def test_logistics_route_and_turn_reset():
    assert routing.INTENT_TO_ROUTE["物流"] == "logistics"
    turn = runtime._graph_input("u-logistics", "订单3001停滞", 1, 2, "", 0)
    assert turn["logistics_decision"] == {} and turn["logistics_order_id"] == ""


@pytest.mark.asyncio
async def test_missing_order_clarifies_and_recent_user_order_survives_next_turn(monkeypatch):
    async def fake_model(_):
        return logistics.LogisticsExtraction(claim_type="stalled", order_id=None,
            missing_fields=["order_id"], needs_policy=False, user_claim="物流没更新")

    from langchain_core.runnables import RunnableLambda
    monkeypatch.setattr(logistics.llm, "structured", lambda *a, **kw: RunnableLambda(fake_model))
    missing = await logistics.extract_logistics({"messages": [HumanMessage("物流没更新")]})
    assert missing["logistics_decision"]["decision"] == "clarify"
    assert logistics.route_after_extract(missing) == "done"
    followup = await logistics.extract_logistics({"messages": [
        HumanMessage("订单3001的物流没更新"), HumanMessage("它还是没动")
    ]})
    assert followup["logistics_order_id"] == "3001"
    assert followup["trace"]["logistics_extraction"]["order_origin"] == "recent_user_history"
    assert logistics.route_after_extract(followup) == "investigate"
    ambiguous = await logistics.extract_logistics({"messages": [HumanMessage("3001 和 3002 都查一下")]})
    assert ambiguous["logistics_decision"]["decision"] == "clarify"


@pytest.mark.asyncio
async def test_investigation_tool_chain_and_no_auto_write(monkeypatch):
    async def specs():
        return [SimpleNamespace(name=name) for name in ("query_order", "query_logistics", "query_faq")]

    seen = []
    async def execute(tc, cid, tool_specs, **kwargs):
        seen.append(tc["name"])
        assert tc["name"] != "create_ticket"
        if tc["name"] == "query_order":
            result = business.order_snapshot(tc["args"]["order_id"])
        else:
            result = {"tracking_no": tc["args"]["tracking_no"],
                      "status": "运输中", "stalled_hours": 72,
                      "current_city": "杭州", "trace": ["72 小时无新扫描记录"]}
        return SimpleNamespace(ok=True, status="成功", duration_ms=1, retry_count=0,
                               tool_message=SimpleNamespace(content=json.dumps(result, ensure_ascii=False)))

    async def draft(*args):
        return "物流三天没更新", "model"

    monkeypatch.setattr(logistics.registry, "get_all_specs", specs)
    monkeypatch.setattr(logistics.engine, "execute_tool_call", execute)
    monkeypatch.setattr(logistics, "_draft_summary", draft)
    out = await logistics.investigate_logistics({
        "conversation_id": 1, "user_id": "u-logistics", "logistics_order_id": "3001",
        "logistics_claim_type": "stalled", "logistics_claim": "物流三天没更新",
        "logistics_needs_policy": False, "messages": [HumanMessage("订单3001物流三天没更新")],
    })
    assert seen == ["query_order", "query_logistics"]
    assert out["logistics_decision"]["decision"] == "offer_ticket"
    assert {e["source"] for e in out["logistics_decision"]["evidence"]} == {"order", "logistics"}
    assert out["suggested_actions"][0]["type"] == "create_ticket"


@pytest.mark.asyncio
async def test_investigation_denies_nonowner_and_handles_tool_outage(monkeypatch):
    denied = await logistics.investigate_logistics({
        "conversation_id": 1, "user_id": "u-logistics", "logistics_order_id": "9999"})
    assert denied["logistics_decision"]["decision"] == "deny"

    async def specs():
        return [SimpleNamespace(name="query_order"), SimpleNamespace(name="query_logistics")]
    async def execute(tc, cid, tool_specs, **kwargs):
        if tc["name"] == "query_order":
            value = business.order_snapshot("3001")
            return SimpleNamespace(ok=True, status="成功", duration_ms=1, retry_count=0,
                                   tool_message=SimpleNamespace(content=json.dumps(value)))
        return SimpleNamespace(ok=False, status="超时", duration_ms=1000, retry_count=1,
                               tool_message=SimpleNamespace(content="工具暂时不可用"))
    monkeypatch.setattr(logistics.registry, "get_all_specs", specs)
    monkeypatch.setattr(logistics.engine, "execute_tool_call", execute)
    failed = await logistics.investigate_logistics({
        "conversation_id": 1, "user_id": "u-logistics", "logistics_order_id": "3001",
        "logistics_claim_type": "stalled", "logistics_needs_policy": False})
    assert failed["logistics_decision"]["decision"] == "human_review"
    assert failed["logistics_decision"]["tool_trace"][-1]["status"] == "超时"
    assert all(a["type"] != "create_ticket" for a in failed["suggested_actions"])


@pytest.mark.asyncio
async def test_compiled_graph_routes_logistics_to_evidence_decision(monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graph import build, nodes

    async def resolve(_state):
        return {"resolved_query": "订单3001三天没更新"}
    async def classify(_state):
        return {"intent": "物流", "route": "logistics"}
    async def no_db_log(_state):
        return {}
    monkeypatch.setattr(nodes, "resolve_reference", resolve)
    monkeypatch.setattr(nodes, "classify_intent", classify)
    monkeypatch.setattr(nodes, "log_node", no_db_log)

    def structured(schema, **kwargs):
        async def invoke(_value):
            if schema is logistics.LogisticsExtraction:
                return logistics.LogisticsExtraction(
                    claim_type="stalled", order_id="3001", missing_fields=[],
                    needs_policy=False, user_claim="三天没更新")
            return logistics.LogisticsDraft(issue_summary="三天没更新")
        return RunnableLambda(invoke)
    monkeypatch.setattr(logistics.llm, "structured", structured)

    async def specs():
        return [SimpleNamespace(name="query_order"), SimpleNamespace(name="query_logistics")]
    async def execute(tc, cid, tool_specs, **kwargs):
        if tc["name"] == "query_order":
            result = business.order_snapshot("3001")
        else:
            result = {"tracking_no": "SFDEMO3001", "status": "运输中",
                      "stalled_hours": 72, "current_city": "杭州", "trace": []}
        return SimpleNamespace(ok=True, status="成功", duration_ms=1, retry_count=0,
                               tool_message=SimpleNamespace(content=json.dumps(result, ensure_ascii=False)))
    monkeypatch.setattr(logistics.registry, "get_all_specs", specs)
    monkeypatch.setattr(logistics.engine, "execute_tool_call", execute)

    graph = build.build_graph(checkpointer=InMemorySaver())
    final = await graph.ainvoke({"messages": [HumanMessage("订单3001三天没更新")],
                                "user_id": "u-graph", "conversation_id": 91,
                                "trace": None},
                               {"configurable": {"thread_id": "logistics-graph-test"}})
    assert final["logistics_decision"]["decision"] == "offer_ticket"
    assert final["logistics_decision"]["issue_type"] == "stalled_48h"
    assert final["route"] == "logistics"
    assert final["answer"] and final["suggested_actions"][0]["type"] == "create_ticket"
