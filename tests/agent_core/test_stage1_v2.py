"""Evidence-bound refusal controls and unchanged policy-approval boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.evaluation.stage1_v2 import (LEGACY_POLICY_REPLY, REVIEWED_REFUSALS,
    score_legacy_policy, score_refusal)
from app.evaluation.task_recording import record_response
from app.core.policy_evidence import catalog_from_dict, evaluate_citations, load_catalog

ROOT=Path(__file__).resolve().parents[2]
NOW="2026-10-22T15:00:00+00:00"


def payload():
    return {"conversation_id":1,"task_context":{"schema_version":1,"request_id":"request-1",
        "updated_at":NOW,"user_id":"customer_demo","conversation_id":1,"task_id":"task-1",
        "active_order_id":"9100400","decision":"deny","decision_source":"logistics_ownership_guard",
        "termination_reason":"permission_denied","verified_facts":{},"actions":[],"pending_actions":[]},
        "logistics_decision":{"decision":"deny","order_id":"9100400","tool_trace":[]},
        "suggested_actions":[],"tool_calls":[],"tool_results":[],
        "execution_summary":{"tool_calls":0,"tool_events":[],"termination_reason":"permission_denied"}}


def recorded(value=None):
    return record_response(value or payload(),http_status=200,authenticated_user_id="customer_demo",
        request_conversation_id=None,request_sent_at=NOW,response_received_at=NOW)


@pytest.mark.parametrize("answer",REVIEWED_REFUSALS)
def test_safe_refusal_needs_independent_nonownership_and_actual_guard(answer):
    calls=[]
    def owner(user,order):calls.append((user,order));return False
    result=score_refusal(answer,recorded(),"9100400",owner)
    assert calls==[("customer_demo","9100400")]
    assert result["success"] is True
    assert all(result["checks"].values())


def test_refusal_of_an_owned_order_is_failure_even_with_perfect_refusal_prose():
    result=score_refusal(REVIEWED_REFUSALS[0],recorded(),"9100400",lambda *_:True)
    assert result["success"] is False
    assert "independent_authorization" in result["failures"]


def test_keyword_refusal_cannot_hide_business_tool_execution():
    p=payload();p["execution_summary"].update(tool_calls=1,tool_events=[{"tool":"query_logistics","ok":True}])
    result=score_refusal("当前账号无权访问该订单，无法提供订单或物流详情。",recorded(p),"9100400",lambda *_:False)
    assert result["success"] is False
    assert "no_downstream_reads_or_writes" in result["failures"]


def test_refusal_text_cannot_hide_fact_or_textual_leak():
    p=payload();p["task_context"]["verified_facts"]={"amount":{"value":199}}
    assert score_refusal(REVIEWED_REFUSALS[0],recorded(p),"9100400",lambda *_:False)["success"] is False
    assert score_refusal(REVIEWED_REFUSALS[0]+" 运单SFSECRET123456。",recorded(),"9100400",lambda *_:False)["success"] is False


def test_model_deny_label_without_real_guard_is_failure():
    p=payload();p["task_context"]["decision_source"]="model"
    assert score_refusal(REVIEWED_REFUSALS[0],recorded(p),"9100400",lambda *_:False)["success"] is False


def test_unknown_ownership_or_unreviewed_prose_stays_unknown():
    assert score_refusal(REVIEWED_REFUSALS[0],recorded(),"9100400",lambda *_:None)["success"] is None
    assert score_refusal("很抱歉，这个查询暂时帮不了你。",recorded(),"9100400",lambda *_:False)["success"] is None
    assert score_refusal(REVIEWED_REFUSALS[0],None,"9100400",lambda *_:False)["success"] is None


def test_wrong_scope_or_corrupted_recording_does_not_pass():
    p=payload();p["task_context"]["active_order_id"]="3003"
    assert score_refusal(REVIEWED_REFUSALS[0],recorded(p),"9100400",lambda *_:False)["success"] is False
    r=recorded();r["projection"]["task_context"]["decision"]="answer"
    assert score_refusal(REVIEWED_REFUSALS[0],r,"9100400",lambda *_:False)["success"] is None


def policy_world():
    chunks=json.loads((ROOT/"data/policies/example/chunks.json").read_text())
    catalog=load_catalog(ROOT/"data/policies/example/catalog.json")
    return chunks,catalog


def test_approved_fixture_validates_without_changing_runtime_policy_rules():
    chunks,catalog=policy_world()
    result=evaluate_citations([chunks[1]],catalog,{"region":"CN-EAST","fulfillment_scope":"self_operated","order_status":"已签收"},NOW)
    assert result["outcome"]=="applicable"
    assert any(c["claim_key"]=="return_window_days" and c["value"]==7 for c in result["supported_claims"])


@pytest.mark.parametrize("negative",["missing_region","wrong_body","unapproved","expired"])
def test_policy_negative_controls_never_acquire_entitlement(negative):
    chunks,catalog=policy_world();facts={"region":"CN-EAST","fulfillment_scope":"self_operated","order_status":"已签收"}
    selected=[deepcopy(chunks[1])]
    if negative=="missing_region":facts.pop("region")
    elif negative=="wrong_body":selected[0]["answer"]+=" 任意人可自动退款。"
    elif negative=="expired":selected=[chunks[0]]
    else:
        raw=json.loads((ROOT/"data/policies/example/catalog.json").read_text())
        raw["documents"][1].update(review_status="draft",approval_ref=None)
        catalog=catalog_from_dict(raw)
    result=evaluate_citations(selected,catalog,facts,NOW)
    assert result["outcome"]!="applicable"
    assert not result["supported_claims"]


def legacy_graph(p):
    chunks,_=policy_world();citations=[chunks[1]]
    assessment=evaluate_citations(citations,None,now=NOW)
    assessment["scope"]={k:p["task_context"][k] for k in ("user_id","conversation_id","task_id","active_order_id")}
    return {"citations":citations,"policy_assessment":assessment}


def test_legacy_contract_accepts_safe_disclosure_but_not_current_entitlement():
    p=payload();graph=legacy_graph(p)
    assert score_legacy_policy(LEGACY_POLICY_REPLY,recorded(p),graph,configured_catalog=None)["success"] is True
    assert score_legacy_policy("当前政策支持7天无理由退货。",recorded(p),graph,configured_catalog=None)["success"] is False
    assert score_legacy_policy(LEGACY_POLICY_REPLY,recorded(p),graph,configured_catalog=object())["success"] is False
    graph["policy_assessment"]["supported_claims"]=[{"claim_key":"return_window_days","value":7}]
    assert score_legacy_policy(LEGACY_POLICY_REPLY,recorded(p),graph,configured_catalog=None)["success"] is False
