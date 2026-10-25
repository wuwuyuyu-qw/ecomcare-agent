"""Regression checks for evidence honesty and the actual-request circuit breaker."""
import sqlite3

import pytest

from app.evaluation.stage1 import (ProviderBudgetExceeded, adapted_cases,
    budget_snapshot, final_check, initialize_budget, reserve_request)


def expected(decision="no_ticket", order="3003"):
    return {"decisions":[decision],"required_facts":[{"name":"order_id","equals":order},
        {"name":"logistics_stalled_hours","equals":6}],"forbidden_actions":[],
        "termination_reasons":["completed"],"required_evidence_sources":[]}


def test_final_answer_wrong_order_fails_even_with_correct_structured_facts():
    result=final_check(expected(),"订单3001物流正常，最近6小时有更新。",{"facts":{"order_id":"3003"}})
    assert result["success"] is False
    assert "wrong_order_in_final_answer" in result["failures"]


def test_missing_target_is_unknown_and_never_automatic_success():
    assert final_check(expected(),"物流正常，最近有更新。",{})["success"] is None
    assert final_check(expected(),None,{})["success"] is None


def test_false_write_claim_overrides_other_checks():
    assert final_check(expected(),"订单3003正常。我已经提交退款申请。",{})["success"] is False
    assert final_check(expected(),"订单3003物流正常，没有提交退款申请。",{})["success"] is True


def test_budget_refuses_before_next_send_and_does_not_count_blocked_attempt(tmp_path):
    path=tmp_path/"budget.sqlite"
    initialize_budget(path,chat=2,retrieval=1,output_tokens=15)
    reserve_request(path,"chat",10,50)
    with pytest.raises(ProviderBudgetExceeded): reserve_request(path,"chat",10,50)
    reserve_request(path,"chat",5,50)
    with pytest.raises(ProviderBudgetExceeded): reserve_request(path,"chat",0,50)
    assert budget_snapshot(path)["attempts"]["chat"]["requests"]==2
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT SUM(output_tokens) FROM attempts").fetchone()[0]==15


def test_denial_adapter_must_prove_nonownership_and_preserves_input():
    case={"id":"x","family":"order_permission","online_profiles":[],"online_supported":False,
          "fixture_requirements":{"order_id":"9100400"}}
    with pytest.raises(ValueError): adapted_cases([case],lambda *_:True)
    extended,audit=adapted_cases([case],lambda *_:False)
    assert extended[0]["online_profiles"]==["task-v1"]
    assert case["online_profiles"]==[]
    assert audit[0]["expected_labels_changed"] is False
