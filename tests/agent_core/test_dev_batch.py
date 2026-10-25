"""Zero-provider-cost checks: selection, shared caps, retry counts and exit ledger."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.evaluation import dev_batch as batch
from app.evaluation.stage1 import ProviderBudgetExceeded, adapted_cases, budget_snapshot
from app.evaluation.task_cases import load_dataset
from app.tools.business import owns_order

ROOT = Path(__file__).resolve().parents[2]


def selected():
    cases, manifest = load_dataset(ROOT/"data/agent_eval", "dev")
    cases, _ = adapted_cases(cases, owns_order)
    return batch.select_cases(cases, batch.CASE_IDS), manifest


@pytest.mark.parametrize("ids", [None, [], list(batch.CASE_IDS[:-1]), list(reversed(batch.CASE_IDS)),
    [*batch.CASE_IDS, "task-01-03"], [*batch.CASE_IDS[:-1], "task-01-07"], [batch.CASE_IDS[0]]*12])
def test_reject_any_selection_other_than_frozen_whitelist(ids):
    with pytest.raises(ValueError): batch.validate_ids(ids)


def test_whitelist_exact_count_split_original_expected_and_identity(monkeypatch):
    monkeypatch.setenv("MEWHELP_FIXTURE_PROFILE", "task-v1")
    cases, _ = selected()
    assert [c["id"] for c in cases] == list(batch.CASE_IDS)
    assert len(cases) == 12 and sum(len(c["turns"]) for c in cases) == 13
    assert {c["split"] for c in cases} == {"dev"}
    originals, _ = load_dataset(ROOT/"data/agent_eval", "dev")
    for case in cases:
        assert case["turns"] == next(c["turns"] for c in originals if c["id"] == case["id"])
        if case["family"] in {"order_permission", "prompt_injection"}:
            assert not owns_order("customer_demo", case["fixture_requirements"]["order_id"])
    assert batch.definition()["authorization"].startswith("pending")


@pytest.mark.parametrize("kind,cap,other", [("chat", 80, "retrieval"), ("retrieval", 20, "chat")])
def test_either_cap_blocks_both_providers_and_counts_every_reservation(tmp_path, kind, cap, other):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    for i in range(cap): batch.reserve(path, kind, 1 if kind == "chat" else 0, 1)
    assert batch.stopped(path) == "provider_request_cap_reached"
    for provider in (kind, other):
        with pytest.raises(ProviderBudgetExceeded): batch.reserve(path, provider, 1, 1)
    assert sum(v["requests"] for v in budget_snapshot(path)["attempts"].values()) == cap


def test_atomic_concurrent_reservations_never_exceed_global_cap(tmp_path):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    def send(_):
        try: batch.reserve(path, "retrieval", 0, 1); return True
        except ProviderBudgetExceeded: return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(send, range(28))) == 20
    assert budget_snapshot(path)["attempts"]["retrieval"]["requests"] == 20


def test_output_quota_not_actual_token_or_currency_accounting(tmp_path):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    with sqlite3.connect(path) as conn: conn.execute("UPDATE limits SET output_tokens=3")
    batch.reserve(path, "chat", 2, 1)
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(path, "chat", 2, 1)
    assert batch.stopped(path) == "provider_gate_blocked"
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(path, "retrieval", 0, 1)
    assert budget_snapshot(path)["attempts"]["chat"]["requested_output_tokens"] == 2


def test_http_transport_retry_counts_and_usage_no_secrets(tmp_path, monkeypatch):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(path, {"api.deepseek.com", "api.siliconflow.cn"})
    calls = []
    def transport(request):
        calls.append(request)
        if len(calls) == 1: raise httpx.ConnectError("synthetic retry")
        return httpx.Response(200, json={"model": "deepseek-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_cache_hit_tokens": 5}})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        for i in range(2):
            try: client.post("https://api.deepseek.com/chat/completions", json={"model": "deepseek-v4-flash", "max_tokens": 10000})
            except httpx.ConnectError: pass
        with pytest.raises(ProviderBudgetExceeded):
            client.post("https://api.siliconflow.cn/v1/embeddings", json={"model": "Pro/BAAI/bge-m3"})
    assert len(calls) == 2
    snap = budget_snapshot(path)
    assert snap["attempts"]["chat"]["requests"] == 2 and snap["http_status_counts"] == {"None": 1, "200": 1}
    with sqlite3.connect(path) as conn: assert conn.execute("SELECT input_tokens,output_tokens,cache_read FROM provider_usage").fetchall() == [(10,3,5)]


def test_limit_request_can_complete_but_next_provider_never_reaches_transport(tmp_path, monkeypatch):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    for _ in range(19): batch.reserve(path, "retrieval", 0, 1)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(path, {"api.deepseek.com", "api.siliconflow.cn"})
    sent = []
    def transport(request):
        sent.append(request.url.host)
        return httpx.Response(200, json={"usage": {"total_tokens": 2}})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        assert client.post("https://api.siliconflow.cn/v1/embeddings", json={"model": "BAAI/bge-m3"}).status_code == 200
        with pytest.raises(ProviderBudgetExceeded):
            client.post("https://api.deepseek.com/chat/completions", json={"model": "deepseek-v4-flash", "max_tokens":10000})
    assert sent == ["api.siliconflow.cn"] and batch.stopped(path) == "provider_request_cap_reached"


def test_config_drift_requires_new_review_without_loading_real_environment():
    frozen = dict(chat_base_url="https://api.deepseek.com/v1", embed_base_url="https://api.siliconflow.cn/v1",
        rerank_base_url="https://api.siliconflow.cn/v1", chat_model="deepseek-v4-flash", embed_model="BAAI/bge-m3",
        rerank_model="BAAI/bge-reranker-v2-m3", max_output_tokens=10000, intent_model="", summary_model="",
        intent_base_url="", summary_base_url="", policy_catalog_path="", compound_agent_enabled=False,
        compound_joint_triage_enabled=False, evidence_selection_enabled=False, skip_coref_without_history=False,
        max_agent_steps=6, max_task_tool_calls=12, max_identical_tool_calls=2, turn_timeout_seconds=120.0,
        mcp_tool_timeout=10.0, chat_thinking="adaptive", chat_reasoning_effort="", retrieval_collection="knowledge")
    batch.verify_configuration(SimpleNamespace(**frozen))
    for key, value in [("chat_model", "deepseek-v4-pro"), ("embed_model", "Pro/BAAI/bge-m3"),
                       ("compound_agent_enabled", True), ("max_output_tokens", 20000), ("max_task_tool_calls", 50),
                       ("chat_thinking", "disabled")]:
        with pytest.raises(ValueError): batch.verify_configuration(SimpleNamespace(**{**frozen, key: value}))


def test_chat_total_tokens_are_not_misreported_as_input_tokens(tmp_path, monkeypatch):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(path, {"api.deepseek.com"})
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"usage":{"total_tokens":10,"completion_tokens":3}}))) as client:
        client.post("https://api.deepseek.com/chat/completions", json={"model":"deepseek-v4-flash","max_tokens":10000})
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT input_tokens,output_tokens,complete FROM provider_usage").fetchall() == [(None,3,0)]


async def test_stream_usage_chunks_are_counted_without_storing_model_content(tmp_path, monkeypatch):
    path = tmp_path/"budget.sqlite"; batch.initialize(path)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(path, {"api.deepseek.com"})
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for value in [b'data: {"choices":[{"delta":{"content":"synthetic-private-prose"}}]}\n\n',
                          b'data: {"model":"deepseek-flash","usage":{"prompt_to',
                          b'kens":20,"completion_tokens":4}}\n\ndata: [DONE]\n\n']:
                yield value
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200,stream=Stream()))) as client:
        async with client.stream("POST", "https://api.deepseek.com/chat/completions", json={"model":"deepseek-v4-flash","max_tokens":10000}) as response:
            assert b"synthetic-private-prose" in await response.aread()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT input_tokens,output_tokens FROM provider_usage").fetchall() == [(20,4)]
    assert b"synthetic-private-prose" not in path.read_bytes()


def test_budget_confirmation_and_full_dev_holdout_still_fail_before_io(tmp_path):
    from scripts.eval_stage1_baseline import run
    for phase in ("dev", "holdout"):
        with pytest.raises(ValueError, match="bounded smoke"):
            run(SimpleNamespace(phase=phase, contract="v2", authorized_full=True))
    with pytest.raises(ValueError, match="confirmed batch budget"):
        run(SimpleNamespace(phase="dev12", contract="v2", authorized_full=False,
            directory=tmp_path, case_ids=list(batch.CASE_IDS), budget_confirmed=False))
    assert list(tmp_path.iterdir()) == []


def test_denominator_skip_unknown_refusal_and_explicit_failure_are_distinct(monkeypatch):
    monkeypatch.setenv("MEWHELP_FIXTURE_PROFILE", "task-v1")
    cases, _ = selected()
    row = {"turns": [{"score": {"success": True}, "final_score": {"success": True}, "response_recording": {"http_status": 200}}]}
    result = batch.finalize(deepcopy(row), cases[6])
    assert result["task_success"] is True and result["failure_type"] == "correct_refusal"
    unknown = deepcopy(row); unknown["turns"][0]["final_score"]["success"] = None
    assert batch.finalize(unknown, cases[6])["task_success"] is None
    wrong = deepcopy(row); wrong["turns"][0]["final_score"]["success"] = False
    assert batch.finalize(wrong, cases[6])["task_success"] is False
    interrupted = batch.finalize(deepcopy(row), cases[-1], "provider_request_cap_reached")
    assert interrupted["task_success"] is None and interrupted["failure_type"] == "budget_interrupted"
    rows = [batch.not_run(c, "budget") for c in cases]; rows[0] = dict(interrupted, id=cases[0]["id"], family=cases[0]["family"])
    counts = batch.denominator(rows)
    assert counts["requested"] == counts["saved"] == 12
    assert (counts["unknown"], counts["skipped"], counts["aborted"], counts["not_run"]) == (0,0,1,11)
    assert sum(counts[k] for k in ("pass", "fail", "unknown", "skipped", "aborted", "not_run")) == 12
    assert counts["baseline_eligible"] == 0


def test_business_skip_unknown_and_protected_pass_remain_separate():
    counts = batch.denominator([
        {"family":"f","status":"skipped","task_success":None},
        {"family":"f","status":"evaluated","task_success":None,"baseline_eligible":True},
        {"family":"f","status":"evaluated","task_success":False,"baseline_eligible":True},
        {"family":"f","status":"evaluated","task_success":True,"baseline_eligible":False},
        {"family":"f","status":"NOT_RUN","task_success":None},
        {"family":"f","status":"ABORTED","task_success":None},
    ])
    assert all(counts[k] == 1 for k in ("pass","fail","unknown","skipped","not_run","aborted"))
    assert counts["baseline_eligible"] == 2


def test_probe_zero_retrieval_budget_allows_two_chat_requests_only(tmp_path):
    path=tmp_path/"probe.sqlite"; batch.initialize(path)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE limits SET chat=2,retrieval=0,output_tokens=20000")
    assert batch.stopped(path) is None
    batch.reserve(path,"chat",10000,100)
    batch.reserve(path,"chat",10000,100)
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(path,"chat",10000,100)
    other=tmp_path/"retrieval.sqlite"; batch.initialize(other)
    with sqlite3.connect(other) as conn:
        conn.execute("UPDATE limits SET chat=2,retrieval=0,output_tokens=20000")
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(other,"retrieval",0,100)
    with sqlite3.connect(other) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_online_before_turn_exit_and_incremental_raw_save(monkeypatch):
    from scripts import eval_task_agent as runner
    monkeypatch.setenv("MEWHELP_FIXTURE_PROFILE", "task-v1")
    cases, manifest = selected(); case = cases[-1]
    profile = manifest["fixture_profiles"]["task-v1"]
    config = {"fixture_profile": "task-v1", "fixture_version": profile["version"], "fixture_sha256": profile["sha256"]}
    me = {**config, "user_id": "customer_demo"}
    monkeypatch.setenv("OFFLINE_USER", "customer_demo"); monkeypatch.setenv("OFFLINE_PASSWORD", "offline-placeholder")
    sent, saved = [], []
    def post(base, endpoint, body, timeout, token=""):
        sent.append(endpoint)
        if endpoint.endswith("login"): return 200, {"access_token": "synthetic"}
        return 200, {"conversation_id": 7, "answer": "synthetic"}
    monkeypatch.setattr(runner, "_post", post)
    monkeypatch.setattr(runner, "_get", lambda *args: (200, me))
    options = SimpleNamespace(base_url="http://127.0.0.1:9", username_env="OFFLINE_USER", password_env="OFFLINE_PASSWORD", timeout=1,
                              before_turn=lambda: len(saved) == 0, after_turn=lambda turn: saved.append(deepcopy(turn)))
    rows, _ = runner._online(options, [case], manifest, config)
    assert sent.count("/api/agent") == 1 and len(saved) == len(rows[0]["turns"]) == 1
    assert rows[0]["expected_turn_count"] == 2 and "incomplete_turn_sequence" in rows[0]["failures"]


async def test_tool_observer_retains_actual_arguments_and_validated_results(tmp_path, monkeypatch):
    from app.core.execution import execution_scope
    from app.tools import engine, registry
    from app.tools.builtin.orders import query_product
    from unittest.mock import AsyncMock
    original = engine.execute_tool_call
    monkeypatch.setattr(engine, "execute_tool_call", original)
    monkeypatch.setattr(engine, "_audit", AsyncMock())
    path = tmp_path/"tools.jsonl"
    batch.install_tool_observer(path)
    spec = registry.spec_from_langchain_tool(query_product, source="builtin")
    with execution_scope(1) as scope:
        run = await engine.execute_tool_call({"name":"query_product","id":"offline", "args":{"product_name":"iPhone"}},
            91, {spec.name:spec}, user_id="customer_demo")
    row = json.loads(path.read_text())
    assert row["request_id"] == scope.request_id and row["authenticated_user_id"] == "customer_demo"
    assert row["tool_call"]["args"] == {"product_name":"iPhone"}
    assert row["validated_result"] == run.data and row["ok"] is True


async def test_tool_observer_propagates_cancellation_and_retains_no_result(tmp_path, monkeypatch):
    import asyncio
    from app.tools import engine
    from unittest.mock import AsyncMock
    monkeypatch.setattr(engine, "execute_tool_call", AsyncMock(side_effect=asyncio.CancelledError()))
    path = tmp_path/"tools.jsonl"; batch.install_tool_observer(path)
    with pytest.raises(asyncio.CancelledError):
        await engine.execute_tool_call({"name":"query_order","id":"offline","args":{"order_id":"3003"}}, 91, {}, user_id="customer_demo")
    row = json.loads(path.read_text())
    assert row["raised_error_class"] == "CancelledError" and row["validated_result"] is None
