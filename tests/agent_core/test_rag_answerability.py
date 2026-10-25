from pathlib import Path
from app.evaluation.rag_answerability import load_bundle, confusion, threshold_scan


def test_new_corpus_is_self_authored_and_balanced_development_only():
    docs, cases, manifest = load_bundle(Path(__file__).resolve().parents[2] / "data/rag_answerability")
    assert len(docs) == 9 and len(cases) == 24 and sum(c["answerable"] for c in cases) == 12
    assert manifest["split"] == "development_only" and manifest["limits"]


def test_failure_not_silently_counted_as_correct_reject():
    rows = [{"answerable": True, "accepted": True, "status": "observed"},
            {"answerable": False, "accepted": False, "status": "observed"},
            {"answerable": False, "accepted": False, "status": "failed"}]
    result = confusion(rows)
    assert result["true_accept"] == result["true_reject"] == result["unknown"] == 1
    assert result["requested"] == 3


def test_threshold_scan_is_diagnostic_and_preserves_failed_denominator():
    rows = [{"answerable": True, "retrieval_status": "observed", "top_score": 0.8},
            {"answerable": False, "retrieval_status": "observed", "top_score": 0.8},
            {"answerable": False, "retrieval_status": "failed", "top_score": 0.0}]
    result = threshold_scan(rows)
    assert not result["runtime_threshold_changed"] and result["omitted_retrieval_failures"] == 1
    assert result["scan"][10]["false_accept"] == 1


def test_component_experiment_calls_the_same_final_guard_contract():
    from scripts.eval_rag_answerability import finalize_component_answer
    state = {"route": "knowledge", "evidence_strong": True, "evidence": "操作方法",
             "citations": [{"id": 43001}], "task_context": {}}
    result = finalize_component_answer(state, "通过聊天提供本人订单号查询物流[1]。")
    assert result["allowed"] and result["answer"].endswith("[1]。")


def test_reused_judge_adapter_is_actual_chat_model_and_function_calling():
    import asyncio
    from langchain_core.language_models.chat_models import BaseChatModel
    from scripts.judge_common_answers import create_judge_client
    from app.evaluation.common_judge import OUTPUT_SCHEMA
    client = create_judge_client({"CHAT_BASE_URL": "http://127.0.0.1:9/v1", "CHAT_API_KEY": "synthetic-placeholder"},
                                model="synthetic-offline", timeout=1, callbacks=[])
    try:
        assert isinstance(client, BaseChatModel)
        assert client.with_structured_output(OUTPUT_SCHEMA) is not None
        assert client.temperature == 0 and client.streaming and client.max_retries == 0
    finally:
        asyncio.run(client.root_async_client.close())
        client.root_client.close()
