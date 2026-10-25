import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage

from app.core import evidence_selection as selection
from app.config import settings
from app.graph import nodes

HITS = [{"id": 1, "question": "方法", "answer": "仅是方法", "rerank_score": 0.1},
        {"id": 2, "question": "物流", "answer": "需要本人订单", "rerank_score": 0.2}]


def retrieval_stub(monkeypatch):
    import app.core
    value = SimpleNamespace(search_knowledge=AsyncMock(return_value=HITS), arrange_head_tail=lambda hits: hits)
    monkeypatch.setitem(sys.modules, "app.core.retrieval", value)
    monkeypatch.setattr(app.core, "retrieval", value, raising=False)
    return value


@pytest.mark.parametrize("status,indices", [("full", []), ("none", [1]), ("conflict", [1]),
    ("full", [True]), ("full", [1.0]), ("full", [1, 1]), ("full", [0]), ("full", [3])])
def test_sources_must_be_valid_unique_existing_and_consistent(status, indices):
    with pytest.raises((ValueError, ValidationError)):
        selection.bind_selection({"status": status, "supporting_indices": indices, "reason": "检查"}, HITS)


def test_binding_keeps_retrieval_order_and_partial_never_releases():
    result = selection.bind_selection({"status": "full", "supporting_indices": [2, 1], "reason": "够用"}, HITS)
    assert result["selected_ids"] == ["1", "2"] and result["useful"]
    partial = selection.bind_selection({"status": "partial", "supporting_indices": [1], "reason": "缺第二项"}, HITS)
    assert not partial["useful"]
    with pytest.raises(ValueError):
        selection.bind_selection({"status": "full", "supporting_indices": [1], "reason": "检查"}, [HITS[0], HITS[0]])


def test_empty_evidence_skips_model(monkeypatch):
    call = AsyncMock()
    monkeypatch.setattr(selection.llm, "structured", lambda *_a, **_k: SimpleNamespace(ainvoke=call))
    assert asyncio.run(selection.select_evidence("问题", []))["status"] == "none"
    call.assert_not_called()


def test_failure_closes_gate_and_cancellation_propagates(monkeypatch):
    call = AsyncMock(side_effect=RuntimeError("synthetic failure"))
    monkeypatch.setattr(selection.llm, "structured", lambda *_a, **_k: SimpleNamespace(ainvoke=call))
    assert asyncio.run(selection.select_evidence("问题", HITS))["status"] == "unavailable"
    call.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(selection.select_evidence("问题", HITS))


def test_original_question_and_selected_sources_survive_query_rewrite(monkeypatch):
    retrieval = retrieval_stub(monkeypatch)
    call = AsyncMock(return_value={"status": "full", "supporting_indices": [2], "reason": "够用"})
    monkeypatch.setattr(settings, "evidence_selection_enabled", True)
    monkeypatch.setattr(selection.llm, "structured", lambda *_a, **_k: SimpleNamespace(ainvoke=call))
    monkeypatch.setattr(nodes.query_understanding, "understand", AsyncMock(return_value={"standard": "已经改写", "expanded": []}))
    monkeypatch.setattr(retrieval, "search_knowledge", AsyncMock(return_value=HITS))
    monkeypatch.setattr(nodes.policy_runtime, "build_citations", lambda hits: [dict(h, n=i) for i, h in enumerate(hits, 1)])
    result = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("只学方法，不读订单3003")]}))
    assert "只学方法，不读订单3003" in call.call_args.args[0][1].content
    assert "已经改写" not in call.call_args.args[0][1].content
    assert [c["id"] for c in result["citations"]] == [2] and result["evidence_strong"]


@pytest.mark.parametrize("status", ["partial", "none", "conflict", "unavailable"])
def test_graph_rejections_clear_previous_citations(monkeypatch, status):
    retrieval = retrieval_stub(monkeypatch)
    monkeypatch.setattr(settings, "evidence_selection_enabled", True)
    monkeypatch.setattr(nodes.query_understanding, "understand", AsyncMock(return_value={"standard": "问题", "expanded": []}))
    monkeypatch.setattr(retrieval, "search_knowledge", AsyncMock(return_value=HITS))
    rejection = {"protocol": selection.PROTOCOL, "status": status, "reason": "不足", "selected_ids": [], "useful": False}
    monkeypatch.setattr(selection, "select_evidence", AsyncMock(return_value=rejection))
    result = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("问题")], "citations": [{"id": 999}]}))
    assert not result["evidence_strong"] and result["citations"] == [] and result["evidence"] == ""


def test_faq_uses_original_keyword_and_selected_citations(monkeypatch):
    retrieval_stub(monkeypatch)
    from app.tools.builtin import faq
    monkeypatch.setattr(settings, "evidence_selection_enabled", True)
    monkeypatch.setattr(faq.query_understanding, "understand", AsyncMock(return_value={"standard": "改写", "expanded": []}))
    monkeypatch.setattr(faq.retrieval, "search_knowledge", AsyncMock(return_value=HITS))
    gate = AsyncMock(return_value={"useful": True, "selected": [HITS[1]]})
    monkeypatch.setattr(selection, "select_evidence", gate)
    monkeypatch.setattr(faq, "build_citations", lambda hits: [dict(h, n=i) for i, h in enumerate(hits, 1)])
    result = asyncio.run(faq.query_faq.ainvoke({"keyword": "原始教程问题"}))
    assert gate.call_args.args[0] == "原始教程问题" and [c["id"] for c in result["citations"]] == [2]
