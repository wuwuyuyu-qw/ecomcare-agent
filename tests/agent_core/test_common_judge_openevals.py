"""Exercise the installed OpenEvals interface with a local model stub only."""
import pytest

pytest.importorskip("openevals", reason="optional eval dependency; install --group eval for this contract check")
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context
from app.evaluation.common_judge import create_evaluator
from app.evaluation.task_review import CRITERIA


class LocalJudge(BaseChatModel):
    @property
    def _llm_type(self):
        return "synthetic-local-judge"

    def _generate(self, *args, **kwargs):
        raise AssertionError("stub must never invoke a provider")

    def with_structured_output(self, schema=None, **kwargs):
        assert schema["properties"]["criteria"]["required"] == list(CRITERIA)

        def result(messages):
            assert "合成用户输入" in str(messages)
            return {"criteria": {key: {"outcome": "unknown", "rationale": "本地合成接口检查",
                                       "evidence_refs": []} for key in CRITERIA}}
        return RunnableLambda(result)


@pytest.mark.asyncio
async def test_upstream_async_custom_schema_interface_without_network():
    evaluator = create_evaluator(LocalJudge())
    with tracing_context(enabled=False):
        value = await evaluator(inputs="合成用户输入", outputs="合成答案", reference_outputs="合成证据")
    assert set(value["criteria"]) == set(CRITERIA)
    assert all(item["outcome"] == "unknown" for item in value["criteria"].values())
