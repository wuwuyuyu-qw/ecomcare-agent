# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from openai import AsyncOpenAI

from app.config import settings
from app.core.execution import record_usage, remaining_seconds
from app.core.vector_contract import validate_embedding_batch

_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    """直连嵌入上游(硅基流动的 bge-m3,OpenAI 兼容)。
    进程内单例:复用同一 httpx 连接池,避免每次 embed 新建客户端累积连接/fd。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)
    return _CLIENT


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("嵌入输入必须是非空文本列表")
    if not texts:
        return []
    remaining = remaining_seconds()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("本轮执行期限已到，未发起嵌入请求")
    kwargs = {"timeout": remaining} if remaining is not None else {}
    try:
        resp = await _client().embeddings.create(model=settings.embed_model, input=texts, **kwargs)
    except BaseException:
        record_usage("embedding", settings.embed_model, None, failed=True)
        raise
    usage = resp.usage
    record_usage("embedding", settings.embed_model,
                 {"input_tokens": usage.prompt_tokens, "output_tokens": 0} if usage else None)
    # A successful HTTP response can still be unsafe to attach to source rows.
    # Account for reported usage first: malformed output may have incurred cost.
    return validate_embedding_batch(resp.data, len(texts))


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.close()
        _CLIENT = None
