# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.core.auth import Principal, require_principal, require_conversation_owner
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> Iterator[str]:
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/chat")
async def chat(req: ChatRequest, principal: Principal = Depends(require_principal)):
    if req.conversation_id is not None:
        await require_conversation_owner(req.conversation_id, principal)
    async def event_stream() -> AsyncIterator[str]:
        try:
            async with aclosing(runtime.stream_turn(principal.user_id, req.message, req.conversation_id)) as events:
                async for ev in events:
                    if ev["type"] == "tool":
                        yield _sse({"event": "tool", "name": ev["name"], "ok": ev.get("ok")})
                    elif ev["type"] == "delta":
                        yield _sse({"delta": ev["text"]})
                    elif ev["type"] == "citations":
                        yield _sse({"event": "citations", "items": ev["items"]})
                    elif ev["type"] == "logistics_decision":
                        yield _sse({"event": "logistics_decision",
                                    **{k: v for k, v in ev.items() if k != "type"}})
                    elif ev["type"] == "interrupt":
                        # 中断负载保留订单选择信息，恢复时再次核验登录身份。
                        yield _sse({"event": "interrupt",
                                    **{k: v for k, v in ev.items() if k != "type"}})
                    elif ev["type"] == "actions":
                        yield _sse({"event": "actions", "items": ev["items"]})
                    elif ev["type"] in ("termination", "execution_summary"):
                        yield _sse({"event": ev["type"], **{k: v for k, v in ev.items() if k != "type"}})
                    elif ev["type"] == "done":
                        yield _sse({"event": "done", **{k: v for k, v in ev.items() if k != "type"}})
        except runtime.ConversationNotFound:
            for f in _sse_error("会话不存在"):
                yield f
            return
        except runtime.ConversationBusy:
            for f in _sse_error("该会话正在处理，请等待当前回复结束"):
                yield f
            return
        except runtime.TurnExecutionTimeout:
            for f in _sse_error("本轮查询尚未完成，已停止自动判断；请稍后重试"):
                yield f
            return
        except SQLAlchemyError:
            logger.exception("数据库错误 user_id=%s", principal.user_id)
            for f in _sse_error("数据库暂时不可用,请稍后重试"):
                yield f
            return
        except Exception:
            logger.exception("图编排失败 user_id=%s", principal.user_id)
            for f in _sse_error("上游模型暂时不可用,请稍后重试"):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
