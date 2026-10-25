# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.auth import Principal, require_principal, require_conversation_owner
from app.graph import runtime
from app.schemas.actions import (
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ResumeRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest, principal: Principal = Depends(require_principal)) -> CreateTicketResponse:
    """旧按钮不能绕过 Agent 确认、权限与统一幂等写入边界。"""
    await require_conversation_owner(req.conversation_id, principal)
    raise HTTPException(status_code=410, detail="此直接建单入口已关闭；当前演示提供只读调查与人工交接，写入服务完善后再开放")


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(req: CreateRefundRequest, principal: Principal = Depends(require_principal)) -> CreateRefundResponse:
    """不把客户端表单当成退款授权；后续统一服务实现后再开放。"""
    await require_conversation_owner(req.conversation_id, principal)
    raise HTTPException(status_code=410, detail="此直接退款入口已关闭；当前演示不执行退款，写入服务完善后再开放")


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest, principal: Principal = Depends(require_principal)):
    """Only an owned-order selector can resume the readonly graph (SSE)."""
    await require_conversation_owner(req.conversation_id, principal)
    if (req.order_id is None) == (req.confirmed is None):
        raise HTTPException(status_code=400, detail="order_id 与 confirmed 必须且只能传一个")
    if req.confirmed is not None:
        raise HTTPException(status_code=409, detail="当前演示只支持订单选择续跑，尚未开放写入确认")
    resume_value = req.order_id

    async def event_stream() -> AsyncIterator[str]:
        try:
            async with aclosing(runtime.stream_resume(req.conversation_id, resume_value, user_id=principal.user_id)) as events:
                async for ev in events:
                    if ev["type"] == "tool":
                        yield _sse({"event": "tool", "name": ev["name"], "ok": ev.get("ok")})
                    elif ev["type"] == "delta":
                        yield _sse({"delta": ev["text"]})
                    elif ev["type"] == "citations":
                        yield _sse({"event": "citations", "items": ev["items"]})
                    elif ev["type"] == "logistics_decision":
                        yield _sse({"event": "logistics_decision", **{k: v for k, v in ev.items() if k != "type"}})
                    elif ev["type"] == "actions":
                        yield _sse({"event": "actions", "items": ev["items"]})
                    elif ev["type"] in ("termination", "execution_summary"):
                        yield _sse({"event": ev["type"], **{k: v for k, v in ev.items() if k != "type"}})
                    elif ev["type"] == "interrupt":
                        yield _sse({"event": "interrupt",
                                    **{k: v for k, v in ev.items() if k != "type"}})
                    elif ev["type"] == "done":
                        yield _sse({"event": "done", **{k: v for k, v in ev.items() if k != "type"}})
        except runtime.ConversationNotFound:
            yield "event: error\n"
            yield _sse({"message": "会话不存在"})
            return
        except runtime.ConversationBusy:
            yield "event: error\n"
            yield _sse({"message": "该会话正在处理，请等待当前回复结束"})
            return
        except runtime.InvalidResume:
            yield "event: error\n"
            yield _sse({"message": "恢复内容与当前暂停步骤不匹配，请重新选择自己的订单"})
            return
        except runtime.TurnExecutionTimeout:
            yield "event: error\n"
            yield _sse({"message": "本轮续跑尚未完成，已停止自动判断；请稍后重试"})
            return
        except Exception:
            logger.exception("续跑失败 conv=%s", req.conversation_id)
            yield "event: error\n"
            yield _sse({"message": "上游暂时不可用,请稍后重试"})
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
