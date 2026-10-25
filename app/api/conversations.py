# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch07 验收配套:会话列表 + 历史消息回载(只读,前端多会话切换用)。"""
from fastapi import APIRouter, Depends

from app.db import repository
from app.core.auth import Principal, require_principal, require_conversation_owner

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(principal: Principal = Depends(require_principal)):
    items = await repository.list_conversations(principal.user_id)
    return {"items": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: int, principal: Principal = Depends(require_principal)):
    await require_conversation_owner(conversation_id, principal)
    msgs = await repository.list_dialog_messages(conversation_id)
    return {"items": [
        {"role": m.role, "content": m.content or "",
         "created_at": m.created_at.isoformat() if m.created_at else None}
        for m in msgs
    ]}
