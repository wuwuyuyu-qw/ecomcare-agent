# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    conversation_id: int = Field(gt=0)
    rating: Literal["up", "down"]
    question: str = Field(min_length=1, max_length=12000)  # 被评价那轮的用户原话


class FeedbackResponse(BaseModel):
    ok: bool = True
    pooled: bool           # down 且成功落池才为 True
