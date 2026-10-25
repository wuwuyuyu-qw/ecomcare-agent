# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from pydantic import BaseModel, Field
from app.schemas.chat import ChatRequest


class AgentRequest(ChatRequest):
    """与流式入口共享输入预算；用户身份由服务端登录态决定。"""


class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict


class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str


class AgentResponse(BaseModel):
    request_triage: dict | None = Field(default=None, description="原话分流及请求核查项，不包含内部推理")
    compound_plan: dict | None = Field(default=None, description="有限业务核查项，不含内部推理")
    compound_completion: dict | None = Field(default=None, description="程序观察的核查覆盖，不等于业务完成或政策适用")
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
    suggested_actions: list = Field(default_factory=list)
    logistics_decision: dict | None = Field(
        default=None, description="物流场景的程序判定、证据和工具轨迹")
    task_context: dict | None = Field(default=None, description="任务进度与停止原因，不包含内部推理")
    execution_summary: dict | None = Field(default=None, description="本轮耗时、调用与实际计量摘要")
    answer_validation: dict | None = Field(default=None, description="输出前证据作用域检查；不代表完整语义证明")
    citations: list[dict] = Field(default_factory=list, description="实际检索来源与经绑定的政策版本")
    policy_assessment: dict | None = Field(default=None, description="本地审批目录与订单条件核验；不代表任意生成文本的语义证明")
    interrupt: dict | None = Field(
        default=None,
        description="中断负载(含 type:select_order 带 orders / confirm_ticket 带 preview);无则 None")
