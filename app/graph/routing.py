# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition
import re

from app.config import settings

# 九类意图 → 五出口(写死的分流规则,spec §3.1 / README route_by_intent;
# 返回值 = build.py 条件边映射键,单一来源不漂移)
INTENT_TO_ROUTE: dict[str, str] = {
    "投诉": "escalate",
    "闲聊": "fallback_script",
    "其他": "fallback_script",
    "商品咨询": "knowledge",
    "退款退货": "refund_flow",
    "售后": "refund_flow",
    "人工": "business",   # ch08:明确要求建工单/转人工 → 主力 Agent 走 create_ticket 确认流
    "物流": "logistics",
    "订单": "business",
}

_AFTERSALES_GOAL = re.compile(r"退款|退货|换货|保修|质保|维修")
_HANDOFF_GOAL = re.compile(r"转人工|找人工|人工客服|我要投诉|投诉客服")
_LOGISTICS_SUBJECT = re.compile(r"物流|快递|包裹|运输|签收|妥投")
_LOGISTICS_LOOKUP = re.compile(r"查一下|查询|查查|查物流|查看|帮我查|帮忙查|核对|核实|到哪|哪里|哪了|进度|状态")
_LOGISTICS_PROBLEM = re.compile(r"没(?:有)?收到|未收到|没拿到|未拿到|不是我签收|不是本人签收|非本人签收|签收异常|没更新|未更新|不动|停滞|卡住|卡了")
_POLICY_GOAL = re.compile(r"政策|规则|规定|运费|赔偿|补偿")
_LOGISTICS_EXPLANATION = re.compile(r"是什么意思|什么叫|如何查看|如何查询|在哪里投诉|哪里投诉|投诉渠道")
_HYPOTHETICAL_LOGISTICS = re.compile(r"(?:如果|假如|假设|比如|要是).{0,100}(?:怎么办|怎么处理|应该怎么|能否|可不可以)")

# Full utterance templates deliberately cover only named-product live lookups.
# A keyword in a policy, explanation, quotation or rewritten history is not an action.
_PRODUCT_NON_LOOKUP = re.compile(
    r"订单|物流|快递|运费|退款|退货|换货|售后|保修|质保|维修|政策|规则|规定|"
    r"是什么|什么意思|什么叫|如何|怎么|为什么|原理|介绍|说明|参数|规格|知识|"
    r"如果|假如|假设|比如|要是|明天|昨天|历史|以前|将来|预计|"
    r"不要|不用|无需|不需要|不想|不查|不是|别|取消|停止|只讲|只想|而非|而不是|"
    r"[“”‘’\"'<>：:，,；;。！？!?\n]"
)
_PRODUCT_LOOKUP = re.compile(
    r"(?:(?:请|麻烦)?(?:帮我|帮忙)?(?:查一下|查询|查查|查看|看看|看下|查下)\s*)?"
    r"(?P<product>[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff ._()+-]{0,63}?)\s*"
    r"(?:现在|目前|当前|今天)?\s*(?:的)?\s*"
    r"(?:还有货吗|有没有货|有货吗|还有库存吗|有没有库存|有库存吗|"
    r"库存(?:还有)?(?:多少|有多少|是多少)|售价(?:多少|是多少)|价格(?:多少|是多少)|多少钱|"
    r"(?:当前)?库存|(?:当前)?售价|(?:当前)?价格)"
)
_UNNAMED_PRODUCT = re.compile(r"^(?:这个|那个|这款|那款|它|商品|产品|东西|现在|目前|当前|今天|请|帮我|查一下|查询|查看)+$")


def product_live_lookup(user_text: str) -> bool:
    """Recognize a small set of explicit, current stock/price requests only."""
    if not isinstance(user_text, str) or len(user_text) > 160:
        return False
    text = user_text.strip().rstrip("？?。！! ")
    if _PRODUCT_NON_LOOKUP.search(text):
        return False
    match = _PRODUCT_LOOKUP.fullmatch(text)
    if not match:
        return False
    product = match["product"].strip()
    return bool(product and not product.isdigit() and not _UNNAMED_PRODUCT.fullmatch(product))


def logistics_route_override(user_text: str) -> str | None:
    """Narrow routing guard grounded in this turn's original words.

    This selects a readonly investigator, not an abnormality verdict. Refund,
    warranty and policy-only requests retain model routing. Neither a rewritten
    query nor a persisted topic can manufacture a current logistics request.
    """
    if (not isinstance(user_text, str) or _AFTERSALES_GOAL.search(user_text)
            or _HANDOFF_GOAL.search(user_text)):
        return None
    if not _LOGISTICS_SUBJECT.search(user_text):
        return None
    if _HYPOTHETICAL_LOGISTICS.search(user_text):
        return None
    explicit_order = re.search(r"订单(?:号)?\s*[:：#]?\s*\d{4,}", user_text)
    if _LOGISTICS_EXPLANATION.search(user_text) and not explicit_order:
        return None
    if (_POLICY_GOAL.search(user_text)
            and not explicit_order):
        return None
    if _LOGISTICS_PROBLEM.search(user_text):
        return "explicit_logistics_problem"
    if _POLICY_GOAL.search(user_text):
        return None
    if _LOGISTICS_LOOKUP.search(user_text):
        return "explicit_logistics_lookup"
    return None


def route_by_intent(state) -> str:
    """按意图分流;未知意图保守归 business(让 Agent 自己应对)。"""
    if settings.compound_agent_enabled and settings.compound_joint_triage_enabled:
        from app.graph import triage
        if triage.active(state):
            return triage.route(state)
    if settings.compound_agent_enabled and state.get("intent") in {"物流", "订单", "售后", "退款退货"}:
        return "compound"
    if state.get("intent") == "商品咨询":
        latest = next((m for m in reversed(state.get("messages", []))
                       if isinstance(m, HumanMessage)), None)
        if latest is not None and product_live_lookup(latest.content):
            return "business"
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")


def confidence_gate(state) -> str:
    """知识路生成前证据闸:强放行、弱兜底。"""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state) -> str:
    """ReAct 停止条件:无 tool_calls 就收敛,步数封顶则强停,否则继续。

    只用步数封顶,不在这里卡 token。防打转靠的是步数:一步一次模型调用,数得清、好解释,
    各家 Agent SDK 给的也是 max_turns / max_iterations 这类步数上限。
    token 花销是另一件事——它管的是烧钱和撑爆上下文,不是死循环,阈值要按真实用量分布标定,
    归第 9 章的成本控制一起讲。混在这里卡,一次估偏就把工具链整条掐断,而且掐在模型
    已经决定调工具之后,用户只看到半句「我这就去查」。"""
    last = state["messages"][-1]
    # Upstream LangGraph MIT component; keep our business budgets outside it.
    has_tool_calls = isinstance(last, AIMessage) and tools_condition(state) == "tools"
    task = state.get("task_context") or {}
    if task.get("termination_reason") in {"tool_budget_exceeded", "repeated_call_limit", "step_budget_exceeded"}:
        return "limit"
    if not has_tool_calls:
        from app.graph.compound import active
        return "compound_stop" if active(state) else "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "limit"
    return "continue"
