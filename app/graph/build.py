# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from langgraph.graph import END, START, StateGraph

from app.graph import compound, logistics, nodes
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def _builder() -> StateGraph:
    b = StateGraph(ConversationState)
    b.add_node("task_entry", nodes.task_entry)
    b.add_node("budget_reply", nodes.budget_reply)
    b.add_node("evidence_guard", nodes.evidence_guard)
    # 节点
    b.add_node("resolve_reference", nodes.resolve_reference)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    b.add_node("confidence_check", nodes.confidence_check)  # 实体节点(trace 门控);判断在其后条件边
    b.add_node("main_agent", nodes.main_agent)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("compound_prepare", compound.prepare)
    b.add_node("compound_finalize", compound.finalize)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("script_reply", nodes.script_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_node("retrieve_policy", nodes.retrieve_policy)
    b.add_node("log", nodes.log_node)
    b.add_node("logistics_extract", logistics.extract_logistics)
    b.add_node("logistics_investigate", logistics.investigate_logistics)

    # 骨架:消解 → 意图 → 分流
    b.add_edge(START, "task_entry")
    b.add_conditional_edges("task_entry", lambda state: "cancelled" if
        (state.get("task_context") or {}).get("cancelled") else "continue",
        {"cancelled": "log", "continue": "resolve_reference"})
    b.add_edge("resolve_reference", "classify_intent")
    b.add_conditional_edges("classify_intent", route_by_intent, {
        "escalate": "complaint_reply",
        "fallback_script": "script_reply",
        "knowledge": "retrieve_knowledge",
        "refund_flow": "fetch_order",
        "business": "main_agent",
        "logistics": "logistics_extract",
        "compound": "compound_prepare",
    })
    # 售后只读链:取单 → 政策检索与充分性检查 → 共用证据门 → 条件说明/拒答
    b.add_edge("fetch_order", "retrieve_policy")
    b.add_edge("retrieve_policy", "confidence_check")
    b.add_conditional_edges("logistics_extract", logistics.route_after_extract, {
        "investigate": "logistics_investigate", "done": "log",
    })
    b.add_edge("logistics_investigate", "log")
    # 知识路:检索 → 生成前证据闸
    b.add_edge("retrieve_knowledge", "confidence_check")
    b.add_conditional_edges("confidence_check", confidence_gate, {
        "strong": "main_agent",
        "weak": "fallback_reply",
    })
    # 主力 Agent ReAct 环
    b.add_conditional_edges("main_agent", should_continue, {
        "continue": "agent_tools",
        "stop": "evidence_guard",
        "compound_stop": "compound_finalize",
        "limit": "budget_reply",
    })
    b.add_edge("agent_tools", "main_agent")
    b.add_conditional_edges("compound_prepare", compound.route_after_prepare,
                            {"continue": "main_agent", "done": "log"})
    b.add_edge("compound_finalize", "evidence_guard")
    # 确定性出口 → 日志 → END
    b.add_edge("complaint_reply", "log")
    b.add_edge("script_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("budget_reply", "evidence_guard")
    b.add_edge("evidence_guard", "log")
    b.add_edge("log", END)
    return b


def build_graph(checkpointer=None):
    return _builder().compile(checkpointer=checkpointer)
