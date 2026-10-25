"""仅根据已核验的工具证据判定物流异常，不读模型输出里的事实字段。"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class LogisticsDecision:
    verdict: str             # no_ticket | offer_ticket | human_review
    issue_type: str          # normal | stalled_48h | signed_not_received | signed_dispute | manual_review
    reason: str


def delivery_claim_type(user_text: str) -> str:
    """Separate the user's receipt assertion from the tool's delivery status.

    A later explicit receipt update supersedes an earlier missing/disputed
    statement. These narrow phrases are a conservative helper, not general
    negation understanding. Ambiguous model-only claims still need human review.
    """
    text = user_text.lower()
    events: list[tuple[int, str]] = []
    missing = re.compile(r"(?:没(?:有)?(?:收到|拿到)|未(?:收到|拿到))(?!\s*(?:短信|通知|消息|邮件|赔偿|补偿|退款))|not received|never received")
    disputed = re.compile(r"不是我签收|不是本人签收|非本人签收|签收异常")
    received = re.compile(
        r"(?:已经|已|确实|确认)(?:收到|拿到)(?!\s*(?:短信|通知|消息|邮件|赔偿|补偿|退款))"
        r"|(?:我|我们)(?:已经|已)?(?:收到|拿到)了(?!\s*(?:短信|通知|消息|邮件|赔偿|补偿|退款))")
    for pattern, kind in ((missing, "signed_not_received"), (disputed, "signed_dispute"),
                          (received, "received")):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 6):match.start()]
            sentence_prefix = re.split(r"[。！？!?；;\n]", text[:match.start()])[-1]
            if re.search(r"(?:如果|假如|假设|比如|要是).{0,100}$", sentence_prefix):
                continue
            if re.search(r"(?:不是|并非|不代表|没有说|没说|并未|尚未|未|没|如果|假如|假设|要是|是否|以为)(?:我|我们|你|您)?\s*$", prefix):
                continue
            # '收到了签收短信' is a notification, not physical receipt. Check
            # the remaining local clause, including intervening particles.
            suffix = re.split(r"[，,。！？!?；;\n]", text[match.end():], maxsplit=1)[0][:20]
            if kind != "signed_dispute" and re.search(r"短信|通知|消息|邮件|验证码|赔偿|补偿|退款", suffix):
                continue
            events.append((match.start(), kind))
    if events:
        return max(events, key=lambda item: item[0])[1]
    implicit = (any(s in text for s in ("签收", "妥投", "送达"))
                and any(s in text for s in ("没有货", "收件处没有", "实际情况不一致")))
    return "signed_not_received" if implicit else ""


def decide_logistics(*, claim_type: str, user_text: str, order: dict,
                     logistics: dict) -> LogisticsDecision:
    """可离线评测的纯守卫。模型只提供用户诉求类别；状态和时长必须来自工具。"""
    tracking = str(order.get("tracking_no") or "")
    if not tracking or logistics.get("tracking_no") != tracking:
        return LogisticsDecision("human_review", "manual_review", "订单与物流单号无法对应，需人工核查")

    status = logistics.get("status_code") or logistics.get("status")
    grounded_claim = delivery_claim_type(user_text)
    if status in {"DELIVERED", "已签收"}:
        if grounded_claim == "received":
            return LogisticsDecision("no_ticket", "normal", "物流显示已签收，用户最新表述为已收到")
        if grounded_claim == "signed_dispute":
            return LogisticsDecision("offer_ticket", "signed_dispute",
                                     "物流显示签收，用户对签收人提出争议，需人工核实签收凭证")
        if claim_type in {"signed_not_received", "signed_dispute"} or grounded_claim == "signed_not_received":
            if grounded_claim != "signed_not_received":
                return LogisticsDecision("human_review", "manual_review",
                                         "模型识别到收件争议，但用户原话缺少明确表述，需人工核对")
            return LogisticsDecision("offer_ticket", "signed_not_received",
                                     "物流显示签收，但用户明确表示未收到")
        return LogisticsDecision("no_ticket", "normal", "物流显示已签收，当前未发现需建单的异常")

    if status in {"IN_TRANSIT", "运输中", "PICKED_UP", "已揽件", "DELIVERING", "派送中"}:
        hours = logistics.get("stalled_hours")
        if (not isinstance(hours, (int, float)) or isinstance(hours, bool)
                or not math.isfinite(hours) or hours < 0):
            return LogisticsDecision("human_review", "manual_review",
                                     "缺少可信的最近物流更新时间，不能判断是否停滞")
        if hours >= 48:
            return LogisticsDecision("offer_ticket", "stalled_48h",
                                     f"物流连续 {hours:g} 小时没有新记录，达到演示异常阈值 48 小时")
        if claim_type in {"signed_not_received", "signed_dispute"} and grounded_claim != "received":
            return LogisticsDecision("human_review", "manual_review",
                                     "用户描述与物流状态不一致，需人工核对")
        return LogisticsDecision("no_ticket", "normal", f"最近 {hours:g} 小时内有物流更新")

    return LogisticsDecision("human_review", "manual_review", "物流状态缺失或无法识别，不能自动判断")
