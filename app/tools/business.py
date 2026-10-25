# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""mock 数据源(纯函数,随机种子固定 → 同键稳定)。
ch08 起工具实现迁至 app/tools/builtin/(注册即定义),本模块只留被 graph 节点
(fetch_order)与 builtin 工具共用的快照函数,不再定义任何 @tool。"""
import random

from app.tools import synthetic_fixtures

# Validate the opt-in world once at process startup. Default demo opens no file.
_FIXTURE_INFO = synthetic_fixtures.profile_info()


def order_snapshot(order_id: str) -> dict:
    """订单快照(纯函数,随机种子固定 → 同 order_id 稳定)。query_order 工具与 fetch_order 节点同源。"""
    fixture = synthetic_fixtures.order_snapshot(order_id)
    if fixture is not None:
        return fixture
    # 独立的物流评测单:固定证据使异常/正常案例可复现,不改变课程演示单 1001/2002。
    demo = {
        "3001": {"status": "已发货", "tracking_no": "SFDEMO3001", "product": "猫粮 5kg"},
        "3002": {"status": "已签收", "tracking_no": "SFDEMO3002", "product": "智能猫砂盆"},
        "3003": {"status": "已发货", "tracking_no": "SFDEMO3003", "product": "猫爬架"},
    }
    if order_id in demo:
        return {"order_id": order_id, "amount": 199, "created_at": "2026-09-25 10:00",
                "destination_region": "CN-EAST", "fulfillment_type": "self_operated",
                **demo[order_id]}
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": f"SF{rng.randint(10**11, 10**12 - 1)}",
    }


def owns_order(user_id: str, order_id: str) -> bool:
    """这一单是不是这个人的。归属判断只有这一处,工具和节点都调它,不各判各的。

    空 user_id 一律不放行:身份是注入进来的,注入没接上就是空串,那种情况下放行等于没做校验。"""
    if not user_id or not order_id:
        return False
    if order_id in synthetic_fixtures.reserved_order_ids():
        return synthetic_fixtures.owns_order(user_id, order_id)
    return any(o["order_id"] == order_id for o in list_user_orders(user_id))


# 课程演示单。文档、curl 例子、各章验收脚本里到处写着 1001 和 2002,让每个账号名下都有
# 这两笔,例子拿来即跑。想看归属校验拦人,随便报一个别的号(如 9999)就会被挡下。
DEMO_ORDER_IDS = ("1001", "2002")
# Shared course examples above remain explicit exceptions. New logistics orders
# belong to the primary synthetic customer; the second login cannot read them.
LOGISTICS_ORDER_IDS = ("3001", "3002", "3003")
LOGISTICS_DEMO_USERS = frozenset({"customer_demo", "customer1", "u-logistics", "u-graph"})


def list_user_orders(user_id: str) -> list[dict]:
    """按 user_id 稳定列出订单(mock,不落库):课程单、明确归属的可选 fixture 和随机单。
    每笔用 order_snapshot 同源,前端选中后回填 order_id 即可 query_order。"""
    rng = random.Random(f"user_orders:{user_id}")
    ids = list(DEMO_ORDER_IDS)
    if user_id in LOGISTICS_DEMO_USERS:
        ids.extend(LOGISTICS_ORDER_IDS)
    ids.extend(synthetic_fixtures.order_ids_for(user_id))
    reserved = synthetic_fixtures.reserved_order_ids()
    for _ in range(rng.randint(2, 4)):
        oid = str(rng.randint(1000, 9999))
        if (oid not in ids and oid != "9999" and oid not in LOGISTICS_ORDER_IDS
                and oid not in reserved):
            ids.append(oid)
    out = []
    for oid in ids:
        s = order_snapshot(oid)
        out.append({"order_id": oid, "product": s["product"],
                    "status": s["status"], "amount": s["amount"]})
    return out
