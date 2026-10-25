# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""工具注册中心。MCP 工具仅按本地 (server, name) 允许表暴露给模型。"""
import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.tools import contracts

logger = logging.getLogger(__name__)

# 不能根据 MCP Server 自报的描述或工具名猜测是否只读。新工具须审查后显式登记。
WRITE_TOOLS: set[str] = {"create_ticket"}
MCP_READ_TOOLS: dict[str, frozenset[str]] = {
    "logistics": frozenset({"query_logistics"}),
    "aftersales": frozenset({"query_warranty", "query_return_status"}),
}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict            # 参数 JSON Schema(校验用;模型可见口径,不含注入参数)
    tool: BaseTool
    permission: str              # "read" | "write" | "deny"
    source: str                  # "builtin" | "mcp"
    mcp_server: str | None = None
    timeout: float | None = None            # None → engine 按来源取默认
    max_retries: int | None = None          # None → engine 按默认;0 = 不重试(如 query_faq RAG 管线)
    inject_conversation: bool = False       # 执行前注入 conversation_id
    inject_user_id: bool = False            # 执行前注入 user_id(身份从可信通道来,不许模型填)
    format_result: Callable[[dict], dict] | None = None  # 挑字段/枚举翻人话,None 透传
    result_contract: str | None = None      # 本地返回契约,仅说明用途;engine仍按工具名重新查验


_BUILTIN: dict[str, ToolSpec] = {}
_scanned = False


def permission_for(name: str, *, source: str = "builtin", mcp_server: str | None = None) -> str:
    """Fail closed for remote tools; local builtin modules remain an explicit trusted code boundary."""
    if source == "mcp":
        return "read" if name in MCP_READ_TOOLS.get(mcp_server or "", ()) else "deny"
    if source != "builtin":
        return "deny"
    return "write" if name in WRITE_TOOLS else "read"


def _json_schema_of(tool: BaseTool) -> dict:
    """三样齐的 schema 口径:MCP 工具 args_schema 本就是 dict;内置 pydantic 模型用
    tool_call_schema(排除 InjectedToolArg,模型可见口径)转 JSON Schema。"""
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict):
        return raw
    tcs = getattr(tool, "tool_call_schema", None) or raw
    return tcs.model_json_schema()


def spec_from_langchain_tool(tool: BaseTool, *, source: str, mcp_server: str | None = None,
                             timeout: float | None = None, max_retries: int | None = None,
                             inject_conversation: bool = False, inject_user_id: bool = False,
                             format_result: Callable | None = None) -> ToolSpec:
    return ToolSpec(name=tool.name, description=tool.description or "",
                    json_schema=_json_schema_of(tool), tool=tool,
                    permission=permission_for(tool.name, source=source, mcp_server=mcp_server),
                    source=source, mcp_server=mcp_server,
                    timeout=timeout, max_retries=max_retries,
                    inject_conversation=inject_conversation, inject_user_id=inject_user_id,
                    format_result=format_result,
                    result_contract=contracts.CONTRACT_VERSION if contracts.has_contract(tool.name) else None)


def register(spec: ToolSpec) -> None:
    if spec.name in _BUILTIN:
        logger.warning("工具重名,丢弃后注册者 name=%s(先到者保留)", spec.name)
        return
    _BUILTIN[spec.name] = spec


def scan_builtin() -> None:
    """服务启动(lifespan)时调用:导入 builtin/ 包全部模块,模块 import 即注册。幂等。"""
    global _scanned
    if _scanned:
        return
    _scanned = True
    from app.tools import builtin as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        try:
            importlib.import_module(f"{pkg.__name__}.{m.name}")
        except Exception:  # noqa: BLE001 坏文件跳过告警,不拖垮其余工具注册(丢文件即插即用的稳健性)
            logger.exception("内置工具模块导入失败,跳过 module=%s", m.name)
    logger.info("内置工具注册完成:%s", sorted(_BUILTIN))


def builtin_specs() -> list[ToolSpec]:
    scan_builtin()
    return list(_BUILTIN.values())


def get_builtin_spec(name: str) -> ToolSpec | None:
    scan_builtin()
    return _BUILTIN.get(name)


async def get_all_specs() -> list[ToolSpec]:
    """内置 + MCP 现拉合并;重名 builtin 优先、后到丢弃告警(本章无重名:内置 query_logistics 已下线)。

    返回顺序必须稳定,别改成 set 或加 sorted:tools 定义跟 system 一样在模型请求的可缓存
    前缀里,顺序抖一下前缀就整段 miss。现在的顺序由三件事保证——scan_builtin 走的 pkgutil
    内部有 filenames.sort()、_BUILTIN 与 merged 都是 dict(保插入序)、_connections() 是字面量
    dict。真会让它变的是 MCP Server 不可达跳过或 Server 侧热加工具,那是「现问现拿」的固有
    代价,排序治不了。"""
    from app.tools import mcp_client   # 延迟导入避免环(mcp_client 也 import registry)
    merged: dict[str, ToolSpec] = {s.name: s for s in builtin_specs()}
    for s in await mcp_client.fetch_mcp_specs():
        if permission_for(s.name, source="mcp", mcp_server=s.mcp_server) == "deny":
            logger.warning("MCP 工具未在本地允许表中,不暴露 name=%s server=%s", s.name, s.mcp_server)
            continue
        if not contracts.has_contract(s.name):
            logger.warning("MCP 工具缺少本地返回契约,不暴露 name=%s server=%s", s.name, s.mcp_server)
            continue
        if s.name in merged:
            logger.warning("MCP 工具与已注册工具重名,丢弃 name=%s server=%s", s.name, s.mcp_server)
            continue
        merged[s.name] = s
    return list(merged.values())
