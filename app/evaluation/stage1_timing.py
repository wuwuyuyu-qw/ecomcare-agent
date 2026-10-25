"""Evaluation-process-only timing observers; no prompts, answers or keys logged."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import time
import uuid

_PARENT=ContextVar("evaluation_parent_span",default=None)


def install_observers(path):
    from app.core.execution import current_execution
    def start(kind,name):
        scope=current_execution()
        return {"span_id":uuid.uuid4().hex,"parent_id":_PARENT.get(),"kind":kind,"name":name,
            "request_id":scope.request_id if scope else None,"task_id":scope.task_id if scope else None,
            "started_at":datetime.now(timezone.utc).isoformat()}
    def finish(row,started,error=None,**fields):
        item={**row,**fields,"elapsed_ms":round((time.perf_counter()-started)*1000,3),
            "ended_at":datetime.now(timezone.utc).isoformat(),"error_type":type(error).__name__ if error else None}
        with path.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(item,ensure_ascii=False)+"\n")
    def instrument(module,name,kind):
        original=getattr(module,name)
        @wraps(original)
        async def measured(*args,**kwargs):
            row=start(kind,name);begin=time.perf_counter();token=_PARENT.set(row["span_id"])
            error=None
            try:return await original(*args,**kwargs)
            except BaseException as exc:error=exc;raise
            finally:
                finish(row,begin,error);_PARENT.reset(token)
        setattr(module,name,measured)
    # Install before app.main constructs its real graph. These decorators return
    # the original values/errors and do not change routing, guards or models.
    from app.graph import nodes,logistics
    from app.core import retrieval
    from app.tools import engine
    for name in ("task_entry","resolve_reference","classify_intent","retrieve_knowledge","confidence_check",
                 "main_agent","agent_tools","evidence_guard","log_node","fetch_order","retrieve_policy","fallback_reply"):
        instrument(nodes,name,"graph_node")
    for name in ("extract_logistics","investigate_logistics"):
        instrument(logistics,name,"graph_node")
    instrument(retrieval,"search_knowledge","rag_inclusive")
    instrument(engine,"execute_tool_call","tool_inclusive")
    import httpx
    original=httpx.AsyncClient.send
    async def sending(self,request,*args,**kwargs):
        if request.url.host in {"127.0.0.1","localhost","::1"}:
            return await original(self,request,*args,**kwargs)
        kind="provider_chat" if request.url.path.endswith("/chat/completions") else "provider_retrieval"
        row=start(kind,"httpx_send");begin=time.perf_counter()
        try:
            response=await original(self,request,*args,**kwargs)
        except BaseException as exc:
            finish(row,begin,exc,status=None,measurement="send_exception")
            raise
        header_ms=round((time.perf_counter()-begin)*1000,3)
        if not kwargs.get("stream",False):
            finish(row,begin,status=response.status_code,headers_ms=header_ms,measurement="complete_nonstream_response")
            return response
        close=response.aclose
        finished=False
        async def closing():
            nonlocal finished
            try:await close()
            finally:
                if not finished:
                    finished=True
                    finish(row,begin,status=response.status_code,headers_ms=header_ms,measurement="stream_lifetime_until_close_not_provider_compute")
        response.aclose=closing
        return response
    httpx.AsyncClient.send=sending
