"""Four model-free HTTP permission cases using real routes/auth and private SQL.

This deliberately mounts only production auth/conversation/action/jobs routers,
without the full application's Graph/Milvus startup. A Graph-entry tripwire and
a zero-request provider budget make accidental model execution fail closed.
Scores are HTTP authorization boundary probes, never Agent task success.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))


def write(path,value):
    with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def service(args):
    from app.evaluation.stage1 import install_http_budget
    install_http_budget(args.output/"provider-budget.sqlite",set())
    from fastapi import FastAPI
    from app.graph import runtime
    def hit():
        with (args.output/"graph-entry-tripwire.jsonl").open("a") as stream:stream.write('{"unexpected_graph_entry":true}\n')
        raise RuntimeError("model-free permission probe must reject before Graph entry")
    async def no_turn(*a,**k):hit()
    async def no_resume(*a,**k):
        hit()
        yield {}
    runtime.run_turn=no_turn;runtime.stream_resume=no_resume
    from app.api import auth,conversations,actions,jobs
    app=FastAPI()
    for module in (auth,conversations,actions,jobs):app.include_router(module.router)
    @app.get("/healthz")
    async def health():return {"scope":"production permission routes; Graph not started"}
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=args.port,log_level="warning")


def run(args):
    from dotenv import load_dotenv
    from eval_stage1_baseline import prepare_database
    from run_agent_diagnostics import OwnedProcesses,Redactor
    from app.evaluation.stage1 import initialize_budget,budget_snapshot
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    load_dotenv(args.env_file.resolve(),override=False)
    os.chdir(args.output)
    os.environ.update(AUTH_DB_PATH=str(args.output/"private-auth.sqlite"),CHECKPOINTER_DB_PATH=str(args.output/"private-checkpoints.sqlite"),
        LANGFUSE_PUBLIC_KEY="",LANGFUSE_SECRET_KEY="",LANGFUSE_BASE_URL="",LANGSMITH_TRACING="false",LANGCHAIN_TRACING_V2="false",
        PYTHONPATH=str(ROOT),PYTHONUTF8="1",PYTHONUNBUFFERED="1")
    from app.config import settings
    redactor=Redactor(dict(os.environ))
    asyncio.run(prepare_database(args.output,settings,redactor,copy_knowledge=False))
    from app.core.auth import create_demo_account
    passwords={u:secrets.token_urlsafe(24) for u in ("customer_demo","customer_other","admin_probe")}
    for user,password in passwords.items():
        redactor.add(password)
        create_demo_account(user,password,role="admin" if user=="admin_probe" else "customer",path=settings.auth_db_path)
    async def seed():
        import asyncmy
        from sqlalchemy.engine import make_url
        u=make_url(os.environ["DATABASE_URL"])
        conn=await asyncmy.connect(host=u.host,port=u.port or 3306,user=u.username,password=u.password,db=u.database)
        try:
            async with conn.cursor() as cur:
                ids={}
                for user in ("customer_demo","customer_other"):
                    await cur.execute("INSERT INTO conversations(user_id) VALUES (%s)",(user,));ids[user]=cur.lastrowid
                    await cur.execute("INSERT INTO messages(conversation_id,role,content) VALUES (%s,'user',%s)",(ids[user],"SYNTHETIC_PRIVATE_"+user))
            await conn.commit()
            return ids
        finally:conn.close()
    ids=asyncio.run(seed())
    initialize_budget(args.output/"provider-budget.sqlite",chat=0,retrieval=0,output_tokens=0)
    owned=OwnedProcesses(args.output,redactor);rows=[];tokens={}
    try:
        import httpx
        owned.start("permission-api",[sys.executable,str(Path(__file__).resolve()),"service","--output",str(args.output),"--port",str(args.port)],dict(os.environ))
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}",timeout=12,follow_redirects=False,trust_env=False) as client:
            for _ in range(80):
                try:
                    if client.get("/healthz").status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.25)
            else:raise RuntimeError("permission API unavailable")
            for user,password in passwords.items():
                body={"username":user,"password":password}
                if user=="customer_demo":body.update(user_id="customer_other",role="admin")
                response=client.post("/api/auth/login",json=body);response.raise_for_status()
                tokens[user]=response.json()["access_token"];redactor.add(tokens[user])
            def request(method,path,user,body=None):
                return client.request(method,path,headers={"Authorization":"Bearer "+tokens[user]},json=body)
            def case(key,checks,observed):
                row={"id":key,"checks":checks,"success":all(checks.values()),"observed":observed,
                     "measurement":"HTTP authentication/authorization boundary; no model or Graph"}
                rows.append(row);write(args.output/(key+".json"),row)
            me=request("GET","/api/auth/me?user_id=customer_other&role=admin","customer_demo")
            principal=me.json()
            resume=request("POST","/api/actions/resume","customer_other",{"conversation_id":ids['customer_demo'],"order_id":"3003"})
            invalid=request("POST","/api/actions/resume","customer_demo",{"conversation_id":ids['customer_demo'],"confirmed":True})
            malformed=request("POST","/api/actions/resume","customer_demo",{"conversation_id":ids['customer_demo'],"order_id":"3003","confirmed":True})
            case("task-11-07",{"server_identity_not_client_claim":me.status_code==200 and principal["user_id"]=="customer_demo" and principal['role']=="customer",
                "foreign_resume_denied":resume.status_code==404,"write_confirmation_rejected":invalid.status_code==409,"ambiguous_resume_rejected":malformed.status_code==400},
                {"me_identity":principal["user_id"],"me_role":principal["role"],"foreign_resume_status":resume.status_code,"confirmed_resume_status":invalid.status_code,"ambiguous_resume_status":malformed.status_code})
            foreign=request("GET",f"/api/conversations/{ids['customer_demo']}/messages","customer_other")
            own=request("GET",f"/api/conversations/{ids['customer_demo']}/messages","customer_demo")
            other=request("GET",f"/api/conversations/{ids['customer_other']}/messages","customer_other")
            missing=request("GET","/api/conversations/9223372036854770000/messages","customer_other")
            case("task-11-08",{"own_history_positive":own.status_code==200 and "SYNTHETIC_PRIVATE_customer_demo" in own.text,
                "other_own_positive":other.status_code==200,"cross_user_denied":foreign.status_code==404,
                "no_private_marker_leak":"SYNTHETIC_PRIVATE_customer_demo" not in foreign.text,"same_as_missing":missing.status_code==foreign.status_code},
                {"owner_status":own.status_code,"other_owner_status":other.status_code,"foreign_status":foreign.status_code,"missing_status":missing.status_code})
            denied=request("GET","/api/jobs","customer_demo");allowed=request("GET","/api/jobs","admin_probe")
            case("task-11-09",{"forged_role_did_not_escalate":denied.status_code==403 and principal["role"]=="customer","admin_positive_control":allowed.status_code==200},
                {"customer_jobs_status":denied.status_code,"admin_jobs_status":allowed.status_code})
            with sqlite3.connect(settings.auth_db_path) as conn:
                conn.execute("UPDATE demo_sessions SET expires_at=? WHERE token_digest=?",(int(time.time())-5,hashlib.sha256(tokens['customer_other'].encode()).hexdigest()))
            expired=request("GET","/api/auth/me","customer_other")
            expired_resume=request("POST","/api/actions/resume","customer_other",{"conversation_id":ids['customer_other'],"order_id":"3003"})
            active=request("GET","/api/auth/me","customer_demo")
            case("task-11-10",{"expired_identity_rejected":expired.status_code==401,"expired_resume_rejected":expired_resume.status_code==401,"control_token_still_active":active.status_code==200},
                {"expired_me_status":expired.status_code,"expired_resume_status":expired_resume.status_code,"active_control_status":active.status_code})
            for user in tokens:request("POST","/api/auth/logout",user)
    finally:
        write(args.output/"process-cleanup.json",owned.close())
        write(args.output/"provider-budget.json",budget_snapshot(args.output/"provider-budget.sqlite"))
        write(args.output/"result.json",{"schema":"model-free-session-boundaries-v1","requested":4,"executed":len(rows),"pass":sum(r["success"] for r in rows),"fail":sum(not r["success"] for r in rows),
            "rows":rows,"graph_started":False,"unexpected_graph_entries":(args.output/"graph-entry-tripwire.jsonl").exists(),
            "separate_from_agent_task_success":True,"fixture":"new private accounts/SQL conversations/auth SQLite; actual production route dependencies",
            "finished_at":datetime.now(timezone.utc).isoformat()})
    if len(rows)!=4 or not all(r['success'] for r in rows):raise RuntimeError("permission boundary check failed")
    print(json.dumps({"measurement":"model-free permission probes","pass":4,"requested":4,"model_calls":0}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("run","service"));p.add_argument("--output",type=Path,required=True);p.add_argument("--env-file",type=Path);p.add_argument("--port",type=int,default=18600)
    a=p.parse_args()
    try:service(a) if a.mode=="service" else run(a)
    except Exception as exc:
        print(json.dumps({"status":"stopped","error_type":type(exc).__name__}));raise SystemExit(1) from None
