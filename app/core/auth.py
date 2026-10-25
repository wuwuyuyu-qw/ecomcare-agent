"""Local demo identity boundary: password hashes and revocable, short-lived sessions.

No account is provisioned at application startup. Use scripts/init_demo_auth.py.
Tokens are opaque random values; only their SHA-256 digests are persisted.
This is deliberately a single-host demo, not an identity provider for deployment.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

Role = Literal["customer", "admin"]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BEARER = HTTPBearer(auto_error=False)
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_LOGIN_WINDOW_SECONDS = 300
_LOGIN_MAX_FAILURES = 10
_LOGIN_SLOT = asyncio.Semaphore(1)


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Role
    token_digest: str = ""


def _auth_path(path: str | Path | None = None) -> Path:
    if path is None:
        from app.config import settings
        path = getattr(settings, "auth_db_path", "data/demo_auth.sqlite")
    result = Path(path).expanduser()
    return result if result.is_absolute() else _PROJECT_ROOT / result


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    target = _auth_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    conn = sqlite3.connect(target, timeout=10)
    if not existed and os.name != "nt":
        target.chmod(0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS demo_accounts (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('customer', 'admin')),
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS demo_sessions (
            token_digest TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES demo_accounts(user_id),
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS demo_login_failures (
            account_key TEXT NOT NULL,
            client_key TEXT NOT NULL,
            failed_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS demo_login_failures_account
            ON demo_login_failures(account_key, failed_at);
        CREATE INDEX IF NOT EXISTS demo_login_failures_client
            ON demo_login_failures(client_key, failed_at);
    """)
    return conn


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N,
                          r=_SCRYPT_R, p=_SCRYPT_P, dklen=64, maxmem=64 * 1024**2)


def hash_password(password: str) -> str:
    if not 12 <= len(password) <= 256:
        raise ValueError("演示密码长度必须为 12-256 个字符")
    salt = secrets.token_bytes(16)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${_derive(password, salt).hex()}"


def _verify_password(password: str, encoded: str | None) -> bool:
    # Unknown accounts still do one scrypt operation and get the same HTTP error.
    if encoded is None:
        _derive(password, b"mewhelp-dummy-v1")
        return False
    try:
        kind, n, r, p, salt_hex, digest_hex = encoded.split("$")
        if (kind, int(n), int(r), int(p)) != ("scrypt", _SCRYPT_N, _SCRYPT_R, _SCRYPT_P):
            return False
        salt, digest = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
        if len(salt) != 16 or len(digest) != 64:
            return False
        return hmac.compare_digest(_derive(password, salt), digest)
    except (ValueError, TypeError):
        return False


def create_demo_account(user_id: str, password: str, role: Role = "customer",
                        *, path: str | Path | None = None) -> None:
    """Create only. Existing identities and their sessions are never overwritten."""
    if not user_id or len(user_id) > 64 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in user_id):
        raise ValueError("账号仅允许 1-64 位英文字母、数字、下划线和短横线")
    if role not in ("customer", "admin"):
        raise ValueError("未知账号角色")
    encoded = hash_password(password)
    conn = _connect(path)
    try:
        with conn:
            conn.execute("INSERT INTO demo_accounts(user_id, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                         (user_id, encoded, role, int(time.time())))
    finally:
        conn.close()


def _login(user_id: str, password: str, client: str) -> tuple[str, Principal, int]:
    from app.config import settings
    now = int(time.time())
    ttl = int(getattr(settings, "auth_token_ttl_seconds", 3600))
    if not 60 <= ttl <= 86400:
        raise RuntimeError("AUTH_TOKEN_TTL_SECONDS 必须为 60-86400 秒")
    account_key = hashlib.sha256(user_id.encode()).hexdigest()
    client_key = hashlib.sha256(client.encode()).hexdigest()
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM demo_login_failures WHERE failed_at < ?", (now - _LOGIN_WINDOW_SECONDS,))
            conn.execute("DELETE FROM demo_sessions WHERE expires_at <= ?", (now,))
            count = conn.execute("SELECT COUNT(*) FROM demo_login_failures WHERE account_key = ? OR client_key = ?",
                                 (account_key, client_key)).fetchone()[0]
        if count >= _LOGIN_MAX_FAILURES:
            raise HTTPException(status_code=429, detail="登录尝试过多，请 5 分钟后重试", headers={"Retry-After": "300"})
        account = conn.execute("SELECT * FROM demo_accounts WHERE user_id = ?", (user_id,)).fetchone()
        valid = _verify_password(password, account["password_hash"] if account else None)
        if not valid or account is None or not account["enabled"]:
            with conn:
                conn.execute("INSERT INTO demo_login_failures VALUES (?, ?, ?)", (account_key, client_key, now))
            raise HTTPException(status_code=401, detail="账号或密码不正确", headers={"WWW-Authenticate": "Bearer"})
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with conn:
            conn.execute("INSERT INTO demo_sessions VALUES (?, ?, ?, ?)", (digest, user_id, now + ttl, now))
        return token, Principal(user_id, account["role"], digest), ttl
    finally:
        conn.close()


async def login(user_id: str, password: str, client: str) -> tuple[str, Principal, int]:
    # Avoid parallel scrypt work exhausting memory in the local demo. Serialize
    # attempts so the persisted failure counter is also read after the prior one.
    try:
        await asyncio.wait_for(_LOGIN_SLOT.acquire(), timeout=0.5)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="登录服务正忙，请稍后重试", headers={"Retry-After": "1"}) from exc
    task = asyncio.create_task(asyncio.to_thread(_login, user_id, password, client))
    release_now = True
    try:
        # A canceled HTTP request cannot stop a running scrypt thread. Keep its
        # slot until it really exits, otherwise repeated disconnects bypass the cap.
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        release_now = False

        def release_when_finished(done: asyncio.Task) -> None:
            if not done.cancelled():
                done.exception()  # retrieve a detached worker error without logging credentials
            _LOGIN_SLOT.release()

        task.add_done_callback(release_when_finished)
        raise
    finally:
        if release_now:
            _LOGIN_SLOT.release()


def _principal_for_token(token: str) -> Principal | None:
    if not 32 <= len(token) <= 128:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    conn = _connect()
    try:
        row = conn.execute("""SELECT a.user_id, a.role FROM demo_sessions s
            JOIN demo_accounts a ON a.user_id = s.user_id
            WHERE s.token_digest = ? AND s.expires_at > ? AND a.enabled = 1""",
                           (digest, int(time.time()))).fetchone()
        return Principal(row["user_id"], row["role"], digest) if row else None
    finally:
        conn.close()


async def require_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
) -> Principal:
    principal = None
    if credentials and credentials.scheme.lower() == "bearer":
        principal = await asyncio.to_thread(_principal_for_token, credentials.credentials)
    if principal is None:
        raise HTTPException(status_code=401, detail="请先登录，或重新登录已过期的会话", headers={"WWW-Authenticate": "Bearer"})
    return principal


async def require_admin(principal: Annotated[Principal, Depends(require_principal)]) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="此操作需要管理员权限")
    return principal


async def require_conversation_owner(conversation_id: int, principal: Principal):
    from app.db import repository
    from sqlalchemy.exc import SQLAlchemyError
    try:
        conv = await repository.get_conversation(conversation_id)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试") from exc
    if conv is None or conv.user_id != principal.user_id:
        # Do not reveal whether a different customer's conversation exists.
        raise HTTPException(status_code=404, detail="会话不存在或不可访问")
    return conv


async def logout(principal: Principal) -> None:
    def revoke() -> None:
        conn = _connect()
        try:
            with conn:
                conn.execute("DELETE FROM demo_sessions WHERE token_digest = ?", (principal.token_digest,))
        finally:
            conn.close()
    await asyncio.to_thread(revoke)


def login_client_key(request: Request) -> str:
    # No trust in X-Forwarded-For for the local demo.
    return request.client.host if request.client else "local-unknown"
