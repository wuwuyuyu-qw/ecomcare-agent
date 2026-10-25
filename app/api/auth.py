"""Explicit local demo login; no anonymous or client-selected identity mode."""
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field, field_validator

from app.core import auth

router = APIRouter(prefix="/api/auth", tags=["身份"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=1, max_length=256)

    @field_validator("password")
    @classmethod
    def _valid_password_text(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("密码包含非法文本字符") from exc
        return value


@router.post("/login")
async def login(req: LoginRequest, request: Request, response: Response):
    token, principal, ttl = await auth.login(req.username, req.password, auth.login_client_key(request))
    response.headers["Cache-Control"] = "no-store"
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl,
            "user": {"user_id": principal.user_id, "role": principal.role}}


@router.get("/me")
async def me(response: Response, principal: Annotated[auth.Principal, Depends(auth.require_principal)]):
    response.headers["Cache-Control"] = "no-store"
    from app.tools.synthetic_fixtures import profile_info
    from app.config import settings
    return {"user_id": principal.user_id, "role": principal.role, **profile_info(),
            "execution_policy": {"skip_coref_without_history": settings.skip_coref_without_history}}


@router.post("/logout", status_code=204)
async def logout(principal: Annotated[auth.Principal, Depends(auth.require_principal)]):
    await auth.logout(principal)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
