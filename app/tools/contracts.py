"""Locally owned tool return contracts; remote content is data, never execution authority.

Validation happens before *and* after a local formatter. A normal function return
can still be a business refusal. Structured data remains available to guards even
when the model-facing ToolMessage is shortened.
"""
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CONTRACT_VERSION = "tool-contracts/1"
MAX_RESULT_CHARS = 131_072
BusinessStatus = Literal["ok", "not_found", "forbidden", "insufficient",
                         "transient_failure", "permanent_failure", "untrusted"]
Text = Annotated[str, Field(min_length=1, max_length=512)]
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
LongText = Annotated[str, Field(min_length=1, max_length=60_000)]
Amount = Annotated[int, Field(ge=0)] | Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ContractError(ValueError):
    """Safe, fixed errors: never retain a vendor error, payload or credential."""

    def __init__(self, code: str = "invalid_tool_result", message: str = "工具返回不符合本地契约，不能作为证据。"):
        super().__init__(message)
        self.code = code


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("created_at", "warranty_until", "updated_at", "delivered_at", check_fields=False)
    @classmethod
    def valid_calendar_date(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value


class OrderResult(_Strict):
    order_id: Identifier
    status: Literal["待付款", "已付款", "已发货", "已签收"]
    amount: Amount
    created_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$")]
    product: Text
    tracking_no: Identifier
    destination_region: Text | None = None
    fulfillment_type: Text | None = None
    delivered_at: Annotated[str, Field(max_length=64)] | None = None


class LogisticsResult(_Strict):
    tracking_no: Identifier
    status_code: Literal["PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"]
    current_city: Text
    trace: Annotated[list[Text], Field(max_length=100)]
    stalled_hours: Amount | None = None
    carrier_code: Text | None = None
    fixture_profile: Literal["demo", "task-v1"] | None = None
    fixture_version: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    fixture_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class LogisticsDisplay(_Strict):
    tracking_no: Identifier
    status: Literal["已揽件", "运输中", "派送中", "已签收"]
    current_city: Text
    trace: Annotated[list[Text], Field(max_length=100)]
    stalled_hours: Amount | None = None


class PolicySource(_Strict):
    body_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    catalog_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    catalog_revision: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    policy_id: Text | None = None
    version: Text | None = None
    source: Annotated[str, Field(max_length=2048)] | None = None
    effective_from: Annotated[str, Field(max_length=64)] | None = None
    effective_to: Annotated[str, Field(max_length=64)] | None = None


class Citation(_Strict):
    n: Annotated[int, Field(ge=1)]
    id: Annotated[int, Field(ge=1)] | Identifier
    section_path: Annotated[str, Field(max_length=512)] | None = None
    question: LongText
    answer: LongText
    content_type: Annotated[str, Field(max_length=128)] | None = None
    policy_source: PolicySource | None = None


class FaqResult(_Strict):
    sufficient: bool
    evidence: LongText | None = None
    citations: Annotated[list[Citation], Field(max_length=100)]
    source: Literal["retrieval_low_conf", "self_check"] | None = None
    reason: Annotated[str, Field(max_length=1024)] | None = None

    @model_validator(mode="after")
    def check_evidence(self):
        if self.sufficient:
            if not self.citations or not self.evidence:
                raise ValueError("sufficient requires citations and evidence")
            numbers = [c.n for c in self.citations]
            ids = [str(c.id) for c in self.citations]
            if numbers != list(range(1, len(self.citations) + 1)) or len(set(ids)) != len(ids):
                raise ValueError("invalid citation identities")
            assembled = "\n".join(f"[{c.n}] {c.question}: {c.answer}" for c in self.citations)
            if self.evidence != assembled:
                raise ValueError("evidence disagrees with citations")
        elif self.citations or self.evidence or not self.reason:
            raise ValueError("insufficient result must carry no evidence")
        return self


class WarrantyResult(_Strict):
    order_id: Identifier
    warranty_code: Literal["IN_WARRANTY", "EXPIRED"]
    warranty_until: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    policy_ref: Text | None = None


class WarrantyDisplay(_Strict):
    order_id: Identifier
    warranty: Literal["在保", "已过保"]
    warranty_until: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class ReturnResult(_Strict):
    order_id: Identifier
    return_code: Literal["AUDITING", "RETURNING", "REFUNDED", "NONE"]
    updated_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$")]


class ReturnDisplay(_Strict):
    order_id: Identifier
    return_status: Literal["审核中", "退货中", "已退款", "无退货记录"]
    updated_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$")]


class ProductResult(_Strict):
    product_name: Text
    price: Amount
    stock: Annotated[int, Field(ge=0)]
    spec: Text


class RefundResult(_Strict):
    status: Literal["待用户确认"]
    order_id: Identifier


class TicketResult(_Strict):
    ticket_no: Text
    status: Literal["已转人工"]


class OrderInput(_Strict):
    order_id: Identifier


class LogisticsInput(_Strict):
    tracking_no: Identifier


class FaqInput(_Strict):
    keyword: Annotated[str, Field(min_length=1, max_length=4096)]
    category: Annotated[str, Field(max_length=128)] | None = None


class ProductInput(_Strict):
    product_name: Text


class RefundInput(OrderInput):
    reason: Annotated[str, Field(max_length=1024)] | None = None


class TicketInput(_Strict):
    description: Annotated[str, Field(min_length=1, max_length=8192)]
    ticket_type: Literal["售后", "投诉", "咨询"]


_RAW_MODELS = {
    "query_order": OrderResult, "query_logistics": LogisticsResult, "query_faq": FaqResult,
    "query_warranty": WarrantyResult, "query_return_status": ReturnResult,
    "query_product": ProductResult, "submit_refund": RefundResult, "create_ticket": TicketResult,
}
_DISPLAY_MODELS = {"query_logistics": LogisticsDisplay, "query_warranty": WarrantyDisplay,
                   "query_return_status": ReturnDisplay}
_INPUT_MODELS = {"query_order": OrderInput, "query_logistics": LogisticsInput,
                 "query_warranty": OrderInput, "query_return_status": OrderInput,
                 "query_faq": FaqInput, "query_product": ProductInput,
                 "submit_refund": RefundInput, "create_ticket": TicketInput}
_SAFE_ERRORS = {
    "order_not_owned": ("forbidden", "没有找到您的这笔订单"),
    "not_found": ("not_found", "没有找到符合条件的记录"),
    "forbidden": ("forbidden", "没有权限访问该记录"),
}


@dataclass(frozen=True)
class ContractResult:
    data: dict
    business_status: BusinessStatus = "ok"
    error_code: str | None = None


def has_contract(name: str) -> bool:
    return name in _RAW_MODELS


def validate_input(name: str, args: dict) -> None:
    """The server cannot relax locally registered tool inputs or accept identity injection."""
    model = _INPUT_MODELS.get(name)
    if model is not None:
        try:
            model.model_validate(args)
        except (ValidationError, ValueError, TypeError):
            raise ContractError("invalid_arguments", "参数不符合本地工具定义，缺少信息时请先追问。") from None


def check_schema_refs(value, *, depth: int = 0) -> None:
    """Do not resolve an untrusted schema URI over the network during input validation."""
    if depth > 24:
        raise ContractError("invalid_tool_schema")
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$id"} and (not isinstance(child, str) or not child.startswith("#")):
                raise ContractError("invalid_tool_schema")
            check_schema_refs(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            check_schema_refs(child, depth=depth + 1)


def _object_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ContractError("ambiguous_tool_result")
        out[key] = value
    return out


def _bad_constant(_value):
    raise ContractError()


def parse_result(result, *, source: str) -> dict:
    """Accept one unambiguous JSON object. MCP multimedia/free text is not evidence."""
    if source == "mcp" and isinstance(result, list):
        if (len(result) != 1 or not isinstance(result[0], dict)
                or result[0].get("type") != "text" or not isinstance(result[0].get("text"), str)):
            raise ContractError("unsupported_mcp_content", "MCP 返回了无法核验的内容块，已拒绝使用。")
        result = result[0]["text"]
    if isinstance(result, str):
        if len(result) > MAX_RESULT_CHARS:
            raise ContractError("tool_result_too_large")
        try:
            result = json.loads(result, object_pairs_hook=_object_pairs, parse_constant=_bad_constant)
        except ContractError:
            raise
        except (ValueError, TypeError, RecursionError):
            raise ContractError("invalid_mcp_json" if source == "mcp" else "invalid_tool_result") from None
    if not isinstance(result, dict):
        raise ContractError()
    try:
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ContractError() from None
    if len(encoded) > MAX_RESULT_CHARS:
        raise ContractError("tool_result_too_large")
    return result


def validate_result(name: str, result: dict, args: dict, *, formatted: bool = False) -> ContractResult:
    """Reject unknown fields/enums/coercions and result identifiers that differ from requests."""
    if "error" in result or "code" in result:
        code = result.get("code")
        if (set(result) != {"error", "code"} or not isinstance(code, str)
                or code not in _SAFE_ERRORS or not isinstance(result.get("error"), str)):
            raise ContractError()
        # Vendor free-text errors never get model authority or enter logs.
        status, message = _SAFE_ERRORS[code]
        return ContractResult({"error": message, "code": code}, status, code)
    model = _DISPLAY_MODELS.get(name) if formatted else None
    model = model or _RAW_MODELS.get(name)
    if model is None:
        # Local extension code is trusted; remote tools require a locally registered contract.
        return ContractResult(redact(result, audit=False))
    try:
        value = model.model_validate(result)
    except (ValidationError, ValueError, TypeError):
        raise ContractError() from None
    data = value.model_dump(exclude_none=True)
    for field in ("order_id", "tracking_no", "product_name"):
        if field in data and field in args and data[field] != args[field]:
            raise ContractError("tool_result_id_mismatch", "工具返回的对象与本次请求不一致，已拒绝使用。")
    if name == "query_faq" and not data["sufficient"]:
        return ContractResult(data, "insufficient", "insufficient_evidence")
    return ContractResult(data)


def public_data(name: str, data: dict) -> dict:
    """Remote internal fields cannot leak when a formatter is missing or misregistered."""
    if name == "query_logistics":
        return {**{k: data[k] for k in ("tracking_no", "current_city", "trace", "stalled_hours") if k in data},
                "status": {"PICKED_UP": "已揽件", "IN_TRANSIT": "运输中", "DELIVERING": "派送中",
                           "DELIVERED": "已签收"}[data["status_code"]]}
    if name == "query_warranty":
        return {"order_id": data["order_id"], "warranty_until": data["warranty_until"],
                "warranty": {"IN_WARRANTY": "在保", "EXPIRED": "已过保"}[data["warranty_code"]]}
    if name == "query_return_status":
        return {"order_id": data["order_id"], "updated_at": data["updated_at"],
                "return_status": {"AUDITING": "审核中", "RETURNING": "退货中", "REFUNDED": "已退款",
                                  "NONE": "无退货记录"}[data["return_code"]]}
    return data


_SECRET_KEY = re.compile(r"password|passwd|secret|token|api.?key|authorization|cookie|credential", re.I)
_IDENTITY_KEYS = {"user_id", "order_id", "tracking_no", "email", "phone", "mobile", "address", "recipient"}
_SECRET_TEXT = re.compile(r"(?i)(?:bearer\s+[^\s,;]+|sk-[A-Za-z0-9_-]{8,}|(?:password|api_key|token)\s*[:=]\s*[^\s,;]+)")
_AUDIT_FIELDS = _IDENTITY_KEYS | {"conversation_id", "description", "ticket_type", "reason", "keyword",
    "category", "product_name", "status", "status_code", "amount", "created_at", "product", "current_city",
    "trace", "stalled_hours", "sufficient", "evidence", "citations", "source", "n", "id", "section_path",
    "question", "answer", "content_type", "error", "code", "price", "stock", "spec", "ticket_no",
    "warranty", "warranty_until", "return_status", "updated_at"}


def redact(value, *, audit: bool = True, key: str = "", depth: int = 0):
    """Audit stores shapes/counts and pseudonymous IDs, never raw business prose/secrets."""
    if depth > 12:
        return "[nested]"
    if _SECRET_KEY.search(key):
        return "[redacted]"
    if audit and key.lower() in _IDENTITY_KEYS and value is not None:
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
        return f"[id:{digest}]"
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            raw_key = str(k)
            safe_key = raw_key[:128]
            if audit and raw_key not in _AUDIT_FIELDS:
                safe_key = "[field:" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:8] + "]"
            elif not audit:
                safe_key = _SECRET_TEXT.sub("[redacted]", safe_key)
            out[safe_key] = redact(v, audit=audit, key=raw_key, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, audit=audit, key=key, depth=depth + 1) for v in value[:100]]
    if isinstance(value, str):
        if audit:
            return f"[text:{len(value)} chars]"
        return _SECRET_TEXT.sub("[redacted]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return "[invalid number]"
    return value if value is None or isinstance(value, (int, float, bool)) else "[unsupported]"
