"""Controlled policy metadata and a deliberately limited applicability check.

Only a validated, locally controlled catalog supplies approval and policy scope.
Retrieval supplies an identifier and exact answer body, never approval authority.
All evaluation is pure: callers supply verified plain order values and an aware
clock. This module does not prove that arbitrary generated prose follows a policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_CATALOG_BYTES = 2_097_152
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")
_TIMEBASES = {"now", "created_at", "paid_at", "shipped_at", "delivered_at"}
_MISSING = object()


class CatalogValidationError(ValueError):
    """A fixed diagnostic code and structural path, without document contents."""

    def __init__(self, code: str, path: str = "$"):
        self.code, self.path = code, path
        super().__init__(f"{code} at {path}")


@dataclass(frozen=True)
class Condition:
    fact: str
    op: str
    value: Any


@dataclass(frozen=True)
class PolicyClause:
    clause_id: str
    claim_key: str
    value: Any
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class PolicyEntry:
    document_id: str
    body_sha256: str
    policy_id: str
    version: str
    policy_group: str
    review_status: str
    approval_ref: str | None
    effective_from: datetime
    effective_to: datetime | None
    revision_at: datetime
    timebasis: str
    regions: tuple[str, ...]
    fulfillment_scopes: tuple[str, ...]
    source: str
    clauses: tuple[PolicyClause, ...]


@dataclass(frozen=True)
class PolicyCatalog:
    catalog_revision: str
    digest: str
    complete_groups: frozenset[str]
    documents: tuple[PolicyEntry, ...]


def _fail(code: str, path: str) -> None:
    raise CatalogValidationError(code, path)


def _object(value: Any, keys: set[str], path: str) -> dict:
    if not isinstance(value, dict):
        _fail("expected_object", path)
    if set(value) != keys:
        _fail("missing_or_unknown_fields", path)
    return value


def _text(value: Any, path: str, limit: int = 128, *, key: bool = False) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > limit or any(ord(c) < 32 for c in value)):
        _fail("invalid_text", path)
    if key and not _KEY.fullmatch(value):
        _fail("invalid_identifier", path)
    try:
        value.encode("utf-8")
    except UnicodeError:
        _fail("invalid_utf8_text", path)
    return value


def _id(value: Any) -> str | None:
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and _KEY.fullmatch(value):
        return value
    return None


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and _STAMP.fullmatch(value):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    try:
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            return None
        return stamp.astimezone(timezone.utc)
    except (ValueError, OverflowError, TypeError):
        # Valid local calendar values can overflow when converted to UTC.
        return None


def _timestamp(value: Any, path: str) -> datetime:
    # JSON catalogs use one explicit, timezone-aware ISO format.
    if not isinstance(value, str) or (stamp := _datetime(value)) is None:
        _fail("invalid_aware_timestamp", path)
    return stamp


def _list(value: Any, path: str, maximum: int) -> list:
    if not isinstance(value, list) or len(value) > maximum:
        _fail("invalid_list", path)
    return value


def _scopes(value: Any, path: str) -> tuple[str, ...]:
    values = _list(value, path, 256)
    if not values:
        _fail("empty_scope", path)
    result = tuple(_text(item, f"{path}[{i}]") for i, item in enumerate(values))
    if len(set(result)) != len(result) or ("*" in result and len(result) != 1):
        _fail("ambiguous_scope", path)
    return result


def _scalar(value: Any, path: str) -> Any:
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return _text(value, path, 2048)
    if type(value) is int or type(value) is float and math.isfinite(value):
        return value
    _fail("invalid_scalar", path)


def _clause(value: Any, path: str) -> PolicyClause:
    row = _object(value, {"clause_id", "claim_key", "value", "conditions"}, path)
    conditions = []
    seen = set()
    for i, item in enumerate(_list(row["conditions"], f"{path}.conditions", 32)):
        cp = f"{path}.conditions[{i}]"
        condition = _object(item, {"fact", "op", "value"}, cp)
        fact = _text(condition["fact"], f"{cp}.fact", key=True)
        op = condition["op"]
        if not isinstance(op, str) or op not in {"eq", "in", "gte", "lte"}:
            _fail("invalid_condition_operator", f"{cp}.op")
        if op == "in":
            raw = _list(condition["value"], f"{cp}.value", 128)
            if not raw:
                _fail("empty_condition_values", f"{cp}.value")
            expected = tuple(_scalar(v, f"{cp}.value[{n}]") for n, v in enumerate(raw))
        else:
            expected = _scalar(condition["value"], f"{cp}.value")
            if op in {"gte", "lte"} and type(expected) not in {int, float}:
                _fail("numeric_condition_required", f"{cp}.value")
        signature = json.dumps([fact, op, expected], ensure_ascii=False, allow_nan=False)
        if signature in seen:
            _fail("duplicate_condition", cp)
        seen.add(signature)
        conditions.append(Condition(fact, op, expected))
    return PolicyClause(
        _text(row["clause_id"], f"{path}.clause_id", key=True),
        _text(row["claim_key"], f"{path}.claim_key", key=True),
        _scalar(row["value"], f"{path}.value"), tuple(conditions))


def _entry(value: Any, path: str) -> PolicyEntry:
    row = _object(value, {"document_id", "body_sha256", "policy_id", "version",
        "policy_group", "review_status", "approval_ref", "effective_from",
        "effective_to", "revision_at", "timebasis", "regions",
        "fulfillment_scopes", "source", "clauses"}, path)
    document_id = _id(row["document_id"])
    if document_id is None:
        _fail("invalid_document_id", f"{path}.document_id")
    digest = row["body_sha256"]
    if not isinstance(digest, str) or not _SHA.fullmatch(digest):
        _fail("invalid_body_sha256", f"{path}.body_sha256")
    status = row["review_status"]
    if not isinstance(status, str) or status not in {"approved", "draft", "rejected"}:
        _fail("invalid_review_status", f"{path}.review_status")
    approval = row["approval_ref"]
    if approval is not None:
        approval = _text(approval, f"{path}.approval_ref", 512)
    if status == "approved" and approval is None:
        _fail("approved_without_approval_ref", f"{path}.approval_ref")
    start = _timestamp(row["effective_from"], f"{path}.effective_from")
    end = None if row["effective_to"] is None else _timestamp(row["effective_to"], f"{path}.effective_to")
    if end is not None and end <= start:
        _fail("invalid_effective_interval", f"{path}.effective_to")
    basis = row["timebasis"]
    if not isinstance(basis, str) or basis not in _TIMEBASES:
        _fail("invalid_timebasis", f"{path}.timebasis")
    clauses = tuple(_clause(v, f"{path}.clauses[{i}]")
                    for i, v in enumerate(_list(row["clauses"], f"{path}.clauses", 256)))
    if len({c.clause_id for c in clauses}) != len(clauses):
        _fail("duplicate_clause_id", f"{path}.clauses")
    return PolicyEntry(document_id, digest,
        _text(row["policy_id"], f"{path}.policy_id", key=True),
        _text(row["version"], f"{path}.version"),
        _text(row["policy_group"], f"{path}.policy_group", key=True), status,
        approval, start, end, _timestamp(row["revision_at"], f"{path}.revision_at"),
        basis, _scopes(row["regions"], f"{path}.regions"),
        _scopes(row["fulfillment_scopes"], f"{path}.fulfillment_scopes"),
        _text(row["source"], f"{path}.source", 2048), clauses)


def catalog_from_dict(data: dict, *, digest: str | None = None) -> PolicyCatalog:
    """Validate controlled JSON data; ``digest`` can bind the exact file bytes.

    Callers own the trust boundary. A retrieval result must never be passed here
    as a replacement catalog. Version labels are opaque, never sorted.
    """
    row = _object(data, {"schema_version", "catalog_revision", "policy_groups", "documents"}, "$")
    if type(row["schema_version"]) is not int or row["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported_schema_version", "$.schema_version")
    revision = _text(row["catalog_revision"], "$.catalog_revision")
    groups, complete = set(), set()
    for i, item in enumerate(_list(row["policy_groups"], "$.policy_groups", 1024)):
        path = f"$.policy_groups[{i}]"
        group = _object(item, {"policy_group", "complete"}, path)
        name = _text(group["policy_group"], f"{path}.policy_group", key=True)
        if name in groups:
            _fail("duplicate_policy_group", path)
        if type(group["complete"]) is not bool:
            _fail("invalid_group_completeness", f"{path}.complete")
        groups.add(name)
        if group["complete"]:
            complete.add(name)
    documents = tuple(_entry(v, f"$.documents[{i}]")
                      for i, v in enumerate(_list(row["documents"], "$.documents", 4096)))
    identities, versions = set(), {}
    for i, entry in enumerate(documents):
        path = f"$.documents[{i}]"
        if entry.policy_group not in groups:
            _fail("undeclared_policy_group", f"{path}.policy_group")
        binding = (entry.document_id, entry.body_sha256)
        if binding in identities:
            _fail("duplicate_document_binding", path)
        identities.add(binding)
        # Multiple chunks of one version are allowed, with identical authority.
        identity = (entry.policy_id, entry.version)
        metadata = (entry.policy_group, entry.review_status, entry.approval_ref,
                    entry.effective_from, entry.effective_to, entry.revision_at,
                    entry.timebasis, entry.regions, entry.fulfillment_scopes, entry.source)
        if identity in versions and versions[identity] != metadata:
            _fail("inconsistent_version_metadata", path)
        versions[identity] = metadata
    if digest is None:
        try:
            encoded = json.dumps(data, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError):
            _fail("invalid_json_data", "$")
        digest = hashlib.sha256(encoded).hexdigest()
    elif not isinstance(digest, str) or not _SHA.fullmatch(digest):
        _fail("invalid_catalog_digest", "$")
    return PolicyCatalog(revision, digest, frozenset(complete), documents)


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", "$")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    _fail("nonfinite_json_number", "$")


def load_catalog(path: str | Path | None) -> PolicyCatalog | None:
    """Read one explicit local JSON catalog. None disables controlled evidence.

    A configured file which is missing or malformed raises a fixed diagnostic;
    it never quietly falls back to approval inferred from retrieved documents.
    """
    if path is None:
        return None
    if (not isinstance(path, (str, Path)) or "://" in str(path)
            or str(path).startswith(("\\\\", "//")) or "\x00" in str(path)):
        _fail("invalid_catalog_path", "$")
    catalog_path = Path(path)
    if catalog_path.suffix.lower() != ".json":
        _fail("catalog_must_be_local_json", "$")
    try:
        with catalog_path.open("rb") as stream:
            body = stream.read(MAX_CATALOG_BYTES + 1)
    except (OSError, ValueError):
        _fail("catalog_unavailable", "$")
    if len(body) > MAX_CATALOG_BYTES:
        _fail("catalog_too_large", "$")
    try:
        data = json.loads(body.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=_constant)
    except CatalogValidationError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError):
        _fail("invalid_catalog_json", "$")
    return catalog_from_dict(data, digest=hashlib.sha256(body).hexdigest())


def _iso(stamp: datetime | None) -> str | None:
    return None if stamp is None else stamp.isoformat().replace("+00:00", "Z")


def _reason(code: str, **fields: Any) -> dict:
    return {"code": code, **fields}


def _reference(entry: PolicyEntry, facts: Mapping, now: datetime | None) -> datetime | None:
    return now if entry.timebasis == "now" else _datetime(facts.get(entry.timebasis))


def _metadata_check(entry: PolicyEntry, facts: Mapping, now: datetime | None) -> tuple[str, list[dict], list[str]]:
    reasons, missing = [], []
    if entry.review_status != "approved":
        return "inapplicable", [_reason("policy_not_approved")], []
    if now is None:
        reasons.append(_reason("evaluation_time_unknown"))
        missing.append("now")
    reference = _reference(entry, facts, now)
    if reference is None and entry.timebasis != "now":
        reasons.append(_reason("policy_timebasis_unknown", fact=entry.timebasis))
        missing.append(entry.timebasis)
    if now is not None and entry.revision_at > now:
        reasons.append(_reason("policy_revision_in_future"))
    if now is not None and entry.effective_from > now:
        reasons.append(_reason("policy_not_yet_effective"))
    if reference is not None:
        if reference < entry.effective_from:
            reasons.append(_reason("before_effective_interval"))
        elif entry.effective_to is not None and reference >= entry.effective_to:
            reasons.append(_reason("after_effective_interval"))
        if now is not None and reference > now:
            reasons.append(_reason("order_time_in_future", fact=entry.timebasis))
    for name, allowed in (("region", entry.regions), ("fulfillment_scope", entry.fulfillment_scopes)):
        if allowed == ("*",):
            continue
        value = facts.get(name)
        if not isinstance(value, str) or not value or value != value.strip():
            reasons.append(_reason("policy_scope_unknown", fact=name))
            missing.append(name)
        elif value not in allowed:
            reasons.append(_reason("policy_scope_mismatch", fact=name))
    definite = {"policy_revision_in_future", "policy_not_yet_effective", "before_effective_interval",
                "after_effective_interval", "order_time_in_future", "policy_scope_mismatch"}
    outcome = "inapplicable" if any(r["code"] in definite for r in reasons) else "unknown" if missing else "applicable"
    return outcome, reasons, list(dict.fromkeys(missing))


def _same(actual: Any, expected: Any) -> bool:
    # JSON booleans must not compare equal to numeric 0/1.
    if type(actual) is bool or type(expected) is bool:
        return type(actual) is type(expected) and actual == expected
    if type(actual) in {int, float} and type(expected) in {int, float}:
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def _condition(condition: Condition, facts: Mapping) -> bool | None:
    actual = facts.get(condition.fact, _MISSING)
    if (actual is _MISSING or actual is None or isinstance(actual, (dict, list, tuple))
            or type(actual) not in {str, bool, int, float}
            or (type(actual) is float and not math.isfinite(actual))):
        return None
    if condition.op == "eq":
        return _same(actual, condition.value)
    if condition.op == "in":
        return any(_same(actual, expected) for expected in condition.value)
    if type(actual) not in {int, float}:
        return None
    return actual >= condition.value if condition.op == "gte" else actual <= condition.value


def _claims(entry: PolicyEntry, facts: Mapping, citation_id: str) -> tuple[list[dict], list[dict], list[str]]:
    supported, diagnostics, missing = [], [], []
    for clause in entry.clauses:
        checks = [_condition(c, facts) for c in clause.conditions]
        absent = [c.fact for c, result in zip(clause.conditions, checks) if result is None]
        # A known false conjunct settles this clause even if another fact is absent.
        outcome = "inapplicable" if False in checks else "unknown" if absent else "applicable"
        diagnostics.append({"clause_id": clause.clause_id, "outcome": outcome,
                            "missing_facts": absent if outcome == "unknown" else []})
        if outcome == "unknown":
            missing.extend(absent)
        elif outcome == "applicable":
            supported.append({"citation_id": citation_id, "policy_id": entry.policy_id,
                "version": entry.version, "clause_id": clause.clause_id,
                "claim_key": clause.claim_key, "value": clause.value,
                "support_kind": "catalog_structured_clause"})
    return supported, diagnostics, list(dict.fromkeys(missing))


def _identity(entry: PolicyEntry) -> tuple[str, str]:
    return entry.policy_id, entry.version


def _scopes_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return "*" in left or "*" in right or bool(set(left) & set(right))


def _overlap(left: PolicyEntry, right: PolicyEntry) -> bool:
    if not (_scopes_overlap(left.regions, right.regions)
            and _scopes_overlap(left.fulfillment_scopes, right.fulfillment_scopes)):
        return False
    # Different governing timebases cannot be safely reconciled as one chain.
    if left.timebasis != right.timebasis:
        return True
    return ((left.effective_to is None or right.effective_from < left.effective_to)
            and (right.effective_to is None or left.effective_from < right.effective_to))


def evaluate_citations(citations: Sequence[Mapping], catalog: PolicyCatalog | None,
                       order_facts: Mapping | None = None, now: datetime | str | None = None,
                       require_latest: bool = False) -> dict:
    """Assess metadata, exact body binding and explicitly structured clauses.

    ``order_facts`` must be caller-selected fresh, verified plain values, never
    user claims or unfiltered model metadata. An incomplete group cannot prove
    latest. Missing current versions cannot be repaired by selecting an old hit.
    Result dictionaries contain no assertion that generated prose was checked.
    """
    facts = order_facts if isinstance(order_facts, Mapping) else {}
    clock = _datetime(now)
    result = {"outcome": "unknown", "reasons": [], "missing_facts": [], "citations": [],
        "supported_claims": [], "catalog_digest": catalog.digest if isinstance(catalog, PolicyCatalog) else None,
        "catalog_revision": catalog.catalog_revision if isinstance(catalog, PolicyCatalog) else None,
        "checked_at": _iso(clock), "reference_times": [], "require_latest": require_latest is True,
        "claim_support_scope": "catalog_structured_clauses_only"}
    if catalog is not None and not isinstance(catalog, PolicyCatalog):
        result["reasons"] = [_reason("invalid_catalog_object")]
        return result
    if not isinstance(citations, (list, tuple)) or not citations:
        result["reasons"] = [_reason("no_citations")]
        return result
    rows, bindings = [], {}
    for index, citation in enumerate(citations):
        citation_id = _id(citation.get("id")) if isinstance(citation, Mapping) else None
        body = citation.get("answer") if isinstance(citation, Mapping) else None
        row = {"citation_id": citation_id, "outcome": "unknown", "policy_id": None,
               "version": None, "policy_group": None, "reasons": [], "missing_facts": [],
               "supported_claims": [], "clause_assessments": []}
        rows.append(row)
        if citation_id is None or not isinstance(body, str) or not body:
            row["reasons"] = [_reason("invalid_citation", citation_index=index)]
            continue
        try:
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        except UnicodeError:
            row["reasons"] = [_reason("invalid_citation_utf8", citation_index=index)]
            continue
        row["body_sha256"] = digest
        if catalog is None:
            row.update(outcome="legacy_information", reasons=[_reason("no_controlled_catalog")])
            continue
        candidates = [entry for entry in catalog.documents if entry.document_id == citation_id]
        matched = [entry for entry in candidates if entry.body_sha256 == digest]
        if not candidates:
            row.update(outcome="legacy_information", reasons=[_reason("document_not_in_catalog")])
            continue
        if len(matched) != 1:
            row["reasons"] = [_reason("document_body_mismatch")]
            continue
        entry = matched[0]
        row.update(policy_id=entry.policy_id, version=entry.version, policy_group=entry.policy_group,
                   approval_ref=entry.approval_ref, source=entry.source, timebasis=entry.timebasis,
                   effective_from=_iso(entry.effective_from), effective_to=_iso(entry.effective_to),
                   revision_at=_iso(entry.revision_at), reference_time=_iso(_reference(entry, facts, clock)))
        bindings[index] = entry
        status, reasons, missing = _metadata_check(entry, facts, clock)
        row.update(outcome=status, reasons=reasons, missing_facts=missing)
        result["reference_times"].append({"citation_id": citation_id,
            "timebasis": entry.timebasis, "reference_time": row["reference_time"]})

    if catalog is not None:
        for index, entry in bindings.items():
            row = rows[index]
            if row["outcome"] != "applicable":
                continue
            group = [e for e in catalog.documents if e.policy_group == entry.policy_group
                     and e.review_status == "approved"]
            active = [e for e in group if _metadata_check(e, facts, clock)[0] == "applicable"]
            # Full catalog peers matter even when retrieval omitted them.
            competing = [e for e in active if _identity(e) != _identity(entry) and _overlap(entry, e)]
            if competing:
                row.update(outcome="conflict", reasons=[_reason("overlapping_approved_versions",
                    policy_ids=sorted({e.policy_id for e in [entry, *competing]}),
                    versions=sorted({e.version for e in [entry, *competing]}))])
                continue
            uncertain = [e for e in group if _identity(e) != _identity(entry)
                         and _metadata_check(e, facts, clock)[0] == "unknown" and _overlap(entry, e)]
            if uncertain:
                # An unrestricted hit cannot conceal an overlapping restricted
                # version while the order's governing scope/time is unknown.
                row.update(outcome="unknown", reasons=[_reason("overlapping_policy_scope_unknown")])
                row["missing_facts"] = list(dict.fromkeys(f for e in uncertain
                    for f in _metadata_check(e, facts, clock)[2]))
                continue
            if require_latest is True:
                if entry.policy_group not in catalog.complete_groups:
                    row.update(outcome="unknown", reasons=[_reason("policy_group_incomplete")])
                    continue
                newest = max(active, key=lambda e: (e.effective_from, e.revision_at), default=None)
                if newest is None:
                    row.update(outcome="unknown", reasons=[_reason("no_current_approved_policy")])
                    continue
                if _identity(newest) != _identity(entry):
                    retrieved = any(_identity(e) == _identity(newest) for e in bindings.values())
                    row.update(outcome="inapplicable" if retrieved else "unknown",
                        reasons=[_reason("superseded_policy" if retrieved else "latest_policy_not_retrieved",
                            policy_id=newest.policy_id, version=newest.version)])
                    continue
            supported, clauses, missing = _claims(entry, facts, row["citation_id"])
            row["clause_assessments"] = clauses
            if missing:
                row.update(outcome="unknown", reasons=[_reason("clause_facts_unknown")], missing_facts=missing)
            else:
                row["supported_claims"] = supported

    result["citations"] = rows
    result["reasons"] = [{**reason, "citation_id": row["citation_id"],
                           **({"policy_id": row["policy_id"]} if row["policy_id"] else {})}
                          for row in rows for reason in row["reasons"]]
    result["missing_facts"] = list(dict.fromkeys(f for row in rows for f in row["missing_facts"]))
    statuses = {row["outcome"] for row in rows}
    # Mixed unresolved hits cannot silently authorize an unspecified answer.
    if "conflict" in statuses:
        result["outcome"] = "conflict"
    elif "unknown" in statuses:
        result["outcome"] = "unknown"
    elif statuses == {"applicable"}:
        result["outcome"] = "applicable"
    elif statuses == {"legacy_information"}:
        result["outcome"] = "legacy_information"
    elif statuses == {"inapplicable"}:
        result["outcome"] = "inapplicable"
    else:
        result["outcome"] = "unknown"
        result["reasons"].append(_reason("mixed_citation_applicability"))
    result["supported_claims"] = [claim for row in rows if row["outcome"] == "applicable"
                                  for claim in row["supported_claims"]]
    # Distinct groups can still make contradictory statements about one key.
    # Do not combine them into one current-order answer without reconciliation.
    by_key = {}
    for claim in result["supported_claims"]:
        encoded = json.dumps(claim["value"], sort_keys=True, ensure_ascii=False, allow_nan=False)
        by_key.setdefault(claim["claim_key"], set()).add(encoded)
    conflicting = sorted(key for key, values in by_key.items() if len(values) > 1)
    if conflicting:
        result["outcome"] = "conflict"
        result["reasons"].append(_reason("structured_clause_conflict", claim_keys=conflicting))
        result["supported_claims"] = []
    if result["outcome"] != "applicable":
        # Per-citation audit detail may retain a supported partial clause, but
        # an unresolved aggregate cannot release it as a final task conclusion.
        result["supported_claims"] = []
    return result


def render_assessment(assessment: Mapping) -> str:
    """Render controlled clauses for customers, with technical reasons in JSON only."""
    if not isinstance(assessment, Mapping):
        return "政策适用性未知：缺少受控核验结果。"
    labels = {"applicable": "元数据适用条件已核验", "inapplicable": "所检索政策不适用",
              "unknown": "政策适用性未知", "conflict": "政策版本冲突",
              "legacy_information": "仅有历史知识说明，不能确认现行适用性"}
    lines = [labels.get(assessment.get("outcome"), "政策适用性未知") + "。"]
    field_names = {"now": "核验时间", "region": "收货地区", "fulfillment_scope": "配送类型",
                   "created_at": "下单时间", "delivered_at": "签收时间", "order_status": "订单状态"}
    missing = assessment.get("missing_facts")
    if isinstance(missing, list) and missing:
        lines.append("缺少已核验条件：" + "、".join(field_names.get(str(v), "相关订单条件") for v in missing) + "。")
    reasons = assessment.get("reasons")
    if isinstance(reasons, list) and reasons:
        codes = list(dict.fromkeys(r.get("code") for r in reasons if isinstance(r, Mapping)
                                   and isinstance(r.get("code"), str)))
        readable = {"no_citations": "未取得政策来源", "no_controlled_catalog": "资料未登记版本与审批记录",
                    "document_not_in_catalog": "资料未登记版本与审批记录", "document_body_mismatch": "正文与审核版本不一致",
                    "policy_not_approved": "该版本尚未审批通过", "policy_not_yet_effective": "政策尚未生效",
                    "after_effective_interval": "政策已超出适用时间", "before_effective_interval": "订单早于政策生效时间",
                    "policy_scope_mismatch": "地区或配送条件不符合适用范围", "overlapping_approved_versions": "存在互相冲突的有效版本",
                    "latest_policy_not_retrieved": "未取得当前版本正文", "superseded_policy": "所取资料不是当前版本",
                    "structured_clause_conflict": "来源对同一关键条款的规定不一致", "policy_group_incomplete": "版本清单尚未完整核验"}
        notes = list(dict.fromkeys(readable[code] for code in codes if code in readable))
        if notes:
            lines.append("核验结果：" + "；".join(notes) + "。")
    claims = assessment.get("supported_claims")
    if assessment.get("outcome") == "applicable" and isinstance(claims, list) and claims:
        claim_names = {"return_window_days": "申请期限（天）", "shipping_stall_hours": "物流无更新阈值（小时）",
                       "fee_payer": "运费承担方", "compensation_cap_cny": "补偿上限（元）"}
        for claim in claims:
            label = claim_names.get(claim.get("claim_key"), "参考条款")
            value = json.dumps(claim.get("value"), ensure_ascii=False, allow_nan=False)
            lines.append(f"{label}：{value}（来源 {claim.get('policy_id')}，版本 {claim.get('version')}）。")
    lines.append("以上仅说明已核验的政策条件；具体资格及处理结果仍需核实，当前未执行退款或赔付。")
    return "\n".join(lines)
