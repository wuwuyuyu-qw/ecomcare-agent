"""Small runtime adapter for local policy evidence; no model decides authority.

Citation text and metadata can be untrusted. Approval/version come exclusively
from the configured local catalog. Order conditions come from fresh scoped facts.
This module does not infer a shipping region from a current logistics city.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import re

from app.config import settings
from app.core.policy_evidence import evaluate_citations, load_catalog, render_assessment


def current_catalog():
    # Re-read on each assessment so a revoked approval or changed catalog never
    # silently survives a checkpoint. Malformed explicit config fails closed.
    return load_catalog(settings.policy_catalog_path or None)


def build_citations(hits: list[dict]) -> list[dict]:
    """Shared numbering/provenance for FAQ and both forced retrieval nodes."""
    catalog = current_catalog()
    out = []
    for i, h in enumerate(hits):
        citation = {"n": i + 1, "id": h["id"], "section_path": h.get("section_path"),
                    "question": h["question"], "answer": h["answer"],
                    "content_type": h.get("content_type")}
        citation["policy_source"] = {
            "body_sha256": hashlib.sha256(h["answer"].encode("utf-8")).hexdigest(),
            "catalog_digest": catalog.digest if catalog is not None else None,
            "catalog_revision": catalog.catalog_revision if catalog is not None else None,
        }
        matched = [entry for entry in catalog.documents
                   if entry.document_id == str(h["id"])
                   and entry.body_sha256 == citation["policy_source"]["body_sha256"]] if catalog is not None else []
        if len(matched) == 1:
            entry = matched[0]
            citation["policy_source"].update(policy_id=entry.policy_id, version=entry.version,
                source=entry.source, effective_from=entry.effective_from.isoformat(),
                effective_to=entry.effective_to.isoformat() if entry.effective_to else None)
        out.append(citation)
    return out


def requires_policy_application(query: str, route: str = "") -> bool:
    if route == "refund_flow":
        return True
    return bool(re.search(
        r"适用|(?:最新|现行|当前有效).{0,8}(?:政策|规则|退货|退款|运费)|"
        r"赔偿|赔付|补偿|能不能退|能否退|可以退|可退|退款资格|"
        r"运费.{0,10}(?:谁|承担|多少钱)", query or "", re.I))


def verified_order_facts(context: dict, now: str) -> dict:
    """Only successful order evidence from this exact task, fresh within 15 min."""
    scope = {key: context.get(key) for key in ("user_id", "conversation_id", "task_id", "active_order_id")}
    if not all(scope.values()):
        return {}
    facts = context.get("verified_facts") or {}
    if not isinstance(facts, dict):
        return {}

    def value(name):
        fact = facts.get(name)
        if (not isinstance(fact, dict) or fact.get("scope") != scope or fact.get("source") != "query_order"
                or not isinstance(fact.get("evidence_id"), str) or not fact["evidence_id"]):
            return None
        try:
            checked = datetime.fromisoformat(now.replace("Z", "+00:00"))
            observed = datetime.fromisoformat(fact["observed_at"].replace("Z", "+00:00"))
            age = (checked - observed).total_seconds()
            if checked.tzinfo is None or observed.tzinfo is None or not 0 <= age <= 900:
                return None
        except (ValueError, TypeError, AttributeError, KeyError):
            return None
        return fact.get("value")

    if value("order_owned") is not True or value("order_id") != scope["active_order_id"]:
        return {}
    mapping = {"destination_region": "region", "fulfillment_type": "fulfillment_scope",
               "created_at": "created_at", "delivered_at": "delivered_at", "order_status": "order_status"}
    return {target: found for source, target in mapping.items() if (found := value(source)) is not None}


def assess_context(citations: list[dict], context: dict, query: str, now: str) -> dict:
    assessment = evaluate_citations(citations, current_catalog(), verified_order_facts(context, now),
                                    now=now, require_latest=bool(re.search(r"最新|现行|latest", query, re.I)))
    # Metadata applicability alone is not an answer to the requested policy.
    # These narrow intent rules only choose a clause type; values still come
    # exclusively from approved, body-bound catalog entries and scoped facts.
    requested = set()
    if re.search(r"赔偿|赔付|补偿|compensation", query, re.I):
        requested.add("compensation_cap_cny")
    if re.search(r"运费|费用.{0,6}(?:谁|承担)|fee", query, re.I):
        requested.add("fee_payer")
    if re.search(r"退货|退款|return", query, re.I):
        requested.add("return_window_days")
    if re.search(r"物流|配送|包裹|快递|shipping", query, re.I):
        requested.add("shipping_stall_hours")
    supported = {c.get("claim_key") for c in assessment.get("supported_claims", [])}
    assessment["requested_claim_keys"] = sorted(requested)
    assessment["unsupported_requested_claim_keys"] = sorted(requested - supported)
    if assessment.get("outcome") == "applicable" and requested - supported:
        assessment.update(outcome="unknown", supported_claims=[], reason="requested_clause_not_supported")
    # Provenance for an assessment is not a verified fact or an execution receipt.
    assessment["scope"] = {key: context.get(key) for key in
                           ("user_id", "conversation_id", "task_id", "active_order_id")}
    assessment["verification_kind"] = "local_structured_policy_conditions"
    assessment["application_requested"] = requires_policy_application(query)
    return assessment


def context_citations(context: dict) -> list[dict]:
    """Fallback for query_faq tool results, which do not write the graph channel."""
    facts = context.get("verified_facts") or {}
    if not isinstance(facts, dict):
        return []
    scope = {key: context.get(key) for key in ("user_id", "conversation_id", "task_id", "active_order_id")}
    out = []
    for key, fact in facts.items():
        if (not key.startswith("policy_chunk_") or not isinstance(fact, dict)
                or fact.get("scope") != scope or fact.get("source") != "query_faq"
                or not isinstance(fact.get("value"), dict)):
            continue
        try:
            checked = datetime.fromisoformat(context["updated_at"].replace("Z", "+00:00"))
            observed = datetime.fromisoformat(fact["observed_at"].replace("Z", "+00:00"))
            if checked.tzinfo is None or observed.tzinfo is None or not 0 <= (checked - observed).total_seconds() <= 900:
                continue
        except (ValueError, TypeError, AttributeError, KeyError):
            continue
        out.append(fact["value"])
    return out


def assessment_text(assessment: dict) -> str:
    return render_assessment(assessment)
