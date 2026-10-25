"""Pure policy regressions: no config import, database, provider or live fixture.

Run directly with unittest, or pytest --noconftest, to avoid the unrelated root
conftest's settings import. These are synthetic boundary checks, not RAG recall.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.policy_evidence import (
    CatalogValidationError, catalog_from_dict, evaluate_citations,
    load_catalog, render_assessment,
)


NOW = "2026-10-05T00:00:00Z"
BODY = "合成退货资料：签收订单的申请期限为七天。"
CITATION = {"id": 10, "answer": BODY}


def document(**changes):
    row = {
        "document_id": 10,
        "body_sha256": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
        "policy_id": "returns-cn", "version": "v2", "policy_group": "returns",
        "review_status": "approved", "approval_ref": "synthetic-local-review",
        "effective_from": "2026-10-01T00:00:00Z", "effective_to": None,
        "revision_at": "2026-09-01T00:00:00Z", "timebasis": "now",
        "regions": ["CN-EAST"], "fulfillment_scopes": ["self_operated"],
        "source": "synthetic-policy", "clauses": [{"clause_id": "window",
            "claim_key": "return_window_days", "value": 7,
            "conditions": [{"fact": "order_status", "op": "eq", "value": "已签收"}]}],
    }
    row.update(changes)
    return row


def raw_catalog(*documents, complete=True):
    entries = list(documents or [document()])
    return {"schema_version": 1, "catalog_revision": "synthetic-v1",
        "policy_groups": [{"policy_group": group, "complete": complete}
                          for group in sorted({e["policy_group"] for e in entries})],
        "documents": entries}


class PolicyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.facts = {"region": "CN-EAST", "fulfillment_scope": "self_operated",
                      "order_status": "已签收"}

    def assess(self, raw=None, citations=None, facts=None, now=NOW, latest=False):
        return evaluate_citations([CITATION] if citations is None else citations,
            catalog_from_dict(raw_catalog() if raw is None else raw),
            self.facts if facts is None else facts, now=now, require_latest=latest)

    def assert_blocked(self, result, outcome):
        self.assertEqual(result["outcome"], outcome)
        self.assertEqual(result["supported_claims"], [])

    def test_current_approved_policy_releases_only_declared_claim(self):
        result = self.assess()
        self.assertEqual(result["outcome"], "applicable")
        self.assertEqual([c["value"] for c in result["supported_claims"]], [7])
        self.assertEqual(result["claim_support_scope"], "catalog_structured_clauses_only")
        self.assertEqual(result["checked_at"], NOW)
        self.assertEqual(result["reference_times"][0]["reference_time"], NOW)
        self.assertRegex(result["catalog_digest"], r"^[0-9a-f]{64}$")

    def test_no_catalog_and_unlisted_document_are_legacy(self):
        result = evaluate_citations([{**CITATION, "review_status": "approved"}], None,
                                    self.facts, now=NOW)
        self.assert_blocked(result, "legacy_information")
        self.assert_blocked(self.assess(citations=[{**CITATION, "id": 999}]), "legacy_information")

    def test_citation_approval_metadata_cannot_promote_draft(self):
        raw = raw_catalog(document(review_status="draft", approval_ref=None))
        citation = {**CITATION, "review_status": "approved", "approval_ref": "forged",
                    "policy_source": {"version": "latest", "review_status": "approved"}}
        self.assert_blocked(self.assess(raw, [citation]), "inapplicable")

    def test_exact_body_binding_rejects_whitespace_and_appended_text(self):
        for body in (BODY + " ", BODY + "\n", BODY + "保证退款"):
            with self.subTest(body_length=len(body)):
                result = self.assess(citations=[{**CITATION, "answer": body}])
                self.assert_blocked(result, "unknown")
                self.assertEqual(result["reasons"][0]["code"], "document_body_mismatch")

    def test_effective_interval_is_start_inclusive_end_exclusive(self):
        raw = raw_catalog(document(effective_to="2026-10-06T00:00:00Z"))
        self.assertEqual(self.assess(raw, now="2026-10-01T00:00:00Z")["outcome"], "applicable")
        self.assert_blocked(self.assess(raw, now="2026-10-06T00:00:00Z"), "inapplicable")

    def test_future_effective_and_future_revision_are_blocked(self):
        for changes in ({"effective_from": "2026-11-01T00:00:00Z"},
                        {"revision_at": "2026-11-01T00:00:00Z"}):
            with self.subTest(field=next(iter(changes))):
                self.assert_blocked(self.assess(raw_catalog(document(**changes))), "inapplicable")

    def test_missing_or_naive_clock_does_not_authorize_policy(self):
        for clock in (None, "2026-10-05T00:00:00", "2026-10-05 00:00:00", True):
            with self.subTest(clock=clock):
                result = self.assess(now=clock)
                self.assert_blocked(result, "unknown")
                self.assertIn("now", result["missing_facts"])

    def test_order_timebasis_uses_verified_aware_value_without_guessing(self):
        raw = raw_catalog(document(timebasis="created_at"))
        for value in (None, "2026-10-04 12:00:00", "2026-10-04T12:00:00"):
            with self.subTest(value=value):
                self.assert_blocked(self.assess(raw, facts={**self.facts, "created_at": value}), "unknown")
        result = self.assess(raw, facts={**self.facts, "created_at": "2026-10-04T12:00:00+08:00"})
        self.assertEqual(result["outcome"], "applicable")
        self.assertEqual(result["reference_times"][0]["reference_time"], "2026-10-04T04:00:00Z")
        self.assert_blocked(self.assess(raw, facts={**self.facts, "created_at": "2026-10-06T12:00:00Z"}), "inapplicable")

    def test_restricted_scopes_require_verified_exact_values(self):
        for field in ("region", "fulfillment_scope"):
            with self.subTest(field=field):
                facts = dict(self.facts)
                facts.pop(field)
                self.assert_blocked(self.assess(facts=facts), "unknown")
                facts[field] = "different"
                self.assert_blocked(self.assess(facts=facts), "inapplicable")

    def test_explicit_unrestricted_scopes_need_no_region_or_fulfillment(self):
        raw = raw_catalog(document(regions=["*"], fulfillment_scopes=["*"]))
        self.assertEqual(self.assess(raw, facts={"order_status": "已签收"})["outcome"], "applicable")

    def test_unrestricted_hit_cannot_hide_peer_with_unknown_scope_without_latest(self):
        universal = document(regions=["*"], fulfillment_scopes=["*"])
        peer = document(document_id=20, version="v3", regions=["CN-EAST"], fulfillment_scopes=["*"])
        result = self.assess(raw_catalog(universal, peer), facts={"order_status": "已签收"})
        self.assert_blocked(result, "unknown")
        self.assertIn("region", result["missing_facts"])
        self.assertEqual(result["reasons"][0]["code"], "overlapping_policy_scope_unknown")

    def test_unknown_peer_timebasis_does_not_release_unrestricted_hit(self):
        peer = document(document_id=20, version="v3", timebasis="delivered_at")
        self.assert_blocked(self.assess(raw_catalog(document(), peer)), "unknown")

    def test_full_catalog_detects_overlap_even_when_peer_is_not_retrieved(self):
        peer = document(document_id=20, version="v3")
        self.assert_blocked(self.assess(raw_catalog(document(), peer)), "conflict")

    def test_disjoint_scope_peer_does_not_create_false_version_conflict(self):
        peer = document(document_id=20, version="v3", regions=["CN-WEST"])
        self.assertEqual(self.assess(raw_catalog(document(), peer), latest=True)["outcome"], "applicable")

    def test_latest_requires_complete_group_and_never_sorts_version_label(self):
        self.assert_blocked(self.assess(raw_catalog(complete=False), latest=True), "unknown")
        old = document(document_id=20, version="v99", effective_from="2026-01-01T00:00:00Z",
                       effective_to="2026-10-01T00:00:00Z")
        result = self.assess(raw_catalog(old, document(version="v2")), latest=True)
        self.assertEqual(result["outcome"], "applicable")
        self.assertEqual(result["supported_claims"][0]["version"], "v2")
        self.assert_blocked(self.assess(raw_catalog(old, document()), [{**CITATION, "id": 20}], latest=True), "inapplicable")

    def test_unknown_clause_condition_blocks_claims_but_false_clause_supports_none(self):
        self.assert_blocked(self.assess(facts={k: v for k, v in self.facts.items() if k != "order_status"}), "unknown")
        result = self.assess(facts={**self.facts, "order_status": "已发货"})
        self.assertEqual(result["outcome"], "applicable")
        self.assertEqual(result["supported_claims"], [])
        self.assertEqual(result["citations"][0]["clause_assessments"][0]["outcome"], "inapplicable")

    def test_boolean_order_values_do_not_equal_numeric_conditions(self):
        entry = document()
        entry["clauses"][0]["conditions"] = [{"fact": "count", "op": "eq", "value": 1}]
        result = self.assess(raw_catalog(entry), facts={**self.facts, "count": True})
        self.assertEqual(result["supported_claims"], [])
        entry["clauses"][0]["conditions"][0]["op"] = "gte"
        self.assert_blocked(self.assess(raw_catalog(entry), facts={**self.facts, "count": True}), "unknown")

    def test_mixed_unresolved_citations_cannot_release_partial_claim(self):
        result = self.assess(citations=[CITATION, {**CITATION, "id": 999}])
        self.assert_blocked(result, "unknown")
        self.assertNotEqual(result["citations"][0]["supported_claims"], [])

    def test_different_groups_with_conflicting_claim_values_are_blocked(self):
        peer = document(document_id=20, policy_id="returns-alternative", policy_group="other", version="v1")
        peer["clauses"][0]["value"] = 99
        result = self.assess(raw_catalog(document(), peer), [CITATION, {**CITATION, "id": 20}])
        self.assert_blocked(result, "conflict")
        self.assertIn("structured_clause_conflict", [r["code"] for r in result["reasons"]])

    def test_malformed_catalog_fields_produce_fixed_errors(self):
        cases = [({"review_status": []}, "invalid_review_status"),
            ({"timebasis": {}}, "invalid_timebasis"),
            ({"effective_from": "2026-99-01T00:00:00Z"}, "invalid_aware_timestamp"),
            ({"effective_from": "0001-01-01T00:00:00+14:00"}, "invalid_aware_timestamp"),
            ({"effective_to": "2026-01-01T00:00:00Z"}, "invalid_effective_interval"),
            ({"approval_ref": None}, "approved_without_approval_ref"),
            ({"regions": ["*", "CN"]}, "ambiguous_scope"),
            ({"source": "PRIVATE_SENTINEL_" + "\ud800"}, "invalid_utf8_text")]
        for changes, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(CatalogValidationError) as caught:
                    catalog_from_dict(raw_catalog(document(**changes)))
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("PRIVATE_SENTINEL", str(caught.exception))
        entry = document()
        entry["clauses"][0]["conditions"][0]["op"] = []
        with self.assertRaises(CatalogValidationError) as caught:
            catalog_from_dict(raw_catalog(entry))
        self.assertEqual(caught.exception.code, "invalid_condition_operator")

    def test_large_integer_condition_does_not_raise_overflow(self):
        entry = document()
        entry["clauses"][0]["value"] = 10 ** 400
        entry["clauses"][0]["conditions"] = [{"fact": "count", "op": "gte", "value": 10 ** 400}]
        result = self.assess(raw_catalog(entry), facts={**self.facts, "count": 10 ** 400})
        self.assertEqual(result["outcome"], "applicable")
        self.assertEqual(result["supported_claims"][0]["value"], 10 ** 400)

    def test_invalid_citation_unicode_is_unknown_without_body_leak(self):
        result = self.assess(citations=[{**CITATION, "answer": "PRIVATE_SENTINEL_" + "\ud800"}])
        self.assert_blocked(result, "unknown")
        self.assertEqual(result["reasons"][0]["code"], "invalid_citation_utf8")
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))

    def test_loader_binds_exact_file_bytes_and_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            body = json.dumps(raw_catalog(), ensure_ascii=False, indent=2).encode("utf-8")
            path.write_bytes(body)
            catalog = load_catalog(path)
            self.assertEqual(catalog.digest, hashlib.sha256(body).hexdigest())
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(CatalogValidationError) as caught:
                load_catalog(path)
            self.assertEqual(caught.exception.code, "duplicate_json_key")
            path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaises(CatalogValidationError) as caught:
                load_catalog(path)
            self.assertEqual(caught.exception.code, "nonfinite_json_number")

    def test_loader_rejects_remote_and_nul_paths_before_file_access(self):
        for path in ("https://example.invalid/catalog.json", "\\\\server\\catalog.json", "catalog\x00.json"):
            with self.subTest(path_kind=path[:2]):
                with self.assertRaises(CatalogValidationError) as caught:
                    load_catalog(path)
                self.assertEqual(caught.exception.code, "invalid_catalog_path")

    def test_renderer_exposes_structured_claims_only_for_applicable_result(self):
        rendered = render_assessment(self.assess())
        self.assertIn("申请期限（天）：7", rendered)
        self.assertNotIn("claim_key", rendered)
        self.assertNotIn("catalog_digest", rendered)
        blocked = self.assess(citations=[CITATION, {**CITATION, "id": 999}])
        self.assertNotIn("申请期限（天）：7", render_assessment(blocked))
        self.assertNotIn("PRIVATE_SENTINEL", render_assessment(self.assess(
            citations=[{**CITATION, "answer": "PRIVATE_SENTINEL"}])))


if __name__ == "__main__":
    unittest.main()
