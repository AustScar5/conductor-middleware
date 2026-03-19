"""
tests/test_adjudication.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pytest suite covering the AdjudicationOracle module.

Required paths
--------------
1. valid_artifact_passes          — schema matches, no reviewer → PASS
2. schema_invalid_artifact_fails  — missing/wrong field → FAIL, field_errors set
3. reviewer_rejected_returns_retry — schema valid, reviewer rejects → RETRY
4. tampered_evidence_hash_detected — artifact modified after hashing → mismatch

Additional paths
----------------
5. hard_reviewer_rejection        — reviewer.retry_eligible=False → FAIL
6. multiple_field_errors          — two missing fields → two field_errors entries
7. evidence_hash_is_deterministic — same artifact always yields same hash
8. evidence_hash_changes_on_mutation
9. log_records_every_verdict      — AdjudicationLog captures all calls
10. log_verify_record_integrity    — log.verify_record_integrity detects tamper
11. log_export_structure          — export() is JSON-round-trippable
12. reviewer_with_multiple_validators
13. schema_name_in_verdict
14. retry_eligible_false_on_schema_fail
15. retry_eligible_true_on_reviewer_retry

Each test that produces a TerminalResult/verdict writes a structured JSON log
to logs/adjudication-<scenario>.json.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from conductor_middleware.adjudication import (
    AdjudicationLog,
    AdjudicationOracle,
    AdjudicationRecord,
    AdjudicationVerdict,
    ReviewerPolicy,
    VerdictCode,
)

LOGS = Path(__file__).parent.parent / "logs"
LOGS.mkdir(exist_ok=True)


# ── Shared schemas ─────────────────────────────────────────────────────────────

class SentimentArtifact(BaseModel):
    label:      str
    confidence: float
    reasoning:  str


class DeliveryArtifact(BaseModel):
    summary:  str
    word_count: int
    sources:  list[str]


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def oracle() -> AdjudicationOracle:
    return AdjudicationOracle()


@pytest.fixture
def logged_oracle() -> tuple[AdjudicationOracle, AdjudicationLog]:
    log    = AdjudicationLog()
    oracle = AdjudicationOracle(log=log)
    return oracle, log


@pytest.fixture
def valid_sentiment() -> dict:
    return {"label": "positive", "confidence": 0.92, "reasoning": "Strong positive markers."}


@pytest.fixture
def valid_delivery() -> dict:
    return {"summary": "Task complete.", "word_count": 2, "sources": ["doc-1", "doc-2"]}


# ── Helper ─────────────────────────────────────────────────────────────────────

def write_log(name: str, verdict: AdjudicationVerdict) -> None:
    payload = {
        "scenario":   name,
        "logged_at":  datetime.now(timezone.utc).isoformat(),
        "verdict":    verdict.model_dump(mode="json"),
    }
    (LOGS / f"adjudication-{name}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Path 1 — Valid artifact passes
# ══════════════════════════════════════════════════════════════════════════════

class TestValidArtifactPasses:

    def test_verdict_is_pass(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert v.verdict == VerdictCode.PASS

    def test_evidence_hash_is_64_hex_chars(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert len(v.evidence_hash) == 64
        int(v.evidence_hash, 16)   # must be valid hex

    def test_no_field_errors_on_pass(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert v.field_errors == []

    def test_retry_eligible_false_on_pass(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert v.retry_eligible is False

    def test_schema_name_recorded(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert v.schema_name == "SentimentArtifact"

    def test_timestamp_is_utc(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert v.timestamp.tzinfo is not None

    def test_pydantic_roundtrip(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        AdjudicationVerdict.model_validate(v.model_dump(mode="json"))

    def test_writes_log(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        write_log("valid_artifact_passes", v)
        data = json.loads((LOGS / "adjudication-valid_artifact_passes.json").read_text())
        assert data["verdict"]["verdict"] == "pass"
        assert len(data["verdict"]["evidence_hash"]) == 64


# ══════════════════════════════════════════════════════════════════════════════
# Path 2 — Schema-invalid artifact fails with specific field errors
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaInvalidArtifactFails:

    def test_verdict_is_fail(self, oracle):
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        assert v.verdict == VerdictCode.FAIL

    def test_field_errors_populated(self, oracle):
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        assert len(v.field_errors) >= 2   # confidence + reasoning missing

    def test_field_errors_name_missing_fields(self, oracle):
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        combined = " ".join(v.field_errors)
        assert "confidence" in combined
        assert "reasoning"  in combined

    def test_field_errors_format_is_path_colon_message(self, oracle):
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        for err in v.field_errors:
            assert ":" in err, f"Expected 'field: message' format, got {err!r}"

    def test_retry_eligible_false_on_schema_fail(self, oracle):
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        assert v.retry_eligible is False

    def test_evidence_hash_still_set_on_fail(self, oracle):
        # Hash is computed before schema validation — always present
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        assert len(v.evidence_hash) == 64

    def test_wrong_type_produces_field_error(self, oracle):
        v = oracle.adjudicate(
            {"label": "positive", "confidence": "not-a-float", "reasoning": "x"},
            SentimentArtifact,
        )
        assert v.verdict == VerdictCode.FAIL
        assert any("confidence" in e for e in v.field_errors)

    def test_writes_log(self, oracle):
        v = oracle.adjudicate({"label": "positive"}, SentimentArtifact)
        write_log("schema_invalid_artifact_fails", v)
        data = json.loads((LOGS / "adjudication-schema_invalid_artifact_fails.json").read_text())
        assert data["verdict"]["verdict"] == "fail"
        assert len(data["verdict"]["field_errors"]) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# Path 3 — Reviewer-rejected artifact returns RETRY verdict
# ══════════════════════════════════════════════════════════════════════════════

class TestReviewerRejectedArtifactReturnsRetry:

    @pytest.fixture
    def low_confidence_policy(self) -> ReviewerPolicy:
        def require_high_confidence(obj: SentimentArtifact) -> str | None:
            if obj.confidence < 0.85:
                return f"confidence {obj.confidence:.2f} is below threshold 0.85"

        return ReviewerPolicy(
            "require_high_confidence",
            [require_high_confidence],
            retry_eligible=True,
        )

    @pytest.fixture
    def low_confidence_artifact(self) -> dict:
        return {"label": "positive", "confidence": 0.60, "reasoning": "Somewhat positive."}

    def test_verdict_is_retry(self, oracle, low_confidence_artifact, low_confidence_policy):
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, low_confidence_policy)
        assert v.verdict == VerdictCode.RETRY

    def test_retry_eligible_true(self, oracle, low_confidence_artifact, low_confidence_policy):
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, low_confidence_policy)
        assert v.retry_eligible is True

    def test_reason_contains_policy_name(self, oracle, low_confidence_artifact, low_confidence_policy):
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, low_confidence_policy)
        assert "require_high_confidence" in v.reason

    def test_reason_contains_rejection_message(self, oracle, low_confidence_artifact, low_confidence_policy):
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, low_confidence_policy)
        assert "threshold" in v.reason

    def test_no_field_errors_on_reviewer_retry(self, oracle, low_confidence_artifact, low_confidence_policy):
        # Schema was valid — field_errors should be empty
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, low_confidence_policy)
        assert v.field_errors == []

    def test_passing_artifact_bypasses_reviewer(self, oracle, valid_sentiment, low_confidence_policy):
        # High-confidence artifact should pass even with the strict policy
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact, low_confidence_policy)
        assert v.verdict == VerdictCode.PASS

    def test_hard_reviewer_rejection_produces_fail(self, oracle, low_confidence_artifact):
        hard_policy = ReviewerPolicy(
            "hard_reject",
            [lambda obj: "always reject"],
            retry_eligible=False,
        )
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, hard_policy)
        assert v.verdict == VerdictCode.FAIL
        assert v.retry_eligible is False

    def test_writes_log(self, oracle, low_confidence_artifact, low_confidence_policy):
        v = oracle.adjudicate(low_confidence_artifact, SentimentArtifact, low_confidence_policy)
        write_log("reviewer_rejected_returns_retry", v)
        data = json.loads((LOGS / "adjudication-reviewer_rejected_returns_retry.json").read_text())
        assert data["verdict"]["verdict"] == "retry"
        assert data["verdict"]["retry_eligible"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Path 4 — Tampered artifact is detected via evidence hash mismatch
# ══════════════════════════════════════════════════════════════════════════════

class TestTamperedArtifactDetected:

    def test_original_artifact_verifies_clean(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert oracle.verify_integrity(valid_sentiment, v.evidence_hash) is True

    def test_mutated_value_detected(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        tampered = {**valid_sentiment, "confidence": 0.01}   # value changed
        assert oracle.verify_integrity(tampered, v.evidence_hash) is False

    def test_added_field_detected(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        tampered = {**valid_sentiment, "injected": "extra_field"}
        assert oracle.verify_integrity(tampered, v.evidence_hash) is False

    def test_removed_field_detected(self, oracle, valid_sentiment):
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        tampered = {k: val for k, val in valid_sentiment.items() if k != "reasoning"}
        assert oracle.verify_integrity(tampered, v.evidence_hash) is False

    def test_hash_is_deterministic(self, oracle, valid_sentiment):
        v1 = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        v2 = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert v1.evidence_hash == v2.evidence_hash

    def test_hash_independent_of_dict_insertion_order(self, oracle):
        a = {"label": "positive", "confidence": 0.9, "reasoning": "good"}
        b = {"reasoning": "good", "label": "positive", "confidence": 0.9}
        va = oracle.adjudicate(a, SentimentArtifact)
        vb = oracle.adjudicate(b, SentimentArtifact)
        assert va.evidence_hash == vb.evidence_hash

    def test_log_verify_record_integrity_clean(self, logged_oracle, valid_sentiment):
        oracle, log = logged_oracle
        v = oracle.adjudicate(valid_sentiment, SentimentArtifact, task_id="task-42")
        rec_id = log._records[-1].record_id
        assert log.verify_record_integrity(rec_id, valid_sentiment) is True

    def test_log_verify_record_integrity_tampered(self, logged_oracle, valid_sentiment):
        oracle, log = logged_oracle
        oracle.adjudicate(valid_sentiment, SentimentArtifact, task_id="task-43")
        rec_id   = log._records[-1].record_id
        tampered = {**valid_sentiment, "label": "negative"}
        assert log.verify_record_integrity(rec_id, tampered) is False

    def test_writes_log(self, oracle, valid_sentiment):
        v          = oracle.adjudicate(valid_sentiment, SentimentArtifact)
        tampered   = {**valid_sentiment, "confidence": 0.01}
        is_clean   = oracle.verify_integrity(valid_sentiment, v.evidence_hash)
        is_tampered = oracle.verify_integrity(tampered, v.evidence_hash)
        payload = {
            "scenario":         "tampered_evidence_hash_detected",
            "original_verdict": v.model_dump(mode="json"),
            "integrity_checks": {
                "original_matches":  is_clean,
                "tampered_matches":  is_tampered,
            },
        }
        path = LOGS / "adjudication-tampered_evidence_hash_detected.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        data = json.loads(path.read_text())
        assert data["integrity_checks"]["original_matches"] is True
        assert data["integrity_checks"]["tampered_matches"] is False


# ══════════════════════════════════════════════════════════════════════════════
# AdjudicationLog
# ══════════════════════════════════════════════════════════════════════════════

class TestAdjudicationLog:

    def test_log_records_all_calls(self, logged_oracle, valid_sentiment):
        oracle, log = logged_oracle
        oracle.adjudicate(valid_sentiment, SentimentArtifact)
        oracle.adjudicate({"label": "x"}, SentimentArtifact)    # fail
        assert len(log) == 2

    def test_log_export_is_json_serialisable(self, logged_oracle, valid_sentiment):
        oracle, log = logged_oracle
        oracle.adjudicate(valid_sentiment, SentimentArtifact)
        raw    = log.export()
        dumped = json.dumps(raw, default=str)
        parsed = json.loads(dumped)
        assert parsed["record_count"] == 1

    def test_log_export_contains_evidence_hash(self, logged_oracle, valid_sentiment):
        oracle, log = logged_oracle
        oracle.adjudicate(valid_sentiment, SentimentArtifact)
        rec = log.export()["records"][0]
        assert len(rec["verdict"]["evidence_hash"]) == 64

    def test_log_unknown_record_id_raises(self, logged_oracle):
        oracle, log = logged_oracle
        with pytest.raises(KeyError):
            log.verify_record_integrity("nonexistent", {})

    def test_log_persists_to_file(self, tmp_path, valid_sentiment):
        log_path = tmp_path / "audit.json"
        log      = AdjudicationLog(path=log_path)
        oracle   = AdjudicationOracle(log=log)
        oracle.adjudicate(valid_sentiment, SentimentArtifact)
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert data["record_count"] == 1

    def test_log_task_id_recorded(self, logged_oracle, valid_sentiment):
        oracle, log = logged_oracle
        oracle.adjudicate(valid_sentiment, SentimentArtifact, task_id="task-99")
        assert log._records[-1].task_id == "task-99"
