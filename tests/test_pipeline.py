"""
tests/test_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~
End-to-end pytest suite for TaskNodeConductor.

Required paths
--------------
1. full_success                — all four stages pass; payout blob returned
2. schema_validation_failure   — bad artifact; stops at verification, REJECTED
3. adjudication_retry          — reviewer policy rejects once, RETRY disposition
4. unauthorized_wallet_block   — wallet not on allowlist; stops at auth, BLOCKED
5. duplicate_verdict_hash      — same artifact submitted twice; DUPLICATE
6. dry_run_mode                — tx_blob returned without network submission

Additional paths
----------------
7.  suspended_wallet_blocked
8.  adjudication_hard_fail      — FAIL verdict (not RETRY)
9.  pipeline_result_audit_trail — all stage names present in order
10. stage_accessor_helper
11. passed_all_stages_property
12. missing_field_stops_at_verification
13. pydantic_roundtrip_pipeline_result
14. extra_validator_triggers_retry
15. xrp_address_accepted_in_allowlist

All tests use offline fixtures — no XRPL network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from xrpl.wallet import Wallet

from conductor_middleware.adjudication import ReviewerPolicy, VerdictCode
from conductor_middleware.authorization import (
    AllowlistConfig,
    AuthorizationGate,
    WalletEntry,
    WalletStatus,
)
from conductor_middleware.payout import IdempotencyStore, PayoutConfig
from conductor_middleware.pipeline import (
    Disposition,
    PipelineResult,
    StageVerdict,
    TaskNodeConductor,
)

LOGS = Path(__file__).parent.parent / "logs"
LOGS.mkdir(exist_ok=True)


# ── Artifact schema used in all tests ─────────────────────────────────────────

class TaskArtifact(BaseModel):
    summary:    str
    confidence: float
    sources:    list[str]


VALID_ARTIFACT = {
    "summary":    "Analysis complete.",
    "confidence": 0.92,
    "sources":    ["doc-1", "doc-2"],
}


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def issuer_wallet() -> Wallet:
    return Wallet.create()


@pytest.fixture(scope="session")
def contributor_wallet() -> Wallet:
    return Wallet.create()


@pytest.fixture(scope="session")
def unauthorized_wallet() -> Wallet:
    return Wallet.create()


@pytest.fixture(scope="session")
def suspended_wallet() -> Wallet:
    return Wallet.create()


@pytest.fixture(scope="session")
def payout_config(issuer_wallet) -> PayoutConfig:
    return PayoutConfig(
        issuer_seed=issuer_wallet.seed,
        pft_issuer=issuer_wallet.classic_address,
        pft_currency="PFT",
        pft_amount="100",
        base_fee_drops="12",
    )


@pytest.fixture(scope="session")
def gate(contributor_wallet, suspended_wallet) -> AuthorizationGate:
    return AuthorizationGate(config=AllowlistConfig(allowlist=[
        WalletEntry(wallet=contributor_wallet.classic_address,
                    status=WalletStatus.AUTHORIZED),
        WalletEntry(wallet=suspended_wallet.classic_address,
                    status=WalletStatus.SUSPENDED,
                    suspension_reason="test suspension"),
    ]))


def make_conductor(payout_config, gate, policy=None, extra=None, store=None) -> TaskNodeConductor:
    """Fresh conductor with its own idempotency store per test."""
    return TaskNodeConductor(
        gate=gate,
        payout_config=payout_config,
        reviewer_policy=policy,
        extra_validators=extra,
        idempotency_store=store if store is not None else IdempotencyStore(),
        # Inject sequence overrides via PayoutTrigger
    )


def make_conductor_with_seq(payout_config, gate, policy=None, extra=None, store=None):
    """Conductor whose PayoutTrigger uses fixed sequence numbers (offline-safe)."""
    from conductor_middleware.payout import PayoutTrigger
    c = TaskNodeConductor.__new__(TaskNodeConductor)
    from conductor_middleware.adjudication import AdjudicationOracle
    from conductor_middleware.oracle import VerificationOracle
    c._gate    = gate
    c._oracle  = AdjudicationOracle()
    c._policy  = policy
    c._extra   = extra or []
    c._store   = store if store is not None else IdempotencyStore()
    c._payout  = PayoutTrigger(
        config=payout_config,
        store=c._store,
        sequence_override=1,
        last_ledger_override=99_999_999,
    )
    return c


def write_log(name: str, result: PipelineResult) -> None:
    path = LOGS / f"pipeline-{name}.json"
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2,
                               default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Path 1 — Full success: all four stages pass, payout blob returned
# ══════════════════════════════════════════════════════════════════════════════

class TestFullSuccess:

    def test_disposition_is_completed(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.COMPLETED

    def test_all_four_stages_present(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        names = [s.stage for s in r.stages]
        assert names == ["verification", "adjudication", "authorization", "payout"]

    def test_all_stages_passed(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert all(s.passed for s in r.stages)

    def test_tx_blob_is_hex(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.tx_blob and len(r.tx_blob) > 100
        bytes.fromhex(r.tx_blob)  # must be valid hex

    def test_artifact_hash_set(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.artifact_hash and len(r.artifact_hash) == 64

    def test_passed_all_stages_true(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.passed_all_stages is True

    def test_timestamps_set(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.started_at is not None
        assert r.completed_at is not None
        assert r.completed_at >= r.started_at

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        write_log("full_success", r)
        data = json.loads((LOGS / "pipeline-full_success.json").read_text())
        assert data["disposition"] == "completed"
        assert len(data["stages"]) == 4


# ══════════════════════════════════════════════════════════════════════════════
# Path 2 — Schema validation failure stops at verification
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidationFailure:

    def test_disposition_is_rejected(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline({"summary": "only one field"}, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.REJECTED

    def test_only_verification_stage_recorded(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline({"summary": "only one field"}, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert len(r.stages) == 1
        assert r.stages[0].stage == "verification"

    def test_verification_stage_failed(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline({"summary": "only one field"}, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.stages[0].passed is False

    def test_field_errors_in_stage_detail(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline({"summary": "only one field"}, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        errors = r.stages[0].detail.get("field_errors", [])
        combined = " ".join(errors)
        assert "confidence" in combined or "sources" in combined

    def test_no_tx_blob_on_schema_fail(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline({}, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.tx_blob is None

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline({"summary": "bad"}, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        write_log("schema_validation_failure", r)
        data = json.loads((LOGS / "pipeline-schema_validation_failure.json").read_text())
        assert data["disposition"] == "rejected"
        assert data["stages"][0]["stage"] == "verification"


# ══════════════════════════════════════════════════════════════════════════════
# Path 3 — Adjudication rejection triggers RETRY disposition
# ══════════════════════════════════════════════════════════════════════════════

class TestAdjudicationRetry:

    @pytest.fixture
    def low_confidence_policy(self) -> ReviewerPolicy:
        def require_high_confidence(obj: TaskArtifact) -> str | None:
            if obj.confidence < 0.95:
                return f"confidence {obj.confidence:.2f} is below threshold 0.95"
        return ReviewerPolicy("high_confidence", [require_high_confidence],
                              retry_eligible=True)

    def test_disposition_is_retry(self, payout_config, gate, contributor_wallet,
                                   low_confidence_policy):
        c = make_conductor_with_seq(payout_config, gate, policy=low_confidence_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.RETRY

    def test_two_stages_recorded(self, payout_config, gate, contributor_wallet,
                                  low_confidence_policy):
        c = make_conductor_with_seq(payout_config, gate, policy=low_confidence_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        names = [s.stage for s in r.stages]
        assert names == ["verification", "adjudication"]

    def test_adjudication_stage_failed(self, payout_config, gate, contributor_wallet,
                                        low_confidence_policy):
        c = make_conductor_with_seq(payout_config, gate, policy=low_confidence_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        adj = r.stage("adjudication")
        assert adj is not None and adj.passed is False

    def test_artifact_hash_set_on_retry(self, payout_config, gate, contributor_wallet,
                                         low_confidence_policy):
        c = make_conductor_with_seq(payout_config, gate, policy=low_confidence_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.artifact_hash and len(r.artifact_hash) == 64

    def test_no_tx_blob_on_retry(self, payout_config, gate, contributor_wallet,
                                  low_confidence_policy):
        c = make_conductor_with_seq(payout_config, gate, policy=low_confidence_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.tx_blob is None

    def test_writes_log(self, payout_config, gate, contributor_wallet,
                         low_confidence_policy):
        c = make_conductor_with_seq(payout_config, gate, policy=low_confidence_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        write_log("adjudication_retry", r)
        data = json.loads((LOGS / "pipeline-adjudication_retry.json").read_text())
        assert data["disposition"] == "retry"


# ══════════════════════════════════════════════════════════════════════════════
# Path 4 — Unauthorized wallet blocks before payout
# ══════════════════════════════════════════════════════════════════════════════

class TestUnauthorizedWalletBlock:

    def test_disposition_is_blocked(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.BLOCKED

    def test_three_stages_recorded(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        names = [s.stage for s in r.stages]
        assert names == ["verification", "adjudication", "authorization"]

    def test_authorization_stage_failed(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        auth = r.stage("authorization")
        assert auth is not None and auth.passed is False

    def test_auth_stage_wallet_status_in_detail(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        assert r.stage("authorization").detail["wallet_status"] == "unauthorized"

    def test_no_tx_blob_when_blocked(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        assert r.tx_blob is None

    def test_writes_log(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        write_log("unauthorized_wallet_block", r)
        data = json.loads((LOGS / "pipeline-unauthorized_wallet_block.json").read_text())
        assert data["disposition"] == "blocked"


# ══════════════════════════════════════════════════════════════════════════════
# Path 5 — Duplicate verdict hash idempotency
# ══════════════════════════════════════════════════════════════════════════════

class TestDuplicateVerdictHash:

    def test_second_submission_is_duplicate(self, payout_config, gate, contributor_wallet):
        store = IdempotencyStore()
        c = make_conductor_with_seq(payout_config, gate, store=store)
        c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                       contributor_wallet.classic_address, dry_run=True)
        r2 = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                            contributor_wallet.classic_address, dry_run=True)
        assert r2.disposition == Disposition.DUPLICATE

    def test_first_call_succeeds(self, payout_config, gate, contributor_wallet):
        store = IdempotencyStore()
        c = make_conductor_with_seq(payout_config, gate, store=store)
        r1 = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                            contributor_wallet.classic_address, dry_run=True)
        assert r1.disposition == Disposition.COMPLETED

    def test_different_artifact_not_blocked(self, payout_config, gate, contributor_wallet):
        store = IdempotencyStore()
        c = make_conductor_with_seq(payout_config, gate, store=store)
        c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                       contributor_wallet.classic_address, dry_run=True)
        alt = {**VALID_ARTIFACT, "summary": "different summary"}
        r2 = c.run_pipeline(alt, TaskArtifact,
                            contributor_wallet.classic_address, dry_run=True)
        assert r2.disposition == Disposition.COMPLETED

    def test_duplicate_payout_stage_recorded(self, payout_config, gate, contributor_wallet):
        store = IdempotencyStore()
        c = make_conductor_with_seq(payout_config, gate, store=store)
        c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                       contributor_wallet.classic_address, dry_run=True)
        r2 = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                            contributor_wallet.classic_address, dry_run=True)
        payout_stage = r2.stage("payout")
        assert payout_stage is not None
        assert payout_stage.passed is False

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        store = IdempotencyStore()
        c = make_conductor_with_seq(payout_config, gate, store=store)
        c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                       contributor_wallet.classic_address, dry_run=True)
        r2 = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                            contributor_wallet.classic_address, dry_run=True)
        write_log("duplicate_verdict_hash", r2)
        data = json.loads((LOGS / "pipeline-duplicate_verdict_hash.json").read_text())
        assert data["disposition"] == "duplicate"


# ══════════════════════════════════════════════════════════════════════════════
# Path 6 — Dry-run mode returns blob without network submission
# ══════════════════════════════════════════════════════════════════════════════

class TestDryRunMode:

    def test_dry_run_true_in_result(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.dry_run is True

    def test_dry_run_tx_hash_is_none(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.tx_hash is None

    def test_dry_run_blob_present(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.tx_blob and len(r.tx_blob) > 100

    def test_dry_run_blob_decodes_correct_destination(self, payout_config, gate,
                                                       contributor_wallet):
        from xrpl.core.binarycodec import decode
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        decoded = decode(r.tx_blob)
        assert decoded["Destination"] == contributor_wallet.classic_address

    def test_dry_run_disposition_completed(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.COMPLETED

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        write_log("dry_run_mode", r)
        data = json.loads((LOGS / "pipeline-dry_run_mode.json").read_text())
        assert data["dry_run"] is True
        assert data["tx_hash"] is None
        assert len(data["tx_blob"]) > 100


# ══════════════════════════════════════════════════════════════════════════════
# Additional paths
# ══════════════════════════════════════════════════════════════════════════════

class TestSuspendedWalletBlocked:

    def test_suspended_wallet_is_blocked(self, payout_config, gate, suspended_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           suspended_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.BLOCKED
        assert r.stage("authorization").detail["wallet_status"] == "suspended"


class TestAdjudicationHardFail:

    def test_hard_reject_gives_rejected_not_retry(self, payout_config, gate,
                                                    contributor_wallet):
        hard_policy = ReviewerPolicy(
            "always_reject",
            [lambda obj: "always reject"],
            retry_eligible=False,
        )
        c = make_conductor_with_seq(payout_config, gate, policy=hard_policy)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.disposition == Disposition.REJECTED


class TestPipelineResultHelpers:

    def test_stage_accessor_returns_correct_stage(self, payout_config, gate,
                                                   contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        assert r.stage("verification") is not None
        assert r.stage("payout") is not None
        assert r.stage("nonexistent") is None

    def test_pydantic_roundtrip(self, payout_config, gate, contributor_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        PipelineResult.model_validate(r.model_dump(mode="json"))

    def test_passed_all_stages_false_on_blocked(self, payout_config, gate,
                                                 unauthorized_wallet):
        c = make_conductor_with_seq(payout_config, gate)
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           unauthorized_wallet.classic_address, dry_run=True)
        assert r.passed_all_stages is False


class TestExtraValidators:

    def test_extra_validator_stops_at_verification(self, payout_config, gate,
                                                    contributor_wallet):
        def require_many_sources(obj: TaskArtifact) -> str | None:
            if len(obj.sources) < 5:
                return f"need at least 5 sources, got {len(obj.sources)}"

        c = make_conductor_with_seq(payout_config, gate,
                                    extra=[require_many_sources])
        r = c.run_pipeline(VALID_ARTIFACT, TaskArtifact,
                           contributor_wallet.classic_address, dry_run=True)
        # VerificationOracle wraps extra_validators — failure → REJECTED
        assert r.disposition == Disposition.REJECTED
        assert r.stages[0].stage == "verification"
