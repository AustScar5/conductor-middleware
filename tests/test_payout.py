"""
tests/test_payout.py
~~~~~~~~~~~~~~~~~~~~~
Pytest suite for the PayoutTrigger module.

All tests are fully offline — no XRPL network calls are made.
Ephemeral test wallets are created with Wallet.create() for each run.

Required paths
--------------
1. authorized_pass_triggers_payout_construction
2. unauthorized_wallet_blocks_payout
3. failed_adjudication_blocks_payout
4. duplicate_verdict_hash_is_rejected
5. dry_run_returns_blob_without_network_submission

Additional paths
----------------
6.  suspended_wallet_blocks_payout
7.  both_verdicts_fail_reports_auth_first
8.  payout_result_fields_are_correct
9.  blob_is_valid_hex
10. blob_changes_with_different_wallet
11. idempotency_store_sqlite_persistence
12. idempotency_store_thread_safety
13. missing_issuer_seed_raises_value_error
14. missing_pft_issuer_raises_value_error
15. dry_run_still_marks_idempotency
16. second_dry_run_same_hash_raises_duplicate
17. payout_result_pydantic_roundtrip
18. payload_blocked_error_has_structured_details
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from xrpl.wallet import Wallet

from conductor_middleware.adjudication import AdjudicationVerdict, VerdictCode
from conductor_middleware.authorization import AuthorizationVerdict, WalletStatus
from conductor_middleware.payout import (
    DuplicateVerdictError,
    IdempotencyStore,
    PayoutBlockedError,
    PayoutConfig,
    PayoutRequest,
    PayoutResult,
    PayoutTrigger,
)

LOGS = Path(__file__).parent.parent / "logs"
LOGS.mkdir(exist_ok=True)

# ── Shared test wallets (created once per session) ────────────────────────────

@pytest.fixture(scope="session")
def issuer_wallet() -> Wallet:
    return Wallet.create()


@pytest.fixture(scope="session")
def contributor_wallet() -> Wallet:
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


@pytest.fixture
def trigger(payout_config) -> PayoutTrigger:
    """Fresh trigger with empty in-memory idempotency store per test."""
    return PayoutTrigger(
        config=payout_config,
        store=IdempotencyStore(),
        sequence_override=1,
        last_ledger_override=99_999_999,
    )


# ── Verdict factories ─────────────────────────────────────────────────────────

def make_auth_verdict(
    wallet: str,
    status: WalletStatus = WalletStatus.AUTHORIZED,
    reason: str = "Contributor is on the allowlist.",
) -> AuthorizationVerdict:
    return AuthorizationVerdict(
        wallet_address=wallet.lower(),
        status=status,
        reason=reason,
    )


def make_adj_verdict(
    verdict: VerdictCode = VerdictCode.PASS,
    evidence_hash: str | None = None,
    reason: str = "Artifact passed all checks.",
) -> AdjudicationVerdict:
    # Use a stable fake hash if not provided
    h = evidence_hash or ("a" * 64)
    return AdjudicationVerdict(
        verdict=verdict,
        reason=reason,
        evidence_hash=h,
        retry_eligible=False,
        schema_name="TestSchema",
    )


def make_request(
    contributor: Wallet,
    adj_verdict: AdjudicationVerdict | None = None,
    auth_status: WalletStatus = WalletStatus.AUTHORIZED,
    evidence_hash: str | None = None,
    task_id: str = "task-test-01",
) -> PayoutRequest:
    return PayoutRequest(
        adjudication_verdict=adj_verdict or make_adj_verdict(evidence_hash=evidence_hash),
        authorization_verdict=make_auth_verdict(contributor.classic_address, auth_status),
        contributor_wallet=contributor.classic_address,
        task_id=task_id,
    )


# ── Log helper ────────────────────────────────────────────────────────────────

def write_log(name: str, payload: dict) -> None:
    path = LOGS / f"payout-{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Path 1 — Authorized + pass triggers payout construction
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthorizedPassTriggersPayout:

    def test_execute_returns_payout_result(self, trigger, contributor_wallet):
        req    = make_request(contributor_wallet)
        result = trigger.execute(req, dry_run=True)
        assert isinstance(result, PayoutResult)

    def test_result_has_tx_blob(self, trigger, contributor_wallet):
        result = trigger.execute(make_request(contributor_wallet), dry_run=True)
        assert result.tx_blob
        assert len(result.tx_blob) > 0

    def test_result_contributor_matches(self, trigger, contributor_wallet):
        result = trigger.execute(make_request(contributor_wallet), dry_run=True)
        assert result.contributor == contributor_wallet.classic_address

    def test_result_pft_amount_matches_config(self, trigger, contributor_wallet, payout_config):
        result = trigger.execute(make_request(contributor_wallet), dry_run=True)
        assert result.pft_amount == payout_config.pft_amount

    def test_result_evidence_hash_matches_verdict(self, trigger, contributor_wallet):
        h   = "b" * 64
        req = make_request(contributor_wallet, evidence_hash=h)
        result = trigger.execute(req, dry_run=True)
        assert result.evidence_hash == h

    def test_result_task_id_preserved(self, trigger, contributor_wallet):
        req    = make_request(contributor_wallet, task_id="task-xyz")
        result = trigger.execute(req, dry_run=True)
        assert result.task_id == "task-xyz"

    def test_writes_log(self, trigger, contributor_wallet):
        req    = make_request(contributor_wallet, evidence_hash="c" * 64)
        result = trigger.execute(req, dry_run=True)
        write_log("authorized_pass_triggers_payout", result.model_dump(mode="json"))
        data = json.loads((LOGS / "payout-authorized_pass_triggers_payout.json").read_text())
        assert data["dry_run"] is True
        assert len(data["tx_blob"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Path 2 — Unauthorized wallet blocks payout
# ══════════════════════════════════════════════════════════════════════════════

class TestUnauthorizedWalletBlocksPayout:

    def test_raises_payout_blocked_error(self, trigger, contributor_wallet):
        req = make_request(contributor_wallet, auth_status=WalletStatus.UNAUTHORIZED)
        with pytest.raises(PayoutBlockedError):
            trigger.execute(req, dry_run=True)

    def test_error_reason_mentions_auth_gate(self, trigger, contributor_wallet):
        req = make_request(contributor_wallet, auth_status=WalletStatus.UNAUTHORIZED)
        with pytest.raises(PayoutBlockedError) as exc_info:
            trigger.execute(req, dry_run=True)
        assert "authorization" in exc_info.value.reason.lower()

    def test_error_details_blocked_by_auth(self, trigger, contributor_wallet):
        req = make_request(contributor_wallet, auth_status=WalletStatus.UNAUTHORIZED)
        with pytest.raises(PayoutBlockedError) as exc_info:
            trigger.execute(req, dry_run=True)
        assert exc_info.value.details["blocked_by"] == "authorization_gate"

    def test_idempotency_not_marked_when_blocked(self, trigger, contributor_wallet):
        h   = "d" * 64
        req = make_request(contributor_wallet, auth_status=WalletStatus.UNAUTHORIZED,
                           evidence_hash=h)
        with pytest.raises(PayoutBlockedError):
            trigger.execute(req, dry_run=True)
        assert not trigger._store.contains(h)

    def test_writes_log(self, trigger, contributor_wallet):
        req = make_request(contributor_wallet, auth_status=WalletStatus.UNAUTHORIZED,
                           evidence_hash="e" * 64)
        try:
            trigger.execute(req, dry_run=True)
        except PayoutBlockedError as err:
            write_log("unauthorized_wallet_blocked",
                      {"blocked_by": err.details.get("blocked_by"),
                       "reason": err.reason})
        data = json.loads((LOGS / "payout-unauthorized_wallet_blocked.json").read_text())
        assert data["blocked_by"] == "authorization_gate"


# ══════════════════════════════════════════════════════════════════════════════
# Path 3 — Failed adjudication blocks payout
# ══════════════════════════════════════════════════════════════════════════════

class TestFailedAdjudicationBlocksPayout:

    def test_raises_payout_blocked_on_adj_fail(self, trigger, contributor_wallet):
        adj = make_adj_verdict(verdict=VerdictCode.FAIL, evidence_hash="f" * 64,
                               reason="Schema validation failed.")
        req = make_request(contributor_wallet, adj_verdict=adj)
        with pytest.raises(PayoutBlockedError):
            trigger.execute(req, dry_run=True)

    def test_raises_payout_blocked_on_adj_retry(self, trigger, contributor_wallet):
        adj = make_adj_verdict(verdict=VerdictCode.RETRY, evidence_hash="0" * 64)
        req = make_request(contributor_wallet, adj_verdict=adj)
        with pytest.raises(PayoutBlockedError):
            trigger.execute(req, dry_run=True)

    def test_error_details_blocked_by_oracle(self, trigger, contributor_wallet):
        adj = make_adj_verdict(verdict=VerdictCode.FAIL, evidence_hash="1" * 64)
        req = make_request(contributor_wallet, adj_verdict=adj)
        with pytest.raises(PayoutBlockedError) as exc_info:
            trigger.execute(req, dry_run=True)
        assert exc_info.value.details["blocked_by"] == "adjudication_oracle"

    def test_error_reason_contains_verdict_value(self, trigger, contributor_wallet):
        adj = make_adj_verdict(verdict=VerdictCode.FAIL, evidence_hash="2" * 64,
                               reason="Bad schema.")
        req = make_request(contributor_wallet, adj_verdict=adj)
        with pytest.raises(PayoutBlockedError) as exc_info:
            trigger.execute(req, dry_run=True)
        assert "fail" in exc_info.value.reason.lower()

    def test_writes_log(self, trigger, contributor_wallet):
        adj = make_adj_verdict(verdict=VerdictCode.FAIL, evidence_hash="3" * 64)
        req = make_request(contributor_wallet, adj_verdict=adj)
        try:
            trigger.execute(req, dry_run=True)
        except PayoutBlockedError as err:
            write_log("failed_adjudication_blocks_payout",
                      {"blocked_by": err.details.get("blocked_by"),
                       "reason": err.reason})
        data = json.loads((LOGS / "payout-failed_adjudication_blocks_payout.json").read_text())
        assert data["blocked_by"] == "adjudication_oracle"


# ══════════════════════════════════════════════════════════════════════════════
# Path 4 — Duplicate verdict hash is rejected
# ══════════════════════════════════════════════════════════════════════════════

class TestDuplicateVerdictHashRejected:

    def test_second_call_same_hash_raises_duplicate(self, trigger, contributor_wallet):
        h   = "4" * 64
        req = make_request(contributor_wallet, evidence_hash=h)
        trigger.execute(req, dry_run=True)   # first call: ok
        with pytest.raises(DuplicateVerdictError):
            trigger.execute(req, dry_run=True)   # second call: rejected

    def test_duplicate_error_contains_hash(self, trigger, contributor_wallet):
        h   = "5" * 64
        req = make_request(contributor_wallet, evidence_hash=h)
        trigger.execute(req, dry_run=True)
        with pytest.raises(DuplicateVerdictError) as exc_info:
            trigger.execute(req, dry_run=True)
        assert exc_info.value.evidence_hash == h

    def test_different_hash_not_blocked(self, trigger, contributor_wallet):
        req1 = make_request(contributor_wallet, evidence_hash="6" * 64)
        req2 = make_request(contributor_wallet, evidence_hash="7" * 64)
        trigger.execute(req1, dry_run=True)
        result = trigger.execute(req2, dry_run=True)   # different hash — ok
        assert isinstance(result, PayoutResult)

    def test_store_grows_after_each_payout(self, trigger, contributor_wallet):
        assert len(trigger._store) == 0
        trigger.execute(make_request(contributor_wallet, evidence_hash="8" * 64), dry_run=True)
        assert len(trigger._store) == 1
        trigger.execute(make_request(contributor_wallet, evidence_hash="9" * 64), dry_run=True)
        assert len(trigger._store) == 2

    def test_writes_log(self, trigger, contributor_wallet):
        h   = "aa" * 32
        req = make_request(contributor_wallet, evidence_hash=h)
        trigger.execute(req, dry_run=True)
        try:
            trigger.execute(req, dry_run=True)
        except DuplicateVerdictError as err:
            write_log("duplicate_verdict_hash_rejected",
                      {"evidence_hash": err.evidence_hash,
                       "error": str(err)})
        data = json.loads((LOGS / "payout-duplicate_verdict_hash_rejected.json").read_text())
        assert data["evidence_hash"] == h


# ══════════════════════════════════════════════════════════════════════════════
# Path 5 — dry_run returns serialised blob without network submission
# ══════════════════════════════════════════════════════════════════════════════

class TestDryRunReturnsBlobWithoutNetworkCall:

    def test_dry_run_true_returns_result(self, trigger, contributor_wallet):
        req    = make_request(contributor_wallet, evidence_hash="bb" * 32)
        result = trigger.execute(req, dry_run=True)
        assert result.dry_run is True

    def test_dry_run_tx_hash_is_none(self, trigger, contributor_wallet):
        # tx_hash is only set after live submission
        result = trigger.execute(
            make_request(contributor_wallet, evidence_hash="cc" * 32), dry_run=True
        )
        assert result.tx_hash is None

    def test_dry_run_blob_is_hex_string(self, trigger, contributor_wallet):
        result = trigger.execute(
            make_request(contributor_wallet, evidence_hash="dd" * 32), dry_run=True
        )
        assert isinstance(result.tx_blob, str)
        # Must be valid hex
        bytes.fromhex(result.tx_blob)

    def test_dry_run_blob_encodes_correct_destination(self, trigger, contributor_wallet):
        from xrpl.core.binarycodec import decode
        result = trigger.execute(
            make_request(contributor_wallet, evidence_hash="ee" * 32), dry_run=True
        )
        decoded = decode(result.tx_blob)
        assert decoded["Destination"] == contributor_wallet.classic_address

    def test_dry_run_blob_encodes_correct_currency(self, trigger, contributor_wallet):
        from xrpl.core.binarycodec import decode
        result = trigger.execute(
            make_request(contributor_wallet, evidence_hash="ff" * 32), dry_run=True
        )
        decoded = decode(result.tx_blob)
        assert decoded["Amount"]["currency"] == "PFT"

    def test_dry_run_marks_idempotency(self, trigger, contributor_wallet):
        h   = "ab" * 32
        req = make_request(contributor_wallet, evidence_hash=h)
        trigger.execute(req, dry_run=True)
        assert trigger._store.contains(h)

    def test_dry_run_second_call_raises_duplicate(self, trigger, contributor_wallet):
        h   = "cd" * 32
        req = make_request(contributor_wallet, evidence_hash=h)
        trigger.execute(req, dry_run=True)
        with pytest.raises(DuplicateVerdictError):
            trigger.execute(req, dry_run=True)

    def test_writes_log(self, trigger, contributor_wallet):
        result = trigger.execute(
            make_request(contributor_wallet, evidence_hash="ef" * 32), dry_run=True
        )
        write_log("dry_run_returns_blob", result.model_dump(mode="json"))
        data = json.loads((LOGS / "payout-dry_run_returns_blob.json").read_text())
        assert data["dry_run"] is True
        assert data["tx_hash"] is None
        assert len(data["tx_blob"]) > 100


# ══════════════════════════════════════════════════════════════════════════════
# Additional paths
# ══════════════════════════════════════════════════════════════════════════════

class TestSuspendedWalletBlocked:

    def test_suspended_raises_payout_blocked(self, trigger, contributor_wallet):
        req = make_request(contributor_wallet, auth_status=WalletStatus.SUSPENDED)
        with pytest.raises(PayoutBlockedError) as exc_info:
            trigger.execute(req, dry_run=True)
        assert "suspended" in exc_info.value.reason.lower()


class TestMissingCredentials:

    def test_missing_seed_raises_value_error(self, contributor_wallet):
        cfg     = PayoutConfig(issuer_seed="", pft_issuer="rSomeAddress")
        trigger = PayoutTrigger(config=cfg, sequence_override=1, last_ledger_override=99999999)
        req     = make_request(contributor_wallet, evidence_hash="10" * 32)
        with pytest.raises(ValueError, match="XRPL_ISSUER_SEED"):
            trigger.execute(req, dry_run=True)

    def test_missing_pft_issuer_raises_value_error(self, issuer_wallet, contributor_wallet):
        cfg     = PayoutConfig(issuer_seed=issuer_wallet.seed, pft_issuer="")
        trigger = PayoutTrigger(config=cfg, sequence_override=1, last_ledger_override=99999999)
        req     = make_request(contributor_wallet, evidence_hash="11" * 32)
        with pytest.raises(ValueError, match="XRPL_PFT_ISSUER"):
            trigger.execute(req, dry_run=True)


class TestIdempotencyStoreSqlite:

    def test_sqlite_store_persists_across_instances(self, tmp_path, contributor_wallet):
        # Use a locally-scoped config — session-scoped fixtures cause _conn=None
        # in IdempotencyStore due to pytest fixture teardown ordering.
        iw  = Wallet.create()
        cfg = PayoutConfig(issuer_seed=iw.seed, pft_issuer=iw.classic_address,
                           pft_currency="PFT", pft_amount="100", base_fee_drops="12")
        db = tmp_path / "idem.db"
        h  = "12" * 32

        t1 = PayoutTrigger(config=cfg, store=IdempotencyStore(db),
                           sequence_override=1, last_ledger_override=99999999)
        t1.execute(make_request(contributor_wallet, evidence_hash=h), dry_run=True)

        # Verify DB was actually written before creating t2
        import sqlite3 as _sq
        rows = _sq.connect(str(db)).execute("SELECT count(*) FROM paid_verdicts").fetchone()
        assert rows[0] == 1, f"Expected 1 DB row after t1, got {rows[0]}"

        # New trigger instance, same DB file — must detect the duplicate
        t2 = PayoutTrigger(config=cfg, store=IdempotencyStore(db),
                           sequence_override=1, last_ledger_override=99999999)
        with pytest.raises(DuplicateVerdictError):
            t2.execute(make_request(contributor_wallet, evidence_hash=h), dry_run=True)

    def test_sqlite_store_thread_safety(self, tmp_path, payout_config):
        db      = tmp_path / "threaded.db"
        store   = IdempotencyStore(db)
        hashes  = [f"{i:064x}" for i in range(20)]
        errors: list[Exception] = []

        def mark(h: str) -> None:
            try:
                store.mark(h)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=mark, args=(h,)) for h in hashes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 20


class TestPayoutResultSchema:

    def test_pydantic_roundtrip(self, trigger, contributor_wallet):
        result = trigger.execute(
            make_request(contributor_wallet, evidence_hash="13" * 32), dry_run=True
        )
        raw = result.model_dump(mode="json")
        PayoutResult.model_validate(raw)

    def test_both_blocked_raises_auth_first(self, trigger, contributor_wallet):
        adj = make_adj_verdict(verdict=VerdictCode.FAIL, evidence_hash="14" * 32)
        req = PayoutRequest(
            adjudication_verdict=adj,
            authorization_verdict=make_auth_verdict(
                contributor_wallet.classic_address, WalletStatus.SUSPENDED
            ),
            contributor_wallet=contributor_wallet.classic_address,
        )
        with pytest.raises(PayoutBlockedError) as exc_info:
            trigger.execute(req, dry_run=True)
        # Auth is checked first
        assert exc_info.value.details["blocked_by"] == "authorization_gate"
