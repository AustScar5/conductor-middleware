"""
tests/test_authorization.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pytest suite for the AuthorizationGate module and its integration
with AdjudicationOracle via gated_adjudicate().

Required paths
--------------
1. authorized_wallet_passes       — gate passes, oracle adjudicates normally
2. unauthorized_wallet_blocked    — not on allowlist → FAIL before oracle
3. suspended_wallet_blocked       — on allowlist but suspended → FAIL before oracle
4. malformed_wallet_rejected      — invalid address format → structured error verdict

Additional paths
----------------
5.  expired_wallet_blocked
6.  toml_config_loads_correctly
7.  case_insensitive_wallet_matching
8.  suspended_reason_surfaces_in_verdict
9.  gated_adjudicate_schema_fail_for_authorized
10. gated_adjudicate_returns_oracle_verdict_on_pass
11. evidence_hash_present_on_blocked_verdict
12. allowlist_config_rejects_duplicate_wallets
13. gate_with_empty_allowlist
14. add_entry_at_runtime
15. reload_config_from_disk
16. missing_config_file_raises_clearly

JSON logs for the four required paths are written to logs/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from conductor_middleware.adjudication import AdjudicationOracle, VerdictCode
from conductor_middleware.authorization import (
    AllowlistConfig,
    AuthorizationGate,
    AuthorizationVerdict,
    WalletEntry,
    WalletStatus,
    gated_adjudicate,
)

LOGS = Path(__file__).parent.parent / "logs"
LOGS.mkdir(exist_ok=True)

# ── Wallets used across tests ──────────────────────────────────────────────────

AUTHORIZED_WALLET   = "0xAABBCCDDEEFF00112233445566778899AABBCCDD"
AUTHORIZED_LOWER    = AUTHORIZED_WALLET.lower()
SECOND_AUTHORIZED   = "0x1111111111111111111111111111111111111111"
SUSPENDED_WALLET    = "0xDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF"
UNKNOWN_WALLET      = "0x0000000000000000000000000000000000000000"
EXPIRED_WALLET      = "0xEEEE000000000000000000000000000000000000"

MALFORMED_CASES = [
    "not-a-wallet",
    "0x",
    "0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",  # non-hex chars
    "AABBCCDDEE" * 4,                                  # missing 0x
    "",
    "   ",
]


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gate() -> AuthorizationGate:
    config_path = Path(__file__).parent.parent / "allowlist.json"
    return AuthorizationGate(config_path)


@pytest.fixture(scope="module")
def oracle() -> AdjudicationOracle:
    return AdjudicationOracle()


@pytest.fixture
def task_artifact() -> dict:
    return {"label": "positive", "confidence": 0.91, "reasoning": "Clear markers."}


class TaskOutput(BaseModel):
    label:      str
    confidence: float
    reasoning:  str


# ── Log helper ─────────────────────────────────────────────────────────────────

def write_log(name: str, auth_verdict: AuthorizationVerdict | None,
              adj_verdict=None) -> None:
    payload: dict = {"scenario": name, "logged_at": datetime.now(timezone.utc).isoformat()}
    if auth_verdict:
        payload["auth_verdict"] = auth_verdict.model_dump(mode="json")
    if adj_verdict:
        payload["adj_verdict"] = adj_verdict.model_dump(mode="json")
    (LOGS / f"authorization-{name}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Path 1 — Authorized wallet passes through to normal adjudication
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthorizedWalletPasses:

    def test_gate_check_returns_authorized(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET)
        assert v.status == WalletStatus.AUTHORIZED

    def test_is_authorized_property_true(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET)
        assert v.is_authorized is True

    def test_wallet_normalised_in_verdict(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET)
        assert v.wallet_address == AUTHORIZED_LOWER

    def test_gated_adjudicate_reaches_oracle(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, AUTHORIZED_WALLET, task_artifact, TaskOutput)
        assert v.verdict == VerdictCode.PASS

    def test_gated_adjudicate_evidence_hash_set(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, AUTHORIZED_WALLET, task_artifact, TaskOutput)
        assert len(v.evidence_hash) == 64

    def test_timestamp_is_utc(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET)
        assert v.timestamp.tzinfo is not None

    def test_pydantic_roundtrip(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET)
        AuthorizationVerdict.model_validate(v.model_dump(mode="json"))

    def test_writes_log(self, gate, oracle, task_artifact):
        auth = gate.gate_check(AUTHORIZED_WALLET)
        adj  = gated_adjudicate(gate, oracle, AUTHORIZED_WALLET, task_artifact, TaskOutput)
        write_log("authorized_wallet_passes", auth, adj)
        data = json.loads((LOGS / "authorization-authorized_wallet_passes.json").read_text())
        assert data["auth_verdict"]["status"] == "authorized"
        assert data["adj_verdict"]["verdict"] == "pass"


# ══════════════════════════════════════════════════════════════════════════════
# Path 2 — Unauthorized wallet is blocked before schema validation
# ══════════════════════════════════════════════════════════════════════════════

class TestUnauthorizedWalletBlocked:

    def test_gate_check_returns_unauthorized(self, gate):
        v = gate.gate_check(UNKNOWN_WALLET)
        assert v.status == WalletStatus.UNAUTHORIZED

    def test_is_authorized_false(self, gate):
        v = gate.gate_check(UNKNOWN_WALLET)
        assert v.is_authorized is False

    def test_reason_mentions_allowlist(self, gate):
        v = gate.gate_check(UNKNOWN_WALLET)
        assert "allowlist" in v.reason.lower()

    def test_gated_adjudicate_returns_fail(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, UNKNOWN_WALLET, task_artifact, TaskOutput)
        assert v.verdict == VerdictCode.FAIL

    def test_fail_reason_contains_unauthorized_contributor(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, UNKNOWN_WALLET, task_artifact, TaskOutput)
        assert "unauthorized_contributor" in v.reason

    def test_fail_verdict_not_retry_eligible(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, UNKNOWN_WALLET, task_artifact, TaskOutput)
        assert v.retry_eligible is False

    def test_oracle_never_called_for_invalid_schema(self, gate, oracle):
        # Even with a schema-breaking artifact the gate blocks before oracle
        v = gated_adjudicate(gate, oracle, UNKNOWN_WALLET, {}, TaskOutput)
        assert v.verdict == VerdictCode.FAIL
        assert "unauthorized_contributor" in v.reason

    def test_evidence_hash_present_even_when_blocked(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, UNKNOWN_WALLET, task_artifact, TaskOutput)
        assert len(v.evidence_hash) == 64

    def test_writes_log(self, gate, oracle, task_artifact):
        auth = gate.gate_check(UNKNOWN_WALLET)
        adj  = gated_adjudicate(gate, oracle, UNKNOWN_WALLET, task_artifact, TaskOutput)
        write_log("unauthorized_wallet_blocked", auth, adj)
        data = json.loads((LOGS / "authorization-unauthorized_wallet_blocked.json").read_text())
        assert data["auth_verdict"]["status"] == "unauthorized"
        assert data["adj_verdict"]["verdict"] == "fail"
        assert "unauthorized_contributor" in data["adj_verdict"]["reason"]


# ══════════════════════════════════════════════════════════════════════════════
# Path 3 — Suspended wallet is blocked
# ══════════════════════════════════════════════════════════════════════════════

class TestSuspendedWalletBlocked:

    def test_gate_check_returns_suspended(self, gate):
        v = gate.gate_check(SUSPENDED_WALLET)
        assert v.status == WalletStatus.SUSPENDED

    def test_is_authorized_false_for_suspended(self, gate):
        v = gate.gate_check(SUSPENDED_WALLET)
        assert v.is_authorized is False

    def test_suspension_reason_surfaced(self, gate):
        v = gate.gate_check(SUSPENDED_WALLET)
        assert "suspended" in v.reason.lower()

    def test_suspension_detail_included(self, gate):
        # allowlist.json carries suspension_reason for this wallet
        v = gate.gate_check(SUSPENDED_WALLET)
        assert "repeated verification failures" in v.reason

    def test_gated_adjudicate_returns_fail_for_suspended(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, SUSPENDED_WALLET, task_artifact, TaskOutput)
        assert v.verdict == VerdictCode.FAIL

    def test_fail_reason_contains_status_suspended(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, SUSPENDED_WALLET, task_artifact, TaskOutput)
        assert "suspended" in v.reason

    def test_writes_log(self, gate, oracle, task_artifact):
        auth = gate.gate_check(SUSPENDED_WALLET)
        adj  = gated_adjudicate(gate, oracle, SUSPENDED_WALLET, task_artifact, TaskOutput)
        write_log("suspended_wallet_blocked", auth, adj)
        data = json.loads((LOGS / "authorization-suspended_wallet_blocked.json").read_text())
        assert data["auth_verdict"]["status"] == "suspended"
        assert data["adj_verdict"]["verdict"] == "fail"


# ══════════════════════════════════════════════════════════════════════════════
# Path 4 — Malformed wallet input raises a structured validation error
# ══════════════════════════════════════════════════════════════════════════════

class TestMalformedWalletRejected:

    @pytest.mark.parametrize("bad_address", MALFORMED_CASES)
    def test_malformed_returns_verdict_not_raises(self, gate, bad_address):
        # gate_check must never raise — always returns a verdict
        v = gate.gate_check(bad_address)
        assert isinstance(v, AuthorizationVerdict)

    @pytest.mark.parametrize("bad_address", MALFORMED_CASES)
    def test_malformed_verdict_is_unauthorized(self, gate, bad_address):
        v = gate.gate_check(bad_address)
        assert v.status == WalletStatus.UNAUTHORIZED

    @pytest.mark.parametrize("bad_address", MALFORMED_CASES)
    def test_malformed_reason_says_malformed(self, gate, bad_address):
        v = gate.gate_check(bad_address)
        assert "malformed" in v.reason.lower() or "wallet" in v.reason.lower()

    def test_missing_0x_prefix_detected(self, gate):
        v = gate.gate_check("AABBCCDDEE112233445566778899AABBCCDDEE11")
        assert v.status == WalletStatus.UNAUTHORIZED
        assert "0x" in v.reason

    def test_wrong_length_detected(self, gate):
        v = gate.gate_check("0xABCD")   # too short
        assert v.status == WalletStatus.UNAUTHORIZED

    def test_gated_adjudicate_with_malformed_returns_fail(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, "not-a-wallet", task_artifact, TaskOutput)
        assert v.verdict == VerdictCode.FAIL
        assert "unauthorized_contributor" in v.reason

    def test_writes_log(self, gate, oracle, task_artifact):
        auth = gate.gate_check("not-a-wallet")
        adj  = gated_adjudicate(gate, oracle, "not-a-wallet", task_artifact, TaskOutput)
        write_log("malformed_wallet_rejected", auth, adj)
        data = json.loads((LOGS / "authorization-malformed_wallet_rejected.json").read_text())
        assert data["auth_verdict"]["status"] == "unauthorized"
        assert data["adj_verdict"]["verdict"] == "fail"


# ══════════════════════════════════════════════════════════════════════════════
# Additional paths
# ══════════════════════════════════════════════════════════════════════════════

class TestExpiredWallet:

    def test_expired_wallet_returns_expired_status(self, gate):
        v = gate.gate_check(EXPIRED_WALLET)
        assert v.status == WalletStatus.EXPIRED

    def test_expired_wallet_not_authorized(self, gate):
        v = gate.gate_check(EXPIRED_WALLET)
        assert v.is_authorized is False

    def test_expired_reason_mentions_expired(self, gate):
        v = gate.gate_check(EXPIRED_WALLET)
        assert "expired" in v.reason.lower()

    def test_gated_adjudicate_blocks_expired(self, gate, oracle, task_artifact):
        v = gated_adjudicate(gate, oracle, EXPIRED_WALLET, task_artifact, TaskOutput)
        assert v.verdict == VerdictCode.FAIL


class TestTomlConfig:

    def test_toml_loads_correctly(self):
        path = Path(__file__).parent.parent / "allowlist.toml"
        gate = AuthorizationGate(path)
        v = gate.gate_check(AUTHORIZED_WALLET)
        assert v.status == WalletStatus.AUTHORIZED

    def test_toml_suspended_wallet(self):
        path = Path(__file__).parent.parent / "allowlist.toml"
        gate = AuthorizationGate(path)
        v = gate.gate_check(SUSPENDED_WALLET)
        assert v.status == WalletStatus.SUSPENDED


class TestCaseInsensitiveMatching:

    def test_uppercase_wallet_matches(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET.upper())
        assert v.status == WalletStatus.AUTHORIZED

    def test_lowercase_wallet_matches(self, gate):
        v = gate.gate_check(AUTHORIZED_WALLET.lower())
        assert v.status == WalletStatus.AUTHORIZED

    def test_mixed_case_wallet_matches(self, gate):
        mixed = "0xaAbBcCdDeEfF00112233445566778899AaBbCcDd"
        v = gate.gate_check(mixed)
        assert v.status == WalletStatus.AUTHORIZED


class TestSchemaFailForAuthorizedWallet:

    def test_authorized_wallet_with_bad_artifact_gets_schema_fail(self, gate, oracle):
        v = gated_adjudicate(gate, oracle, AUTHORIZED_WALLET,
                             {"label": "positive"},   # missing fields
                             TaskOutput)
        assert v.verdict == VerdictCode.FAIL
        assert len(v.field_errors) >= 2   # confidence + reasoning missing

    def test_oracle_verdict_not_unauthorized_reason(self, gate, oracle):
        v = gated_adjudicate(gate, oracle, AUTHORIZED_WALLET,
                             {"label": "positive"}, TaskOutput)
        assert "unauthorized_contributor" not in v.reason


class TestRuntimeModification:

    def test_add_new_wallet_at_runtime(self):
        gate = AuthorizationGate()   # empty allowlist
        new_wallet = "0xAAAA000000000000000000000000000000000000"
        v_before = gate.gate_check(new_wallet)
        assert v_before.status == WalletStatus.UNAUTHORIZED

        gate.add(WalletEntry(wallet=new_wallet, status=WalletStatus.AUTHORIZED))
        v_after = gate.gate_check(new_wallet)
        assert v_after.status == WalletStatus.AUTHORIZED

    def test_reload_from_file(self, tmp_path):
        config = {"allowlist": [{"wallet": AUTHORIZED_WALLET, "status": "authorized"}]}
        config_file = tmp_path / "dynamic.json"
        config_file.write_text(json.dumps(config))

        gate = AuthorizationGate(config_file)
        assert gate.gate_check(AUTHORIZED_WALLET).status == WalletStatus.AUTHORIZED

        # Now rewrite the file to revoke access
        revoked = {"allowlist": []}
        config_file.write_text(json.dumps(revoked))
        gate.reload(config_file)
        assert gate.gate_check(AUTHORIZED_WALLET).status == WalletStatus.UNAUTHORIZED


class TestAllowlistValidation:

    def test_duplicate_wallets_rejected(self):
        raw = {
            "allowlist": [
                {"wallet": AUTHORIZED_WALLET, "status": "authorized"},
                {"wallet": AUTHORIZED_WALLET, "status": "authorized"},
            ]
        }
        with pytest.raises(Exception, match="[Dd]uplicate"):
            AllowlistConfig.model_validate(raw)

    def test_missing_config_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            AuthorizationGate(tmp_path / "nonexistent.json")

    def test_unsupported_file_format_raises(self, tmp_path):
        bad = tmp_path / "config.yaml"
        bad.write_text("allowlist: []")
        with pytest.raises(ValueError, match="Unsupported"):
            AuthorizationGate(bad)

    def test_empty_allowlist_blocks_everyone(self):
        gate = AuthorizationGate()
        v = gate.gate_check(AUTHORIZED_WALLET)
        assert v.status == WalletStatus.UNAUTHORIZED
