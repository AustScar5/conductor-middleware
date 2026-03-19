"""
tests/test_mcp_server.py
~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end tests for the MCP server tool handlers.

Tests call the handler functions directly (no network/transport layer needed)
so they run offline and are fast. Transport integration is verified by a
separate smoke test that checks the server builds without error.

Paths covered
-------------
1. run_pipeline — authorized wallet, valid artifact → COMPLETED + blob
2. run_pipeline — unauthorized wallet → BLOCKED before payout
3. run_pipeline — schema failure → REJECTED at verification
4. check_authorization — authorized, unauthorized, suspended wallets
5. dry_run_payout — authorized wallet → blob without submission
6. dry_run_payout — unauthorized wallet → structured error, no blob
7. duplicate evidence hash → structured duplicate error
8. MCP Server builds and lists three tools correctly
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest
from xrpl.wallet import Wallet

from conductor_middleware.adjudication import AdjudicationOracle
from conductor_middleware.authorization import (
    AllowlistConfig, AuthorizationGate, WalletEntry, WalletStatus,
)
from conductor_middleware.mcp_server import (
    TOOL_CHECK_AUTHORIZATION,
    TOOL_DRY_RUN_PAYOUT,
    TOOL_RUN_PIPELINE,
    _handle_check_authorization,
    _handle_dry_run_payout,
    _handle_run_pipeline,
    build_server,
)
from conductor_middleware.payout import IdempotencyStore, PayoutConfig, PayoutTrigger
from conductor_middleware.pipeline import Disposition, TaskNodeConductor

LOGS = Path(__file__).parent.parent / "logs"
LOGS.mkdir(exist_ok=True)


# ── Session-scoped fixtures ────────────────────────────────────────────────────

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


def make_conductor(payout_config, gate) -> TaskNodeConductor:
    """Fresh conductor with isolated idempotency store per test."""
    return TaskNodeConductor(
        gate=gate,
        payout_config=payout_config,
        idempotency_store=IdempotencyStore(),
    )


def make_trigger(payout_config, store=None) -> PayoutTrigger:
    return PayoutTrigger(
        config=payout_config,
        store=store if store is not None else IdempotencyStore(),
        sequence_override=1,
        last_ledger_override=99_999_999,
    )


VALID_ARTIFACT   = {"summary": "Analysis complete.", "confidence": 0.92}
VALID_SCHEMA_DEF = {"summary": "str", "confidence": "float"}


def write_log(name: str, result: dict) -> None:
    (LOGS / f"mcp-{name}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )


# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline — authorized wallet
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipelineAuthorized:

    def test_disposition_completed(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        assert r["disposition"] == "completed"

    def test_four_stages_in_result(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        assert [s["stage"] for s in r["stages"]] == [
            "verification", "adjudication", "authorization", "payout"
        ]

    def test_tx_blob_in_result(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        assert r["tx_blob"] and len(r["tx_blob"]) > 100
        bytes.fromhex(r["tx_blob"])

    def test_artifact_hash_set(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        assert r["artifact_hash"] and len(r["artifact_hash"]) == 64

    def test_result_is_json_serialisable(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        json.dumps(r, default=str)  # must not raise

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        write_log("run_pipeline_authorized", r)
        data = json.loads((LOGS / "mcp-run_pipeline_authorized.json").read_text())
        assert data["disposition"] == "completed"
        assert len(data["stages"]) == 4


# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline — unauthorized wallet blocked
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipelineUnauthorized:

    def test_disposition_blocked(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": unauthorized_wallet.classic_address,
            "dry_run": True,
        })
        assert r["disposition"] == "blocked"

    def test_three_stages_only(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": unauthorized_wallet.classic_address,
            "dry_run": True,
        })
        names = [s["stage"] for s in r["stages"]]
        assert names == ["verification", "adjudication", "authorization"]
        assert "payout" not in names

    def test_no_tx_blob_when_blocked(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": unauthorized_wallet.classic_address,
            "dry_run": True,
        })
        assert r["tx_blob"] is None

    def test_writes_log(self, payout_config, gate, unauthorized_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": VALID_ARTIFACT,
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": unauthorized_wallet.classic_address,
            "dry_run": True,
        })
        write_log("run_pipeline_unauthorized", r)
        data = json.loads((LOGS / "mcp-run_pipeline_unauthorized.json").read_text())
        assert data["disposition"] == "blocked"
        assert data["tx_blob"] is None


# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline — schema failure stops at verification
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipelineSchemaFailure:

    def test_disposition_rejected(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": {"only_bad_field": True},
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        assert r["disposition"] == "rejected"

    def test_only_verification_stage(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": {},
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        assert len(r["stages"]) == 1
        assert r["stages"][0]["stage"] == "verification"
        assert r["stages"][0]["passed"] is False

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        c = make_conductor(payout_config, gate)
        r = _handle_run_pipeline(c, {
            "artifact": {},
            "schema_fields": VALID_SCHEMA_DEF,
            "contributor_wallet": contributor_wallet.classic_address,
            "dry_run": True,
        })
        write_log("run_pipeline_schema_failure", r)
        data = json.loads((LOGS / "mcp-run_pipeline_schema_failure.json").read_text())
        assert data["disposition"] == "rejected"
        assert data["stages"][0]["stage"] == "verification"


# ══════════════════════════════════════════════════════════════════════════════
# check_authorization — all three wallet states
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckAuthorization:

    def test_authorized_wallet(self, gate, contributor_wallet):
        r = _handle_check_authorization(gate,
                {"wallet_address": contributor_wallet.classic_address})
        assert r["status"] == "authorized"
        assert r["is_authorized"] is True

    def test_unauthorized_wallet(self, gate, unauthorized_wallet):
        r = _handle_check_authorization(gate,
                {"wallet_address": unauthorized_wallet.classic_address})
        assert r["status"] == "unauthorized"
        assert r["is_authorized"] is False

    def test_suspended_wallet(self, gate, suspended_wallet):
        r = _handle_check_authorization(gate,
                {"wallet_address": suspended_wallet.classic_address})
        assert r["status"] == "suspended"
        assert r["is_authorized"] is False

    def test_reason_populated(self, gate, contributor_wallet):
        r = _handle_check_authorization(gate,
                {"wallet_address": contributor_wallet.classic_address})
        assert r["reason"] and len(r["reason"]) > 0

    def test_timestamp_in_result(self, gate, contributor_wallet):
        r = _handle_check_authorization(gate,
                {"wallet_address": contributor_wallet.classic_address})
        assert "timestamp" in r

    def test_malformed_wallet_returns_unauthorized(self, gate):
        r = _handle_check_authorization(gate, {"wallet_address": "not-a-wallet"})
        assert r["status"] == "unauthorized"
        assert "malformed" in r["reason"].lower() or "wallet" in r["reason"].lower()

    def test_result_is_json_serialisable(self, gate, contributor_wallet):
        r = _handle_check_authorization(gate,
                {"wallet_address": contributor_wallet.classic_address})
        json.dumps(r, default=str)

    def test_writes_log(self, gate, contributor_wallet, unauthorized_wallet):
        auth   = _handle_check_authorization(gate, {"wallet_address": contributor_wallet.classic_address})
        unauth = _handle_check_authorization(gate, {"wallet_address": unauthorized_wallet.classic_address})
        write_log("check_authorization", {"authorized": auth, "unauthorized": unauth})
        data = json.loads((LOGS / "mcp-check_authorization.json").read_text())
        assert data["authorized"]["status"] == "authorized"
        assert data["unauthorized"]["status"] == "unauthorized"


# ══════════════════════════════════════════════════════════════════════════════
# dry_run_payout — authorized wallet returns blob
# ══════════════════════════════════════════════════════════════════════════════

class TestDryRunPayout:

    def test_returns_tx_blob(self, payout_config, gate, contributor_wallet):
        t = make_trigger(payout_config)
        r = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": "a" * 64,
        })
        assert "tx_blob" in r
        assert len(r["tx_blob"]) > 100

    def test_blob_is_valid_hex(self, payout_config, gate, contributor_wallet):
        t = make_trigger(payout_config)
        r = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": "b" * 64,
        })
        bytes.fromhex(r["tx_blob"])

    def test_dry_run_flag_set(self, payout_config, gate, contributor_wallet):
        t = make_trigger(payout_config)
        r = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": "c" * 64,
        })
        assert r["dry_run"] is True

    def test_contributor_in_result(self, payout_config, gate, contributor_wallet):
        t = make_trigger(payout_config)
        r = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": "d" * 64,
        })
        assert r["contributor"] == contributor_wallet.classic_address

    def test_unauthorized_wallet_returns_error(self, payout_config, gate,
                                                unauthorized_wallet):
        t = make_trigger(payout_config)
        r = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": unauthorized_wallet.classic_address,
            "evidence_hash": "e" * 64,
        })
        assert "error" in r
        assert r["error"] == "unauthorized_contributor"
        assert "tx_blob" not in r

    def test_duplicate_evidence_hash_returns_error(self, payout_config, gate,
                                                    contributor_wallet):
        store = IdempotencyStore()
        t = make_trigger(payout_config, store=store)
        h = "f" * 64
        _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": h,
        })
        r2 = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": h,
        })
        assert r2["error"] == "duplicate_verdict"

    def test_writes_log(self, payout_config, gate, contributor_wallet):
        t = make_trigger(payout_config)
        r = _handle_dry_run_payout(t, gate, {
            "contributor_wallet": contributor_wallet.classic_address,
            "evidence_hash": "00" * 32,
            "task_id": "mcp-test-task",
        })
        write_log("dry_run_payout", r)
        data = json.loads((LOGS / "mcp-dry_run_payout.json").read_text())
        assert "tx_blob" in data
        assert data["dry_run"] is True


# ══════════════════════════════════════════════════════════════════════════════
# MCP Server — tool registration
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPServerTools:

    def test_build_server_returns_server(self, tmp_path):
        # build_server with no allowlist (empty gate)
        server = build_server(allowlist_path=str(tmp_path / "nonexistent.json"))
        assert server is not None
        assert server.name == "task-node-conductor"

    def test_list_tools_returns_three_tools(self, tmp_path):
        import mcp.types as mt
        server = build_server(allowlist_path=str(tmp_path / "nonexistent.json"))

        async def run():
            res = await server.request_handlers[mt.ListToolsRequest](
                mt.ListToolsRequest()
            )
            # ServerResult wraps the actual result in .root
            return res.root.tools

        tools = asyncio.run(run())
        names = {t.name for t in tools}
        assert TOOL_RUN_PIPELINE        in names
        assert TOOL_CHECK_AUTHORIZATION in names
        assert TOOL_DRY_RUN_PAYOUT      in names

    def test_tool_schemas_have_required_fields(self, tmp_path):
        import mcp.types as mt
        server = build_server(allowlist_path=str(tmp_path / "nonexistent.json"))

        async def run():
            res = await server.request_handlers[mt.ListToolsRequest](
                mt.ListToolsRequest()
            )
            return {t.name: t for t in res.root.tools}

        tools = asyncio.run(run())
        assert "contributor_wallet" in str(tools[TOOL_RUN_PIPELINE].inputSchema)
        assert "wallet_address"     in str(tools[TOOL_CHECK_AUTHORIZATION].inputSchema)
        assert "evidence_hash"      in str(tools[TOOL_DRY_RUN_PAYOUT].inputSchema)

    def test_call_tool_unknown_name_returns_error(self, tmp_path):
        import mcp.types as mt
        server = build_server(allowlist_path=str(tmp_path / "nonexistent.json"))

        async def run():
            res = await server.request_handlers[mt.CallToolRequest](
                mt.CallToolRequest(
                    params=mt.CallToolRequestParams(
                        name="nonexistent_tool",
                        arguments={},
                    )
                )
            )
            # ServerResult.root is CallToolResult; content is on root
            return json.loads(res.root.content[0].text)

        r = asyncio.run(run())
        assert "error" in r
