"""
conductor_middleware.payout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XRPL PFT payout executor for Task Node conductors.

Accepts AdjudicationVerdict + AuthorizationVerdict from the existing middleware
stack, validates both are 'pass', enforces an idempotency guard keyed on the
verdict evidence hash, and constructs a signed XRP Ledger Payment transaction
that emits PFT tokens to the contributor's wallet.

Supports dry_run=True to return the fully serialised transaction blob without
submitting to the network — safe for testing and pipeline validation.

Public API
----------
PayoutConfig       — XRPL credentials + PFT issuer config (env-var backed)
PayoutRequest      — bundles both verdicts + contributor wallet for one payout
PayoutResult       — structured outcome: tx blob, hash, dry_run flag, timestamp
PayoutBlockedError — raised when verdicts don't authorise a payout
DuplicateVerdictError — raised when the evidence hash has already been paid out
IdempotencyStore   — pluggable duplicate-hash guard (in-memory or SQLite)
PayoutTrigger      — the main callable; wires everything together

Environment variables
---------------------
XRPL_ISSUER_SEED      — base58 family seed of the PFT-issuing account (required)
XRPL_PFT_ISSUER       — classic address of the PFT issuer (required)
XRPL_PFT_CURRENCY     — currency code, default "PFT"
XRPL_NODE_URL         — websocket URL of XRPL node, default mainnet
XRPL_BASE_FEE_DROPS   — network fee in drops, default "12"
XRPL_PFT_AMOUNT       — tokens to pay per passing verdict, default "100"
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from xrpl.models.amounts import IssuedCurrencyAmount
from xrpl.models.transactions import Payment
from xrpl.transaction import sign
from xrpl.wallet import Wallet

from .adjudication import AdjudicationVerdict, VerdictCode
from .authorization import AuthorizationVerdict, WalletStatus


# ── Exceptions ─────────────────────────────────────────────────────────────────

class PayoutBlockedError(Exception):
    """
    Raised when one or both verdicts do not authorise a payout.

    Attributes
    ----------
    reason:   Human-readable explanation
    details:  Dict with the blocking verdict(s) for structured logging
    """

    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason  = reason
        self.details = details or {}


class DuplicateVerdictError(Exception):
    """
    Raised when the evidence hash has already been processed.

    Attributes
    ----------
    evidence_hash:  The hash that triggered the guard
    """

    def __init__(self, evidence_hash: str) -> None:
        super().__init__(
            f"Verdict evidence_hash={evidence_hash!r} has already been paid out. "
            "Duplicate submission rejected."
        )
        self.evidence_hash = evidence_hash


# ── Idempotency store ──────────────────────────────────────────────────────────

class IdempotencyStore:
    """
    Thread-safe duplicate-hash guard.

    Parameters
    ----------
    db_path:
        Path to a SQLite file for persistent storage.
        Pass ``None`` (default) for in-memory-only storage — safe for testing
        and single-process conductors; lost on restart.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._memory: set[str] = set()
        self._db_path = str(db_path) if db_path else None
        self._conn: sqlite3.Connection | None = None

        if self._db_path:
            # Use the default isolation_level (deferred transactions) so that
            # explicit conn.commit() calls flush immediately and are visible
            # to any new connection. isolation_level=None (autocommit) sounds
            # right but interacts badly with CREATE TABLE implicit transactions.
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS paid_verdicts "
                "(evidence_hash TEXT PRIMARY KEY, paid_at TEXT NOT NULL)"
            )
            self._conn.commit()
            # Warm memory set from DB so in-memory checks are still O(1)
            cur = self._conn.execute("SELECT evidence_hash FROM paid_verdicts")
            self._memory = {row[0] for row in cur.fetchall()}

    def contains(self, evidence_hash: str) -> bool:
        with self._lock:
            return evidence_hash in self._memory

    def mark(self, evidence_hash: str) -> None:
        with self._lock:
            self._memory.add(evidence_hash)
            if self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO paid_verdicts VALUES (?, ?)",
                    (evidence_hash, datetime.now(timezone.utc).isoformat()),
                )
                self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return len(self._memory)


# ── Config ─────────────────────────────────────────────────────────────────────

class PayoutConfig(BaseModel):
    """
    XRPL credentials and PFT token configuration.

    All fields have env-var defaults; override explicitly for testing.

    Fields
    ------
    issuer_seed:    base58 family seed of the signing/issuing wallet
    pft_issuer:     classic address of the PFT token issuer
    pft_currency:   currency code (3-char or 40-char hex), default "PFT"
    pft_amount:     token units to pay per passing verdict, default "100"
    node_url:       XRPL websocket node URL (only used for live submission)
    base_fee_drops: network fee string in drops
    """

    issuer_seed:    str   = Field(default_factory=lambda: os.environ.get("XRPL_ISSUER_SEED", ""))
    pft_issuer:     str   = Field(default_factory=lambda: os.environ.get("XRPL_PFT_ISSUER", ""))
    pft_currency:   str   = Field(default_factory=lambda: os.environ.get("XRPL_PFT_CURRENCY", "PFT"))
    pft_amount:     str   = Field(default_factory=lambda: os.environ.get("XRPL_PFT_AMOUNT", "100"))
    node_url:       str   = Field(
        default_factory=lambda: os.environ.get(
            "XRPL_NODE_URL", "wss://xrplcluster.com"
        )
    )
    base_fee_drops: str   = Field(default_factory=lambda: os.environ.get("XRPL_BASE_FEE_DROPS", "12"))

    model_config = {"frozen": True}


# ── Payout request / result ────────────────────────────────────────────────────

class PayoutRequest(BaseModel):
    """Bundles both verdicts and the contributor wallet for a single payout."""

    adjudication_verdict:  AdjudicationVerdict
    authorization_verdict: AuthorizationVerdict
    contributor_wallet:    str     # XRP classic address (r...)
    task_id:               str | None = None

    model_config = {"frozen": True}


class PayoutResult(BaseModel):
    """Structured outcome returned by PayoutTrigger.execute()."""

    task_id:        str | None
    contributor:    str
    pft_amount:     str
    evidence_hash:  str
    tx_blob:        str      # hex-encoded signed transaction
    tx_hash:        str | None = None   # set after live submission
    dry_run:        bool
    submitted_at:   datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── PayoutTrigger ──────────────────────────────────────────────────────────────

class PayoutTrigger:
    """
    Validates the verdict chain, enforces idempotency, and constructs (or
    submits) an XRPL PFT Payment transaction.

    Parameters
    ----------
    config:
        :class:`PayoutConfig` with XRPL credentials. If omitted, values are
        read from environment variables.
    store:
        :class:`IdempotencyStore` instance. Defaults to a new in-memory store.
    sequence_override:
        Fixed ledger sequence number to use in transactions — set this in tests
        to avoid needing a live account lookup.
    last_ledger_override:
        Fixed last_ledger_sequence — set in tests to avoid network calls.
    """

    def __init__(
        self,
        config:               PayoutConfig | None = None,
        store:                IdempotencyStore | None = None,
        sequence_override:    int | None = None,
        last_ledger_override: int | None = None,
    ) -> None:
        self._config   = config or PayoutConfig()
        self._store    = store if store is not None else IdempotencyStore()
        self._seq      = sequence_override
        self._last_seq = last_ledger_override

    # ── Public interface ───────────────────────────────────────────────────────

    def execute(self, request: PayoutRequest, dry_run: bool = False) -> PayoutResult:
        """
        Validate verdicts, check idempotency, build and optionally submit tx.

        Parameters
        ----------
        request:
            :class:`PayoutRequest` with both verdicts and the contributor wallet.
        dry_run:
            If ``True``, return the serialised blob without submitting.
            Safe for pipeline testing; no network calls are made.

        Returns
        -------
        :class:`PayoutResult`

        Raises
        ------
        PayoutBlockedError
            If either verdict is not a pass.
        DuplicateVerdictError
            If the evidence hash has already been paid out.
        ValueError
            If XRPL credentials are missing or malformed.
        """
        # ── Step 1: validate verdict chain ────────────────────────────────────
        self._validate_verdicts(request)

        # ── Step 2: idempotency guard ─────────────────────────────────────────
        evidence_hash = request.adjudication_verdict.evidence_hash
        if self._store.contains(evidence_hash):
            raise DuplicateVerdictError(evidence_hash)

        # ── Step 3: build + sign transaction ──────────────────────────────────
        signed_tx = self._build_transaction(request)
        blob      = signed_tx.blob()

        # ── Step 4: submit or dry-run ──────────────────────────────────────────
        tx_hash: str | None = None
        if not dry_run:
            tx_hash = self._submit(signed_tx)

        # ── Step 5: mark idempotency (after successful build/submit) ───────────
        self._store.mark(evidence_hash)

        return PayoutResult(
            task_id=request.task_id,
            contributor=request.contributor_wallet,
            pft_amount=self._config.pft_amount,
            evidence_hash=evidence_hash,
            tx_blob=blob,
            tx_hash=tx_hash,
            dry_run=dry_run,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate_verdicts(self, request: PayoutRequest) -> None:
        """
        Enforce the full verdict chain:
          1. authorization_verdict.status must be AUTHORIZED
          2. adjudication_verdict.verdict must be PASS
        Both must hold; raises PayoutBlockedError on the first failure.
        """
        auth = request.authorization_verdict
        adj  = request.adjudication_verdict

        if auth.status != WalletStatus.AUTHORIZED:
            raise PayoutBlockedError(
                reason=(
                    f"Authorization gate failed: contributor "
                    f"{auth.wallet_address!r} has status={auth.status.value!r}. "
                    f"Reason: {auth.reason}"
                ),
                details={
                    "blocked_by":          "authorization_gate",
                    "wallet_status":       auth.status.value,
                    "authorization_reason": auth.reason,
                },
            )

        if adj.verdict != VerdictCode.PASS:
            raise PayoutBlockedError(
                reason=(
                    f"Adjudication gate failed: verdict={adj.verdict.value!r}. "
                    f"Reason: {adj.reason}"
                ),
                details={
                    "blocked_by":         "adjudication_oracle",
                    "verdict":            adj.verdict.value,
                    "adjudication_reason": adj.reason,
                    "field_errors":       adj.field_errors,
                },
            )

    def _build_transaction(self, request: PayoutRequest) -> Any:
        """Construct and sign a Payment transaction using xrpl-py."""
        cfg = self._config

        if not cfg.issuer_seed:
            raise ValueError(
                "XRPL_ISSUER_SEED is not set. "
                "Set the environment variable or pass PayoutConfig(issuer_seed=...)."
            )
        if not cfg.pft_issuer:
            raise ValueError(
                "XRPL_PFT_ISSUER is not set. "
                "Set the environment variable or pass PayoutConfig(pft_issuer=...)."
            )

        issuer_wallet = Wallet.from_seed(cfg.issuer_seed)

        # Sequence numbers: use override (tests) or defaults that will be
        # updated by the node on live submission via autofill.
        seq      = self._seq      if self._seq      is not None else 1
        last_seq = self._last_seq if self._last_seq is not None else 99_999_999

        tx = Payment(
            account=issuer_wallet.classic_address,
            amount=IssuedCurrencyAmount(
                currency=cfg.pft_currency,
                issuer=cfg.pft_issuer,
                value=cfg.pft_amount,
            ),
            destination=request.contributor_wallet,
            sequence=seq,
            fee=cfg.base_fee_drops,
            last_ledger_sequence=last_seq,
        )

        return sign(tx, issuer_wallet)

    def _submit(self, signed_tx: Any) -> str:
        """Submit the signed transaction to the XRPL network."""
        # Import here so the module loads cleanly without a network connection
        from xrpl.clients import WebsocketClient
        from xrpl.transaction import submit_and_wait

        with WebsocketClient(self._config.node_url) as client:
            response = submit_and_wait(signed_tx, client)
            return response.result.get("tx_json", {}).get("hash", "")
