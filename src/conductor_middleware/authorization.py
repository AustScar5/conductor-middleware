"""
conductor_middleware.authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Contributor authorization enforcement for Task Node conductors.

Enforces an allowlist check before any reward dispatch or task-completion
acceptance.  Integrates with AdjudicationOracle so the oracle automatically
returns a FAIL verdict for any unauthorized or suspended contributor.

Public API
----------
WalletStatus          — authorized / unauthorized / suspended
AuthorizationVerdict  — structured gate decision with reason and timestamp
WalletEntry           — one row in the allowlist config
AllowlistConfig       — the full parsed config (JSON or TOML)
AuthorizationGate     — loads config, exposes gate_check(wallet) → verdict
gated_adjudicate()    — drop-in replacement for oracle.adjudicate() that runs
                        the gate before schema validation

Usage::

    from conductor_middleware.authorization import AuthorizationGate
    from conductor_middleware.adjudication  import AdjudicationOracle

    gate   = AuthorizationGate("allowlist.json")
    oracle = AdjudicationOracle()

    verdict = gate.gate_check("0xAABB...CCDD")
    if verdict.status == WalletStatus.AUTHORIZED:
        adj = oracle.adjudicate(artifact, MySchema)
    else:
        # blocked — verdict.reason explains why

    # Or use the integrated helper (gate + oracle in one call):
    from conductor_middleware.authorization import gated_adjudicate
    adj = gated_adjudicate(gate, oracle, wallet, artifact, MySchema)
"""

from __future__ import annotations

import json
import re
import tomllib
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .adjudication import AdjudicationOracle, AdjudicationVerdict, ReviewerPolicy, VerdictCode


# ── Wallet address validation ──────────────────────────────────────────────────

# Accepts 0x-prefixed 40-hex-char Ethereum-style addresses.
# Deliberately broad — does not enforce EIP-55 checksum so both lower and
# upper-case addresses are accepted.
_WALLET_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _validate_wallet(address: str) -> str:
    """
    Normalise and validate a wallet address string.

    Raises ``ValueError`` with a human-readable message on any of:
      - non-string input
      - wrong length / missing 0x prefix
      - non-hex characters after 0x
    """
    if not isinstance(address, str):
        raise ValueError(
            f"wallet_address must be a string, got {type(address).__name__!r}"
        )
    stripped = address.strip()
    if not stripped:
        raise ValueError("wallet_address must not be empty or whitespace")
    # Normalise prefix to lowercase before any checks so '0X...' is accepted
    if len(stripped) >= 2:
        stripped = "0x" + stripped[2:] if stripped[:2].lower() == "0x" else stripped
    if not stripped.startswith("0x"):
        raise ValueError(
            f"wallet_address must start with '0x', got {stripped[:6]!r}…"
        )
    if not _WALLET_RE.match(stripped):
        raise ValueError(
            f"wallet_address must be a 0x-prefixed 40-hex-character string "
            f"(got {len(stripped) - 2} hex chars in {stripped!r})"
        )
    return stripped.lower()   # normalise to lowercase for comparison


# ── Enums and models ───────────────────────────────────────────────────────────

class WalletStatus(str, Enum):
    AUTHORIZED   = "authorized"
    UNAUTHORIZED = "unauthorized"
    SUSPENDED    = "suspended"
    EXPIRED      = "expired"      # was authorized but expires_at is in the past


class AuthorizationVerdict(BaseModel):
    """
    Structured result of a gate_check call.

    Fields
    ------
    wallet_address:  Normalised (lowercase) address that was checked
    status:          WalletStatus enum value
    reason:          Human-readable explanation
    timestamp:       UTC datetime of the check
    """

    wallet_address: str
    status:         WalletStatus
    reason:         str
    timestamp:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_authorized(self) -> bool:
        return self.status == WalletStatus.AUTHORIZED


class WalletEntry(BaseModel):
    """One row in the allowlist config."""

    wallet:            str
    status:            WalletStatus
    label:             str | None = None
    expires_at:        datetime   | None = None
    suspension_reason: str        | None = None

    @field_validator("wallet", mode="before")
    @classmethod
    def _normalise_wallet(cls, v: Any) -> str:
        return _validate_wallet(str(v))


class AllowlistConfig(BaseModel):
    """The full parsed allowlist configuration."""

    allowlist: list[WalletEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_no_duplicate_wallets(self) -> "AllowlistConfig":
        seen: set[str] = set()
        for entry in self.allowlist:
            if entry.wallet in seen:
                raise ValueError(
                    f"Duplicate wallet entry in config: {entry.wallet!r}"
                )
            seen.add(entry.wallet)
        return self


# ── AuthorizationGate ──────────────────────────────────────────────────────────

class AuthorizationGate:
    """
    Loads an allowlist from a JSON or TOML config file and enforces
    contributor authorization before task dispatch or reward acceptance.

    Parameters
    ----------
    config_path:
        Path to a ``.json`` or ``.toml`` allowlist file.
        Pass ``None`` to start with an empty allowlist (useful in tests).
    config:
        Supply an ``AllowlistConfig`` directly instead of a file path.
        ``config_path`` takes precedence if both are provided.

    Examples
    --------
    >>> gate = AuthorizationGate("allowlist.json")
    >>> verdict = gate.gate_check("0xAABB...CCDD")
    >>> verdict.is_authorized
    True
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config:      AllowlistConfig   | None = None,
    ) -> None:
        if config_path is not None:
            self._config = self._load(Path(config_path))
        elif config is not None:
            self._config = config
        else:
            self._config = AllowlistConfig(allowlist=[])

        # Build O(1) lookup by normalised wallet address
        self._index: dict[str, WalletEntry] = {
            entry.wallet: entry for entry in self._config.allowlist
        }

    # ── Public interface ───────────────────────────────────────────────────────

    def gate_check(self, wallet_address: str) -> AuthorizationVerdict:
        """
        Check whether ``wallet_address`` is authorised to participate.

        Returns an :class:`AuthorizationVerdict` for all cases including
        malformed addresses — never raises.

        Precedence
        ----------
        1. Validate the address format.
        2. Check allowlist lookup:
           - Not found → UNAUTHORIZED
           - Found, status=suspended → SUSPENDED
           - Found, status=authorized but expires_at is past → EXPIRED
           - Found, status=authorized, not expired → AUTHORIZED
        """
        # Step 1 — format validation
        try:
            normalised = _validate_wallet(wallet_address)
        except ValueError as exc:
            return AuthorizationVerdict(
                wallet_address=wallet_address,
                status=WalletStatus.UNAUTHORIZED,
                reason=f"Malformed wallet address: {exc}",
            )

        # Step 2 — allowlist lookup
        entry = self._index.get(normalised)

        if entry is None:
            return AuthorizationVerdict(
                wallet_address=normalised,
                status=WalletStatus.UNAUTHORIZED,
                reason="Wallet address is not on the contributor allowlist.",
            )

        if entry.status == WalletStatus.SUSPENDED:
            base = "Contributor is suspended."
            reason = (
                f"{base} Reason: {entry.suspension_reason}"
                if entry.suspension_reason
                else base
            )
            return AuthorizationVerdict(
                wallet_address=normalised,
                status=WalletStatus.SUSPENDED,
                reason=reason,
            )

        # Check expiry
        if entry.expires_at is not None:
            now = datetime.now(timezone.utc)
            exp = (
                entry.expires_at
                if entry.expires_at.tzinfo
                else entry.expires_at.replace(tzinfo=timezone.utc)
            )
            if now > exp:
                return AuthorizationVerdict(
                    wallet_address=normalised,
                    status=WalletStatus.EXPIRED,
                    reason=f"Contributor authorization expired at {exp.isoformat()}.",
                )

        return AuthorizationVerdict(
            wallet_address=normalised,
            status=WalletStatus.AUTHORIZED,
            reason="Contributor is on the allowlist and authorization is current.",
        )

    def add(self, entry: WalletEntry) -> None:
        """Add or replace a wallet entry in the in-memory allowlist."""
        self._config.allowlist.append(entry)
        self._index[entry.wallet] = entry

    def reload(self, config_path: str | Path) -> None:
        """Reload the allowlist from disk without recreating the gate."""
        self._config = self._load(Path(config_path))
        self._index  = {e.wallet: e for e in self._config.allowlist}

    # ── Loader ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> AllowlistConfig:
        if not path.exists():
            raise FileNotFoundError(f"Allowlist config not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        else:
            raise ValueError(
                f"Unsupported config format {suffix!r}. Use .json or .toml"
            )
        return AllowlistConfig.model_validate(raw)


# ── Oracle integration ─────────────────────────────────────────────────────────

def gated_adjudicate(
    gate:            AuthorizationGate,
    oracle:          AdjudicationOracle,
    wallet_address:  str,
    artifact:        dict[str, Any],
    schema:          type,
    reviewer_policy: ReviewerPolicy | None = None,
    task_id:         str | None = None,
) -> AdjudicationVerdict:
    """
    Drop-in replacement for ``oracle.adjudicate()`` that enforces the
    ``AuthorizationGate`` before any schema validation occurs.

    If the gate returns anything other than AUTHORIZED, this function
    immediately returns a FAIL ``AdjudicationVerdict`` with
    ``reason='unauthorized_contributor'`` and the gate's reason appended.
    The oracle is never called for unauthorized or suspended wallets.

    Parameters
    ----------
    gate:            Configured :class:`AuthorizationGate` instance
    oracle:          :class:`AdjudicationOracle` instance
    wallet_address:  Contributor wallet to check before adjudication
    artifact:        Task completion artifact dict
    schema:          Pydantic model class to validate the artifact against
    reviewer_policy: Optional domain-level reviewer rules
    task_id:         Optional task identifier for audit logging

    Returns
    -------
    :class:`AdjudicationVerdict` — FAIL if gate blocks, otherwise the
    oracle's normal verdict
    """
    auth = gate.gate_check(wallet_address)

    if not auth.is_authorized:
        # Compute hash of artifact even for blocked submissions so the
        # audit log can prove what was received
        evidence_hash = AdjudicationOracle._hash(artifact)
        return AdjudicationVerdict(
            verdict=VerdictCode.FAIL,
            reason=(
                f"unauthorized_contributor: {auth.reason} "
                f"(wallet={auth.wallet_address}, status={auth.status.value})"
            ),
            evidence_hash=evidence_hash,
            retry_eligible=False,
            schema_name=schema.__name__ if hasattr(schema, "__name__") else str(schema),
        )

    return oracle.adjudicate(
        artifact=artifact,
        schema=schema,
        reviewer_policy=reviewer_policy,
        task_id=task_id,
    )
