"""
conductor_middleware.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TaskNodeConductor — sequences all four middleware modules into a single
importable pipeline for Task Node operators.

Pipeline stages
---------------
1. Verification   — VerificationOracle validates the artifact against
                    the caller-supplied Pydantic schema
2. Adjudication   — AdjudicationOracle computes evidence hash, applies
                    optional reviewer policy, and emits a structured verdict
3. Authorization  — AuthorizationGate checks the contributor wallet against
                    the allowlist before any payout is constructed
4. Payout         — PayoutTrigger builds (and optionally submits) the XRPL
                    PFT Payment transaction; idempotency guard prevents
                    double-spend on duplicate evidence hashes

Each stage short-circuits on failure, returning a PipelineResult with the
completed audit trail and a Disposition explaining why the pipeline stopped.

Public API
----------
Disposition      — completed / rejected / retry / blocked / duplicate / error
StageVerdict     — one stage's outcome captured in the audit trail
PipelineResult   — full structured result returned by run_pipeline()
TaskNodeConductor — the conductor; call run_pipeline() to execute

Usage::

    from conductor_middleware import TaskNodeConductor
    from conductor_middleware import AuthorizationGate, PayoutConfig
    from pydantic import BaseModel

    class MyArtifact(BaseModel):
        summary: str
        confidence: float

    conductor = TaskNodeConductor(
        gate=AuthorizationGate("allowlist.json"),
        payout_config=PayoutConfig(issuer_seed="s...", pft_issuer="r..."),
    )
    result = conductor.run_pipeline(
        artifact={"summary": "task done", "confidence": 0.95},
        schema=MyArtifact,
        contributor_wallet="0xAABB...CCDD",
        dry_run=True,
    )
    print(result.disposition)   # Disposition.COMPLETED
    print(result.tx_blob)       # signed XRPL transaction hex
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .adjudication import AdjudicationLog, AdjudicationOracle, AdjudicationVerdict, ReviewerPolicy, VerdictCode
from .authorization import AuthorizationGate, AuthorizationVerdict, WalletStatus
from .oracle import VerificationOracle
from .payout import (
    DuplicateVerdictError,
    IdempotencyStore,
    PayoutBlockedError,
    PayoutConfig,
    PayoutRequest,
    PayoutResult,
    PayoutTrigger,
)


# ── Disposition enum ───────────────────────────────────────────────────────────

class Disposition(str, Enum):
    COMPLETED  = "completed"   # all stages passed; payout constructed/submitted
    REJECTED   = "rejected"    # schema or adjudication hard failure; no retry
    RETRY      = "retry"       # adjudication or verification soft failure; resubmit
    BLOCKED    = "blocked"     # authorization gate denied the contributor
    DUPLICATE  = "duplicate"   # evidence hash already processed; idempotent no-op
    ERROR      = "error"       # unexpected exception; pipeline aborted


# ── Stage verdict ──────────────────────────────────────────────────────────────

class StageVerdict(BaseModel):
    """One stage's outcome captured in the pipeline audit trail."""
    stage:          str
    passed:         bool
    reason:         str
    evidence_hash:  str | None = None
    timestamp:      datetime   = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail:         dict[str, Any] = Field(default_factory=dict)


# ── PipelineResult ─────────────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """
    Full structured result of a TaskNodeConductor.run_pipeline() call.

    Fields
    ------
    disposition:    Final outcome enum — what happened to the pipeline
    contributor:    Normalised contributor wallet address
    artifact_hash:  SHA-256 of the submitted artifact (set after adjudication)
    stages:         Ordered list of StageVerdict, one per executed stage
    tx_blob:        Signed XRPL transaction hex (set on COMPLETED or dry-run)
    tx_hash:        On-chain transaction hash (set only after live submission)
    dry_run:        Whether payout was constructed but not submitted
    started_at:     UTC timestamp when run_pipeline() was called
    completed_at:   UTC timestamp when run_pipeline() returned
    error:          Exception message if disposition == ERROR
    """

    disposition:    Disposition
    contributor:    str
    artifact_hash:  str | None         = None
    stages:         list[StageVerdict] = Field(default_factory=list)
    tx_blob:        str | None         = None
    tx_hash:        str | None         = None
    dry_run:        bool               = False
    started_at:     datetime           = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at:   datetime | None    = None
    error:          str | None         = None

    # ── Convenience accessors ──────────────────────────────────────────────────

    def stage(self, name: str) -> StageVerdict | None:
        """Return the StageVerdict for the named stage, or None if not reached."""
        for s in self.stages:
            if s.stage == name:
                return s
        return None

    @property
    def passed_all_stages(self) -> bool:
        return self.disposition == Disposition.COMPLETED


# ── TaskNodeConductor ──────────────────────────────────────────────────────────

class TaskNodeConductor:
    """
    Sequences VerificationOracle → AdjudicationOracle → AuthorizationGate
    → PayoutTrigger into a single run_pipeline() call.

    Parameters
    ----------
    gate:
        Configured :class:`AuthorizationGate`. Required — no payout can be
        constructed for unverified contributors.
    payout_config:
        XRPL credentials for :class:`PayoutTrigger`. If omitted, the trigger
        reads from environment variables.
    reviewer_policy:
        Optional :class:`ReviewerPolicy` passed to AdjudicationOracle.
    extra_validators:
        List of callables passed to :class:`VerificationOracle` for domain
        rules beyond schema validation.
    idempotency_store:
        :class:`IdempotencyStore` for duplicate-hash detection. Defaults to
        a new in-memory store; pass a SQLite-backed instance for persistence.
    adjudication_log:
        Optional :class:`AdjudicationLog` for adjudication audit persistence.
    """

    def __init__(
        self,
        gate:               AuthorizationGate,
        payout_config:      PayoutConfig | None        = None,
        reviewer_policy:    ReviewerPolicy | None      = None,
        extra_validators:   list | None                = None,
        idempotency_store:  IdempotencyStore | None    = None,
        adjudication_log:   AdjudicationLog | None     = None,
    ) -> None:
        self._gate       = gate
        self._oracle     = AdjudicationOracle(log=adjudication_log)
        self._policy     = reviewer_policy
        self._extra      = extra_validators or []
        self._store      = idempotency_store if idempotency_store is not None else IdempotencyStore()
        self._payout     = PayoutTrigger(
            config=payout_config,
            store=self._store,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def run_pipeline(
        self,
        artifact:           dict[str, Any],
        schema:             type[BaseModel],
        contributor_wallet: str,
        dry_run:            bool = False,
        task_id:            str | None = None,
    ) -> PipelineResult:
        """
        Run the full verification → adjudication → authorization → payout pipeline.

        Fails fast: each stage short-circuits on failure, populating the
        PipelineResult audit trail with every stage that was reached.

        Parameters
        ----------
        artifact:
            Raw task-completion artifact dict from the worker agent.
        schema:
            Pydantic model class the artifact is validated against.
        contributor_wallet:
            Contributor's wallet address (XRP classic address r...).
        dry_run:
            If True, payout is constructed and serialised but not submitted.
        task_id:
            Optional identifier threaded through the audit log.

        Returns
        -------
        :class:`PipelineResult`
        """
        started_at = datetime.now(timezone.utc)
        stages: list[StageVerdict] = []

        def finish(disposition: Disposition, **kwargs) -> PipelineResult:
            return PipelineResult(
                disposition=disposition,
                contributor=contributor_wallet,
                stages=stages,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                dry_run=dry_run,
                **kwargs,
            )

        # ── Stage 1: Verification ──────────────────────────────────────────────
        v_oracle = VerificationOracle(schema, extra_validators=self._extra)
        v_verdict = v_oracle(artifact)

        stages.append(StageVerdict(
            stage="verification",
            passed=v_verdict.passed,
            reason="; ".join(v_verdict.reasons) if not v_verdict.passed
                   else "Schema validation passed.",
            detail={"field_errors": v_verdict.reasons},
        ))

        if not v_verdict.passed:
            # Hard schema failure — REJECTED, not retryable at this stage
            return finish(Disposition.REJECTED)

        # ── Stage 2: Adjudication ──────────────────────────────────────────────
        adj_verdict: AdjudicationVerdict = self._oracle.adjudicate(
            artifact=artifact,
            schema=schema,
            reviewer_policy=self._policy,
            task_id=task_id,
        )

        stages.append(StageVerdict(
            stage="adjudication",
            passed=(adj_verdict.verdict == VerdictCode.PASS),
            reason=adj_verdict.reason,
            evidence_hash=adj_verdict.evidence_hash,
            detail={
                "verdict":      adj_verdict.verdict.value,
                "field_errors": adj_verdict.field_errors,
                "schema_name":  adj_verdict.schema_name,
            },
        ))

        if adj_verdict.verdict == VerdictCode.RETRY:
            return finish(Disposition.RETRY, artifact_hash=adj_verdict.evidence_hash)

        if adj_verdict.verdict != VerdictCode.PASS:
            return finish(Disposition.REJECTED, artifact_hash=adj_verdict.evidence_hash)

        # ── Stage 3: Authorization ─────────────────────────────────────────────
        auth_verdict: AuthorizationVerdict = self._gate.gate_check(contributor_wallet)

        stages.append(StageVerdict(
            stage="authorization",
            passed=auth_verdict.is_authorized,
            reason=auth_verdict.reason,
            detail={
                "wallet_status": auth_verdict.status.value,
                "wallet":        auth_verdict.wallet_address,
            },
        ))

        if not auth_verdict.is_authorized:
            return finish(Disposition.BLOCKED, artifact_hash=adj_verdict.evidence_hash)

        # ── Stage 4: Payout ────────────────────────────────────────────────────
        payout_request = PayoutRequest(
            adjudication_verdict=adj_verdict,
            authorization_verdict=auth_verdict,
            contributor_wallet=contributor_wallet,
            task_id=task_id,
        )

        try:
            payout_result: PayoutResult = self._payout.execute(
                payout_request, dry_run=dry_run
            )
        except DuplicateVerdictError as exc:
            stages.append(StageVerdict(
                stage="payout",
                passed=False,
                reason=str(exc),
                evidence_hash=adj_verdict.evidence_hash,
                detail={"error": "duplicate_verdict"},
            ))
            return finish(Disposition.DUPLICATE, artifact_hash=adj_verdict.evidence_hash)
        except PayoutBlockedError as exc:
            # Shouldn't happen given we checked auth above, but handle defensively
            stages.append(StageVerdict(
                stage="payout",
                passed=False,
                reason=exc.reason,
                evidence_hash=adj_verdict.evidence_hash,
                detail=exc.details,
            ))
            return finish(Disposition.BLOCKED, artifact_hash=adj_verdict.evidence_hash)
        except Exception as exc:
            stages.append(StageVerdict(
                stage="payout",
                passed=False,
                reason=f"{type(exc).__name__}: {exc}",
                evidence_hash=adj_verdict.evidence_hash,
                detail={"error": "unexpected_exception"},
            ))
            return finish(Disposition.ERROR,
                          artifact_hash=adj_verdict.evidence_hash,
                          error=f"{type(exc).__name__}: {exc}")

        stages.append(StageVerdict(
            stage="payout",
            passed=True,
            reason="Payout transaction constructed successfully.",
            evidence_hash=adj_verdict.evidence_hash,
            detail={"dry_run": dry_run},
        ))

        return finish(
            Disposition.COMPLETED,
            artifact_hash=adj_verdict.evidence_hash,
            tx_blob=payout_result.tx_blob,
            tx_hash=payout_result.tx_hash,
        )
