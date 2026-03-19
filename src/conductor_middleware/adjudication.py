"""
conductor_middleware.adjudication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AdjudicationOracle — programmatic Authorization Gate enforcement for Task Node conductors.

Validates task completion artifacts against caller-defined Pydantic schemas,
computes SHA-256 evidence hashes, applies optional reviewer policy rules,
and produces an auditable adjudication log.

Public API
----------
VerdictCode          — PASS / FAIL / RETRY enum
AdjudicationVerdict  — structured verdict with hash, timestamp, retry_eligible flag
ReviewerPolicy       — injectable domain rules applied after schema validation
AdjudicationOracle   — the callable gate; produces AdjudicationVerdict
AdjudicationRecord   — one entry in the audit log
AdjudicationLog      — append-only audit log with integrity-verification support

Usage::

    from conductor_middleware.adjudication import AdjudicationOracle, ReviewerPolicy
    from pydantic import BaseModel

    class DeliveryArtifact(BaseModel):
        summary: str
        confidence: float
        sources: list[str]

    def require_sources(obj: DeliveryArtifact) -> str | None:
        if not obj.sources:
            return "sources must not be empty"

    oracle = AdjudicationOracle()
    verdict = oracle.adjudicate(
        artifact={"summary": "done", "confidence": 0.9, "sources": ["doc1"]},
        schema=DeliveryArtifact,
        reviewer_policy=ReviewerPolicy("require_sources", [require_sources]),
    )

    assert verdict.verdict == VerdictCode.PASS
    assert verdict.evidence_hash  # SHA-256 of the serialised artifact
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError


# ── Verdict enum ───────────────────────────────────────────────────────────────

class VerdictCode(str, Enum):
    PASS  = "pass"   # artifact satisfies schema and all reviewer rules
    FAIL  = "fail"   # permanent failure — schema violation or hard reviewer reject
    RETRY = "retry"  # reviewer rejection flagged as transient; conductor should retry


# ── AdjudicationVerdict ────────────────────────────────────────────────────────

class AdjudicationVerdict(BaseModel):
    """
    Structured verdict returned by the AdjudicationOracle.

    Fields
    ------
    verdict:         PASS / FAIL / RETRY
    reason:          Human-readable explanation of the decision
    evidence_hash:   SHA-256 hex digest of the *canonically serialised* artifact
                     at adjudication time.  Re-compute and compare to detect tampering.
    timestamp:       UTC datetime of adjudication
    retry_eligible:  True iff the conductor may re-submit the artifact
    field_errors:    Pydantic field-level errors (populated on schema FAIL only)
    schema_name:     Name of the Pydantic model class used for validation
    """

    verdict:        VerdictCode
    reason:         str
    evidence_hash:  str   = Field(..., min_length=64, max_length=64,
                                  description="SHA-256 hex digest of the canonicalised artifact")
    timestamp:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retry_eligible: bool
    field_errors:   list[str] = Field(default_factory=list)
    schema_name:    str       = ""


# ── ReviewerPolicy ─────────────────────────────────────────────────────────────

class ReviewerPolicy:
    """
    Injectable domain rules applied *after* schema validation passes.

    Parameters
    ----------
    name:
        Human-readable label for the policy (appears in verdict reason).
    validators:
        List of callables ``(validated_instance: BaseModel) -> str | None``.
        Return a non-empty string to reject with that message; return ``None`` to pass.
    retry_eligible:
        Whether a rejection by this policy should produce ``VerdictCode.RETRY``
        (True, default) or ``VerdictCode.FAIL`` (False — hard stop).
    """

    def __init__(
        self,
        name:           str,
        validators:     list[Callable[[BaseModel], str | None]],
        retry_eligible: bool = True,
    ) -> None:
        self.name           = name
        self.validators     = validators
        self.retry_eligible = retry_eligible

    def apply(self, validated_obj: BaseModel) -> list[str]:
        """Run all validators; return list of rejection reasons (empty = passed)."""
        reasons: list[str] = []
        for fn in self.validators:
            try:
                result = fn(validated_obj)
                if result:
                    reasons.append(str(result))
            except Exception as exc:
                reasons.append(f"validator {fn.__name__!r} raised {type(exc).__name__}: {exc}")
        return reasons

    def __repr__(self) -> str:
        return f"ReviewerPolicy({self.name!r}, validators={len(self.validators)}, retry={self.retry_eligible})"


# ── AdjudicationOracle ─────────────────────────────────────────────────────────

class AdjudicationOracle:
    """
    Programmatic Authorization Gate for Task Node conductors.

    Call ``adjudicate()`` to validate an artifact dict and receive a structured
    ``AdjudicationVerdict``.  Optionally attach an ``AdjudicationLog`` instance
    to record every verdict for later audit or integrity verification.

    Parameters
    ----------
    log:
        Optional ``AdjudicationLog`` to record every verdict automatically.
    """

    def __init__(self, log: AdjudicationLog | None = None) -> None:
        self._log = log

    # ── Primary interface ──────────────────────────────────────────────────────

    def adjudicate(
        self,
        artifact:        dict[str, Any],
        schema:          type[BaseModel],
        reviewer_policy: ReviewerPolicy | None = None,
        task_id:         str | None            = None,
    ) -> AdjudicationVerdict:
        """
        Validate ``artifact`` and return a structured ``AdjudicationVerdict``.

        Steps
        -----
        1. Compute SHA-256 evidence hash of the canonically serialised artifact.
        2. Validate artifact against ``schema`` (Pydantic model_validate).
        3. Apply ``reviewer_policy`` validators if provided.
        4. Return verdict; record in ``self._log`` if attached.

        Never raises — all errors are captured in the verdict.
        """
        evidence_hash = self._hash(artifact)

        # ── Step 2: schema validation ──────────────────────────────────────────
        try:
            validated = schema.model_validate(artifact)
        except ValidationError as exc:
            field_errors = [
                f"{'.'.join(str(loc) for loc in err['loc']) or 'root'}: {err['msg']}"
                for err in exc.errors()
            ]
            verdict = AdjudicationVerdict(
                verdict=VerdictCode.FAIL,
                reason=f"Schema validation failed: {len(field_errors)} field error(s).",
                evidence_hash=evidence_hash,
                retry_eligible=False,
                field_errors=field_errors,
                schema_name=schema.__name__,
            )
            self._record(verdict, task_id)
            return verdict

        # ── Step 3: reviewer policy ────────────────────────────────────────────
        if reviewer_policy:
            reasons = reviewer_policy.apply(validated)
            if reasons:
                code = VerdictCode.RETRY if reviewer_policy.retry_eligible else VerdictCode.FAIL
                verdict = AdjudicationVerdict(
                    verdict=code,
                    reason=f"Reviewer policy {reviewer_policy.name!r} rejected: {'; '.join(reasons)}",
                    evidence_hash=evidence_hash,
                    retry_eligible=reviewer_policy.retry_eligible,
                    schema_name=schema.__name__,
                )
                self._record(verdict, task_id)
                return verdict

        # ── Step 4: pass ───────────────────────────────────────────────────────
        verdict = AdjudicationVerdict(
            verdict=VerdictCode.PASS,
            reason="Artifact passed schema validation and all reviewer checks.",
            evidence_hash=evidence_hash,
            retry_eligible=False,
            schema_name=schema.__name__,
        )
        self._record(verdict, task_id)
        return verdict

    # ── Integrity verification ─────────────────────────────────────────────────

    def verify_integrity(self, artifact: dict[str, Any], expected_hash: str) -> bool:
        """
        Re-compute the evidence hash and compare to ``expected_hash``.

        Returns ``True`` if the artifact is unmodified since adjudication,
        ``False`` if it has been tampered with.
        """
        return self._hash(artifact) == expected_hash

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(artifact: dict[str, Any]) -> str:
        """
        SHA-256 of the artifact serialised with sorted keys and no whitespace.
        Deterministic across Python versions and dict insertion orders.
        """
        canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"),
                               default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record(self, verdict: AdjudicationVerdict, task_id: str | None) -> None:
        if self._log is not None:
            self._log.append(AdjudicationRecord(
                task_id=task_id,
                verdict=verdict,
            ))


# ── AdjudicationRecord + AdjudicationLog ──────────────────────────────────────

class AdjudicationRecord(BaseModel):
    """Single entry in the adjudication audit log."""

    record_id:       str      = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id:         str | None = None
    verdict:         AdjudicationVerdict
    adjudicated_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdjudicationLog:
    """
    Append-only in-memory audit log with optional JSON persistence.

    Parameters
    ----------
    path:
        If provided, the log is written to this file after every append.
    """

    def __init__(self, path: "Path | None" = None) -> None:  # noqa: F821
        self._records: list[AdjudicationRecord] = []
        self._path = path

    def append(self, record: AdjudicationRecord) -> None:
        self._records.append(record)
        if self._path is not None:
            self._flush()

    def verify_record_integrity(
        self,
        record_id: str,
        artifact:  dict[str, Any],
    ) -> bool:
        """
        Check whether ``artifact`` still matches the evidence hash stored
        in the log record identified by ``record_id``.

        Raises ``KeyError`` if the record is not found.
        """
        rec = self._find(record_id)
        stored_hash = rec.verdict.evidence_hash
        recomputed  = AdjudicationOracle._hash(artifact)
        return recomputed == stored_hash

    def export(self) -> dict[str, Any]:
        """Return the full log as a plain dict (JSON-serialisable)."""
        return {
            "record_count": len(self._records),
            "records": [r.model_dump(mode="json") for r in self._records],
        }

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def _find(self, record_id: str) -> AdjudicationRecord:
        for rec in self._records:
            if rec.record_id == record_id:
                return rec
        raise KeyError(f"No record with record_id={record_id!r}")

    def _flush(self) -> None:
        import json as _json
        from pathlib import Path
        Path(self._path).write_text(
            _json.dumps(self.export(), indent=2, default=str),
            encoding="utf-8",
        )
