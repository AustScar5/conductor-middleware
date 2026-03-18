"""
conductor_middleware.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic v2 data models for the conductor-middleware interface spec.

Public types
------------
ErrorCode          — typed enum with transient/permanent classification
RetryPolicy        — backoff config; owns delay_for() computation
ErrorEnvelope      — structured error with computed is_transient field
TaskDispatch       — what a conductor sends to a worker
WorkerResponse     — what a worker returns; output/error are mutually exclusive
ReviewerVerdict    — pass/fail result from the VerificationOracle
AttemptRecord      — one entry in the per-task attempt log
AttemptState       — mutable state tracked by the RetryEngine across attempts
TerminalResult     — final sealed outcome; success requires a passing verdict
RetryDecision      — conductor decision after evaluating a response
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


# ── Error taxonomy ─────────────────────────────────────────────────────────────

class ErrorCode(str, Enum):
    # Transient — safe to retry
    TIMEOUT              = "TIMEOUT"
    RATE_LIMITED         = "RATE_LIMITED"
    PARSE_FAILURE        = "PARSE_FAILURE"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    WORKER_OVERLOADED    = "WORKER_OVERLOADED"

    # Permanent — do not retry
    INVALID_INPUT     = "INVALID_INPUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_IMPLEMENTED   = "NOT_IMPLEMENTED"
    SCHEMA_VIOLATION  = "SCHEMA_VIOLATION"
    CONTEXT_TOO_LONG  = "CONTEXT_TOO_LONG"


TRANSIENT_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.TIMEOUT,
    ErrorCode.RATE_LIMITED,
    ErrorCode.PARSE_FAILURE,
    ErrorCode.UPSTREAM_UNAVAILABLE,
    ErrorCode.WORKER_OVERLOADED,
})


# ── Retry policy ───────────────────────────────────────────────────────────────

class BackoffStrategy(str, Enum):
    FIXED       = "fixed"
    LINEAR      = "linear"
    EXPONENTIAL = "exponential"


class RetryPolicy(BaseModel):
    """
    Governs how the RetryEngine retries failed tasks.

    ``max_attempts`` is the total number of attempts including the first.
    ``backoff_seconds`` is the base unit for delay computation.
    """

    max_attempts:     int             = Field(3,   ge=1, le=20)
    backoff_seconds:  float           = Field(1.0, ge=0.0)
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter:           bool            = Field(False, description="Add ±25% randomness to backoff")

    def delay_for(self, attempt: int) -> float:
        """
        Sleep duration (seconds) before ``attempt`` number.
        ``attempt`` is 1-indexed; the first call passes attempt=1 → 0.0 delay.
        """
        if attempt <= 1:
            return 0.0
        n = attempt - 1
        match self.backoff_strategy:
            case BackoffStrategy.FIXED:
                delay = self.backoff_seconds
            case BackoffStrategy.LINEAR:
                delay = self.backoff_seconds * n
            case BackoffStrategy.EXPONENTIAL:
                delay = self.backoff_seconds * (2 ** (n - 1))
        if self.jitter:
            import random
            delay *= 0.75 + random.random() * 0.5
        return round(delay, 3)

    model_config = {"frozen": True}


# ── Error envelope ─────────────────────────────────────────────────────────────

class ErrorEnvelope(BaseModel):
    """
    Structured error returned by a worker on failure.

    ``is_transient`` is a *computed* field derived from ``error_code``.
    Callers set only ``error_code``; the classification is enforced by
    the schema and cannot be overridden.
    """

    error_code:    ErrorCode
    message:       str       = Field(..., min_length=1)
    attempt_count: int       = Field(..., ge=1)
    timestamp:     datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail:        str | None = None

    @computed_field
    @property
    def is_transient(self) -> bool:
        return self.error_code in TRANSIENT_CODES


# ── Task dispatch ──────────────────────────────────────────────────────────────

class TaskDispatch(BaseModel):
    """
    What the conductor sends to a worker for a single attempt.

    ``attempt_count`` is set to 1 on first dispatch and incremented by
    the RetryEngine before each re-dispatch.
    """

    task_id:       str            = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    instruction:   str            = Field(..., min_length=1)
    context:       dict[str, Any] = Field(default_factory=dict)
    constraints:   list[str]      = Field(default_factory=list)
    retry_policy:  RetryPolicy    = Field(default_factory=RetryPolicy)
    attempt_count: int            = Field(1, ge=1)
    dispatched_at: datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}


# ── Worker response ────────────────────────────────────────────────────────────

class WorkerResponse(BaseModel):
    """
    What a worker returns to the conductor.

    ``output`` and ``error`` are mutually exclusive.
    - status=complete: output required, error must be None
    - status=failed:   error required, output must be None
    """

    task_id:       str
    status:        Literal["complete", "failed"]
    attempt_count: int            = Field(..., ge=1)
    output:        dict[str, Any] | None = None
    error:         ErrorEnvelope  | None = None
    responded_at:  datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _enforce_output_xor_error(self) -> "WorkerResponse":
        if self.status == "complete":
            if self.output is None:
                raise ValueError("output is required when status='complete'")
            if self.error is not None:
                raise ValueError("error must be None when status='complete'")
        if self.status == "failed":
            if self.error is None:
                raise ValueError("error is required when status='failed'")
            if self.output is not None:
                raise ValueError("output must be None when status='failed'")
        return self


# ── Reviewer verdict ───────────────────────────────────────────────────────────

class ReviewerVerdict(BaseModel):
    """
    Result from the VerificationOracle.

    ``passed=True`` means the output satisfies the expected schema and
    any extra validators. ``passed=False`` blocks the output from being
    marked terminal-success.
    """

    passed:           bool
    reasons:          list[str]        = Field(default_factory=list)
    validated_output: dict[str, Any]   | None = None  # set on pass
    reviewed_at:      datetime         = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Attempt tracking ───────────────────────────────────────────────────────────

class RetryDecision(str, Enum):
    RETRY     = "retry"      # transient error, attempts remain
    ESCALATE  = "escalate"   # permanent error, stopped immediately
    TERMINATE = "terminate"  # attempts exhausted (transient or oracle rejections)


class AttemptRecord(BaseModel):
    """One entry in the per-task retry log."""

    attempt:                    int
    task_id:                    str
    dispatched_at:              datetime
    responded_at:               datetime
    worker_status:              Literal["complete", "failed"]
    error:                      ErrorEnvelope  | None = None
    oracle_verdict:             ReviewerVerdict | None = None
    retry_decision:             RetryDecision  | None = None
    backoff_applied_seconds:    float = 0.0
    cumulative_elapsed_seconds: float = 0.0


class AttemptState(BaseModel):
    """Mutable attempt counter and full history for one logical task."""

    task_id:       str
    attempt_count: int                  = Field(1, ge=1)
    history:       list[AttemptRecord]  = Field(default_factory=list)

    @computed_field
    @property
    def total_attempts_made(self) -> int:
        return len(self.history)


# ── Terminal result ────────────────────────────────────────────────────────────

class TerminalResult(BaseModel):
    """
    Final sealed outcome of a task run.

    ``status='success'`` is only valid when:
      - ``output`` is set
      - ``verdict.passed is True``

    This invariant is enforced by the model validator. The VerificationOracle
    is the mandatory gate that must pass before any output can be marked final.
    """

    task_id:       str
    status:        Literal["success", "failure"]
    output:        dict[str, Any]   | None = None
    final_error:   ErrorEnvelope    | None = None
    verdict:       ReviewerVerdict  | None = None
    attempt_state: AttemptState
    completed_at:  datetime         = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _success_requires_passing_verdict(self) -> "TerminalResult":
        if self.status == "success":
            if self.output is None:
                raise ValueError("output is required on a successful TerminalResult")
            if self.verdict is None:
                raise ValueError("a ReviewerVerdict is required on a successful TerminalResult")
            if not self.verdict.passed:
                raise ValueError(
                    "TerminalResult status='success' requires verdict.passed=True — "
                    "the VerificationOracle must approve the output before it can be marked final"
                )
        return self
