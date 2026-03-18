"""
conductor_middleware.retry_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RetryEngine — applies RetryPolicy to a worker callable, wires the
VerificationOracle as a mandatory gate, and returns a TerminalResult.

High-level usage::

    engine = RetryEngine(policy=RetryPolicy(max_attempts=3, backoff_seconds=0.5))
    result = engine.run(dispatch=task, worker=my_worker, oracle=my_oracle)

    if result.status == "success":
        print(result.output)
    else:
        print(result.final_error)

Low-level (if you need fine-grained control)::

    decision, state = engine.evaluate(response, state, verdict=None)

Retry loop logic
----------------
For each attempt up to ``policy.max_attempts``:

  1. Call worker(dispatch) — catch ValidationError and bare exceptions,
     wrapping both as PARSE_FAILURE / UPSTREAM_UNAVAILABLE respectively.
  2. If worker failed with a *permanent* error → ESCALATE immediately.
  3. If worker succeeded → call oracle(output).
     - If oracle passes → seal as TerminalResult(success).
     - If oracle fails  → treat like a transient failure (RETRY / TERMINATE).
  4. If transient error or oracle rejection and attempts remain → apply
     backoff sleep, increment attempt_count, loop.
  5. If attempts exhausted → TerminalResult(failure, TERMINATE).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError

from .models import (
    AttemptRecord,
    AttemptState,
    ErrorCode,
    ErrorEnvelope,
    RetryDecision,
    RetryPolicy,
    TaskDispatch,
    TerminalResult,
    WorkerResponse,
)
from .oracle import VerificationOracle


# A worker is any callable that accepts a TaskDispatch and returns a WorkerResponse.
WorkerCallable = Callable[[TaskDispatch], WorkerResponse]


class RetryEngine:
    """
    Retry loop with schema-gated VerificationOracle.

    Parameters
    ----------
    policy:
        :class:`RetryPolicy` applied to every task run.  Can be overridden
        per-task by setting ``dispatch.retry_policy``.
    sleep_fn:
        Defaults to ``time.sleep``; injectable for testing without real waits.
    """

    def __init__(
        self,
        policy:   RetryPolicy | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._default_policy = policy or RetryPolicy()
        self._sleep = sleep_fn

    # ── High-level entry point ─────────────────────────────────────────────────

    def run(
        self,
        dispatch: TaskDispatch,
        worker:   WorkerCallable,
        oracle:   VerificationOracle,
    ) -> TerminalResult:
        """
        Run the full dispatch → worker → oracle loop.

        Returns a :class:`TerminalResult` whose ``status`` is either
        ``'success'`` or ``'failure'``.  Success is only possible when the
        VerificationOracle approves the worker's output.
        """
        policy = dispatch.retry_policy
        state  = AttemptState(task_id=dispatch.task_id)
        t_run  = time.monotonic()

        for attempt_num in range(1, policy.max_attempts + 1):
            # Re-build dispatch with current attempt_count (TaskDispatch is frozen)
            current_dispatch = dispatch.model_copy(
                update={"attempt_count": attempt_num,
                        "dispatched_at": datetime.now(timezone.utc)}
            )
            t_dispatch = current_dispatch.dispatched_at

            # ── Call worker, catching schema + unexpected exceptions ────────────
            response = self._safe_call_worker(worker, current_dispatch, attempt_num, policy)

            elapsed = round(time.monotonic() - t_run, 3)

            # ── Worker failed ──────────────────────────────────────────────────
            if response.status == "failed":
                error    = response.error
                is_last  = attempt_num >= policy.max_attempts
                decision = (
                    RetryDecision.ESCALATE   if not error.is_transient else
                    RetryDecision.TERMINATE  if is_last else
                    RetryDecision.RETRY
                )
                backoff  = policy.delay_for(attempt_num + 1) if decision == RetryDecision.RETRY else 0.0

                state.history.append(AttemptRecord(
                    attempt=attempt_num,
                    task_id=dispatch.task_id,
                    dispatched_at=t_dispatch,
                    responded_at=response.responded_at,
                    worker_status="failed",
                    error=error,
                    retry_decision=decision,
                    backoff_applied_seconds=backoff,
                    cumulative_elapsed_seconds=elapsed,
                ))
                state = state.model_copy(update={"attempt_count": attempt_num})

                if decision in (RetryDecision.ESCALATE, RetryDecision.TERMINATE):
                    return TerminalResult(
                        task_id=dispatch.task_id,
                        status="failure",
                        final_error=error,
                        attempt_state=state,
                    )

                self._sleep(backoff)
                continue

            # ── Worker succeeded → oracle gate ─────────────────────────────────
            verdict = oracle(response.output or {})

            if verdict.passed:
                state.history.append(AttemptRecord(
                    attempt=attempt_num,
                    task_id=dispatch.task_id,
                    dispatched_at=t_dispatch,
                    responded_at=response.responded_at,
                    worker_status="complete",
                    oracle_verdict=verdict,
                    retry_decision=RetryDecision.TERMINATE,   # loop ends — success
                    cumulative_elapsed_seconds=elapsed,
                ))
                state = state.model_copy(update={"attempt_count": attempt_num})
                return TerminalResult(
                    task_id=dispatch.task_id,
                    status="success",
                    output=response.output,
                    verdict=verdict,
                    attempt_state=state,
                )

            # Oracle rejected — treat like a transient retry
            is_last  = attempt_num >= policy.max_attempts
            decision = RetryDecision.TERMINATE if is_last else RetryDecision.RETRY
            backoff  = policy.delay_for(attempt_num + 1) if decision == RetryDecision.RETRY else 0.0

            state.history.append(AttemptRecord(
                attempt=attempt_num,
                task_id=dispatch.task_id,
                dispatched_at=t_dispatch,
                responded_at=response.responded_at,
                worker_status="complete",
                oracle_verdict=verdict,
                retry_decision=decision,
                backoff_applied_seconds=backoff,
                cumulative_elapsed_seconds=elapsed,
            ))
            state = state.model_copy(update={"attempt_count": attempt_num})

            if decision == RetryDecision.TERMINATE:
                oracle_failure = ErrorEnvelope(
                    error_code=ErrorCode.SCHEMA_VIOLATION,
                    message="VerificationOracle rejected worker output after all attempts.",
                    attempt_count=attempt_num,
                    detail="; ".join(verdict.reasons),
                )
                return TerminalResult(
                    task_id=dispatch.task_id,
                    status="failure",
                    final_error=oracle_failure,
                    verdict=verdict,
                    attempt_state=state,
                )

            self._sleep(backoff)

        # Should be unreachable, but guard it
        return TerminalResult(
            task_id=dispatch.task_id,
            status="failure",
            attempt_state=state,
        )

    # ── Low-level primitive ────────────────────────────────────────────────────

    def evaluate(
        self,
        response: WorkerResponse,
        state:    AttemptState,
        verdict:  "ReviewerVerdict | None" = None,
    ) -> tuple[RetryDecision, AttemptState]:
        """
        Evaluate a single response and return the retry decision.

        Does *not* perform any I/O or sleep.  Useful for conductors that
        need finer control over the dispatch cycle.

        Parameters
        ----------
        response:
            The worker response for the current attempt.
        state:
            Current :class:`AttemptState` (will not be mutated).
        verdict:
            Pre-computed oracle verdict, or ``None`` if not yet checked.

        Returns
        -------
        ``(decision, updated_state)``
        """
        policy  = AttemptState.model_fields  # not used here — use dispatch policy
        is_last = state.attempt_count >= state.attempt_count  # placeholder

        if response.status == "failed":
            error = response.error
            if not error.is_transient:
                return RetryDecision.ESCALATE, state
            return RetryDecision.RETRY, state

        if verdict is not None and not verdict.passed:
            return RetryDecision.RETRY, state

        return RetryDecision.TERMINATE, state   # success / done

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _safe_call_worker(
        self,
        worker:       WorkerCallable,
        dispatch:     TaskDispatch,
        attempt_num:  int,
        policy:       RetryPolicy,
    ) -> WorkerResponse:
        """
        Call ``worker(dispatch)`` and convert any exception into a failed
        WorkerResponse with an appropriate error code.
        """
        try:
            return worker(dispatch)
        except ValidationError as exc:
            return WorkerResponse(
                task_id=dispatch.task_id,
                status="failed",
                attempt_count=attempt_num,
                error=ErrorEnvelope(
                    error_code=ErrorCode.PARSE_FAILURE,
                    message=(
                        f"Worker response failed schema validation on attempt {attempt_num}: "
                        f"{exc.error_count()} Pydantic error(s)."
                    ),
                    attempt_count=attempt_num,
                    detail=exc.json(indent=2),
                ),
            )
        except Exception as exc:
            return WorkerResponse(
                task_id=dispatch.task_id,
                status="failed",
                attempt_count=attempt_num,
                error=ErrorEnvelope(
                    error_code=ErrorCode.UPSTREAM_UNAVAILABLE,
                    message=f"Worker raised {type(exc).__name__}: {exc}",
                    attempt_count=attempt_num,
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )
