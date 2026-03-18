"""
tests/test_middleware.py
~~~~~~~~~~~~~~~~~~~~~~~~
Pytest suite exercising all four middleware paths:

  1. success_path              — worker succeeds first try, oracle passes
  2. transient_retry_recovery  — worker fails (TIMEOUT) twice, succeeds third
  3. permanent_failure_stop    — worker fails (INVALID_INPUT), no retry
  4. reviewer_rejection_redispatch — worker output fails oracle once,
                                    second attempt passes

Each test writes a structured JSON log to logs/<test_name>.json.
No real API calls — all workers are pure-Python callables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from conductor_middleware import (
    BackoffStrategy,
    ErrorCode,
    ErrorEnvelope,
    RetryDecision,
    RetryEngine,
    RetryPolicy,
    ReviewerVerdict,
    TaskDispatch,
    TerminalResult,
    VerificationOracle,
    WorkerResponse,
)

LOGS = Path(__file__).parent.parent / "logs"
LOGS.mkdir(exist_ok=True)


# ── Shared output schema ───────────────────────────────────────────────────────

class SentimentOutput(BaseModel):
    label:      str
    confidence: float
    reasoning:  str


# ── Log helper ─────────────────────────────────────────────────────────────────

def write_log(name: str, result: TerminalResult) -> None:
    payload = {
        "test":         name,
        "logged_at":    datetime.now(timezone.utc).isoformat(),
        "terminal_result": result.model_dump(mode="json"),
    }
    path = LOGS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def fast_policy() -> RetryPolicy:
    """Zero-sleep exponential policy for fast test runs."""
    return RetryPolicy(
        max_attempts=4,
        backoff_seconds=0.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
    )


@pytest.fixture
def oracle() -> VerificationOracle:
    return VerificationOracle(SentimentOutput)


@pytest.fixture
def engine(fast_policy: RetryPolicy) -> RetryEngine:
    return RetryEngine(policy=fast_policy, sleep_fn=lambda _: None)


@pytest.fixture
def base_dispatch(fast_policy: RetryPolicy) -> TaskDispatch:
    return TaskDispatch(
        instruction="Classify the sentiment of the provided text.",
        context={"text": "I absolutely loved the new release!"},
        retry_policy=fast_policy,
    )


# ── Helper workers ─────────────────────────────────────────────────────────────

def _good_response(dispatch: TaskDispatch) -> WorkerResponse:
    return WorkerResponse(
        task_id=dispatch.task_id,
        status="complete",
        attempt_count=dispatch.attempt_count,
        output={"label": "positive", "confidence": 0.95, "reasoning": "Strong positive language."},
    )


def _transient_fail_then_succeed(fail_times: int):
    call_count = 0

    def worker(dispatch: TaskDispatch) -> WorkerResponse:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            return WorkerResponse(
                task_id=dispatch.task_id,
                status="failed",
                attempt_count=dispatch.attempt_count,
                error=ErrorEnvelope(
                    error_code=ErrorCode.TIMEOUT,
                    message=f"Timed out on call {call_count} (simulated).",
                    attempt_count=dispatch.attempt_count,
                ),
            )
        return WorkerResponse(
            task_id=dispatch.task_id,
            status="complete",
            attempt_count=dispatch.attempt_count,
            output={"label": "positive", "confidence": 0.91, "reasoning": "Recovered after retry."},
        )

    return worker


def _always_permanent_fail(dispatch: TaskDispatch) -> WorkerResponse:
    return WorkerResponse(
        task_id=dispatch.task_id,
        status="failed",
        attempt_count=dispatch.attempt_count,
        error=ErrorEnvelope(
            error_code=ErrorCode.INVALID_INPUT,
            message="Instruction is structurally invalid — cannot process.",
            attempt_count=dispatch.attempt_count,
        ),
    )


def _output_improves_on_retry(bad_output: dict, good_output: dict):
    """Returns bad output first, then good output on subsequent calls."""
    call_count = 0

    def worker(dispatch: TaskDispatch) -> WorkerResponse:
        nonlocal call_count
        call_count += 1
        output = bad_output if call_count == 1 else good_output
        return WorkerResponse(
            task_id=dispatch.task_id,
            status="complete",
            attempt_count=dispatch.attempt_count,
            output=output,
        )

    return worker


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSuccessPath:
    """Worker succeeds first try; oracle approves; TerminalResult=success."""

    def test_status_is_success(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        assert result.status == "success"

    def test_output_matches_worker(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        assert result.output["label"] == "positive"
        assert result.output["confidence"] == pytest.approx(0.95)

    def test_exactly_one_attempt(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        assert result.attempt_state.total_attempts_made == 1

    def test_verdict_passed(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        assert result.verdict is not None
        assert result.verdict.passed is True

    def test_single_attempt_decision_is_terminate(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        record = result.attempt_state.history[0]
        assert record.retry_decision == RetryDecision.TERMINATE

    def test_pydantic_roundtrip(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        raw = result.model_dump(mode="json")
        TerminalResult.model_validate(raw)  # must not raise

    def test_success_requires_passing_verdict(self, engine, base_dispatch):
        """TerminalResult.status='success' must be rejected if verdict is absent."""
        with pytest.raises(Exception, match="ReviewerVerdict"):
            TerminalResult(
                task_id=base_dispatch.task_id,
                status="success",
                output={"label": "positive", "confidence": 0.9, "reasoning": "test"},
                verdict=None,
                attempt_state=AttemptState(task_id=base_dispatch.task_id),
            )

    def test_writes_log(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _good_response, oracle)
        write_log("success_path", result)
        log = json.loads((LOGS / "success_path.json").read_text())
        assert log["terminal_result"]["status"] == "success"


class TestTransientRetryRecovery:
    """Worker fails with TIMEOUT twice, then succeeds on third attempt."""

    def test_final_status_success(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        assert result.status == "success"

    def test_three_attempts_made(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        assert result.attempt_state.total_attempts_made == 3

    def test_first_two_are_timeout(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        for rec in result.attempt_state.history[:2]:
            assert rec.error is not None
            assert rec.error.error_code == ErrorCode.TIMEOUT

    def test_timeout_is_transient(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        for rec in result.attempt_state.history[:2]:
            assert rec.error.is_transient is True

    def test_first_two_decisions_are_retry(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        assert result.attempt_state.history[0].retry_decision == RetryDecision.RETRY
        assert result.attempt_state.history[1].retry_decision == RetryDecision.RETRY

    def test_third_attempt_succeeds(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        assert result.attempt_state.history[2].retry_decision == RetryDecision.TERMINATE
        assert result.attempt_state.history[2].oracle_verdict.passed is True

    def test_writes_log(self, engine, base_dispatch, oracle):
        worker = _transient_fail_then_succeed(fail_times=2)
        result = engine.run(base_dispatch, worker, oracle)
        write_log("transient_retry_recovery", result)
        log = json.loads((LOGS / "transient_retry_recovery.json").read_text())
        attempts = log["terminal_result"]["attempt_state"]["history"]
        assert len(attempts) == 3
        assert attempts[0]["error"]["error_code"] == "TIMEOUT"
        assert attempts[0]["error"]["is_transient"] is True
        assert log["terminal_result"]["status"] == "success"


class TestPermanentFailureStop:
    """Worker fails with INVALID_INPUT; no retry attempted; immediate terminal."""

    def test_final_status_failure(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        assert result.status == "failure"

    def test_exactly_one_attempt(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        assert result.attempt_state.total_attempts_made == 1

    def test_error_code_is_invalid_input(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        assert result.final_error.error_code == ErrorCode.INVALID_INPUT

    def test_is_not_transient(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        assert result.final_error.is_transient is False

    def test_decision_is_escalate(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        assert result.attempt_state.history[0].retry_decision == RetryDecision.ESCALATE

    def test_no_backoff_applied(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        assert result.attempt_state.history[0].backoff_applied_seconds == 0.0

    def test_writes_log(self, engine, base_dispatch, oracle):
        result = engine.run(base_dispatch, _always_permanent_fail, oracle)
        write_log("permanent_failure_stop", result)
        log = json.loads((LOGS / "permanent_failure_stop.json").read_text())
        assert log["terminal_result"]["status"] == "failure"
        attempts = log["terminal_result"]["attempt_state"]["history"]
        assert len(attempts) == 1
        assert attempts[0]["retry_decision"] == "escalate"
        assert attempts[0]["error"]["is_transient"] is False


class TestReviewerRejectionRedispatch:
    """Worker output fails oracle on first attempt; second attempt passes."""

    def test_final_status_success(self, engine, base_dispatch, oracle):
        worker = _output_improves_on_retry(
            bad_output={"label": "positive"},          # missing confidence + reasoning
            good_output={"label": "positive", "confidence": 0.88, "reasoning": "Clear markers."},
        )
        result = engine.run(base_dispatch, worker, oracle)
        assert result.status == "success"

    def test_two_attempts_made(self, engine, base_dispatch, oracle):
        worker = _output_improves_on_retry(
            bad_output={"label": "positive"},
            good_output={"label": "positive", "confidence": 0.88, "reasoning": "Clear markers."},
        )
        result = engine.run(base_dispatch, worker, oracle)
        assert result.attempt_state.total_attempts_made == 2

    def test_first_verdict_failed(self, engine, base_dispatch, oracle):
        worker = _output_improves_on_retry(
            bad_output={"label": "positive"},
            good_output={"label": "positive", "confidence": 0.88, "reasoning": "Clear markers."},
        )
        result = engine.run(base_dispatch, worker, oracle)
        first = result.attempt_state.history[0]
        assert first.worker_status == "complete"   # worker didn't fail — oracle did
        assert first.oracle_verdict is not None
        assert first.oracle_verdict.passed is False
        assert first.retry_decision == RetryDecision.RETRY

    def test_first_verdict_has_reasons(self, engine, base_dispatch, oracle):
        worker = _output_improves_on_retry(
            bad_output={"label": "positive"},
            good_output={"label": "positive", "confidence": 0.88, "reasoning": "Clear markers."},
        )
        result = engine.run(base_dispatch, worker, oracle)
        first_verdict = result.attempt_state.history[0].oracle_verdict
        assert len(first_verdict.reasons) > 0

    def test_second_verdict_passed(self, engine, base_dispatch, oracle):
        worker = _output_improves_on_retry(
            bad_output={"label": "positive"},
            good_output={"label": "positive", "confidence": 0.88, "reasoning": "Clear markers."},
        )
        result = engine.run(base_dispatch, worker, oracle)
        second = result.attempt_state.history[1]
        assert second.oracle_verdict.passed is True

    def test_extra_validator_rejection(self, base_dispatch, engine):
        """Extra domain validator can also trigger rejection and re-dispatch."""
        def require_high_confidence(obj: SentimentOutput) -> str | None:
            if obj.confidence < 0.9:
                return f"confidence {obj.confidence:.2f} is below threshold 0.90"

        strict_oracle = VerificationOracle(SentimentOutput, extra_validators=[require_high_confidence])

        low_then_high = _output_improves_on_retry(
            bad_output={"label": "positive", "confidence": 0.75, "reasoning": "OK."},
            good_output={"label": "positive", "confidence": 0.95, "reasoning": "Strong."},
        )
        result = engine.run(base_dispatch, low_then_high, strict_oracle)
        assert result.status == "success"
        assert result.attempt_state.total_attempts_made == 2
        first_reason = result.attempt_state.history[0].oracle_verdict.reasons[0]
        assert "threshold" in first_reason

    def test_writes_log(self, engine, base_dispatch, oracle):
        worker = _output_improves_on_retry(
            bad_output={"label": "positive"},
            good_output={"label": "positive", "confidence": 0.88, "reasoning": "Clear markers."},
        )
        result = engine.run(base_dispatch, worker, oracle)
        write_log("reviewer_rejection_redispatch", result)
        log = json.loads((LOGS / "reviewer_rejection_redispatch.json").read_text())
        attempts = log["terminal_result"]["attempt_state"]["history"]
        assert len(attempts) == 2
        assert attempts[0]["oracle_verdict"]["passed"] is False
        assert attempts[0]["retry_decision"] == "retry"
        assert attempts[1]["oracle_verdict"]["passed"] is True
        assert log["terminal_result"]["status"] == "success"


class TestVerificationOracle:
    """Unit tests for the oracle in isolation."""

    def test_valid_input_passes(self, oracle):
        v = oracle({"label": "negative", "confidence": 0.8, "reasoning": "Sad words."})
        assert v.passed is True
        assert v.validated_output is not None

    def test_missing_field_fails(self, oracle):
        v = oracle({"label": "negative"})
        assert v.passed is False
        assert len(v.reasons) > 0
        assert any("confidence" in r or "reasoning" in r for r in v.reasons)

    def test_wrong_type_fails(self, oracle):
        v = oracle({"label": "negative", "confidence": "not-a-float", "reasoning": "x"})
        assert v.passed is False

    def test_extra_validator_triggered(self):
        def no_neutral(obj: SentimentOutput) -> str | None:
            if obj.label == "neutral":
                return "neutral label is not allowed in this context"

        strict = VerificationOracle(SentimentOutput, extra_validators=[no_neutral])
        v = strict({"label": "neutral", "confidence": 0.6, "reasoning": "Meh."})
        assert v.passed is False
        assert "neutral" in v.reasons[0]

    def test_oracle_repr(self, oracle):
        assert "SentimentOutput" in repr(oracle)


class TestRetryPolicyBackoff:
    """Verify backoff computation for all strategies."""

    def test_exponential(self):
        p = RetryPolicy(backoff_seconds=1.0, backoff_strategy=BackoffStrategy.EXPONENTIAL)
        assert p.delay_for(1) == 0.0
        assert p.delay_for(2) == 1.0
        assert p.delay_for(3) == 2.0
        assert p.delay_for(4) == 4.0

    def test_linear(self):
        p = RetryPolicy(backoff_seconds=0.5, backoff_strategy=BackoffStrategy.LINEAR)
        assert p.delay_for(2) == 0.5
        assert p.delay_for(3) == 1.0
        assert p.delay_for(4) == 1.5

    def test_fixed(self):
        p = RetryPolicy(backoff_seconds=2.0, backoff_strategy=BackoffStrategy.FIXED)
        assert p.delay_for(2) == 2.0
        assert p.delay_for(3) == 2.0
        assert p.delay_for(10) == 2.0

    def test_first_attempt_no_delay(self):
        p = RetryPolicy(backoff_seconds=5.0)
        assert p.delay_for(1) == 0.0


class TestSchemaInvariants:
    """Pydantic model validation rules."""

    def test_worker_response_complete_requires_output(self):
        with pytest.raises(Exception):
            WorkerResponse(task_id="t1", status="complete", attempt_count=1, output=None)

    def test_worker_response_failed_requires_error(self):
        with pytest.raises(Exception):
            WorkerResponse(task_id="t1", status="failed", attempt_count=1, error=None)

    def test_timeout_is_transient(self):
        e = ErrorEnvelope(error_code=ErrorCode.TIMEOUT, message="timed out", attempt_count=1)
        assert e.is_transient is True

    def test_invalid_input_is_not_transient(self):
        e = ErrorEnvelope(error_code=ErrorCode.INVALID_INPUT, message="bad", attempt_count=1)
        assert e.is_transient is False

    def test_terminal_result_success_needs_verdict(self):
        from conductor_middleware.models import AttemptState
        with pytest.raises(Exception, match="ReviewerVerdict"):
            TerminalResult(
                task_id="t1",
                status="success",
                output={"label": "positive", "confidence": 0.9, "reasoning": "ok"},
                verdict=None,
                attempt_state=AttemptState(task_id="t1"),
            )

    def test_terminal_result_success_needs_passing_verdict(self):
        from conductor_middleware.models import AttemptState
        with pytest.raises(Exception, match="passed=True"):
            TerminalResult(
                task_id="t1",
                status="success",
                output={"label": "positive", "confidence": 0.9, "reasoning": "ok"},
                verdict=ReviewerVerdict(passed=False, reasons=["failed"]),
                attempt_state=AttemptState(task_id="t1"),
            )


# Import AttemptState for inline use in schema tests above
from conductor_middleware.models import AttemptState  # noqa: E402
