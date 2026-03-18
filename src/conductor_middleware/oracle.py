"""
conductor_middleware.oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~~
VerificationOracle — the mandatory reviewer gate.

Usage::

    from conductor_middleware import VerificationOracle
    from pydantic import BaseModel

    class SentimentOutput(BaseModel):
        label: str
        confidence: float

    oracle = VerificationOracle(SentimentOutput)
    verdict = oracle({"label": "positive", "confidence": 0.92})
    assert verdict.passed

    verdict = oracle({"label": "positive"})        # missing field
    assert not verdict.passed
    assert "confidence" in verdict.reasons[0]

Extra validators
----------------
Pass a list of callables as ``extra_validators``.  Each receives the
*validated* Pydantic instance and may return a non-empty string to fail::

    def must_be_confident(obj: SentimentOutput) -> str | None:
        if obj.confidence < 0.5:
            return f"confidence {obj.confidence} is below threshold 0.5"

    oracle = VerificationOracle(SentimentOutput, extra_validators=[must_be_confident])
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .models import ReviewerVerdict


class VerificationOracle:
    """
    Callable that validates an arbitrary dict against a supplied Pydantic schema.

    Returns a :class:`ReviewerVerdict`.  ``verdict.passed=True`` is the only
    condition under which the :class:`RetryEngine` will seal the output as a
    successful :class:`TerminalResult`.

    Parameters
    ----------
    schema:
        Any Pydantic ``BaseModel`` subclass.  The oracle will attempt
        ``schema.model_validate(output)`` before running extra validators.
    extra_validators:
        Optional list of callables ``(validated_instance) -> str | None``.
        Return a non-empty string to fail with that message; return ``None``
        (or an empty string) to pass.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        extra_validators: list[Callable[[BaseModel], str | None]] | None = None,
    ) -> None:
        self._schema    = schema
        self._extra_validators = extra_validators or []

    # ── callable interface ─────────────────────────────────────────────────────

    def __call__(self, output: dict[str, Any]) -> ReviewerVerdict:
        """
        Validate ``output`` dict and return a :class:`ReviewerVerdict`.

        Never raises; all errors are captured as verdict reasons.
        """
        # Step 1 — structural validation via Pydantic
        try:
            validated = self._schema.model_validate(output)
        except ValidationError as exc:
            reasons = [
                f"{'.'.join(str(loc) for loc in err['loc']) or 'root'}: {err['msg']}"
                for err in exc.errors()
            ]
            return ReviewerVerdict(passed=False, reasons=reasons)

        # Step 2 — extra domain validators
        extra_failures: list[str] = []
        for validator in self._extra_validators:
            try:
                result = validator(validated)
                if result:   # non-empty string = failure
                    extra_failures.append(str(result))
            except Exception as exc:  # validator itself crashed
                extra_failures.append(f"validator {validator.__name__!r} raised {type(exc).__name__}: {exc}")

        if extra_failures:
            return ReviewerVerdict(passed=False, reasons=extra_failures)

        return ReviewerVerdict(
            passed=True,
            validated_output=output,
        )

    def __repr__(self) -> str:
        extras = len(self._extra_validators)
        return (
            f"VerificationOracle(schema={self._schema.__name__!r}, "
            f"extra_validators={extras})"
        )
