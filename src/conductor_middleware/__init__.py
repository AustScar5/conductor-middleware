"""
conductor_middleware
~~~~~~~~~~~~~~~~~~~~
Pip-installable verification and retry middleware for Task Node agents.

Quick start::

    from conductor_middleware import (
        RetryPolicy,
        RetryEngine,
        VerificationOracle,
        TaskDispatch,
    )
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        result: str
        confidence: float

    oracle = VerificationOracle(MyOutput)
    engine = RetryEngine(policy=RetryPolicy(max_attempts=3, backoff_seconds=0.5))

    dispatch = TaskDispatch(instruction="Do something useful.")
    result   = engine.run(dispatch=dispatch, worker=my_worker, oracle=oracle)
"""

from .models import (
    AttemptRecord,
    AttemptState,
    BackoffStrategy,
    ErrorCode,
    ErrorEnvelope,
    RetryDecision,
    RetryPolicy,
    ReviewerVerdict,
    TaskDispatch,
    TerminalResult,
    TRANSIENT_CODES,
    WorkerResponse,
)
from .oracle import VerificationOracle
from .retry_engine import RetryEngine
from .adjudication import (
    AdjudicationLog,
    AdjudicationOracle,
    AdjudicationRecord,
    AdjudicationVerdict,
    ReviewerPolicy,
    VerdictCode,
)

__all__ = [
    # models
    "AttemptRecord",
    "AttemptState",
    "BackoffStrategy",
    "ErrorCode",
    "ErrorEnvelope",
    "RetryDecision",
    "RetryPolicy",
    "ReviewerVerdict",
    "TaskDispatch",
    "TerminalResult",
    "TRANSIENT_CODES",
    "WorkerResponse",
    # oracle
    "VerificationOracle",
    # engine
    "RetryEngine",
    # adjudication
    "AdjudicationLog",
    "AdjudicationOracle",
    "AdjudicationRecord",
    "AdjudicationVerdict",
    "ReviewerPolicy",
    "VerdictCode",
]

__version__ = "0.2.0"
