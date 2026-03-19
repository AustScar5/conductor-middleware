# conductor-middleware

Schema-validated retry and reviewer-gating middleware for Task Node conductor/agent workflows.

Any agent developer can `pip install` this package to get:

- **Pydantic models** for typed dispatch/response contracts with enforced error classification
- **RetryEngine** with configurable exponential, linear, or fixed backoff
- **VerificationOracle** — a mandatory schema-validated gate that must approve any output before it can be marked final
- Structured JSON attempt logs for every run

---

## Installation

```bash
# From PyPI (once published)
pip install conductor-middleware

# Directly from GitHub
pip install git+https://github.com/<your-org>/conductor-middleware.git
```

Requires Python 3.11+ and Pydantic v2.

---

## Quick start

```python
from conductor_middleware import (
    RetryPolicy,
    RetryEngine,
    VerificationOracle,
    TaskDispatch,
    WorkerResponse,
    ErrorEnvelope,
    ErrorCode,
)
from pydantic import BaseModel


# 1. Define your expected output schema
class SentimentOutput(BaseModel):
    label:      str
    confidence: float
    reasoning:  str


# 2. Create the oracle and engine
oracle = VerificationOracle(SentimentOutput)
engine = RetryEngine(
    policy=RetryPolicy(
        max_attempts=3,
        backoff_seconds=1.0,
        backoff_strategy="exponential",
    )
)


# 3. Write your worker as a plain callable
def my_sentiment_worker(dispatch: TaskDispatch) -> WorkerResponse:
    text = dispatch.context.get("text", "")
    # ... call your LLM or service here ...
    return WorkerResponse(
        task_id=dispatch.task_id,
        status="complete",
        attempt_count=dispatch.attempt_count,
        output={"label": "positive", "confidence": 0.92, "reasoning": "Strong positive markers."},
    )


# 4. Run
dispatch = TaskDispatch(
    instruction="Classify the sentiment of this tweet.",
    context={"text": "I absolutely loved the new release!"},
)

result = engine.run(dispatch=dispatch, worker=my_sentiment_worker, oracle=oracle)

if result.status == "success":
    print(result.output)          # validated dict
    print(result.verdict.passed)  # True — oracle approved
else:
    print(result.final_error)     # ErrorEnvelope with error_code + is_transient
```

---

## Retry behaviour

| Scenario | Engine action |
|---|---|
| Worker success + oracle pass | `TerminalResult(status="success")` |
| Worker failure, `is_transient=True`, attempts remain | Sleep (backoff), re-dispatch |
| Worker failure, `is_transient=False` | `TerminalResult(status="failure")`, `decision=ESCALATE` |
| Worker failure, transient, attempts exhausted | `TerminalResult(status="failure")`, `decision=TERMINATE` |
| Oracle rejection, attempts remain | Re-dispatch worker (oracle rejection = transient) |
| Oracle rejection, attempts exhausted | `TerminalResult(status="failure")` |

### Error classification

`ErrorCode` values are automatically classified as transient or permanent via the `is_transient` computed field — callers never set it manually:

```python
# Transient (retryable)
ErrorCode.TIMEOUT
ErrorCode.RATE_LIMITED
ErrorCode.PARSE_FAILURE
ErrorCode.UPSTREAM_UNAVAILABLE
ErrorCode.WORKER_OVERLOADED

# Permanent (terminal)
ErrorCode.INVALID_INPUT
ErrorCode.PERMISSION_DENIED
ErrorCode.NOT_IMPLEMENTED
ErrorCode.SCHEMA_VIOLATION
ErrorCode.CONTEXT_TOO_LONG
```

---

## VerificationOracle

The oracle is the mandatory reviewer gate. `TerminalResult(status="success")` is a **Pydantic-enforced invariant** — it cannot be constructed without a `ReviewerVerdict(passed=True)`.

```python
from conductor_middleware import VerificationOracle
from pydantic import BaseModel

class MyOutput(BaseModel):
    label: str
    confidence: float

oracle = VerificationOracle(MyOutput)

verdict = oracle({"label": "positive", "confidence": 0.9})
assert verdict.passed is True

verdict = oracle({"label": "positive"})   # missing field
assert verdict.passed is False
assert "confidence" in verdict.reasons[0]
```

### Extra validators

Pass callables that receive the validated Pydantic instance. Return a non-empty string to fail:

```python
def require_high_confidence(obj: MyOutput) -> str | None:
    if obj.confidence < 0.8:
        return f"confidence {obj.confidence:.2f} is below threshold 0.80"

oracle = VerificationOracle(MyOutput, extra_validators=[require_high_confidence])
```

---

## Wiring into an existing conductor

```python
from conductor_middleware import RetryEngine, RetryPolicy, VerificationOracle, TaskDispatch

class MyConductor:
    def __init__(self):
        self.engine = RetryEngine(
            policy=RetryPolicy(max_attempts=4, backoff_seconds=0.5)
        )

    def run_task(self, instruction: str, worker, output_schema) -> dict:
        oracle   = VerificationOracle(output_schema)
        dispatch = TaskDispatch(instruction=instruction)
        result   = self.engine.run(dispatch=dispatch, worker=worker, oracle=oracle)

        if result.status == "success":
            return result.output

        raise RuntimeError(
            f"Task failed after {result.attempt_state.total_attempts_made} attempts: "
            f"{result.final_error.message}"
        )
```

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest -v
```

Structured JSON logs for each test scenario are written to `logs/` after the test run.

---

## Package layout

```
src/conductor_middleware/
    __init__.py        # public API exports
    models.py          # Pydantic models (ErrorCode, RetryPolicy, TaskDispatch, …)
    oracle.py          # VerificationOracle
    retry_engine.py    # RetryEngine with full dispatch-execute-review loop
tests/
    test_middleware.py  # pytest suite — 4 scenarios, ~30 assertions
logs/                   # structured JSON logs emitted by tests
pyproject.toml
```

---

## Conductor Integration

Wire `AuthorizationGate` and `AdjudicationOracle` together in any conductor
reward-dispatch pipeline with about 10 lines:

```python
from conductor_middleware import (
    AuthorizationGate,
    AdjudicationOracle,
    AdjudicationLog,
    gated_adjudicate,
)
from pydantic import BaseModel

# Define the expected artifact schema for this task type
class TaskArtifact(BaseModel):
    summary:    str
    confidence: float
    sources:    list[str]

# Load the contributor allowlist and wire the oracle
gate   = AuthorizationGate("allowlist.json")   # JSON or TOML
log    = AdjudicationLog(path="logs/adjudication.json")
oracle = AdjudicationOracle(log=log)

# In your reward-dispatch handler:
verdict = gated_adjudicate(
    gate=gate,
    oracle=oracle,
    wallet_address=contributor_wallet,   # from task submission
    artifact=task_output,                # worker's completion artifact
    schema=TaskArtifact,
    task_id="task-abc123",
)

match verdict.verdict:
    case "pass":
        dispatch_reward(contributor_wallet, task_output)
    case "retry":
        redispatch_task(task_id, hint=verdict.reason)
    case "fail":
        # Check whether it was an auth block or a schema failure
        if "unauthorized_contributor" in verdict.reason:
            log_auth_violation(contributor_wallet, verdict.reason)
        else:
            escalate(verdict.field_errors)
```

### Allowlist config format

**JSON** (`allowlist.json`):
```json
{
  "allowlist": [
    {"wallet": "0xAABB...CCDD", "status": "authorized", "label": "node-alpha"},
    {"wallet": "0xDEAD...BEEF", "status": "suspended",
     "suspension_reason": "repeated verification failures"}
  ]
}
```

**TOML** (`allowlist.toml`):
```toml
[[allowlist]]
wallet = "0xAABB...CCDD"
status = "authorized"
label  = "node-alpha"

[[allowlist]]
wallet            = "0xDEAD...BEEF"
status            = "suspended"
suspension_reason = "repeated verification failures"
```

### Gate verdict status values

| Status | Meaning | `is_authorized` |
|---|---|---|
| `authorized` | On allowlist, not expired | `True` |
| `unauthorized` | Not on allowlist or malformed address | `False` |
| `suspended` | On allowlist but explicitly suspended | `False` |
| `expired` | Was authorized but `expires_at` is past | `False` |

Only `authorized` wallets reach the `AdjudicationOracle`.
All others receive an immediate `FAIL` verdict with `reason="unauthorized_contributor: ..."`.

---

## Task Node Integration

Any Task Node conductor can import the `AdjudicationOracle` and wire it as an
Authorization Gate with about 10 lines of code:

```python
from conductor_middleware import AdjudicationOracle, AdjudicationLog, ReviewerPolicy
from pydantic import BaseModel

# 1. Define the expected artifact schema for this task type
class DeliveryArtifact(BaseModel):
    summary:    str
    word_count: int
    sources:    list[str]

# 2. Optional domain rule — sources must not be empty
def require_sources(obj: DeliveryArtifact) -> str | None:
    if not obj.sources:
        return "sources list must not be empty"

# 3. Create oracle (attach a log for auditability)
log    = AdjudicationLog(path="logs/adjudication.json")
oracle = AdjudicationOracle(log=log)
policy = ReviewerPolicy("require_sources", [require_sources])

# 4. Call the gate — artifact comes from your worker agent
verdict = oracle.adjudicate(
    artifact=worker_output,
    schema=DeliveryArtifact,
    reviewer_policy=policy,
    task_id="task-abc123",
)

# 5. Route on the verdict
match verdict.verdict:
    case "pass":
        mark_task_complete(worker_output)
    case "retry":
        redispatch(worker, hint=verdict.reason)
    case "fail":
        escalate(verdict.reason, verdict.field_errors)

# 6. Verify artifact integrity later (tamper detection)
assert oracle.verify_integrity(worker_output, verdict.evidence_hash)
```

The `AdjudicationLog` writes a JSON audit trail to disk after every call.
Each record stores the verdict, evidence hash, and timestamp so any artifact
can be re-verified against its original hash at any time.

---

## License

MIT
