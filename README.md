# conductor-middleware

Pip-installable middleware stack for Task Node conductors. Sequences
verification → adjudication → authorization → XRPL payout into a single
importable pipeline.

---

## MCP Server

The conductor pipeline is also exposed as a Model Context Protocol (MCP) server,
letting any AI agent or external tool invoke the three core operations over
stdio or HTTP-SSE without importing Python directly.

### Architecture

```
  AI agent / external tool
         │
         │  MCP protocol (stdio or HTTP-SSE)
         ▼
  ┌─────────────────────────────────────────┐
  │         MCP Server                       │
  │  task-node-conductor                     │
  │                                          │
  │  ┌────────────────────────────────────┐  │
  │  │  run_pipeline tool                  │  │
  │  │  check_authorization tool           │  │
  │  │  dry_run_payout tool                │  │
  │  └───────────────┬────────────────────┘  │
  └──────────────────┼──────────────────────┘
                     │  Python function calls
                     ▼
  ┌──────────────────────────────────────────┐
  │        TaskNodeConductor                  │
  │                                           │
  │  VerificationOracle                       │
  │       ↓                                   │
  │  AdjudicationOracle                       │
  │       ↓                                   │
  │  AuthorizationGate  ←  allowlist.json     │
  │       ↓                                   │
  │  PayoutTrigger      →  XRPL PFT Payment   │
  └──────────────────────────────────────────┘
```

### Quickstart

**Install:**
```bash
pip install git+https://github.com/AustScar5/conductor-middleware.git
```

**Configure:**
```bash
export XRPL_ISSUER_SEED="sYourIssuerSeed"
export XRPL_PFT_ISSUER="rYourIssuerAddress"
export CONDUCTOR_ALLOWLIST="allowlist.json"
```

**Start (stdio — for AI agent integration):**
```bash
python -m conductor_middleware.mcp_server
# or:
conductor-mcp
```

**Start (HTTP-SSE — for HTTP clients):**
```bash
conductor-mcp --transport sse --port 8000 --host 127.0.0.1
```

---

### Tool reference

#### `run_pipeline`

Full verification → adjudication → authorization → payout flow.

**Input:**
```json
{
  "artifact": {"summary": "Analysis complete.", "confidence": 0.95},
  "schema_fields": {"summary": "str", "confidence": "float"},
  "contributor_wallet": "rContribXRPAddress...",
  "task_id": "task-abc123",
  "dry_run": true
}
```

**Response (COMPLETED):**
```json
{
  "disposition": "completed",
  "contributor": "rContribXRPAddress...",
  "artifact_hash": "3cbddd698cd2a4...",
  "tx_blob": "1200002400000001...",
  "tx_hash": null,
  "dry_run": true,
  "stages": [
    {"stage": "verification",   "passed": true,  "reason": "Schema validation passed."},
    {"stage": "adjudication",   "passed": true,  "reason": "Artifact passed all checks."},
    {"stage": "authorization",  "passed": true,  "reason": "Contributor is on the allowlist."},
    {"stage": "payout",         "passed": true,  "reason": "Payout transaction constructed."}
  ]
}
```

**Response (BLOCKED — unauthorized wallet):**
```json
{
  "disposition": "blocked",
  "tx_blob": null,
  "stages": [
    {"stage": "verification",  "passed": true},
    {"stage": "adjudication",  "passed": true},
    {"stage": "authorization", "passed": false,
     "reason": "Wallet address is not on the contributor allowlist.",
     "detail": {"wallet_status": "unauthorized"}}
  ]
}
```

---

#### `check_authorization`

**Input:**
```json
{"wallet_address": "rContribXRPAddress..."}
```

**Response:**
```json
{
  "wallet_address": "rcontribxrpaddress...",
  "status": "authorized",
  "reason": "Contributor is on the allowlist and authorization is current.",
  "is_authorized": true,
  "timestamp": "2026-03-19T22:56:30Z"
}
```

---

#### `dry_run_payout`

**Input:**
```json
{
  "contributor_wallet": "rContribXRPAddress...",
  "evidence_hash": "3cbddd698cd2a47c6731b264fbe138f2..."
}
```

**Response:**
```json
{
  "tx_blob": "1200002400000001201B05F5E0FF6140...",
  "contributor": "rContribXRPAddress...",
  "pft_amount": "100",
  "evidence_hash": "3cbddd698cd2a47c...",
  "dry_run": true
}
```

---

## Architecture

```
  Raw artifact + contributor wallet
            │
            ▼
  ┌─────────────────────┐
  │  VerificationOracle  │  Validates artifact against caller-supplied
  │  (Stage 1)           │  Pydantic schema + extra domain validators
  └─────────┬───────────┘
            │ pass
            ▼
  ┌─────────────────────┐
  │  AdjudicationOracle │  Computes SHA-256 evidence hash, applies
  │  (Stage 2)           │  ReviewerPolicy, emits structured verdict
  └─────────┬───────────┘
            │ pass
            ▼
  ┌─────────────────────┐
  │  AuthorizationGate  │  Checks contributor wallet against JSON/TOML
  │  (Stage 3)           │  allowlist; rejects unauthorized/suspended
  └─────────┬───────────┘
            │ authorized
            ▼
  ┌─────────────────────┐
  │   PayoutTrigger     │  Idempotency guard → builds signed XRPL PFT
  │  (Stage 4)           │  Payment tx; dry_run=True skips submission
  └─────────┬───────────┘
            │
            ▼
      PipelineResult
   disposition + audit trail
```

Each stage short-circuits on failure. `PipelineResult.disposition` is one of:
`completed` · `rejected` · `retry` · `blocked` · `duplicate` · `error`

---

## Installation

```bash
pip install git+https://github.com/AustScar5/conductor-middleware.git
```

Requires Python 3.11+.

---

## Quickstart

```python
from conductor_middleware import (
    TaskNodeConductor,
    AuthorizationGate,
    AllowlistConfig,
    WalletEntry,
    WalletStatus,
    PayoutConfig,
)
from pydantic import BaseModel

# 1. Define your task completion artifact schema
class TaskArtifact(BaseModel):
    summary:    str
    confidence: float
    sources:    list[str]

# 2. Configure the gate (JSON/TOML file or inline config)
gate = AuthorizationGate(config=AllowlistConfig(allowlist=[
    WalletEntry(
        wallet="rContributorXRPAddress...",
        status=WalletStatus.AUTHORIZED,
    ),
]))

# 3. Configure XRPL payout (or set XRPL_ISSUER_SEED / XRPL_PFT_ISSUER env vars)
payout_config = PayoutConfig(
    issuer_seed="sYourIssuerSeed",
    pft_issuer="rYourPFTIssuerAddress",
)

# 4. Build the conductor
conductor = TaskNodeConductor(gate=gate, payout_config=payout_config)

# 5. Run the pipeline
result = conductor.run_pipeline(
    artifact={
        "summary":    "Analysis complete.",
        "confidence": 0.95,
        "sources":    ["doc-1", "doc-2"],
    },
    schema=TaskArtifact,
    contributor_wallet="rContributorXRPAddress...",
    dry_run=True,   # set False to submit on-chain
)

print(result.disposition)    # Disposition.COMPLETED
print(result.tx_blob)        # hex-encoded signed XRPL Payment tx
print(result.artifact_hash)  # SHA-256 evidence hash

# Inspect the per-stage audit trail
for stage in result.stages:
    print(f"  {stage.stage}: passed={stage.passed}  {stage.reason[:60]}")
```

---

## Conductor Integration

```python
from conductor_middleware import (
    AuthorizationGate, AdjudicationLog, IdempotencyStore,
    PayoutConfig, PayoutTrigger, ReviewerPolicy,
    TaskNodeConductor, Disposition,
)

# Optional: persist idempotency guard and adjudication log across restarts
store = IdempotencyStore(db_path="payout_guard.db")
log   = AdjudicationLog(path="logs/adjudication.json")

# Optional: domain-level reviewer rule
def require_sources(obj) -> str | None:
    if not obj.sources:
        return "sources must not be empty"

conductor = TaskNodeConductor(
    gate=AuthorizationGate("allowlist.json"),
    payout_config=PayoutConfig(),        # reads XRPL_* env vars
    reviewer_policy=ReviewerPolicy("require_sources", [require_sources]),
    idempotency_store=store,
    adjudication_log=log,
)

result = conductor.run_pipeline(
    artifact=task_output,
    schema=MyArtifactSchema,
    contributor_wallet=contributor_wallet,
    task_id="task-abc123",
    dry_run=False,
)

match result.disposition:
    case Disposition.COMPLETED:
        print("Payout submitted:", result.tx_hash)
    case Disposition.RETRY:
        redispatch_task(task_id)
    case Disposition.BLOCKED:
        log_auth_violation(contributor_wallet)
    case Disposition.DUPLICATE:
        pass   # already paid — safe to ignore
    case Disposition.REJECTED | Disposition.ERROR:
        escalate(result)
```

### Allowlist config (JSON)

```json
{
  "allowlist": [
    {"wallet": "rContrib...", "status": "authorized"},
    {"wallet": "rBadActor...", "status": "suspended",
     "suspension_reason": "repeated failures"}
  ]
}
```

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `XRPL_ISSUER_SEED` | base58 family seed of the PFT issuer wallet | — (required) |
| `XRPL_PFT_ISSUER` | classic address of the PFT token issuer | — (required) |
| `XRPL_PFT_CURRENCY` | currency code | `PFT` |
| `XRPL_PFT_AMOUNT` | token units per passing verdict | `100` |
| `XRPL_NODE_URL` | XRPL websocket node | `wss://xrplcluster.com` |
| `XRPL_BASE_FEE_DROPS` | network fee in drops | `12` |

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest -v
```

226 tests across 7 modules. All offline — no XRPL network calls.

---

## Package layout

```
src/conductor_middleware/
    __init__.py        — public API exports
    models.py          — RetryPolicy, TaskDispatch, WorkerResponse, …
    oracle.py          — VerificationOracle
    adjudication.py    — AdjudicationOracle, AdjudicationVerdict, ReviewerPolicy
    authorization.py   — AuthorizationGate, AllowlistConfig, WalletEntry
    payout.py          — PayoutTrigger, IdempotencyStore, PayoutConfig
    pipeline.py        — TaskNodeConductor, PipelineResult, Disposition
    retry_engine.py    — RetryEngine (lower-level retry loop)
tests/
    test_middleware.py
    test_adjudication.py
    test_authorization.py
    test_payout.py
    test_pipeline.py
logs/                  — structured JSON run logs written by tests
allowlist.json         — sample allowlist config (JSON)
allowlist.toml         — sample allowlist config (TOML)
pyproject.toml
```

---

## Task Node Integration

Wire `AuthorizationGate` and `AdjudicationOracle` together in any conductor
reward-dispatch pipeline with about 10 lines:

```python
from conductor_middleware import AdjudicationOracle, AdjudicationLog, ReviewerPolicy
from conductor_middleware import AuthorizationGate
from conductor_middleware import gated_adjudicate
from pydantic import BaseModel

class DeliveryArtifact(BaseModel):
    summary:    str
    word_count: int
    sources:    list[str]

def require_sources(obj: DeliveryArtifact) -> str | None:
    if not obj.sources:
        return "sources list must not be empty"

log    = AdjudicationLog(path="logs/adjudication.json")
oracle = AdjudicationOracle(log=log)
gate   = AuthorizationGate("allowlist.json")
policy = ReviewerPolicy("require_sources", [require_sources])

verdict = gated_adjudicate(gate, oracle, contributor_wallet,
                            worker_output, DeliveryArtifact,
                            reviewer_policy=policy,
                            task_id="task-abc123")

match verdict.verdict:
    case "pass":
        dispatch_reward(contributor_wallet, worker_output)
    case "retry":
        redispatch(task_id, hint=verdict.reason)
    case "fail":
        if "unauthorized_contributor" in verdict.reason:
            log_auth_violation(contributor_wallet, verdict.reason)
        else:
            escalate(verdict.field_errors)

assert oracle.verify_integrity(worker_output, verdict.evidence_hash)
```

---

## License

MIT
