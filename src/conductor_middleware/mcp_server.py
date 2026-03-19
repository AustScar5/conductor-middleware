"""
conductor_middleware.mcp_server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP server that exposes the Task Node conductor pipeline as three callable
tools over stdio or HTTP-SSE transport.

Tools
-----
run_pipeline
    Full verification → adjudication → authorization → payout flow.
    Returns a serialised PipelineResult with stage-by-stage audit trail
    and Disposition.

check_authorization
    Single gate_check call against the configured allowlist.
    Returns a serialised AuthorizationVerdict (status, reason, timestamp).

dry_run_payout
    Constructs and signs an XRPL PFT Payment transaction for an already-
    adjudicated artifact without submitting it to the network.
    Returns the hex-encoded transaction blob.

Transport
---------
stdio (default):
    python -m conductor_middleware.mcp_server

HTTP-SSE:
    python -m conductor_middleware.mcp_server --transport sse --port 8000

Configuration
-------------
All XRPL credentials are read from environment variables (see PayoutConfig).
The allowlist path is set via CONDUCTOR_ALLOWLIST (default: allowlist.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .adjudication import AdjudicationOracle
from .authorization import (
    AllowlistConfig,
    AuthorizationGate,
    WalletEntry,
    WalletStatus,
)
from .payout import IdempotencyStore, PayoutConfig, PayoutRequest, PayoutTrigger
from .pipeline import Disposition, TaskNodeConductor

# ── Tool name constants ────────────────────────────────────────────────────────

TOOL_RUN_PIPELINE        = "run_pipeline"
TOOL_CHECK_AUTHORIZATION = "check_authorization"
TOOL_DRY_RUN_PAYOUT      = "dry_run_payout"


# ── Tool input schemas ─────────────────────────────────────────────────────────

_RUN_PIPELINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "artifact": {
            "type": "object",
            "description": "Task completion artifact dict from the worker agent.",
        },
        "schema_fields": {
            "type": "object",
            "description": (
                "Inline schema definition as {field_name: type_string} mapping. "
                "Supported types: str, float, int, bool, list. "
                "Example: {\"summary\": \"str\", \"confidence\": \"float\"}. "
                "Alternatively omit to use the default TaskArtifact schema."
            ),
        },
        "contributor_wallet": {
            "type": "string",
            "description": "XRP classic address (r...) of the contributor.",
        },
        "task_id": {
            "type": "string",
            "description": "Optional task identifier for audit logging.",
        },
        "dry_run": {
            "type": "boolean",
            "description": "If true, construct but do not submit the payout tx.",
            "default": True,
        },
    },
    "required": ["artifact", "contributor_wallet"],
}

_CHECK_AUTHORIZATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "wallet_address": {
            "type": "string",
            "description": "Wallet address to check (XRP classic r... or Ethereum 0x...).",
        },
    },
    "required": ["wallet_address"],
}

_DRY_RUN_PAYOUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contributor_wallet": {
            "type": "string",
            "description": "XRP classic address (r...) to receive the PFT payment.",
        },
        "evidence_hash": {
            "type": "string",
            "description": "64-char SHA-256 hex string from a prior adjudication.",
        },
        "task_id": {
            "type": "string",
            "description": "Optional task identifier.",
        },
    },
    "required": ["contributor_wallet", "evidence_hash"],
}


# ── Schema builder ─────────────────────────────────────────────────────────────

def _build_schema_from_fields(schema_fields: dict[str, str]):
    """
    Build a temporary Pydantic model from a {field: type_string} dict.
    Used when callers pass inline schema definitions rather than a class.
    """
    from pydantic import BaseModel, create_model

    type_map = {"str": str, "float": float, "int": int, "bool": bool, "list": list}
    field_defs = {
        name: (type_map.get(type_str, str), ...)
        for name, type_str in schema_fields.items()
    }
    return create_model("DynamicArtifact", **field_defs)


def _default_schema():
    """Default artifact schema used when schema_fields is omitted."""
    from pydantic import BaseModel

    class DefaultArtifact(BaseModel):
        summary:    str
        confidence: float

    return DefaultArtifact


# ── Server factory ─────────────────────────────────────────────────────────────

def build_server(
    allowlist_path: str | None = None,
    payout_config:  PayoutConfig | None = None,
) -> Server:
    """
    Build and return a configured MCP Server instance.

    Parameters
    ----------
    allowlist_path:
        Path to JSON or TOML allowlist file.
        Falls back to CONDUCTOR_ALLOWLIST env var, then 'allowlist.json'.
    payout_config:
        XRPL credentials. Falls back to XRPL_* env vars.
    """
    resolved_path = (
        allowlist_path
        or os.environ.get("CONDUCTOR_ALLOWLIST")
        or "allowlist.json"
    )

    # Load gate — tolerate missing allowlist gracefully (empty = block all)
    try:
        gate = AuthorizationGate(resolved_path)
    except FileNotFoundError:
        gate = AuthorizationGate(config=AllowlistConfig(allowlist=[]))

    payout_cfg = payout_config or PayoutConfig()
    store      = IdempotencyStore()
    conductor  = TaskNodeConductor(
        gate=gate,
        payout_config=payout_cfg,
        idempotency_store=store,
    )

    # Separate trigger for dry_run_payout tool (shares same idempotency store)
    payout_trigger = PayoutTrigger(
        config=payout_cfg,
        store=store,
        sequence_override=1,
        last_ledger_override=99_999_999,
    )

    server = Server("task-node-conductor")

    # ── list_tools ─────────────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=TOOL_RUN_PIPELINE,
                description=(
                    "Run a task completion artifact through the full conductor "
                    "pipeline: VerificationOracle → AdjudicationOracle → "
                    "AuthorizationGate → PayoutTrigger. Returns a PipelineResult "
                    "with disposition (completed/rejected/retry/blocked/duplicate) "
                    "and a per-stage audit trail."
                ),
                inputSchema=_RUN_PIPELINE_SCHEMA,
            ),
            Tool(
                name=TOOL_CHECK_AUTHORIZATION,
                description=(
                    "Check whether a wallet address is authorized to participate "
                    "as a Task Node contributor. Returns an AuthorizationVerdict "
                    "with status (authorized/unauthorized/suspended/expired) and "
                    "a reason string."
                ),
                inputSchema=_CHECK_AUTHORIZATION_SCHEMA,
            ),
            Tool(
                name=TOOL_DRY_RUN_PAYOUT,
                description=(
                    "Construct and sign an XRPL PFT Payment transaction for an "
                    "authorized contributor without submitting it to the network. "
                    "Returns the hex-encoded signed transaction blob. Requires "
                    "XRPL_ISSUER_SEED and XRPL_PFT_ISSUER to be set."
                ),
                inputSchema=_DRY_RUN_PAYOUT_SCHEMA,
            ),
        ]

    # ── call_tool ──────────────────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            if name == TOOL_RUN_PIPELINE:
                result = _handle_run_pipeline(conductor, arguments)
            elif name == TOOL_CHECK_AUTHORIZATION:
                result = _handle_check_authorization(gate, arguments)
            elif name == TOOL_DRY_RUN_PAYOUT:
                result = _handle_dry_run_payout(payout_trigger, gate, arguments)
            else:
                result = {"error": f"Unknown tool: {name!r}"}
        except Exception as exc:
            # Return structured error rather than letting the exception propagate
            result = {
                "error":      type(exc).__name__,
                "message":    str(exc),
                "tool":       name,
            }

        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


# ── Tool handlers ──────────────────────────────────────────────────────────────

def _handle_run_pipeline(
    conductor: TaskNodeConductor,
    args: dict[str, Any],
) -> dict[str, Any]:
    artifact   = args.get("artifact", {})
    wallet     = args.get("contributor_wallet", "")
    task_id    = args.get("task_id")
    dry_run    = args.get("dry_run", True)
    sf         = args.get("schema_fields")

    schema = _build_schema_from_fields(sf) if sf else _default_schema()

    result = conductor.run_pipeline(
        artifact=artifact,
        schema=schema,
        contributor_wallet=wallet,
        dry_run=dry_run,
        task_id=task_id,
    )
    return result.model_dump(mode="json")


def _handle_check_authorization(
    gate: AuthorizationGate,
    args: dict[str, Any],
) -> dict[str, Any]:
    wallet  = args.get("wallet_address", "")
    verdict = gate.gate_check(wallet)
    result  = verdict.model_dump(mode="json")
    # is_authorized is a @property, not a Pydantic field — add explicitly
    result["is_authorized"] = verdict.is_authorized
    return result


def _handle_dry_run_payout(
    trigger:  PayoutTrigger,
    gate:     AuthorizationGate,
    args:     dict[str, Any],
) -> dict[str, Any]:
    from .adjudication import AdjudicationVerdict, VerdictCode
    from .payout import DuplicateVerdictError, PayoutBlockedError

    wallet        = args.get("contributor_wallet", "")
    evidence_hash = args.get("evidence_hash", "")
    task_id       = args.get("task_id")

    auth_verdict = gate.gate_check(wallet)
    if not auth_verdict.is_authorized:
        return {
            "error":  "unauthorized_contributor",
            "status": auth_verdict.status.value,
            "reason": auth_verdict.reason,
            "wallet": auth_verdict.wallet_address,
        }

    adj_verdict = AdjudicationVerdict(
        verdict=VerdictCode.PASS,
        reason="Pre-adjudicated artifact supplied directly to dry_run_payout.",
        evidence_hash=evidence_hash,
        retry_eligible=False,
        schema_name="external",
    )

    request = PayoutRequest(
        adjudication_verdict=adj_verdict,
        authorization_verdict=auth_verdict,
        contributor_wallet=wallet,
        task_id=task_id,
    )

    try:
        payout_result = trigger.execute(request, dry_run=True)
    except DuplicateVerdictError as exc:
        return {"error": "duplicate_verdict", "evidence_hash": exc.evidence_hash}
    except PayoutBlockedError as exc:
        return {"error": "payout_blocked", "reason": exc.reason, "details": exc.details}
    except ValueError as exc:
        return {"error": "configuration_error", "message": str(exc)}

    return {
        "tx_blob":       payout_result.tx_blob,
        "contributor":   payout_result.contributor,
        "pft_amount":    payout_result.pft_amount,
        "evidence_hash": payout_result.evidence_hash,
        "dry_run":       True,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task Node Conductor MCP Server")
    p.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                   help="Transport protocol (default: stdio)")
    p.add_argument("--port", type=int, default=8000,
                   help="HTTP port for SSE transport (default: 8000)")
    p.add_argument("--host", default="127.0.0.1",
                   help="HTTP host for SSE transport (default: 127.0.0.1)")
    p.add_argument("--allowlist", default=None,
                   help="Path to allowlist JSON/TOML file")
    return p.parse_args()


async def _run_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


async def _run_sse(server: Server, host: str, port: int) -> None:
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read, write):
            await server.run(read, write, server.create_initialization_options())

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ])
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main() -> None:
    args   = _parse_args()
    server = build_server(allowlist_path=args.allowlist)

    if args.transport == "stdio":
        asyncio.run(_run_stdio(server))
    else:
        asyncio.run(_run_sse(server, args.host, args.port))


if __name__ == "__main__":
    main()
