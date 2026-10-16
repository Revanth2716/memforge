"""MCP self-test for MemForge (local, offline).

Drives the real MCP server in-process and walks the full tool lifecycle:

    tools/list
        -> memory.store
        -> memory.search
        -> memory.recall
        -> memory.compile_context
        -> memory.explain
        -> memory.update
        -> memory.forget
        -> validation and error paths
        -> trace correlation (M6)

**Zero network**: the server is built over the local SQLite repository and the
memory tools never reach a language-model provider. Run it with::

    python scripts/mcp_selftest.py

It exits non-zero when a check fails.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

__all__ = ["main", "run_checks"]


class CheckFailed(AssertionError):
    """A mandatory MCP check did not hold."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`CheckFailed` with a readable message."""
    if not condition:
        raise CheckFailed(message)


async def build_stack(directory: Path) -> tuple[MCPServer, Any, Any]:
    """Build a traced MemForge MCP server over a throwaway database."""
    from app.domain.ids import SequentialIdGenerator
    from app.domain.time import ManualClock
    from app.mcp.server import build_mcp_server
    from app.observability.instrumentation import build_traced_operations
    from app.observability.service import ObservabilityService, SequentialIdFactory
    from app.services.context_service import ContextCompilerService
    from app.services.memory_service import MemoryService
    from app.services.retrieval_service import MemoryRetrievalService
    from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
    from app.storage.sqlite.repository import SqliteMemoryRepository
    from app.storage.sqlite.trace_store import SqliteTraceStore

    db_path = directory / "mcp-selftest.sqlite"
    clock = ManualClock()
    repository = SqliteMemoryRepository(db_path)
    repository.initialize()
    memories = MemoryService(
        repository, clock=clock, id_generator=SequentialIdGenerator(prefix="mcp")
    )
    retrieval = MemoryRetrievalService(
        repository, clock=clock, embedding_store=SqliteEmbeddingStore(db_path)
    )
    compiler = ContextCompilerService(retrieval)
    observability = ObservabilityService(
        SqliteTraceStore(db_path),
        clock=clock,
        id_factory=SequentialIdFactory(prefix="mcp-trace"),
    )
    operations = build_traced_operations(memories, retrieval, compiler, observability)
    return build_mcp_server(operations, observability=observability), operations, observability


async def check_tools_list(server: MCPServer) -> None:
    """All seven tools are advertised, each with input and output schemas."""
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    expected = {
        "memory.store",
        "memory.search",
        "memory.recall",
        "memory.update",
        "memory.forget",
        "memory.compile_context",
        "memory.explain",
    }
    require(names == expected, f"unexpected tool list: {sorted(names)}")
    for tool in tools:
        require(bool(tool.description), f"{tool.name} has no description")
        require(tool.input_schema.get("type") == "object", f"{tool.name} has no object input")
        require(bool(tool.output_schema), f"{tool.name} publishes no output schema")
    print(f"    {len(tools)} tools, all with input and output schemas")


async def check_store(server: MCPServer) -> str:
    """``memory.store`` writes a memory and returns its provenance."""
    result = await server.call_tool(
        "memory.store",
        {
            "content": "User prefers Python for backend work.",
            "memory_type": "PROFILE",
            "source": "user:selftest",
            "confidence": 0.9,
            "importance": 0.8,
        },
    )
    require(not result.is_error, f"store failed: {result.content}")
    payload = result.structured_content
    require(payload["memory_type"] == "PROFILE", "wrong memory type")
    require(payload["confidence"] == 0.9, "wrong confidence")
    print(f"    stored {payload['id']}")
    return str(payload["id"])


async def check_store_duplicate(server: MCPServer) -> None:
    """A near-duplicate is detected through the same M2 path."""
    await server.call_tool(
        "memory.store",
        {
            "content": "The user prefers python for backend development work",
            "memory_type": "OBSERVATION",
        },
    )
    result = await server.call_tool("memory.search", {"query": "python backend"})
    require(not result.is_error, "search failed")
    require(result.structured_content["candidates_considered"] >= 2, "search considered too little")


async def check_search(server: MCPServer) -> None:
    """``memory.search`` returns ranked, explained results."""
    result = await server.call_tool("memory.search", {"query": "python backend", "top_k": 5})
    require(not result.is_error, "search failed")
    payload = result.structured_content
    require(payload["results"], "search returned nothing")
    require(payload["retrieval_mode"] == "lexical", "default mode must stay lexical")
    for item in payload["results"]:
        require(bool(item["explanation"]), "a result carries no explanation")
    print(f"    search returned {len(payload['results'])} explained result(s)")


async def check_recall(server: MCPServer) -> None:
    """``memory.recall`` gates out zero-relevance memories."""
    result = await server.call_tool("memory.recall", {"query": "python backend", "top_k": 5})
    require(not result.is_error, "recall failed")
    payload = result.structured_content
    require(payload["recalled"] >= 1, "recall found nothing")
    require(
        all(item["relevance"] > 0.0 for item in payload["results"]),
        "recall returned a zero-relevance memory",
    )
    empty = await server.call_tool("memory.recall", {"query": "kubernetes ingress"})
    require(
        empty.structured_content["recalled"] == 0,
        "an unrelated query recalled something",
    )
    print(f"    recalled {payload['recalled']}; unrelated query recalled 0")


async def check_compile_context(server: MCPServer) -> None:
    """``memory.compile_context`` honours the budget and keeps provenance."""
    result = await server.call_tool(
        "memory.compile_context",
        {"query": "python backend", "token_budget": 120, "top_k": 5},
    )
    require(not result.is_error, "compile failed")
    payload = result.structured_content
    require(payload["within_budget"] is True, "the compiled context exceeded its budget")
    require(payload["estimated_tokens"] <= payload["token_budget"], "token budget violated")
    require(payload["blocks"], "no block was selected")
    for block in payload["blocks"]:
        require(bool(block["content"]), "a block carries no content")
        require(block["estimated_tokens"] > 0, "a block reports no token estimate")
    print(
        f"    compiled {payload['selected_count']} block(s), "
        f"{payload['estimated_tokens']}/{payload['token_budget']} estimated tokens"
    )


async def check_explain(server: MCPServer, memory_id: str) -> None:
    """``memory.explain`` returns the M3 breakdown for one memory."""
    result = await server.call_tool("memory.explain", {"memory_id": memory_id, "query": "python"})
    require(not result.is_error, "explain failed")
    payload = result.structured_content
    require(payload["final_score"] > 0.0, "explain produced no score")
    require(bool(payload["explanation"]), "explain produced no explanation")
    print(f"    explained {memory_id} -> {payload['final_score']:.4f}")


async def check_update_and_forget(server: MCPServer, memory_id: str) -> None:
    """``memory.update`` patches; ``memory.forget`` soft-deletes."""
    updated = await server.call_tool("memory.update", {"memory_id": memory_id, "importance": 0.95})
    require(not updated.is_error, "update failed")
    require(updated.structured_content["importance"] == 0.95, "update did not apply")
    require(updated.structured_content["version"] == 2, "update did not bump the version")

    forgotten = await server.call_tool("memory.forget", {"memory_id": memory_id})
    require(not forgotten.is_error, "forget failed")
    require(forgotten.structured_content["status"] == "DELETED", "forget did not soft-delete")
    print("    updated (v2) and forgotten (soft delete)")


async def check_validation(server: MCPServer) -> None:
    """Invalid inputs are refused, never crash and never leak a secret."""
    invalid_type = await server.call_tool(
        "memory.store", {"content": "x", "memory_type": "NOT_A_TYPE"}
    )
    require(invalid_type.is_error, "an unknown memory_type was accepted")
    require(
        invalid_type.structured_content["error"]["code"] == "VALIDATION_ERROR",
        "an unknown memory_type did not produce a validation error",
    )

    missing = await server.call_tool(
        "memory.forget", {"memory_id": "00000000-0000-5000-8000-000000000000"}
    )
    require(missing.is_error, "an unknown memory id was accepted")
    require(
        missing.structured_content["error"]["code"] == "MEMORY_NOT_FOUND",
        "an unknown memory id did not produce MEMORY_NOT_FOUND",
    )

    # Schema violations (a zero budget, an empty query) are refused by the SDK's
    # own validation before the tool body runs; either a raised ToolError or a
    # structured error result is an acceptable refusal.
    for invalid in (
        ("memory.compile_context", {"query": "x", "token_budget": 0}),
        ("memory.search", {"query": ""}),
    ):
        tool, arguments = invalid
        try:
            outcome = await server.call_tool(tool, arguments)
        except ToolError:
            continue
        require(outcome.is_error, f"{tool} accepted {arguments}")

    # No tool response may carry anything that looks like a credential.
    for tool, arguments in (
        ("memory.search", {"query": "python"}),
        ("memory.recall", {"query": "python"}),
        ("memory.compile_context", {"query": "python", "token_budget": 120}),
    ):
        result = await server.call_tool(tool, arguments)
        rendered = str(result.structured_content).lower()
        for secret_marker in ("authorization", "bearer ", "api_key", "api-key"):
            require(secret_marker not in rendered, f"{tool} leaked {secret_marker!r}")
    print("    invalid inputs refused; no secret-shaped text in any response")


async def check_trace_correlation(server: MCPServer, observability: Any) -> None:
    """Every MCP call is traceable: a trace, its events, and no secrets in them."""
    before = len(observability.list_traces(limit=200))
    await server.call_tool(
        "memory.store",
        {
            "content": "The user keeps a standing desk for deep work sessions.",
            "memory_type": "FACT",
            "source": "user:selftest",
        },
    )
    traces = observability.list_traces(limit=200)
    require(len(traces) == before + 1, "an MCP call produced no trace")

    trace = traces[0]
    require(trace.kind.value == "mcp_request", f"unexpected trace kind {trace.kind.value}")
    require(trace.operation == "mcp.memory.store", f"unexpected operation {trace.operation}")

    events = observability.get_events(trace.trace_id)
    kinds = [event.kind.value for event in events]
    require(kinds[0] == "mcp_request", f"a trace did not open with mcp_request: {kinds}")
    require(kinds[-1] == "mcp_response", f"a trace did not close with mcp_response: {kinds}")
    require("memory_write" in kinds, f"the store was not recorded inside the trace: {kinds}")

    sequences = [event.sequence for event in events]
    require(sequences == list(range(len(sequences))), f"event order is not total: {sequences}")

    summary = observability.summarize_trace(trace.trace_id)
    require(summary is not None, "the trace produced no summary")
    require(summary.succeeded, "a successful MCP call was summarised as failed")
    require(bool(summary.memory_ids), "the stored memory id was not recorded")

    # The same invariant the REST smoke test checks, enforced on the MCP path.
    rendered = str([event.to_dict() for event in events]).lower()
    for marker in ("authorization", "bearer ", "api_key", "api-key", "password"):
        require(marker not in rendered, f"a trace payload leaked {marker!r}")
    print(
        f"    {len(traces)} traces; last={trace.trace_id} "
        f"with {len(events)} events, {summary.event_count} summarised"
    )


async def check_trace_error(server: MCPServer, observability: Any) -> None:
    """A failed tool call still produces a trace, and it records the failure."""
    before = len(observability.list_traces(limit=200))
    result = await server.call_tool(
        "memory.forget", {"memory_id": "00000000-0000-5000-8000-000000000000"}
    )
    require(result.is_error, "an unknown memory id was accepted")
    traces = observability.list_traces(limit=200)
    require(len(traces) == before + 1, "a failed MCP call produced no trace")
    summary = observability.summarize_trace(traces[0].trace_id)
    require(summary is not None, "the failed call produced no summary")
    require(bool(summary.errors), "a failed tool call was summarised as clean")
    print(f"    failed call recorded as {traces[0].trace_id} with {summary.errors}")


async def run_checks() -> int:
    """Run every MCP check; return a process exit code."""
    logging.disable(logging.WARNING)
    failures: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="memforge-mcp-") as directory:
        server, _, observability = await build_stack(Path(directory))
        memory_id = ""
        checks: tuple[tuple[str, Any], ...] = (
            ("tools/list", check_tools_list),
            ("memory.store", check_store),
            ("memory.store (duplicate)", check_store_duplicate),
            ("memory.search", check_search),
            ("memory.recall", check_recall),
            ("memory.compile_context", check_compile_context),
            ("memory.explain", lambda s: check_explain(s, memory_id)),
            ("memory.update + memory.forget", lambda s: check_update_and_forget(s, memory_id)),
            ("validation + error paths", check_validation),
            ("trace correlation", lambda s: check_trace_correlation(s, observability)),
            ("trace error path", lambda s: check_trace_error(s, observability)),
        )
        print("MemForge MCP self-test (local, offline)")
        print("=" * 60)
        for index, (title, check) in enumerate(checks, start=1):
            try:
                produced = await check(server)
            except CheckFailed as failure:
                failures.append((title, str(failure)))
                print(f"[{index}/{len(checks)}] FAIL  {title}")
                print(f"          {failure}")
                continue
            if isinstance(produced, str) and not memory_id:
                memory_id = produced
            print(f"[{index}/{len(checks)}] PASS  {title}")

    print("=" * 60)
    passed = len(checks) - len(failures)
    print(f"{passed}/{len(checks)} mandatory MCP checks passed")
    if failures:
        print("\nFailures:")
        for title, message in failures:
            print(f"  - {title}: {message}")
        return 1
    print("MCP self-test finished successfully.")
    return 0


def main() -> int:
    """Entry point."""
    return asyncio.run(run_checks())


if __name__ == "__main__":
    raise SystemExit(main())
