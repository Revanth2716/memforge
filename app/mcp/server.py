"""The MemForge MCP server (M5).

Built on the official Python SDK (``mcp`` >= 2.0). Each tool is a small
adapter over :class:`~app.services.memory_operations.MemoryOperationsService`:

* **Input** -- flat, typed, annotated parameters, so the SDK derives a strict
  JSON schema (required fields, bounds, enums) that a client can validate before
  calling;
* **Output** -- a pydantic model, so the SDK publishes an output schema *and*
  validates the value the tool returns;
* **Errors** -- a MemForge exception is converted into the REST error envelope
  and returned with ``isError: true``. Nothing raises out of a tool for an
  expected failure, and an unexpected crash is reported by the SDK without the
  message reaching the client.

Deliberately absent: any provider/model call. An MCP client already *is* the
model; these tools expose the memory layer, not a second inference path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeVar, cast

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from app.domain.errors import (
    AgentValidationError,
    ContextValidationError,
    LLMError,
    MemforgeError,
    MemoryNotFoundError,
    MemoryValidationError,
    NoConflictError,
    RepositoryError,
    RetrievalValidationError,
)
from app.domain.models import MemoryStatus, MemoryType
from app.domain.retrieval import RetrievalMode
from app.observability.instrumentation import McpTracer
from app.observability.service import ObservabilityService
from app.services.memory_operations import MemoryOperationsService, RecallPolicy

#: Server name advertised over MCP.
SERVER_NAME = "memforge"

MEMFORGE_MCP_INSTRUCTIONS = """
MemForge exposes its memory layer to an agent over MCP.

Every tool delegates to the same services the REST API uses, so the memory
semantics are identical: importance scoring, duplicate suppression, conflict
resolution and token-budgeted context compilation.

Typical flow:
  1. memory.recall          -- what do we know that is relevant?
  2. memory.compile_context -- build a budgeted context with provenance
  3. (the client generates the answer itself)
  4. memory.store           -- persist anything durable the user stated

Token figures are MemForge estimates, not provider tokenisations.
""".strip()

#: Error code per domain error family, so clients can branch without parsing text.
_ERROR_CODES: tuple[tuple[type[MemforgeError], str], ...] = (
    (MemoryNotFoundError, "MEMORY_NOT_FOUND"),
    (MemoryValidationError, "MEMORY_VALIDATION_ERROR"),
    (NoConflictError, "NO_CONFLICT"),
    (RetrievalValidationError, "RETRIEVAL_VALIDATION_ERROR"),
    (ContextValidationError, "CONTEXT_VALIDATION_ERROR"),
    (AgentValidationError, "AGENT_VALIDATION_ERROR"),
    (RepositoryError, "REPOSITORY_ERROR"),
    (LLMError, "LLM_ERROR"),
)


def _error_result(error: Exception) -> CallToolResult:
    """Turn an expected failure into a structured tool error."""
    code = "MEMFORGE_ERROR"
    for error_type, name in _ERROR_CODES:
        if isinstance(error, error_type):
            code = name
            break
    payload = {"error": {"code": code, "message": str(error), "details": {}}}
    return CallToolResult(
        content=[TextContent(type="text", text=_dumps(payload))],
        structured_content=payload,
        is_error=True,
    )


def _dumps(payload: Any) -> str:
    """Deterministic JSON text for the unstructured content block."""
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _fail(message: str, code: str = "VALIDATION_ERROR") -> CallToolResult:
    """Return a tool-level validation error (before any service call)."""
    payload = {"error": {"code": code, "message": message, "details": {}}}
    return CallToolResult(
        content=[TextContent(type="text", text=_dumps(payload))],
        structured_content=payload,
        is_error=True,
    )


_ResultT = TypeVar("_ResultT")


def _tool_error(error: Exception) -> _ResultT:  # type: ignore[type-var]
    """Return a structured tool error from a tool whose success type is a model.

    The MCP SDK checks for a ``CallToolResult`` *before* it validates a result
    against the tool's published output schema, so this is the supported way for a
    tool to report an expected failure without raising. The SDK refuses a union
    that mentions ``CallToolResult``, so the one documented cast lives here
    instead of in every tool body.
    """
    return cast("_ResultT", _error_result(error))


def _tool_failure(message: str, code: str = "VALIDATION_ERROR") -> _ResultT:  # type: ignore[type-var]
    """Return a tool-level validation error raised before any service call."""
    return cast("_ResultT", _fail(message, code))


# -- output models ------------------------------------------------------------


class ToolModel(BaseModel):
    """Base model for every tool result: strict, no unknown fields."""

    model_config = ConfigDict(extra="forbid")


class StoredMemory(ToolModel):
    """A memory that was stored."""

    id: str
    content: str
    memory_type: str
    source: str | None = None
    confidence: float
    importance: float
    version: int


class SearchedMemory(ToolModel):
    """One ranked result."""

    memory_id: str
    content: str
    memory_type: str
    final_score: float
    relevance: float
    explanation: str


class SearchResult(ToolModel):
    """The result of ``memory.search``."""

    query: str
    retrieval_mode: str
    candidates_considered: int
    results: list[SearchedMemory]
    suppressed: list[str]


class RecallResult(ToolModel):
    """The result of ``memory.recall``."""

    query: str
    recalled: int
    memory_ids: list[str]
    results: list[SearchedMemory]


class ContextBlockOut(ToolModel):
    """One compiled block with provenance."""

    memory_id: str
    memory_type: str
    section: str
    order: int
    content: str
    utility_score: float
    estimated_tokens: int


class CompiledContext(ToolModel):
    """The result of ``memory.compile_context``."""

    query: str
    token_budget: int
    estimated_tokens: int
    remaining_tokens: int
    within_budget: bool
    selected_count: int
    suppressed_count: int
    context_text: str
    blocks: list[ContextBlockOut]


class ExplainedMemory(ToolModel):
    """The result of ``memory.explain``."""

    memory_id: str
    query: str
    final_score: float
    relevance: float
    explanation: str
    duplicate_of: str | None = None
    superseded_by: str | None = None


class ForgetResult(ToolModel):
    """The result of ``memory.forget``."""

    id: str
    status: str
    version: int


# -- server -------------------------------------------------------------------


def build_mcp_server(
    operations: MemoryOperationsService,
    *,
    observability: ObservabilityService | None = None,
) -> MCPServer:
    """Build the MemForge MCP server bound to one operations service.

    ``observability`` is optional. When supplied, every tool call runs inside a
    trace of its own (``mcp.<tool>``) and the memory operations it performs are
    recorded against that same trace id. The tools' behaviour is identical with
    or without it -- a trace is added, never required.
    """
    server = MCPServer(
        name=SERVER_NAME,
        title="MemForge memory layer",
        instructions=MEMFORGE_MCP_INSTRUCTIONS,
        version="1.0.0",
    )
    _register_tools(server, operations, observability)
    return server


def _register_tools(
    server: MCPServer,
    operations: MemoryOperationsService,
    observability: ObservabilityService | None = None,
) -> None:
    """Attach the seven MemForge tools to a server.

    Each tool body is wrapped in :func:`_traced` when an observability service
    was supplied. The wrapper opens the trace, binds its id for the duration of
    the call and closes it afterwards; it never changes what the tool returns.
    """

    def _traced(tool: str, arguments: Mapping[str, Any], call: Callable[[], _ResultT]) -> _ResultT:
        """Run a tool body inside an MCP trace, when tracing is enabled."""
        if observability is None:
            return call()
        return cast("_ResultT", McpTracer(observability).call(tool, arguments, call))

    @server.tool(
        name="memory.store",
        title="Store a memory",
        description=(
            "Store one durable memory with its type, confidence and importance. "
            "Returns the stored memory exactly as it was written."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    def memory_store(
        content: Annotated[str, Field(min_length=1, max_length=4000)],
        memory_type: Annotated[
            str, Field(description="One of FACT, PREFERENCE, PROFILE, EVENT, TASK, OBSERVATION.")
        ],
        source: Annotated[str | None, Field(max_length=200)] = None,
        confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
        importance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
    ) -> StoredMemory:
        parsed = _memory_type(memory_type)
        if parsed is None:
            return _tool_failure(f"unknown memory_type {memory_type!r}")

        def call() -> StoredMemory:
            try:
                memory = operations.store(
                    content,
                    parsed,
                    source=source,
                    confidence=confidence,
                    importance=importance,
                )
            except MemforgeError as error:
                return _tool_error(error)
            return StoredMemory(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type.value,
                source=memory.source,
                confidence=memory.confidence,
                importance=memory.importance,
                version=memory.version,
            )

        return _traced(
            "memory.store",
            {
                "content_length": len(content),
                "memory_type": parsed.value,
                "check": True,
            },
            call,
        )

    @server.tool(
        name="memory.search",
        title="Search memories",
        description=(
            "Ranked lexical / semantic / hybrid search over the corpus, with a "
            "per-result explanation and duplicate suppression. Use memory.recall "
            "when you want a short list for a prompt."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def memory_search(
        query: Annotated[str, Field(min_length=1, max_length=1000)],
        top_k: Annotated[int, Field(ge=1, le=100)] = 10,
        retrieval_mode: Annotated[
            str, Field(description="lexical, semantic or hybrid")
        ] = "lexical",
    ) -> SearchResult:
        mode = _retrieval_mode(retrieval_mode)
        if mode is None:
            return _tool_failure(f"unknown retrieval_mode {retrieval_mode!r}")

        def call() -> SearchResult:
            try:
                response = operations.search(query, top_k=top_k, mode=mode)
            except MemforgeError as error:
                return _tool_error(error)
            return SearchResult(
                query=response.query,
                retrieval_mode=response.retrieval_mode.value,
                candidates_considered=response.candidates_considered,
                results=[_searched(result) for result in response.results],
                suppressed=[item.memory_id for item in response.suppressed],
            )

        return _traced(
            "memory.search",
            {"query_length": len(query), "top_k": top_k, "retrieval_mode": mode.value},
            call,
        )

    @server.tool(
        name="memory.recall",
        title="Recall memories for an agent",
        description=(
            "Shortlist the memories worth putting in front of a model: ranked "
            "search plus a relevance gate, so zero-relevance memories are dropped."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def memory_recall(
        query: Annotated[str, Field(min_length=1, max_length=1000)],
        top_k: Annotated[int, Field(ge=1, le=100)] = 5,
        min_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    ) -> RecallResult:
        def call() -> RecallResult:
            try:
                results = operations.recall(
                    query,
                    top_k=top_k,
                    policy=RecallPolicy(min_score=min_score, max_results=top_k),
                )
            except (MemforgeError, ValueError) as error:
                return _tool_error(error)
            return RecallResult(
                query=query,
                recalled=len(results),
                memory_ids=[result.memory.id for result in results],
                results=[_searched(result) for result in results],
            )

        return _traced(
            "memory.recall",
            {"query_length": len(query), "top_k": top_k, "min_score": min_score},
            call,
        )

    @server.tool(
        name="memory.update",
        title="Update a memory",
        description="Apply a partial update to a stored memory. Omitted fields keep their value.",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    def memory_update(
        memory_id: Annotated[str, Field(min_length=1)],
        content: Annotated[str | None, Field(min_length=1, max_length=4000)] = None,
        importance: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        confidence: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        status: Annotated[str | None, Field(description="ACTIVE or ARCHIVED")] = None,
    ) -> StoredMemory:
        parsed_status = _memory_status(status)
        if status is not None and parsed_status is None:
            return _tool_failure(f"unknown status {status!r}")

        def call() -> StoredMemory:
            try:
                memory = operations.update(
                    memory_id,
                    content=content,
                    confidence=confidence,
                    importance=importance,
                    status=parsed_status,
                )
            except MemforgeError as error:
                return _tool_error(error)
            return StoredMemory(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type.value,
                source=memory.source,
                confidence=memory.confidence,
                importance=memory.importance,
                version=memory.version,
            )

        return _traced(
            "memory.update",
            {
                "memory_id": memory_id,
                "fields": sorted(
                    name
                    for name, value in (
                        ("content", content),
                        ("importance", importance),
                        ("confidence", confidence),
                        ("status", status),
                    )
                    if value is not None
                ),
            },
            call,
        )

    @server.tool(
        name="memory.forget",
        title="Forget a memory",
        description=(
            "Soft-delete a memory: it disappears from normal reads but is never "
            "destroyed, and its history stays available."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    def memory_forget(
        memory_id: Annotated[str, Field(min_length=1)],
    ) -> ForgetResult:
        def call() -> ForgetResult:
            try:
                memory = operations.forget(memory_id)
            except MemforgeError as error:
                return _tool_error(error)
            return ForgetResult(id=memory.id, status=memory.status.value, version=memory.version)

        return _traced("memory.forget", {"memory_id": memory_id}, call)

    @server.tool(
        name="memory.compile_context",
        title="Compile a budgeted context",
        description=(
            "Turn retrieved memories into a token-budgeted, ordered, explained "
            "agent context with provenance for every block. Token counts are "
            "MemForge estimates, not provider tokenisations."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def memory_compile_context(
        query: Annotated[str, Field(min_length=1, max_length=1000)],
        token_budget: Annotated[int, Field(ge=1, description="Hard budget in estimated tokens")],
        top_k: Annotated[int, Field(ge=1, le=100)] = 10,
        retrieval_mode: Annotated[
            str, Field(description="lexical, semantic or hybrid")
        ] = "lexical",
    ) -> CompiledContext:
        mode = _retrieval_mode(retrieval_mode)
        if mode is None:
            return _tool_failure(f"unknown retrieval_mode {retrieval_mode!r}")

        def call() -> CompiledContext:
            try:
                result = operations.compile_context(
                    query, token_budget=token_budget, top_k=top_k, retrieval_mode=mode
                )
            except MemforgeError as error:
                return _tool_error(error)
            return CompiledContext(
                query=result.query,
                token_budget=result.token_budget,
                estimated_tokens=result.estimated_tokens,
                remaining_tokens=result.remaining_tokens,
                within_budget=result.within_budget,
                selected_count=result.selected_count,
                suppressed_count=result.suppressed_count,
                context_text=result.context_text,
                blocks=[
                    ContextBlockOut(
                        memory_id=block.memory_id,
                        memory_type=block.memory_type.value,
                        section=block.section.value,
                        order=block.order,
                        content=block.rendered_content,
                        utility_score=block.utility_score,
                        estimated_tokens=block.estimated_tokens,
                    )
                    for block in result.blocks
                ],
            )

        return _traced(
            "memory.compile_context",
            {
                "query_length": len(query),
                "token_budget": token_budget,
                "top_k": top_k,
                "retrieval_mode": mode.value,
            },
            call,
        )

    @server.tool(
        name="memory.explain",
        title="Explain a memory's ranking",
        description="Show exactly how one memory would rank for a query, with its factors.",
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def memory_explain(
        memory_id: Annotated[str, Field(min_length=1)],
        query: Annotated[str, Field(min_length=1, max_length=1000)],
    ) -> ExplainedMemory:
        def call() -> ExplainedMemory:
            try:
                result = operations.explain(memory_id, query)
            except MemforgeError as error:
                return _tool_error(error)
            return ExplainedMemory(
                memory_id=result.memory.id,
                query=query,
                final_score=result.final_score,
                relevance=result.relevance,
                explanation=result.explanation,
                duplicate_of=result.duplicate_of,
                superseded_by=result.superseded_by,
            )

        return _traced(
            "memory.explain",
            {"memory_id": memory_id, "query_length": len(query)},
            call,
        )


def run_mcp_server(
    operations: MemoryOperationsService,
    *,
    transport: Literal["stdio"] = "stdio",
    observability: ObservabilityService | None = None,
) -> None:
    """Run the MCP server over stdio (the transport MCP clients spawn)."""
    build_mcp_server(operations, observability=observability).run(transport=transport)


# -- helpers ------------------------------------------------------------------


def _memory_type(value: str) -> MemoryType | None:
    """Parse a memory type, or ``None`` when it is unknown."""
    try:
        return MemoryType(value.upper())
    except ValueError:
        return None


def _memory_status(value: str | None) -> MemoryStatus | None:
    """Parse a status, or ``None`` when it is unknown."""
    if value is None:
        return None
    try:
        return MemoryStatus(value.upper())
    except ValueError:
        return None


def _retrieval_mode(value: str) -> RetrievalMode | None:
    """Parse a retrieval mode, or ``None`` when it is unknown."""
    try:
        return RetrievalMode(value.lower())
    except ValueError:
        return None


def _searched(result: Any) -> SearchedMemory:
    """Condense a retrieval result for the wire."""
    return SearchedMemory(
        memory_id=result.memory.id,
        content=result.memory.content,
        memory_type=result.memory.memory_type.value,
        final_score=result.final_score,
        relevance=result.relevance,
        explanation=result.explanation,
    )


def tool_names() -> Sequence[str]:
    """The tool names this server exposes (kept in one place for tests)."""
    return (
        "memory.store",
        "memory.search",
        "memory.recall",
        "memory.update",
        "memory.forget",
        "memory.compile_context",
        "memory.explain",
    )


__all__ = [
    "MEMFORGE_MCP_INSTRUCTIONS",
    "SERVER_NAME",
    "CompiledContext",
    "ExplainedMemory",
    "ForgetResult",
    "RecallResult",
    "SearchResult",
    "SearchedMemory",
    "StoredMemory",
    "build_mcp_server",
    "run_mcp_server",
    "tool_names",
]
