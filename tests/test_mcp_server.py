"""M5 MCP server tests: tool contract, validation, errors and integration.

Every call goes through the real MCP SDK against a real (temporary) MemForge
stack -- no network, no provider, no credential.

The MCP protocol is asynchronous, so the tests drive it through a small ``run()``
wrapper instead of depending on an async pytest plugin: the suite stays
dependency-free and the event loop stays explicit.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.config import Settings
from app.domain.errors import MemoryNotFoundError, RepositoryError
from app.domain.ids import SequentialIdGenerator
from app.domain.models import MemoryType
from app.domain.time import ManualClock
from app.mcp import cli as mcp_cli
from app.mcp.cli import build_operations
from app.mcp.server import (
    MEMFORGE_MCP_INSTRUCTIONS,
    SERVER_NAME,
    build_mcp_server,
    tool_names,
)
from app.services.context_service import ContextCompilerService
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository

PYTHON = "User prefers Python for backend work."
NEAR_DUPLICATE = "The user prefers python for backend development work"
UNKNOWN_ID = "00000000-0000-5000-8000-000000000000"

#: Tools that mutate storage, and tools that only read.
WRITE_TOOLS = {"memory.store", "memory.update", "memory.forget"}
READ_TOOLS = {"memory.search", "memory.recall", "memory.compile_context", "memory.explain"}


@pytest.fixture(autouse=True)
def _quiet_sdk() -> None:
    """The SDK warns about dotted tool names; keep the test output readable."""
    logging.disable(logging.WARNING)


@pytest.fixture
def memories(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryService:
    """A deterministic memory service."""
    return MemoryService(repository, clock=clock, id_generator=SequentialIdGenerator(prefix="mcp"))


@pytest.fixture
def operations(
    repository: SqliteMemoryRepository, memories: MemoryService, clock: ManualClock
) -> MemoryOperationsService:
    """The memory-operations facade the tools are bound to."""
    retrieval = MemoryRetrievalService(repository, clock=clock)
    return MemoryOperationsService(memories, retrieval, ContextCompilerService(retrieval))


@pytest.fixture
def server(operations: MemoryOperationsService) -> MCPServer:
    """A MemForge MCP server."""
    return build_mcp_server(operations)


def run(coro: Any) -> Any:
    """Run one coroutine to completion (the MCP protocol is asynchronous)."""
    return asyncio.run(coro)


def tools(server: MCPServer) -> list[Any]:
    """The advertised tool list."""
    return cast("list[Any]", run(server.list_tools()))


def call(server: MCPServer, tool: str, **arguments: Any) -> Any:
    """Call a tool and return the raw result (errors included)."""
    return run(server.call_tool(tool, arguments))


def payload(server: MCPServer, tool: str, **arguments: Any) -> dict[str, Any]:
    """Call a tool and return its structured payload."""
    return cast("dict[str, Any]", call(server, tool, **arguments).structured_content)


def store(server: MCPServer, content: str = PYTHON) -> str:
    """Store one memory and return its id."""
    result = call(server, "memory.store", content=content, memory_type="PROFILE", confidence=0.9)
    assert not result.is_error, result.content
    return str(result.structured_content["id"])


class TestServerMetadata:
    def test_all_seven_tools_are_exposed(self, server: MCPServer) -> None:
        names = {tool.name for tool in tools(server)}
        assert names == set(tool_names())
        assert len(names) == 7

    def test_read_and_write_tools_partition_the_surface(self) -> None:
        assert set(tool_names()) == WRITE_TOOLS | READ_TOOLS
        assert not WRITE_TOOLS & READ_TOOLS

    def test_every_tool_documents_its_input_and_output(self, server: MCPServer) -> None:
        for tool in tools(server):
            assert tool.description, tool.name
            assert tool.input_schema.get("type") == "object", tool.name
            assert tool.output_schema, tool.name

    def test_required_inputs_are_declared(self, server: MCPServer) -> None:
        schemas = {tool.name: tool.input_schema for tool in tools(server)}
        assert set(schemas["memory.store"]["required"]) == {"content", "memory_type"}
        assert "query" in schemas["memory.search"]["required"]
        assert set(schemas["memory.compile_context"]["required"]) == {"query", "token_budget"}
        assert set(schemas["memory.explain"]["required"]) == {"memory_id", "query"}

    def test_bounds_appear_in_the_schema(self, server: MCPServer) -> None:
        schemas = {tool.name: tool.input_schema for tool in tools(server)}
        confidence = schemas["memory.store"]["properties"]["confidence"]
        assert confidence["minimum"] == 0.0
        assert confidence["maximum"] == 1.0

    def test_annotations_declare_the_side_effects(self, server: MCPServer) -> None:
        for tool in tools(server):
            annotations = tool.annotations
            assert annotations is not None, tool.name
            if tool.name in WRITE_TOOLS:
                assert annotations.read_only_hint is False, tool.name
            else:
                assert annotations.read_only_hint is True, tool.name
        forget = next(tool for tool in tools(server) if tool.name == "memory.forget")
        assert forget.annotations is not None
        assert forget.annotations.destructive_hint is True

    def test_server_is_named_and_documented(self, server: MCPServer) -> None:
        assert server.name == SERVER_NAME
        assert "memory" in MEMFORGE_MCP_INSTRUCTIONS


class TestStoreTool:
    def test_store_writes_a_memory(self, server: MCPServer) -> None:
        body = payload(
            server,
            "memory.store",
            content=PYTHON,
            memory_type="PROFILE",
            source="user:test",
            confidence=0.9,
            importance=0.8,
        )
        assert body["memory_type"] == "PROFILE"
        assert body["source"] == "user:test"
        assert body["confidence"] == 0.9
        assert body["version"] == 1

    @pytest.mark.parametrize("memory_type", ["NOT_A_TYPE", "profile2", ""])
    def test_unknown_types_are_refused(self, server: MCPServer, memory_type: str) -> None:
        result = call(server, "memory.store", content="x", memory_type=memory_type)
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "VALIDATION_ERROR"

    def test_an_out_of_range_score_is_refused(self, server: MCPServer) -> None:
        # Bounds are enforced by the published input schema, before the tool runs.
        with pytest.raises(ToolError):
            call(server, "memory.store", content="x", memory_type="FACT", confidence=5.0)

    def test_a_missing_required_field_is_refused(self, server: MCPServer) -> None:
        with pytest.raises(ToolError):
            call(server, "memory.store", memory_type="FACT")


class TestSearchTool:
    def test_search_returns_ranked_results(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        operations.store("User is learning TypeScript", MemoryType.OBSERVATION)

        body = payload(server, "memory.search", query="python backend", top_k=5)
        assert body["retrieval_mode"] == "lexical"
        assert body["results"][0]["content"] == PYTHON
        assert body["results"][0]["explanation"]

    def test_search_suppresses_duplicates(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        memory_id = store(server)
        operations.store(NEAR_DUPLICATE, MemoryType.OBSERVATION, importance=0.2)
        operations.memory_service.find_duplicates(memory_id)

        body = payload(server, "memory.search", query="python backend")
        assert len(body["results"]) == 1
        assert body["suppressed"]

    def test_an_unknown_mode_is_refused(self, server: MCPServer) -> None:
        result = call(server, "memory.search", query="python", retrieval_mode="telepathy")
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "VALIDATION_ERROR"

    def test_an_empty_corpus_returns_nothing(self, server: MCPServer) -> None:
        body = payload(server, "memory.search", query="python")
        assert body["results"] == []
        assert body["candidates_considered"] == 0


class TestRecallTool:
    def test_recall_returns_matches(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        body = payload(server, "memory.recall", query="python backend", top_k=5)
        assert body["recalled"] == 1
        assert body["memory_ids"]

    def test_recall_drops_unrelated_memories(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        body = payload(server, "memory.recall", query="kubernetes ingress")
        assert body["recalled"] == 0
        assert body["results"] == []

    def test_recall_honours_the_score_floor(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        body = payload(server, "memory.recall", query="python backend", min_score=0.99)
        assert body["recalled"] == 0


class TestCompileContextTool:
    def test_compile_returns_a_budgeted_context(self, server: MCPServer) -> None:
        store(server)
        body = payload(server, "memory.compile_context", query="python backend", token_budget=200)
        assert body["within_budget"] is True
        assert body["estimated_tokens"] <= body["token_budget"]
        assert body["context_text"].strip()
        assert body["blocks"][0]["content"] == PYTHON

    def test_compile_honours_a_tiny_budget(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        for index in range(4):
            operations.store(f"python backend note {index}", MemoryType.FACT)
        for budget in (1, 10, 30, 100, 500):
            body = payload(
                server, "memory.compile_context", query="python backend", token_budget=budget
            )
            assert body["estimated_tokens"] <= budget

    @pytest.mark.parametrize("budget", [0, -1])
    def test_invalid_budgets_are_refused(self, server: MCPServer, budget: int) -> None:
        with pytest.raises(ToolError):
            call(server, "memory.compile_context", query="python", token_budget=budget)

    def test_an_unknown_mode_is_refused(self, server: MCPServer) -> None:
        result = call(
            server,
            "memory.compile_context",
            query="python",
            token_budget=100,
            retrieval_mode="nonsense",
        )
        assert result.is_error is True


class TestUpdateTool:
    def test_update_applies_a_partial_change(self, server: MCPServer) -> None:
        memory_id = store(server)
        body = payload(server, "memory.update", memory_id=memory_id, importance=0.95)
        assert body["importance"] == 0.95
        assert body["content"] == PYTHON
        assert body["version"] == 2

    def test_update_can_archive(self, server: MCPServer) -> None:
        memory_id = store(server)
        result = call(server, "memory.update", memory_id=memory_id, status="ARCHIVED")
        assert result.is_error is False
        assert result.structured_content["content"] == PYTHON

    def test_an_unknown_id_is_normalised(self, server: MCPServer) -> None:
        result = call(server, "memory.update", memory_id=UNKNOWN_ID, importance=0.5)
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "MEMORY_NOT_FOUND"

    def test_an_unknown_status_is_refused(self, server: MCPServer) -> None:
        memory_id = store(server)
        result = call(server, "memory.update", memory_id=memory_id, status="GONE")
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "VALIDATION_ERROR"


class TestForgetTool:
    def test_forget_is_a_soft_delete(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        memory_id = store(server)
        body = payload(server, "memory.forget", memory_id=memory_id)
        assert body["status"] == "DELETED"
        assert body["version"] == 2
        assert len(operations.memory_service.get_history(memory_id)) == 2

    def test_forget_is_idempotent(self, server: MCPServer) -> None:
        memory_id = store(server)
        call(server, "memory.forget", memory_id=memory_id)
        assert call(server, "memory.forget", memory_id=memory_id).is_error is False

    def test_an_unknown_id_is_normalised(self, server: MCPServer) -> None:
        result = call(server, "memory.forget", memory_id=UNKNOWN_ID)
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "MEMORY_NOT_FOUND"


class TestExplainTool:
    def test_explain_scores_one_memory(self, server: MCPServer) -> None:
        memory_id = store(server)
        body = payload(server, "memory.explain", memory_id=memory_id, query="python")
        assert body["final_score"] > 0.0
        assert body["explanation"]

    def test_an_unknown_id_is_normalised(self, server: MCPServer) -> None:
        result = call(server, "memory.explain", memory_id=UNKNOWN_ID, query="python")
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "MEMORY_NOT_FOUND"


class TestErrorNormalisation:
    def test_the_envelope_matches_the_rest_api(self, server: MCPServer) -> None:
        result = call(server, "memory.forget", memory_id=UNKNOWN_ID)
        envelope = result.structured_content["error"]
        assert set(envelope) == {"code", "message", "details"}
        assert envelope["message"]

    def test_domain_errors_keep_their_codes(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        with pytest.raises(MemoryNotFoundError):
            operations.explain(UNKNOWN_ID, "python")
        result = call(server, "memory.explain", memory_id=UNKNOWN_ID, query="python")
        assert result.structured_content["error"]["code"] == "MEMORY_NOT_FOUND"

    def test_errors_are_also_text_content(self, server: MCPServer) -> None:
        result = call(server, "memory.forget", memory_id=UNKNOWN_ID)
        assert result.content
        assert "MEMORY_NOT_FOUND" in result.content[0].text

    def test_a_malformed_request_never_raises(self, server: MCPServer) -> None:
        # Every expected failure comes back as data, never as an exception.
        for tool, arguments in (
            ("memory.store", {"content": "x", "memory_type": "NOPE"}),
            ("memory.search", {"query": "python", "retrieval_mode": "nope"}),
            ("memory.forget", {"memory_id": UNKNOWN_ID}),
            ("memory.explain", {"memory_id": UNKNOWN_ID, "query": "python"}),
        ):
            assert call(server, tool, **arguments).is_error is True, tool

    @pytest.mark.parametrize(
        ("tool", "arguments", "code"),
        [
            ("memory.store", {"content": "   ", "memory_type": "FACT"}, "MEMORY_VALIDATION_ERROR"),
            ("memory.search", {"query": "   "}, "RETRIEVAL_VALIDATION_ERROR"),
            ("memory.recall", {"query": "   "}, "RETRIEVAL_VALIDATION_ERROR"),
            (
                "memory.compile_context",
                {"query": "   ", "token_budget": 100},
                "CONTEXT_VALIDATION_ERROR",
            ),
        ],
    )
    def test_domain_failures_inside_a_tool_are_normalised(
        self, server: MCPServer, tool: str, arguments: dict[str, Any], code: str
    ) -> None:
        # Whitespace-only input satisfies the published schema but violates an
        # M1/M3/M4 rule: the tool must report it, not crash the server.
        result = call(server, tool, **arguments)
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == code


class TestSecretSafety:
    def test_no_response_carries_credential_shaped_text(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        for tool, arguments in (
            ("memory.store", {"content": PYTHON, "memory_type": "PROFILE"}),
            ("memory.search", {"query": "python"}),
            ("memory.recall", {"query": "python"}),
            ("memory.compile_context", {"query": "python", "token_budget": 100}),
            ("memory.explain", {"memory_id": UNKNOWN_ID, "query": "python"}),
        ):
            rendered = str(payload(server, tool, **arguments)).lower()
            for marker in ("authorization", "bearer", "api_key", "api-key", "sk-"):
                assert marker not in rendered, f"{tool} leaked {marker!r}"

    def test_the_mcp_layer_never_touches_the_provider(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        # The MCP tools expose memory only; there is no inference path to leak a
        # credential through, and the operations service holds no provider.
        assert not hasattr(operations, "provider")
        assert not hasattr(server, "provider")


class TestDeterminism:
    def test_repeated_calls_are_identical(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        assert payload(server, "memory.search", query="python backend") == payload(
            server, "memory.search", query="python backend"
        )

    def test_tool_list_is_stable(self, server: MCPServer) -> None:
        assert [tool.name for tool in tools(server)] == [tool.name for tool in tools(server)]


class TestServerEntryPoint:
    def test_the_cli_builds_a_working_stack(self, tmp_path: Any) -> None:
        operations = build_operations(Settings(database_path=tmp_path / "mcp-cli.sqlite"))
        assert isinstance(operations, MemoryOperationsService)

        server = build_mcp_server(operations)
        assert {tool.name for tool in tools(server)} == set(tool_names())

    def test_the_cli_starts_the_stdio_server(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[object] = []

        def fake_run(self: Any, transport: str = "stdio") -> None:
            started.append(transport)

        monkeypatch.setattr("mcp.server.mcpserver.MCPServer.run", fake_run, raising=True)
        exit_code = mcp_cli.main(["--db", str(tmp_path / "cli.sqlite")])

        assert exit_code == 0
        assert started == ["stdio"]
        assert (tmp_path / "cli.sqlite").exists()

    def test_the_cli_reports_configuration_failures(
        self, tmp_path: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(path: Any) -> None:
            raise RepositoryError("storage unavailable")

        monkeypatch.setattr("app.mcp.cli.SqliteMemoryRepository", explode)
        exit_code = mcp_cli.main(["--db", str(tmp_path / "broken.sqlite")])

        assert exit_code == 1
        assert "memforge-mcp failed: storage unavailable" in capsys.readouterr().err

    def test_tools_share_one_service_instance(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        store(server)
        assert payload(server, "memory.search", query="python")["candidates_considered"] == 1

    def test_a_write_through_mcp_is_visible_to_the_services(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        memory_id = store(server)
        assert operations.memory_service.get_memory(memory_id) is not None
        assert operations.recall("python backend")

    def test_m2_relationships_are_honoured(
        self, server: MCPServer, operations: MemoryOperationsService
    ) -> None:
        memory_id = store(server)
        operations.store(NEAR_DUPLICATE, MemoryType.OBSERVATION)
        operations.memory_service.find_duplicates(memory_id)
        assert payload(server, "memory.recall", query="python backend")["recalled"] == 1
