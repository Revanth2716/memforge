"""Tests for the M6 observability domain model and redaction helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from app.domain.errors import ObservabilityError
from app.domain.observability import (
    FORBIDDEN_METADATA_KEYS,
    EventKind,
    EventStatus,
    OperationMetrics,
    Trace,
    TraceEvent,
    TraceKind,
    TraceSummary,
    redact_secrets_in,
    safe_detail,
    safe_metadata,
)
from app.domain.time import ManualClock


class TestEventKind:
    """The lifecycle vocabulary."""

    def test_start_and_complete_kinds_are_disjoint(self) -> None:
        starts = {kind for kind in EventKind if kind.is_start}
        completes = {kind for kind in EventKind if kind.is_complete}
        assert not starts & completes
        assert EventKind.RETRIEVAL_COMPLETE.is_complete
        assert EventKind.CONTEXT_COMPILE_START.is_start
        # A gate is a standalone decision, not half of a span.
        assert EventKind.MEMORY_GATE.is_start is False
        assert EventKind.MEMORY_GATE.is_complete is False

    def test_every_kind_maps_to_an_operation(self) -> None:
        operations = {kind.operation for kind in EventKind}
        assert "agent.request" in operations
        assert "memory.retrieval" in operations
        assert "context.compile" in operations
        assert "llm.generate" in operations
        assert "memory.write" in operations
        assert "mcp.request" in operations
        # Exactly four kinds are the opening half of a span whose closing half
        # reports the outcome under the same operation name.
        assert sum(1 for kind in EventKind if kind.is_start) == 4
        assert len(operations) == len(EventKind) - 4

    def test_the_documented_agent_lifecycle_is_representable(self) -> None:
        order = (
            EventKind.REQUEST,
            EventKind.MEMORY_GATE,
            EventKind.RETRIEVAL_START,
            EventKind.RETRIEVAL_COMPLETE,
            EventKind.CONTEXT_COMPILE_START,
            EventKind.CONTEXT_COMPILE_COMPLETE,
            EventKind.LLM_REQUEST,
            EventKind.LLM_RESPONSE,
            EventKind.MEMORY_CANDIDATE,
            EventKind.MEMORY_WRITE,
            EventKind.RESPONSE,
        )
        assert len(order) == 11
        assert all(isinstance(kind, EventKind) for kind in order)


class TestTrace:
    """Trace validation and serialisation."""

    def test_a_trace_requires_an_id_and_an_operation(self) -> None:
        now = ManualClock().now()
        with pytest.raises(ObservabilityError):
            Trace(
                trace_id="", kind=TraceKind.AGENT_REQUEST, operation="agent.respond", started_at=now
            )
        with pytest.raises(ObservabilityError):
            Trace(
                trace_id="trace-1",
                kind=TraceKind.AGENT_REQUEST,
                operation="   ",
                started_at=now,
            )

    def test_to_dict_is_json_ready(self) -> None:
        trace = Trace(
            trace_id="trace-1",
            kind=TraceKind.MCP_REQUEST,
            operation="mcp.memory.store",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            attributes={"tool": "memory.store"},
        )
        payload = trace.to_dict()
        assert payload["kind"] == "mcp_request"
        assert payload["attributes"] == {"tool": "memory.store"}
        assert payload["started_at"].startswith("2026-01-01")


class TestTraceEvent:
    """Event validation and terminal detection."""

    @staticmethod
    def _event(**overrides: object) -> TraceEvent:
        """Build a valid event, overriding what a test needs to change."""
        now = ManualClock().now()
        fields: dict[str, object] = {
            "trace_id": "trace-1",
            "event_id": "trace-1-e0000",
            "sequence": 0,
            "kind": EventKind.REQUEST,
            "operation": "agent.request",
            "status": EventStatus.OK,
            "timestamp": now,
        }
        fields.update(overrides)
        return TraceEvent(**fields)  # type: ignore[arg-type]

    def test_a_negative_sequence_or_duration_is_refused(self) -> None:
        with pytest.raises(ObservabilityError):
            self._event(sequence=-1)
        with pytest.raises(ObservabilityError):
            self._event(duration_ms=-1.0)
        with pytest.raises(ObservabilityError):
            self._event(event_id="")

    def test_terminal_events_are_complete_kinds_or_errors(self) -> None:
        assert self._event(kind=EventKind.LLM_RESPONSE, operation="llm.generate").is_terminal
        assert not self._event(kind=EventKind.LLM_REQUEST, operation="llm.generate").is_terminal
        assert self._event(
            kind=EventKind.LLM_REQUEST, operation="llm.generate", status=EventStatus.ERROR
        ).is_terminal

    def test_to_dict_exposes_every_field(self) -> None:
        now = ManualClock().now()
        payload = TraceEvent(
            trace_id="trace-1",
            event_id="trace-1-e0002",
            sequence=2,
            kind=EventKind.MEMORY_WRITE,
            operation="memory.write",
            status=EventStatus.OK,
            timestamp=now,
            parent_id="trace-1-e0001",
            duration_ms=1.25,
            metadata={"memory_ids": ["mem-1"]},
            detail="stored",
        ).to_dict()
        assert payload["sequence"] == 2
        assert payload["parent_id"] == "trace-1-e0001"
        assert payload["duration_ms"] == 1.25
        assert payload["metadata"] == {"memory_ids": ["mem-1"]}
        assert payload["detail"] == "stored"


class TestTraceSummary:
    """Summary projection."""

    @staticmethod
    def _summary(status: str, errors: tuple[str, ...]) -> TraceSummary:
        """Build a summary with one operation and the given terminal status."""
        return TraceSummary(
            trace_id="trace-1",
            kind=TraceKind.AGENT_REQUEST,
            operation="agent.respond",
            started_at=ManualClock().now(),
            event_count=3,
            operations=MappingProxyType({"agent.response": status}),
            durations_ms=MappingProxyType({"memory.retrieval": 1.0}),
            total_duration_ms=1.0,
            errors=errors,
            memory_ids=("mem-1",),
        )

    def test_succeeded_is_false_when_any_operation_failed(self) -> None:
        clean = self._summary("ok", ())
        assert clean.succeeded
        assert clean.to_dict()["operations"] == {"agent.response": "ok"}
        broken = self._summary("error", ("agent.response:response",))
        assert not broken.succeeded
        assert broken.to_dict()["operations"] == {"agent.response": "error"}
        assert broken.to_dict()["errors"] == ["agent.response:response"]


class TestOperationMetrics:
    """Counter projection."""

    def test_to_dict_sorts_every_counter_map(self) -> None:
        metrics = OperationMetrics(
            traces=MappingProxyType({"mcp_request": 1, "agent_request": 2}),
            operations=MappingProxyType({"memory.write": 1}),
            statuses=MappingProxyType({"ok": 3}),
            memory=MappingProxyType({"written": 1}),
            provider=MappingProxyType({"calls": 2}),
            context=MappingProxyType({"selected": 1}),
            usage_reported=0,
            failed_traces=0,
        )
        payload = metrics.to_dict()
        assert list(payload["traces"]) == ["agent_request", "mcp_request"]
        assert list(payload["operations"]) == ["memory.write"]
        assert payload["usage_reported"] == 0


class TestRedaction:
    """What may and may not reach a trace payload."""

    def test_credential_shaped_keys_are_dropped_not_masked(self) -> None:
        metadata = {
            "api_key": "sk-secret",
            "Authorization": "Bearer abc",
            "password": "hunter2",
            "memory_id": "mem-1",
            "top_k": 5,
        }
        safe = safe_metadata(metadata)
        assert safe == {"memory_id": "mem-1", "top_k": 5}
        assert "sk-secret" not in str(safe)

    def test_a_registered_secret_is_redacted_from_any_value(self) -> None:
        safe = safe_metadata(
            {"endpoint": "https://api.example/v1?key=sk-live-1"}, secrets=("sk-live-1",)
        )
        assert "sk-live-1" not in safe["endpoint"]
        assert "***redacted***" in safe["endpoint"]

    def test_auth_headers_are_masked_by_the_m5_helper(self) -> None:
        safe = safe_metadata({"headers": {"Authorization": "Bearer abc", "Accept": "json"}})
        assert "abc" not in str(safe["headers"])
        assert safe["headers"]["Accept"] == "json"

    def test_content_shaped_keys_never_survive(self) -> None:
        for key in ("prompt", "system_prompt", "answer", "content", "chain_of_thought"):
            assert key in FORBIDDEN_METADATA_KEYS
            assert safe_metadata({key: "sensitive"}) == {}

    def test_counts_and_sizes_are_allowed(self) -> None:
        safe = safe_metadata(
            {"input_characters": 120, "answer_characters": 40, "usage_reported": False}
        )
        assert safe["input_characters"] == 120
        assert safe["usage_reported"] is False

    def test_empty_metadata_is_an_empty_dict(self) -> None:
        assert safe_metadata(None) == {}
        assert safe_metadata({}) == {}

    def test_details_are_redacted_too(self) -> None:
        redacted = safe_detail("token is sk-abc", secrets=("sk-abc",))
        assert redacted is not None
        assert "sk-abc" not in redacted
        assert safe_detail(None) is None

    def test_nested_payloads_are_redacted_recursively(self) -> None:
        payload = redact_secrets_in(
            {"headers": {"Authorization": "Bearer abc"}, "body": ["safe", "sk-live-9"]},
            "sk-live-9",
        )
        assert "abc" not in str(payload["headers"])
        assert "sk-live-9" not in str(payload)
