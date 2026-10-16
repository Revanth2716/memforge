"""Tests for the M6 trace store and the observability service."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.domain.errors import ObservabilityError
from app.domain.observability import EventKind, EventStatus, Trace, TraceEvent, TraceKind
from app.domain.time import ManualClock
from app.observability.service import (
    NullTraceStore,
    ObservabilityService,
    SequentialIdFactory,
)
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.migrations import applied_migrations, run_migrations
from app.storage.sqlite.trace_store import SqliteTraceStore
from tests.m6_support import StepTimer


def _trace(trace_id: str = "trace-1", kind: TraceKind = TraceKind.AGENT_REQUEST) -> Trace:
    """A minimal valid trace."""
    return Trace(
        trace_id=trace_id,
        kind=kind,
        operation="agent.respond",
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        attributes={"tool": "memory.store"},
    )


def _event(
    sequence: int = 0,
    *,
    trace_id: str = "trace-1",
    kind: EventKind = EventKind.REQUEST,
    status: EventStatus = EventStatus.OK,
    metadata: dict[str, object] | None = None,
    duration_ms: float | None = None,
) -> TraceEvent:
    """A minimal valid event."""
    return TraceEvent(
        trace_id=trace_id,
        event_id=f"{trace_id}-e{sequence:04d}",
        sequence=sequence,
        kind=kind,
        operation=kind.operation,
        status=status,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, sequence, tzinfo=UTC),
        metadata=metadata or {},
        duration_ms=duration_ms,
    )


class TestSqliteTraceStore:
    """Append-only persistence."""

    def test_migrations_create_the_observability_tables(self, db_path: Path) -> None:
        SqliteTraceStore(db_path).put_trace(_trace())
        with closing(open_connection(db_path)) as conn:
            run_migrations(conn)
            tables = {
                str(row["name"])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            applied = applied_migrations(conn)
        assert {"traces", "trace_events"} <= tables
        assert "0004_observability" in applied

    def test_a_trace_and_its_events_round_trip(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(_trace())
        store.append_event(_event(0, metadata={"provider": "mock", "usage_reported": False}))
        store.append_event(_event(1, kind=EventKind.RESPONSE, duration_ms=1.5))

        assert store.get_trace("trace-1") == _trace()
        events = store.get_events("trace-1")
        assert [event.sequence for event in events] == [0, 1]
        assert events[0].metadata["provider"] == "mock"
        assert events[1].duration_ms == 1.5
        assert store.count_events() == 2

    def test_an_unknown_trace_reads_as_none(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(_trace())
        assert store.get_trace("missing") is None
        assert store.get_events("missing") == ()

    def test_storing_the_same_trace_twice_is_a_no_op(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(_trace())
        store.put_trace(_trace())
        assert len(store.list_traces(limit=10)) == 1

    def test_a_duplicate_sequence_is_refused(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(_trace())
        store.append_event(_event(0))
        with pytest.raises(ObservabilityError):
            store.append_event(_event(0))

    def test_listing_is_newest_first_and_filterable(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        for index in range(3):
            kind = TraceKind.MCP_REQUEST if index == 2 else TraceKind.AGENT_REQUEST
            store.put_trace(
                Trace(
                    trace_id=f"trace-{index}",
                    kind=kind,
                    operation="op",
                    started_at=datetime(2026, 1, 1, 12, index, tzinfo=UTC),
                )
            )
        newest = store.list_traces(limit=2)
        assert [trace.trace_id for trace in newest] == ["trace-2", "trace-1"]
        only_mcp = store.list_traces(limit=10, kind=TraceKind.MCP_REQUEST)
        assert [trace.trace_id for trace in only_mcp] == ["trace-2"]

    def test_a_limit_below_one_is_refused(self, db_path: Path) -> None:
        with pytest.raises(ObservabilityError):
            SqliteTraceStore(db_path).list_traces(limit=0)

    def test_metadata_is_stored_as_sorted_json(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(_trace())
        store.append_event(_event(0, metadata={"b": 2, "a": 1}))
        with closing(open_connection(db_path)) as conn:
            row = conn.execute("SELECT metadata FROM trace_events").fetchone()
        assert str(row["metadata"]) == json.dumps(
            {"a": 1, "b": 2}, sort_keys=True, separators=(",", ":")
        )

    def test_a_corrupt_row_is_reported_not_silently_dropped(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(_trace())
        store.append_event(_event(0))
        with closing(open_connection(db_path)) as conn:
            conn.execute("UPDATE trace_events SET kind = 'teleportation'")
        with pytest.raises(ObservabilityError):
            store.get_events("trace-1")

    def test_traces_survive_a_new_store_instance(self, db_path: Path) -> None:
        first = SqliteTraceStore(db_path)
        first.put_trace(_trace())
        first.append_event(_event(0))
        second = SqliteTraceStore(db_path)
        assert second.get_trace("trace-1") is not None
        assert len(second.get_events("trace-1")) == 1


class TestNullTraceStore:
    """The in-memory store used when persistence is not wanted."""

    def test_it_behaves_like_the_persistent_one(self) -> None:
        store = NullTraceStore()
        store.put_trace(_trace())
        store.append_event(_event(1))
        store.append_event(_event(0))
        assert [event.sequence for event in store.get_events("trace-1")] == [0, 1]
        assert store.count_events() == 2
        assert store.all_events(limit=1)[0].sequence == 0

    def test_a_duplicate_sequence_is_refused(self) -> None:
        store = NullTraceStore()
        store.put_trace(_trace())
        store.append_event(_event(0))
        with pytest.raises(ObservabilityError):
            store.append_event(_event(0))


class TestSequentialIdFactory:
    """Reproducible trace ids."""

    def test_ids_increase_and_are_reproducible(self) -> None:
        first = SequentialIdFactory(prefix="demo")
        second = SequentialIdFactory(prefix="demo")
        assert [first() for _ in range(3)] == ["demo-0001", "demo-0002", "demo-0003"]
        assert [second() for _ in range(3)] == ["demo-0001", "demo-0002", "demo-0003"]


class TestObservabilityServiceLifecycle:
    """Trace creation, event sequencing and closure."""

    def test_starting_a_trace_records_its_opening_event(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond", attributes={"top_k": 5})
        assert trace.trace_id == "trace-0001"
        events = service.get_events(trace.trace_id)
        assert [event.kind for event in events] == [EventKind.REQUEST]
        assert events[0].metadata == {"top_k": 5}

    def test_an_mcp_trace_opens_with_an_mcp_request(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("mcp.memory.store", kind=TraceKind.MCP_REQUEST)
        events = service.get_events(trace.trace_id)
        assert events[0].kind is EventKind.MCP_REQUEST
        service.end_trace(trace.trace_id)
        assert service.get_events(trace.trace_id)[-1].kind is EventKind.MCP_RESPONSE

    def test_event_sequences_are_total_and_ordered(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        for _index in range(5):
            service.record_event(EventKind.MEMORY_GATE, trace_id=trace.trace_id)
        events = service.get_events(trace.trace_id)
        assert [event.sequence for event in events] == list(range(6))
        assert [event.event_id for event in events] == [
            f"{trace.trace_id}-e{index:04d}" for index in range(6)
        ]

    def test_closing_an_unknown_trace_is_a_no_op(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        assert in_memory_observability.end_trace("never-recorded") is None

    def test_an_open_span_is_closed_as_an_error(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        service.begin_operation(EventKind.RETRIEVAL_START, trace_id=trace.trace_id)
        service.end_trace(trace.trace_id, status=EventStatus.ERROR)
        events = service.get_events(trace.trace_id)
        assert events[-2].kind is EventKind.RETRIEVAL_COMPLETE
        assert events[-2].status is EventStatus.ERROR
        assert events[-1].status is EventStatus.ERROR

    def test_durations_are_measured_not_declared(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        span = service.begin_operation(EventKind.RETRIEVAL_START, trace_id=trace.trace_id)
        event = service.complete_operation(span, EventKind.RETRIEVAL_COMPLETE)
        assert event.duration_ms is not None
        assert event.duration_ms > 0.0
        assert event.parent_id == span.start_event.event_id

    def test_the_operation_context_manager_records_and_reraises(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        with (
            pytest.raises(RuntimeError),
            service.operation(
                EventKind.LLM_REQUEST,
                trace_id=trace.trace_id,
                complete_kind=EventKind.LLM_RESPONSE,
            ),
        ):
            raise RuntimeError("provider exploded")
        response = service.get_events(trace.trace_id)[-1]
        assert response.kind is EventKind.LLM_RESPONSE
        assert response.status is EventStatus.ERROR
        assert response.detail == "RuntimeError"

    def test_the_operation_context_manager_closes_on_success(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        with service.operation(
            EventKind.RETRIEVAL_START,
            trace_id=trace.trace_id,
            complete_kind=EventKind.RETRIEVAL_COMPLETE,
            metadata={"top_k": 3},
        ):
            pass
        event = service.get_events(trace.trace_id)[-1]
        assert event.kind is EventKind.RETRIEVAL_COMPLETE
        assert event.status is EventStatus.OK
        assert event.metadata["top_k"] == 3

    def test_the_current_trace_is_scoped_and_restored(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        before = service.current_trace_id
        assert before is None
        trace = service.start_trace("agent.respond")
        with service.use_trace(trace.trace_id):
            assert service.current_trace_id == trace.trace_id
        restored: str | None = service.current_trace_id
        assert restored is before

    def test_the_current_trace_is_restored_after_an_error(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        with pytest.raises(RuntimeError), service.use_trace(trace.trace_id):
            raise RuntimeError("boom")
        assert service.current_trace_id is None


class TestObservabilityServiceSafety:
    """Nothing sensitive can reach the store."""

    def test_a_registered_secret_never_reaches_an_event(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        service.register_secret("sk-live-do-not-log")
        trace = service.start_trace("agent.respond")
        event = service.record_event(
            EventKind.LLM_RESPONSE,
            trace_id=trace.trace_id,
            metadata={"endpoint": "https://api/v1?key=sk-live-do-not-log"},
            detail="used sk-live-do-not-log",
        )
        assert "sk-live-do-not-log" not in json.dumps(event.to_dict())

    def test_a_credential_shaped_metadata_key_is_dropped(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        event = service.record_event(
            EventKind.LLM_REQUEST, trace_id=trace.trace_id, metadata={"api_key": "sk-1"}
        )
        assert event.metadata == {}

    def test_trace_attributes_are_redacted_at_creation(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        service.register_secret("sk-live")
        trace = service.start_trace("agent.respond", attributes={"key": "sk-live"})
        assert "sk-live" not in json.dumps(trace.to_dict())


class TestObservabilityServiceSummaries:
    """Reduction of events into summaries and counters."""

    def _recorded_turn(self, service: ObservabilityService) -> str:
        """Record one complete, successful agent-shaped trace."""
        trace = service.start_trace("agent.respond")
        span = service.begin_operation(
            EventKind.RETRIEVAL_START,
            trace_id=trace.trace_id,
            metadata={"query_characters": 20},
        )
        service.complete_operation(
            span,
            EventKind.RETRIEVAL_COMPLETE,
            metadata={"recalled": 3, "duplicate_candidates": 1, "candidates": 4},
        )
        span = service.begin_operation(EventKind.CONTEXT_COMPILE_START, trace_id=trace.trace_id)
        service.complete_operation(
            span,
            EventKind.CONTEXT_COMPILE_COMPLETE,
            metadata={"candidates": 4, "selected": 2, "suppressed": 2, "over_budget": False},
        )
        span = service.begin_operation(EventKind.LLM_REQUEST, trace_id=trace.trace_id)
        service.complete_operation(
            span,
            EventKind.LLM_RESPONSE,
            metadata={"provider": "mock", "usage_reported": False, "output_characters": 40},
        )
        service.record_event(
            EventKind.MEMORY_CANDIDATE, trace_id=trace.trace_id, metadata={"candidate_count": 1}
        )
        service.record_event(
            EventKind.MEMORY_WRITE,
            trace_id=trace.trace_id,
            status=EventStatus.SKIPPED,
            metadata={"memory_ids": []},
        )
        service.end_trace(trace.trace_id, metadata={"answer_characters": 40, "used_memory": True})
        return trace.trace_id

    def test_the_summary_reports_one_status_per_operation(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace_id = self._recorded_turn(service)
        summary = service.summarize_trace(trace_id)
        assert summary is not None
        assert summary.operations["agent.request"] == "ok"
        assert summary.operations["memory.retrieval"] == "ok"
        assert summary.operations["context.compile"] == "ok"
        assert summary.operations["llm.generate"] == "ok"
        assert summary.operations["memory.write"] == "skipped"
        assert summary.succeeded
        assert summary.errors == ()
        assert summary.durations_ms["memory.retrieval"] > 0.0

    def test_the_summary_collects_memory_ids(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("mcp.memory.store", kind=TraceKind.MCP_REQUEST)
        service.record_event(
            EventKind.MEMORY_WRITE, trace_id=trace.trace_id, metadata={"memory_ids": ["mem-1"]}
        )
        summary = service.summarize_trace(trace.trace_id)
        assert summary is not None
        assert summary.memory_ids == ("mem-1",)

    def test_an_unknown_trace_summarises_to_none(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        assert in_memory_observability.summarize_trace("missing") is None

    def test_counters_are_derived_from_real_events(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        self._recorded_turn(service)
        metrics = service.get_metrics()
        assert metrics.traces == {"agent_request": 1}
        assert metrics.operations["context.compile"] == 1
        assert metrics.statuses["ok"] > 0
        assert metrics.statuses["skipped"] == 1
        assert metrics.memory["recalled"] == 3
        assert metrics.memory["duplicate_candidates"] == 1
        assert metrics.memory["candidates"] == 1
        assert metrics.context == {"candidates": 4, "selected": 2, "suppressed": 2}
        assert metrics.provider["calls"] == 1
        assert metrics.usage_reported == 0
        assert metrics.failed_traces == 0

    def test_a_skipped_write_is_counted_as_skipped_not_written(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        metrics = in_memory_observability.get_metrics()
        assert metrics.memory.get("written", 0) == 0

    def test_a_failed_trace_is_counted(self, in_memory_observability: ObservabilityService) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        service.record_event(
            EventKind.ERROR, trace_id=trace.trace_id, status=EventStatus.ERROR, detail="Boom"
        )
        service.end_trace(trace.trace_id, status=EventStatus.ERROR)
        metrics = service.get_metrics()
        assert metrics.failed_traces == 1
        assert metrics.statuses["error"] == 2

    def test_usage_is_counted_only_when_the_provider_reported_it(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        span = service.begin_operation(EventKind.LLM_REQUEST, trace_id=trace.trace_id)
        service.complete_operation(
            span,
            EventKind.LLM_RESPONSE,
            metadata={"usage_reported": True, "usage_total_tokens": 42},
        )
        metrics = service.get_metrics()
        assert metrics.usage_reported == 1
        assert metrics.provider["usage_reported"] == 1

    def test_a_failed_provider_call_is_counted_as_a_failure(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        span = service.begin_operation(EventKind.LLM_REQUEST, trace_id=trace.trace_id)
        service.complete_operation(
            span, EventKind.LLM_RESPONSE, status=EventStatus.ERROR, detail="LLMTimeoutError"
        )
        assert service.get_metrics().provider["failures"] == 1

    def test_an_empty_store_reports_no_counters_rather_than_zeroes(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        metrics = in_memory_observability.get_metrics()
        assert metrics.traces == {}
        assert metrics.memory == {}
        assert metrics.usage_reported == 0

    def test_counters_ignore_a_non_counted_metadata_value(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        service.record_event(
            EventKind.MEMORY_CANDIDATE,
            trace_id=trace.trace_id,
            metadata={"candidate_count": "two"},
        )
        assert service.get_metrics().memory.get("candidates", 0) == 0


class TestLifecycleCounters:
    """Counters that only a real write, update or forget can move."""

    def _record(
        self,
        service: ObservabilityService,
        kind: EventKind,
        memory_ids: tuple[str, ...],
        *,
        status: EventStatus = EventStatus.OK,
    ) -> None:
        """Record one lifecycle event against a fresh trace."""
        trace = service.start_trace("mcp.memory.store", kind=TraceKind.MCP_REQUEST)
        service.record_event(
            kind, trace_id=trace.trace_id, status=status, metadata={"memory_ids": list(memory_ids)}
        )

    def test_updates_and_forgets_are_counted_separately(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        self._record(service, EventKind.MEMORY_UPDATE, ("mem-1",))
        self._record(service, EventKind.MEMORY_FORGET, ("mem-2",))
        metrics = service.get_metrics()
        assert metrics.memory["updated"] == 1
        assert metrics.memory["forgotten"] == 1
        assert metrics.memory["memory_ids_touched"] == 2

    def test_an_over_budget_compilation_is_counted(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("agent.respond")
        service.record_event(
            EventKind.CONTEXT_COMPILE_COMPLETE,
            trace_id=trace.trace_id,
            metadata={"over_budget": True, "selected": 3},
        )
        metrics = service.get_metrics()
        assert metrics.context["over_budget"] == 1
        assert metrics.context["selected"] == 3

    def test_mcp_calls_and_failures_are_counted(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        self._record(service, EventKind.MEMORY_WRITE, ("mem-1",))
        failed = service.start_trace("mcp.memory.forget", kind=TraceKind.MCP_REQUEST)
        service.record_event(
            EventKind.MCP_RESPONSE, trace_id=failed.trace_id, status=EventStatus.ERROR
        )
        metrics = service.get_metrics()
        assert metrics.provider["mcp_calls"] == 2
        assert metrics.provider["mcp_failures"] == 1
        assert metrics.memory["written"] == 1
        assert metrics.memory["created"] == 1

    def test_a_memory_id_given_as_a_bare_string_is_collected(
        self, in_memory_observability: ObservabilityService
    ) -> None:
        service = in_memory_observability
        trace = service.start_trace("mcp.memory.store", kind=TraceKind.MCP_REQUEST)
        service.record_event(
            EventKind.MEMORY_WRITE, trace_id=trace.trace_id, metadata={"memory_id": "mem-1"}
        )
        summary = service.summarize_trace(trace.trace_id)
        assert summary is not None
        assert summary.memory_ids == ("mem-1",)


class TestPersistenceAcrossRestart:
    """Traces outlive the process that wrote them."""

    def test_a_new_service_reads_the_previous_run(
        self, db_path: Path, clock: ManualClock, timer: StepTimer
    ) -> None:
        first = ObservabilityService(
            SqliteTraceStore(db_path),
            clock=clock,
            timer=timer,
            id_factory=SequentialIdFactory(),
        )
        trace = first.start_trace("agent.respond")
        first.record_event(EventKind.MEMORY_GATE, trace_id=trace.trace_id)
        first.end_trace(trace.trace_id)

        second = ObservabilityService(SqliteTraceStore(db_path), clock=clock)
        assert second.get_trace("trace-0001") is not None
        events = second.get_events("trace-0001")
        assert [event.kind.value for event in events] == [
            "request",
            "memory_gate",
            "response",
        ]
        summary = second.summarize_trace("trace-0001")
        assert summary is not None
        assert summary.event_count == 3
