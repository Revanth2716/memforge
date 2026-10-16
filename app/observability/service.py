"""The observability service (M6).

One service owns the trace lifecycle so that services do not sprout ad-hoc
logging. It is independent of FastAPI, of the API layer and of any provider, and
it has exactly one job: turn operations into safe, ordered, persisted events.

::

    ObservabilityService.start_trace(...)
        ├─ record_event(...)
        ├─ begin_operation(...) -> complete_operation(...)
        └─ get_trace / summarize_trace / get_metrics

Determinism
-----------
* event ordering comes from a per-trace ``sequence``, never from comparing
  timestamps (two events can land in the same microsecond);
* identifiers are derived from the trace id plus the sequence, so an identical
  execution produces identical event ids;
* durations are **measured** with an injectable timer and are never asserted on
  in a test; the HTTP smoke test compares the event *sequence*, not the numbers.

Safe by construction
--------------------
Every metadata mapping passes through
:func:`~app.domain.observability.safe_metadata` before it can reach the store, so
a caller cannot accidentally persist an API key, an authorization header or a
prompt by passing it as metadata. A secret can additionally be *registered*
(:meth:`ObservabilityService.register_secret`); from then on its literal value is
redacted out of every recorded string, wherever it came from.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.domain.errors import ObservabilityError
from app.domain.observability import (
    EventKind,
    EventStatus,
    OperationMetrics,
    Trace,
    TraceEvent,
    TraceKind,
    TraceSummary,
    safe_detail,
    safe_metadata,
)
from app.domain.time import Clock, SystemClock
from app.storage.sqlite.trace_store import TraceStore

#: Prefix of every identifier this service mints.
ID_PREFIX = "trace"

#: Placeholder used by the demo and tests for reproducible trace ids.
SEQUENTIAL_ID_PREFIX = "trace"


class NullTraceStore(TraceStore):
    """A store that keeps nothing on disk, for callers that want no persistence.

    Still fully functional -- events are counted in memory -- so instrumented
    code paths behave identically with and without a database.
    """

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}
        self._events: dict[str, list[TraceEvent]] = {}

    def put_trace(self, trace: Trace) -> None:
        """Keep the trace in memory (first write wins)."""
        self._traces.setdefault(trace.trace_id, trace)

    def append_event(self, event: TraceEvent) -> None:
        """Keep the event in memory, refusing a duplicate sequence."""
        events = self._events.setdefault(event.trace_id, [])
        if any(existing.sequence == event.sequence for existing in events):
            raise ObservabilityError(
                f"duplicate sequence {event.sequence} in trace {event.trace_id}"
            )
        events.append(event)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return an in-memory trace."""
        return self._traces.get(trace_id)

    def get_events(self, trace_id: str) -> tuple[TraceEvent, ...]:
        """Return in-memory events, ordered by sequence."""
        return tuple(sorted(self._events.get(trace_id, ()), key=lambda e: e.sequence))

    def list_traces(self, *, limit: int = 50, kind: TraceKind | None = None) -> tuple[Trace, ...]:
        """Return in-memory traces, newest first, ties broken by id."""
        selected = [trace for trace in self._traces.values() if kind is None or trace.kind is kind]
        ordered = sorted(selected, key=lambda t: (t.started_at, t.trace_id), reverse=True)
        return tuple(ordered[:limit])

    def count_events(self) -> int:
        """Total in-memory events."""
        return sum(len(events) for events in self._events.values())

    def all_events(self, *, limit: int | None = None) -> tuple[TraceEvent, ...]:
        """Every in-memory event, in a stable order."""
        events = [
            event
            for trace_events in self._events.values()
            for event in sorted(trace_events, key=lambda e: e.sequence)
        ]
        return tuple(events[:limit] if limit is not None else events)


class SequentialIdFactory:
    """Reproducible trace ids: ``trace-0001``, ``trace-0002``, ...

    Used by the demo and by tests so that output is byte-identical across runs.
    Production uses :class:`app.domain.ids.UuidIdGenerator` by default.
    """

    def __init__(self, prefix: str = SEQUENTIAL_ID_PREFIX, width: int = 4) -> None:
        self._prefix = prefix
        self._width = width
        self._counter = 0

    def __call__(self) -> str:
        """Return the next id."""
        self._counter += 1
        return f"{self._prefix}-{self._counter:0{self._width}d}"


@dataclass(slots=True)
class OperationSpan:
    """An operation that has been started but not yet completed."""

    trace_id: str
    start_event: TraceEvent
    #: Timer reading taken when the operation started.
    started_perf: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservabilityService:
    """Records what happened, safely and in order."""

    def __init__(
        self,
        store: TraceStore | None = None,
        *,
        clock: Clock | None = None,
        timer: Callable[[], float] = time.perf_counter,
        id_factory: Callable[[], str] | None = None,
        secrets: Sequence[str | None] = (),
    ) -> None:
        """Args:
        store: Where traces are persisted (``None`` keeps them in memory).
        clock: Source of event timestamps (injected, so tests are deterministic).
        timer: Monotonic timer used to measure durations.
        id_factory: Source of new trace ids (injected for reproducible runs).
        secrets: Literal values that must never appear in a recorded string.
        """
        self._store: TraceStore = store if store is not None else NullTraceStore()
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._timer = timer
        self._id_factory = id_factory
        self._secrets: tuple[str | None, ...] = tuple(secret for secret in secrets if secret)
        self._sequence: dict[str, int] = {}
        self._open: dict[str, list[OperationSpan]] = {}
        #: The trace every instrumented call inside ``use_trace`` belongs to.
        self._current: ContextVar[str | None] = ContextVar("memforge_current_trace", default=None)

    # -- introspection ----------------------------------------------------

    @property
    def store(self) -> TraceStore:
        """The backing store."""
        return self._store

    @property
    def current_trace_id(self) -> str | None:
        """The trace currently in scope, or ``None`` outside a traced call."""
        return self._current.get()

    @contextmanager
    def use_trace(self, trace_id: str) -> Iterator[str]:
        """Bind ``trace_id`` for everything nested inside this block.

        A :class:`~contextvars.ContextVar` rather than instance state, so two
        concurrent requests cannot capture each other's events. The token is
        restored on exit, including when the block raises.
        """
        token: Token[str | None] = self._current.set(trace_id)
        try:
            yield trace_id
        finally:
            self._current.reset(token)

    def register_secret(self, secret: str | None) -> None:
        """Redact ``secret`` from every string recorded from now on.

        The value is kept in memory only; it is never written to the store, an
        event, a log line or a response body.
        """
        if secret and secret not in self._secrets:
            self._secrets = (*self._secrets, secret)

    # -- traces -----------------------------------------------------------

    def start_trace(
        self,
        operation: str,
        *,
        kind: TraceKind = TraceKind.AGENT_REQUEST,
        attributes: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Trace:
        """Open a trace and record its opening ``REQUEST``/``MCP_REQUEST`` event."""
        resolved_id = trace_id or self._new_trace_id()
        trace = Trace(
            trace_id=resolved_id,
            kind=kind,
            operation=operation,
            started_at=self._clock.now(),
            attributes=safe_metadata(attributes, secrets=self._secrets),
        )
        self._store.put_trace(trace)
        self._sequence[resolved_id] = 0
        self._open[resolved_id] = []
        self.record_event(
            EventKind.MCP_REQUEST if kind is TraceKind.MCP_REQUEST else EventKind.REQUEST,
            trace_id=resolved_id,
            metadata=trace.attributes,
        )
        return trace

    def end_trace(
        self,
        trace_id: str,
        *,
        status: EventStatus = EventStatus.OK,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraceEvent | None:
        """Close a trace with its terminal event.

        Any operation still open is closed as ``error`` first, so a trace never
        ends with a dangling span. Returns ``None`` for an unknown trace:
        closing one is a caller bug, not a reason to fail a request.
        """
        trace = self._store.get_trace(trace_id)
        if trace is None:
            return None
        open_spans = self._open.get(trace_id, [])
        while open_spans:
            span = open_spans.pop()
            self.complete_operation(
                span,
                _COMPLETE_FOR_START.get(span.start_event.kind, span.start_event.kind),
                status=EventStatus.ERROR,
                detail="trace ended while the operation was still open",
            )
        closing = (
            EventKind.MCP_RESPONSE if trace.kind is TraceKind.MCP_REQUEST else EventKind.RESPONSE
        )
        return self.record_event(closing, trace_id=trace_id, status=status, metadata=metadata)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return a stored trace."""
        return self._store.get_trace(trace_id)

    def get_events(self, trace_id: str) -> tuple[TraceEvent, ...]:
        """Every event of a trace, in order."""
        return self._store.get_events(trace_id)

    def list_traces(self, *, limit: int = 50, kind: TraceKind | None = None) -> tuple[Trace, ...]:
        """Recent traces, newest first."""
        return self._store.list_traces(limit=limit, kind=kind)

    # -- events -----------------------------------------------------------

    def record_event(
        self,
        kind: EventKind,
        *,
        trace_id: str,
        status: EventStatus = EventStatus.OK,
        metadata: Mapping[str, Any] | None = None,
        detail: str | None = None,
        parent_id: str | None = None,
        duration_ms: float | None = None,
    ) -> TraceEvent:
        """Append one event to a trace and return it."""
        sequence = self._sequence.get(trace_id, 0)
        self._sequence[trace_id] = sequence + 1
        event = TraceEvent(
            trace_id=trace_id,
            event_id=f"{trace_id}-e{sequence:04d}",
            sequence=sequence,
            kind=kind,
            operation=kind.operation,
            status=status,
            timestamp=self._clock.now(),
            parent_id=parent_id,
            duration_ms=None if duration_ms is None else round(float(duration_ms), 3),
            metadata=safe_metadata(metadata, secrets=self._secrets),
            detail=safe_detail(detail, secrets=self._secrets),
        )
        self._store.append_event(event)
        return event

    def begin_operation(
        self,
        kind: EventKind,
        *,
        trace_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> OperationSpan:
        """Open a timed operation and return its span handle."""
        start = self.record_event(kind, trace_id=trace_id, metadata=metadata)
        span = OperationSpan(
            trace_id=trace_id,
            start_event=start,
            started_perf=self._timer(),
            metadata=metadata or {},
        )
        self._open.setdefault(trace_id, []).append(span)
        return span

    def complete_operation(
        self,
        span: OperationSpan,
        complete_kind: EventKind,
        *,
        status: EventStatus = EventStatus.OK,
        metadata: Mapping[str, Any] | None = None,
        detail: str | None = None,
    ) -> TraceEvent:
        """Close a span, recording its *measured* duration."""
        open_spans = self._open.get(span.trace_id, [])
        if span in open_spans:
            open_spans.remove(span)
        merged: dict[str, Any] = dict(span.metadata)
        merged.update(metadata or {})
        duration = max(0.0, self._timer() - span.started_perf) * 1000.0
        return self.record_event(
            complete_kind,
            trace_id=span.trace_id,
            status=status,
            metadata=merged,
            detail=detail,
            parent_id=span.start_event.event_id,
            duration_ms=duration,
        )

    @contextmanager
    def operation(
        self,
        kind: EventKind,
        *,
        trace_id: str,
        complete_kind: EventKind | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[OperationSpan]:
        """Time an operation as a context manager.

        On an exception the operation is recorded as ``error`` and the exception
        is re-raised: observability never swallows a failure.
        """
        span = self.begin_operation(kind, trace_id=trace_id, metadata=metadata)
        try:
            yield span
        except Exception as error:
            self.complete_operation(
                span,
                complete_kind or kind,
                status=EventStatus.ERROR,
                detail=type(error).__name__,
            )
            raise
        self.complete_operation(span, complete_kind or kind, metadata=metadata)

    # -- aggregation ------------------------------------------------------

    def summarize_trace(self, trace_id: str) -> TraceSummary | None:
        """Reduce one trace to a summary by counting its events."""
        trace = self._store.get_trace(trace_id)
        if trace is None:
            return None
        events = self._store.get_events(trace_id)
        operations: dict[str, str] = {}
        durations: dict[str, float] = {}
        errors: list[str] = []
        memory_ids: list[str] = []
        total = 0.0
        for event in events:
            # The last event that closes (or *is*) an operation decides that
            # operation's status. Start events never do: they are the opening
            # half of a span, and the complete event carries the outcome.
            if not event.kind.is_start:
                operations[event.operation] = event.status.value
            if event.duration_ms is not None:
                durations[event.operation] = durations.get(event.operation, 0.0) + event.duration_ms
                total += event.duration_ms
            if event.status is EventStatus.ERROR:
                errors.append(f"{event.operation}:{event.kind.value}")
            for key in ("memory_id", "memory_ids"):
                memory_ids.extend(_as_ids(event.metadata.get(key)))
        return TraceSummary(
            trace_id=trace.trace_id,
            kind=trace.kind,
            operation=trace.operation,
            started_at=trace.started_at,
            event_count=len(events),
            operations=operations,
            durations_ms=durations,
            total_duration_ms=total,
            errors=tuple(errors),
            memory_ids=tuple(dict.fromkeys(memory_ids)),
        )

    def get_metrics(self) -> OperationMetrics:
        """Aggregate every recorded event into counters.

        Each counter counts something that actually happened. Provider usage is
        counted only when the provider reported it (``usage_reported``); it is
        never derived from a token estimate or from a character count.
        """
        traces = _Counters()
        operations = _Counters()
        statuses = _Counters()
        memory = _Counters()
        provider = _Counters()
        context = _Counters()
        usage_reported = 0
        failed_traces = 0

        for trace in self._store.list_traces(limit=_UNBOUNDED):
            traces.add(trace.kind.value, 1)

        for event in self._store.all_events():
            statuses.add(event.status.value, 1)
            if event.kind.is_complete:
                operations.add(event.operation, 1)
            # -- memory lifecycle -------------------------------------------
            if event.kind is EventKind.MEMORY_WRITE:
                if event.status is EventStatus.OK:
                    memory.add("written", 1)
                    memory.add("created", _as_size(event.metadata.get("memory_ids")))
                else:
                    memory.add("writes_skipped", 1)
            elif event.kind is EventKind.MEMORY_UPDATE:
                memory.add("updated", 1)
            elif event.kind is EventKind.MEMORY_FORGET:
                memory.add("forgotten", 1)
            if event.kind in _LIFECYCLE_KINDS:
                memory.add("memory_ids_touched", _as_size(event.metadata.get("memory_ids")))
            elif event.kind is EventKind.MEMORY_CANDIDATE:
                memory.add("candidates", _as_count(event.metadata.get("candidate_count")))
            elif event.kind is EventKind.RETRIEVAL_COMPLETE:
                memory.add("recalled", _as_count(event.metadata.get("recalled")))
                memory.add(
                    "duplicate_candidates", _as_count(event.metadata.get("duplicate_candidates"))
                )
            # -- context ----------------------------------------------------
            elif event.kind is EventKind.CONTEXT_COMPILE_COMPLETE:
                context.add("candidates", _as_count(event.metadata.get("candidates")))
                context.add("selected", _as_count(event.metadata.get("selected")))
                context.add("suppressed", _as_count(event.metadata.get("suppressed")))
                context.add(
                    "conflicts_suppressed",
                    _as_count(event.metadata.get("conflicts_suppressed")),
                )
                if event.metadata.get("over_budget"):
                    context.add("over_budget", 1)
            # -- provider ---------------------------------------------------
            elif event.kind is EventKind.LLM_REQUEST:
                provider.add("calls", 1)
            elif event.kind is EventKind.LLM_RESPONSE:
                if event.status is EventStatus.ERROR:
                    provider.add("failures", 1)
                elif event.metadata.get("usage_reported"):
                    usage_reported += 1
                    provider.add("usage_reported", 1)
            elif event.kind is EventKind.MCP_REQUEST:
                provider.add("mcp_calls", 1)
            elif event.kind is EventKind.MCP_RESPONSE and event.status is EventStatus.ERROR:
                provider.add("mcp_failures", 1)

        for trace in self._store.list_traces(limit=_UNBOUNDED):
            if any(
                event.status is EventStatus.ERROR
                for event in self._store.get_events(trace.trace_id)
            ):
                failed_traces += 1

        return OperationMetrics(
            traces=traces.as_mapping(),
            operations=operations.as_mapping(),
            statuses=statuses.as_mapping(),
            memory=memory.as_mapping(),
            provider=provider.as_mapping(),
            context=context.as_mapping(),
            usage_reported=usage_reported,
            failed_traces=failed_traces,
        )

    # -- ids --------------------------------------------------------------

    def _new_trace_id(self) -> str:
        """Mint a new trace id."""
        if self._id_factory is not None:
            return self._id_factory()
        from app.domain.ids import UuidIdGenerator

        return f"{ID_PREFIX}-{UuidIdGenerator().next_id()}"


class _Counters:
    """A tiny counter map that omits untouched keys.

    Only counters that were actually incremented appear in the report: a zero
    that was never measured is not a measurement.
    """

    def __init__(self) -> None:
        self._values: dict[str, int] = {}

    def add(self, key: str, value: int) -> None:
        """Add ``value`` to ``key`` (no-op when the increment is zero)."""
        if value:
            self._values[key] = self._values.get(key, 0) + value

    def as_mapping(self) -> Mapping[str, int]:
        """The counters that were actually incremented, sorted by key."""
        return {key: self._values[key] for key in sorted(self._values)}


def _as_ids(value: Any) -> list[str]:
    """Coerce a metadata value into a list of memory ids."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _as_count(value: Any) -> int:
    """Coerce a metadata value into a non-negative count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _as_size(value: Any) -> int:
    """Coerce a metadata value into a size: a number, or the length of a list.

    Lifecycle events record ``memory_ids`` as a list; some record a bare count.
    Both are real measurements, so both are read.
    """
    if isinstance(value, (list, tuple)):
        return len(value)
    return _as_count(value)


#: Sentinel meaning "no limit" for the store's own aggregation query.
_UNBOUNDED = 1_000_000

#: The closing event kind of each opening event kind, used when a trace ends
#: while a span is still open.
_COMPLETE_FOR_START: Mapping[EventKind, EventKind] = {
    EventKind.RETRIEVAL_START: EventKind.RETRIEVAL_COMPLETE,
    EventKind.CONTEXT_COMPILE_START: EventKind.CONTEXT_COMPILE_COMPLETE,
    EventKind.LLM_REQUEST: EventKind.LLM_RESPONSE,
    EventKind.MCP_REQUEST: EventKind.MCP_RESPONSE,
}

#: The memory lifecycle kinds, which all carry the ids they touched.
_LIFECYCLE_KINDS = frozenset(
    {EventKind.MEMORY_WRITE, EventKind.MEMORY_UPDATE, EventKind.MEMORY_FORGET}
)


__all__ = [
    "ID_PREFIX",
    "SEQUENTIAL_ID_PREFIX",
    "NullTraceStore",
    "ObservabilityService",
    "OperationSpan",
    "SequentialIdFactory",
]
