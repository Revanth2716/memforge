"""Observability domain for MemForge (M6).

Two questions, two models:

1. *What happened?* — :class:`Trace` and :class:`TraceEvent`. An append-only,
   deterministic tree of operational events, correlated by one ``trace_id``.
2. *How well did it work?* — :class:`TraceSummary` and :class:`OperationMetrics`,
   counters derived from those events by counting, never by guessing.

What is deliberately **not** recorded
-------------------------------------
- **Chain-of-thought.** Only what operation ran, on how many items, with what
  outcome and how long. No reasoning, no deliberation, no model scratchpad.
- **Prompts and answers.** Not the text. Only sizes (``input_characters``,
  ``output_characters``), so a payload can be correlated without the payload.
- **Secrets.** Nothing reaches a :class:`TraceEvent` that was not already safe:
  :func:`safe_metadata` runs every mapping through the M5 redaction helpers and
  through a deny-list of credential-shaped keys, so an accidental
  ``metadata={"api_key": ...}`` is dropped rather than persisted.

Determinism
-----------
Events carry a monotonically increasing ``sequence`` per trace, so ordering is
total and identical for identical execution. Timestamps come from an injected
clock; duration is measured with an injectable timer and is never asserted on in
a test.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

from app.domain.errors import ObservabilityError
from app.domain.llm import REDACTED, redact_mapping, redact_text

#: Keys that must never be recorded, whatever their value.
FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "auth",
        "bearer",
        "token",
        "access_token",
        "secret",
        "password",
        "passwd",
        "credential",
        "credentials",
        "private_key",
        "session",
        "cookie",
        "set-cookie",
        "x-api-key",
        "prompt",
        "system_prompt",
        "completion",
        "response_text",
        "answer",
        "content",
        "chain_of_thought",
        "reasoning",
    }
)

#: Decimal places kept in reported ratios.
_PRECISION = 6


class EventKind(enum.StrEnum):
    """Every operation MemForge records.

    The lifecycle an agent turn emits is always, in this order:
    ``REQUEST`` -> ``MEMORY_GATE`` -> ``RETRIEVAL_START/COMPLETE`` ->
    ``CONTEXT_COMPILE_START/COMPLETE`` -> ``LLM_REQUEST/RESPONSE`` ->
    ``MEMORY_CANDIDATE`` -> ``MEMORY_WRITE`` -> ``RESPONSE``.
    """

    REQUEST = "request"
    RESPONSE = "response"
    MEMORY_GATE = "memory_gate"
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    CONTEXT_COMPILE_START = "context_compile_start"
    CONTEXT_COMPILE_COMPLETE = "context_compile_complete"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    MEMORY_CANDIDATE = "memory_candidate"
    MEMORY_WRITE = "memory_write"
    MEMORY_UPDATE = "memory_update"
    MEMORY_FORGET = "memory_forget"
    MCP_REQUEST = "mcp_request"
    MCP_RESPONSE = "mcp_response"
    ERROR = "error"

    @property
    def is_start(self) -> bool:
        """True for the opening event of an operation."""
        return self in _START_KINDS

    @property
    def is_complete(self) -> bool:
        """True for the closing event of an operation."""
        return self in _COMPLETE_KINDS

    @property
    def operation(self) -> str:
        """The operation this event belongs to (start and complete share it)."""
        return _OPERATIONS[self]


_START_KINDS = frozenset(
    {
        EventKind.RETRIEVAL_START,
        EventKind.CONTEXT_COMPILE_START,
        EventKind.LLM_REQUEST,
        EventKind.MCP_REQUEST,
    }
)

_COMPLETE_KINDS = frozenset(
    {
        EventKind.RESPONSE,
        EventKind.RETRIEVAL_COMPLETE,
        EventKind.CONTEXT_COMPILE_COMPLETE,
        EventKind.LLM_RESPONSE,
        EventKind.MCP_RESPONSE,
    }
)

#: Event kind -> the operation it belongs to.
_OPERATIONS: Mapping[EventKind, str] = MappingProxyType(
    {
        EventKind.REQUEST: "agent.request",
        EventKind.MEMORY_GATE: "agent.memory_gate",
        EventKind.RETRIEVAL_START: "memory.retrieval",
        EventKind.RETRIEVAL_COMPLETE: "memory.retrieval",
        EventKind.CONTEXT_COMPILE_START: "context.compile",
        EventKind.CONTEXT_COMPILE_COMPLETE: "context.compile",
        EventKind.LLM_REQUEST: "llm.generate",
        EventKind.LLM_RESPONSE: "llm.generate",
        EventKind.MEMORY_CANDIDATE: "agent.memory_candidate",
        EventKind.MEMORY_WRITE: "memory.write",
        EventKind.MEMORY_UPDATE: "memory.update",
        EventKind.MEMORY_FORGET: "memory.forget",
        EventKind.MCP_REQUEST: "mcp.request",
        EventKind.MCP_RESPONSE: "mcp.request",
        EventKind.ERROR: "error",
        EventKind.RESPONSE: "agent.response",
    }
)


class EventStatus(enum.StrEnum):
    """Outcome of an event."""

    #: The operation ran and produced its result.
    OK = "ok"
    #: The operation ran and decided not to act (e.g. storage not permitted).
    SKIPPED = "skipped"
    #: The operation failed.
    ERROR = "error"


class TraceKind(enum.StrEnum):
    """What started a trace."""

    AGENT_REQUEST = "agent_request"
    MCP_REQUEST = "mcp_request"
    API_REQUEST = "api_request"
    EVALUATION_CASE = "evaluation_case"
    CLI = "cli"


@dataclass(frozen=True, slots=True)
class Trace:
    """One correlated unit of work."""

    trace_id: str
    kind: TraceKind
    operation: str
    started_at: datetime
    #: Safe, structured attributes of the trace itself.
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id or not self.trace_id.strip():
            raise ObservabilityError("trace_id must be a non-empty string")
        if not self.operation or not self.operation.strip():
            raise ObservabilityError("operation must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "trace_id": self.trace_id,
            "kind": self.kind.value,
            "operation": self.operation,
            "started_at": self.started_at.isoformat(),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One event inside a trace.

    ``sequence`` makes the order total and reproducible; ``parent_id`` points at
    the event that opened the current operation, which is what turns a flat list
    into an inspectable tree.
    """

    trace_id: str
    event_id: str
    sequence: int
    kind: EventKind
    operation: str
    status: EventStatus
    timestamp: datetime
    parent_id: str | None = None
    #: Measured wall-clock duration in milliseconds (``None`` when not timed).
    duration_ms: float | None = None
    #: Structured, redacted operational metadata.
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Human-readable reason for a non-OK status.
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ObservabilityError("event_id must be a non-empty string")
        if self.sequence < 0:
            raise ObservabilityError("sequence must be >= 0")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ObservabilityError("duration_ms must be >= 0")

    @property
    def is_terminal(self) -> bool:
        """True when this event closes its operation."""
        return self.kind.is_complete or self.status is EventStatus.ERROR

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "operation": self.operation,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """Everything an operator needs from one trace, derived by counting events."""

    trace_id: str
    kind: TraceKind
    operation: str
    started_at: datetime
    event_count: int
    #: Operation -> terminal status of that operation.
    operations: Mapping[str, str]
    #: Operation -> total measured duration (sum of its complete events).
    durations_ms: Mapping[str, float]
    total_duration_ms: float
    errors: tuple[str, ...]
    memory_ids: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        """True when no operation ended in an error."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serializable form (stable key order)."""
        return {
            "trace_id": self.trace_id,
            "kind": self.kind.value,
            "operation": self.operation,
            "started_at": self.started_at.isoformat(),
            "event_count": self.event_count,
            "operations": dict(sorted(self.operations.items())),
            "durations_ms": {
                name: round(value, 3) for name, value in sorted(self.durations_ms.items())
            },
            "total_duration_ms": round(self.total_duration_ms, 3),
            "errors": list(self.errors),
            "memory_ids": list(self.memory_ids),
            "succeeded": self.succeeded,
        }


@dataclass(frozen=True, slots=True)
class OperationMetrics:
    """Counters derived from recorded events.

    Every counter is a count of something that actually happened. Nothing here is
    a rate, a ratio, an estimate or a benchmark.
    """

    #: Traces recorded, by kind (``{"agent_request": 3}``).
    traces: Mapping[str, int]
    #: Completed operations by name (``{"memory.retrieval": 3}``).
    operations: Mapping[str, int]
    #: Terminal statuses across all events (``{"ok": 9, "skipped": 1}``).
    statuses: Mapping[str, int]
    #: Memory lifecycle counters (each one counts a real write).
    memory: Mapping[str, int]
    #: Provider counters (``{"calls": 3, "failures": 0, "usage_reported": 0}``).
    provider: Mapping[str, int]
    #: Context counters (``{"candidates": 12, "selected": 5, "suppressed": 7}``).
    context: Mapping[str, int]
    #: Number of provider calls that reported usage (never inferred).
    usage_reported: int
    #: Number of traces that ended with at least one error.
    failed_traces: int

    def to_dict(self) -> dict[str, Any]:
        """Serializable form (stable key order) for dashboards and diffs."""
        return {
            "traces": dict(sorted(self.traces.items())),
            "operations": dict(sorted(self.operations.items())),
            "statuses": dict(sorted(self.statuses.items())),
            "memory": dict(sorted(self.memory.items())),
            "provider": dict(sorted(self.provider.items())),
            "context": dict(sorted(self.context.items())),
            "usage_reported": self.usage_reported,
            "failed_traces": self.failed_traces,
        }


def safe_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    secrets: Sequence[str | None] = (),
) -> dict[str, Any]:
    """Return a log-safe copy of a metadata mapping.

    Three passes, in order:

    1. **Drop** any key that is credential- or content-shaped (see
       :data:`FORBIDDEN_METADATA_KEYS`), because a trace must never hold a
       secret or a prompt.
    2. **Redact** auth-bearing headers through the M5 helper.
    3. **Redact** the literal values of any secret the caller knows about.
    """
    if not metadata:
        return {}
    filtered = {
        key: value
        for key, value in metadata.items()
        if str(key).lower() not in FORBIDDEN_METADATA_KEYS
    }
    return redact_mapping(filtered, *secrets)


def safe_detail(detail: str | None, *, secrets: Sequence[str | None] = ()) -> str | None:
    """Return a log-safe event detail."""
    if detail is None:
        return None
    return redact_text(detail, *secrets)


def redact_secrets_in(payload: Any, *secrets: str | None) -> Any:
    """Recursively redact secret values from any JSON-shaped payload."""
    if isinstance(payload, str):
        return redact_text(payload, *secrets)
    if isinstance(payload, Mapping):
        return redact_mapping(payload, *secrets)
    if isinstance(payload, (list, tuple)):
        return [redact_secrets_in(item, *secrets) for item in payload]
    return payload


__all__ = [
    "FORBIDDEN_METADATA_KEYS",
    "REDACTED",
    "EventKind",
    "EventStatus",
    "OperationMetrics",
    "Trace",
    "TraceEvent",
    "TraceKind",
    "TraceSummary",
    "redact_secrets_in",
    "safe_detail",
    "safe_metadata",
]
