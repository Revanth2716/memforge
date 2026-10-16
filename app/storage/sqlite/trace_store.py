"""Append-only persistence for traces and trace events (M6).

Traces live in the same SQLite file as everything else: no external
observability dependency, no second database, no telemetry service. Two tables,
written once, never updated -- which is what makes an audit trustworthy.

The store is deliberately dumb. It stores what it is given. Redaction is the
*service's* job (see :mod:`app.observability.service`), because a store that
silently rewrites its input is a store you cannot trust when auditing.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app.domain.errors import ObservabilityError
from app.domain.observability import EventKind, EventStatus, Trace, TraceEvent, TraceKind
from app.storage.sqlite.connection import open_connection, transaction
from app.storage.sqlite.migrations import run_migrations

_TRACE_COLUMNS = "trace_id, kind, operation, started_at, attributes"
_EVENT_COLUMNS = (
    "event_id, trace_id, sequence, kind, operation, status, timestamp, "
    "parent_id, duration_ms, metadata, detail"
)


def _dumps(payload: Mapping[str, object]) -> str:
    """Deterministic JSON: sorted keys, no whitespace drift between runs."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _loads(payload: str) -> dict[str, object]:
    """Decode a stored JSON mapping, tolerating a corrupt row."""
    try:
        decoded = json.loads(payload)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


class TraceStore(ABC):
    """Contract for trace persistence."""

    @abstractmethod
    def put_trace(self, trace: Trace) -> None:
        """Store a new trace (idempotent on re-insert of the same id)."""

    @abstractmethod
    def append_event(self, event: TraceEvent) -> None:
        """Append one event to an existing trace."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> Trace | None:
        """Return a trace, or ``None`` when it was never recorded."""

    @abstractmethod
    def get_events(self, trace_id: str) -> tuple[TraceEvent, ...]:
        """Every event of a trace, ordered by sequence."""

    @abstractmethod
    def list_traces(self, *, limit: int = 50, kind: TraceKind | None = None) -> tuple[Trace, ...]:
        """Recent traces, newest first (ties broken by id)."""

    @abstractmethod
    def count_events(self) -> int:
        """Total number of recorded events."""

    @abstractmethod
    def all_events(self, *, limit: int | None = None) -> tuple[TraceEvent, ...]:
        """Every recorded event, ordered by (trace_id, sequence)."""


class SqliteTraceStore(TraceStore):
    """Trace persistence backed by the MemForge SQLite file."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._initialised = False

    @property
    def db_path(self) -> Path:
        """The file this store writes to."""
        return self._db_path

    def initialize(self) -> None:
        """Ensure the trace tables exist.

        Idempotent, and called automatically before the first write, so a store
        handed a brand new path works without a separate migration step. It is
        the *same* migration runner the rest of MemForge uses -- there is no
        second schema mechanism.
        """
        with closing(open_connection(self._db_path)) as conn:
            run_migrations(conn)
        self._initialised = True

    def _ensure_schema(self) -> None:
        """Migrate once, lazily, before the first write."""
        if not self._initialised:
            self.initialize()

    def put_trace(self, trace: Trace) -> None:
        """Store a trace; re-inserting the same id is a no-op."""
        self._ensure_schema()
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            conn.execute(
                f"INSERT OR IGNORE INTO traces ({_TRACE_COLUMNS}) VALUES (?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.kind.value,
                    trace.operation,
                    trace.started_at.isoformat(),
                    _dumps(dict(trace.attributes)),
                ),
            )

    def append_event(self, event: TraceEvent) -> None:
        """Append one event; the ``(trace_id, sequence)`` pair must be unique."""
        self._ensure_schema()
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            try:
                conn.execute(
                    "INSERT INTO trace_events "
                    "(event_id, trace_id, sequence, kind, operation, status, "
                    "timestamp, parent_id, duration_ms, metadata, detail) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.trace_id,
                        event.sequence,
                        event.kind.value,
                        event.operation,
                        event.status.value,
                        event.timestamp.isoformat(),
                        event.parent_id,
                        event.duration_ms,
                        _dumps(dict(event.metadata)),
                        event.detail,
                    ),
                )
            except Exception as error:  # sqlite3.IntegrityError and friends
                raise ObservabilityError(
                    f"could not append event {event.event_id!r}: {error}"
                ) from error

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return a trace by id, or ``None``."""
        self._ensure_schema()
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute(
                f"SELECT {_TRACE_COLUMNS} FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return _trace_from_row(row) if row is not None else None

    def get_events(self, trace_id: str) -> tuple[TraceEvent, ...]:
        """Every event of a trace, ordered by sequence."""
        self._ensure_schema()
        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT {_EVENT_COLUMNS} FROM trace_events WHERE trace_id = ? ORDER BY sequence",
                (trace_id,),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def list_traces(self, *, limit: int = 50, kind: TraceKind | None = None) -> tuple[Trace, ...]:
        """Recent traces, newest first."""
        if limit < 1:
            raise ObservabilityError("limit must be >= 1")
        self._ensure_schema()
        query = f"SELECT {_TRACE_COLUMNS} FROM traces"
        params: list[object] = []
        if kind is not None:
            query += " WHERE kind = ?"
            params.append(kind.value)
        query += " ORDER BY started_at DESC, trace_id DESC LIMIT ?"
        params.append(limit)
        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(query, params).fetchall()
        return tuple(_trace_from_row(row) for row in rows)

    def count_events(self) -> int:
        """Total number of recorded events."""
        self._ensure_schema()
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM trace_events").fetchone()
        return int(row["total"]) if row is not None else 0

    def all_events(self, *, limit: int | None = None) -> tuple[TraceEvent, ...]:
        """Every recorded event, in a stable order."""
        self._ensure_schema()
        query = f"SELECT {_EVENT_COLUMNS} FROM trace_events ORDER BY trace_id, sequence"
        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(query).fetchall()
        events = [_event_from_row(row) for row in rows]
        selected = tuple(events[:limit]) if limit is not None else tuple(events)
        return selected


def _trace_from_row(row: Mapping[str, object]) -> Trace:
    """Rebuild a trace from a database row."""
    try:
        return Trace(
            trace_id=str(row["trace_id"]),
            kind=TraceKind(str(row["kind"])),
            operation=str(row["operation"]),
            started_at=datetime.fromisoformat(str(row["started_at"])),
            attributes=_loads(str(row["attributes"])),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ObservabilityError(f"corrupt trace row: {error}") from error


def _event_from_row(row: Mapping[str, object]) -> TraceEvent:
    """Rebuild an event from a database row."""
    duration = row["duration_ms"]
    parent = row["parent_id"]
    detail = row["detail"]
    try:
        return TraceEvent(
            trace_id=str(row["trace_id"]),
            event_id=str(row["event_id"]),
            sequence=_as_int(row["sequence"]),
            kind=EventKind(str(row["kind"])),
            operation=str(row["operation"]),
            status=EventStatus(str(row["status"])),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            parent_id=None if parent is None else str(parent),
            duration_ms=None if duration is None else float(duration),  # type: ignore[arg-type]
            metadata=_loads(str(row["metadata"])),
            detail=None if detail is None else str(detail),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ObservabilityError(f"corrupt trace event row: {error}") from error


def _as_int(value: object) -> int:
    """Coerce a stored integer column, reporting a corrupt row rather than lying."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{value!r} is not an integer column")
    return int(value)


__all__ = ["SqliteTraceStore", "TraceStore"]
