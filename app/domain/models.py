"""Core memory domain model.

Only the standard library is used here. The :class:`Memory` entity is a frozen
(value) dataclass: mutations always produce a new instance, which makes the
versioning rules easy to reason about and to test.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from app.domain.errors import MemoryValidationError


class MemoryType(enum.StrEnum):
    """Supported memory categories (M1 set).

    ``StrEnum`` based so values serialise directly to JSON and to SQLite text.
    """

    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    PROFILE = "PROFILE"
    EVENT = "EVENT"
    TASK = "TASK"
    OBSERVATION = "OBSERVATION"


class MemoryStatus(enum.StrEnum):
    """Lifecycle status of a memory.

    ``DELETED`` is a *soft* delete ("forget"): the memory disappears from
    normal reads/lists but its history is kept forever.
    """

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class VersionChange(enum.StrEnum):
    """Why a new version of a memory was written."""

    CREATED = "CREATED"
    UPDATED = "UPDATED"
    FORGOTTEN = "FORGOTTEN"


def _validate_unit_score(field: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryValidationError(f"{field} must be a number between 0.0 and 1.0")
    if not 0.0 <= float(value) <= 1.0:
        raise MemoryValidationError(f"{field} must be between 0.0 and 1.0 (got {value})")


@dataclass(frozen=True, slots=True)
class Memory:
    """A single stored memory.

    Attributes:
        id: Stable identifier (UUID string).
        content: The memory text itself.
        memory_type: Category of the memory.
        source: Optional provenance, e.g. ``user:revanth`` or ``cli:demo``.
        confidence: How sure we are the content is accurate, in ``[0, 1]``.
            This is *accuracy*, not importance -- the two are never conflated.
        importance: Caller-supplied importance in ``[0, 1]`` (an input to the
            computed importance score, see ``app.domain.intelligence``).
        status: Lifecycle status.
        created_at: Creation timestamp (timezone aware, UTC).
        updated_at: Last-modification timestamp (>= ``created_at``). Access
            telemetry deliberately does *not* touch this.
        version: Monotonically increasing version counter, starts at 1.
        access_count: How many times the memory was explicitly retrieved.
            Telemetry: it never creates a history version and never bumps
            ``version``.
        last_accessed_at: Timestamp of the most recent recorded access.
    """

    id: str
    content: str
    memory_type: MemoryType
    source: str | None
    confidence: float
    importance: float
    status: MemoryStatus
    created_at: datetime
    updated_at: datetime
    version: int
    access_count: int = 0
    last_accessed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise MemoryValidationError("Memory id must be a non-empty string")
        if not isinstance(self.content, str) or not self.content.strip():
            raise MemoryValidationError("Memory content must be a non-empty string")
        _validate_unit_score("confidence", self.confidence)
        _validate_unit_score("importance", self.importance)
        if self.version < 1:
            raise MemoryValidationError("Memory version must be >= 1")
        if self.updated_at < self.created_at:
            raise MemoryValidationError("updated_at must not be earlier than created_at")
        if isinstance(self.access_count, bool) or not isinstance(self.access_count, int):
            raise MemoryValidationError("access_count must be an integer")
        if self.access_count < 0:
            raise MemoryValidationError("access_count must be >= 0")
        if self.last_accessed_at is not None and self.last_accessed_at < self.created_at:
            raise MemoryValidationError("last_accessed_at must not be earlier than created_at")


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    """An immutable snapshot of a memory at a point in time.

    Versions are append-only: updating or forgetting a memory never rewrites or
    removes an earlier version, which is what makes history inspection possible.
    """

    memory_id: str
    version: int
    change: VersionChange
    content: str
    memory_type: MemoryType
    source: str | None
    confidence: float
    importance: float
    status: MemoryStatus
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise MemoryValidationError("memory_id must be a non-empty string")
        if self.version < 1:
            raise MemoryValidationError("Version number must be >= 1")


def build_version(
    memory: Memory,
    *,
    change: VersionChange,
    recorded_at: datetime,
) -> MemoryVersion:
    """Create a history snapshot from the current state of ``memory``."""
    return MemoryVersion(
        memory_id=memory.id,
        version=memory.version,
        change=change,
        content=memory.content,
        memory_type=memory.memory_type,
        source=memory.source,
        confidence=memory.confidence,
        importance=memory.importance,
        status=memory.status,
        recorded_at=recorded_at,
    )
