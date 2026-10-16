"""Shared test helpers (importable as ``tests.support``)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.ids import SequentialIdGenerator
from app.domain.models import (
    Memory,
    MemoryStatus,
    MemoryType,
)
from app.domain.time import ManualClock

#: Fixed, timezone-aware instant used across the suite.
FIXED_START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC)


def fixed_clock(start: datetime | None = None) -> ManualClock:
    """A clock frozen at :data:`FIXED_START`."""
    return ManualClock(start or FIXED_START)


def sequential_ids(prefix: str = "test") -> SequentialIdGenerator:
    """Deterministic identifier generator."""
    return SequentialIdGenerator(prefix=prefix)


def make_memory(**overrides: object) -> Memory:
    """Build a valid :class:`Memory` with sensible defaults."""
    defaults: dict[str, object] = {
        "id": "11111111-1111-5111-8111-111111111111",
        "content": "User likes strong coffee.",
        "memory_type": MemoryType.PREFERENCE,
        "source": "user:test",
        "confidence": 0.8,
        "importance": 0.5,
        "status": MemoryStatus.ACTIVE,
        "created_at": FIXED_START,
        "updated_at": FIXED_START,
        "version": 1,
    }
    defaults.update(overrides)
    return Memory(**defaults)  # type: ignore[arg-type]
