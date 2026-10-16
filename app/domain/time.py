"""Time source abstraction.

All timestamps are timezone-aware UTC datetimes. The clock is injectable so
that behaviour stays deterministic in tests and in the CLI demo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Returns the current time."""

    def now(self) -> datetime:
        """The current instant as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """Real wall clock (used in production and by the REST API)."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """Deterministic clock for demos and tests.

    ``now()`` always returns the same instant until :meth:`advance` is called.
    """

    def __init__(self, start: datetime | None = None) -> None:
        if start is None:
            start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def advance(
        self,
        *,
        days: float = 0,
        hours: float = 0,
        minutes: float = 0,
        seconds: float = 0,
    ) -> datetime:
        """Move the clock forward and return the new instant."""
        self._now = self._now + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return self._now
