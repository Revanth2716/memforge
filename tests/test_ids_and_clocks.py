"""Tests for the injectable id generator and clock ports."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.domain.ids import SequentialIdGenerator, UuidIdGenerator
from app.domain.time import ManualClock, SystemClock


class TestIdGenerators:
    def test_uuid_generator_produces_valid_unique_ids(self) -> None:
        generator = UuidIdGenerator()
        ids = {generator.next_id() for _ in range(50)}
        assert len(ids) == 50
        for value in ids:
            assert uuid.UUID(value)  # parses without error

    def test_sequential_generator_is_deterministic(self) -> None:
        first = [SequentialIdGenerator().next_id() for _ in range(3)]
        second = [SequentialIdGenerator().next_id() for _ in range(3)]

        # Two independent generators produce the same sequence...
        assert first == second
        # ...and each generator produces distinct ids within that sequence.
        generator = SequentialIdGenerator()
        assert len({generator.next_id() for _ in range(5)}) == 5

    def test_sequential_generator_ids_are_valid_uuids(self) -> None:
        value = SequentialIdGenerator().next_id()
        assert uuid.UUID(value)

    def test_sequential_prefixes_do_not_collide(self) -> None:
        a = SequentialIdGenerator(prefix="demo").next_id()
        b = SequentialIdGenerator(prefix="test").next_id()
        assert a != b


class TestClocks:
    def test_manual_clock_is_frozen_until_advanced(self) -> None:
        clock = ManualClock(datetime(2026, 1, 1, tzinfo=UTC))
        first = clock.now()
        assert clock.now() == first
        clock.advance(minutes=5)
        assert clock.now() == first.replace(minute=5)

    def test_manual_clock_defaults_to_utc(self) -> None:
        assert ManualClock().now().tzinfo is not None

    def test_system_clock_is_utc_now(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert abs((datetime.now(UTC) - now).total_seconds()) < 5
