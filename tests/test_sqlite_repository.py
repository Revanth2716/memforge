"""SQLite repository tests (data access only, no service layer)."""

from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from app.domain.errors import MemoryAlreadyExistsError, MemoryNotFoundError, RepositoryError
from app.domain.models import MemoryStatus, MemoryType, VersionChange
from app.storage.base import MemoryQuery
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START, fixed_clock, make_memory

UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"


class TestInsertAndGet:
    def test_insert_then_get_round_trips_every_field(
        self, repository: SqliteMemoryRepository
    ) -> None:
        memory = make_memory(
            id="78888888-8888-4888-8888-888888888888",
            content="User works from Hyderabad on Mondays.",
            memory_type=MemoryType.EVENT,
            source="calendar:import",
            confidence=0.42,
            importance=0.99,
        )
        written_version = repository.insert(memory, change=VersionChange.CREATED)

        fetched = repository.get(memory.id)
        assert fetched == memory
        assert written_version.change is VersionChange.CREATED
        assert written_version.version == 1
        assert written_version.content == memory.content

    def test_get_unknown_id_returns_none(self, repository: SqliteMemoryRepository) -> None:
        assert repository.get(UNKNOWN_ID) is None

    def test_duplicate_insert_is_rejected(self, repository: SqliteMemoryRepository) -> None:
        memory = make_memory()
        repository.insert(memory, change=VersionChange.CREATED)
        with pytest.raises(MemoryAlreadyExistsError):
            repository.insert(memory, change=VersionChange.CREATED)

    def test_deleted_memory_is_hidden_unless_requested(
        self, repository: SqliteMemoryRepository
    ) -> None:
        memory = make_memory()
        repository.insert(memory, change=VersionChange.CREATED)
        repository.save(
            dataclasses.replace(memory, status=MemoryStatus.DELETED, version=2),
            change=VersionChange.FORGOTTEN,
        )

        assert repository.get(memory.id) is None
        assert repository.get(memory.id, include_deleted=True) is not None

    @pytest.mark.parametrize(
        "content",
        [
            "'; DROP TABLE memories; --",
            'He said "hello" and left.',
            "100% sure? yes & no.",
            "Unicode: café 日本語",
            "line1\nline2\ttabbed",
        ],
    )
    def test_content_round_trips_safely(
        self, repository: SqliteMemoryRepository, content: str
    ) -> None:
        memory = make_memory(content=content)
        repository.insert(memory, change=VersionChange.CREATED)

        fetched = repository.get(memory.id)
        assert fetched is not None
        assert fetched.content == content

        # The injection attempt must not have destroyed the table.
        assert repository.ping() is True
        assert repository.find(MemoryQuery(limit=10))[1] == 1


class TestFind:
    @pytest.fixture
    def seeded(self, repository: SqliteMemoryRepository) -> SqliteMemoryRepository:
        clock = fixed_clock(FIXED_START)
        for index in range(3):
            stamp = clock.now() + timedelta(minutes=index)
            repository.insert(
                make_memory(
                    id=f"{index + 1}",
                    content=f"memory {index}",
                    memory_type=MemoryType.TASK,
                    created_at=stamp,
                    updated_at=stamp,
                ),
                change=VersionChange.CREATED,
            )
        return repository

    def test_default_returns_all_newest_first(self, seeded: SqliteMemoryRepository) -> None:
        items, total = seeded.find(MemoryQuery())
        assert total == 3
        assert [item.content for item in items] == ["memory 2", "memory 1", "memory 0"]

    def test_pagination(self, seeded: SqliteMemoryRepository) -> None:
        first_page, total = seeded.find(MemoryQuery(limit=2, offset=0))
        second_page, _ = seeded.find(MemoryQuery(limit=2, offset=2))

        assert total == 3
        assert [item.content for item in first_page] == ["memory 2", "memory 1"]
        assert [item.content for item in second_page] == ["memory 0"]

    def test_offset_beyond_end_returns_empty_page(self, seeded: SqliteMemoryRepository) -> None:
        items, total = seeded.find(MemoryQuery(limit=10, offset=10))
        assert items == []
        assert total == 3

    def test_filter_by_type(self, seeded: SqliteMemoryRepository) -> None:
        items, total = seeded.find(MemoryQuery(memory_type=MemoryType.TASK))
        assert total == 3
        assert len(items) == 3

    def test_filter_by_status(self, seeded: SqliteMemoryRepository) -> None:
        items, total = seeded.find(MemoryQuery(status=MemoryStatus.ARCHIVED))
        assert items == []
        assert total == 0

    def test_invalid_query_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryQuery(limit=0)
        with pytest.raises(ValueError):
            MemoryQuery(offset=-1)


class TestSaveAndVersions:
    def test_db_path_property(self, repository: SqliteMemoryRepository, db_path: Path) -> None:
        assert repository.db_path == db_path

    def test_naive_timestamps_are_read_as_utc(
        self, repository: SqliteMemoryRepository, db_path: Path
    ) -> None:
        memory = make_memory()
        repository.insert(memory, change=VersionChange.CREATED)

        # Simulate a row written by another tool without timezone information.
        with closing(open_connection(db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                ("2026-01-01T09:00:00.000000", "2026-01-01T09:00:00.000000", memory.id),
            )
            conn.execute(
                "INSERT INTO memory_versions (memory_id, version, change, content, memory_type,"
                " source, confidence, importance, status, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    2,
                    "UPDATED",
                    memory.content,
                    memory.memory_type.value,
                    memory.source,
                    memory.confidence,
                    memory.importance,
                    memory.status.value,
                    "2026-01-01T09:00:00.000000",
                ),
            )
            conn.execute("COMMIT")

        stored = repository.get(memory.id)
        assert stored is not None
        assert stored.created_at.tzinfo is not None
        assert stored.created_at == FIXED_START
        assert repository.versions(memory.id)[1].recorded_at == FIXED_START

    def test_integrity_errors_are_translated_to_domain_errors(
        self, repository: SqliteMemoryRepository, db_path: Path
    ) -> None:
        memory = make_memory()
        repository.insert(memory, change=VersionChange.CREATED)

        # Pre-create version 2 so the UPDATE/INSERT in save() hits the unique
        # constraint on (memory_id, version).
        with closing(open_connection(db_path)) as conn:
            conn.execute(
                "INSERT INTO memory_versions (memory_id, version, change, content, memory_type,"
                " source, confidence, importance, status, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    2,
                    "UPDATED",
                    "clashing version",
                    memory.memory_type.value,
                    None,
                    0.5,
                    0.5,
                    "ACTIVE",
                    "2026-01-01T09:00:00.000000+00:00",
                ),
            )

        with pytest.raises(MemoryAlreadyExistsError):
            repository.save(
                dataclasses.replace(memory, content="new", version=2),
                change=VersionChange.UPDATED,
            )

    def test_integrity_error_mapping(self) -> None:
        unique = SqliteMemoryRepository._integrity_error(
            sqlite3.IntegrityError("UNIQUE constraint failed: memories.id"), "x"
        )
        assert isinstance(unique, MemoryAlreadyExistsError)

        other = SqliteMemoryRepository._integrity_error(
            sqlite3.IntegrityError("FOREIGN KEY constraint failed"), "x"
        )
        assert isinstance(other, RepositoryError)

    def test_save_updates_row_and_appends_version(self, repository: SqliteMemoryRepository) -> None:
        original = make_memory()
        repository.insert(original, change=VersionChange.CREATED)

        updated = dataclasses.replace(original, content="updated", importance=0.1, version=2)
        written_version = repository.save(updated, change=VersionChange.UPDATED)

        assert repository.get(original.id) == updated
        assert written_version.change is VersionChange.UPDATED
        assert written_version.version == 2

    def test_save_requires_consecutive_versions(self, repository: SqliteMemoryRepository) -> None:
        original = make_memory()
        repository.insert(original, change=VersionChange.CREATED)

        jumped = dataclasses.replace(original, content="x", version=5)
        with pytest.raises(RepositoryError):
            repository.save(jumped, change=VersionChange.UPDATED)

    def test_save_unknown_memory_raises_not_found(self, repository: SqliteMemoryRepository) -> None:
        ghost = make_memory(id=UNKNOWN_ID, version=2)
        with pytest.raises(MemoryNotFoundError):
            repository.save(ghost, change=VersionChange.UPDATED)

    def test_versions_are_ordered_oldest_first(self, repository: SqliteMemoryRepository) -> None:
        original = make_memory()
        repository.insert(original, change=VersionChange.CREATED)
        for number in (2, 3):
            bumped = dataclasses.replace(original, content=f"v{number}", version=number)
            repository.save(bumped, change=VersionChange.UPDATED)

        versions = repository.versions(original.id)
        assert [version.version for version in versions] == [1, 2, 3]
        assert [version.content for version in versions] == [original.content, "v2", "v3"]
        assert all(version.change is VersionChange.UPDATED for version in versions[1:])

    def test_versions_for_unknown_memory_is_empty(self, repository: SqliteMemoryRepository) -> None:
        assert repository.versions(UNKNOWN_ID) == []

    def test_history_survives_a_forget(self, repository: SqliteMemoryRepository) -> None:
        original = make_memory()
        repository.insert(original, change=VersionChange.CREATED)
        repository.save(
            dataclasses.replace(original, content="middle", version=2),
            change=VersionChange.UPDATED,
        )
        repository.save(
            dataclasses.replace(original, content="middle", status=MemoryStatus.DELETED, version=3),
            change=VersionChange.FORGOTTEN,
        )

        versions = repository.versions(original.id)
        assert [version.change for version in versions] == [
            VersionChange.CREATED,
            VersionChange.UPDATED,
            VersionChange.FORGOTTEN,
        ]
        assert versions[-1].status is MemoryStatus.DELETED
        assert repository.get(original.id, include_deleted=True).version == 3  # type: ignore[union-attr]
