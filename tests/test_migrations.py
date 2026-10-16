"""Schema initialisation, migrations and SQLite constraint tests."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.domain.models import MemoryType
from app.services.memory_service import MemoryService
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.migrations import applied_migrations, run_migrations, schema_version
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import fixed_clock, make_memory, sequential_ids

#: Every migration the current codebase knows how to apply.
EXPECTED_MIGRATIONS = [
    "0001_initial_memory_core",
    "0002_memory_intelligence",
    "0003_memory_retrieval",
    "0004_observability",
]

_VERSION_INSERT = (
    "INSERT INTO memory_versions (memory_id, version, change, content, memory_type, "
    "source, confidence, importance, status, recorded_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def test_fresh_database_has_expected_objects(db_path: Path) -> None:
    SqliteMemoryRepository(db_path).initialize()

    with closing(open_connection(db_path)) as conn:
        tables = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        indexes = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }

    assert {"memories", "memory_versions", "_memforge_migrations"} <= tables
    assert {
        "idx_memories_status_type",
        "idx_memories_created_at",
        "idx_memory_versions_lookup",
    } <= indexes


def test_migrations_are_recorded_and_idempotent(db_path: Path) -> None:
    SqliteMemoryRepository(db_path).initialize()

    with closing(open_connection(db_path)) as conn:
        assert applied_migrations(conn) == EXPECTED_MIGRATIONS
        assert schema_version(conn) == EXPECTED_MIGRATIONS[-1]
        # Re-running applies nothing new.
        assert run_migrations(conn) == []
        assert applied_migrations(conn) == EXPECTED_MIGRATIONS

    # A second repository instance over the same file is also a no-op.
    assert SqliteMemoryRepository(db_path).schema_version() == EXPECTED_MIGRATIONS[-1]


def test_foreign_keys_are_enforced(db_path: Path, repository: SqliteMemoryRepository) -> None:
    with closing(open_connection(db_path)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            _VERSION_INSERT,
            (
                "00000000-0000-4000-8000-000000000000",  # unknown memory_id
                1,
                "CREATED",
                "orphan",
                "FACT",
                None,
                0.5,
                0.5,
                "ACTIVE",
                "2026-01-01T00:00:00.000000+00:00",
            ),
        )


def test_duplicate_version_numbers_are_rejected(
    db_path: Path, repository: SqliteMemoryRepository, service: MemoryService
) -> None:
    memory = service.create_memory(content="hello", memory_type=MemoryType.OBSERVATION)

    with closing(open_connection(db_path)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            _VERSION_INSERT,
            (
                memory.id,
                1,
                "UPDATED",
                "duplicate",
                "OBSERVATION",
                None,
                0.5,
                0.5,
                "ACTIVE",
                "2026-01-01T00:00:00.000000+00:00",
            ),
        )


def test_empty_database_reports_no_schema_version(db_path: Path) -> None:
    with closing(open_connection(db_path)) as conn:
        assert schema_version(conn) is None
        assert applied_migrations(conn) == []


def test_parent_directories_are_created(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "mem.sqlite"
    SqliteMemoryRepository(nested).initialize()
    assert nested.exists()


def test_ping(tmp_path: Path) -> None:
    assert SqliteMemoryRepository(tmp_path / "ping.sqlite").ping() is True


def test_deterministic_helpers_are_usable() -> None:
    assert fixed_clock().now().year == 2026
    assert sequential_ids().next_id()
    assert make_memory().version == 1
