"""Persistence across process/instance boundaries.

The whole point of a *persistent* memory layer is that data outlives the
objects that wrote it, so these tests close everything down and start again.
"""

from __future__ import annotations

import dataclasses
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.models import MemoryStatus, MemoryType, VersionChange
from app.domain.relationships import RelationshipType
from app.domain.updates import MemoryUpdate
from app.main import create_app
from app.services.memory_service import MemoryService
from app.storage.base import MemoryQuery
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.migrations import applied_migrations, run_migrations
from app.storage.sqlite.repository import SqliteMemoryRepository

#: Every migration the current codebase knows how to apply.
EXPECTED_MIGRATIONS = [
    "0001_initial_memory_core",
    "0002_memory_intelligence",
    "0003_memory_retrieval",
    "0004_observability",
]


def test_data_survives_a_new_repository_instance(tmp_path: Path, db_path: Path) -> None:
    service = MemoryService(SqliteMemoryRepository(db_path))
    created = service.create_memory(
        content="persisted across restarts",
        memory_type=MemoryType.FACT,
        source="user:revanth",
        confidence=0.9,
        importance=0.6,
    )
    service.update_memory(created.id, MemoryUpdate(importance=0.95))
    service.forget_memory(created.id)
    del service  # nothing in memory survives this line

    # A brand new repository + service over the same file sees everything.
    reloaded = MemoryService(SqliteMemoryRepository(db_path))
    stored = reloaded.repository.get(created.id, include_deleted=True)

    assert stored is not None
    assert stored.content == "persisted across restarts"
    assert stored.source == "user:revanth"
    assert stored.confidence == 0.9
    assert stored.importance == 0.95
    assert stored.status is MemoryStatus.DELETED
    assert stored.version == 3

    history = reloaded.get_history(created.id)
    assert [version.change for version in history] == [
        VersionChange.CREATED,
        VersionChange.UPDATED,
        VersionChange.FORGOTTEN,
    ]
    assert [version.importance for version in history] == [0.6, 0.95, 0.95]


def test_data_survives_a_new_api_application(tmp_path: Path, db_path: Path) -> None:
    settings = Settings(database_path=db_path)

    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/memories",
            json={"content": "survive the app lifecycle", "memory_type": "TASK"},
        ).json()
        client.patch(f"/api/v1/memories/{created['id']}", json={"content": "updated"})

    # A second application instance, same database file.
    with TestClient(create_app(settings)) as client:
        response = client.get(f"/api/v1/memories/{created['id']}")
        assert response.status_code == 200
        body = response.json()
        assert body["content"] == "updated"
        assert body["version"] == 2

        history = client.get(f"/api/v1/memories/{created['id']}/history").json()
        assert [version["content"] for version in history["items"]] == [
            "survive the app lifecycle",
            "updated",
        ]


def test_schema_is_stable_across_restarts(tmp_path: Path, db_path: Path) -> None:
    SqliteMemoryRepository(db_path).initialize()

    with closing(open_connection(db_path)) as conn:
        assert run_migrations(conn) == []
        assert applied_migrations(conn) == EXPECTED_MIGRATIONS

    assert SqliteMemoryRepository(db_path).schema_version() == EXPECTED_MIGRATIONS[-1]


def test_separate_databases_are_isolated(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"

    MemoryService(SqliteMemoryRepository(first)).create_memory(
        content="only in first", memory_type=MemoryType.FACT
    )
    other = MemoryService(SqliteMemoryRepository(second))

    assert other.list_memories().total == 0
    assert other.repository.find(MemoryQuery()) == ([], 0)


DUPLICATE_A = "User prefers Python"
DUPLICATE_B = "Python is the user's preferred language"


def test_intelligence_data_survives_a_restart(tmp_path: Path, db_path: Path) -> None:
    service = MemoryService(SqliteMemoryRepository(db_path))
    first = service.create_memory(content=DUPLICATE_A, memory_type=MemoryType.FACT, importance=0.7)
    second = service.create_memory(content=DUPLICATE_B, memory_type=MemoryType.FACT, importance=0.2)

    service.find_duplicates(first.id)
    service.record_access(first.id)
    del service

    # A brand new service over the same file sees the intelligence artefacts.
    reloaded = MemoryService(SqliteMemoryRepository(db_path))
    stored = reloaded.get_memory(first.id)
    assert stored is not None
    assert stored.access_count == 1
    assert reloaded.score_memory(first.id).score > 0.0

    relationships = reloaded.get_relationships(first.id)
    assert [rel.relationship for rel in relationships] == [RelationshipType.DUPLICATE_OF]
    assert relationships[0].target_id == second.id


def test_migrations_upgrade_an_existing_database(db_path: Path) -> None:
    """M2, M3 and M6 migrations upgrade a database created by M1 in place."""
    SqliteMemoryRepository(db_path)
    assert SqliteMemoryRepository(db_path).schema_version() == "0004_observability"

    with closing(open_connection(db_path)) as conn:
        tables = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        memory_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        indexes = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }

    assert "memory_relationships" in tables
    assert "memory_embeddings" in tables
    assert {"access_count", "last_accessed_at"} <= memory_columns
    assert {"idx_embeddings_provider", "idx_embeddings_memory"} <= indexes


def test_domain_objects_are_plain_data(tmp_path: Path, db_path: Path) -> None:
    service = MemoryService(SqliteMemoryRepository(db_path))
    created = service.create_memory(content="plain data", memory_type=MemoryType.PROFILE)

    dumped = dataclasses.asdict(created)
    assert dumped["content"] == "plain data"
    assert dumped["memory_type"] is MemoryType.PROFILE
    assert dumped["status"] is MemoryStatus.ACTIVE
