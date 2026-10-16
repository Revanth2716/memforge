"""Storage-level tests for the M2 intelligence support.

Covers access telemetry, the relationship table (constraints, indexes,
idempotency) and the active-corpus scan used by detection.
"""

from __future__ import annotations

import dataclasses
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from app.domain.errors import (
    InvalidRelationshipError,
    MemoryNotFoundError,
    RelationshipExistsError,
)
from app.domain.models import Memory, MemoryStatus, MemoryType, VersionChange
from app.domain.relationships import MemoryRelationship, RelationshipType
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START, make_memory

UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"
OTHER_ID = "33333333-3333-4333-8333-333333333333"


def _mutate(memory: Memory | None, **changes: object) -> Memory:
    """Fetch a memory and apply changes, bumping its version by one."""
    if memory is None:  # pragma: no cover - defensive
        raise AssertionError("the memory is expected to exist")
    return dataclasses.replace(memory, version=memory.version + 1, **changes)  # type: ignore[arg-type]


class TestAccessTelemetry:
    def test_access_is_recorded_and_persisted(
        self, repository: SqliteMemoryRepository, db_path: Path
    ) -> None:
        memory_id = repository.insert(make_memory(), change=VersionChange.CREATED).memory_id
        assert repository.get(memory_id).access_count == 0  # type: ignore[union-attr]

        for expected in (1, 2, 3):
            touched = repository.record_access(
                memory_id, at=FIXED_START + timedelta(hours=expected)
            )
            assert touched.access_count == expected
            assert touched.last_accessed_at == FIXED_START + timedelta(hours=expected)

        # The counters are durable.
        reloaded = repository.get(memory_id)
        assert reloaded is not None
        assert reloaded.access_count == 3
        assert reloaded.last_accessed_at == FIXED_START + timedelta(hours=3)

    def test_access_is_telemetry_not_an_edit(
        self, repository: SqliteMemoryRepository, db_path: Path
    ) -> None:
        memory = make_memory()
        memory_id = repository.insert(memory, change=VersionChange.CREATED).memory_id
        repository.record_access(memory_id, at=FIXED_START + timedelta(days=5))

        stored = repository.get(memory_id)
        assert stored is not None
        assert stored.version == 1
        assert stored.updated_at == memory.updated_at
        assert stored.created_at == memory.created_at
        assert stored.content == memory.content
        # Telemetry is not part of the history snapshots.
        assert len(repository.versions(memory_id)) == 1

    def test_access_is_refused_for_unknown_memories(
        self, repository: SqliteMemoryRepository
    ) -> None:
        with pytest.raises(MemoryNotFoundError):
            repository.record_access(UNKNOWN_ID, at=FIXED_START)

    def test_access_is_refused_for_forgotten_memories(
        self, repository: SqliteMemoryRepository
    ) -> None:
        memory_id = repository.insert(make_memory(), change=VersionChange.CREATED).memory_id
        repository.save(
            _mutate(repository.get(memory_id), status=MemoryStatus.DELETED),
            change=VersionChange.FORGOTTEN,
        )
        with pytest.raises(MemoryNotFoundError):
            repository.record_access(memory_id, at=FIXED_START)


class TestActiveMemories:
    def test_only_active_memories_are_returned(self, repository: SqliteMemoryRepository) -> None:
        active = repository.insert(make_memory(), change=VersionChange.CREATED).memory_id
        archived = repository.insert(
            make_memory(id=OTHER_ID), change=VersionChange.CREATED
        ).memory_id
        repository.save(
            _mutate(repository.get(archived), status=MemoryStatus.ARCHIVED),
            change=VersionChange.UPDATED,
        )
        forgotten = repository.insert(
            make_memory(id=UNKNOWN_ID), change=VersionChange.CREATED
        ).memory_id
        repository.save(
            _mutate(repository.get(forgotten), status=MemoryStatus.DELETED),
            change=VersionChange.FORGOTTEN,
        )

        ids = [memory.id for memory in repository.active_memories()]
        assert ids == [active]


class TestRelationships:
    def _link(self, repository: SqliteMemoryRepository, **overrides: object) -> MemoryRelationship:
        defaults: dict[str, object] = {
            "source_id": "11111111-1111-5111-8111-111111111111",
            "target_id": OTHER_ID,
            "relationship": RelationshipType.DUPLICATE_OF,
            "similarity": 0.9,
            "created_at": FIXED_START,
        }
        defaults.update(overrides)
        return repository.add_relationship(
            MemoryRelationship(**defaults)  # type: ignore[arg-type]
        )

    @pytest.fixture
    def two_memories(self, repository: SqliteMemoryRepository) -> SqliteMemoryRepository:
        repository.insert(make_memory(), change=VersionChange.CREATED)
        repository.insert(make_memory(id=OTHER_ID), change=VersionChange.CREATED)
        return repository

    def test_relationship_round_trips(self, two_memories: SqliteMemoryRepository) -> None:
        stored = self._link(two_memories)
        assert stored.relationship is RelationshipType.DUPLICATE_OF
        assert stored.similarity == 0.9
        assert stored.created_at == FIXED_START

    def test_relationship_is_visible_from_both_sides(
        self, two_memories: SqliteMemoryRepository
    ) -> None:
        self._link(two_memories)
        left = two_memories.relationships("11111111-1111-5111-8111-111111111111")
        right = two_memories.relationships(OTHER_ID)
        assert len(left) == len(right) == 1
        assert left[0].source_id == right[0].source_id

    def test_duplicate_relationship_is_rejected(self, two_memories: SqliteMemoryRepository) -> None:
        self._link(two_memories)
        with pytest.raises(RelationshipExistsError):
            self._link(two_memories)

    def test_same_pair_can_hold_different_relationships(
        self, two_memories: SqliteMemoryRepository
    ) -> None:
        self._link(two_memories)
        self._link(two_memories, relationship=RelationshipType.CONFLICTS_WITH, similarity=None)
        kinds = {rel.relationship for rel in two_memories.relationships(OTHER_ID)}
        assert kinds == {RelationshipType.DUPLICATE_OF, RelationshipType.CONFLICTS_WITH}

    def test_relationship_direction_matters(self, two_memories: SqliteMemoryRepository) -> None:
        self._link(two_memories, relationship=RelationshipType.SUPERSEDES)
        with pytest.raises(RelationshipExistsError):
            self._link(two_memories, relationship=RelationshipType.SUPERSEDES)

    def test_self_relationship_is_rejected(self, two_memories: SqliteMemoryRepository) -> None:
        with pytest.raises(InvalidRelationshipError):
            self._link(
                two_memories,
                source_id="11111111-1111-5111-8111-111111111111",
                target_id="11111111-1111-5111-8111-111111111111",
            )

    def test_relationship_to_unknown_memory_is_rejected(
        self, repository: SqliteMemoryRepository
    ) -> None:
        repository.insert(make_memory(), change=VersionChange.CREATED)
        with pytest.raises(MemoryNotFoundError):
            self._link(repository)

    def test_invalid_similarity_is_rejected(self, two_memories: SqliteMemoryRepository) -> None:
        with pytest.raises(InvalidRelationshipError):
            self._link(two_memories, similarity=1.5)

    def test_relationship_table_has_indexes_and_constraints(
        self, repository: SqliteMemoryRepository, db_path: Path
    ) -> None:
        repository.insert(make_memory(), change=VersionChange.CREATED)
        repository.insert(make_memory(id=OTHER_ID), change=VersionChange.CREATED)

        with closing(open_connection(db_path)) as conn:
            sql = str(
                conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = 'memory_relationships'"
                ).fetchone()["sql"]
            )
            indexes = {
                str(row["name"])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
            }

        assert "UNIQUE (source_id, target_id, relationship)" in sql
        assert "CHECK (source_id <> target_id)" in sql
        assert "REFERENCES memories (id) ON DELETE CASCADE" in sql
        assert {"idx_relationships_source", "idx_relationships_target"} <= indexes

    def test_relationships_are_ordered_deterministically(
        self, two_memories: SqliteMemoryRepository
    ) -> None:
        self._link(two_memories)
        self._link(two_memories, relationship=RelationshipType.CONFLICTS_WITH)
        kinds = [rel.relationship for rel in two_memories.relationships(OTHER_ID)]
        assert kinds == sorted(kinds, key=lambda kind: kind.value)

    def test_superseded_pairs(self, two_memories: SqliteMemoryRepository) -> None:
        self._link(two_memories, relationship=RelationshipType.SUPERSEDES, similarity=None)
        pairs = two_memories.superseded_pairs(OTHER_ID)
        assert pairs == {frozenset({"11111111-1111-5111-8111-111111111111", OTHER_ID})}

        assert two_memories.superseded_pairs(UNKNOWN_ID) == set()

    def test_deleting_a_memory_cascades_its_relationships(
        self, two_memories: SqliteMemoryRepository, db_path: Path
    ) -> None:
        self._link(two_memories)
        with closing(open_connection(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM memories WHERE id = ?", (OTHER_ID,))
        assert two_memories.relationships(OTHER_ID) == []


def test_relationship_type_values() -> None:
    assert [kind.value for kind in RelationshipType] == [
        "DUPLICATE_OF",
        "CONFLICTS_WITH",
        "SUPERSEDES",
    ]


def test_memory_type_weights_are_documented() -> None:
    """Every memory type must have a scoring weight."""
    from app.domain.intelligence import DEFAULT_TYPE_WEIGHTS

    assert set(DEFAULT_TYPE_WEIGHTS) == set(MemoryType)
