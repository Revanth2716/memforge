"""SQLite implementation of the memory repository.

Design notes
------------
* **Parameterised SQL only.** No value is ever interpolated into a statement;
  only static SQL fragments (``WHERE``/``ORDER BY`` clauses) are assembled.
* **One connection per operation.** Statements are wrapped in a single
  transaction per write, so a memory and its history snapshot are always
  consistent with each other.
* **Soft deletes.** Forgetting a memory flips ``status`` to ``DELETED`` and
  appends a history version; rows are never physically removed, so history
  survives forever.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from app.domain.errors import (
    InvalidRelationshipError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    RelationshipExistsError,
    RepositoryError,
)
from app.domain.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    VersionChange,
    build_version,
)
from app.domain.relationships import MemoryRelationship, RelationshipType
from app.storage.base import CandidateQuery, MemoryQuery, MemoryRepository
from app.storage.sqlite.connection import open_connection, transaction
from app.storage.sqlite.migrations import run_migrations, schema_version

_MEMORY_COLUMNS = (
    "id, content, memory_type, source, confidence, importance, "
    "status, created_at, updated_at, version, access_count, last_accessed_at"
)
#: Columns written when a memory is first created (telemetry starts empty).
_MEMORY_INSERT_COLUMNS = (
    "id, content, memory_type, source, confidence, importance, "
    "status, created_at, updated_at, version"
)
_VERSION_COLUMNS = (
    "memory_id, version, change, content, memory_type, source, "
    "confidence, importance, status, recorded_at"
)
#: Columns stored/read for relationships (``id`` is database assigned).
_RELATIONSHIP_COLUMNS = "source_id, target_id, relationship, similarity, created_at"


def _format_timestamp(value: datetime) -> str:
    """Serialise a datetime as a fixed-width UTC ISO-8601 string."""
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, assuming UTC when no offset is present."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=str(row["id"]),
        content=str(row["content"]),
        memory_type=MemoryType(str(row["memory_type"])),
        source=None if row["source"] is None else str(row["source"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        status=MemoryStatus(str(row["status"])),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
        version=int(row["version"]),
        access_count=int(row["access_count"]),
        last_accessed_at=(
            None
            if row["last_accessed_at"] is None
            else _parse_timestamp(str(row["last_accessed_at"]))
        ),
    )


def _row_to_version(row: sqlite3.Row) -> MemoryVersion:
    return MemoryVersion(
        memory_id=str(row["memory_id"]),
        version=int(row["version"]),
        change=VersionChange(str(row["change"])),
        content=str(row["content"]),
        memory_type=MemoryType(str(row["memory_type"])),
        source=None if row["source"] is None else str(row["source"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        status=MemoryStatus(str(row["status"])),
        recorded_at=_parse_timestamp(str(row["recorded_at"])),
    )


def _row_to_relationship(row: sqlite3.Row) -> MemoryRelationship:
    similarity = row["similarity"]
    return MemoryRelationship(
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        relationship=RelationshipType(str(row["relationship"])),
        similarity=None if similarity is None else float(similarity),
        created_at=_parse_timestamp(str(row["created_at"])),
    )


class SqliteMemoryRepository(MemoryRepository):
    """Memory persistence in a single SQLite database file.

    The schema is created (or upgraded) eagerly on construction, so a freshly
    built repository is always ready to use. ``initialize()`` remains public and
    idempotent for explicit startup checks.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self.initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        """Create the schema on first use; no-op on later calls."""
        with closing(open_connection(self._db_path)) as conn:
            run_migrations(conn)

    def ping(self) -> bool:
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute("SELECT 1 AS ok").fetchone()
        return row is not None and int(row["ok"]) == 1

    def schema_version(self) -> str | None:
        with closing(open_connection(self._db_path)) as conn:
            return schema_version(conn)

    # -- writes -----------------------------------------------------------

    def insert(self, memory: Memory, *, change: VersionChange) -> MemoryVersion:
        version = build_version(memory, change=change, recorded_at=memory.updated_at)
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            try:
                conn.execute(
                    f"INSERT INTO memories ({_MEMORY_INSERT_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory.id,
                        memory.content,
                        memory.memory_type.value,
                        memory.source,
                        memory.confidence,
                        memory.importance,
                        memory.status.value,
                        _format_timestamp(memory.created_at),
                        _format_timestamp(memory.updated_at),
                        memory.version,
                    ),
                )
                conn.execute(
                    f"INSERT INTO memory_versions ({_VERSION_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._version_params(memory, version),
                )
            except sqlite3.IntegrityError as exc:
                raise self._integrity_error(exc, memory.id) from exc
        return version

    def save(self, memory: Memory, *, change: VersionChange) -> MemoryVersion:
        version = build_version(memory, change=change, recorded_at=memory.updated_at)
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            row = conn.execute("SELECT version FROM memories WHERE id = ?", (memory.id,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(memory.id)
            stored_version = int(row["version"])
            if memory.version != stored_version + 1:
                raise RepositoryError(
                    f"Version conflict for memory {memory.id!r}: "
                    f"stored version {stored_version}, tried to write {memory.version}"
                )
            try:
                conn.execute(
                    "UPDATE memories SET content = ?, memory_type = ?, source = ?, "
                    "confidence = ?, importance = ?, status = ?, created_at = ?, "
                    "updated_at = ?, version = ? WHERE id = ?",
                    (
                        memory.content,
                        memory.memory_type.value,
                        memory.source,
                        memory.confidence,
                        memory.importance,
                        memory.status.value,
                        _format_timestamp(memory.created_at),
                        _format_timestamp(memory.updated_at),
                        memory.version,
                        memory.id,
                    ),
                )
                conn.execute(
                    f"INSERT INTO memory_versions ({_VERSION_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._version_params(memory, version),
                )
            except sqlite3.IntegrityError as exc:
                raise self._integrity_error(exc, memory.id) from exc
        return version

    @staticmethod
    def _version_params(memory: Memory, version: MemoryVersion) -> tuple[object, ...]:
        return (
            memory.id,
            memory.version,
            version.change.value,
            memory.content,
            memory.memory_type.value,
            memory.source,
            memory.confidence,
            memory.importance,
            memory.status.value,
            _format_timestamp(version.recorded_at),
        )

    @staticmethod
    def _integrity_error(exc: sqlite3.IntegrityError, memory_id: str) -> Exception:
        message = str(exc).upper()
        if "UNIQUE" in message or "PRIMARY KEY" in message:
            return MemoryAlreadyExistsError(memory_id)
        return RepositoryError(f"Database rejected the write for {memory_id!r}: {exc}")

    # -- reads ------------------------------------------------------------

    def get(self, memory_id: str, *, include_deleted: bool = False) -> Memory | None:
        sql = f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?"
        params: list[object] = [memory_id]
        if not include_deleted:
            sql += " AND status <> ?"
            params.append(MemoryStatus.DELETED.value)
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute(sql, params).fetchone()
        return None if row is None else _row_to_memory(row)

    def find(self, query: MemoryQuery) -> tuple[list[Memory], int]:
        where: list[str] = []
        params: list[object] = []
        if query.status is not None:
            where.append("status = ?")
            params.append(query.status.value)
        if query.memory_type is not None:
            where.append("memory_type = ?")
            params.append(query.memory_type.value)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""

        with closing(open_connection(self._db_path)) as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM memories{where_sql}", params
            ).fetchone()
            total = int(total_row["total"])
            rows = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories{where_sql} "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                [*params, query.limit, query.offset],
            ).fetchall()
        return [_row_to_memory(row) for row in rows], total

    def versions(self, memory_id: str) -> list[MemoryVersion]:
        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT {_VERSION_COLUMNS} FROM memory_versions "
                "WHERE memory_id = ? ORDER BY version ASC",
                (memory_id,),
            ).fetchall()
        return [_row_to_version(row) for row in rows]

    # -- M2: intelligence support ------------------------------------------

    def active_memories(self) -> list[Memory]:
        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE status = ? "
                "ORDER BY created_at DESC, id DESC",
                (MemoryStatus.ACTIVE.value,),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def record_access(self, memory_id: str, *, at: datetime) -> Memory:
        """Metadata-only telemetry write (no history version, no updated_at)."""
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                "WHERE id = ? AND status <> ?",
                (_format_timestamp(at), memory_id, MemoryStatus.DELETED.value),
            )
            if cursor.rowcount == 0:
                raise MemoryNotFoundError(memory_id)
            row = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:  # pragma: no cover - the row was updated a moment ago
            raise MemoryNotFoundError(memory_id)
        return _row_to_memory(row)

    def add_relationship(self, relationship: MemoryRelationship) -> MemoryRelationship:
        recorded_at = relationship.created_at
        stored = MemoryRelationship(
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            relationship=relationship.relationship,
            similarity=relationship.similarity,
            created_at=recorded_at,
        )
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            existing = conn.execute(
                "SELECT id FROM memories WHERE id IN (?, ?)",
                (stored.source_id, stored.target_id),
            ).fetchall()
            known = {str(row["id"]) for row in existing}
            for memory_id in (stored.source_id, stored.target_id):
                if memory_id not in known:
                    raise MemoryNotFoundError(memory_id)
            try:
                conn.execute(
                    f"INSERT INTO memory_relationships ({_RELATIONSHIP_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        stored.source_id,
                        stored.target_id,
                        stored.relationship.value,
                        stored.similarity,
                        _format_timestamp(recorded_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._relationship_integrity_error(exc, stored) from exc
        return stored

    @staticmethod
    def _relationship_integrity_error(
        exc: sqlite3.IntegrityError, relationship: MemoryRelationship
    ) -> Exception:
        message = str(exc).upper()
        if "UNIQUE" in message:
            return RelationshipExistsError(
                relationship.source_id, relationship.target_id, relationship.relationship.value
            )
        if "FOREIGN KEY" in message:
            return MemoryNotFoundError(relationship.source_id)
        return InvalidRelationshipError(f"Database rejected the relationship: {exc}")

    def relationships(self, memory_id: str) -> list[MemoryRelationship]:
        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT {_RELATIONSHIP_COLUMNS} FROM memory_relationships "
                "WHERE source_id = ? OR target_id = ? "
                "ORDER BY relationship ASC, source_id ASC, target_id ASC",
                (memory_id, memory_id),
            ).fetchall()
        return [_row_to_relationship(row) for row in rows]

    def superseded_pairs(self, memory_id: str) -> set[frozenset[str]]:
        pairs: set[frozenset[str]] = set()
        for relationship in self.relationships(memory_id):
            if relationship.relationship is RelationshipType.SUPERSEDES:
                pairs.add(frozenset({relationship.source_id, relationship.target_id}))
        return pairs

    # -- M3: retrieval support --------------------------------------------

    def search_candidates(self, query: CandidateQuery) -> list[Memory]:
        """Candidate generation for retrieval (all filtering happens in SQL)."""
        where: list[str] = []
        params: list[object] = []

        if query.ids is not None:
            if not query.ids:
                return []
            placeholders = ", ".join("?" for _ in query.ids)
            where.append(f"id IN ({placeholders})")
            params.extend(query.ids)
        if query.status is not None:
            where.append("status = ?")
            params.append(query.status.value)
        if query.memory_type is not None:
            where.append("memory_type = ?")
            params.append(query.memory_type.value)
        if query.min_confidence is not None:
            where.append("confidence >= ?")
            params.append(query.min_confidence)
        if query.min_importance is not None:
            where.append("importance >= ?")
            params.append(query.min_importance)
        if query.created_after is not None:
            where.append("created_at >= ?")
            params.append(_format_timestamp(query.created_after))
        if query.created_before is not None:
            where.append("created_at <= ?")
            params.append(_format_timestamp(query.created_before))
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""

        with closing(open_connection(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories{where_sql} "
                "ORDER BY created_at DESC, id DESC",
                params,
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def searchable_corpus_signature(self) -> tuple[int, str]:
        """Cheap signature of the searchable corpus (count + newest updated_at).

        The searchable corpus is everything that is not forgotten; the status
        filter is applied at query time instead, so an explicit
        ``status=ARCHIVED`` request can still find archived memories.
        """
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, COALESCE(MAX(updated_at), '') AS newest "
                "FROM memories WHERE status <> ?",
                (MemoryStatus.DELETED.value,),
            ).fetchone()
        return int(row["total"]), str(row["newest"])
