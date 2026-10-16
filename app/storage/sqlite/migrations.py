"""Tiny, explicit schema migration runner.

Migrations are ordered tuples of SQL statements that are applied exactly once
each; applied migration ids are recorded in the ``_memforge_migrations`` table so
the schema can be upgraded (and inspected) without an ORM.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from app.storage.sqlite.connection import transaction

#: Bookkeeping table that records which migrations have been applied.
MIGRATION_TABLE = "_memforge_migrations"

_BOOTSTRAP = f"""
CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
    id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class Migration:
    """A single, atomically applied schema change."""

    id: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        id="0001_initial_memory_core",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                source      TEXT,
                confidence  REAL NOT NULL,
                importance  REAL NOT NULL,
                status      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                version     INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id   TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                version     INTEGER NOT NULL,
                change      TEXT NOT NULL,
                content     TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                source      TEXT,
                confidence  REAL NOT NULL,
                importance  REAL NOT NULL,
                status      TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                CONSTRAINT uq_memory_versions_version UNIQUE (memory_id, version)
            )
            """,
            # Common lookups: filtered listings and newest-first pagination.
            "CREATE INDEX IF NOT EXISTS idx_memories_status_type ON memories (status, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_versions_lookup ON memory_versions (memory_id, version)",
        ),
    ),
    Migration(
        id="0002_memory_intelligence",
        statements=(
            # Access telemetry for the access-frequency factor of the importance
            # score. Defaults keep every existing M1 row valid.
            "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN last_accessed_at TEXT",
            """
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                target_id   TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                relationship TEXT NOT NULL,
                similarity  REAL,
                created_at  TEXT NOT NULL,
                CONSTRAINT uq_memory_relationships UNIQUE (source_id, target_id, relationship),
                CONSTRAINT chk_relationship_not_self CHECK (source_id <> target_id),
                CONSTRAINT chk_relationship_kind
                    CHECK (relationship IN ('DUPLICATE_OF', 'CONFLICTS_WITH', 'SUPERSEDES')),
                CONSTRAINT chk_relationship_similarity
                    CHECK (similarity IS NULL OR (similarity >= 0.0 AND similarity <= 1.0))
            )
            """,
            # "what is related to this memory?" in both directions.
            "CREATE INDEX IF NOT EXISTS idx_relationships_source "
            "ON memory_relationships (source_id, relationship)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target "
            "ON memory_relationships (target_id, relationship)",
        ),
    ),
    Migration(
        id="0003_memory_retrieval",
        statements=(
            # Cached embeddings: one row per (memory, provider, model). No
            # external vector database is introduced; the vector is a packed
            # little-endian float array and the provenance columns make a cached
            # row self-describing.
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id   TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                provider    TEXT NOT NULL,
                model       TEXT NOT NULL,
                dimensions  INTEGER NOT NULL,
                vector      BLOB NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                CONSTRAINT uq_memory_embeddings UNIQUE (memory_id, provider, model),
                CONSTRAINT chk_embedding_dimensions CHECK (dimensions > 0),
                CONSTRAINT chk_embedding_provider CHECK (length(provider) > 0),
                CONSTRAINT chk_embedding_model CHECK (length(model) > 0)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_embeddings_provider "
            "ON memory_embeddings (provider, model)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_memory ON memory_embeddings (memory_id)",
        ),
    ),
    Migration(
        id="0004_observability",
        statements=(
            # One row per correlated unit of work. Traces are append-only: a
            # finished trace is never rewritten, so an audit survives restarts.
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id    TEXT PRIMARY KEY,
                kind        TEXT NOT NULL,
                operation   TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                attributes  TEXT NOT NULL,
                CONSTRAINT chk_trace_kind CHECK (kind IN (
                    'agent_request', 'mcp_request', 'api_request',
                    'evaluation_case', 'cli'
                ))
            )
            """,
            # One row per event. ``sequence`` makes the order total and
            # reproducible; ``parent_id`` turns the flat list into an inspectable
            # tree. Metadata is JSON that the service has already redacted.
            """
            CREATE TABLE IF NOT EXISTS trace_events (
                event_id    TEXT PRIMARY KEY,
                trace_id    TEXT NOT NULL REFERENCES traces (trace_id) ON DELETE CASCADE,
                sequence    INTEGER NOT NULL,
                kind        TEXT NOT NULL,
                operation   TEXT NOT NULL,
                status      TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                parent_id   TEXT,
                duration_ms REAL,
                metadata    TEXT NOT NULL,
                detail      TEXT,
                CONSTRAINT uq_trace_event_sequence UNIQUE (trace_id, sequence),
                CONSTRAINT chk_event_status CHECK (status IN ('ok', 'skipped', 'error')),
                CONSTRAINT chk_event_duration CHECK (duration_ms IS NULL OR duration_ms >= 0)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_trace_events_trace "
            "ON trace_events (trace_id, sequence)",
            "CREATE INDEX IF NOT EXISTS idx_trace_events_operation "
            "ON trace_events (operation, status)",
            "CREATE INDEX IF NOT EXISTS idx_traces_started ON traces (started_at DESC, trace_id)",
        ),
    ),
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def applied_migrations(conn: sqlite3.Connection) -> list[str]:
    """Migration ids already applied, in the order they were applied."""
    if not _table_exists(conn, MIGRATION_TABLE):
        return []
    rows = conn.execute(f"SELECT id FROM {MIGRATION_TABLE} ORDER BY id").fetchall()
    return [str(row["id"]) for row in rows]


def run_migrations(conn: sqlite3.Connection) -> list[str]:
    """Apply every pending migration; return the ids that were just applied."""
    with transaction(conn):
        conn.execute(_BOOTSTRAP)
        already_applied = set(applied_migrations(conn))
        newly_applied: list[str] = []
        for migration in MIGRATIONS:
            if migration.id in already_applied:
                continue
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                f"INSERT INTO {MIGRATION_TABLE} (id, applied_at) VALUES (?, ?)",
                (migration.id, datetime.now(UTC).isoformat(timespec="microseconds")),
            )
            newly_applied.append(migration.id)
    return newly_applied


def schema_version(conn: sqlite3.Connection) -> str | None:
    """Id of the latest applied migration, or ``None`` for an empty database."""
    applied = applied_migrations(conn)
    return applied[-1] if applied else None
