"""Persistent storage for memory embeddings.

Embeddings are stored in SQLite (no external vector database) keyed by
``(memory_id, provider, model)``, together with the provenance needed to decide
whether a stored vector is still usable: provider name, model name, dimensions
and timestamps.

The store never computes embeddings itself; it asks the configured
:class:`~app.domain.embeddings.EmbeddingProvider` on a cache miss, which keeps
the "provider is optional" rule intact: with no provider configured the store is
simply never used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from app.domain.embeddings import Embedding, EmbeddingProvider
from app.domain.errors import EmbeddingProviderError, MemoryNotFoundError
from app.storage.sqlite.connection import open_connection, transaction
from app.storage.sqlite.migrations import run_migrations

_COLUMNS = "memory_id, provider, model, dimensions, vector, created_at, updated_at"


def _encode(values: tuple[float, ...]) -> bytes:
    """Pack floats little-endian (8 bytes each) — compact and dependency-free."""
    import struct

    return struct.pack(f"<{len(values)}d", *values)


def _decode(blob: bytes, dimensions: int) -> tuple[float, ...]:
    import struct

    if len(blob) != dimensions * 8:
        raise EmbeddingProviderError(
            f"stored vector is {len(blob)} bytes but {dimensions} dimensions were declared"
        )
    return tuple(struct.unpack(f"<{dimensions}d", blob))


class EmbeddingStore(ABC):
    """Contract for caching embeddings for memories."""

    @abstractmethod
    def get(self, memory_id: str, *, provider: str, model: str) -> Embedding | None:
        """Return a stored embedding, or ``None`` when it is not cached."""

    @abstractmethod
    def put(self, memory_id: str, embedding: Embedding, *, at: datetime) -> None:
        """Store (or replace) the embedding for a memory."""

    @abstractmethod
    def get_or_embed(
        self, memory_id: str, text: str, provider: EmbeddingProvider, *, at: datetime
    ) -> Embedding | None:
        """Return the cached embedding, computing and caching it when missing."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored embeddings."""

    @abstractmethod
    def dimensions_for(self, *, provider: str, model: str) -> int | None:
        """Stored dimensionality for a provider/model pair, if any."""


class SqliteEmbeddingStore(EmbeddingStore):
    """SQLite-backed embedding cache."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self.initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        """Create/upgrade the schema; safe to call repeatedly."""
        with closing(open_connection(self._db_path)) as conn:
            run_migrations(conn)

    def get(self, memory_id: str, *, provider: str, model: str) -> Embedding | None:
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM memory_embeddings WHERE memory_id = ? "
                "AND provider = ? AND model = ?",
                (memory_id, provider, model),
            ).fetchone()
        if row is None:
            return None
        dimensions = int(row["dimensions"])
        return Embedding(
            values=_decode(bytes(row["vector"]), dimensions),
            provider=str(row["provider"]),
            model=str(row["model"]),
            dimensions=dimensions,
            created_at=_parse_timestamp(str(row["created_at"])),
        )

    def put(self, memory_id: str, embedding: Embedding, *, at: datetime) -> None:
        with closing(open_connection(self._db_path)) as conn, transaction(conn):
            row = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(memory_id)
            conn.execute(
                "INSERT INTO memory_embeddings (memory_id, provider, model, dimensions, "
                "vector, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (memory_id, provider, model) DO UPDATE SET "
                "dimensions = excluded.dimensions, vector = excluded.vector, "
                "updated_at = excluded.updated_at",
                (
                    memory_id,
                    embedding.provider,
                    embedding.model,
                    embedding.dimensions,
                    _encode(embedding.values),
                    _format_timestamp(embedding.created_at),
                    _format_timestamp(at),
                ),
            )

    def get_or_embed(
        self, memory_id: str, text: str, provider: EmbeddingProvider, *, at: datetime
    ) -> Embedding | None:
        if not provider.is_available():
            return None
        cached = self.get(memory_id, provider=provider.name, model=provider.model)
        if cached is not None:
            return cached
        embedding = provider.embed(text)
        self.put(memory_id, embedding, at=at)
        return embedding

    def count(self) -> int:
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM memory_embeddings").fetchone()
        return int(row["total"])

    def dimensions_for(self, *, provider: str, model: str) -> int | None:
        with closing(open_connection(self._db_path)) as conn:
            row = conn.execute(
                "SELECT dimensions FROM memory_embeddings WHERE provider = ? AND model = ? LIMIT 1",
                (provider, model),
            ).fetchone()
        return None if row is None else int(row["dimensions"])


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


__all__ = ["EmbeddingStore", "SqliteEmbeddingStore"]
