"""SQLite embedding store tests."""

from __future__ import annotations

from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from app.domain.embeddings import (
    DEFAULT_HASHED_DIMENSIONS,
    Embedding,
    EmbeddingProviderError,
    HashedLexicalEmbeddingProvider,
)
from app.domain.errors import MemoryNotFoundError
from app.domain.models import VersionChange
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from tests.support import FIXED_START, make_memory

OTHER_ID = "33333333-3333-4333-8333-333333333333"
PROVIDER = HashedLexicalEmbeddingProvider.name
MODEL = HashedLexicalEmbeddingProvider.model


@pytest.fixture
def store(db_path: Path) -> SqliteEmbeddingStore:
    return SqliteEmbeddingStore(db_path)


@pytest.fixture
def seeded(store: SqliteEmbeddingStore, repository: object) -> SqliteEmbeddingStore:
    """Store plus one memory to attach embeddings to."""
    from app.storage.sqlite.repository import SqliteMemoryRepository

    assert isinstance(repository, SqliteMemoryRepository)
    repository.insert(make_memory(), change=VersionChange.CREATED)
    repository.insert(make_memory(id=OTHER_ID), change=VersionChange.CREATED)
    return store


def _embedding(
    values: tuple[float, ...], provider: str = PROVIDER, model: str = MODEL
) -> Embedding:
    return Embedding(
        values=values,
        provider=provider,
        model=model,
        dimensions=len(values),
        created_at=FIXED_START,
    )


class TestRoundTrip:
    def test_put_then_get(self, seeded: SqliteEmbeddingStore) -> None:
        embedding = seeded.get_or_embed(
            "11111111-1111-5111-8111-111111111111",
            "user prefers python",
            HashedLexicalEmbeddingProvider(dimensions=32),
            at=FIXED_START,
        )
        assert embedding is not None
        assert embedding.provider == PROVIDER
        assert embedding.model == MODEL
        assert seeded.count() == 1

        cached = seeded.get("11111111-1111-5111-8111-111111111111", provider=PROVIDER, model=MODEL)
        assert cached is not None
        assert cached.values == pytest.approx(embedding.values)

    def test_cache_hit_does_not_recompute(self, seeded: SqliteEmbeddingStore) -> None:
        provider = HashedLexicalEmbeddingProvider(dimensions=32)
        memory_id = "11111111-1111-5111-8111-111111111111"
        first = seeded.get_or_embed(memory_id, "text", provider, at=FIXED_START)
        assert first is not None
        assert seeded.count() == 1

        # A second call is served from the cache: identical vector, same row.
        second = seeded.get_or_embed(
            memory_id, "text", provider, at=FIXED_START + timedelta(hours=1)
        )
        assert second is not None
        assert second.values == pytest.approx(first.values)
        assert seeded.count() == 1

        cached = seeded.get(memory_id, provider=PROVIDER, model=MODEL)
        assert cached is not None
        assert cached.created_at == first.created_at  # creation time is preserved

    def test_put_replaces_an_existing_row(self, seeded: SqliteEmbeddingStore) -> None:
        memory_id = "11111111-1111-5111-8111-111111111111"
        seeded.put(memory_id, _embedding((1.0, 0.0)), at=FIXED_START)
        seeded.put(memory_id, _embedding((0.0, 1.0)), at=FIXED_START + timedelta(hours=1))

        assert seeded.count() == 1
        cached = seeded.get(memory_id, provider=PROVIDER, model=MODEL)
        assert cached is not None
        assert cached.values == pytest.approx((0.0, 1.0))

    def test_unknown_memory_is_rejected(self, store: SqliteEmbeddingStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            store.put("00000000-0000-4000-8000-000000000000", _embedding((1.0,)), at=FIXED_START)

    def test_dimensions_are_recorded(self, seeded: SqliteEmbeddingStore) -> None:
        seeded.put("11111111-1111-5111-8111-111111111111", _embedding((1.0, 0.0)), at=FIXED_START)
        assert seeded.dimensions_for(provider=PROVIDER, model=MODEL) == 2
        assert seeded.dimensions_for(provider="other", model="other") is None

    def test_provenance_separates_providers(self, seeded: SqliteEmbeddingStore) -> None:
        memory_id = "11111111-1111-5111-8111-111111111111"
        seeded.put(memory_id, _embedding((1.0,)), at=FIXED_START)
        seeded.put(memory_id, _embedding((1.0,), provider="other", model="m"), at=FIXED_START)

        assert seeded.count() == 2
        assert seeded.get(memory_id, provider="other", model="m") is not None

    def test_unavailable_provider_returns_none(self, seeded: SqliteEmbeddingStore) -> None:
        from app.domain.embeddings import OllamaEmbeddingProvider

        assert (
            seeded.get_or_embed(
                "11111111-1111-5111-8111-111111111111",
                "text",
                OllamaEmbeddingProvider(timeout_seconds=0.2),
                at=FIXED_START,
            )
            is None
        )


class TestSchema:
    def test_table_constraints(self, store: SqliteEmbeddingStore, db_path: Path) -> None:
        with closing(open_connection(db_path)) as conn:
            sql = str(
                conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = 'memory_embeddings'"
                ).fetchone()["sql"]
            )
            indexes = {
                str(row["name"])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
            }

        assert "UNIQUE (memory_id, provider, model)" in sql
        assert "REFERENCES memories (id) ON DELETE CASCADE" in sql
        assert "CHECK (dimensions > 0)" in sql
        assert {"idx_embeddings_provider", "idx_embeddings_memory"} <= indexes

    def test_cascade_deletes_embeddings(self, seeded: SqliteEmbeddingStore, db_path: Path) -> None:
        seeded.put("11111111-1111-5111-8111-111111111111", _embedding((1.0,)), at=FIXED_START)
        with closing(open_connection(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM memories WHERE id = ?", (OTHER_ID,))
        assert seeded.count() == 1

    def test_corrupted_vector_is_detected(
        self, seeded: SqliteEmbeddingStore, db_path: Path
    ) -> None:
        memory_id = "11111111-1111-5111-8111-111111111111"
        seeded.put(memory_id, _embedding((1.0, 0.0)), at=FIXED_START)
        with closing(open_connection(db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE memory_embeddings SET vector = ? WHERE memory_id = ?",
                (b"\x00\x01", memory_id),
            )
            conn.execute("COMMIT")

        with pytest.raises(EmbeddingProviderError):
            seeded.get(memory_id, provider=PROVIDER, model=MODEL)

    def test_default_dimensions(self, seeded: SqliteEmbeddingStore) -> None:
        embedding = seeded.get_or_embed(
            "11111111-1111-5111-8111-111111111111",
            "user prefers python",
            HashedLexicalEmbeddingProvider(),
            at=FIXED_START,
        )
        assert embedding is not None
        assert embedding.dimensions == DEFAULT_HASHED_DIMENSIONS
