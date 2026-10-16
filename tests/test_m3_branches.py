"""Covers the remaining M3 branches (run once; new test module)."""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.embeddings import (
    Embedding,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
)
from app.domain.errors import (
    InvalidRetrievalConfigError,
    RetrievalValidationError,
)
from app.domain.models import MemoryStatus
from app.domain.retrieval import RetrievalMode, RetrievalQuery
from app.main import create_app
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.base import CandidateQuery
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START, make_memory


class TestCandidateQueryValidation:
    def test_confidence_bound(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            CandidateQuery(min_confidence=1.5)

    def test_importance_bound(self) -> None:
        with pytest.raises(ValueError, match="min_importance"):
            CandidateQuery(min_importance=-0.5)

    def test_inverted_window(self) -> None:
        with pytest.raises(ValueError, match="created_before"):
            CandidateQuery(
                created_after=FIXED_START + timedelta(days=5), created_before=FIXED_START
            )

    def test_created_before_filter(self, repository: SqliteMemoryRepository) -> None:
        from app.domain.models import VersionChange

        memory = make_memory(created_at=FIXED_START, updated_at=FIXED_START)
        repository.insert(memory, change=VersionChange.CREATED)
        found = repository.search_candidates(
            CandidateQuery(
                created_before=FIXED_START + timedelta(days=1), status=MemoryStatus.ACTIVE
            )
        )
        missing = repository.search_candidates(
            CandidateQuery(
                created_before=FIXED_START - timedelta(days=1), status=MemoryStatus.ACTIVE
            )
        )
        assert [item.id for item in found] == [memory.id]
        assert missing == []


class TestServiceBranches:
    def test_describe_provider_without_one(self, repository: SqliteMemoryRepository) -> None:
        retriever = MemoryRetrievalService(repository)
        assert retriever.describe_provider() == "none (lexical only)"
        assert retriever.ranking_config is not None
        assert retriever.semantic_provider is None
        assert retriever.semantic_model is None
        assert retriever.semantic_available is False

    def test_repository_without_a_path_must_be_given_a_store(self) -> None:
        """A backend without a file location needs an explicit embedding store."""

        class _NoPathRepository:
            db_path = None

            def search_candidates(self, query: object) -> list[object]:
                return []

            def searchable_corpus_signature(self) -> tuple[int, str]:
                return 0, ""

        with pytest.raises(TypeError, match="db_path"):
            MemoryRetrievalService(_NoPathRepository())  # type: ignore[arg-type]

    def test_dimension_mismatch_scores_zero(self, repository: SqliteMemoryRepository) -> None:
        """A cached vector of a different width degrades to 0, it never raises."""
        from app.domain.embeddings import Embedding, HashedLexicalEmbeddingProvider
        from app.domain.models import VersionChange

        memory_id = repository.insert(
            make_memory(content="alpha beta"), change=VersionChange.CREATED
        ).memory_id

        # Cache a wide vector under the provider's identity ...
        wide = HashedLexicalEmbeddingProvider(dimensions=64)
        store = SqliteEmbeddingStore(repository.db_path)
        store.get_or_embed(memory_id, "alpha beta", wide, at=FIXED_START)

        class _NarrowProvider(HashedLexicalEmbeddingProvider):
            @property
            def dimensions(self) -> int:
                return 4

            def embed(self, text: str) -> Embedding:
                embedding = HashedLexicalEmbeddingProvider.embed(self, text)
                return Embedding(
                    values=embedding.values[:4],
                    provider=embedding.provider,
                    model=embedding.model,
                    dimensions=4,
                    created_at=embedding.created_at,
                )

        retriever = MemoryRetrievalService(
            repository, embedding_provider=_NarrowProvider(), embedding_store=store
        )
        response = retriever.search(RetrievalQuery(text="alpha beta", mode=RetrievalMode.SEMANTIC))
        assert response.results
        assert response.results[0].semantic_score == 0.0

    def test_embedding_failure_degrades_to_zero(self, repository: SqliteMemoryRepository) -> None:
        from app.domain.embeddings import EmbeddingProviderError, HashedLexicalEmbeddingProvider
        from app.domain.models import VersionChange

        repository.insert(make_memory(content="alpha beta"), change=VersionChange.CREATED)

        class _BrokenProvider(HashedLexicalEmbeddingProvider):
            def embed(self, text: str) -> Embedding:
                raise EmbeddingProviderError("boom")

        retriever = MemoryRetrievalService(repository, embedding_provider=_BrokenProvider())
        response = retriever.search(RetrievalQuery(text="alpha beta", mode=RetrievalMode.HYBRID))
        assert response.retrieval_mode is RetrievalMode.HYBRID
        assert response.results
        assert response.results[0].semantic_score == 0.0

    def test_explain_reports_the_lexical_component(
        self, repository: SqliteMemoryRepository, clock: object
    ) -> None:
        from app.domain.models import VersionChange
        from app.domain.time import ManualClock

        memory = make_memory(content="alpha beta gamma")
        repository.insert(memory, change=VersionChange.CREATED)
        retriever = MemoryRetrievalService(repository, clock=ManualClock(FIXED_START))
        result = retriever.explain(memory.id, RetrievalQuery(text="alpha", top_k=1))
        assert result.final_score > 0.0
        assert result.relevance == 1.0


def service_of(client: TestClient) -> object:
    """The retrieval service attached to the running application."""
    return cast(FastAPI, client.app).state.retrieval_service


class TestApiBranches:
    def test_retrieval_validation_error_is_mapped(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = create_app(settings)
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            monkeypatch.setattr(
                service_of(client),
                "explain",
                lambda *a, **k: (_ for _ in ()).throw(RetrievalValidationError("bad query")),
            )
            created = client.post(
                "/api/v1/memories", json={"content": "x", "memory_type": "FACT"}
            ).json()
            response = client.get(
                f"/api/v1/memories/{created['id']}/explain", params={"query": "x"}
            )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "RETRIEVAL_VALIDATION_ERROR"

    def test_invalid_retrieval_config_is_mapped(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = create_app(settings)
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            monkeypatch.setattr(
                service_of(client),
                "search",
                lambda *a, **k: (_ for _ in ()).throw(
                    InvalidRetrievalConfigError("weights must sum to 1.0")
                ),
            )
            response = client.post("/api/v1/memories/search", json={"query": "x"})
            assert response.status_code == 500
            assert response.json()["error"]["code"] == "INVALID_CONFIGURATION"

    def test_app_without_lifespan_still_serves_search(self, settings: Settings) -> None:
        from fastapi.testclient import TestClient

        app = create_app(settings)
        client = TestClient(app)
        try:
            created = client.post(
                "/api/v1/memories", json={"content": "alpha beta", "memory_type": "FACT"}
            ).json()
            response = client.post("/api/v1/memories/search", json={"query": "alpha"})
            assert response.status_code == 200
            assert response.json()["candidates_considered"] == 1
            assert created["id"]
        finally:
            client.close()


class TestOllamaOverHttp:
    """Exercise the real HTTP code path against a local stub server."""

    def test_embed_over_http(self, repository: SqliteMemoryRepository) -> None:
        vector = [0.1, 0.2, 0.3, 0.4]

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = json.dumps({"models": [{"name": "nomic-embed-text:latest"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                assert body.get("model") == "nomic-embed-text"
                payload = json.dumps({"embedding": vector}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: object) -> None:  # silence the test output
                return None

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OllamaEmbeddingProvider(
                endpoint=f"http://127.0.0.1:{server.server_port}", timeout_seconds=5.0
            )
            assert provider.is_available() is True
            embedding = provider.embed("alpha beta")
            assert embedding.dimensions == len(vector)
            assert embedding.provider == "ollama"
            assert list(embedding.values) == pytest.approx(vector)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_missing_model_is_reported_as_unavailable(self) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = json.dumps({"models": [{"name": "other-model"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: object) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OllamaEmbeddingProvider(
                endpoint=f"http://127.0.0.1:{server.server_port}", timeout_seconds=5.0
            )
            assert provider.is_available() is False
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_base_provider_is_abstract(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = EmbeddingProvider().dimensions

    def test_embedding_store_rejects_bad_dimensions(self, db_path: Path) -> None:
        store = SqliteEmbeddingStore(db_path)
        assert store.db_path == db_path
        assert store.dimensions_for(provider="x", model="y") is None
