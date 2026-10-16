"""Covers the remaining defensive branches (run once; appended as a new module)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import app.cli.demo as demo_module
from app.domain.embeddings import (
    Embedding,
    EmbeddingProvider,
    HashedLexicalEmbeddingProvider,
    OllamaEmbeddingProvider,
    build_provider,
)
from app.domain.errors import EmbeddingProviderError
from app.domain.models import VersionChange
from app.domain.retrieval import RetrievalMode, RetrievalQuery
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START, make_memory


class TestProviderEdges:
    def test_hashed_provider_rejects_non_strings(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            HashedLexicalEmbeddingProvider().embed(5)  # type: ignore[arg-type]

    def test_base_embed_is_abstract(self) -> None:
        with pytest.raises(NotImplementedError):
            EmbeddingProvider().embed("text")

    def test_base_provider_describes_itself(self) -> None:
        class _Named(EmbeddingProvider):
            name = "demo"
            model = "demo-v1"

            @property
            def dimensions(self) -> int:
                return 8

            def is_available(self) -> bool:
                return True

            def embed(self, text: str) -> Embedding:
                return Embedding(
                    values=(1.0, 0.0),
                    provider=self.name,
                    model=self.model,
                    dimensions=2,
                    created_at=FIXED_START,
                )

        assert _Named().describe() == "demo/demo-v1 (8d)"

    def test_build_provider_for_ollama(self) -> None:
        provider = build_provider("ollama", model="m", endpoint="http://x", timeout_seconds=1)
        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.model == "m"

    def test_ollama_http_failure_is_wrapped(self) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = json.dumps({"models": [{"name": "nomic-embed-text"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                self.send_response(500)
                self.end_headers()

            def log_message(self, *_: object) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OllamaEmbeddingProvider(
                endpoint=f"http://127.0.0.1:{server.server_port}", timeout_seconds=5.0
            )
            with pytest.raises(EmbeddingProviderError):
                provider.embed("text")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_ollama_empty_vector_is_rejected(self) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = json.dumps({"models": [{"name": "nomic-embed-text"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                payload = json.dumps({"embedding": []}).encode()
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
            with pytest.raises(EmbeddingProviderError, match="no embedding"):
                provider.embed("text")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class TestRetrievalDefensiveBranches:
    def test_query_with_only_unknown_terms(self, repository: SqliteMemoryRepository) -> None:
        from app.domain.time import ManualClock

        repository.insert(make_memory(content="alpha beta"), change=VersionChange.CREATED)
        retriever = MemoryRetrievalService(repository, clock=ManualClock(FIXED_START))
        # "kubernetes" is not in the corpus, so no term is scorable.
        response = retriever.search(RetrievalQuery(text="kubernetes"))
        assert response.results == ()
        assert response.candidates_considered == 0

    def test_conflict_info_for_a_memory_without_attributes(
        self, repository: SqliteMemoryRepository
    ) -> None:
        """Conflict metadata degrades gracefully when no attribute was parsed."""
        from app.domain.relationships import MemoryRelationship, RelationshipType

        memory_id = repository.insert(
            make_memory(content="plain memory"), change=VersionChange.CREATED
        ).memory_id
        other_id = repository.insert(
            make_memory(id="33333333-3333-4333-8333-333333333333", content="other"),
            change=VersionChange.CREATED,
        ).memory_id
        repository.add_relationship(
            MemoryRelationship(
                source_id=memory_id,
                target_id=other_id,
                relationship=RelationshipType.CONFLICTS_WITH,
                similarity=None,
                created_at=FIXED_START,
            )
        )
        repository.add_relationship(
            MemoryRelationship(
                source_id=other_id,
                target_id=memory_id,
                relationship=RelationshipType.SUPERSEDES,
                similarity=None,
                created_at=FIXED_START,
            )
        )

        retriever = MemoryRetrievalService(repository)
        result = retriever.explain(memory_id, RetrievalQuery(text="plain memory"))
        assert result.conflicts
        assert result.conflicts[0].attribute == ""
        assert result.superseded_by == other_id

    def test_explain_of_an_unknown_memory(self, repository: SqliteMemoryRepository) -> None:
        from app.domain.errors import MemoryNotFoundError

        retriever = MemoryRetrievalService(repository)
        with pytest.raises(MemoryNotFoundError):
            retriever.explain("00000000-0000-4000-8000-000000000000", RetrievalQuery(text="x"))


class TestDemoDegradationOutput:
    def test_degraded_reason_is_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The demo prints the degradation reason instead of hiding it."""
        import tempfile

        db = Path(tempfile.mkdtemp()) / "degraded.sqlite"
        repository = SqliteMemoryRepository(db)
        from app.domain.time import ManualClock

        retriever = MemoryRetrievalService(repository, clock=ManualClock(FIXED_START))
        memory = retriever._repository.insert(
            make_memory(content="alpha beta"), change=VersionChange.CREATED
        )
        del memory

        class _Service:
            """Stand-in exposing only what the demo prints."""

            def get_memory(self, memory_id: str) -> object:  # pragma: no cover
                return None

        demo_module._run_retrieval(
            _Service(),  # type: ignore[arg-type]
            retriever,
            "alpha beta",
            RetrievalMode.SEMANTIC,
            semantic_weight=None,
        )
        out = capsys.readouterr().out
        assert "mode=lexical" in out
        assert "degraded:" in out
        assert "no embedding provider" in out


class TestLastBranches:
    def test_lexical_relevance_with_a_degenerate_idf_denominator(self) -> None:
        """A non-positive idf mass scores zero instead of dividing by it."""
        from app.domain.retrieval import lexical_relevance

        # A document frequency larger than the corpus makes every idf negative,
        # so the sum of the query mass cannot be positive.
        score = lexical_relevance(
            ("alpha",),
            frozenset({"alpha"}),
            corpus_size=1,
            document_frequencies={"alpha": 100},
        )
        assert score == 0.0

    def test_semantic_search_without_a_query_embedding(
        self, repository: SqliteMemoryRepository
    ) -> None:
        """When the query cannot be embedded, every similarity is zero."""
        from app.domain.time import ManualClock

        repository.insert(make_memory(content="alpha beta"), change=VersionChange.CREATED)

        class _BrokenProvider(HashedLexicalEmbeddingProvider):
            def embed(self, text: str) -> Embedding:
                raise EmbeddingProviderError("boom")

        retriever = MemoryRetrievalService(
            repository, clock=ManualClock(FIXED_START), embedding_provider=_BrokenProvider()
        )
        response = retriever.search(RetrievalQuery(text="alpha beta", mode=RetrievalMode.SEMANTIC))
        assert response.results
        assert response.results[0].semantic_score == 0.0

    def test_conflict_winner_is_reported_from_this_side(
        self, repository: SqliteMemoryRepository
    ) -> None:
        from app.domain.relationships import MemoryRelationship, RelationshipType

        memory_id = repository.insert(
            make_memory(content="plain memory"), change=VersionChange.CREATED
        ).memory_id
        other_id = repository.insert(
            make_memory(id="33333333-3333-4333-8333-333333333333", content="other"),
            change=VersionChange.CREATED,
        ).memory_id
        # This memory supersedes the other one.
        repository.add_relationship(
            MemoryRelationship(
                source_id=memory_id,
                target_id=other_id,
                relationship=RelationshipType.SUPERSEDES,
                similarity=None,
                created_at=FIXED_START,
            )
        )
        retriever = MemoryRetrievalService(repository)
        result = retriever.explain(memory_id, RetrievalQuery(text="plain memory"))
        assert result.conflicts == ()
        assert result.superseded_by is None

    def test_factor_value_defaults_to_zero(self, repository: SqliteMemoryRepository) -> None:
        from app.domain.time import ManualClock

        memory = make_memory(content="alpha beta")
        repository.insert(memory, change=VersionChange.CREATED)
        retriever = MemoryRetrievalService(repository, clock=ManualClock(FIXED_START))
        result = retriever.explain(memory.id, RetrievalQuery(text="alpha"))
        empty = type(result)(
            memory=memory,
            lexical_score=0.0,
            semantic_score=None,
            relevance=0.0,
            final_score=0.0,
            components=(),
            explanation="",
            conflicts=(),
            superseded_by=None,
            duplicate_of=None,
            suppressed=False,
            suppression_reason=None,
        )
        assert MemoryRetrievalService._factor_value(empty, "relevance") == 0.0
        assert result.components
