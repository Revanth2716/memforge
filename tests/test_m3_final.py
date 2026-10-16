"""Covers the last defensive guards (run once; appended)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.domain.attributes import extract_attribute_values
from app.domain.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from app.domain.models import VersionChange
from app.domain.relationships import MemoryRelationship, RelationshipType
from app.domain.retrieval import RetrievalMode, RetrievalQuery
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START, make_memory


class TestFinalGuards:
    def test_base_provider_is_not_available(self) -> None:
        with pytest.raises(NotImplementedError):
            EmbeddingProvider().is_available()

    def test_ollama_probe_with_an_unexpected_payload(self) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = json.dumps({"unexpected": True}).encode()
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

    def test_supersedes_link_between_other_memories_is_ignored(
        self, repository: SqliteMemoryRepository
    ) -> None:
        """A SUPERSEDES row for a different pair does not affect this pair."""
        first = repository.insert(
            make_memory(content="alpha one"), change=VersionChange.CREATED
        ).memory_id
        second = repository.insert(
            make_memory(id="33333333-3333-4333-8333-333333333333", content="alpha two"),
            change=VersionChange.CREATED,
        ).memory_id
        third = repository.insert(
            make_memory(id="44444444-4444-4444-8444-444444444444", content="alpha three"),
            change=VersionChange.CREATED,
        ).memory_id
        conflict_id = repository.add_relationship(
            MemoryRelationship(
                source_id=first,
                target_id=second,
                relationship=RelationshipType.CONFLICTS_WITH,
                similarity=None,
                created_at=FIXED_START,
            )
        )
        assert conflict_id is not None
        fourth = repository.insert(
            make_memory(id="55555555-5555-4555-8555-555555555555", content="alpha four"),
            change=VersionChange.CREATED,
        ).memory_id
        repository.add_relationship(
            MemoryRelationship(
                source_id=third,
                target_id=fourth,
                relationship=RelationshipType.SUPERSEDES,
                similarity=None,
                created_at=FIXED_START,
            )
        )

        retriever = MemoryRetrievalService(repository)
        result = retriever.explain(first, RetrievalQuery(text="alpha one"))
        assert result.conflicts
        assert result.conflicts[0].resolution == "unresolved"
        assert extract_attribute_values("alpha one") == ()

    def test_semantic_mode_with_a_provider_that_disappears(
        self, repository: SqliteMemoryRepository
    ) -> None:
        """If the provider vanishes between probe and use, similarity is zero."""
        from app.domain.embeddings import HashedLexicalEmbeddingProvider

        repository.insert(make_memory(content="alpha beta"), change=VersionChange.CREATED)

        class _VanishingProvider(HashedLexicalEmbeddingProvider):
            def is_available(self) -> bool:
                return False

        retriever = MemoryRetrievalService(repository, embedding_provider=_VanishingProvider())
        response = retriever.search(RetrievalQuery(text="alpha beta", mode=RetrievalMode.SEMANTIC))
        # Unavailable provider: the request degraded to lexical.
        assert response.retrieval_mode is RetrievalMode.LEXICAL
        assert response.semantic_available is False
