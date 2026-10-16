"""Embedding provider abstraction tests."""

from __future__ import annotations

import math

import pytest

from app.domain.embeddings import (
    DEFAULT_HASHED_DIMENSIONS,
    Embedding,
    EmbeddingProvider,
    HashedLexicalEmbeddingProvider,
    OllamaEmbeddingProvider,
    build_provider,
    cosine_similarity,
    l2_normalize,
)
from app.domain.errors import EmbeddingProviderError
from app.domain.similarity import normalized_tokens
from tests.support import FIXED_START


class TestEmbeddingValidation:
    def test_dimensions_must_match_values(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            Embedding(
                values=(1.0, 0.0),
                provider="p",
                model="m",
                dimensions=3,
                created_at=FIXED_START,
            )

    def test_empty_vector_is_rejected(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            Embedding(values=(), provider="p", model="m", dimensions=0, created_at=FIXED_START)

    @pytest.mark.parametrize("value", [float("inf"), float("nan")])
    def test_non_finite_values_are_rejected(self, value: float) -> None:
        with pytest.raises(EmbeddingProviderError):
            Embedding(
                values=(value, 0.0),
                provider="p",
                model="m",
                dimensions=2,
                created_at=FIXED_START,
            )

    def test_provenance_is_required(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            Embedding(values=(1.0,), provider="", model="m", dimensions=1, created_at=FIXED_START)


class TestVectorMath:
    def test_l2_normalize_produces_unit_vectors(self) -> None:
        normalized = l2_normalize([3.0, 4.0])
        assert normalized == (0.6, 0.8)
        assert math.isclose(math.sqrt(sum(v * v for v in normalized)), 1.0)

    def test_l2_normalize_leaves_zero_vectors_alone(self) -> None:
        assert l2_normalize([0.0, 0.0]) == (0.0, 0.0)

    def test_cosine_is_bounded_and_symmetric(self) -> None:
        left, right = (1.0, 0.0), (0.0, 1.0)
        assert cosine_similarity(left, left) == 1.0
        assert cosine_similarity(left, right) == 0.0
        assert cosine_similarity(left, right) == cosine_similarity(right, left)

    def test_cosine_of_opposite_vectors_is_clamped_to_zero(self) -> None:
        assert cosine_similarity((1.0,), (-1.0,)) == 0.0

    def test_cosine_requires_matching_dimensions(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            cosine_similarity((1.0, 0.0), (1.0,))


class TestHashedProvider:
    def test_is_always_available(self) -> None:
        assert HashedLexicalEmbeddingProvider().is_available() is True

    def test_dimensions_are_validated(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            HashedLexicalEmbeddingProvider(dimensions=4)

    def test_embeddings_are_deterministic(self) -> None:
        provider = HashedLexicalEmbeddingProvider(dimensions=32)
        assert (
            provider.embed("user prefers python").values
            == provider.embed("user prefers python").values
        )

    def test_identical_text_scores_one(self) -> None:
        provider = HashedLexicalEmbeddingProvider(dimensions=64)
        vector = provider.embed("user prefers python").values
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_similar_text_scores_higher_than_unrelated_text(self) -> None:
        provider = HashedLexicalEmbeddingProvider(dimensions=256)
        query = provider.embed("preferred backend language").values
        similar = provider.embed("preferred backend language = Python").values
        unrelated = provider.embed("the build server is in hyderabad").values
        assert cosine_similarity(query, similar) > cosine_similarity(query, unrelated)

    def test_morphological_variants_match(self) -> None:
        provider = HashedLexicalEmbeddingProvider(dimensions=256)
        first = provider.embed("user prefers python").values
        second = provider.embed("python is the user's preferred language").values
        # 3 shared tokens out of 3 and 4: cosine = 3 / sqrt(3 * 4) ~= 0.87.
        assert cosine_similarity(first, second) == pytest.approx(0.866, abs=0.02)

    def test_embeddings_are_unit_length(self) -> None:
        vector = HashedLexicalEmbeddingProvider(dimensions=64).embed("alpha beta gamma").values
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0)

    def test_embedding_carries_provenance(self) -> None:
        embedding = HashedLexicalEmbeddingProvider().embed("text")
        assert embedding.provider == HashedLexicalEmbeddingProvider.name
        assert embedding.model == HashedLexicalEmbeddingProvider.model
        assert embedding.dimensions == DEFAULT_HASHED_DIMENSIONS

    def test_empty_text_produces_a_zero_vector(self) -> None:
        provider = HashedLexicalEmbeddingProvider()
        vector = provider.embed("   ").values
        assert set(vector) == {0.0}
        assert normalized_tokens("   ") == ()


class TestOllamaProvider:
    def test_unavailable_without_a_server(self) -> None:
        provider = OllamaEmbeddingProvider(timeout_seconds=0.2)
        assert provider.is_available() is False

    def test_embedding_fails_loudly_when_unavailable(self) -> None:
        provider = OllamaEmbeddingProvider(timeout_seconds=0.2)
        with pytest.raises(EmbeddingProviderError):
            provider.embed("text")

    def test_configuration_is_validated(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            OllamaEmbeddingProvider(timeout_seconds=0)


class TestBuildProvider:
    def test_default_is_none(self) -> None:
        assert build_provider(None) is None
        assert build_provider("") is None
        assert build_provider("none") is None

    def test_hashed_provider(self) -> None:
        provider = build_provider("hashed", dimensions=32)
        assert isinstance(provider, HashedLexicalEmbeddingProvider)
        assert provider.dimensions == 32

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            build_provider("openai")

    def test_protocol_is_satisfied(self) -> None:
        assert isinstance(HashedLexicalEmbeddingProvider(), EmbeddingProvider)
