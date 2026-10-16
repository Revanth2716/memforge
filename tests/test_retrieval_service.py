"""Memory-aware retrieval service tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.embeddings import HashedLexicalEmbeddingProvider
from app.domain.errors import RetrievalValidationError
from app.domain.ids import SequentialIdGenerator
from app.domain.models import Memory, MemoryStatus, MemoryType, VersionChange
from app.domain.retrieval import RankingConfig, RetrievalFilters, RetrievalMode, RetrievalQuery
from app.domain.time import ManualClock
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START

PYTHON = "User prefers Python for backend work"
TYPESCRIPT = "User is learning TypeScript for frontend work"
NEAR_DUPLICATE = "The user prefers python for backend development work"


@pytest.fixture
def memories(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryService:
    """A deterministic memory service over the same repository."""
    return MemoryService(repository, clock=clock, id_generator=SequentialIdGenerator(prefix="mem"))


@pytest.fixture
def retriever(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryRetrievalService:
    """Lexical-only retrieval (no embedding provider)."""
    return MemoryRetrievalService(repository, clock=clock)


@pytest.fixture
def semantic_retriever(
    repository: SqliteMemoryRepository, clock: ManualClock
) -> MemoryRetrievalService:
    """Retrieval with the deterministic local embedding provider."""
    return MemoryRetrievalService(
        repository, clock=clock, embedding_provider=HashedLexicalEmbeddingProvider(dimensions=64)
    )


class TestLexicalRetrieval:
    def test_returns_matching_memories(self, retriever: MemoryRetrievalService) -> None:
        assert retriever.search(RetrievalQuery(text="python", top_k=5)).candidates_considered == 0

    def test_ranks_relevant_memory_first(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=TYPESCRIPT, memory_type=MemoryType.OBSERVATION)
        python = memories.create_memory(
            content=PYTHON, memory_type=MemoryType.PREFERENCE, confidence=0.9, importance=0.8
        )

        response = retriever.search(RetrievalQuery(text="python backend work", top_k=5))
        assert response.retrieval_mode is RetrievalMode.LEXICAL
        assert response.results[0].memory.id == python.id
        assert response.results[0].lexical_score == 1.0
        assert response.results[0].semantic_score is None
        assert response.candidates_returned == 2
        assert response.latency_ms >= 0.0

    def test_only_active_memories_by_default(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        from app.domain.updates import MemoryUpdate

        created = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        memories.forget_memory(created.id)
        response = retriever.search(RetrievalQuery(text="python"))
        assert response.candidates_considered == 0

        archived = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        memories.update_memory(archived.id, MemoryUpdate(status=MemoryStatus.ARCHIVED))
        assert retriever.search(RetrievalQuery(text="python")).candidates_considered == 0
        assert (
            retriever.search(
                RetrievalQuery(
                    text="python", filters=RetrievalFilters(status=MemoryStatus.ARCHIVED)
                )
            ).candidates_considered
            == 1
        )

    def test_top_k_limits_results(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        for index in range(5):
            memories.create_memory(
                content=f"python backend work number {index}", memory_type=MemoryType.TASK
            )
        response = retriever.search(RetrievalQuery(text="python backend", top_k=2))
        assert response.candidates_considered == 5
        assert response.candidates_returned == 2
        assert response.top_k == 2

    def test_top_k_is_validated(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        with pytest.raises(RetrievalValidationError):
            RetrievalQuery(text="python", top_k=0)

    def test_empty_query_is_rejected(self, retriever: MemoryRetrievalService) -> None:
        with pytest.raises(RetrievalValidationError):
            RetrievalQuery(text="  ")

    def test_query_with_no_index_match_returns_nothing(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = retriever.search(RetrievalQuery(text="kubernetes"))
        assert response.candidates_considered == 0
        assert response.results == ()


class TestRanking:
    def test_priority_beats_a_weaker_match(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(
            content="python", memory_type=MemoryType.OBSERVATION, confidence=0.1, importance=0.0
        )
        strong = memories.create_memory(
            content="python for backend work",
            memory_type=MemoryType.PROFILE,
            confidence=1.0,
            importance=1.0,
        )
        response = retriever.search(RetrievalQuery(text="python for backend work", top_k=5))
        assert response.results[0].memory.id == strong.id
        assert response.results[0].final_score > response.results[-1].final_score

    def test_freshness_matters(
        self, retriever: MemoryRetrievalService, memories: MemoryService, clock: ManualClock
    ) -> None:
        memories.create_memory(content="python backend", memory_type=MemoryType.FACT)
        clock.advance(days=200)
        fresh = memories.create_memory(content="python backend", memory_type=MemoryType.FACT)

        response = retriever.search(RetrievalQuery(text="python backend", top_k=5))
        assert response.results[0].memory.id == fresh.id
        assert response.results[0].final_score > response.results[1].final_score

    def test_scores_are_bounded_and_explained(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(
            content=PYTHON, memory_type=MemoryType.PREFERENCE, confidence=0.9, importance=0.8
        )
        response = retriever.search(RetrievalQuery(text="python backend"))
        for item in response.results:
            assert 0.0 <= item.final_score <= 1.0
            assert 0.0 <= item.relevance <= 1.0
            assert len(item.components) == 5
            assert sum(factor.contribution for factor in item.components) <= 1.0 + 1e-9
            assert "final " in item.explanation
            for factor in item.components:
                assert factor.explanation
                assert factor.contribution == pytest.approx(factor.weight * factor.value)

    def test_ranking_is_deterministic(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        for content in (PYTHON, TYPESCRIPT, NEAR_DUPLICATE):
            memories.create_memory(content=content, memory_type=MemoryType.PREFERENCE)
        first = retriever.search(RetrievalQuery(text="python backend work"))
        second = retriever.search(RetrievalQuery(text="python backend work"))
        assert [r.memory.id for r in first.results] == [r.memory.id for r in second.results]
        assert [r.final_score for r in first.results] == [r.final_score for r in second.results]

    def test_identical_candidates_break_ties_by_id(
        self, retriever: MemoryRetrievalService, repository: SqliteMemoryRepository
    ) -> None:
        first = repository.insert(
            _memory("11111111-1111-5111-8111-111111111111", PYTHON),
            change=VersionChange.CREATED,
        )
        second = repository.insert(
            _memory("22222222-2222-4222-8222-222222222222", PYTHON),
            change=VersionChange.CREATED,
        )
        response = retriever.search(RetrievalQuery(text="python backend work"))
        ids = [result.memory.id for result in response.results]
        assert ids == sorted(ids)
        assert set(ids) == {first.memory_id, second.memory_id}

    def test_ranking_weights_are_configurable(
        self, repository: SqliteMemoryRepository, clock: ManualClock, memories: MemoryService
    ) -> None:
        memories.create_memory(
            content="python", memory_type=MemoryType.OBSERVATION, confidence=0.0, importance=0.0
        )
        memories.create_memory(
            content="python for backend work",
            memory_type=MemoryType.OBSERVATION,
            confidence=1.0,
            importance=1.0,
        )
        confidence_first = MemoryRetrievalService(
            repository,
            clock=clock,
            ranking=RankingConfig(
                relevance_weight=0.1,
                priority_weight=0.1,
                confidence_weight=0.6,
                freshness_weight=0.1,
                type_weight=0.1,
            ),
        )
        relevance_first = MemoryRetrievalService(
            repository,
            clock=clock,
            ranking=RankingConfig(
                relevance_weight=0.8,
                priority_weight=0.05,
                confidence_weight=0.05,
                freshness_weight=0.05,
                type_weight=0.05,
            ),
        )
        by_confidence = confidence_first.search(RetrievalQuery(text="python for backend work"))
        by_relevance = relevance_first.search(RetrievalQuery(text="python for backend work"))
        assert by_confidence.results[0].final_score != by_relevance.results[0].final_score


class TestFilters:
    def test_type_filter(self, retriever: MemoryRetrievalService, memories: MemoryService) -> None:
        memories.create_memory(content="python backend", memory_type=MemoryType.TASK)
        memories.create_memory(content="python backend", memory_type=MemoryType.PREFERENCE)

        response = retriever.search(
            RetrievalQuery(
                text="python backend", filters=RetrievalFilters(memory_type=MemoryType.TASK)
            )
        )
        assert response.candidates_considered == 1
        assert response.results[0].memory.memory_type is MemoryType.TASK

    def test_confidence_filter(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content="python", memory_type=MemoryType.FACT, confidence=0.2)
        memories.create_memory(content="python", memory_type=MemoryType.FACT, confidence=0.9)

        response = retriever.search(
            RetrievalQuery(text="python", filters=RetrievalFilters(min_confidence=0.5))
        )
        assert response.candidates_considered == 1
        assert response.results[0].memory.confidence == 0.9

    def test_importance_filter(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content="python", memory_type=MemoryType.FACT, importance=0.1)
        memories.create_memory(content="python", memory_type=MemoryType.FACT, importance=0.9)

        response = retriever.search(
            RetrievalQuery(text="python", filters=RetrievalFilters(min_importance=0.5))
        )
        assert response.candidates_considered == 1
        assert response.results[0].memory.importance == 0.9

    def test_creation_window(
        self, retriever: MemoryRetrievalService, memories: MemoryService, clock: ManualClock
    ) -> None:
        old = memories.create_memory(content="python", memory_type=MemoryType.FACT)
        clock.advance(days=10)
        new = memories.create_memory(content="python", memory_type=MemoryType.FACT)

        response = retriever.search(
            RetrievalQuery(
                text="python",
                filters=RetrievalFilters(created_after=FIXED_START + timedelta(days=5)),
            )
        )
        assert [result.memory.id for result in response.results] == [new.id]
        assert old.id not in [result.memory.id for result in response.results]

    def test_invalid_filters_are_rejected(self, retriever: MemoryRetrievalService) -> None:
        with pytest.raises(RetrievalValidationError):
            RetrievalFilters(
                created_after=FIXED_START + timedelta(days=5), created_before=FIXED_START
            )


class TestDuplicateSuppression:
    def test_duplicates_are_suppressed(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        first = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        second = memories.create_memory(content=NEAR_DUPLICATE, memory_type=MemoryType.PREFERENCE)
        # Seed the M2 duplicate relationship that suppression relies on.
        memories.find_duplicates(first.id)
        assert memories.get_relationships(first.id)

        response = retriever.search(RetrievalQuery(text="python backend work", top_k=5))
        returned = [result.memory.id for result in response.results]
        assert first.id in returned
        assert second.id not in returned
        assert response.candidates_suppressed == 1
        suppressed = response.suppressed[0]
        assert suppressed.memory_id == second.id
        assert first.id in suppressed.reason
        assert suppressed.final_score < 1.0

    def test_nothing_is_deleted(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        first = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        second = memories.create_memory(content=NEAR_DUPLICATE, memory_type=MemoryType.PREFERENCE)
        memories.find_duplicates(first.id)
        retriever.search(RetrievalQuery(text="python backend work", top_k=1))

        assert memories.get_memory(second.id) is not None
        assert len(memories.get_history(second.id)) == 1

    def test_undetected_duplicates_are_both_returned(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        """Suppression only uses recorded M2 relationships, never a rescan."""
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        memories.create_memory(content=NEAR_DUPLICATE, memory_type=MemoryType.PREFERENCE)

        response = retriever.search(RetrievalQuery(text="python backend work", top_k=5))
        assert response.candidates_returned == 2
        assert response.candidates_suppressed == 0

    def test_suppressed_results_still_expose_evidence(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        first = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        second = memories.create_memory(content=NEAR_DUPLICATE, memory_type=MemoryType.PREFERENCE)
        memories.find_duplicates(first.id)

        response = retriever.search(RetrievalQuery(text="python backend work", top_k=5))
        assert response.suppressed[0].memory_id == second.id
        assert "duplicate_of" in response.suppressed[0].reason


class TestConflictAwareness:
    def test_winner_is_reported_first(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        loser = memories.create_memory(
            content="preferred backend language = Python",
            memory_type=MemoryType.FACT,
            importance=0.2,
        )
        winner = memories.create_memory(
            content="preferred backend language = TypeScript",
            memory_type=MemoryType.FACT,
            importance=0.9,
        )
        memories.detect_conflicts(winner.id)
        resolution = memories.resolve_conflict(winner.id)
        assert resolution.winner.id == winner.id
        assert resolution.loser.id == loser.id

        response = retriever.search(RetrievalQuery(text="preferred backend language", top_k=5))
        top = response.results[0]
        assert top.memory.id == winner.id
        assert top.conflicts
        assert top.conflicts[0].resolution == "supersedes_other"
        assert top.superseded_by is None

    def test_superseded_memory_is_demoted(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        loser = memories.create_memory(
            content="preferred backend language = Python",
            memory_type=MemoryType.FACT,
            importance=0.2,
        )
        winner = memories.create_memory(
            content="preferred backend language = TypeScript",
            memory_type=MemoryType.FACT,
            importance=0.9,
        )
        memories.detect_conflicts(winner.id)
        memories.resolve_conflict(winner.id)

        # Resolution archived the loser, so the default (ACTIVE) view is clean.
        response = retriever.search(RetrievalQuery(text="preferred backend language", top_k=10))
        assert [result.memory.id for result in response.results] == [winner.id]

        # Asking for every status shows both, with the winner still first and
        # the loser annotated with why it lost.
        everything = retriever.search(
            RetrievalQuery(
                text="preferred backend language",
                filters=RetrievalFilters(status=None),
                top_k=10,
            )
        )
        assert everything.results[0].memory.id == winner.id
        loser_result = next(result for result in everything.results if result.memory.id == loser.id)
        assert loser_result.superseded_by == winner.id
        assert loser_result.conflicts[0].resolution == "superseded_by_other"
        assert loser_result.final_score < everything.results[0].final_score

    def test_conflict_penalty_appears_in_the_explanation(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        left = memories.create_memory(
            content="preferred backend language = Python",
            memory_type=MemoryType.FACT,
            importance=0.9,
        )
        right = memories.create_memory(
            content="preferred backend language = TypeScript",
            memory_type=MemoryType.FACT,
            importance=0.5,
        )
        memories.detect_conflicts(right.id)
        # Resolve so that `left` wins: left has the higher importance.
        memories.resolve_conflict(left.id)

        response = retriever.search(RetrievalQuery(text="preferred backend language", top_k=10))
        winner = next(result for result in response.results if result.memory.id == left.id)
        assert winner.conflicts[0].resolution == "supersedes_other"

    def test_unresolved_conflicts_are_visible(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        left = memories.create_memory(
            content="preferred backend language = Python", memory_type=MemoryType.FACT
        )
        right = memories.create_memory(
            content="preferred backend language = TypeScript", memory_type=MemoryType.FACT
        )
        memories.detect_conflicts(right.id)

        response = retriever.search(
            RetrievalQuery(
                text="preferred backend language",
                filters=RetrievalFilters(status=None),
                top_k=10,
            )
        )
        results = {result.memory.id: result for result in response.results}
        assert results[left.id].conflicts[0].resolution == "unresolved"
        assert results[right.id].conflicts[0].resolution == "unresolved"


class TestSemanticRetrieval:
    def test_semantic_mode_reports_the_provider(
        self, semantic_retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = semantic_retriever.search(
            RetrievalQuery(text="python", mode=RetrievalMode.HYBRID)
        )
        assert response.semantic_available is True
        assert response.semantic_provider == HashedLexicalEmbeddingProvider.name
        assert response.semantic_model == HashedLexicalEmbeddingProvider.model
        assert response.retrieval_mode is RetrievalMode.HYBRID
        assert response.results[0].semantic_score is not None

    def test_hybrid_blends_both_scores(
        self, semantic_retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = semantic_retriever.search(
            RetrievalQuery(text="python", mode=RetrievalMode.HYBRID, semantic_weight=0.5)
        )
        result = response.results[0]
        assert result.relevance == pytest.approx(
            0.5 * result.lexical_score + 0.5 * (result.semantic_score or 0.0)
        )

    def test_semantic_only_ignores_lexical(
        self, semantic_retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(
            content="the build server is in hyderabad", memory_type=MemoryType.OBSERVATION
        )
        response = semantic_retriever.search(
            RetrievalQuery(text="python", mode=RetrievalMode.SEMANTIC)
        )
        assert response.candidates_considered == 1
        assert response.results[0].lexical_score == 0.0
        assert response.results[0].relevance == response.results[0].semantic_score

    def test_embeddings_are_cached(
        self, semantic_retriever: MemoryRetrievalService, memories: MemoryService, db_path: object
    ) -> None:
        created = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        semantic_retriever.search(RetrievalQuery(text="python", mode=RetrievalMode.HYBRID))
        assert semantic_retriever._store.count() == 1
        assert (
            semantic_retriever._store.get(
                created.id,
                provider=HashedLexicalEmbeddingProvider.name,
                model=HashedLexicalEmbeddingProvider.model,
            )
            is not None
        )

    def test_semantic_weight_is_configurable(
        self, semantic_retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        lexical_heavy = semantic_retriever.search(
            RetrievalQuery(text="python", mode=RetrievalMode.HYBRID, semantic_weight=0.0)
        )
        semantic_heavy = semantic_retriever.search(
            RetrievalQuery(text="python", mode=RetrievalMode.HYBRID, semantic_weight=1.0)
        )
        assert lexical_heavy.results[0].relevance != semantic_heavy.results[0].relevance

    def test_semantic_scores_stay_in_range(
        self, semantic_retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = semantic_retriever.search(
            RetrievalQuery(text="python", mode=RetrievalMode.SEMANTIC)
        )
        semantic_score = response.results[0].semantic_score
        assert semantic_score is not None
        assert 0.0 <= semantic_score <= 1.0


class TestDegradation:
    def test_semantic_without_a_provider_degrades(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = retriever.search(RetrievalQuery(text="python", mode=RetrievalMode.SEMANTIC))

        assert response.requested_mode is RetrievalMode.SEMANTIC
        assert response.retrieval_mode is RetrievalMode.LEXICAL
        assert response.semantic_available is False
        assert response.semantic_provider is None
        assert response.degraded_reason is not None
        assert "no embedding provider" in response.degraded_reason
        assert response.results[0].semantic_score is None

    def test_lexical_mode_never_degrades(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = retriever.search(RetrievalQuery(text="python", mode=RetrievalMode.LEXICAL))
        assert response.degraded_reason is None
        assert response.retrieval_mode is RetrievalMode.LEXICAL

    def test_unreachable_provider_degrades(
        self, repository: SqliteMemoryRepository, clock: ManualClock, memories: MemoryService
    ) -> None:
        from app.domain.embeddings import OllamaEmbeddingProvider

        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        retriever = MemoryRetrievalService(
            repository, clock=clock, embedding_provider=OllamaEmbeddingProvider(timeout_seconds=0.2)
        )
        response = retriever.search(RetrievalQuery(text="python", mode=RetrievalMode.HYBRID))
        assert response.semantic_available is False
        assert response.retrieval_mode is RetrievalMode.LEXICAL


class TestIndexManagement:
    def test_index_is_rebuilt_when_the_corpus_changes(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        assert retriever.search(RetrievalQuery(text="python")).candidates_considered == 0
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        assert retriever.search(RetrievalQuery(text="python")).candidates_considered == 1

        created = memories.create_memory(content=PYTHON + " again", memory_type=MemoryType.FACT)
        assert retriever.search(RetrievalQuery(text="python")).candidates_considered == 2
        memories.forget_memory(created.id)
        assert retriever.search(RetrievalQuery(text="python")).candidates_considered == 1

    def test_rebuild_is_explicitly_available(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        retriever.rebuild_index()
        assert retriever.search(RetrievalQuery(text="python")).candidates_considered == 1

    def test_repeated_queries_do_not_rebuild(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        retriever.search(RetrievalQuery(text="python"))
        signature = retriever._signature
        retriever.search(RetrievalQuery(text="python"))
        assert retriever._signature == signature


class TestExplain:
    def test_explain_returns_one_fully_explained_result(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        created = memories.create_memory(
            content=PYTHON, memory_type=MemoryType.PREFERENCE, confidence=0.9
        )
        result = retriever.explain(created.id, RetrievalQuery(text="python backend"))
        assert result.memory.id == created.id
        assert len(result.components) == 5
        assert result.final_score > 0.0
        assert "python backend" in result.explanation

    def test_explain_unknown_memory(self, retriever: MemoryRetrievalService) -> None:
        from app.domain.errors import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError):
            retriever.explain("00000000-0000-4000-8000-000000000000", RetrievalQuery(text="x"))

    def test_explain_reports_factors_by_name(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        created = memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        result = retriever.explain(created.id, RetrievalQuery(text="python"))
        names = [factor.name for factor in result.components]
        assert names == [
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
        ]


class TestMetadata:
    def test_response_metadata_is_reported(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        for index in range(3):
            memories.create_memory(
                content=f"python backend work {index}", memory_type=MemoryType.TASK
            )
        response = retriever.search(RetrievalQuery(text="python backend work", top_k=2))
        assert response.query == "python backend work"
        assert response.candidates_considered == 3
        assert response.candidates_returned == 2
        assert response.candidates_suppressed == 0
        assert response.latency_ms >= 0.0
        assert response.requested_mode is RetrievalMode.LEXICAL
        assert response.retrieval_mode is RetrievalMode.LEXICAL

    def test_latency_is_a_measurement_not_a_claim(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        memories.create_memory(content=PYTHON, memory_type=MemoryType.PREFERENCE)
        response = retriever.search(RetrievalQuery(text="python"))
        assert 0.0 <= response.latency_ms < 60_000.0


def _memory(memory_id: str, content: str) -> Memory:
    from tests.support import make_memory

    return make_memory(id=memory_id, content=content)
