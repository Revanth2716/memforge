"""Domain-level tests for the retrieval ranking model."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.errors import InvalidRetrievalConfigError, RetrievalValidationError
from app.domain.models import MemoryStatus, MemoryType
from app.domain.retrieval import (
    RETRIEVAL_TYPE_WEIGHTS,
    ConflictInfo,
    RankingConfig,
    RankingFactor,
    RetrievalFilters,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
    conflict_penalty_from,
    final_score,
    freshness,
    hybrid_relevance,
    idf,
    lexical_relevance,
    ranking_explanation,
    semantic_relevance,
    sort_key,
    type_preference,
)
from tests.support import FIXED_START, make_memory

LATER = FIXED_START + timedelta(days=15)


class TestQueryValidation:
    def test_defaults(self) -> None:
        query = RetrievalQuery(text="python")
        assert query.top_k == 10
        assert query.mode is RetrievalMode.LEXICAL
        assert query.filters.status is MemoryStatus.ACTIVE
        assert query.terms == ("python",)

    def test_terms_are_normalised_and_de_duplicated(self) -> None:
        query = RetrievalQuery(text="The user's preferences, the user!")
        assert query.terms == ("user", "preference")

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_query_is_rejected(self, text: str) -> None:
        with pytest.raises(RetrievalValidationError):
            RetrievalQuery(text=text)

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(RetrievalValidationError):
            RetrievalQuery(text="python", top_k=0)

    def test_semantic_weight_must_be_bounded(self) -> None:
        with pytest.raises(RetrievalValidationError):
            RetrievalQuery(text="python", semantic_weight=1.5)

    def test_query_only_of_stop_words_has_no_terms(self) -> None:
        assert RetrievalQuery(text="the and of").terms == ()


class TestFilters:
    def test_defaults_to_active_only(self) -> None:
        filters = RetrievalFilters()
        assert filters.status is MemoryStatus.ACTIVE
        assert filters.matches(make_memory()) is True

    def test_status_filter(self) -> None:
        filters = RetrievalFilters(status=MemoryStatus.ARCHIVED)
        assert filters.matches(make_memory()) is False
        assert filters.matches(make_memory(status=MemoryStatus.ARCHIVED)) is True

    def test_status_none_includes_everything(self) -> None:
        filters = RetrievalFilters(status=None)
        assert filters.matches(make_memory(status=MemoryStatus.DELETED)) is True

    def test_type_filter(self) -> None:
        filters = RetrievalFilters(memory_type=MemoryType.TASK)
        assert filters.matches(make_memory(memory_type=MemoryType.FACT)) is False
        assert filters.matches(make_memory(memory_type=MemoryType.TASK)) is True

    def test_confidence_and_importance_bounds(self) -> None:
        filters = RetrievalFilters(min_confidence=0.7, min_importance=0.8)
        assert filters.matches(make_memory(confidence=0.6, importance=0.9)) is False
        assert filters.matches(make_memory(confidence=0.7, importance=0.9)) is True
        assert filters.matches(make_memory(confidence=0.9, importance=0.7)) is False

    def test_creation_window(self) -> None:
        filters = RetrievalFilters(
            created_after=FIXED_START + timedelta(days=1),
            created_before=FIXED_START + timedelta(days=10),
        )
        assert filters.matches(make_memory()) is False
        inside = FIXED_START + timedelta(days=5)
        assert filters.matches(make_memory(created_at=inside, updated_at=inside)) is True
        outside = FIXED_START + timedelta(days=20)
        assert filters.matches(make_memory(created_at=outside, updated_at=outside)) is False

    def test_inverted_window_is_rejected(self) -> None:
        with pytest.raises(RetrievalValidationError):
            RetrievalFilters(
                created_after=FIXED_START + timedelta(days=10),
                created_before=FIXED_START,
            )

    def test_unbounded_scores_are_rejected(self) -> None:
        with pytest.raises(InvalidRetrievalConfigError):
            RetrievalFilters(min_confidence=1.5)


class TestRankingConfig:
    def test_defaults(self) -> None:
        config = RankingConfig()
        weights = config.component_weights
        assert set(weights) == {
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_component_weights_must_sum_to_one(self) -> None:
        with pytest.raises(InvalidRetrievalConfigError):
            RankingConfig(relevance_weight=0.9)

    def test_component_weights_must_be_bounded(self) -> None:
        with pytest.raises(InvalidRetrievalConfigError):
            RankingConfig(relevance_weight=1.5, priority_weight=-0.5)

    def test_penalties_must_be_bounded(self) -> None:
        with pytest.raises(InvalidRetrievalConfigError):
            RankingConfig(conflict_penalty_weight=1.5)
        with pytest.raises(InvalidRetrievalConfigError):
            RankingConfig(redundancy_penalty_weight=-0.1)
        with pytest.raises(InvalidRetrievalConfigError):
            RankingConfig(default_semantic_weight=2.0)

    def test_half_life_must_be_positive(self) -> None:
        with pytest.raises(InvalidRetrievalConfigError):
            RankingConfig(decay_half_life_days=0)

    def test_semantic_weight_resolution(self) -> None:
        config = RankingConfig(default_semantic_weight=0.4)
        assert config.semantic_weight_for(None) == 0.4
        assert config.semantic_weight_for(0.9) == 0.9

    def test_type_weights_are_documented(self) -> None:
        assert set(RETRIEVAL_TYPE_WEIGHTS) == set(MemoryType)
        assert (
            RETRIEVAL_TYPE_WEIGHTS[MemoryType.PROFILE]
            > RETRIEVAL_TYPE_WEIGHTS[MemoryType.OBSERVATION]
        )


class TestRelevance:
    def test_idf_is_positive_and_decreases_with_frequency(self) -> None:
        rare = idf(100, 1)
        common = idf(100, 50)
        assert rare > common > 0.0
        assert idf(100, 0) == 0.0

    def test_full_coverage_scores_one(self) -> None:
        score = lexical_relevance(
            ("python",),
            frozenset({"python"}),
            corpus_size=10,
            document_frequencies={"python": 3},
        )
        assert score == 1.0

    def test_missing_term_reduces_the_score(self) -> None:
        score = lexical_relevance(
            ("python", "java"),
            frozenset({"python"}),
            corpus_size=10,
            document_frequencies={"python": 1, "java": 1},
        )
        assert score == pytest.approx(0.5)

    def test_rare_terms_weigh_more(self) -> None:
        # Only the rare term is present: covering it is worth more than
        # covering a common term.
        covered_rare = lexical_relevance(
            ("rare", "common"),
            frozenset({"rare"}),
            corpus_size=100,
            document_frequencies={"rare": 1, "common": 90},
        )
        covered_common = lexical_relevance(
            ("rare", "common"),
            frozenset({"common"}),
            corpus_size=100,
            document_frequencies={"rare": 1, "common": 90},
        )
        assert covered_rare > covered_common
        assert 0.0 < covered_common < covered_rare < 1.0

    def test_terms_missing_from_the_corpus_are_ignored(self) -> None:
        score = lexical_relevance(
            ("python", "unicorn"),
            frozenset({"python"}),
            corpus_size=10,
            document_frequencies={"python": 2},
        )
        assert score == 1.0

    def test_empty_query_scores_zero(self) -> None:
        assert (
            lexical_relevance(
                (),
                frozenset({"python"}),
                corpus_size=10,
                document_frequencies={"python": 2},
            )
            == 0.0
        )

    def test_relevance_is_bounded(self) -> None:
        score = lexical_relevance(
            ("a", "b", "c"),
            frozenset({"a"}),
            corpus_size=10,
            document_frequencies={"a": 1, "b": 5, "c": 9},
        )
        assert 0.0 <= score <= 1.0

    def test_semantic_relevance_is_clamped(self) -> None:
        assert semantic_relevance(1.7) == 1.0
        assert semantic_relevance(-0.3) == 0.0
        assert semantic_relevance(0.42) == 0.42

    @pytest.mark.parametrize(
        ("lexical", "semantic", "weight", "expected"),
        [
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 0.0),
            (1.0, 1.0, 0.5, 1.0),
            (0.0, 1.0, 0.5, 0.5),
        ],
    )
    def test_hybrid_blend(
        self, lexical: float, semantic: float, weight: float, expected: float
    ) -> None:
        assert hybrid_relevance(lexical, semantic, semantic_weight=weight) == pytest.approx(
            expected
        )

    def test_hybrid_weight_is_validated(self) -> None:
        with pytest.raises(InvalidRetrievalConfigError):
            hybrid_relevance(0.5, 0.5, semantic_weight=1.5)


class TestComponents:
    def test_freshness_halves_every_half_life(self) -> None:
        memory = make_memory()
        assert freshness(memory, now=FIXED_START, half_life_days=10.0) == 1.0
        assert freshness(
            memory, now=FIXED_START + timedelta(days=10), half_life_days=10.0
        ) == pytest.approx(0.5)
        assert freshness(
            memory, now=FIXED_START + timedelta(days=20), half_life_days=10.0
        ) == pytest.approx(0.25)

    def test_freshness_never_exceeds_one(self) -> None:
        memory = make_memory()
        assert freshness(memory, now=FIXED_START - timedelta(days=5), half_life_days=10.0) == 1.0

    def test_type_preference_uses_the_config(self) -> None:
        config = RankingConfig()
        assert type_preference(make_memory(memory_type=MemoryType.PROFILE), config) == 1.0
        assert type_preference(make_memory(memory_type=MemoryType.OBSERVATION), config) == 0.4

    @pytest.mark.parametrize(
        ("resolutions", "expected"),
        [
            ([], 0.0),
            (["supersedes_other"], 0.0),
            (["unresolved"], 0.5),
            (["unresolved", "superseded_by_other"], 1.0),
            (["superseded_by_other"], 1.0),
        ],
    )
    def test_conflict_penalty(self, resolutions: list[str], expected: float) -> None:
        assert conflict_penalty_from(resolutions) == expected


class TestFinalScore:
    def _config(self) -> RankingConfig:
        return RankingConfig()

    def test_perfect_memory_scores_one(self) -> None:
        score = final_score(
            relevance=1.0,
            priority=1.0,
            confidence=1.0,
            freshness_value=1.0,
            type_value=1.0,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=0.0,
            config=self._config(),
        )
        assert score == pytest.approx(1.0)

    def test_worst_case_stays_non_negative(self) -> None:
        score = final_score(
            relevance=0.0,
            priority=0.0,
            confidence=0.0,
            freshness_value=0.0,
            type_value=0.0,
            conflict_penalty_value=1.0,
            redundancy_penalty_value=1.0,
            config=self._config(),
        )
        assert score == 0.0

    def test_conflict_penalty_reduces_the_score(self) -> None:
        clean = final_score(
            relevance=1.0,
            priority=1.0,
            confidence=1.0,
            freshness_value=1.0,
            type_value=1.0,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=0.0,
            config=self._config(),
        )
        penalised = final_score(
            relevance=1.0,
            priority=1.0,
            confidence=1.0,
            freshness_value=1.0,
            type_value=1.0,
            conflict_penalty_value=1.0,
            redundancy_penalty_value=0.0,
            config=self._config(),
        )
        assert penalised < clean
        assert penalised == pytest.approx(clean * 0.5)

    def test_redundancy_penalty_reduces_the_score(self) -> None:
        clean = final_score(
            relevance=0.5,
            priority=0.5,
            confidence=0.5,
            freshness_value=0.5,
            type_value=0.5,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=0.0,
            config=self._config(),
        )
        penalised = final_score(
            relevance=0.5,
            priority=0.5,
            confidence=0.5,
            freshness_value=0.5,
            type_value=0.5,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=1.0,
            config=self._config(),
        )
        assert penalised == pytest.approx(clean * 0.3)

    def test_weights_change_the_outcome(self) -> None:
        relevance_first = RankingConfig(
            relevance_weight=0.8,
            priority_weight=0.05,
            confidence_weight=0.05,
            freshness_weight=0.05,
            type_weight=0.05,
        )
        priority_first = RankingConfig(
            relevance_weight=0.05,
            priority_weight=0.8,
            confidence_weight=0.05,
            freshness_weight=0.05,
            type_weight=0.05,
        )
        relevant = final_score(
            relevance=1.0,
            priority=0.0,
            confidence=0.0,
            freshness_value=0.0,
            type_value=0.0,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=0.0,
            config=relevance_first,
        )
        prioritised = final_score(
            relevance=0.0,
            priority=1.0,
            confidence=0.0,
            freshness_value=0.0,
            type_value=0.0,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=0.0,
            config=priority_first,
        )
        assert relevant == pytest.approx(0.8)
        assert prioritised == pytest.approx(0.8)


class TestExplainability:
    def test_explanation_lists_every_factor(self) -> None:
        config = RankingConfig()
        factors = tuple(
            _factor(name)
            for name in (
                "relevance",
                "effective_priority",
                "confidence",
                "freshness",
                "type_preference",
            )
        )
        explanation = ranking_explanation(
            query="python",
            factors=factors,
            conflict_penalty_value=1.0,
            redundancy_penalty_value=1.0,
            config=config,
            final=0.15,
        )
        assert "final 0.1500" in explanation
        for name in (
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
        ):
            assert name in explanation
        assert "conflict penalty" in explanation
        assert "redundancy penalty" in explanation

    def test_explanation_without_penalties(self) -> None:
        config = RankingConfig()
        factors = (_factor("relevance", weight=0.45),)
        explanation = ranking_explanation(
            query="python",
            factors=factors,
            conflict_penalty_value=0.0,
            redundancy_penalty_value=0.0,
            config=config,
            final=0.45,
        )
        assert "penalty" not in explanation


def _factor(name: str, weight: float = 0.2, value: float = 1.0) -> RankingFactor:
    """Real ranking factor used by the explanation tests."""
    return RankingFactor(
        name=name,
        weight=weight,
        value=value,
        contribution=round(weight * value, 6),
        explanation=f"{name} explained",
    )


class TestSortKey:
    def _result(self, memory_id: str, score: float) -> RetrievalResult:
        from app.domain.retrieval import RetrievalResult

        return RetrievalResult(
            memory=make_memory(id=memory_id),
            lexical_score=score,
            semantic_score=None,
            relevance=score,
            final_score=score,
            components=(),
            explanation="",
            conflicts=(),
            superseded_by=None,
            duplicate_of=None,
            suppressed=False,
            suppression_reason=None,
        )

    def test_orders_by_score_then_id(self) -> None:
        high = self._result("11111111-1111-5111-8111-111111111111", 0.9)
        low = self._result("22222222-2222-4222-8222-222222222222", 0.4)
        tie_a = self._result("11111111-1111-5111-8111-111111111111", 0.5)
        tie_b = self._result("22222222-2222-4222-8222-222222222222", 0.5)

        ordered = sorted([low, high, tie_b, tie_a], key=sort_key)
        assert [item.memory.id for item in ordered] == [
            "11111111-1111-5111-8111-111111111111",
            "11111111-1111-5111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "22222222-2222-4222-8222-222222222222",
        ]


def test_conflict_info_is_immutable() -> None:
    info = ConflictInfo(
        target_memory_id="x",
        relationship="CONFLICTS_WITH",
        attribute="a",
        source_value="b",
        target_value="c",
        resolution="unresolved",
        resolution_reason=None,
    )
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass assignment
        info.resolution = "superseded_by_other"  # type: ignore[misc]
