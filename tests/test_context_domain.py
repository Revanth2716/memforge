"""M4 context domain tests: validation, utility, compression, ordering, budget.

Everything here is a pure function of (candidates, policy, request, budget), so
the compiler's decision logic is proved without a database, a clock or a server.
"""

from __future__ import annotations

import pytest

from app.domain.context import (
    CompressionStep,
    ContextCandidate,
    ContextPolicy,
    ContextRequest,
    ContextSection,
    SelectionReason,
    apply_admission_filters,
    block_sort_key,
    compress_content,
    cost_penalty,
    factor_value,
    order_blocks,
    render_block_line,
    render_context,
    render_section,
    score_candidate,
    select_within_budget,
    split_segments,
    suppress_redundant,
)
from app.domain.errors import ContextValidationError, InvalidContextConfigError
from app.domain.models import Memory, MemoryStatus, MemoryType
from app.domain.retrieval import (
    ConflictInfo,
    RankingFactor,
    RetrievalFilters,
    RetrievalMode,
    RetrievalResult,
)
from app.domain.tokens import TokenEstimatorConfig
from tests.support import make_memory

PYTHON = "User prefers Python for backend work."
TYPESCRIPT = "User is learning TypeScript for frontend work."


def build_result(
    memory: Memory,
    *,
    final_score: float = 0.5,
    relevance: float = 0.5,
    priority: float = 0.5,
    freshness: float = 0.5,
    type_preference: float = 0.5,
    duplicate_of: str | None = None,
    superseded_by: str | None = None,
    conflicts: tuple[ConflictInfo, ...] = (),
) -> RetrievalResult:
    """A retrieval result with explicit, controllable M3 signals."""
    factors = (
        RankingFactor("relevance", 0.45, relevance, 0.45 * relevance, "test relevance"),
        RankingFactor("effective_priority", 0.20, priority, 0.20 * priority, "test priority"),
        RankingFactor(
            "confidence", 0.15, memory.confidence, 0.15 * memory.confidence, "test confidence"
        ),
        RankingFactor("freshness", 0.10, freshness, 0.10 * freshness, "test freshness"),
        RankingFactor(
            "type_preference", 0.10, type_preference, 0.10 * type_preference, "test type"
        ),
    )
    return RetrievalResult(
        memory=memory,
        lexical_score=relevance,
        semantic_score=None,
        relevance=relevance,
        final_score=final_score,
        components=factors,
        explanation="test result",
        conflicts=conflicts,
        superseded_by=superseded_by,
        duplicate_of=duplicate_of,
        suppressed=False,
        suppression_reason=None,
    )


def candidate_for(
    content: str,
    memory_id: str,
    *,
    memory_type: MemoryType = MemoryType.PREFERENCE,
    confidence: float = 0.8,
    importance: float = 0.5,
    relevance: float = 0.8,
    priority: float = 0.5,
    freshness: float = 0.9,
    type_preference: float = 0.8,
    duplicate_of: str | None = None,
    superseded_by: str | None = None,
    policy: ContextPolicy | None = None,
) -> ContextCandidate:
    """Score a one-line memory into a context candidate."""
    memory = make_memory(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        importance=importance,
    )
    return score_candidate(
        build_result(
            memory,
            final_score=relevance,
            relevance=relevance,
            priority=priority,
            freshness=freshness,
            type_preference=type_preference,
            duplicate_of=duplicate_of,
            superseded_by=superseded_by,
        ),
        policy=policy,
    )


# -- policy -------------------------------------------------------------------


class TestPolicyValidation:
    def test_default_policy_is_valid(self) -> None:
        policy = ContextPolicy()
        assert policy.compress is True
        assert policy.cost_reference_tokens == 120

    def test_utility_weights_must_sum_to_one(self) -> None:
        with pytest.raises(InvalidContextConfigError, match=r"sum to 1\.0"):
            ContextPolicy(relevance_weight=0.5, priority_weight=0.5, confidence_weight=0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"relevance_weight": 1.5},
            {"relevance_weight": -0.1},
            {"cost_weight": 2.0},
            {"min_candidate_score": -0.5},
            {"cost_reference_tokens": 0},
            {"cost_reference_tokens": True},
            {"max_block_tokens": 0},
        ],
    )
    def test_out_of_range_knobs_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(InvalidContextConfigError):
            ContextPolicy(**kwargs)  # type: ignore[arg-type]

    def test_type_sections_must_cover_every_memory_type(self) -> None:
        with pytest.raises(InvalidContextConfigError, match="missing memory types"):
            ContextPolicy(type_sections={MemoryType.FACT: ContextSection.FACTS})

    def test_type_sections_must_map_to_sections(self) -> None:
        broken: dict[MemoryType, str] = dict.fromkeys(MemoryType, "FACTS")
        with pytest.raises(InvalidContextConfigError, match="not a ContextSection"):
            ContextPolicy(type_sections=broken)  # type: ignore[arg-type]

    def test_section_order_must_list_every_section_once(self) -> None:
        with pytest.raises(InvalidContextConfigError, match="exactly once"):
            ContextPolicy(section_order=(ContextSection.FACTS, ContextSection.FACTS))

    def test_section_helpers(self) -> None:
        policy = ContextPolicy()
        assert policy.section_for(MemoryType.PREFERENCE) is ContextSection.PROFILE
        assert policy.section_for(MemoryType.OBSERVATION) is ContextSection.OBSERVATIONS
        assert policy.section_rank(ContextSection.PROFILE) == 0
        assert policy.section_rank(ContextSection.OBSERVATIONS) == 4

    def test_formula_is_documented(self) -> None:
        formula = ContextPolicy().formula()
        assert "utility = value" in formula
        assert "0.4*relevance" in formula

    def test_utility_weights_view(self) -> None:
        weights = ContextPolicy().utility_weights
        assert set(weights) == {
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
        }


# -- request ------------------------------------------------------------------


class TestRequestValidation:
    def test_minimal_request_is_valid(self) -> None:
        request = ContextRequest(query="python backend", token_budget=100)
        assert request.top_k == 20
        assert request.retrieval_mode is RetrievalMode.LEXICAL

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_query_is_rejected(self, query: str) -> None:
        with pytest.raises(ContextValidationError, match="query"):
            ContextRequest(query=query, token_budget=100)

    @pytest.mark.parametrize("budget", [0, -1])
    def test_non_positive_budget_is_rejected(self, budget: int) -> None:
        with pytest.raises(ContextValidationError, match="token_budget"):
            ContextRequest(query="python", token_budget=budget)

    def test_boolean_budget_is_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="integer"):
            ContextRequest(query="python", token_budget=True)

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(ContextValidationError, match="top_k"):
            ContextRequest(query="python", token_budget=10, top_k=0)

    @pytest.mark.parametrize(
        "field", ["min_confidence", "min_importance", "min_candidate_score", "semantic_weight"]
    )
    def test_unit_bounds_are_validated(self, field: str) -> None:
        with pytest.raises(ContextValidationError, match=field):
            ContextRequest(query="python", token_budget=10, **{field: 1.5})  # type: ignore[arg-type]

    def test_empty_preference_list_is_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="preferred_memory_types"):
            ContextRequest(query="python", token_budget=10, preferred_memory_types=())

    def test_repeated_preference_is_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="repeat"):
            ContextRequest(
                query="python",
                token_budget=10,
                preferred_memory_types=(MemoryType.FACT, MemoryType.FACT),
            )

    def test_unknown_preference_type_is_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="unknown memory type"):
            ContextRequest(
                query="python",
                token_budget=10,
                preferred_memory_types=("NOPE",),  # type: ignore[arg-type]
            )

    def test_blank_ids_are_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="include_ids"):
            ContextRequest(query="python", token_budget=10, include_ids=(" ",))

    def test_overlapping_id_lists_are_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="must not overlap"):
            ContextRequest(query="python", token_budget=10, include_ids=("a",), exclude_ids=("a",))

    def test_max_block_tokens_is_validated(self) -> None:
        with pytest.raises(ContextValidationError, match="max_block_tokens"):
            ContextRequest(query="python", token_budget=10, max_block_tokens=0)

    def test_helpers(self) -> None:
        request = ContextRequest(
            query="python backend",
            token_budget=10,
            preferred_memory_types=(MemoryType.FACT, MemoryType.TASK),
            include_ids=("a",),
            exclude_ids=("b",),
        )
        assert request.prefers(MemoryType.FACT) is True
        assert request.prefers(MemoryType.EVENT) is False
        assert request.admits_id("a") is True
        assert request.admits_id("b") is False
        assert request.admits_id("c") is False
        # Two preferred types cannot be expressed as one M3 filter.
        assert request.single_preferred_type is None

    def test_single_preference_is_pushed_into_the_m3_filter(self) -> None:
        request = ContextRequest(
            query="python", token_budget=10, preferred_memory_types=(MemoryType.FACT,)
        )
        assert request.single_preferred_type is MemoryType.FACT

    def test_no_preference_admits_everything(self) -> None:
        request = ContextRequest(query="python", token_budget=10)
        assert request.prefers(MemoryType.EVENT) is True
        assert request.admits_id("anything") is True

    def test_translation_reuses_the_m3_contract(self) -> None:
        request = ContextRequest(
            query="python backend",
            token_budget=250,
            top_k=7,
            retrieval_mode=RetrievalMode.HYBRID,
            semantic_weight=0.25,
            min_confidence=0.4,
            min_importance=0.2,
            preferred_memory_types=(MemoryType.FACT,),
        )
        query = request.to_retrieval_query()
        assert isinstance(query.filters, RetrievalFilters)
        assert query.text == "python backend"
        assert query.top_k == 7
        assert query.mode is RetrievalMode.HYBRID
        assert query.semantic_weight == 0.25
        assert query.filters.memory_type is MemoryType.FACT
        assert query.filters.status is MemoryStatus.ACTIVE
        assert query.filters.min_confidence == 0.4
        assert query.filters.min_importance == 0.2

    def test_terms_are_normalised(self) -> None:
        assert ContextRequest(query="Python backend", token_budget=5).terms == (
            "python",
            "backend",
        )


# -- utility ------------------------------------------------------------------


class TestUtilityScoring:
    def test_value_is_a_bounded_convex_combination(self) -> None:
        candidate = candidate_for(PYTHON, "m-1")
        assert 0.0 <= candidate.value_score <= 1.0
        assert 0.0 <= candidate.utility_score <= candidate.value_score + 1e-9

    def test_memory_is_scored_with_its_own_signals(self) -> None:
        memory = make_memory(id="m-1", content=PYTHON, confidence=0.8, importance=0.5)
        result = build_result(memory, final_score=0.5, relevance=0.5)
        factors = {factor.name: factor.value for factor in result.components}

        assert factors["relevance"] == 0.5
        assert factors["effective_priority"] == 0.5
        assert factors["freshness"] == 0.5
        assert factors["type_preference"] == 0.5
        assert factors["confidence"] == 0.8

    def test_confidence_raises_utility(self) -> None:
        low = candidate_for(PYTHON, "m-1", confidence=0.1)
        high = candidate_for(PYTHON, "m-2", confidence=1.0)
        assert high.utility_score > low.utility_score

    def test_importance_raises_utility_through_priority(self) -> None:
        low = candidate_for(PYTHON, "m-1", importance=0.0, priority=0.0)
        high = candidate_for(PYTHON, "m-2", importance=1.0, priority=1.0)
        assert high.utility_score > low.utility_score

    def test_freshness_raises_utility(self) -> None:
        stale = candidate_for(PYTHON, "m-1", freshness=0.0)
        fresh = candidate_for(PYTHON, "m-2", freshness=1.0)
        assert fresh.utility_score > stale.utility_score

    def test_memory_type_preference_raises_utility(self) -> None:
        weak = candidate_for(PYTHON, "m-1", type_preference=0.0)
        strong = candidate_for(PYTHON, "m-2", type_preference=1.0)
        assert strong.utility_score > weak.utility_score

    def test_token_cost_reduces_utility(self) -> None:
        short = candidate_for("Python.", "m-1")
        long = candidate_for("Python " * 200, "m-2")
        assert long.estimated_tokens > short.estimated_tokens
        assert long.value_score == pytest.approx(short.value_score)
        assert long.utility_score < short.utility_score

    def test_relevance_versus_cost_trade_off(self) -> None:
        cheap_relevant = candidate_for(PYTHON, "m-1", relevance=0.7)
        expensive_irrelevant = candidate_for(TYPESCRIPT * 20, "m-2", relevance=0.2)
        assert expensive_irrelevant.utility_score < cheap_relevant.utility_score

    def test_cost_weight_zero_makes_size_irrelevant(self) -> None:
        policy = ContextPolicy(cost_weight=0.0)
        short = candidate_for(PYTHON, "m-1", policy=policy)
        long = candidate_for(PYTHON * 50, "m-2", policy=policy)
        assert short.cost_penalty != long.cost_penalty
        assert short.utility_score == long.utility_score

    def test_cost_penalty_is_bounded_and_saturating(self) -> None:
        policy = ContextPolicy()
        assert cost_penalty(0, policy) == 0.0
        assert cost_penalty(policy.cost_reference_tokens, policy) == 1.0
        assert cost_penalty(10 * policy.cost_reference_tokens, policy) == 1.0
        assert cost_penalty(60, policy) == 0.5

    def test_every_factor_is_explained(self) -> None:
        candidate = candidate_for(PYTHON, "m-1")
        names = {factor.name for factor in candidate.factors}
        assert names == {
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
            "cost_penalty",
        }
        assert all(factor.explanation for factor in candidate.factors)

    def test_factor_value_falls_back_to_zero(self) -> None:
        assert factor_value(build_result(make_memory()), "does_not_exist") == 0.0

    def test_candidate_exposes_its_memory(self) -> None:
        candidate = candidate_for(PYTHON, "m-1")
        assert candidate.memory_id == "m-1"
        assert candidate.rendered_content == PYTHON


# -- compression --------------------------------------------------------------


class TestCompression:
    def test_whitespace_is_normalised(self) -> None:
        result = compress_content("a   b\n\nc  ")
        assert result.compressed_text == "a b c"
        assert CompressionStep.WHITESPACE in result.applied
        assert result.original_text == "a   b\n\nc  "
        assert result.changed is True

    def test_the_estimator_is_whitespace_insensitive(self) -> None:
        # Whitespace normalisation tidies the prompt but cannot change the token
        # count: the estimator already normalises before counting. Claiming a
        # saving here would be an invented metric.
        result = compress_content("a   b\n\nc  ")
        assert result.original_tokens == result.compressed_tokens
        assert result.tokens_saved == 0

    def test_compression_can_be_disabled(self) -> None:
        result = compress_content("a   b", enabled=False)
        assert result.compressed_text == "a   b"
        assert result.applied == ()
        assert result.changed is False

    def test_content_is_never_reworded(self) -> None:
        text = "User prefers Python; TypeScript is only for the frontend."
        result = compress_content(text)
        assert result.compressed_text in result.original_text or text in result.compressed_text
        assert "prefers Python" in result.compressed_text

    def test_segmentation(self) -> None:
        assert split_segments("One. Two! Three?") == ("One.", "Two!", "Three?")
        assert split_segments("line one\nline two") == ("line one", "line two")
        assert split_segments("no boundary 3.14 stays") == ("no boundary 3.14 stays",)
        assert split_segments("a; b") == ("a;", "b")
        assert split_segments("   ") == ()

    def test_long_memory_keeps_the_query_relevant_sentences(self) -> None:
        text = (
            "Kubernetes runs the staging cluster. "
            "User prefers Python for backend work. "
            "The office coffee machine is broken."
        )
        result = compress_content(text, query_terms=("python", "backend"), max_block_tokens=8)
        assert CompressionStep.SENTENCES in result.applied
        assert "Python" in result.compressed_text
        assert "coffee machine" not in result.compressed_text
        assert result.truncated is True
        assert result.retained_segments == (1,)

    def test_kept_sentences_stay_in_their_original_order(self) -> None:
        text = "Third about python. First unrelated. Second about backend."
        result = compress_content(text, query_terms=("python", "backend"), max_block_tokens=12)
        assert result.compressed_text == "Third about python. Second about backend."
        assert result.compressed_text.index("Third") < result.compressed_text.index("Second")

    def test_single_sentence_memory_is_never_split(self) -> None:
        text = "A single very long sentence that clearly cannot be reduced at all."
        result = compress_content(text, query_terms=("sentence",), max_block_tokens=2)
        assert result.compressed_text == text
        assert CompressionStep.SENTENCES not in result.applied

    def test_ceiling_is_a_soft_target_not_a_budget_violation(self) -> None:
        text = "One long sentence about python."
        result = compress_content(text, query_terms=("python",), max_block_tokens=1)
        assert result.compressed_tokens >= 1

    def test_policy_ceiling_is_used_by_default(self) -> None:
        policy = ContextPolicy(max_block_tokens=6)
        text = "First about python. Second about backend. Third about testing."
        result = compress_content(text, query_terms=("python",), policy=policy)
        assert CompressionStep.SENTENCES in result.applied

    def test_compression_keeps_the_original_estimate(self) -> None:
        text = "First about python. Second about backend. Third about rest."
        result = compress_content(text, query_terms=("python",), max_block_tokens=8)
        assert result.original_tokens > result.compressed_tokens
        assert result.tokens_saved == result.original_tokens - result.compressed_tokens
        assert result.applied == (CompressionStep.SENTENCES,)

    def test_the_most_relevant_sentence_survives_an_impossible_ceiling(self) -> None:
        # A ceiling smaller than any single sentence cannot be met; compression
        # must never empty the memory out.
        text = "Kubernetes runs staging. User prefers Python. Coffee is broken."
        result = compress_content(text, query_terms=("python",), max_block_tokens=1)
        assert result.compressed_text == "User prefers Python."
        assert result.truncated is True


# -- admission ----------------------------------------------------------------


class TestAdmissionFilters:
    def setup_method(self) -> None:
        self.policy = ContextPolicy()
        self.request = ContextRequest(query="python", token_budget=100)

    def test_explicit_exclusion(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1")]
        request = ContextRequest(query="python", token_budget=100, exclude_ids=("m-1",))
        kept, rejected = apply_admission_filters(
            candidates, request=request, policy=self.policy, token_budget=100
        )
        assert kept == []
        assert rejected[0].reason is SelectionReason.EXCLUDED_BY_ID

    def test_include_list_is_an_allow_list(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1"), candidate_for(TYPESCRIPT, "m-2")]
        request = ContextRequest(query="python", token_budget=100, include_ids=("m-2",))
        kept, rejected = apply_admission_filters(
            candidates, request=request, policy=self.policy, token_budget=100
        )
        assert [candidate.memory_id for candidate in kept] == ["m-2"]
        assert rejected[0].reason is SelectionReason.NOT_IN_INCLUDE_IDS

    def test_memory_type_preference_filters_after_retrieval(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1", memory_type=MemoryType.FACT),
            candidate_for(TYPESCRIPT, "m-2", memory_type=MemoryType.EVENT),
        ]
        request = ContextRequest(
            query="python",
            token_budget=100,
            preferred_memory_types=(MemoryType.FACT, MemoryType.TASK),
        )
        kept, rejected = apply_admission_filters(
            candidates, request=request, policy=self.policy, token_budget=100
        )
        assert [candidate.memory_id for candidate in kept] == ["m-1"]
        assert rejected[0].reason is SelectionReason.MEMORY_TYPE_FILTERED

    def test_minimum_score_uses_the_request_override(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1", relevance=0.2)]
        request = ContextRequest(query="python", token_budget=100, min_candidate_score=0.9)
        kept, rejected = apply_admission_filters(
            candidates, request=request, policy=self.policy, token_budget=100
        )
        assert kept == []
        assert rejected[0].reason is SelectionReason.BELOW_MINIMUM_SCORE

    def test_minimum_score_falls_back_to_the_policy(self) -> None:
        policy = ContextPolicy(min_candidate_score=0.95)
        candidates = [candidate_for(PYTHON, "m-1", relevance=0.2)]
        kept, rejected = apply_admission_filters(
            candidates, request=self.request, policy=policy, token_budget=100
        )
        assert kept == []
        assert rejected[0].reason is SelectionReason.BELOW_MINIMUM_SCORE

    def test_rejections_report_the_full_budget(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1")]
        request = ContextRequest(query="python", token_budget=77, exclude_ids=("m-1",))
        _, rejected = apply_admission_filters(
            candidates, request=request, policy=self.policy, token_budget=77
        )
        assert rejected[0].remaining_tokens == 77

    def test_admitted_candidates_produce_no_rejection(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1")]
        kept, rejected = apply_admission_filters(
            candidates, request=self.request, policy=self.policy, token_budget=100
        )
        assert len(kept) == 1
        assert rejected == []


# -- redundancy ---------------------------------------------------------------


class TestRedundancySuppression:
    def test_superseded_memory_is_suppressed(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1", superseded_by="m-2"),
            candidate_for(TYPESCRIPT, "m-2"),
        ]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert [candidate.memory_id for candidate in kept] == ["m-2"]
        assert suppressed[0].reason is SelectionReason.CONFLICT_SUPERSEDED
        assert suppressed[0].related_memory_id == "m-2"
        assert "m-2" in suppressed[0].detail

    def test_the_conflict_winner_is_kept(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1"), candidate_for(PYTHON, "m-2")]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert len(kept) == 2
        assert suppressed == []

    def test_duplicate_is_suppressed_in_favour_of_the_representative(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1"),
            candidate_for(PYTHON, "m-2", duplicate_of="m-1", relevance=0.2),
        ]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert [candidate.memory_id for candidate in kept] == ["m-1"]
        assert suppressed[0].reason is SelectionReason.DUPLICATE_OF
        assert suppressed[0].related_memory_id == "m-1"

    def test_representative_is_the_higher_utility_member(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1", relevance=0.2, duplicate_of="m-2"),
            candidate_for(PYTHON, "m-2", relevance=0.9),
        ]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert [candidate.memory_id for candidate in kept] == ["m-2"]
        assert suppressed[0].memory_id == "m-1"

    def test_duplicate_chain_is_resolved_transitively(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1", relevance=0.9),
            candidate_for(PYTHON, "m-2", relevance=0.5, duplicate_of="m-1"),
            candidate_for(PYTHON, "m-3", relevance=0.1, duplicate_of="m-2"),
        ]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert [candidate.memory_id for candidate in kept] == ["m-1"]
        assert [entry.related_memory_id for entry in suppressed] == ["m-1", "m-1"]

    def test_duplicate_of_an_absent_partner_is_kept(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1", duplicate_of="not-a-candidate")]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert len(kept) == 1
        assert suppressed == []

    def test_conflict_beats_duplicate_precedence(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1", superseded_by="m-2", duplicate_of="m-2"),
        ]
        kept, suppressed = suppress_redundant(candidates, token_budget=100)
        assert kept == []
        assert suppressed[0].reason is SelectionReason.CONFLICT_SUPERSEDED

    def test_identical_utilities_break_ties_by_identifier(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-b", duplicate_of="m-a"),
            candidate_for(PYTHON, "m-a"),
        ]
        kept, _ = suppress_redundant(candidates, token_budget=100)
        assert [candidate.memory_id for candidate in kept] == ["m-a"]


# -- budget -------------------------------------------------------------------


class TestBudgetSelection:
    def test_everything_fits(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1"), candidate_for(TYPESCRIPT, "m-2")]
        selected, trace = select_within_budget(candidates, token_budget=1000)
        assert len(selected) == 2
        assert {entry.reason for entry in trace} == {SelectionReason.SELECTED}

    def test_ordering_is_utility_first(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-1", relevance=0.2),
            candidate_for(PYTHON, "m-2", relevance=0.95),
        ]
        selected, _ = select_within_budget(candidates, token_budget=1000)
        assert [candidate.memory_id for candidate in selected] == ["m-2", "m-1"]

    def test_identical_utilities_break_ties_by_identifier(self) -> None:
        candidates = [
            candidate_for(PYTHON, "m-b"),
            candidate_for(PYTHON, "m-a"),
        ]
        selected, _ = select_within_budget(candidates, token_budget=1000)
        assert [candidate.memory_id for candidate in selected] == ["m-a", "m-b"]

    def test_exact_budget_is_respected(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1")]
        cost = candidates[0].estimated_tokens
        selected, trace = select_within_budget(candidates, token_budget=cost)
        assert len(selected) == 1
        assert trace[0].remaining_tokens == 0

    def test_one_token_short_of_the_cost_rejects_the_candidate(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1")]
        cost = candidates[0].estimated_tokens
        selected, trace = select_within_budget(candidates, token_budget=cost - 1)
        assert selected == []
        assert trace[0].reason is SelectionReason.INSUFFICIENT_BUDGET
        assert trace[0].remaining_tokens == cost - 1

    def test_every_candidate_too_large_selects_nothing(self) -> None:
        candidates = [candidate_for(PYTHON * 100, "m-1"), candidate_for(TYPESCRIPT * 100, "m-2")]
        selected, trace = select_within_budget(candidates, token_budget=5)
        assert selected == []
        assert all(entry.reason is SelectionReason.INSUFFICIENT_BUDGET for entry in trace)

    def test_a_skipped_candidate_does_not_stop_the_remaining_ones(self) -> None:
        big = candidate_for("python " * 60, "m-big")
        small = candidate_for("ok", "m-small")
        candidates = [big, small]
        # Order them so the expensive candidate is examined first.
        assert big.utility_score != small.utility_score
        budget = small.estimated_tokens
        selected, trace = select_within_budget(candidates, token_budget=budget)
        assert [candidate.memory_id for candidate in selected] == ["m-small"]
        reasons = {entry.memory_id: entry.reason for entry in trace}
        assert reasons["m-small"] is SelectionReason.SELECTED
        assert len(trace) == 2

    def test_many_small_candidates_fill_the_budget(self) -> None:
        candidates = [candidate_for(f"memory {index}", f"m-{index:02d}") for index in range(8)]
        cost = candidates[0].estimated_tokens
        selected, _ = select_within_budget(candidates, token_budget=cost * 8)
        assert len(selected) == 8

    def test_budget_is_never_overspent(self) -> None:
        candidates = [candidate_for(PYTHON, f"m-{index}") for index in range(6)]
        for budget in range(1, 80):
            selected, _ = select_within_budget(candidates, token_budget=budget)
            assert sum(item.estimated_tokens for item in selected) <= budget

    def test_zero_budget_is_rejected(self) -> None:
        with pytest.raises(ContextValidationError, match="token_budget"):
            select_within_budget([], token_budget=0)

    def test_empty_candidate_set(self) -> None:
        selected, trace = select_within_budget([], token_budget=100)
        assert selected == []
        assert trace == []

    def test_insufficient_budget_trace_explains_the_numbers(self) -> None:
        candidates = [candidate_for(PYTHON, "m-1")]
        _, trace = select_within_budget(candidates, token_budget=1)
        assert trace[0].estimated_tokens == candidates[0].estimated_tokens
        assert trace[0].remaining_tokens == 1
        assert "exceeds the remaining budget" in trace[0].detail


# -- ordering and rendering ---------------------------------------------------


class TestOrdering:
    def test_sections_come_before_utility(self) -> None:
        policy = ContextPolicy()
        candidates = [
            candidate_for(TYPESCRIPT, "m-obs", memory_type=MemoryType.OBSERVATION, relevance=1.0),
            candidate_for(PYTHON, "m-pro", memory_type=MemoryType.PROFILE, relevance=0.1),
        ]
        blocks = order_blocks(candidates, policy=policy)
        assert [block.section for block in blocks] == [
            ContextSection.PROFILE,
            ContextSection.OBSERVATIONS,
        ]
        assert [block.order for block in blocks] == [0, 1]

    def test_within_a_section_utility_wins(self) -> None:
        policy = ContextPolicy()
        candidates = [
            candidate_for(PYTHON, "m-a", memory_type=MemoryType.FACT, relevance=0.2),
            candidate_for(PYTHON, "m-b", memory_type=MemoryType.FACT, relevance=0.9),
        ]
        blocks = order_blocks(candidates, policy=policy)
        assert [block.memory_id for block in blocks] == ["m-b", "m-a"]

    def test_ordering_is_independent_of_input_order(self) -> None:
        policy = ContextPolicy()
        candidates = [
            candidate_for(PYTHON, "m-a", memory_type=MemoryType.TASK),
            candidate_for(PYTHON, "m-b", memory_type=MemoryType.PROFILE),
            candidate_for(PYTHON, "m-c", memory_type=MemoryType.FACT),
        ]
        first = order_blocks(candidates, policy=policy)
        second = order_blocks(list(reversed(candidates)), policy=policy)
        assert [block.memory_id for block in first] == [block.memory_id for block in second]
        assert [block.order for block in first] == [0, 1, 2]

    def test_blocks_retain_full_provenance(self) -> None:
        policy = ContextPolicy()
        candidate = candidate_for(PYTHON, "m-1", memory_type=MemoryType.PROFILE)
        block = order_blocks([candidate], policy=policy)[0]
        assert block.memory_id == "m-1"
        assert block.content == PYTHON
        assert block.source == "user:test"
        assert block.confidence == 0.8
        assert block.section_rank == 0
        assert block.retrieval_score == candidate.retrieval_score
        assert block.utility_score == candidate.utility_score
        assert block.compression.original_text == PYTHON

    def test_block_sort_key_is_public_and_stable(self) -> None:
        policy = ContextPolicy()
        block = order_blocks([candidate_for(PYTHON, "m-1")], policy=policy)[0]
        assert block_sort_key(block) == (
            policy.section_rank(block.section),
            -block.utility_score,
            "m-1",
        )

    def test_custom_section_order_is_honoured(self) -> None:
        policy = ContextPolicy(
            section_order=(
                ContextSection.OBSERVATIONS,
                ContextSection.PROFILE,
                ContextSection.FACTS,
                ContextSection.TASKS,
                ContextSection.RECENT_EVENTS,
            )
        )
        candidates = [
            candidate_for(PYTHON, "m-a", memory_type=MemoryType.PROFILE),
            candidate_for(PYTHON, "m-b", memory_type=MemoryType.OBSERVATION),
        ]
        blocks = order_blocks(candidates, policy=policy)
        assert [block.memory_id for block in blocks] == ["m-b", "m-a"]


class TestRendering:
    def test_block_line_is_bulleted(self) -> None:
        assert render_block_line("hello") == "- hello"

    def test_section_has_a_underline(self) -> None:
        rendered = render_section("PROFILE", ["- a", "- b"])
        assert rendered == "PROFILE\n-------\n- a\n- b"

    def test_context_groups_by_section(self) -> None:
        policy = ContextPolicy()
        blocks = order_blocks(
            [
                candidate_for(PYTHON, "m-a", memory_type=MemoryType.PROFILE),
                candidate_for(PYTHON, "m-b", memory_type=MemoryType.FACT),
            ],
            policy=policy,
        )
        rendered = render_context(blocks, policy=policy)
        assert rendered.startswith("PROFILE\n-------\n- User prefers")
        assert "FACTS\n-----" in rendered

    def test_empty_sections_are_omitted(self) -> None:
        policy = ContextPolicy()
        blocks = order_blocks(
            [candidate_for(PYTHON, "m-a", memory_type=MemoryType.FACT)], policy=policy
        )
        assert "PROFILE" not in render_context(blocks, policy=policy)

    def test_empty_context_renders_as_empty_text(self) -> None:
        assert render_context([], policy=ContextPolicy()) == ""

    def test_every_rendered_line_maps_back_to_a_block(self) -> None:
        policy = ContextPolicy()
        blocks = order_blocks(
            [
                candidate_for(PYTHON, "m-a", memory_type=MemoryType.PROFILE),
                candidate_for(TYPESCRIPT, "m-b", memory_type=MemoryType.OBSERVATION),
            ],
            policy=policy,
        )
        rendered = render_context(blocks, policy=policy)
        for block in blocks:
            assert f"- {block.rendered_content}" in rendered


class TestSelectionReasons:
    def test_only_selected_is_an_admission(self) -> None:
        admissions = [reason for reason in SelectionReason if reason.is_admission]
        assert admissions == [SelectionReason.SELECTED]

    def test_every_reason_is_documented(self) -> None:
        for reason in SelectionReason:
            assert reason.description.endswith(".") or "…" not in reason.description
            assert reason.description


class TestSections:
    def test_every_memory_type_maps_to_a_section(self) -> None:
        policy = ContextPolicy()
        for memory_type in MemoryType:
            assert isinstance(policy.section_for(memory_type), ContextSection)

    def test_headings(self) -> None:
        assert ContextSection.RECENT_EVENTS.heading == "RECENT EVENTS"
        assert ContextSection.FACTS.heading == "FACTS"

    def test_estimator_override_flows_into_scoring(self) -> None:
        estimator = TokenEstimatorConfig(chars_per_token=1.0)
        memory = make_memory(id="m-1", content=PYTHON)
        default = score_candidate(build_result(memory))
        conservative = score_candidate(build_result(memory), estimator=estimator)
        assert conservative.estimated_tokens > default.estimated_tokens
