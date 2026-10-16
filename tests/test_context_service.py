"""Context compiler service tests (M4).

These exercise the real composition: M3 retrieval -> budgeted selection ->
ordering -> provenance, over a real SQLite database with a frozen clock.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from app.domain.context import (
    ContextPolicy,
    ContextRequest,
    ContextResult,
    ContextSection,
    SelectionReason,
)
from app.domain.embeddings import HashedLexicalEmbeddingProvider
from app.domain.errors import ContextValidationError, MemoryNotFoundError
from app.domain.ids import SequentialIdGenerator
from app.domain.models import Memory, MemoryStatus, MemoryType
from app.domain.retrieval import RetrievalMode
from app.domain.time import ManualClock
from app.domain.tokens import TokenEstimatorConfig
from app.domain.updates import MemoryUpdate
from app.services.context_service import ContextCompilerService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START

PROFILE = "User prefers Python for backend work."
DUPLICATE = "The user prefers python for backend development work"
OBSERVATION = "Office coffee machine on floor two is broken."
CONFLICT_OLD = "preferred backend language = Python"
CONFLICT_NEW = "preferred backend language = TypeScript"


@pytest.fixture
def memories(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryService:
    """A deterministic memory service over the test repository."""
    return MemoryService(repository, clock=clock, id_generator=SequentialIdGenerator(prefix="ctx"))


@pytest.fixture
def retriever(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryRetrievalService:
    """Lexical-only retrieval (the default, fully offline path)."""
    return MemoryRetrievalService(repository, clock=clock)


@pytest.fixture
def compiler(retriever: MemoryRetrievalService) -> ContextCompilerService:
    """The M4 compiler, composing the M3 retriever."""
    return ContextCompilerService(retriever)


def compile_for(
    compiler: ContextCompilerService,
    query: str = "python backend language",
    budget: int = 400,
    **overrides: Any,
) -> ContextResult:
    """Compile with a small, readable default request."""
    return compiler.compile(ContextRequest(query=query, token_budget=budget, **overrides))


def populate(memories: MemoryService) -> dict[str, Memory]:
    """A realistic corpus: a profile, a near-duplicate, an observation."""
    created: dict[str, Memory] = {}
    created["profile"] = memories.create_memory(
        content=PROFILE,
        memory_type=MemoryType.PROFILE,
        source="user:demo",
        confidence=0.95,
        importance=0.9,
    )
    created["duplicate"] = memories.create_memory(
        content=DUPLICATE,
        memory_type=MemoryType.OBSERVATION,
        source="agent:note",
        confidence=0.5,
        importance=0.3,
    )
    created["observation"] = memories.create_memory(
        content=OBSERVATION,
        memory_type=MemoryType.OBSERVATION,
        source="agent:note",
        confidence=0.4,
        importance=0.2,
    )
    memories.find_duplicates(created["profile"].id)
    return created


class TestHappyPath:
    def test_compiles_a_context_within_budget(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler)

        assert result.within_budget is True
        assert result.estimated_tokens <= result.token_budget
        assert result.remaining_tokens == result.token_budget - result.estimated_tokens
        assert result.selected_count == len(result.blocks)
        assert result.suppressed_count == len(result.suppressed_candidates)
        assert result.selected_count > 0

    def test_result_shape_matches_the_documented_contract(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler)

        assert result.query == "python backend language"
        assert result.token_budget == 400
        assert result.candidates_considered == len(result.selection_trace)
        assert result.retrieval_mode is RetrievalMode.LEXICAL
        assert result.context_text.strip()

    def test_every_candidate_has_exactly_one_decision(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler)

        identifiers = [entry.memory_id for entry in result.selection_trace]
        assert len(identifiers) == len(set(identifiers))
        assert set(identifiers) == {block.memory_id for block in result.blocks} | {
            entry.memory_id for entry in result.suppressed_candidates
        }

    def test_blocks_trace_back_to_stored_memories(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler)

        for block in result.blocks:
            stored = memories.get_memory(block.memory_id)
            assert stored is not None
            assert block.content == stored.content
            assert block.source == stored.source
            assert block.confidence == stored.confidence
            assert block.memory_type == stored.memory_type

    def test_no_content_appears_without_provenance(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler)

        known = {block.rendered_content for block in result.blocks}
        for line in result.context_text.splitlines():
            if line.startswith("- "):
                assert line[2:] in known

    def test_explanation_states_the_estimator_and_heuristics(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        explanation = compile_for(compiler).explanation

        assert explanation.estimator == "memforge-char-word/v1"
        assert "utility = value" in explanation.utility_formula
        assert explanation.ordering_policy == "section-then-utility/v1"
        assert any("not provider tokenisations" in note for note in explanation.notes)
        assert any("no LLM" in note for note in explanation.notes)

    def test_empty_corpus_returns_an_empty_context(self, compiler: ContextCompilerService) -> None:
        result = compile_for(compiler)

        assert result.candidates_considered == 0
        assert result.blocks == ()
        assert result.context_text == ""
        assert result.estimated_tokens == 0
        assert result.remaining_tokens == result.token_budget
        assert result.within_budget is True

    def test_query_without_matches_returns_an_empty_context(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler, query="kubernetes helm charts")

        assert result.candidates_considered == 0
        assert result.blocks == ()


class TestBudgetEnforcement:
    def _corpus(self, memories: MemoryService) -> None:
        for index in range(6):
            memories.create_memory(
                content=f"python backend note number {index}",
                memory_type=MemoryType.FACT,
                confidence=0.9,
                importance=0.7,
            )

    @pytest.mark.parametrize("budget", [1, 5, 12, 25, 60, 150, 400, 5000])
    def test_the_hard_limit_always_holds(
        self, compiler: ContextCompilerService, memories: MemoryService, budget: int
    ) -> None:
        self._corpus(memories)
        result = compile_for(compiler, budget=budget)

        assert result.estimated_tokens <= budget
        assert result.remaining_tokens >= 0
        assert result.within_budget is True

    def test_tiny_budget_selects_nothing_but_explains_why(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        self._corpus(memories)
        result = compile_for(compiler, budget=1)

        assert result.blocks == ()
        assert result.suppressed_count > 0
        assert all(
            entry.reason is SelectionReason.INSUFFICIENT_BUDGET
            for entry in result.suppressed_candidates
        )

    def test_a_generous_budget_selects_everything(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        self._corpus(memories)
        result = compile_for(compiler, budget=5000)

        assert result.selected_count == result.candidates_considered
        assert result.suppressed_count == 0

    def test_exact_fit_is_reported_as_zero_remaining(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        self._corpus(memories)
        generous = compile_for(compiler, budget=5000)
        exact = compile_for(compiler, budget=generous.estimated_tokens)

        assert exact.estimated_tokens == generous.estimated_tokens
        assert exact.remaining_tokens == 0
        assert exact.within_budget is True

    def test_a_skipped_memory_does_not_block_a_smaller_one(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        memories.create_memory(
            content="python backend " + "and more python backend detail " * 40,
            memory_type=MemoryType.FACT,
            importance=1.0,
            confidence=1.0,
        )
        small = memories.create_memory(
            content="python",
            memory_type=MemoryType.FACT,
            importance=0.1,
        )
        result = compile_for(compiler, budget=12)

        assert [block.memory_id for block in result.blocks] == [small.id]

    def test_oversized_budget_is_rejected(self, retriever: MemoryRetrievalService) -> None:
        bounded = ContextCompilerService(retriever, max_token_budget=100)
        with pytest.raises(ContextValidationError, match="configured maximum"):
            bounded.compile(ContextRequest(query="python", token_budget=101))

    def test_budget_at_the_ceiling_is_accepted(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        bounded = ContextCompilerService(retriever, max_token_budget=100)
        result = bounded.compile(ContextRequest(query="python", token_budget=100))
        assert result.token_budget == 100


class TestDuplicateHandling:
    def test_a_near_duplicate_is_suppressed_not_deleted(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = populate(memories)
        result = compile_for(compiler, budget=2000)

        suppressed = {
            entry.memory_id: entry
            for entry in result.suppressed_candidates
            if entry.reason is SelectionReason.DUPLICATE_OF
        }
        assert created["duplicate"].id in suppressed
        assert suppressed[created["duplicate"].id].related_memory_id == created["profile"].id
        # The representative is in the context, the duplicate is still stored.
        assert any(block.memory_id == created["profile"].id for block in result.blocks)
        assert memories.get_memory(created["duplicate"].id) is not None

    def test_only_one_copy_reaches_the_context(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = populate(memories)
        result = compile_for(compiler, budget=2000)
        identifiers = {block.memory_id for block in result.blocks}

        assert created["profile"].id in identifiers
        assert created["duplicate"].id not in identifiers

    def test_suppression_reports_the_representative_and_the_numbers(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        entry = next(
            item
            for item in compile_for(compiler, budget=2000).suppressed_candidates
            if item.reason is SelectionReason.DUPLICATE_OF
        )
        assert "near-duplicate of" in entry.detail
        assert entry.utility_score > 0.0
        assert entry.estimated_tokens > 0
        assert entry.remaining_tokens == 2000


class TestConflictHandling:
    def _conflicting(self, memories: MemoryService) -> dict[str, Memory]:
        old = memories.create_memory(
            content=CONFLICT_OLD,
            memory_type=MemoryType.FACT,
            importance=0.3,
            confidence=0.6,
        )
        new = memories.create_memory(
            content=CONFLICT_NEW,
            memory_type=MemoryType.FACT,
            importance=0.9,
            confidence=0.5,
        )
        memories.detect_conflicts(new.id)
        resolution = memories.resolve_conflict(new.id, archive_loser=False)
        return {"loser": old, "winner": resolution.winner}

    def test_the_superseded_memory_never_reaches_the_context(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        pair = self._conflicting(memories)
        result = compile_for(compiler, query="backend language", budget=2000)

        identifiers = {block.memory_id for block in result.blocks}
        assert pair["winner"].id in identifiers
        assert pair["loser"].id not in identifiers

    def test_suppression_names_the_winner_and_the_reason(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        pair = self._conflicting(memories)
        entry = next(
            item
            for item in compile_for(
                compiler, query="backend language", budget=2000
            ).suppressed_candidates
            if item.reason is SelectionReason.CONFLICT_SUPERSEDED
        )
        assert entry.memory_id == pair["loser"].id
        assert entry.related_memory_id == pair["winner"].id
        assert "M2 resolution" in entry.detail

    def test_the_loser_remains_stored_and_readable(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        pair = self._conflicting(memories)
        compile_for(compiler, query="backend language", budget=2000)
        assert memories.get_memory(pair["loser"].id) is not None

    def test_an_unresolved_conflict_is_kept_with_its_metadata(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        first = memories.create_memory(
            content=CONFLICT_OLD, memory_type=MemoryType.FACT, importance=0.9
        )
        memories.create_memory(content=CONFLICT_NEW, memory_type=MemoryType.FACT, importance=0.4)
        memories.detect_conflicts(first.id)
        result = compile_for(compiler, query="backend language", budget=2000)

        assert not any(
            entry.reason is SelectionReason.CONFLICT_SUPERSEDED
            for entry in result.suppressed_candidates
        )
        conflicted = [
            block
            for block in result.blocks
            if any(conflict.resolution == "unresolved" for conflict in block.conflicts)
        ]
        assert conflicted
        assert conflicted[0].conflicts[0].resolution_reason is None

    def test_a_block_reports_the_conflicts_it_participates_in(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        self._conflicting(memories)
        result = compile_for(compiler, query="backend language", budget=2000)
        winner = next(block for block in result.blocks if CONFLICT_NEW in block.rendered_content)
        assert winner.conflicts
        assert winner.conflicts[0].resolution == "supersedes_other"
        assert winner.conflicts[0].resolution_reason


class TestRequestFilters:
    def test_exclude_ids_removes_a_memory(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = populate(memories)
        result = compile_for(compiler, budget=2000, exclude_ids=(created["profile"].id,))

        assert all(block.memory_id != created["profile"].id for block in result.blocks)
        entry = next(
            item for item in result.suppressed_candidates if item.memory_id == created["profile"].id
        )
        assert entry.reason is SelectionReason.EXCLUDED_BY_ID

    def test_include_ids_is_an_allow_list(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = populate(memories)
        result = compile_for(
            compiler,
            query="coffee machine python",
            budget=2000,
            include_ids=(created["observation"].id,),
        )

        assert [block.memory_id for block in result.blocks] == [created["observation"].id]
        assert all(
            entry.reason is SelectionReason.NOT_IN_INCLUDE_IDS
            for entry in result.suppressed_candidates
        )

    def test_include_ids_of_an_unretrieved_memory_yields_nothing(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        # The allow list cannot conjure a memory retrieval never returned, and
        # the candidates that *were* returned are rejected with a reason.
        created = populate(memories)
        result = compile_for(compiler, budget=2000, include_ids=(created["observation"].id,))

        assert result.blocks == ()
        assert result.candidates_considered > 0
        assert all(
            entry.reason is SelectionReason.NOT_IN_INCLUDE_IDS
            for entry in result.suppressed_candidates
        )

    def test_a_single_preferred_type_filters_in_retrieval(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        # One type is pushed down into the M3 SQL filter, so nothing else is
        # ever a candidate and the trace has nothing to explain away.
        populate(memories)
        result = compile_for(compiler, budget=2000, preferred_memory_types=(MemoryType.PROFILE,))

        assert all(block.memory_type is MemoryType.PROFILE for block in result.blocks)
        assert not any(
            entry.reason is SelectionReason.MEMORY_TYPE_FILTERED for entry in result.selection_trace
        )

    def test_several_preferred_types_are_filtered_by_the_compiler(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        memories.create_memory(
            content="The python backend is a FastAPI service.",
            memory_type=MemoryType.FACT,
        )
        result = compile_for(
            compiler,
            budget=2000,
            preferred_memory_types=(MemoryType.PROFILE, MemoryType.OBSERVATION),
        )

        assert all(
            block.memory_type in (MemoryType.PROFILE, MemoryType.OBSERVATION)
            for block in result.blocks
        )
        filtered = [
            entry
            for entry in result.suppressed_candidates
            if entry.reason is SelectionReason.MEMORY_TYPE_FILTERED
        ]
        assert len(filtered) == 1
        assert filtered[0].memory_type is MemoryType.FACT

    def test_min_confidence_filters_candidates(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler, budget=2000, min_confidence=0.9)

        assert all(block.confidence >= 0.9 for block in result.blocks)

    def test_min_candidate_score_rejects_low_utility_candidates(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler, budget=2000, min_candidate_score=0.99)

        assert result.blocks == ()
        assert all(
            entry.reason is SelectionReason.BELOW_MINIMUM_SCORE
            for entry in result.suppressed_candidates
        )

    def test_policy_minimum_applies_without_a_request_override(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        populate(memories)
        compiler = ContextCompilerService(retriever, policy=ContextPolicy(min_candidate_score=0.99))
        result = compiler.compile(ContextRequest(query="python", token_budget=2000))

        assert result.blocks == ()
        assert result.explanation.selection_counts["below_minimum_score"] > 0

    def test_top_k_limits_the_candidate_pool(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        for index in range(5):
            memories.create_memory(
                content=f"python backend fact {index}", memory_type=MemoryType.FACT
            )
        result = compile_for(compiler, budget=2000, top_k=2)

        assert result.candidates_considered <= 2

    def test_archived_memories_are_not_compiled_by_default(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = populate(memories)
        memories.update_memory(created["profile"].id, MemoryUpdate(status=MemoryStatus.ARCHIVED))
        result = compile_for(compiler, budget=2000)

        assert all(block.memory_id != created["profile"].id for block in result.blocks)


class TestCompression:
    def test_whitespace_is_normalised_in_the_rendered_context(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = memories.create_memory(
            content="python   backend\n\n   note", memory_type=MemoryType.FACT
        )
        result = compiler.compile(
            ContextRequest(query="python backend", token_budget=200, include_ids=(created.id,))
        )
        block = result.blocks[0]

        assert block.content == "python   backend\n\n   note"
        assert block.rendered_content == "python backend note"
        assert block.compression.changed is True

    def test_compression_can_be_disabled_per_request(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        created = memories.create_memory(
            content="python   backend   note", memory_type=MemoryType.FACT
        )
        result = compiler.compile(
            ContextRequest(
                query="python backend",
                token_budget=200,
                include_ids=(created.id,),
                compress=False,
            )
        )
        assert result.blocks[0].rendered_content == "python   backend   note"

    def test_long_memories_lose_irrelevant_sentences(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        created = memories.create_memory(
            content=(
                "Kubernetes runs the staging cluster for releases. "
                "User prefers Python for backend work. "
                "The office coffee machine is broken again."
            ),
            memory_type=MemoryType.FACT,
        )
        compiler = ContextCompilerService(retriever, policy=ContextPolicy(max_block_tokens=12))
        result = compiler.compile(
            ContextRequest(query="python backend", token_budget=400, include_ids=(created.id,))
        )
        block = result.blocks[0]

        assert "Python" in block.rendered_content
        assert "coffee machine" not in block.rendered_content
        assert block.compression.truncated is True
        assert block.content.count(".") == 3

    def test_compression_is_reported_in_the_explanation(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        created = memories.create_memory(
            content=("Kubernetes runs staging. User prefers Python. Coffee is broken."),
            memory_type=MemoryType.FACT,
        )
        compiler = ContextCompilerService(retriever, policy=ContextPolicy(max_block_tokens=12))
        result = compiler.compile(
            ContextRequest(query="python backend", token_budget=400, include_ids=(created.id,))
        )
        assert result.explanation.compressed_blocks == 1
        assert result.explanation.tokens_saved > 0


class TestDeterminism:
    def test_repeated_compilation_is_identical(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        first = compile_for(compiler, budget=60)
        second = compile_for(compiler, budget=60)

        assert first.context_text == second.context_text
        assert first.blocks == second.blocks
        assert first.selection_trace == second.selection_trace
        assert first.explanation == second.explanation

    def test_determinism_survives_a_rebuilt_retriever(
        self, repository: SqliteMemoryRepository, clock: ManualClock, memories: MemoryService
    ) -> None:
        populate(memories)
        first = ContextCompilerService(MemoryRetrievalService(repository, clock=clock)).compile(
            ContextRequest(query="python backend", token_budget=2000)
        )
        second = ContextCompilerService(MemoryRetrievalService(repository, clock=clock)).compile(
            ContextRequest(query="python backend", token_budget=2000)
        )

        assert first.context_text == second.context_text

    def test_ordering_is_stable_regardless_of_insertion_order(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        for index in range(4):
            memories.create_memory(
                content=f"python backend fact {index}",
                memory_type=MemoryType.FACT,
                importance=0.5,
            )
        result = compile_for(compiler, budget=2000)
        utilities = [block.utility_score for block in result.blocks]

        assert utilities == sorted(utilities, reverse=True)


class TestSemanticModes:
    def test_hybrid_mode_is_reported_honestly(
        self,
        repository: SqliteMemoryRepository,
        clock: ManualClock,
        memories: MemoryService,
    ) -> None:
        populate(memories)
        retriever = MemoryRetrievalService(
            repository,
            clock=clock,
            embedding_provider=HashedLexicalEmbeddingProvider(dimensions=64),
        )
        compiler = ContextCompilerService(retriever)
        result = compiler.compile(
            ContextRequest(
                query="python backend",
                token_budget=400,
                retrieval_mode=RetrievalMode.HYBRID,
                semantic_weight=0.5,
            )
        )

        assert result.retrieval_mode is RetrievalMode.HYBRID
        assert result.requested_retrieval_mode is RetrievalMode.HYBRID
        assert result.semantic_available is True
        assert result.degraded_reason is None
        assert result.within_budget is True

    def test_degraded_mode_is_reported_not_hidden(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compiler.compile(
            ContextRequest(
                query="python backend", token_budget=400, retrieval_mode=RetrievalMode.SEMANTIC
            )
        )

        assert result.requested_retrieval_mode is RetrievalMode.SEMANTIC
        assert result.retrieval_mode is RetrievalMode.LEXICAL
        assert result.semantic_available is False
        assert result.degraded_reason is not None


class TestServiceConfiguration:
    def test_default_policy_and_estimator(self, retriever: MemoryRetrievalService) -> None:
        compiler = ContextCompilerService(retriever)
        assert compiler.policy == ContextPolicy()
        assert compiler.estimator_config == TokenEstimatorConfig()
        assert compiler.retrieval_service is retriever

    def test_describe_policy_is_informative(self, retriever: MemoryRetrievalService) -> None:
        description = ContextCompilerService(retriever).describe_policy()
        assert "memforge-char-word/v1" in description
        assert "ordering=section-then-utility/v1" in description
        assert "PROFILE > FACTS > RECENT_EVENTS > TASKS > OBSERVATIONS" in description

    def test_estimator_configuration_is_honoured(
        self, retriever: MemoryRetrievalService, memories: MemoryService
    ) -> None:
        populate(memories)
        conservative = ContextCompilerService(
            retriever, estimator=TokenEstimatorConfig(chars_per_token=1.0)
        )
        result = conservative.compile(ContextRequest(query="python backend", token_budget=2000))
        generous = ContextCompilerService(retriever).compile(
            ContextRequest(query="python backend", token_budget=2000)
        )

        assert result.estimated_tokens > generous.estimated_tokens
        assert result.explanation.estimator_formula.startswith("memforge-char-word/v1")

    def test_missing_memory_during_re_scoring_is_skipped(
        self,
        compiler: ContextCompilerService,
        memories: MemoryService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        created = populate(memories)

        def explode(memory_id: str, query: object) -> object:
            # A memory that vanished between search and re-scoring must not
            # fail the whole compilation.
            raise MemoryNotFoundError(memory_id)

        monkeypatch.setattr(compiler.retrieval_service, "explain", explode)
        result = compile_for(compiler, budget=2000)

        assert result.within_budget is True
        assert created["profile"].id in {block.memory_id for block in result.blocks}


class TestOrderingIntegration:
    def test_sections_are_rendered_in_policy_order(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        memories.create_memory(
            content="task: ship the context compiler", memory_type=MemoryType.TASK
        )
        memories.create_memory(
            content="event: started the context compiler today", memory_type=MemoryType.EVENT
        )
        memories.create_memory(
            content="profile: builds context compiler memory systems",
            memory_type=MemoryType.PROFILE,
        )
        memories.create_memory(
            content="fact: the context compiler is deterministic", memory_type=MemoryType.FACT
        )
        result = compiler.compile(ContextRequest(query="context compiler", token_budget=2000))

        sections = [block.section for block in result.blocks]
        assert set(sections) == {
            ContextSection.PROFILE,
            ContextSection.FACTS,
            ContextSection.RECENT_EVENTS,
            ContextSection.TASKS,
        }
        ranks = [block.section_rank for block in result.blocks]
        assert ranks == sorted(ranks)
        assert result.context_text.startswith("PROFILE")
        assert result.context_text.index("FACTS") < result.context_text.index("TASKS")

    def test_block_order_field_is_dense_and_ordered(
        self, compiler: ContextCompilerService, memories: MemoryService
    ) -> None:
        populate(memories)
        result = compile_for(compiler, budget=2000)
        assert [block.order for block in result.blocks] == list(range(len(result.blocks)))


class TestAgeing:
    def test_old_memories_lose_utility_through_freshness(
        self, compiler: ContextCompilerService, memories: MemoryService, clock: ManualClock
    ) -> None:
        created = memories.create_memory(
            content="python backend stack note", memory_type=MemoryType.FACT
        )
        fresh = compile_for(compiler, budget=2000)
        clock.advance(days=120)
        aged = compile_for(compiler, budget=2000)

        fresh_block = next(b for b in fresh.blocks if b.memory_id == created.id)
        aged_block = next(b for b in aged.blocks if b.memory_id == created.id)
        assert aged_block.utility_score < fresh_block.utility_score
        assert aged_block.retrieval_score <= fresh_block.retrieval_score

    def test_reference_time_is_taken_from_the_injected_clock(
        self, repository: SqliteMemoryRepository, memories: MemoryService
    ) -> None:
        created = memories.create_memory(content="python note", memory_type=MemoryType.FACT)
        early = ContextCompilerService(
            MemoryRetrievalService(repository, clock=ManualClock(FIXED_START))
        ).compile(ContextRequest(query="python", token_budget=2000))
        late = ContextCompilerService(
            MemoryRetrievalService(repository, clock=ManualClock(FIXED_START + timedelta(days=90)))
        ).compile(ContextRequest(query="python", token_budget=2000))

        assert (
            next(block for block in late.blocks if block.memory_id == created.id).utility_score
            < next(block for block in early.blocks if block.memory_id == created.id).utility_score
        )
