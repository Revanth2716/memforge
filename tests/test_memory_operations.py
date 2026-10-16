"""M5 memory-operations tests: the agent-facing facade over M1-M4.

The point of these tests is that the facade *delegates*: every rule it appears
to implement is actually M1-M4's rule, and no behaviour is re-implemented here.
"""

from __future__ import annotations

import pytest

from app.domain.errors import (
    ContextValidationError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from app.domain.ids import SequentialIdGenerator
from app.domain.models import Memory, MemoryStatus, MemoryType
from app.domain.retrieval import RetrievalMode
from app.domain.time import ManualClock
from app.domain.updates import UNSET, Unset
from app.services.context_service import ContextCompilerService
from app.services.memory_operations import MemoryOperationsService, RecallPolicy
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository

PYTHON = "User prefers Python for backend work."
TYPESCRIPT = "User is learning TypeScript for frontend work."
NEAR_DUPLICATE = "The user prefers python for backend development work"


@pytest.fixture
def memories(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryService:
    """A deterministic memory service."""
    return MemoryService(repository, clock=clock, id_generator=SequentialIdGenerator(prefix="ops"))


@pytest.fixture
def operations(
    repository: SqliteMemoryRepository, memories: MemoryService, clock: ManualClock
) -> MemoryOperationsService:
    """The facade over M1, M3 and M4."""
    retrieval = MemoryRetrievalService(repository, clock=clock)
    return MemoryOperationsService(memories, retrieval, ContextCompilerService(retrieval))


class TestStore:
    def test_store_creates_a_memory(self, operations: MemoryOperationsService) -> None:
        memory = operations.store(PYTHON, MemoryType.PROFILE, confidence=0.9, importance=0.8)
        assert memory.content == PYTHON
        assert memory.memory_type is MemoryType.PROFILE
        assert memory.confidence == 0.9
        assert memory.version == 1

    def test_store_applies_domain_validation(self, operations: MemoryOperationsService) -> None:
        with pytest.raises(MemoryValidationError):
            operations.store("   ", MemoryType.FACT)
        with pytest.raises(MemoryValidationError):
            operations.store("content", MemoryType.FACT, confidence=2.0)

    def test_store_keeps_provenance(
        self, operations: MemoryOperationsService, memories: MemoryService
    ) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE, source="user:test")
        assert memories.get_memory(stored.id).source == "user:test"


class TestSearchAndRecall:
    def _corpus(self, operations: MemoryOperationsService) -> None:
        operations.store(PYTHON, MemoryType.PREFERENCE, confidence=0.9, importance=0.8)
        operations.store(TYPESCRIPT, MemoryType.OBSERVATION)

    def test_search_returns_the_m3_response(self, operations: MemoryOperationsService) -> None:
        self._corpus(operations)
        response = operations.search("python backend", top_k=5)
        assert response.candidates_considered == 1
        assert response.results[0].memory.content == PYTHON

    def test_search_supports_every_mode(self, operations: MemoryOperationsService) -> None:
        self._corpus(operations)
        response = operations.search("python", mode=RetrievalMode.HYBRID)
        assert response.requested_mode is RetrievalMode.HYBRID
        assert response.retrieval_mode is RetrievalMode.LEXICAL
        assert response.degraded_reason is not None

    def test_recall_drops_zero_relevance_memories(
        self, operations: MemoryOperationsService
    ) -> None:
        self._corpus(operations)
        recalled = operations.recall("python backend", top_k=5)
        assert len(recalled) == 1
        assert recalled[0].memory.content == PYTHON

    def test_recall_without_matches_is_empty(self, operations: MemoryOperationsService) -> None:
        self._corpus(operations)
        assert operations.recall("kubernetes ingress") == ()

    def test_recall_respects_the_score_floor(self, operations: MemoryOperationsService) -> None:
        self._corpus(operations)
        assert operations.recall("python", policy=RecallPolicy(min_score=0.99)) == ()

    def test_recall_respects_its_result_cap(self, operations: MemoryOperationsService) -> None:
        for index in range(4):
            operations.store(f"python backend note {index}", MemoryType.FACT)
        recalled = operations.recall("python backend", policy=RecallPolicy(max_results=2))
        assert len(recalled) == 2

    def test_recall_keeps_ordering(self, operations: MemoryOperationsService) -> None:
        for index in range(3):
            operations.store(f"python backend note {index}", MemoryType.FACT)
        recalled = operations.recall("python backend", top_k=5)
        scores = [result.final_score for result in recalled]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("kwargs", [{"min_score": 1.5}, {"max_results": 0}])
    def test_recall_policy_is_validated(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            RecallPolicy(**kwargs)  # type: ignore[arg-type]


class TestUpdate:
    def test_update_applies_only_the_given_field(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE, importance=0.5)
        updated = operations.update(stored.id, importance=0.9)
        assert updated.importance == 0.9
        assert updated.content == PYTHON
        assert updated.version == 2

    def test_omitted_fields_mean_unset_not_null(self, operations: MemoryOperationsService) -> None:
        # Passing None must translate to the UNSET sentinel, otherwise M1 would
        # reject it as an attempt to null a non-nullable field.
        stored = operations.store(PYTHON, MemoryType.PROFILE, source="user:test")
        updated = operations.update(stored.id, importance=0.9)
        assert updated.content == PYTHON
        assert updated.source == "user:test"

    def test_update_can_clear_the_source(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE, source="user:test")
        assert operations.update(stored.id, clear_source=True).source is None

    def test_update_can_change_the_status(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE)
        assert operations.update(stored.id, status=MemoryStatus.ARCHIVED).status is (
            MemoryStatus.ARCHIVED
        )

    def test_update_of_an_unknown_memory_raises(self, operations: MemoryOperationsService) -> None:
        with pytest.raises(MemoryNotFoundError):
            operations.update("11111111-1111-5111-8111-111111111111", importance=0.5)

    def test_unset_sentinel_is_the_one_m1_uses(self) -> None:
        # Guards the translation the facade depends on: the facade must send
        # M1's own sentinel, not a substitute.
        assert isinstance(UNSET, Unset)
        assert UNSET is not None


class TestForget:
    def test_forget_is_a_soft_delete(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE)
        forgotten = operations.forget(stored.id)
        assert forgotten.status is MemoryStatus.DELETED
        assert forgotten.version == 2

    def test_forget_keeps_the_history(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE)
        operations.forget(stored.id)
        assert len(operations.memory_service.get_history(stored.id)) == 2

    def test_forget_is_idempotent(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PROFILE)
        operations.forget(stored.id)
        assert operations.forget(stored.id).status is MemoryStatus.DELETED

    def test_forget_of_an_unknown_memory_raises(self, operations: MemoryOperationsService) -> None:
        with pytest.raises(MemoryNotFoundError):
            operations.forget("11111111-1111-5111-8111-111111111111")


class TestCompileContext:
    def test_compile_returns_a_context_result(self, operations: MemoryOperationsService) -> None:
        operations.store(PYTHON, MemoryType.PROFILE)
        result = operations.compile_context("python backend", token_budget=200)
        assert result.selected_count >= 1
        assert result.within_budget is True
        assert result.blocks[0].rendered_content == PYTHON

    def test_compile_honours_the_budget(self, operations: MemoryOperationsService) -> None:
        for index in range(5):
            operations.store(f"python backend note {index}", MemoryType.FACT)
        for budget in (1, 20, 60, 500):
            result = operations.compile_context("python backend", token_budget=budget)
            assert result.estimated_tokens <= budget

    def test_compile_forwards_filters(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.FACT)
        operations.store(TYPESCRIPT, MemoryType.EVENT)
        result = operations.compile_context(
            "python typescript",
            token_budget=500,
            preferred_memory_types=(MemoryType.FACT,),
            exclude_ids=(stored.id,),
        )
        assert result.blocks == ()

    def test_compile_validates_the_budget(self, operations: MemoryOperationsService) -> None:
        with pytest.raises(ContextValidationError):
            operations.compile_context("python", token_budget=0)

    def test_compile_rejects_an_empty_query(self, operations: MemoryOperationsService) -> None:
        with pytest.raises(ContextValidationError):
            operations.compile_context("  ", token_budget=100)


class TestExplain:
    def test_explain_scores_one_memory(self, operations: MemoryOperationsService) -> None:
        stored = operations.store(PYTHON, MemoryType.PREFERENCE)
        result = operations.explain(stored.id, "python backend")
        assert result.memory.id == stored.id
        assert result.final_score > 0.0
        assert result.explanation

    def test_explain_of_an_unknown_memory_raises(self, operations: MemoryOperationsService) -> None:
        with pytest.raises(MemoryNotFoundError):
            operations.explain("11111111-1111-5111-8111-111111111111", "python")


class TestComposition:
    def test_the_facade_exposes_its_services(
        self, operations: MemoryOperationsService, memories: MemoryService
    ) -> None:
        assert operations.memory_service is memories
        assert isinstance(operations.retrieval_service, MemoryRetrievalService)
        assert isinstance(operations.compiler_service, ContextCompilerService)

    def test_recall_uses_the_m3_suppression(self, operations: MemoryOperationsService) -> None:
        profile = operations.store(PYTHON, MemoryType.PROFILE, importance=0.9)
        operations.store(NEAR_DUPLICATE, MemoryType.OBSERVATION, importance=0.2)
        operations.memory_service.find_duplicates(profile.id)

        recalled = operations.recall("python backend", top_k=5)
        assert len(recalled) == 1
        assert recalled[0].memory.id == profile.id

    def test_stored_memory_is_visible_to_compile(self, operations: MemoryOperationsService) -> None:
        memory: Memory = operations.store(PYTHON, MemoryType.PROFILE)
        result = operations.compile_context("python backend", token_budget=200)
        assert [block.memory_id for block in result.blocks] == [memory.id]
