"""Service-level tests for the M2 intelligence operations."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.errors import MemoryNotFoundError, MemoryValidationError, NoConflictError
from app.domain.intelligence import IntelligenceConfig
from app.domain.models import Memory, MemoryStatus, MemoryType, VersionChange
from app.domain.relationships import RelationshipType
from app.domain.updates import MemoryUpdate
from app.services.memory_service import MemoryService
from app.storage.base import MemoryQuery
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.conftest import ClockFixture, IdGeneratorFixture
from tests.support import FIXED_START

UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"

#: "User prefers Python" and "Python is the user's preferred language".
DUPLICATE_A = "User prefers Python"
DUPLICATE_B = "Python is the user's preferred language"

CONFLICT_A = "preferred backend language = Python"
CONFLICT_B = "preferred backend language = TypeScript"


@pytest.fixture
def intelligence_service(
    repository: SqliteMemoryRepository, clock: ClockFixture, id_generator: IdGeneratorFixture
) -> MemoryService:
    return MemoryService(
        repository,
        clock=clock,
        id_generator=id_generator,
        intelligence=IntelligenceConfig(),
    )


class TestImportanceScoring:
    def test_score_is_bounded_and_explained(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers typed Python.", memory_type=MemoryType.PREFERENCE, importance=0.8
        )
        breakdown = intelligence_service.score_memory(memory.id)

        assert 0.0 <= breakdown.score <= 1.0
        assert breakdown.memory_id == memory.id
        assert len(breakdown.factors) == 5
        assert breakdown.explanation.startswith("importance ")
        assert breakdown.computed_at == FIXED_START

    def test_factors_are_documented(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(content="anything", memory_type=MemoryType.FACT)
        breakdown = intelligence_service.score_memory(memory.id)
        assert [factor.name for factor in breakdown.factors] == [
            "memory_type",
            "explicit_importance",
            "recurrence",
            "recency",
            "access_frequency",
        ]
        weights = {factor.name: factor.weight for factor in breakdown.factors}
        assert weights == {
            "memory_type": 0.20,
            "explicit_importance": 0.40,
            "recurrence": 0.15,
            "recency": 0.15,
            "access_frequency": 0.10,
        }

    def test_score_is_not_stored(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="anything", memory_type=MemoryType.FACT, importance=0.5
        )
        _ = intelligence_service.score_memory(memory.id)
        stored = intelligence_service.get_memory(memory.id)

        assert stored.importance == 0.5  # the caller's value is untouched
        assert stored.version == 1

    def test_confidence_and_importance_stay_separate(
        self, intelligence_service: MemoryService
    ) -> None:
        confident = intelligence_service.create_memory(
            content="x", memory_type=MemoryType.FACT, confidence=1.0, importance=0.0
        )
        important = intelligence_service.create_memory(
            content="x", memory_type=MemoryType.FACT, confidence=0.0, importance=1.0
        )
        confident_score = intelligence_service.score_memory(confident.id).score
        important_score = intelligence_service.score_memory(important.id).score

        assert important_score > confident_score
        # Effective priority merges both, so this memory ranks between them.
        assert intelligence_service.get_effective_priority(important.id).priority > 0.0

    def test_score_for_unknown_memory(self, intelligence_service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.score_memory(UNKNOWN_ID)


class TestAccessTelemetry:
    def test_access_raises_the_access_factor(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.PREFERENCE
        )
        before = intelligence_service.score_memory(memory.id).score

        for _ in range(5):
            intelligence_service.record_access(memory.id)

        after = intelligence_service.score_memory(memory.id).score
        assert after > before
        access_factor = next(
            factor
            for factor in intelligence_service.score_memory(memory.id).factors
            if factor.name == "access_frequency"
        )
        assert access_factor.raw_value == 1.0  # saturated at 5 accesses
        assert intelligence_service.get_memory(memory.id).access_count == 5

    def test_access_does_not_touch_history_or_version(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.FACT
        )
        intelligence_service.record_access(memory.id)

        stored = intelligence_service.get_memory(memory.id)
        assert stored.version == 1
        assert stored.updated_at == memory.updated_at
        assert len(intelligence_service.get_history(memory.id)) == 1
        assert stored.last_accessed_at == FIXED_START

    def test_access_is_never_created_implicitly(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.FACT
        )
        for _ in range(5):
            intelligence_service.get_memory(memory.id)
            intelligence_service.list_memories()
            intelligence_service.score_memory(memory.id)
        assert intelligence_service.get_memory(memory.id).access_count == 0

    def test_access_for_unknown_memory(self, intelligence_service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.record_access(UNKNOWN_ID)


class TestTemporalDecay:
    def test_priority_halves_after_one_half_life(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.PREFERENCE, importance=1.0
        )
        fresh = intelligence_service.get_effective_priority(memory.id)
        decayed = intelligence_service.get_effective_priority(
            memory.id, at=FIXED_START + timedelta(days=30)
        )

        assert decayed.decay_multiplier == pytest.approx(0.5)
        assert decayed.priority < fresh.priority
        # The importance score decays too (its recency factor uses the same
        # half-life), so the priority falls by more than a bare halving.
        assert decayed.importance_score < fresh.importance_score

    def test_recency_affects_the_importance_score_too(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.PREFERENCE
        )
        fresh = intelligence_service.score_memory(memory.id)
        stale = intelligence_service.score_memory(memory.id, at=FIXED_START + timedelta(days=60))
        assert stale.score < fresh.score
        assert stale.computed_at == FIXED_START + timedelta(days=60)

    def test_decay_is_computed_not_stored(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.PREFERENCE
        )
        first = intelligence_service.get_effective_priority(memory.id)
        stored = intelligence_service.get_memory(memory.id)

        # 60 days later: different priority, identical stored memory.
        later = intelligence_service.get_effective_priority(
            memory.id, at=FIXED_START + timedelta(days=60)
        )
        assert later.priority < first.priority
        assert intelligence_service.get_memory(memory.id) == stored
        assert stored.version == 1

    def test_reference_time_defaults_to_the_clock(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.FACT
        )
        breakdown = intelligence_service.get_effective_priority(memory.id)
        assert breakdown.computed_at == FIXED_START
        assert breakdown.age_days == 0.0


class TestDuplicateDetection:
    def test_documented_example_is_a_near_duplicate(
        self, intelligence_service: MemoryService
    ) -> None:
        first = intelligence_service.create_memory(
            content=DUPLICATE_A, memory_type=MemoryType.PREFERENCE
        )
        second = intelligence_service.create_memory(
            content=DUPLICATE_B, memory_type=MemoryType.PREFERENCE
        )

        matches = intelligence_service.find_duplicates(first.id)
        assert [match.memory_id for match in matches] == [second.id]
        assert matches[0].similarity == pytest.approx(6 / 7)
        assert set(matches[0].shared_tokens) == {"user", "prefer", "python"}
        assert "Dice" in matches[0].explanation

    def test_identical_texts_are_duplicates(self, intelligence_service: MemoryService) -> None:
        intelligence_service.create_memory(content="alpha beta", memory_type=MemoryType.FACT)
        other = intelligence_service.create_memory(
            content="alpha beta", memory_type=MemoryType.FACT
        )
        matches = intelligence_service.find_duplicates(other.id)
        assert len(matches) == 1
        assert matches[0].similarity == 1.0

    def test_unrelated_memories_are_not_duplicates(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.FACT
        )
        intelligence_service.create_memory(
            content="The build server is in Hyderabad", memory_type=MemoryType.OBSERVATION
        )
        assert intelligence_service.find_duplicates(memory.id) == []

    def test_threshold_boundaries(self, intelligence_service: MemoryService) -> None:
        # "alpha beta gamma delta" vs "alpha beta gamma epsilon zeta eta" -> Dice 0.6
        boundary = intelligence_service.create_memory(
            content="alpha beta gamma delta", memory_type=MemoryType.FACT
        )
        intelligence_service.create_memory(
            content="alpha beta gamma epsilon zeta eta", memory_type=MemoryType.FACT
        )

        assert len(intelligence_service.find_duplicates(boundary.id, threshold=0.6)) == 1
        assert intelligence_service.find_duplicates(boundary.id, threshold=0.61) == []
        assert intelligence_service.find_duplicates(
            boundary.id, threshold=0.0
        )  # sanity: everything matches at 0

    def test_explicit_threshold_outranks_the_config(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content="alpha beta", memory_type=MemoryType.FACT
        )
        intelligence_service.create_memory(content="alpha gamma", memory_type=MemoryType.FACT)
        # Dice is exactly 0.5: excluded at the default threshold, included at 0.5.
        assert intelligence_service.find_duplicates(memory.id) == []
        assert len(intelligence_service.find_duplicates(memory.id, threshold=0.5)) == 1

    def test_matches_are_ranked_deterministically(
        self, intelligence_service: MemoryService
    ) -> None:
        anchor = intelligence_service.create_memory(
            content="alpha beta gamma delta", memory_type=MemoryType.FACT
        )
        intelligence_service.create_memory(content="alpha beta gamma", memory_type=MemoryType.FACT)
        intelligence_service.create_memory(
            content="alpha beta gamma delta epsilon", memory_type=MemoryType.FACT
        )

        matches = intelligence_service.find_duplicates(anchor.id)
        similarities = [match.similarity for match in matches]
        assert similarities == sorted(similarities, reverse=True)

    def test_limit_truncates_the_result(self, intelligence_service: MemoryService) -> None:
        anchor = intelligence_service.create_memory(
            content="alpha beta gamma delta", memory_type=MemoryType.FACT
        )
        for index in range(4):
            intelligence_service.create_memory(
                content=f"alpha beta gamma {index}", memory_type=MemoryType.FACT
            )
        matches = intelligence_service.find_duplicates(anchor.id, limit=2)
        assert len(matches) == 2

    def test_invalid_threshold_is_rejected(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(content="x", memory_type=MemoryType.FACT)
        with pytest.raises(MemoryValidationError):
            intelligence_service.find_duplicates(memory.id, threshold=1.5)
        with pytest.raises(MemoryValidationError):
            intelligence_service.find_duplicates(memory.id, threshold=-0.1)
        with pytest.raises(MemoryValidationError):
            intelligence_service.find_duplicates(memory.id, limit=0)

    def test_duplicate_detection_persists_relationships(
        self, intelligence_service: MemoryService
    ) -> None:
        first = intelligence_service.create_memory(content=DUPLICATE_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(
            content=DUPLICATE_B, memory_type=MemoryType.FACT
        )
        intelligence_service.find_duplicates(first.id)

        relationships = intelligence_service.get_relationships(first.id)
        assert len(relationships) == 1
        assert relationships[0].relationship is RelationshipType.DUPLICATE_OF
        assert relationships[0].target_id == second.id
        assert relationships[0].similarity == pytest.approx(6 / 7)

    def test_repeated_detection_is_idempotent(self, intelligence_service: MemoryService) -> None:
        first = intelligence_service.create_memory(content=DUPLICATE_A, memory_type=MemoryType.FACT)
        intelligence_service.create_memory(content=DUPLICATE_B, memory_type=MemoryType.FACT)
        intelligence_service.find_duplicates(first.id)
        intelligence_service.find_duplicates(first.id)
        assert len(intelligence_service.get_relationships(first.id)) == 1

    def test_detection_can_be_side_effect_free(self, intelligence_service: MemoryService) -> None:
        first = intelligence_service.create_memory(content=DUPLICATE_A, memory_type=MemoryType.FACT)
        intelligence_service.create_memory(content=DUPLICATE_B, memory_type=MemoryType.FACT)
        matches = intelligence_service.find_duplicates(first.id, persist=False)
        assert len(matches) == 1
        assert intelligence_service.get_relationships(first.id) == []

    def test_deleted_memories_are_not_duplicates(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content=DUPLICATE_A, memory_type=MemoryType.FACT
        )
        forgotten = intelligence_service.create_memory(
            content=DUPLICATE_B, memory_type=MemoryType.FACT
        )
        intelligence_service.forget_memory(forgotten.id)

        # The forgotten memory leaves the detection corpus...
        assert intelligence_service.find_duplicates(memory.id, persist=False) == []
        # ...and stays invisible itself.
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.find_duplicates(forgotten.id)

    def test_archived_memories_are_not_duplicates(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content=DUPLICATE_A, memory_type=MemoryType.FACT
        )
        archived = intelligence_service.create_memory(
            content=DUPLICATE_B, memory_type=MemoryType.FACT
        )
        intelligence_service.update_memory(archived.id, MemoryUpdate(status=MemoryStatus.ARCHIVED))

        assert intelligence_service.find_duplicates(memory.id, persist=False) == []
        # An archived memory is still readable and can still be compared.
        assert intelligence_service.get_memory(archived.id).status is MemoryStatus.ARCHIVED
        assert len(intelligence_service.find_duplicates(archived.id, persist=False)) == 1

    def test_a_conflicting_pair_is_not_reported_as_a_duplicate(
        self, intelligence_service: MemoryService
    ) -> None:
        left = intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        intelligence_service.create_memory(content=CONFLICT_B, memory_type=MemoryType.FACT)
        assert intelligence_service.find_duplicates(left.id, persist=False) == []
        assert intelligence_service.get_relationships(left.id) == []


class TestConflictDetection:
    def test_documented_conflict_is_detected(self, intelligence_service: MemoryService) -> None:
        first = intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(
            content=CONFLICT_B, memory_type=MemoryType.FACT, importance=0.9
        )

        conflicts = intelligence_service.detect_conflicts(second.id)
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.source_memory_id == second.id
        assert conflict.target_memory_id == first.id
        assert conflict.attribute == "prefer backend language"
        assert conflict.source_value == "typescript"
        assert conflict.target_value == "python"
        assert conflict.value_similarity == 0.0
        assert "both memories state" in conflict.explanation

    def test_agreeing_memories_do_not_conflict(self, intelligence_service: MemoryService) -> None:
        intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(
            content="preferred backend language = python", memory_type=MemoryType.FACT
        )
        assert intelligence_service.detect_conflicts(second.id) == []

    def test_unrelated_memories_do_not_conflict(self, intelligence_service: MemoryService) -> None:
        intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(
            content="User works from Hyderabad", memory_type=MemoryType.PROFILE
        )
        assert intelligence_service.detect_conflicts(second.id) == []

    def test_conflicts_are_persisted_and_idempotent(
        self, intelligence_service: MemoryService
    ) -> None:
        intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(content=CONFLICT_B, memory_type=MemoryType.FACT)

        intelligence_service.detect_conflicts(second.id)
        intelligence_service.detect_conflicts(second.id)

        relationships = intelligence_service.get_relationships(second.id)
        assert [rel.relationship for rel in relationships] == [RelationshipType.CONFLICTS_WITH]

    def test_conflicts_are_visible_from_both_sides(
        self, intelligence_service: MemoryService
    ) -> None:
        first = intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(content=CONFLICT_B, memory_type=MemoryType.FACT)
        intelligence_service.detect_conflicts(second.id)

        forward = intelligence_service.detect_conflicts(first.id)
        backward = intelligence_service.detect_conflicts(second.id)
        assert len(forward) == len(backward) == 1
        assert forward[0].target_memory_id == second.id
        assert backward[0].target_memory_id == first.id

    def test_deleted_and_archived_memories_are_excluded(
        self, intelligence_service: MemoryService
    ) -> None:
        first = intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        second = intelligence_service.create_memory(content=CONFLICT_B, memory_type=MemoryType.FACT)
        intelligence_service.forget_memory(first.id)
        assert intelligence_service.detect_conflicts(second.id) == []
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.detect_conflicts(first.id)

        third = intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        intelligence_service.update_memory(third.id, MemoryUpdate(status=MemoryStatus.ARCHIVED))
        assert intelligence_service.detect_conflicts(second.id) == []

    def test_conflicts_sorted_deterministically(self, intelligence_service: MemoryService) -> None:
        intelligence_service.create_memory(content="attr one = a", memory_type=MemoryType.FACT)
        intelligence_service.create_memory(content="attr two = b", memory_type=MemoryType.FACT)
        probe = intelligence_service.create_memory(
            content="attr two = c\nattr one = d", memory_type=MemoryType.FACT
        )
        conflicts = intelligence_service.detect_conflicts(probe.id)
        attributes = [conflict.attribute for conflict in conflicts]
        assert attributes == sorted(attributes)

    def test_conflict_for_unknown_memory(self, intelligence_service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.detect_conflicts(UNKNOWN_ID)


class TestConflictResolution:
    @pytest.fixture
    def conflicting(self, intelligence_service: MemoryService) -> tuple[Memory, Memory]:
        first = intelligence_service.create_memory(
            content=CONFLICT_A, memory_type=MemoryType.FACT, importance=0.4, confidence=0.6
        )
        second = intelligence_service.create_memory(
            content=CONFLICT_B, memory_type=MemoryType.FACT, importance=0.9, confidence=0.5
        )
        return first, second

    def test_resolution_returns_winner_and_loser(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        first, second = conflicting
        resolution = intelligence_service.resolve_conflict(second.id)

        assert resolution.winner.id == second.id
        assert resolution.loser.id == first.id
        assert resolution.decided_by == "explicit_importance"
        assert resolution.loser_archived is True
        assert resolution.steps[0].factor == "explicit_importance"

    def test_resolution_records_supersedes(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        _, second = conflicting
        resolution = intelligence_service.resolve_conflict(second.id)

        relationships = intelligence_service.get_relationships(second.id)
        supersedes = [
            rel for rel in relationships if rel.relationship is RelationshipType.SUPERSEDES
        ]
        assert len(supersedes) == 1
        assert supersedes[0].source_id == resolution.winner.id
        assert supersedes[0].target_id == resolution.loser.id

    def test_resolved_conflicts_are_not_reported_again(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        first, second = conflicting
        assert len(intelligence_service.detect_conflicts(second.id)) == 1
        intelligence_service.resolve_conflict(second.id)
        assert intelligence_service.detect_conflicts(second.id) == []
        assert intelligence_service.detect_conflicts(first.id) == []

    def test_resolution_archives_the_loser(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        first, second = conflicting
        intelligence_service.resolve_conflict(second.id)

        assert intelligence_service.get_memory(first.id).status is MemoryStatus.ARCHIVED
        assert intelligence_service.get_memory(second.id).status is MemoryStatus.ACTIVE

    def test_resolution_is_auditable_in_history(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        first, second = conflicting
        intelligence_service.resolve_conflict(second.id)

        history = intelligence_service.get_history(first.id)
        assert [version.change for version in history] == [
            VersionChange.CREATED,
            VersionChange.UPDATED,
        ]
        assert history[-1].status is MemoryStatus.ARCHIVED
        # The winner's history is untouched.
        assert len(intelligence_service.get_history(second.id)) == 1

    def test_resolution_can_skip_archiving(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        first, second = conflicting
        resolution = intelligence_service.resolve_conflict(second.id, archive_loser=False)
        assert resolution.loser_archived is False
        assert intelligence_service.get_memory(first.id).status is MemoryStatus.ACTIVE

    def test_target_memory_can_be_chosen_explicitly(
        self, intelligence_service: MemoryService
    ) -> None:
        first = intelligence_service.create_memory(
            content="preferred backend language = Python",
            memory_type=MemoryType.FACT,
            importance=0.9,
        )
        intelligence_service.create_memory(
            content="preferred backend language = Go",
            memory_type=MemoryType.FACT,
            importance=0.1,
        )
        second = intelligence_service.create_memory(
            content="preferred backend language = TypeScript",
            memory_type=MemoryType.FACT,
            importance=0.5,
        )

        resolution = intelligence_service.resolve_conflict(second.id, target_id=first.id)
        assert resolution.winner.id == first.id
        assert resolution.loser.id == second.id

    def test_resolution_without_conflict_raises(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(content=CONFLICT_A, memory_type=MemoryType.FACT)
        intelligence_service.create_memory(
            content="User works from Hyderabad", memory_type=MemoryType.PROFILE
        )
        with pytest.raises(NoConflictError):
            intelligence_service.resolve_conflict(memory.id)

        with pytest.raises(NoConflictError):
            intelligence_service.resolve_conflict(memory.id, target_id=UNKNOWN_ID)

    def test_resolution_twice_raises(
        self, intelligence_service: MemoryService, conflicting: tuple[Memory, Memory]
    ) -> None:
        _, second = conflicting
        intelligence_service.resolve_conflict(second.id)
        with pytest.raises(NoConflictError):
            intelligence_service.resolve_conflict(second.id)


class TestM1Compatibility:
    """M2 must not change any M1 guarantee."""

    def test_lifecycle_unchanged(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.PREFERENCE
        )
        intelligence_service.update_memory(memory.id, MemoryUpdate(importance=0.9))
        intelligence_service.record_access(memory.id)
        intelligence_service.score_memory(memory.id)
        intelligence_service.find_duplicates(memory.id)

        stored = intelligence_service.get_memory(memory.id)
        assert stored.version == 2
        assert stored.importance == 0.9
        history = intelligence_service.get_history(memory.id)
        assert [version.change for version in history] == [
            VersionChange.CREATED,
            VersionChange.UPDATED,
        ]

    def test_forget_still_hides_and_keeps_history(
        self, intelligence_service: MemoryService
    ) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.FACT
        )
        intelligence_service.forget_memory(memory.id)
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.get_memory(memory.id)
        assert len(intelligence_service.get_history(memory.id)) == 2
        assert intelligence_service.list_memories().total == 0
        assert (
            intelligence_service.list_memories(MemoryQuery(status=MemoryStatus.DELETED)).total == 1
        )

    def test_forgotten_memory_cannot_be_updated(self, intelligence_service: MemoryService) -> None:
        memory = intelligence_service.create_memory(
            content="User prefers Python", memory_type=MemoryType.FACT
        )
        intelligence_service.forget_memory(memory.id)
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.update_memory(memory.id, MemoryUpdate(content="x"))
        with pytest.raises(MemoryNotFoundError):
            intelligence_service.update_memory(memory.id, MemoryUpdate(status=MemoryStatus.ACTIVE))

    def test_pagination_and_filters_unchanged(self, intelligence_service: MemoryService) -> None:
        for index in range(3):
            intelligence_service.create_memory(
                content=f"memory {index}", memory_type=MemoryType.TASK
            )
        page = intelligence_service.list_memories(MemoryQuery(limit=2, offset=0))
        assert page.total == 3
        assert len(page.items) == 2

    def test_config_defaults_are_exposed(self, intelligence_service: MemoryService) -> None:
        config = intelligence_service.intelligence_config
        assert config.similarity_threshold == 0.6
        assert config.decay_half_life_days == 30.0
