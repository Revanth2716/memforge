"""Memory service layer.

This is where the memory lifecycle rules live. The service is transport
agnostic (no FastAPI imports) and depends only on the repository interface, so
it can be exercised directly in tests or reused later by an MCP server.

M1 adds the lifecycle (create/read/update/forget/history); M2 adds the
deterministic *intelligence* operations (importance scoring, temporal decay,
duplicate detection, conflict detection and conflict resolution). Both live
here because both are business rules, not storage or transport concerns.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime

from app.domain.attributes import AttributeValue, extract_attribute_values, find_conflicts
from app.domain.errors import (
    InvalidMemoryStateError,
    MemoryNotFoundError,
    MemoryValidationError,
    NoConflictError,
    RelationshipExistsError,
)
from app.domain.ids import IdGenerator, UuidIdGenerator
from app.domain.intelligence import (
    ImportanceBreakdown,
    IntelligenceConfig,
    PriorityBreakdown,
    ResolutionStep,
    compute_effective_priority,
    compute_importance,
    decide_conflict,
)
from app.domain.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    VersionChange,
)
from app.domain.relationships import MemoryRelationship, RelationshipType
from app.domain.similarity import SimilarityDetail, similarity_details
from app.domain.time import Clock, SystemClock
from app.domain.updates import MemoryUpdate
from app.storage.base import MemoryQuery, MemoryRepository

#: Neutral defaults: callers state confidence/importance explicitly.
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IMPORTANCE = 0.5

#: Maximum number of duplicate matches returned by ``find_duplicates``.
DEFAULT_DUPLICATE_LIMIT = 20

#: One duplicate/conflict candidate: the other memory, its score and the
#: token-level evidence behind that score.
_Candidate = tuple[Memory, float, SimilarityDetail]


@dataclass(frozen=True, slots=True)
class MemoryPage:
    """A page of memories plus the total number of matches."""

    items: list[Memory]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """A near-duplicate of a memory, with the evidence behind the score."""

    memory_id: str
    similarity: float
    content: str
    memory_type: MemoryType
    shared_tokens: tuple[str, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class ConflictCandidate:
    """Two memories that disagree about the same attribute."""

    source_memory_id: str
    target_memory_id: str
    attribute: str
    source_value: str
    target_value: str
    value_similarity: float
    source_content: str
    target_content: str
    explanation: str


@dataclass(frozen=True, slots=True)
class ConflictResolution:
    """The outcome of resolving one conflict, including the explanation."""

    conflict: ConflictCandidate
    winner: Memory
    loser: Memory
    decided_by: str
    explanation: str
    steps: tuple[ResolutionStep, ...]
    loser_archived: bool


class MemoryService:
    """Implements the M1 lifecycle plus the M2 intelligence operations."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        intelligence: IntelligenceConfig | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or SystemClock()
        self._ids = id_generator or UuidIdGenerator()
        self._intelligence = intelligence or IntelligenceConfig()

    @property
    def repository(self) -> MemoryRepository:
        return self._repository

    @property
    def intelligence_config(self) -> IntelligenceConfig:
        """The deterministic M2 configuration in use."""
        return self._intelligence

    # -- lifecycle --------------------------------------------------------

    def initialize(self) -> None:
        """Ensure the storage schema exists."""
        self._repository.initialize()

    def health_check(self) -> bool:
        """True when the storage backend answers a trivial query."""
        return self._repository.ping()

    # -- commands ---------------------------------------------------------

    def create_memory(
        self,
        *,
        content: str,
        memory_type: MemoryType,
        source: str | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> Memory:
        """Store a new ACTIVE memory and write its first history version."""
        now = self._clock.now()
        memory = Memory(
            id=self._ids.next_id(),
            content=content,
            memory_type=memory_type,
            source=source,
            confidence=confidence,
            importance=importance,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            version=1,
        )
        self._repository.insert(memory, change=VersionChange.CREATED)
        return memory

    def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory:
        """Apply a partial update and append a new history version."""
        if update.is_empty():
            raise MemoryValidationError("At least one field must be provided to update a memory")
        current = self._require_memory(memory_id, include_deleted=True)
        if current.status is MemoryStatus.DELETED:
            # Forgotten memories stay invisible: nothing can update them.
            raise MemoryNotFoundError(memory_id)
        candidate = update.apply_to(current)
        if candidate.status is MemoryStatus.DELETED:
            raise InvalidMemoryStateError(
                "A memory cannot be forgotten through an update; use delete/forget instead",
                memory_id=memory_id,
            )
        updated = replace(
            candidate,
            updated_at=self._clock.now(),
            version=current.version + 1,
        )
        self._repository.save(updated, change=VersionChange.UPDATED)
        return updated

    def forget_memory(self, memory_id: str) -> Memory:
        """Soft-delete a memory, keeping its full history available.

        Forgetting an already forgotten memory is a successful no-op.
        """
        current = self._require_memory(memory_id, include_deleted=True)
        if current.status is MemoryStatus.DELETED:
            return current
        forgotten = replace(
            current,
            status=MemoryStatus.DELETED,
            updated_at=self._clock.now(),
            version=current.version + 1,
        )
        self._repository.save(forgotten, change=VersionChange.FORGOTTEN)
        return forgotten

    # -- queries ----------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory:
        """Return a memory, or raise :class:`MemoryNotFoundError`."""
        return self._require_memory(memory_id)

    def list_memories(self, query: MemoryQuery | None = None) -> MemoryPage:
        """Return a page of memories, newest first, with the total match count.

        Without an explicit query only ``ACTIVE`` memories are listed. Pass
        :class:`MemoryQuery` with another ``status`` (or ``status=None`` to
        include every status) to inspect archived or forgotten memories.
        """
        effective_query = query or MemoryQuery(status=MemoryStatus.ACTIVE)
        items, total = self._repository.find(effective_query)
        return MemoryPage(
            items=items,
            total=total,
            limit=effective_query.limit,
            offset=effective_query.offset,
        )

    def get_history(self, memory_id: str) -> list[MemoryVersion]:
        """Return every version of a memory, oldest first.

        History is still available for forgotten memories -- that is the point
        of keeping append-only versions instead of destroying rows.
        """
        self._require_memory(memory_id, include_deleted=True)
        return self._repository.versions(memory_id)

    # -- M2: intelligence -------------------------------------------------

    def score_memory(self, memory_id: str, *, at: datetime | None = None) -> ImportanceBreakdown:
        """Compute the deterministic importance score of a memory.

        The score is never stored: it is recomputed from the memory, the
        configured weights and the reference time, so it stays explainable and
        changes automatically as the memory ages or is retrieved. Pass ``at`` to
        evaluate the score at another instant (useful for "what if" checks).
        """
        memory = self._require_memory(memory_id)
        return self._score_at(memory, at or self._clock.now())

    def record_access(self, memory_id: str) -> Memory:
        """Record a retrieval event (telemetry for the access factor).

        This is the only write that does *not* create a history version and does
        not touch ``updated_at``: reading a memory must never look like
        editing it.
        """
        self._require_memory(memory_id)
        return self._repository.record_access(memory_id, at=self._clock.now())

    def get_effective_priority(
        self, memory_id: str, *, at: datetime | None = None
    ) -> PriorityBreakdown:
        """Compute the decayed priority of a memory at ``at`` (default: now).

        Nothing is mutated: the priority is derived from the stored memory and
        the reference time, which is what keeps temporal decay auditable.
        """
        memory = self._require_memory(memory_id)
        reference = at or self._clock.now()
        breakdown = self._score_at(memory, reference)
        return compute_effective_priority(
            memory,
            config=self._intelligence,
            now=reference,
            importance_score=breakdown.score,
        )

    def find_duplicates(
        self,
        memory_id: str,
        *,
        threshold: float | None = None,
        limit: int = DEFAULT_DUPLICATE_LIMIT,
        persist: bool = True,
    ) -> list[DuplicateMatch]:
        """Return active memories that are near-duplicates of ``memory_id``.

        Similarity is the Dice coefficient over normalised token sets (see
        ``app.domain.similarity``). Matches are ranked by similarity descending
        then by id, and a pair that *conflicts* on an attribute is never
        reported as a duplicate (a disagreement is not a duplicate).

        With ``persist=True`` (default) each match is recorded as a
        ``DUPLICATE_OF`` relationship. Recording is idempotent, and no memory is
        ever modified or removed.
        """
        memory = self._require_memory(memory_id)
        effective_threshold = self._resolve_threshold(threshold)
        if limit < 1:
            raise MemoryValidationError("limit must be >= 1")

        candidates = self._duplicate_candidates(memory, threshold=effective_threshold)[:limit]
        matches = [
            DuplicateMatch(
                memory_id=other.id,
                similarity=round(score, 6),
                content=other.content,
                memory_type=other.memory_type,
                shared_tokens=detail.shared_tokens,
                explanation=detail.explanation,
            )
            for other, score, detail in candidates
        ]
        if persist:
            for match in matches:
                self._link(
                    memory.id, match.memory_id, RelationshipType.DUPLICATE_OF, match.similarity
                )
        return matches

    def detect_conflicts(self, memory_id: str, *, persist: bool = True) -> list[ConflictCandidate]:
        """Return unresolved conflicts between ``memory_id`` and active memories.

        A conflict means "same normalised attribute, different normalised
        value" -- for example ``preferred backend language = Python`` versus
        ``preferred backend language = TypeScript``. Pairs already resolved by
        a ``SUPERSEDES`` link are not reported again.
        """
        memory = self._require_memory(memory_id)
        resolved = self._repository.superseded_pairs(memory_id)
        candidates: list[ConflictCandidate] = []

        for other in self._repository.active_memories():
            if other.id == memory.id:
                continue
            for comparison in find_conflicts(
                memory.content,
                other.content,
                value_similarity_threshold=self._intelligence.value_similarity_threshold,
            ):
                if frozenset({memory.id, other.id}) in resolved:
                    continue
                candidates.append(
                    ConflictCandidate(
                        source_memory_id=memory.id,
                        target_memory_id=other.id,
                        attribute=comparison.attribute,
                        source_value=comparison.left_value,
                        target_value=comparison.right_value,
                        value_similarity=round(comparison.similarity, 6),
                        source_content=memory.content,
                        target_content=other.content,
                        explanation=(
                            f"both memories state '{comparison.attribute}' but with "
                            f"different values ('{comparison.left_value}' vs "
                            f"'{comparison.right_value}'); value similarity "
                            f"{comparison.similarity:.2f} < "
                            f"{self._intelligence.value_similarity_threshold:.2f}"
                        ),
                    )
                )

        candidates.sort(key=lambda candidate: (candidate.attribute, candidate.target_memory_id))
        if persist:
            for candidate in candidates:
                self._link(
                    candidate.source_memory_id,
                    candidate.target_memory_id,
                    RelationshipType.CONFLICTS_WITH,
                )
        return candidates

    def resolve_conflict(
        self,
        memory_id: str,
        *,
        target_id: str | None = None,
        archive_loser: bool = True,
    ) -> ConflictResolution:
        """Resolve one conflict with the deterministic policy.

        The policy compares, in order: explicit importance, confidence,
        effective priority, recency, then identifier (a total order, so the
        outcome never depends on iteration order). A ``SUPERSEDES`` relationship
        is always recorded and, by default, the losing memory is archived --
        which appends a history version, so nothing is silently overwritten.
        """
        memory = self._require_memory(memory_id)
        conflicts = self.detect_conflicts(memory_id)
        if target_id is not None:
            conflicts = [
                candidate for candidate in conflicts if candidate.target_memory_id == target_id
            ]
        if not conflicts:
            raise NoConflictError(memory_id, target_id)

        conflict = conflicts[0]
        other = self._require_memory(conflict.target_memory_id)
        now = self._clock.now()
        decision = decide_conflict(
            memory,
            other,
            config=self._intelligence,
            now=now,
            left_importance=self._score_at(memory, now).score,
            right_importance=self._score_at(other, now).score,
        )

        self._link(decision.winner.id, decision.loser.id, RelationshipType.SUPERSEDES)

        loser_archived = False
        if archive_loser:
            # Archiving appends a history version: resolution is auditable.
            self.update_memory(decision.loser.id, MemoryUpdate(status=MemoryStatus.ARCHIVED))
            loser_archived = True

        return ConflictResolution(
            conflict=conflict,
            winner=decision.winner,
            loser=decision.loser,
            decided_by=decision.decided_by,
            explanation=decision.explanation,
            steps=decision.steps,
            loser_archived=loser_archived,
        )

    def get_relationships(self, memory_id: str) -> list[MemoryRelationship]:
        """Return every relationship recorded for a memory (both directions)."""
        self._require_memory(memory_id, include_deleted=True)
        return self._repository.relationships(memory_id)

    # -- internals --------------------------------------------------------

    def _require_memory(self, memory_id: str, *, include_deleted: bool = False) -> Memory:
        memory = self._repository.get(memory_id, include_deleted=include_deleted)
        if memory is None:
            raise MemoryNotFoundError(memory_id)
        return memory

    def _score_at(self, memory: Memory, at: datetime) -> ImportanceBreakdown:
        return compute_importance(
            memory,
            config=self._intelligence,
            now=at,
            duplicate_count=len(self._duplicate_candidates(memory, now=at)),
            access_count=memory.access_count,
        )

    def _resolve_threshold(self, threshold: float | None) -> float:
        if threshold is None:
            return self._intelligence.similarity_threshold
        if not 0.0 <= threshold <= 1.0:
            raise MemoryValidationError("threshold must be between 0.0 and 1.0")
        return threshold

    def _duplicate_candidates(
        self, memory: Memory, *, threshold: float | None = None, now: datetime | None = None
    ) -> list[_Candidate]:
        """Active memories similar to ``memory``, best match first.

        The scan is O(n) over the active corpus and completely deterministic:
        the corpus comes from the repository, the scores from pure functions,
        and ties are broken by id.
        """
        effective_threshold = self._resolve_threshold(threshold)
        source_pairs = extract_attribute_values(memory.content)
        found: list[_Candidate] = []
        for other in self._repository.active_memories():
            if other.id == memory.id:
                continue
            detail = similarity_details(memory.content, other.content)
            if detail.score < effective_threshold:
                continue
            if _disagrees(source_pairs, other.content, self._intelligence):
                # A disagreement is a conflict, not a duplicate.
                continue
            found.append((other, detail.score, detail))
        found.sort(key=lambda item: (-item[1], item[0].id))
        return found

    def _link(
        self,
        source_id: str,
        target_id: str,
        relationship: RelationshipType,
        similarity: float | None = None,
    ) -> None:
        """Record a relationship idempotently (never modifies memories)."""
        with suppress(RelationshipExistsError):
            self._repository.add_relationship(
                MemoryRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=relationship,
                    similarity=similarity,
                    created_at=self._clock.now(),
                )
            )


def _disagrees(
    source_pairs: tuple[AttributeValue, ...], other_content: str, config: IntelligenceConfig
) -> bool:
    """True when two texts state the same attribute with different values."""
    if not source_pairs:
        return False
    reconstruction = " ".join(f"{pair.raw_attribute} = {pair.raw_value}" for pair in source_pairs)
    return bool(
        find_conflicts(
            reconstruction,
            other_content,
            value_similarity_threshold=config.value_similarity_threshold,
        )
    )
