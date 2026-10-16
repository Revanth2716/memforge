"""Storage abstractions.

The service layer depends on :class:`MemoryRepository` (an abstract base
class), never on a concrete backend. Swapping SQLite for another local store --
or adding an Ollama-backed embedding store later -- means adding a new
implementation of this interface, with no service or API changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from app.domain.models import Memory, MemoryStatus, MemoryType, MemoryVersion, VersionChange
from app.domain.relationships import MemoryRelationship


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Filter + pagination for listing memories."""

    status: MemoryStatus | None = None
    memory_type: MemoryType | None = None
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.offset < 0:
            raise ValueError("offset must be >= 0")


@dataclass(frozen=True, slots=True)
class CandidateQuery:
    """Candidate generation for retrieval (see ``MemoryRepository.search_candidates``).

    ``status=None`` means "no status filter"; ``exclude_deleted`` then decides
    whether forgotten memories are still visible, so callers can ask for
    archived memories without ever surfacing forgotten ones by accident.
    """

    ids: tuple[str, ...] | None = None
    status: MemoryStatus | None = MemoryStatus.ACTIVE
    #: Keep memories that are not forgotten even when ``status`` is ``None``.
    exclude_deleted: bool = True
    memory_type: MemoryType | None = None
    min_confidence: float | None = None
    min_importance: float | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    def __post_init__(self) -> None:
        if self.min_confidence is not None and not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_importance is not None and not 0.0 <= self.min_importance <= 1.0:
            raise ValueError("min_importance must be in [0, 1]")
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_before < self.created_after
        ):
            raise ValueError("created_before must not be earlier than created_after")


class MemoryRepository(ABC):
    """Data-access contract for memory persistence."""

    @abstractmethod
    def initialize(self) -> None:
        """Create/upgrade the schema. Safe to call repeatedly."""

    @abstractmethod
    def ping(self) -> bool:
        """Return True when the backend answers a trivial query."""

    @abstractmethod
    def insert(self, memory: Memory, *, change: VersionChange) -> MemoryVersion:
        """Insert a brand-new memory together with its first history version."""

    @abstractmethod
    def get(self, memory_id: str, *, include_deleted: bool = False) -> Memory | None:
        """Return a memory by id, or ``None`` when it does not exist.

        By default forgotten (soft-deleted) memories are hidden.
        """

    @abstractmethod
    def find(self, query: MemoryQuery) -> tuple[list[Memory], int]:
        """Return a page of memories ordered newest-first plus the total count."""

    @abstractmethod
    def save(self, memory: Memory, *, change: VersionChange) -> MemoryVersion:
        """Persist a new version of an existing memory (row + history snapshot).

        ``memory.version`` must be exactly one greater than the stored version,
        otherwise a :class:`~app.domain.errors.RepositoryError` is raised.
        """

    @abstractmethod
    def versions(self, memory_id: str) -> list[MemoryVersion]:
        """Return the full history of a memory, oldest version first."""

    # -- M2: intelligence support ------------------------------------------

    @abstractmethod
    def active_memories(self) -> list[Memory]:
        """Return every ACTIVE memory (the deterministic scan set).

        Used by the lexical duplicate/conflict scans, which are O(n) over the
        active corpus. That is intentional and documented: M2 has no index
        structure and no embeddings.
        """

    @abstractmethod
    def record_access(self, memory_id: str, *, at: datetime) -> Memory:
        """Increment the access telemetry of a memory and return it.

        Metadata only: no history version, no ``version`` bump, and
        ``updated_at`` is deliberately left untouched. Forgotten memories raise
        :class:`~app.domain.errors.MemoryNotFoundError`.
        """

    @abstractmethod
    def add_relationship(self, relationship: MemoryRelationship) -> MemoryRelationship:
        """Persist a relationship.

        Raises :class:`~app.domain.errors.RelationshipExistsError` when the same
        relationship between the same pair already exists.
        """

    @abstractmethod
    def relationships(self, memory_id: str) -> list[MemoryRelationship]:
        """Return every relationship touching ``memory_id`` (both directions)."""

    @abstractmethod
    def superseded_pairs(self, memory_id: str) -> set[frozenset[str]]:
        """Memory pairs already resolved by a ``SUPERSEDES`` link."""

    # -- M3: retrieval support --------------------------------------------

    @abstractmethod
    def search_candidates(self, query: CandidateQuery) -> list[Memory]:
        """Return memories matching a candidate query, newest first.

        Candidate generation for retrieval: it can be driven by an explicit id
        set (from the inverted index) or by the filters alone, and it applies
        the retrieval filters in SQL so ranking only ever sees admissible rows.
        """

    @abstractmethod
    def searchable_corpus_signature(self) -> tuple[int, str]:
        """Cheap signature of the searchable (non-forgotten) corpus.

        Used to decide whether an in-memory inverted index is still current
        without rescanning the corpus on every query.
        """
