"""Explicit relationships between memories.

M2 keeps exactly three relationship kinds -- no graph infrastructure, no
traversal engine. Each row is an auditable fact recorded by a detection or
resolution operation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from app.domain.errors import InvalidRelationshipError


class RelationshipType(enum.StrEnum):
    """Kind of link recorded between two memories."""

    #: The two memories say (nearly) the same thing.
    DUPLICATE_OF = "DUPLICATE_OF"
    #: The two memories disagree about the same attribute.
    CONFLICTS_WITH = "CONFLICTS_WITH"
    #: The source memory was chosen over the target memory by resolution.
    SUPERSEDES = "SUPERSEDES"


#: Relationships that may be created automatically by detection/resolution.
AUTOMATIC_RELATIONSHIPS = (
    RelationshipType.DUPLICATE_OF,
    RelationshipType.CONFLICTS_WITH,
    RelationshipType.SUPERSEDES,
)


@dataclass(frozen=True, slots=True)
class MemoryRelationship:
    """An auditable link between two distinct memories."""

    source_id: str
    target_id: str
    relationship: RelationshipType
    created_at: datetime
    similarity: float | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise InvalidRelationshipError("source_id must be a non-empty string")
        if not self.target_id or not self.target_id.strip():
            raise InvalidRelationshipError("target_id must be a non-empty string")
        if self.source_id == self.target_id:
            raise InvalidRelationshipError("a memory cannot be related to itself")
        if not isinstance(self.relationship, RelationshipType):
            raise InvalidRelationshipError(f"unknown relationship {self.relationship!r}")
        if self.similarity is not None and not 0.0 <= self.similarity <= 1.0:
            raise InvalidRelationshipError("similarity must be in [0, 1]")
