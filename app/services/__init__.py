"""Service layer: memory lifecycle and intelligence rules."""

from app.services.memory_service import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DUPLICATE_LIMIT,
    DEFAULT_IMPORTANCE,
    ConflictCandidate,
    ConflictResolution,
    DuplicateMatch,
    MemoryPage,
    MemoryService,
)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_DUPLICATE_LIMIT",
    "DEFAULT_IMPORTANCE",
    "ConflictCandidate",
    "ConflictResolution",
    "DuplicateMatch",
    "MemoryPage",
    "MemoryService",
]
