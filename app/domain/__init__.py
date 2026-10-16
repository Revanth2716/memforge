"""MemForge domain layer.

The domain layer is intentionally free of framework and storage concerns: it
only uses the standard library. Everything an agent needs to reason about a
memory (identity, type, status, confidence, importance, versioning, scoring,
relationships) is modelled here and validated here, so invariants hold
regardless of which transport (REST, CLI, future MCP server) or storage backend
is used.
"""

from app.domain.attributes import (
    AttributeValue,
    ValueComparison,
    compare_values,
    extract_attribute_values,
    find_conflicts,
)
from app.domain.errors import (
    InvalidIntelligenceConfigError,
    InvalidMemoryStateError,
    InvalidRelationshipError,
    MemforgeError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryValidationError,
    NoConflictError,
    RelationshipExistsError,
    RepositoryError,
    StorageUnavailableError,
)
from app.domain.ids import IdGenerator, SequentialIdGenerator, UuidIdGenerator
from app.domain.intelligence import (
    DEFAULT_IMPORTANCE_WEIGHTS,
    DEFAULT_PRIORITY_WEIGHTS,
    DEFAULT_TYPE_WEIGHTS,
    FactorScore,
    ImportanceBreakdown,
    IntelligenceConfig,
    PriorityBreakdown,
    ResolutionDecision,
    ResolutionStep,
    age_in_days,
    compute_effective_priority,
    compute_importance,
    decay_multiplier,
    decide_conflict,
    saturation_factor,
)
from app.domain.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    VersionChange,
    build_version,
)
from app.domain.relationships import (
    AUTOMATIC_RELATIONSHIPS,
    MemoryRelationship,
    RelationshipType,
)
from app.domain.similarity import (
    SimilarityDetail,
    dice_similarity,
    normalize,
    normalized_tokens,
    similarity_details,
    stem,
    token_set,
    tokenize,
)
from app.domain.time import Clock, ManualClock, SystemClock
from app.domain.updates import UNSET, MemoryUpdate, Unset

__all__ = [
    "AUTOMATIC_RELATIONSHIPS",
    "DEFAULT_IMPORTANCE_WEIGHTS",
    "DEFAULT_PRIORITY_WEIGHTS",
    "DEFAULT_TYPE_WEIGHTS",
    "UNSET",
    "AttributeValue",
    "Clock",
    "FactorScore",
    "IdGenerator",
    "ImportanceBreakdown",
    "IntelligenceConfig",
    "InvalidIntelligenceConfigError",
    "InvalidMemoryStateError",
    "InvalidRelationshipError",
    "ManualClock",
    "MemforgeError",
    "Memory",
    "MemoryAlreadyExistsError",
    "MemoryNotFoundError",
    "MemoryRelationship",
    "MemoryStatus",
    "MemoryType",
    "MemoryUpdate",
    "MemoryValidationError",
    "MemoryVersion",
    "NoConflictError",
    "PriorityBreakdown",
    "RelationshipExistsError",
    "RelationshipType",
    "RepositoryError",
    "ResolutionDecision",
    "ResolutionStep",
    "SequentialIdGenerator",
    "SimilarityDetail",
    "StorageUnavailableError",
    "SystemClock",
    "Unset",
    "UuidIdGenerator",
    "ValueComparison",
    "VersionChange",
    "age_in_days",
    "build_version",
    "compare_values",
    "compute_effective_priority",
    "compute_importance",
    "decay_multiplier",
    "decide_conflict",
    "dice_similarity",
    "extract_attribute_values",
    "find_conflicts",
    "normalize",
    "normalized_tokens",
    "saturation_factor",
    "similarity_details",
    "stem",
    "token_set",
    "tokenize",
]
