"""Memory-aware retrieval: queries, ranking and explainable results.

Every function in this module is pure: it takes memories, a query, a
configuration and a reference time, and returns scores and explanations. No
clock is read implicitly, no storage is touched and no model is called.

Ranking formula (all inputs bounded in ``[0, 1]``)
-------------------------------------------------
::

    base = w_relevance  * relevance       # lexical, semantic or hybrid blend
         + w_priority   * priority        # M2 effective priority
         + w_confidence * confidence      # stored accuracy
         + w_freshness  * freshness       # M2 decay factor
         + w_type       * type_weight     # documented type preference table

    final = base
          * (1 - w_conflict    * conflict_penalty)
          * (1 - w_redundancy  * redundancy_penalty)

* ``relevance``, ``priority``, ``confidence``, ``freshness`` and ``type`` are the
  five ranking components; their weights must sum to ``1.0`` (validated), so
  ``base`` is a convex combination and therefore in ``[0, 1]``.
* ``conflict_penalty`` comes from the M2 relationship graph: ``0.0`` when the
  memory wins its conflicts, ``0.5`` when a conflict is unresolved, ``1.0`` when
  the memory was superseded by another.
* ``redundancy_penalty`` is ``1.0`` for a candidate that duplicates a
  higher-ranked memory in the same result set and ``0.0`` otherwise; suppression
  happens during selection, so no memory is ever deleted.
* Both penalty strengths are in ``[0, 1]``, so ``final`` stays in ``[0, 1]``.

``relevance`` itself is a documented blend::

    lexical    = idf-weighted coverage of the query terms  (inverted index)
    semantic   = cosine similarity to the query embedding   (optional provider)
    hybrid     = (1 - semantic_weight) * lexical + semantic_weight * semantic

Scores are rounded to six decimals, and every result carries the exact factor
values, weights and contributions that produced it.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from app.domain.errors import InvalidRetrievalConfigError, RetrievalValidationError
from app.domain.intelligence import (
    DEFAULT_TYPE_WEIGHTS,
    decay_multiplier,
)
from app.domain.models import Memory, MemoryStatus, MemoryType
from app.domain.similarity import normalized_tokens

#: Documented type preference for retrieval ranking. Profile and fact memories
#: are treated as more useful answers than observations. These are explicit
#: heuristics, not measurements.
RETRIEVAL_TYPE_WEIGHTS: Mapping[MemoryType, float] = MappingProxyType(
    {
        MemoryType.PROFILE: 1.0,
        MemoryType.FACT: 0.9,
        MemoryType.PREFERENCE: 0.8,
        MemoryType.TASK: 0.7,
        MemoryType.EVENT: 0.6,
        MemoryType.OBSERVATION: 0.4,
    }
)

#: Decimal places kept in reported scores (keeps output stable and readable).
_SCORE_PRECISION = 6


class RetrievalMode(enum.StrEnum):
    """Requested retrieval strategy."""

    #: Deterministic inverted-index search only (the default, always available).
    LEXICAL = "lexical"
    #: Embedding similarity only; requires an available provider.
    SEMANTIC = "semantic"
    #: Documented blend of both; requires an available provider.
    HYBRID = "hybrid"


def _validate_unit(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise InvalidRetrievalConfigError(f"{label} must be in [0, 1] (got {value})")


def _validate_query_unit(value: float, label: str) -> None:
    """Query parameters raise a validation error, not a configuration error."""
    if not 0.0 <= value <= 1.0:
        raise RetrievalValidationError(f"{label} must be in [0, 1] (got {value})")


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Typed filters applied before ranking."""

    memory_type: MemoryType | None = None
    status: MemoryStatus | None = MemoryStatus.ACTIVE
    min_confidence: float | None = None
    min_importance: float | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    def __post_init__(self) -> None:
        if self.min_confidence is not None:
            _validate_unit(self.min_confidence, "min_confidence")
        if self.min_importance is not None:
            _validate_unit(self.min_importance, "min_importance")
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_before < self.created_after
        ):
            raise RetrievalValidationError("created_before must not be earlier than created_after")

    def matches(self, memory: Memory) -> bool:
        """True when a memory satisfies every filter."""
        if self.memory_type is not None and memory.memory_type is not self.memory_type:
            return False
        if self.status is not None and memory.status is not self.status:
            return False
        if self.min_confidence is not None and memory.confidence < self.min_confidence:
            return False
        if self.min_importance is not None and memory.importance < self.min_importance:
            return False
        if self.created_after is not None and memory.created_at < self.created_after:
            return False
        # (negated condition inlined: the memory must not be newer than the bound)
        return self.created_before is None or memory.created_at <= self.created_before


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A retrieval request."""

    text: str
    top_k: int = 10
    mode: RetrievalMode = RetrievalMode.LEXICAL
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    #: Blend weight for the semantic component in ``hybrid`` mode.
    semantic_weight: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise RetrievalValidationError("query text must be a non-empty string")
        if self.top_k < 1:
            raise RetrievalValidationError("top_k must be >= 1")
        if self.semantic_weight is not None:
            _validate_query_unit(self.semantic_weight, "semantic_weight")

    @property
    def terms(self) -> tuple[str, ...]:
        """Normalised, de-duplicated query terms."""
        return tuple(dict.fromkeys(normalized_tokens(self.text)))


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Weights for the memory-aware ranking formula.

    The five component weights must sum to ``1.0``; the two penalty strengths
    are independent scalars in ``[0, 1]``.
    """

    relevance_weight: float = 0.45
    priority_weight: float = 0.20
    confidence_weight: float = 0.15
    freshness_weight: float = 0.10
    type_weight: float = 0.10
    #: Strength of the conflict penalty (1.0 = fully demote a superseded memory).
    conflict_penalty_weight: float = 0.5
    #: Strength of the duplicate penalty (1.0 = fully demote a redundant memory).
    redundancy_penalty_weight: float = 0.7
    #: Default semantic weight used by ``hybrid`` mode.
    default_semantic_weight: float = 0.5
    type_weights: Mapping[MemoryType, float] = field(default=RETRIEVAL_TYPE_WEIGHTS)
    decay_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        weights = {
            "relevance_weight": self.relevance_weight,
            "priority_weight": self.priority_weight,
            "confidence_weight": self.confidence_weight,
            "freshness_weight": self.freshness_weight,
            "type_weight": self.type_weight,
        }
        for name, value in weights.items():
            _validate_unit(value, name)
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise InvalidRetrievalConfigError(f"component weights must sum to 1.0 (got {total})")
        _validate_unit(self.conflict_penalty_weight, "conflict_penalty_weight")
        _validate_unit(self.redundancy_penalty_weight, "redundancy_penalty_weight")
        _validate_unit(self.default_semantic_weight, "default_semantic_weight")
        if self.decay_half_life_days <= 0:
            raise InvalidRetrievalConfigError("decay_half_life_days must be > 0")

    @property
    def component_weights(self) -> Mapping[str, float]:
        """The five component weights, keyed by component name."""
        return MappingProxyType(
            {
                "relevance": self.relevance_weight,
                "effective_priority": self.priority_weight,
                "confidence": self.confidence_weight,
                "freshness": self.freshness_weight,
                "type_preference": self.type_weight,
            }
        )

    def semantic_weight_for(self, requested: float | None) -> float:
        """The semantic blend weight to use for a request."""
        return self.default_semantic_weight if requested is None else requested


@dataclass(frozen=True, slots=True)
class RankingFactor:
    """One component of the final score, with its contribution."""

    name: str
    weight: float
    value: float
    contribution: float
    explanation: str


@dataclass(frozen=True, slots=True)
class ConflictInfo:
    """How a memory relates to a conflict it participates in."""

    target_memory_id: str
    relationship: str
    attribute: str
    source_value: str
    target_value: str
    resolution: str
    resolution_reason: str | None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked memory, with everything needed to explain the ranking."""

    memory: Memory
    lexical_score: float
    semantic_score: float | None
    relevance: float
    final_score: float
    components: tuple[RankingFactor, ...]
    explanation: str
    conflicts: tuple[ConflictInfo, ...]
    superseded_by: str | None
    duplicate_of: str | None
    suppressed: bool
    suppression_reason: str | None


@dataclass(frozen=True, slots=True)
class SuppressedCandidate:
    """A candidate that was deliberately left out of the top-k."""

    memory_id: str
    final_score: float
    reason: str


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """Query-level metadata plus the ranked results."""

    query: str
    top_k: int
    requested_mode: RetrievalMode
    retrieval_mode: RetrievalMode
    semantic_available: bool
    semantic_provider: str | None
    semantic_model: str | None
    degraded_reason: str | None
    candidates_considered: int
    candidates_returned: int
    candidates_suppressed: int
    latency_ms: float
    results: tuple[RetrievalResult, ...]
    suppressed: tuple[SuppressedCandidate, ...]


# -- scoring primitives -------------------------------------------------------


def idf(corpus_size: int, document_frequency: int) -> float:
    """Smoothed inverse document frequency (always positive)."""
    if document_frequency <= 0:
        return 0.0
    return round(math.log((corpus_size + 1) / (document_frequency + 1)) + 1.0, _SCORE_PRECISION)


def lexical_relevance(
    query_terms: Sequence[str],
    document_tokens: frozenset[str],
    *,
    corpus_size: int,
    document_frequencies: Mapping[str, int],
) -> float:
    """idf-weighted coverage of the query terms, bounded in ``[0, 1]``.

    A term that no memory contains cannot be covered, so it is excluded from
    both sides of the ratio. The result is therefore "the share of the query's
    information mass that this memory covers".
    """
    if not query_terms:
        return 0.0
    scored = [term for term in query_terms if document_frequencies.get(term, 0) > 0]
    if not scored:
        return 0.0
    denominator = sum(idf(corpus_size, document_frequencies[term]) for term in scored)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(
        idf(corpus_size, document_frequencies[term]) for term in scored if term in document_tokens
    )
    return round(min(1.0, numerator / denominator), _SCORE_PRECISION)


def semantic_relevance(similarity: float) -> float:
    """Clamp a cosine similarity into ``[0, 1]``."""
    return round(max(0.0, min(1.0, similarity)), _SCORE_PRECISION)


def hybrid_relevance(lexical: float, semantic: float, *, semantic_weight: float) -> float:
    """Documented blend: ``(1 - w) * lexical + w * semantic``."""
    _validate_unit(semantic_weight, "semantic_weight")
    return round((1.0 - semantic_weight) * lexical + semantic_weight * semantic, _SCORE_PRECISION)


def freshness(memory: Memory, *, now: datetime, half_life_days: float) -> float:
    """Decay factor from M2: 1.0 today, 0.5 after one half-life."""
    age_days = max(0.0, (now - memory.updated_at).total_seconds() / 86400.0)
    return round(decay_multiplier(age_days, half_life_days), _SCORE_PRECISION)


def type_preference(memory: Memory, config: RankingConfig) -> float:
    """Documented type preference for retrieval."""
    return float(config.type_weights[memory.memory_type])


def conflict_penalty_from(resolutions: Sequence[str]) -> float:
    """``1.0`` superseded, ``0.5`` unresolved, ``0.0`` winner/no conflict."""
    if not resolutions:
        return 0.0
    if any(resolution == "superseded_by_other" for resolution in resolutions):
        return 1.0
    if any(resolution == "unresolved" for resolution in resolutions):
        return 0.5
    return 0.0


def final_score(
    *,
    relevance: float,
    priority: float,
    confidence: float,
    freshness_value: float,
    type_value: float,
    conflict_penalty_value: float,
    redundancy_penalty_value: float,
    config: RankingConfig,
) -> float:
    """Apply the documented ranking formula; the result is in ``[0, 1]``."""
    weights = config.component_weights
    base = (
        weights["relevance"] * relevance
        + weights["effective_priority"] * priority
        + weights["confidence"] * confidence
        + weights["freshness"] * freshness_value
        + weights["type_preference"] * type_value
    )
    conflict_factor = 1.0 - config.conflict_penalty_weight * conflict_penalty_value
    redundancy_factor = 1.0 - config.redundancy_penalty_weight * redundancy_penalty_value
    return round(base * conflict_factor * redundancy_factor, _SCORE_PRECISION)


def ranking_explanation(
    *,
    query: str,
    factors: Sequence[RankingFactor],
    conflict_penalty_value: float,
    redundancy_penalty_value: float,
    config: RankingConfig,
    final: float,
) -> str:
    """Human readable summary of why a memory scored what it scored."""
    parts = [
        f"{factor.name}={factor.value:.2f} (x{factor.weight:.2f} -> {factor.contribution:.4f})"
        for factor in factors
    ]
    text = f"final {final:.4f} for query {query!r} = " + " + ".join(parts)
    if conflict_penalty_value:
        text += (
            f" | conflict penalty x{config.conflict_penalty_weight * conflict_penalty_value:.2f} "
            f"({conflict_penalty_value:.2f} penalised)"
        )
    if redundancy_penalty_value:
        text += (
            f" | redundancy penalty x{config.redundancy_penalty_weight * redundancy_penalty_value:.2f} "
            f"({redundancy_penalty_value:.2f} penalised)"
        )
    return text


def sort_key(result: RetrievalResult) -> tuple[float, str]:
    """Deterministic ordering: score descending, then id ascending."""
    return (-result.final_score, result.memory.id)


__all__ = [
    "DEFAULT_TYPE_WEIGHTS",
    "RETRIEVAL_TYPE_WEIGHTS",
    "ConflictInfo",
    "RankingConfig",
    "RankingFactor",
    "RetrievalFilters",
    "RetrievalMode",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievalResult",
    "SuppressedCandidate",
    "conflict_penalty_from",
    "final_score",
    "freshness",
    "hybrid_relevance",
    "idf",
    "lexical_relevance",
    "ranking_explanation",
    "semantic_relevance",
    "sort_key",
    "type_preference",
]
