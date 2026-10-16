"""M4 context domain: the deterministic memory-to-context compiler.

    MEMORY RETRIEVAL (M3)
            |
            v
    CONTEXT COMPILER (this module)
            |
            v
    BUDGETED AGENT CONTEXT

This is **not** summarisation, not a chatbot and not another document RAG
pipeline. It is a compiler: given retrieved memories and a hard token budget it
emits the best possible *context*, together with a complete explanation of every
decision. Nothing here reads a clock, touches storage or calls a model; every
function is pure and independently testable, and the service layer
(``app.services.context_service``) is the only thing that composes them.

Utility model (all factors bounded in ``[0, 1]``)
-------------------------------------------------
::

    value = w_relevance      * relevance           # M3 lexical/semantic/hybrid blend
          + w_priority       * effective_priority  # M2 decayed priority
          + w_confidence     * confidence          # stored accuracy
          + w_freshness      * freshness           # M2 decay factor
          + w_type           * type_preference     # documented type table

    cost_penalty = min(1, estimated_tokens / cost_reference_tokens)
    utility      = value * (1 - cost_weight * cost_penalty)

* The five value weights must sum to ``1.0`` (validated), so ``value`` is a
  convex combination and therefore in ``[0, 1]``.
* ``cost_penalty`` is in ``[0, 1]`` and ``cost_weight`` is in ``[0, 1]``, so
  ``utility`` is in ``[0, 1]`` as well: no score can be inflated by a
  misconfiguration, and a memory is never rewarded for being long.
* Utility is **not** the M3 ``final_score``: M3 answers *"how relevant is this?"*,
  M4 answers *"how much of the remaining budget is this worth?"*.

Budget algorithm
----------------
1. every candidate is scored (``value`` and ``cost_penalty``);
2. candidates are filtered by the request (include/exclude ids, memory-type
   preferences, minimum score);
3. redundant candidates are suppressed (M2 duplicates, M2 conflict winners);
4. candidates are ordered by ``(-utility, memory_id)`` -- a *total* order, so
   ties never depend on iteration order;
5. candidates are admitted greedily while ``estimated_tokens <= remaining``;
6. a candidate that does not fit is **reported and skipped, never fatal**: the
   loop continues, so a smaller (lower-utility) candidate can still use the
   leftover budget;
7. the rendered context (section headers included) is measured exactly; if the
   framing pushed it over the limit, the lowest-priority block is dropped until
   it fits. The hard limit therefore holds for the text the agent actually
   receives.

Ordering
--------
Prompt order is **not** retrieval order. Blocks are grouped into documented
sections and ordered by ``(section_rank, -utility, memory_id)``:

===  ==========================================  ==========================
rank section            memory types              rationale
===  ==========================================  ==========================
0    ``PROFILE``         PROFILE, PREFERENCE       stable identity and
                                                  stated preferences first
1    ``FACTS``           FACT                      durable, authoritative
2    ``RECENT_EVENTS``   EVENT                     what happened lately
3    ``TASKS``           TASK                      open work
4    ``OBSERVATIONS``    OBSERVATION               lowest-confidence notes
===  ==========================================  ==========================

Compression
-----------
M4 never paraphrases. It only (a) normalises whitespace and (b) optionally keeps
the highest-scoring *sentences* of an over-long memory, in their original order.
Both are traceable: the original text, the retained segment indices and both
token figures are kept. Compression does **not** guarantee semantic equivalence
(see the README limitations).

Provenance
----------
Every block carries the memory id, type, source, confidence, importance,
retrieval score, selection score, estimated tokens, the compression record and
any conflict it participates in. Every candidate considered -- selected or not
-- appears in ``selection_trace`` with the reason it was admitted or rejected.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from app.domain.errors import ContextValidationError, InvalidContextConfigError
from app.domain.models import Memory, MemoryStatus, MemoryType
from app.domain.retrieval import (
    ConflictInfo,
    RankingFactor,
    RetrievalFilters,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
)
from app.domain.similarity import normalized_tokens
from app.domain.tokens import (
    TokenEstimatorConfig,
    estimate_tokens,
    normalize_whitespace,
)

#: Decimal places kept in reported scores (keeps output stable and readable).
_SCORE_PRECISION = 6

#: Prefix of one rendered context line. Counted in the per-block token cost so
#: the budget accounts for the framing the agent actually receives.
BLOCK_PREFIX = "- "


class ContextSection(enum.StrEnum):
    """A section of the compiled context, in prompt order."""

    PROFILE = "PROFILE"
    FACTS = "FACTS"
    RECENT_EVENTS = "RECENT_EVENTS"
    TASKS = "TASKS"
    OBSERVATIONS = "OBSERVATIONS"

    @property
    def heading(self) -> str:
        """Human readable heading of the section (its wire name, spelled out)."""
        return _SECTION_HEADINGS[self]


_SECTION_HEADINGS: Mapping[ContextSection, str] = MappingProxyType(
    {
        ContextSection.PROFILE: "PROFILE",
        ContextSection.FACTS: "FACTS",
        ContextSection.RECENT_EVENTS: "RECENT EVENTS",
        ContextSection.TASKS: "TASKS",
        ContextSection.OBSERVATIONS: "OBSERVATIONS",
    }
)

#: Default mapping from memory type to context section (a documented heuristic:
#: profile and preference memories are stable identity, observations are notes).
DEFAULT_TYPE_SECTIONS: Mapping[MemoryType, ContextSection] = MappingProxyType(
    {
        MemoryType.PROFILE: ContextSection.PROFILE,
        MemoryType.PREFERENCE: ContextSection.PROFILE,
        MemoryType.FACT: ContextSection.FACTS,
        MemoryType.EVENT: ContextSection.RECENT_EVENTS,
        MemoryType.TASK: ContextSection.TASKS,
        MemoryType.OBSERVATION: ContextSection.OBSERVATIONS,
    }
)

#: Default prompt order of the sections.
DEFAULT_SECTION_ORDER: tuple[ContextSection, ...] = (
    ContextSection.PROFILE,
    ContextSection.FACTS,
    ContextSection.RECENT_EVENTS,
    ContextSection.TASKS,
    ContextSection.OBSERVATIONS,
)


class SelectionReason(enum.StrEnum):
    """Why a candidate was admitted into the context, or rejected from it."""

    #: Admitted because it had the highest utility that still fit.
    SELECTED = "selected"
    #: Redundant: another candidate is a recorded near-duplicate of it.
    DUPLICATE_OF = "duplicate_of"
    #: Contradicted: another candidate won an M2 conflict resolution.
    CONFLICT_SUPERSEDED = "conflict_superseded"
    #: Excluded by the request's minimum score.
    BELOW_MINIMUM_SCORE = "below_minimum_score"
    #: Excluded because its type is not among the requested preferences.
    MEMORY_TYPE_FILTERED = "memory_type_filtered"
    #: Excluded by the request's ``exclude_ids``.
    EXCLUDED_BY_ID = "excluded_by_id"
    #: Not among the request's ``include_ids`` (when that list is non-empty).
    NOT_IN_INCLUDE_IDS = "not_in_include_ids"
    #: Admissible, but it did not fit in the remaining budget.
    INSUFFICIENT_BUDGET = "insufficient_budget"

    @property
    def is_admission(self) -> bool:
        """True when the reason means "the candidate entered the context"."""
        return self is SelectionReason.SELECTED

    @property
    def description(self) -> str:
        """One-line documentation of the reason (used in explanations)."""
        return _REASON_DESCRIPTIONS[self]


_REASON_DESCRIPTIONS: Mapping[SelectionReason, str] = MappingProxyType(
    {
        SelectionReason.SELECTED: "admitted: highest utility that fits the remaining budget",
        SelectionReason.DUPLICATE_OF: (
            "suppressed: a recorded near-duplicate in this candidate set is the representative"
        ),
        SelectionReason.CONFLICT_SUPERSEDED: (
            "suppressed: another candidate won the M2 conflict resolution for this attribute"
        ),
        SelectionReason.BELOW_MINIMUM_SCORE: "excluded: utility below the requested minimum",
        SelectionReason.MEMORY_TYPE_FILTERED: "excluded: memory type not among the requested types",
        SelectionReason.EXCLUDED_BY_ID: "excluded: listed in exclude_ids",
        SelectionReason.NOT_IN_INCLUDE_IDS: "excluded: not listed in include_ids",
        SelectionReason.INSUFFICIENT_BUDGET: (
            "skipped: estimated token cost exceeds the remaining budget"
        ),
    }
)


class CompressionStep(enum.StrEnum):
    """A deterministic transformation applied to memory content."""

    #: Runs of whitespace collapsed to a single space (meaning-preserving).
    WHITESPACE = "whitespace_normalised"
    #: Only the highest-scoring sentences of an over-long memory were kept.
    SENTENCES = "salient_sentences"


# -- configuration ------------------------------------------------------------


def _validate_unit(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InvalidContextConfigError(f"{label} must be in [0, 1] (got {value})")


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Documented, validated knobs of the context compiler.

    Every weight lives here (and not in the environment) so a context policy is
    a reviewable value object. :meth:`app.config.Settings.context_policy` maps
    the operational subset of them from ``MEMFORGE_*`` variables.
    """

    # -- utility weights (must sum to 1.0) --------------------------------
    relevance_weight: float = 0.40
    priority_weight: float = 0.20
    confidence_weight: float = 0.15
    freshness_weight: float = 0.10
    type_weight: float = 0.15

    # -- cost awareness ---------------------------------------------------
    #: How strongly the estimated token cost reduces utility (0 = ignore size).
    cost_weight: float = 0.35
    #: Token count at which the cost penalty reaches its full strength.
    cost_reference_tokens: int = 120

    # -- admission --------------------------------------------------------
    #: Candidates whose utility is below this are never selected.
    min_candidate_score: float = 0.0

    # -- compression ------------------------------------------------------
    #: Apply the deterministic (never paraphrasing) compression rules.
    compress: bool = True
    #: Soft per-block ceiling. Above it, only salient sentences are kept.
    #: ``None`` disables sentence selection. A block that cannot be reduced
    #: below the ceiling is *not* truncated mid-sentence: it is simply more
    #: expensive, and the budget decides whether it fits.
    max_block_tokens: int | None = None

    # -- ordering ---------------------------------------------------------
    type_sections: Mapping[MemoryType, ContextSection] = field(default=DEFAULT_TYPE_SECTIONS)
    section_order: tuple[ContextSection, ...] = field(default=DEFAULT_SECTION_ORDER)

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
        total = math.fsum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise InvalidContextConfigError(f"utility weights must sum to 1.0 (got {total})")
        _validate_unit(self.cost_weight, "cost_weight")
        _validate_unit(self.min_candidate_score, "min_candidate_score")
        if isinstance(self.cost_reference_tokens, bool) or not isinstance(
            self.cost_reference_tokens, int
        ):
            raise InvalidContextConfigError("cost_reference_tokens must be an integer")
        if self.cost_reference_tokens < 1:
            raise InvalidContextConfigError("cost_reference_tokens must be >= 1")
        if self.max_block_tokens is not None and (
            isinstance(self.max_block_tokens, bool) or self.max_block_tokens < 1
        ):
            raise InvalidContextConfigError("max_block_tokens must be >= 1 or None")
        missing_types = set(MemoryType) - set(self.type_sections)
        if missing_types:
            raise InvalidContextConfigError(
                f"type_sections is missing memory types: {sorted(t.value for t in missing_types)}"
            )
        for memory_type, section in self.type_sections.items():
            if not isinstance(section, ContextSection):
                raise InvalidContextConfigError(
                    f"type_sections[{memory_type}] is not a ContextSection"
                )
        if len(set(self.section_order)) != len(ContextSection) or set(self.section_order) != set(
            ContextSection
        ):
            raise InvalidContextConfigError(
                "section_order must list every ContextSection exactly once"
            )

    # -- derived views ----------------------------------------------------

    @property
    def utility_weights(self) -> Mapping[str, float]:
        """The five value weights, keyed by factor name."""
        return MappingProxyType(
            {
                "relevance": self.relevance_weight,
                "effective_priority": self.priority_weight,
                "confidence": self.confidence_weight,
                "freshness": self.freshness_weight,
                "type_preference": self.type_weight,
            }
        )

    def section_for(self, memory_type: MemoryType) -> ContextSection:
        """The section a memory type belongs to."""
        return self.type_sections[memory_type]

    def section_rank(self, section: ContextSection) -> int:
        """Position of a section in the prompt (0 is rendered first)."""
        return self.section_order.index(section)

    def formula(self) -> str:
        """One-line description of the utility model, for explanations."""
        weights = self.utility_weights
        return (
            "utility = value * (1 - "
            f"{self.cost_weight:g} * min(1, tokens/{self.cost_reference_tokens})) where value = "
            + " + ".join(f"{weight:g}*{name}" for name, weight in weights.items())
        )


# -- request ------------------------------------------------------------------


def _validate_optional_unit(value: float | None, label: str) -> None:
    """Request parameters raise a validation error, not a configuration error."""
    if value is not None and not 0.0 <= value <= 1.0:
        raise ContextValidationError(f"{label} must be in [0, 1] (got {value})")


def _validate_ids(ids: tuple[str, ...], label: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ContextValidationError(f"{label} must contain non-empty memory ids")


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """A context compilation request.

    Mirrors the M3 retrieval contract: the same query text, the same filters,
    the same retrieval modes -- plus the strict token budget that makes this an
    M4 request rather than a search.
    """

    #: Query / task the context is compiled for.
    query: str
    #: Hard upper bound in **estimated** tokens (see ``app.domain.tokens``).
    token_budget: int
    #: Maximum number of retrieval candidates handed to the compiler.
    top_k: int = 20
    #: Requested retrieval strategy (may be degraded by M3, and says so).
    retrieval_mode: RetrievalMode = RetrievalMode.LEXICAL
    semantic_weight: float | None = None
    #: Optional memory-type preference. One type is pushed down into the M3
    #: filter; several are applied by the compiler after retrieval.
    preferred_memory_types: tuple[MemoryType, ...] | None = None
    min_confidence: float | None = None
    min_importance: float | None = None
    include_ids: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()
    #: Soft per-block ceiling override (``None`` uses the policy default).
    max_block_tokens: int | None = None
    #: Utility floor override (``None`` uses the policy default).
    min_candidate_score: float | None = None
    #: Disable deterministic compression for this request.
    compress: bool | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ContextValidationError("query must be a non-empty string")
        if isinstance(self.token_budget, bool) or not isinstance(self.token_budget, int):
            raise ContextValidationError("token_budget must be an integer")
        if self.token_budget < 1:
            raise ContextValidationError(
                f"token_budget must be >= 1 estimated token (got {self.token_budget})"
            )
        if isinstance(self.top_k, bool) or self.top_k < 1:
            raise ContextValidationError("top_k must be >= 1")
        if self.preferred_memory_types is not None:
            if not self.preferred_memory_types:
                raise ContextValidationError(
                    "preferred_memory_types must be omitted or contain at least one type"
                )
            if len(set(self.preferred_memory_types)) != len(self.preferred_memory_types):
                raise ContextValidationError("preferred_memory_types must not repeat a type")
            for memory_type in self.preferred_memory_types:
                if not isinstance(memory_type, MemoryType):
                    raise ContextValidationError(
                        f"unknown memory type {memory_type!r} in preferred_memory_types"
                    )
        _validate_optional_unit(self.min_confidence, "min_confidence")
        _validate_optional_unit(self.min_importance, "min_importance")
        _validate_optional_unit(self.min_candidate_score, "min_candidate_score")
        _validate_optional_unit(self.semantic_weight, "semantic_weight")
        _validate_ids(self.include_ids, "include_ids")
        _validate_ids(self.exclude_ids, "exclude_ids")
        overlap = set(self.include_ids) & set(self.exclude_ids)
        if overlap:
            raise ContextValidationError(
                f"include_ids and exclude_ids must not overlap (got {sorted(overlap)})"
            )
        if self.max_block_tokens is not None and self.max_block_tokens < 1:
            raise ContextValidationError("max_block_tokens must be >= 1 or None")

    # -- helpers ----------------------------------------------------------

    @property
    def terms(self) -> tuple[str, ...]:
        """Normalised, de-duplicated query terms (same pipeline as M3)."""
        return tuple(dict.fromkeys(normalized_tokens(self.query)))

    @property
    def single_preferred_type(self) -> MemoryType | None:
        """The preferred type that can be pushed into the M3 filter."""
        if self.preferred_memory_types is not None and len(self.preferred_memory_types) == 1:
            return self.preferred_memory_types[0]
        return None

    def prefers(self, memory_type: MemoryType) -> bool:
        """True when the request does not restrict this memory type."""
        if self.preferred_memory_types is None:
            return True
        return memory_type in self.preferred_memory_types

    def admits_id(self, memory_id: str) -> bool:
        """Apply the ``include_ids`` / ``exclude_ids`` allow and deny lists."""
        if memory_id in self.exclude_ids:
            return False
        return not self.include_ids or memory_id in self.include_ids

    def to_retrieval_query(self) -> RetrievalQuery:
        """Translate the request into an M3 :class:`RetrievalQuery`.

        Retrieval stays M3's job: the compiler never re-implements candidate
        generation, ranking or relevance.
        """
        return RetrievalQuery(
            text=self.query,
            top_k=self.top_k,
            mode=self.retrieval_mode,
            filters=RetrievalFilters(
                memory_type=self.single_preferred_type,
                status=self.status,
                min_confidence=self.min_confidence,
                min_importance=self.min_importance,
            ),
            semantic_weight=self.semantic_weight,
        )


# -- compression --------------------------------------------------------------


def split_segments(text: str) -> tuple[str, ...]:
    """Split text into deterministic segments.

    Segmentation happens at line breaks first (the most explicit structure a
    memory can carry) and then at sentence punctuation -- but only when the
    punctuation really is a boundary (followed by whitespace or the end of the
    line), so ``3.14`` stays intact. No NLP, no model, no randomness.
    """
    segments: list[str] = []
    for line in text.splitlines():
        current: list[str] = []
        index = 0
        length = len(line)
        while index < length:
            character = line[index]
            current.append(character)
            index += 1
            if character in ".!?;" and (index >= length or line[index].isspace()):
                segments.append("".join(current).strip())
                current = []
        tail = "".join(current).strip()
        if tail:
            segments.append(tail)
    return tuple(segment for segment in segments if segment)


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """The traceable record of what compression did to one memory."""

    original_text: str
    compressed_text: str
    applied: tuple[CompressionStep, ...]
    original_tokens: int
    compressed_tokens: int
    retained_segments: tuple[int, ...] = ()

    @property
    def tokens_saved(self) -> int:
        """Tokens the compression saved (never negative)."""
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def changed(self) -> bool:
        """True when any transformation was applied."""
        return bool(self.applied)

    @property
    def truncated(self) -> bool:
        """True when whole segments were dropped (never mid-sentence)."""
        return CompressionStep.SENTENCES in self.applied


def compress_content(
    content: str,
    *,
    query_terms: Sequence[str] = (),
    policy: ContextPolicy | None = None,
    estimator: TokenEstimatorConfig | None = None,
    enabled: bool | None = None,
    max_block_tokens: int | None = None,
) -> CompressionResult:
    """Apply the deterministic, never-paraphrasing compression rules.

    Args:
        content: The original memory content (kept in the result verbatim).
        query_terms: Terms used to rank candidate segments.
        policy: Compression policy (``None`` uses the defaults).
        estimator: Token estimator configuration.
        enabled: Per-request override of ``policy.compress``.
        max_block_tokens: Per-request override of ``policy.max_block_tokens``.

    The result always keeps ``original_text``, so the rendered context can be
    traced back to the exact stored content.
    """
    resolved = policy or ContextPolicy()
    use_compression = resolved.compress if enabled is None else enabled
    ceiling = resolved.max_block_tokens if max_block_tokens is None else max_block_tokens

    text = normalize_whitespace(content) if use_compression else content
    applied: list[CompressionStep] = []
    if use_compression and text != content:
        applied.append(CompressionStep.WHITESPACE)

    original_tokens = estimate_tokens(content, config=estimator).estimated_tokens
    estimated = estimate_tokens(text, config=estimator).estimated_tokens
    retained: tuple[int, ...] = ()

    if ceiling is not None and estimated > ceiling:
        segments = split_segments(content)
        if len(segments) > 1:
            kept = _select_segments(
                segments,
                query_terms=query_terms,
                ceiling=ceiling,
                estimator=estimator,
            )
            if len(kept) < len(segments):
                retained = kept
                applied.append(CompressionStep.SENTENCES)
                text = " ".join(segments[index] for index in kept)
                estimated = estimate_tokens(text, config=estimator).estimated_tokens
    return CompressionResult(
        original_text=content,
        compressed_text=text,
        applied=tuple(applied),
        original_tokens=original_tokens,
        compressed_tokens=estimated,
        retained_segments=retained,
    )


def _segment_key(
    segment: str,
    position: int,
    terms: frozenset[str],
) -> tuple[int, int, int, str]:
    """Deterministic ranking key: most query coverage, then earliest, then text."""
    tokens = frozenset(normalized_tokens(segment))
    return (-len(tokens & terms), position, -len(segment), segment)


def _select_segments(
    segments: Sequence[str],
    *,
    query_terms: Sequence[str],
    ceiling: int,
    estimator: TokenEstimatorConfig | None,
) -> tuple[int, ...]:
    """Greedily keep the most query-relevant segments that fit ``ceiling``.

    Segments are added in relevance order and then re-sorted by their original
    position, so the rendered text always reads in its stored order. The
    single most relevant segment is kept even when it alone exceeds the ceiling:
    dropping everything would be worse than exceeding a *soft* target, and
    whether the resulting block is affordable is the budget's decision, not the
    compressor's.
    """
    terms = frozenset(query_terms)
    ranked = sorted(
        range(len(segments)),
        key=lambda index: _segment_key(segments[index], index, terms),
    )
    kept: list[int] = []
    used = 0
    for index in ranked:
        cost = estimate_tokens(segments[index], config=estimator).estimated_tokens
        if used + cost <= ceiling:
            kept.append(index)
            used += cost
    if not kept and ranked:
        kept.append(ranked[0])
    return tuple(sorted(kept))


# -- candidates ---------------------------------------------------------------


def factor_value(result: RetrievalResult, name: str) -> float:
    """Value of a named M3 ranking factor (0.0 when absent)."""
    for factor in result.components:
        if factor.name == name:
            return factor.value
    return 0.0


def cost_penalty(tokens: int, policy: ContextPolicy) -> float:
    """``min(1, tokens / cost_reference_tokens)`` -- bounded in ``[0, 1]``."""
    if tokens <= 0:
        return 0.0
    return round(min(1.0, tokens / policy.cost_reference_tokens), _SCORE_PRECISION)


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """One retrieval result scored for context inclusion."""

    memory: Memory
    #: M3 ``final_score``: relevance ranking, not a context decision.
    retrieval_score: float
    relevance: float
    effective_priority: float
    confidence: float
    importance: float
    freshness: float
    type_preference: float
    #: Convex combination of the five signals above.
    value_score: float
    #: Bounded size penalty derived from the estimated token cost.
    cost_penalty: float
    #: ``value * (1 - cost_weight * cost_penalty)`` -- the selection signal.
    utility_score: float
    #: Estimated cost of the *rendered* block (content plus framing).
    estimated_tokens: int
    factors: tuple[RankingFactor, ...]
    section: ContextSection
    compression: CompressionResult
    duplicate_of: str | None
    superseded_by: str | None
    conflicts: tuple[ConflictInfo, ...]

    @property
    def memory_id(self) -> str:
        """Shorthand for ``candidate.memory.id``."""
        return self.memory.id

    @property
    def rendered_content(self) -> str:
        """Content after deterministic compression."""
        return self.compression.compressed_text

    def selection_key(self) -> tuple[float, str]:
        """Deterministic admission order: utility descending, id ascending."""
        return (-self.utility_score, self.memory.id)


def score_candidate(
    result: RetrievalResult,
    *,
    policy: ContextPolicy | None = None,
    estimator: TokenEstimatorConfig | None = None,
    query_terms: Sequence[str] = (),
    enabled: bool | None = None,
    max_block_tokens: int | None = None,
) -> ContextCandidate:
    """Turn one M3 result into a scored context candidate.

    Pure: the same result, policy, estimator and query always produce the same
    candidate, which is what makes the whole compiler reproducible.
    """
    resolved = policy or ContextPolicy()
    memory = result.memory
    relevance = result.relevance
    effective_priority = factor_value(result, "effective_priority")
    freshness = factor_value(result, "freshness")
    type_preference = factor_value(result, "type_preference")

    compression = compress_content(
        memory.content,
        query_terms=query_terms,
        policy=resolved,
        estimator=estimator,
        enabled=enabled,
        max_block_tokens=max_block_tokens,
    )
    rendered_tokens = estimate_tokens(
        render_block_line(compression.compressed_text), config=estimator
    ).estimated_tokens

    value = round(
        resolved.relevance_weight * relevance
        + resolved.priority_weight * effective_priority
        + resolved.confidence_weight * memory.confidence
        + resolved.freshness_weight * freshness
        + resolved.type_weight * type_preference,
        _SCORE_PRECISION,
    )
    penalty = cost_penalty(rendered_tokens, resolved)
    utility = round(
        value * (1.0 - resolved.cost_weight * penalty),
        _SCORE_PRECISION,
    )
    factors = (
        RankingFactor(
            name="relevance",
            weight=resolved.relevance_weight,
            value=round(relevance, _SCORE_PRECISION),
            contribution=round(resolved.relevance_weight * relevance, _SCORE_PRECISION),
            explanation=f"M3 relevance {relevance:.4f} for the requested query",
        ),
        RankingFactor(
            name="effective_priority",
            weight=resolved.priority_weight,
            value=round(effective_priority, _SCORE_PRECISION),
            contribution=round(resolved.priority_weight * effective_priority, _SCORE_PRECISION),
            explanation=f"M2 effective priority {effective_priority:.4f} (decayed, never stored)",
        ),
        RankingFactor(
            name="confidence",
            weight=resolved.confidence_weight,
            value=round(memory.confidence, _SCORE_PRECISION),
            contribution=round(resolved.confidence_weight * memory.confidence, _SCORE_PRECISION),
            explanation=f"stored confidence {memory.confidence:.2f}",
        ),
        RankingFactor(
            name="freshness",
            weight=resolved.freshness_weight,
            value=round(freshness, _SCORE_PRECISION),
            contribution=round(resolved.freshness_weight * freshness, _SCORE_PRECISION),
            explanation=f"M2 decay factor {freshness:.4f}",
        ),
        RankingFactor(
            name="type_preference",
            weight=resolved.type_weight,
            value=round(type_preference, _SCORE_PRECISION),
            contribution=round(resolved.type_weight * type_preference, _SCORE_PRECISION),
            explanation=(
                f"{memory.memory_type.value} has type preference {type_preference:.2f} "
                f"and belongs to the {resolved.section_for(memory.memory_type).value} section"
            ),
        ),
        RankingFactor(
            name="cost_penalty",
            weight=resolved.cost_weight,
            value=penalty,
            contribution=round(-resolved.cost_weight * penalty, _SCORE_PRECISION),
            explanation=(
                f"{rendered_tokens} estimated tokens vs a reference of "
                f"{resolved.cost_reference_tokens} tokens"
            ),
        ),
    )
    return ContextCandidate(
        memory=memory,
        retrieval_score=result.final_score,
        relevance=round(relevance, _SCORE_PRECISION),
        effective_priority=round(effective_priority, _SCORE_PRECISION),
        confidence=round(memory.confidence, _SCORE_PRECISION),
        importance=round(memory.importance, _SCORE_PRECISION),
        freshness=round(freshness, _SCORE_PRECISION),
        type_preference=round(type_preference, _SCORE_PRECISION),
        value_score=value,
        cost_penalty=penalty,
        utility_score=utility,
        estimated_tokens=rendered_tokens,
        factors=factors,
        section=resolved.section_for(memory.memory_type),
        compression=compression,
        duplicate_of=result.duplicate_of,
        superseded_by=result.superseded_by,
        conflicts=result.conflicts,
    )


# -- selection ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """The explained outcome of one candidate.

    ``remaining_tokens`` is the budget still available when the decision was
    taken. Candidates rejected *before* budget allocation (allow/deny lists, type
    preference, score floor, redundancy) carry the full budget, because nothing
    had been allocated yet.
    """

    memory_id: str
    memory_type: MemoryType
    reason: SelectionReason
    detail: str
    utility_score: float
    value_score: float
    estimated_tokens: int
    remaining_tokens: int
    selected: bool
    related_memory_id: str | None = None

    def selection_key(self) -> tuple[float, str]:
        """Trace order: utility descending, then memory id ascending."""
        return (-self.utility_score, self.memory_id)


def _decision(
    candidate: ContextCandidate,
    *,
    reason: SelectionReason,
    detail: str,
    remaining: int,
    related: str | None = None,
) -> ContextSelection:
    """Build one trace entry (the single place trace entries are shaped)."""
    return ContextSelection(
        memory_id=candidate.memory_id,
        memory_type=candidate.memory.memory_type,
        reason=reason,
        detail=detail,
        utility_score=candidate.utility_score,
        value_score=candidate.value_score,
        estimated_tokens=candidate.estimated_tokens,
        remaining_tokens=remaining,
        selected=reason.is_admission,
        related_memory_id=related,
    )


def apply_admission_filters(
    candidates: Sequence[ContextCandidate],
    *,
    request: ContextRequest,
    policy: ContextPolicy,
    token_budget: int,
) -> tuple[list[ContextCandidate], list[ContextSelection]]:
    """Apply the request's allow/deny lists, type preference and score floor.

    ``min_candidate_score`` falls back to the policy when the request does not
    override it, and ``include_ids`` is a hard allow list: anything else is
    reported as rejected rather than silently dropped.
    """
    floor = (
        policy.min_candidate_score
        if request.min_candidate_score is None
        else request.min_candidate_score
    )
    kept: list[ContextCandidate] = []
    rejected: list[ContextSelection] = []
    for candidate in candidates:
        memory_id = candidate.memory_id
        if memory_id in request.exclude_ids:
            rejected.append(
                _decision(
                    candidate,
                    reason=SelectionReason.EXCLUDED_BY_ID,
                    detail=SelectionReason.EXCLUDED_BY_ID.description,
                    remaining=token_budget,
                )
            )
        elif not request.admits_id(memory_id):
            rejected.append(
                _decision(
                    candidate,
                    reason=SelectionReason.NOT_IN_INCLUDE_IDS,
                    detail=SelectionReason.NOT_IN_INCLUDE_IDS.description,
                    remaining=token_budget,
                )
            )
        elif not request.prefers(candidate.memory.memory_type):
            rejected.append(
                _decision(
                    candidate,
                    reason=SelectionReason.MEMORY_TYPE_FILTERED,
                    detail=(
                        f"{candidate.memory.memory_type.value} is not among the requested "
                        f"memory types"
                    ),
                    remaining=token_budget,
                )
            )
        elif candidate.utility_score < floor:
            rejected.append(
                _decision(
                    candidate,
                    reason=SelectionReason.BELOW_MINIMUM_SCORE,
                    detail=(
                        f"utility {candidate.utility_score:.4f} is below the requested minimum "
                        f"{floor:.4f}"
                    ),
                    remaining=token_budget,
                )
            )
        else:
            kept.append(candidate)
    return kept, rejected


def suppress_redundant(
    candidates: Sequence[ContextCandidate],
    *,
    token_budget: int,
) -> tuple[list[ContextCandidate], list[ContextSelection]]:
    """Suppress candidates that a better candidate already represents.

    Two M2 relationships are honoured, and **no memory is ever deleted**:

    * ``CONFLICTS_WITH`` + ``SUPERSEDES`` -- the loser of a resolution is
      suppressed whenever M2 recorded that it was superseded. It stays in storage
      and remains reachable through the API; it simply must not reach the prompt
      as if it were still true.
    * ``DUPLICATE_OF`` -- of a group of near-duplicates present in the candidate
      set, the highest utility (ties by id) is kept as the representative and
      every other member is suppressed, including duplicates *of a duplicate*
      (the group is resolved transitively).

    A recorded duplicate whose partner is *not* in the candidate set is kept:
    the compiler only suppresses redundancy it can actually see.
    """
    kept: list[ContextCandidate] = []
    suppressed: list[ContextSelection] = []
    #: Every candidate id -> the surviving representative of its duplicate group.
    representative_of: dict[str, str] = {}

    for candidate in sorted(candidates, key=ContextCandidate.selection_key):
        memory_id = candidate.memory_id
        if candidate.superseded_by is not None:
            suppressed.append(
                _decision(
                    candidate,
                    reason=SelectionReason.CONFLICT_SUPERSEDED,
                    detail=(
                        f"M2 resolution superseded this memory with "
                        f"{candidate.superseded_by}; it stays stored but is not authoritative"
                    ),
                    remaining=token_budget,
                    related=candidate.superseded_by,
                )
            )
            continue
        representative_id = representative_of.get(candidate.duplicate_of or "")
        if representative_id is not None:
            suppressed.append(
                _decision(
                    candidate,
                    reason=SelectionReason.DUPLICATE_OF,
                    detail=(
                        f"near-duplicate of {representative_id}, which represents this "
                        f"duplicate group in the context"
                    ),
                    remaining=token_budget,
                    related=representative_id,
                )
            )
            representative_of[memory_id] = representative_id
            continue
        kept.append(candidate)
        representative_of[memory_id] = memory_id
    return kept, suppressed


def select_within_budget(
    candidates: Sequence[ContextCandidate],
    *,
    token_budget: int,
) -> tuple[list[ContextCandidate], list[ContextSelection]]:
    """Greedily admit candidates while they fit the remaining budget.

    The documented algorithm (see the module docstring): order by
    ``(-utility, memory_id)``, admit while the candidate fits, otherwise record
    ``insufficient_budget`` and **continue** -- which is what lets a smaller,
    lower-utility memory use the leftover budget instead of wasting it.

    A candidate that does not fit on its own turn can never fit later, because
    the remaining budget only shrinks; skipping it and continuing is therefore
    both optimal for this greedy pass and cheap.
    """
    if token_budget < 1:
        raise ContextValidationError(
            f"token_budget must be >= 1 estimated token (got {token_budget})"
        )
    selected: list[ContextCandidate] = []
    trace: list[ContextSelection] = []
    remaining = token_budget
    for candidate in sorted(candidates, key=ContextCandidate.selection_key):
        cost = candidate.estimated_tokens
        if cost <= remaining:
            selected.append(candidate)
            remaining -= cost
            trace.append(
                _decision(
                    candidate,
                    reason=SelectionReason.SELECTED,
                    detail=(
                        f"highest utility ({candidate.utility_score:.4f}) that fits the "
                        f"remaining budget of {remaining + cost} tokens"
                    ),
                    remaining=remaining,
                )
            )
        else:
            trace.append(
                _decision(
                    candidate,
                    reason=SelectionReason.INSUFFICIENT_BUDGET,
                    detail=(
                        f"estimated cost {cost} tokens exceeds the remaining budget of "
                        f"{remaining} tokens"
                    ),
                    remaining=remaining,
                )
            )
    return selected, trace


# -- blocks, rendering and results -------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One memory, its rendered content and its full provenance."""

    memory_id: str
    memory_type: MemoryType
    status: MemoryStatus
    source: str | None
    confidence: float
    importance: float
    #: Original, unmodified memory content.
    content: str
    #: Content after deterministic compression (what the agent reads).
    rendered_content: str
    section: ContextSection
    section_rank: int
    #: Position in the compiled context (0 is first).
    order: int
    relevance: float
    #: M3 ``final_score``.
    retrieval_score: float
    #: M4 selection signal.
    utility_score: float
    estimated_tokens: int
    compression: CompressionResult
    conflicts: tuple[ConflictInfo, ...]


def build_block(
    candidate: ContextCandidate,
    *,
    order: int,
    policy: ContextPolicy,
) -> ContextBlock:
    """Wrap a selected candidate into a context block with provenance."""
    memory = candidate.memory
    return ContextBlock(
        memory_id=memory.id,
        memory_type=memory.memory_type,
        status=memory.status,
        source=memory.source,
        confidence=candidate.confidence,
        importance=candidate.importance,
        content=candidate.compression.original_text,
        rendered_content=candidate.compression.compressed_text,
        section=candidate.section,
        section_rank=policy.section_rank(candidate.section),
        order=order,
        relevance=candidate.relevance,
        retrieval_score=candidate.retrieval_score,
        utility_score=candidate.utility_score,
        estimated_tokens=candidate.estimated_tokens,
        compression=candidate.compression,
        conflicts=candidate.conflicts,
    )


def block_sort_key(block: ContextBlock) -> tuple[int, float, str]:
    """Deterministic prompt order: section, then utility, then id."""
    return (block.section_rank, -block.utility_score, block.memory_id)


def order_blocks(
    selected: Sequence[ContextCandidate], *, policy: ContextPolicy
) -> tuple[ContextBlock, ...]:
    """Order selected candidates into blocks and renumber them.

    The prompt order is ``(section_rank, -utility, memory_id)``: a total order,
    so two compilations of the same candidate set can never differ.
    """
    blocks = [build_block(candidate, order=0, policy=policy) for candidate in selected]
    blocks.sort(key=block_sort_key)
    return tuple(replace(block, order=index) for index, block in enumerate(blocks))


def render_block_line(rendered_content: str) -> str:
    """One line of the compiled context."""
    return f"{BLOCK_PREFIX}{rendered_content}"


def render_section(title: str, lines: Sequence[str]) -> str:
    """One titled section of the compiled context."""
    return f"{title}\n{'-' * len(title)}\n" + "\n".join(lines)


def render_context(blocks: Sequence[ContextBlock], *, policy: ContextPolicy) -> str:
    """Render the whole context: sections, headings and one line per memory.

    Empty sections are omitted entirely, so the rendered text is exactly what
    the token budget is enforced against.
    """
    chunks: list[str] = []
    for section in policy.section_order:
        lines = [
            render_block_line(block.rendered_content)
            for block in blocks
            if block.section is section
        ]
        if lines:
            chunks.append(render_section(section.heading, lines))
    return "\n\n".join(chunks)


@dataclass(frozen=True, slots=True)
class ContextExplanation:
    """Everything needed to audit *how* the context was built."""

    estimator: str
    estimator_formula: str
    utility_formula: str
    utility_weights: Mapping[str, float]
    ordering_policy: str
    section_order: tuple[ContextSection, ...]
    selection_counts: Mapping[str, int]
    compressed_blocks: int
    tokens_saved: int
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContextResult:
    """The compiled context: blocks, budget accounting and the full trace."""

    query: str
    token_budget: int
    estimated_tokens: int
    remaining_tokens: int
    candidates_considered: int
    selected_count: int
    suppressed_count: int
    requested_retrieval_mode: RetrievalMode
    retrieval_mode: RetrievalMode
    semantic_available: bool
    degraded_reason: str | None
    context_text: str
    blocks: tuple[ContextBlock, ...]
    selection_trace: tuple[ContextSelection, ...]
    suppressed_candidates: tuple[ContextSelection, ...]
    explanation: ContextExplanation

    @property
    def within_budget(self) -> bool:
        """The hard guarantee: estimated tokens never exceed the budget."""
        return self.estimated_tokens <= self.token_budget

    @property
    def utilization(self) -> float:
        """Share of the budget the compiled context occupies."""
        if self.token_budget <= 0:  # pragma: no cover - rejected at construction
            return 0.0
        return round(self.estimated_tokens / self.token_budget, _SCORE_PRECISION)


#: The fixed, honest caveats reported with every compiled context.
CONTEXT_NOTES: tuple[str, ...] = (
    "token counts are MemForge estimates from a documented character/word formula, "
    "not provider tokenisations and not billing data",
    "selection is a deterministic, configurable heuristic -- it is not a learned "
    "ranking and makes no claim about optimality",
    "context is compiled with no LLM: content is never paraphrased or summarised",
    "compression never rewrites memory content; it only normalises whitespace and "
    "may drop whole sentences, which does not guarantee semantic equivalence",
)


__all__ = [
    "BLOCK_PREFIX",
    "CONTEXT_NOTES",
    "DEFAULT_SECTION_ORDER",
    "DEFAULT_TYPE_SECTIONS",
    "CompressionResult",
    "CompressionStep",
    "ContextBlock",
    "ContextCandidate",
    "ContextExplanation",
    "ContextPolicy",
    "ContextRequest",
    "ContextResult",
    "ContextSection",
    "ContextSelection",
    "SelectionReason",
    "apply_admission_filters",
    "block_sort_key",
    "build_block",
    "compress_content",
    "cost_penalty",
    "factor_value",
    "order_blocks",
    "render_block_line",
    "render_context",
    "render_section",
    "score_candidate",
    "select_within_budget",
    "split_segments",
    "suppress_redundant",
]
