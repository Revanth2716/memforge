"""Pydantic schemas for the MemForge REST API.

API models are deliberately separate from the domain entities: the API is a
public contract (strict, ``extra="forbid"``, documented), while the domain is a
pure in-process model.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.agent import AgentRequest, AgentResult, CandidateMemory, ContextBlockSummary
from app.domain.agent import OperationRecord as AgentOperationRecord
from app.domain.context import (
    CompressionResult,
    ContextBlock,
    ContextExplanation,
    ContextRequest,
    ContextResult,
    ContextSection,
    ContextSelection,
)
from app.domain.evaluation import METRIC_DEFINITIONS, EvaluationSummary
from app.domain.intelligence import FactorScore
from app.domain.llm import LLMUsage
from app.domain.models import Memory, MemoryStatus, MemoryType, MemoryVersion
from app.domain.observability import (
    OperationMetrics,
    Trace,
    TraceEvent,
    TraceSummary,
)
from app.domain.retrieval import (
    RankingFactor,
    RetrievalFilters,
    RetrievalMode,
    RetrievalQuery,
)
from app.domain.updates import UNSET, MemoryUpdate

#: Fields of an update that may never be sent as JSON ``null``.
_NON_NULLABLE_UPDATE_FIELDS = ("content", "memory_type", "confidence", "importance", "status")


class ApiModel(BaseModel):
    """Base model: strict about unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MemoryCreateRequest(ApiModel):
    """Payload for ``POST /api/v1/memories``."""

    content: str = Field(min_length=1, max_length=4000, description="The memory text.")
    memory_type: MemoryType = Field(description="Category of the memory.")
    source: str | None = Field(
        default=None, max_length=200, description="Optional provenance, e.g. 'user:revanth'."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Accuracy score in [0, 1].")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance score in [0, 1].")


class MemoryUpdateRequest(ApiModel):
    """Payload for ``PATCH /api/v1/memories/{id}``.

    Omitted fields keep their stored value. ``source: null`` clears the
    provenance; every other field rejects an explicit ``null``.
    """

    content: str | None = Field(default=None, min_length=1, max_length=4000)
    memory_type: MemoryType | None = None
    source: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    status: MemoryStatus | None = None

    @model_validator(mode="after")
    def _validate_provided_fields(self) -> MemoryUpdateRequest:
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")
        for name in _NON_NULLABLE_UPDATE_FIELDS:
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"Field '{name}' cannot be null")
        return self

    def to_domain_update(self) -> MemoryUpdate:
        """Translate the request into a domain :class:`MemoryUpdate`."""
        provided = self.model_fields_set
        return MemoryUpdate(
            content=self.content if "content" in provided else UNSET,
            memory_type=self.memory_type if "memory_type" in provided else UNSET,
            source=self.source if "source" in provided else UNSET,
            confidence=self.confidence if "confidence" in provided else UNSET,
            importance=self.importance if "importance" in provided else UNSET,
            status=self.status if "status" in provided else UNSET,
        )


class MemoryResponse(BaseModel):
    """A memory as returned by the API."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    memory_type: MemoryType
    source: str | None = None
    confidence: float
    importance: float
    status: MemoryStatus
    created_at: datetime
    updated_at: datetime
    version: int

    @classmethod
    def from_domain(cls, memory: Memory) -> MemoryResponse:
        return cls(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type,
            source=memory.source,
            confidence=memory.confidence,
            importance=memory.importance,
            status=memory.status,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            version=memory.version,
        )


class MemoryListResponse(BaseModel):
    """A page of memories plus pagination metadata."""

    items: list[MemoryResponse]
    total: int
    limit: int
    offset: int


class MemoryVersionResponse(BaseModel):
    """One immutable snapshot in a memory's history."""

    memory_id: str
    version: int
    change: str
    content: str
    memory_type: MemoryType
    source: str | None = None
    confidence: float
    importance: float
    status: MemoryStatus
    recorded_at: datetime

    @classmethod
    def from_domain(cls, version: MemoryVersion) -> MemoryVersionResponse:
        return cls(
            memory_id=version.memory_id,
            version=version.version,
            change=version.change.value,
            content=version.content,
            memory_type=version.memory_type,
            source=version.source,
            confidence=version.confidence,
            importance=version.importance,
            status=version.status,
            recorded_at=version.recorded_at,
        )


class MemoryHistoryResponse(BaseModel):
    """Full history of a memory, oldest version first."""

    memory_id: str
    current_version: int
    total: int
    items: list[MemoryVersionResponse]


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    service: str
    version: str
    storage: str


# -- M3: retrieval -----------------------------------------------------------


class FactorResponse(BaseModel):
    """One factor of a retrieval ranking score."""

    name: str
    weight: float
    value: float
    contribution: float
    explanation: str

    @classmethod
    def from_domain(cls, factor: RankingFactor) -> FactorResponse:
        return cls(
            name=factor.name,
            weight=factor.weight,
            value=factor.value,
            contribution=factor.contribution,
            explanation=factor.explanation,
        )


# -- M2: intelligence --------------------------------------------------------


class M2FactorResponse(BaseModel):
    """One factor of the computed importance score."""

    name: str
    weight: float
    raw_value: float
    contribution: float
    explanation: str

    @classmethod
    def from_domain(cls, factor: FactorScore) -> M2FactorResponse:
        return cls(
            name=factor.name,
            weight=factor.weight,
            raw_value=factor.raw_value,
            contribution=factor.contribution,
            explanation=factor.explanation,
        )


class IntelligenceResponse(BaseModel):
    """Deterministic intelligence for one memory.

    ``confidence`` (accuracy) and ``stored_importance`` (caller input) are
    stored fields; ``importance_score``, ``effective_priority`` and
    ``decay_multiplier`` are *computed* on request and never stored.
    """

    memory_id: str
    memory_type: MemoryType
    status: MemoryStatus
    version: int
    confidence: float
    stored_importance: float
    importance_score: float
    importance_factors: list[M2FactorResponse]
    importance_explanation: str
    effective_priority: float
    base_priority: float
    decay_multiplier: float
    age_days: float
    priority_explanation: str
    access_count: int
    last_accessed_at: datetime | None
    decay_half_life_days: float
    similarity_threshold: float
    computed_at: datetime


class DuplicateMatchResponse(BaseModel):
    """A near-duplicate of the requested memory."""

    memory_id: str
    similarity: float
    content: str
    memory_type: MemoryType
    shared_tokens: list[str]
    explanation: str


class DuplicatesResponse(BaseModel):
    """Near-duplicates of a memory, best match first."""

    memory_id: str
    threshold: float
    total: int
    items: list[DuplicateMatchResponse]


class ConflictCandidateResponse(BaseModel):
    """A memory that disagrees with the requested memory."""

    target_memory_id: str
    attribute: str
    source_value: str
    target_value: str
    value_similarity: float
    source_content: str
    target_content: str
    explanation: str


class ConflictsResponse(BaseModel):
    """Unresolved conflicts of a memory."""

    memory_id: str
    total: int
    items: list[ConflictCandidateResponse]


class ResolutionStepResponse(BaseModel):
    """One comparison made by the resolution policy."""

    factor: str
    left_value: str
    right_value: str
    winner_id: str


class ConflictResolutionResponse(BaseModel):
    """Result of resolving one conflict deterministically."""

    winner_memory_id: str
    loser_memory_id: str
    decided_by: str
    explanation: str
    steps: list[ResolutionStepResponse]
    loser_archived: bool
    conflict: ConflictCandidateResponse


class RelationshipResponse(BaseModel):
    """A stored relationship between two memories."""

    source_id: str
    target_id: str
    relationship: str
    similarity: float | None
    created_at: datetime


class RelationshipsResponse(BaseModel):
    """Every relationship recorded for a memory."""

    memory_id: str
    total: int
    items: list[RelationshipResponse]


# -- M3: retrieval request/response -----------------------------------------


class RetrievalMode_(enum.StrEnum):
    """Requested retrieval mode (API mirror of the domain enum)."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class MemorySearchRequest(ApiModel):
    """Body for ``POST /api/v1/memories/search``."""

    query: str = Field(min_length=1, max_length=1000, description="Natural language query.")
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum memories to return.")
    retrieval_mode: RetrievalMode_ = Field(default=RetrievalMode_.LEXICAL)
    semantic_weight: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Semantic blend weight for hybrid mode."
    )
    memory_type: MemoryType | None = None
    status: MemoryStatus | None = MemoryStatus.ACTIVE
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    created_after: datetime | None = None
    created_before: datetime | None = None

    @model_validator(mode="after")
    def _validate_date_window(self) -> MemorySearchRequest:
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_before < self.created_after
        ):
            raise ValueError("created_before must not be earlier than created_after")
        return self

    def to_domain_query(self) -> RetrievalQuery:
        """Translate the request into a domain retrieval query."""
        return RetrievalQuery(
            text=self.query,
            top_k=self.top_k,
            mode=RetrievalMode(self.retrieval_mode.value),
            filters=RetrievalFilters(
                memory_type=self.memory_type,
                status=self.status,
                min_confidence=self.min_confidence,
                min_importance=self.min_importance,
                created_after=self.created_after,
                created_before=self.created_before,
            ),
            semantic_weight=self.semantic_weight,
        )


class ConflictInfoResponse(BaseModel):
    """How a retrieved memory relates to a conflict it takes part in."""

    target_memory_id: str
    relationship: str
    attribute: str
    source_value: str
    target_value: str
    resolution: str
    resolution_reason: str | None


class RetrievalResultResponse(BaseModel):
    """One ranked memory, fully explained."""

    memory_id: str
    content: str
    memory_type: MemoryType
    status: MemoryStatus
    confidence: float
    importance: float
    effective_priority: float
    freshness: float
    lexical_score: float
    semantic_score: float | None
    relevance: float
    final_score: float
    factors: list[FactorResponse]
    explanation: str
    duplicate_of: str | None
    superseded_by: str | None
    conflicts: list[ConflictInfoResponse]


class SuppressedCandidateResponse(BaseModel):
    """A candidate deliberately left out of the top-k."""

    memory_id: str
    final_score: float
    reason: str


class MemorySearchResponse(BaseModel):
    """Search response: ranked results plus query-level provenance."""

    query: str
    top_k: int
    requested_mode: str
    retrieval_mode: str
    semantic_available: bool
    semantic_provider: str | None
    semantic_model: str | None
    degraded_reason: str | None
    candidates_considered: int
    candidates_returned: int
    candidates_suppressed: int
    latency_ms: float
    results: list[RetrievalResultResponse]
    suppressed: list[SuppressedCandidateResponse]


# -- M4: context compilation --------------------------------------------------


class ContextCompileRequest(ApiModel):
    """Body for ``POST /api/v1/context/compile``.

    Only ``query`` is required: the token budget falls back to
    ``MEMFORGE_DEFAULT_CONTEXT_TOKEN_BUDGET``. ``token_budget`` is a hard upper
    bound in **estimated** tokens (MemForge's documented estimator -- never a
    provider tokenisation).
    """

    query: str = Field(
        min_length=1, max_length=1000, description="Agent query or task the context is for."
    )
    token_budget: int | None = Field(
        default=None,
        ge=1,
        description="Hard budget in estimated tokens. Defaults to the configured default.",
    )
    top_k: int = Field(default=20, ge=1, le=100, description="Maximum retrieval candidates.")
    retrieval_mode: RetrievalMode_ = Field(default=RetrievalMode_.LEXICAL)
    semantic_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    preferred_memory_types: list[MemoryType] | None = Field(
        default=None, description="Optional memory-type preference (allow list)."
    )
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    include_ids: list[str] = Field(default_factory=list, description="Allow list of memory ids.")
    exclude_ids: list[str] = Field(default_factory=list, description="Deny list of memory ids.")
    max_block_tokens: int | None = Field(
        default=None, ge=1, description="Soft per-block ceiling for sentence selection."
    )
    min_candidate_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Utility floor for a candidate to be selected."
    )
    compress: bool | None = Field(
        default=None, description="Set false to keep the stored content verbatim."
    )
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)

    @model_validator(mode="after")
    def _validate_id_lists(self) -> ContextCompileRequest:
        overlap = set(self.include_ids) & set(self.exclude_ids)
        if overlap:
            raise ValueError(
                f"include_ids and exclude_ids must not overlap (got {sorted(overlap)})"
            )
        if self.preferred_memory_types is not None and not self.preferred_memory_types:
            raise ValueError("preferred_memory_types must be omitted or contain at least one type")
        return self

    def to_domain_request(self, *, default_token_budget: int) -> ContextRequest:
        """Translate the request into a domain :class:`ContextRequest`."""
        preferred = self.preferred_memory_types
        return ContextRequest(
            query=self.query,
            token_budget=(default_token_budget if self.token_budget is None else self.token_budget),
            top_k=self.top_k,
            retrieval_mode=RetrievalMode(self.retrieval_mode.value),
            semantic_weight=self.semantic_weight,
            preferred_memory_types=None if preferred is None else tuple(preferred),
            min_confidence=self.min_confidence,
            min_importance=self.min_importance,
            include_ids=tuple(self.include_ids),
            exclude_ids=tuple(self.exclude_ids),
            max_block_tokens=self.max_block_tokens,
            min_candidate_score=self.min_candidate_score,
            compress=self.compress,
            status=self.status,
        )


class CompressionResponse(BaseModel):
    """The traceable record of what compression did to one memory."""

    original_text: str
    compressed_text: str
    applied: list[str]
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    truncated: bool
    retained_segments: list[int]


class ContextBlockResponse(BaseModel):
    """One compiled block, with full provenance back to its memory."""

    memory_id: str
    memory_type: MemoryType
    status: MemoryStatus
    source: str | None
    confidence: float
    importance: float
    content: str
    rendered_content: str
    section: ContextSection
    section_rank: int
    order: int
    relevance: float
    retrieval_score: float
    utility_score: float
    estimated_tokens: int
    compression: CompressionResponse
    conflicts: list[ConflictInfoResponse]

    @classmethod
    def from_domain(cls, block: ContextBlock) -> ContextBlockResponse:
        compression: CompressionResult = block.compression
        return cls(
            memory_id=block.memory_id,
            memory_type=block.memory_type,
            status=block.status,
            source=block.source,
            confidence=block.confidence,
            importance=block.importance,
            content=block.content,
            rendered_content=block.rendered_content,
            section=block.section,
            section_rank=block.section_rank,
            order=block.order,
            relevance=block.relevance,
            retrieval_score=block.retrieval_score,
            utility_score=block.utility_score,
            estimated_tokens=block.estimated_tokens,
            compression=CompressionResponse(
                original_text=compression.original_text,
                compressed_text=compression.compressed_text,
                applied=[step.value for step in compression.applied],
                original_tokens=compression.original_tokens,
                compressed_tokens=compression.compressed_tokens,
                tokens_saved=compression.tokens_saved,
                truncated=compression.truncated,
                retained_segments=list(compression.retained_segments),
            ),
            conflicts=[
                ConflictInfoResponse(
                    target_memory_id=conflict.target_memory_id,
                    relationship=conflict.relationship,
                    attribute=conflict.attribute,
                    source_value=conflict.source_value,
                    target_value=conflict.target_value,
                    resolution=conflict.resolution,
                    resolution_reason=conflict.resolution_reason,
                )
                for conflict in block.conflicts
            ],
        )


class ContextSelectionResponse(BaseModel):
    """The explained decision taken for one candidate."""

    memory_id: str
    memory_type: MemoryType
    selected: bool
    reason: str
    detail: str
    utility_score: float
    value_score: float
    estimated_tokens: int
    remaining_tokens: int
    related_memory_id: str | None = None

    @classmethod
    def from_domain(cls, selection: ContextSelection) -> ContextSelectionResponse:
        return cls(
            memory_id=selection.memory_id,
            memory_type=selection.memory_type,
            selected=selection.selected,
            reason=selection.reason.value,
            detail=selection.detail,
            utility_score=selection.utility_score,
            value_score=selection.value_score,
            estimated_tokens=selection.estimated_tokens,
            remaining_tokens=selection.remaining_tokens,
            related_memory_id=selection.related_memory_id,
        )


class ContextExplanationResponse(BaseModel):
    """How the context was built: estimator, ordering, counts and caveats."""

    estimator: str
    estimator_formula: str
    utility_formula: str
    utility_weights: dict[str, float]
    ordering_policy: str
    section_order: list[ContextSection]
    selection_counts: dict[str, int]
    compressed_blocks: int
    tokens_saved: int
    notes: list[str]

    @classmethod
    def from_domain(cls, explanation: ContextExplanation) -> ContextExplanationResponse:
        return cls(
            estimator=explanation.estimator,
            estimator_formula=explanation.estimator_formula,
            utility_formula=explanation.utility_formula,
            utility_weights=dict(explanation.utility_weights),
            ordering_policy=explanation.ordering_policy,
            section_order=list(explanation.section_order),
            selection_counts=dict(explanation.selection_counts),
            compressed_blocks=explanation.compressed_blocks,
            tokens_saved=explanation.tokens_saved,
            notes=list(explanation.notes),
        )


class ContextCompileResponse(BaseModel):
    """The compiled context: budget accounting, blocks, trace and explanation."""

    query: str
    token_budget: int
    estimated_tokens: int
    remaining_tokens: int
    within_budget: bool
    utilization: float
    candidates_considered: int
    selected_count: int
    suppressed_count: int
    requested_retrieval_mode: str
    retrieval_mode: str
    semantic_available: bool
    degraded_reason: str | None
    context_text: str
    blocks: list[ContextBlockResponse]
    selection_trace: list[ContextSelectionResponse]
    suppressed_candidates: list[ContextSelectionResponse]
    explanation: ContextExplanationResponse

    @classmethod
    def from_domain(cls, result: ContextResult) -> ContextCompileResponse:
        return cls(
            query=result.query,
            token_budget=result.token_budget,
            estimated_tokens=result.estimated_tokens,
            remaining_tokens=result.remaining_tokens,
            within_budget=result.within_budget,
            utilization=result.utilization,
            candidates_considered=result.candidates_considered,
            selected_count=result.selected_count,
            suppressed_count=result.suppressed_count,
            requested_retrieval_mode=result.requested_retrieval_mode.value,
            retrieval_mode=result.retrieval_mode.value,
            semantic_available=result.semantic_available,
            degraded_reason=result.degraded_reason,
            context_text=result.context_text,
            blocks=[ContextBlockResponse.from_domain(block) for block in result.blocks],
            selection_trace=[
                ContextSelectionResponse.from_domain(entry) for entry in result.selection_trace
            ],
            suppressed_candidates=[
                ContextSelectionResponse.from_domain(entry)
                for entry in result.suppressed_candidates
            ],
            explanation=ContextExplanationResponse.from_domain(result.explanation),
        )


# -- M5: agent + memory operations ------------------------------------------


class LLMUsageResponse(BaseModel):
    """Usage **as reported by the provider**.

    The whole object is ``null`` when the provider reported none; MemForge never
    derives or defaults these numbers.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reported_by: str = "provider"

    @classmethod
    def from_domain(cls, usage: LLMUsage | None) -> LLMUsageResponse | None:
        return (
            None
            if usage is None
            else cls(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                reported_by=usage.reported_by,
            )
        )


class AgentRespondRequest(ApiModel):
    """Body for ``POST /api/v1/agent/respond``."""

    request: str = Field(min_length=1, max_length=4000, description="The user's request.")
    token_budget: int = Field(
        default=800, ge=1, description="Hard budget in estimated tokens for the context (M4)."
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Recall / retrieval candidates.")
    retrieval_mode: RetrievalMode_ = Field(default=RetrievalMode_.LEXICAL)
    semantic_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    use_memory: bool = Field(
        default=True, description="Set false to answer without consulting memory at all."
    )
    context_only: bool = Field(
        default=False, description="Compile the context but never call the provider."
    )
    extract_candidates: bool = Field(
        default=False, description="Ask the provider to propose candidate memories."
    )
    store_candidates: bool = Field(
        default=False,
        description="Persist accepted candidates. Opt-in: nothing is written otherwise.",
    )
    source: str = Field(default="agent:api", max_length=200)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)

    def to_domain_request(self) -> AgentRequest:
        """Translate the request into a domain :class:`AgentRequest`."""
        return AgentRequest(
            request=self.request,
            token_budget=self.token_budget,
            top_k=self.top_k,
            retrieval_mode=RetrievalMode(self.retrieval_mode.value),
            semantic_weight=self.semantic_weight,
            use_memory=self.use_memory,
            context_only=self.context_only,
            extract_candidates=self.extract_candidates,
            store_candidates=self.store_candidates,
            source=self.source,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class CandidateMemoryResponse(BaseModel):
    """A memory the provider proposed (stored only when permitted)."""

    content: str
    memory_type: MemoryType
    confidence: float
    importance: float
    reason: str = ""
    rejection: str | None = None

    @classmethod
    def from_domain(cls, candidate: CandidateMemory) -> CandidateMemoryResponse:
        return cls(
            content=candidate.content,
            memory_type=candidate.memory_type,
            confidence=candidate.confidence,
            importance=candidate.importance,
            reason=candidate.reason,
            rejection=candidate.rejection,
        )


class OperationRecordResponse(BaseModel):
    """Safe metadata about one step of an agent turn."""

    operation: str
    success: bool
    latency_ms: float
    provider: str | None = None
    model: str | None = None
    usage: LLMUsageResponse | None = None
    context_estimated_tokens: int | None = Field(
        default=None, description="M4's own estimate, never a provider count."
    )
    memories_recalled: int | None = None
    memories_stored: int | None = None
    error_kind: str | None = None

    @classmethod
    def from_domain(cls, record: AgentOperationRecord) -> OperationRecordResponse:
        return cls(
            operation=record.operation,
            success=record.success,
            latency_ms=record.latency_ms,
            provider=record.provider,
            model=record.model,
            usage=LLMUsageResponse.from_domain(record.usage),
            context_estimated_tokens=record.context_estimated_tokens,
            memories_recalled=record.memories_recalled,
            memories_stored=record.memories_stored,
            error_kind=record.error_kind,
        )


class AgentContextBlockResponse(BaseModel):
    """Provenance of one compiled block, as used by an agent turn."""

    memory_id: str
    memory_type: MemoryType
    section: ContextSection
    order: int
    utility_score: float
    retrieval_score: float
    estimated_tokens: int

    @classmethod
    def from_domain(cls, block: ContextBlockSummary) -> AgentContextBlockResponse:
        return cls(
            memory_id=block.memory_id,
            memory_type=block.memory_type,
            section=block.section,
            order=block.order,
            utility_score=block.utility_score,
            retrieval_score=block.retrieval_score,
            estimated_tokens=block.estimated_tokens,
        )


class AgentRespondResponse(BaseModel):
    """The answer plus everything needed to audit how it was produced."""

    request: str
    answer: str
    memory_used: bool
    memory_gate: dict[str, str]
    provider: str
    model: str
    #: ``null`` unless the provider reported usage.
    usage: LLMUsageResponse | None = None
    context_text: str
    context_estimated_tokens: int = Field(
        description="M4's own estimate of the compiled context, in tokens."
    )
    context_token_budget: int
    context_blocks: list[AgentContextBlockResponse] = Field(default_factory=list)
    candidate_memories: list[CandidateMemoryResponse] = Field(default_factory=list)
    stored_memory_ids: list[str] = Field(default_factory=list)
    operations: list[OperationRecordResponse] = Field(default_factory=list)
    messages: list[dict[str, str]] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, result: AgentResult) -> AgentRespondResponse:
        return cls(
            request=result.request,
            answer=result.answer,
            memory_used=result.used_memory,
            memory_gate=result.gate.to_dict(),
            provider=result.provider or "",
            model=result.model or "",
            usage=LLMUsageResponse.from_domain(result.usage),
            context_text=result.context_text,
            context_estimated_tokens=result.context_estimated_tokens,
            context_token_budget=result.context_token_budget,
            context_blocks=[
                AgentContextBlockResponse.from_domain(block) for block in result.context_blocks
            ],
            candidate_memories=[
                CandidateMemoryResponse.from_domain(candidate)
                for candidate in result.candidate_memories
            ],
            stored_memory_ids=list(result.stored_memory_ids),
            operations=[OperationRecordResponse.from_domain(record) for record in result.records],
            messages=[message.to_dict() for message in result.messages],
        )


class RememberRequest(ApiModel):
    """Body for ``POST /api/v1/memories/remember``."""

    content: str = Field(min_length=1, max_length=4000, description="The memory text.")
    memory_type: MemoryType = Field(description="Category of the memory.")
    source: str | None = Field(default=None, max_length=200)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Record and report near-duplicates of the new memory.
    check_duplicates: bool = Field(
        default=True, description="Run M2 duplicate detection after storing."
    )

    def to_domain_call(self) -> dict[str, object]:
        """Arguments for :meth:`MemoryOperationsService.store`."""
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "source": self.source,
            "confidence": self.confidence,
            "importance": self.importance,
        }


class RememberResponse(BaseModel):
    """The stored memory plus its detected near-duplicates."""

    memory: MemoryResponse
    duplicate_count: int
    duplicates: list[DuplicateMatchResponse] = Field(default_factory=list)


# -- M6: observability and evaluation ---------------------------------------
# These are projections of frozen domain values, not new state. Each
# ``from_domain`` reads a recorded trace (or a measured evaluation result) and
# nothing else, so the wire shape can never drift from what actually happened.
#
# Every field below is safe by construction: structured metadata, identifiers,
# counts, modes, durations and *provider-reported* usage. There is deliberately
# no field for a credential, a prompt, an answer or a chain of thought, because
# the trace never contains one.


class TraceResponse(ApiModel):
    """A trace header."""

    trace_id: str
    kind: str = Field(description="agent_request, mcp_request, evaluation_case or cli.")
    operation: str
    started_at: str
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Safe, structured trace attributes."
    )

    @classmethod
    def from_domain(cls, trace: Trace) -> TraceResponse:
        """Project a trace."""
        return cls(
            trace_id=trace.trace_id,
            kind=trace.kind.value,
            operation=trace.operation,
            started_at=trace.started_at.isoformat(),
            attributes=dict(trace.attributes),
        )


class TraceEventResponse(ApiModel):
    """One event inside a trace."""

    event_id: str
    trace_id: str
    sequence: int = Field(description="Total, reproducible order within the trace.")
    kind: str
    operation: str
    status: str = Field(description="ok, skipped or error.")
    timestamp: str
    parent_id: str | None = None
    duration_ms: float | None = Field(
        default=None, description="Measured at runtime; never a fixed benchmark."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    detail: str | None = None

    @classmethod
    def from_domain(cls, event: TraceEvent) -> TraceEventResponse:
        """Project one event."""
        return cls(
            event_id=event.event_id,
            trace_id=event.trace_id,
            sequence=event.sequence,
            kind=event.kind.value,
            operation=event.operation,
            status=event.status.value,
            timestamp=event.timestamp.isoformat(),
            parent_id=event.parent_id,
            duration_ms=event.duration_ms,
            metadata=dict(event.metadata),
            detail=event.detail,
        )


class TraceSummaryResponse(ApiModel):
    """One trace reduced to counts by walking its events."""

    trace_id: str
    kind: str
    operation: str
    started_at: str
    event_count: int
    operations: dict[str, str] = Field(
        default_factory=dict, description="Operation -> terminal status."
    )
    durations_ms: dict[str, float] = Field(
        default_factory=dict, description="Operation -> total measured duration."
    )
    total_duration_ms: float
    errors: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    succeeded: bool

    @classmethod
    def from_domain(cls, summary: TraceSummary) -> TraceSummaryResponse:
        """Project a trace summary."""
        return cls(
            trace_id=summary.trace_id,
            kind=summary.kind.value,
            operation=summary.operation,
            started_at=summary.started_at.isoformat(),
            event_count=summary.event_count,
            operations=dict(sorted(summary.operations.items())),
            durations_ms={
                name: round(value, 3) for name, value in sorted(summary.durations_ms.items())
            },
            total_duration_ms=round(summary.total_duration_ms, 3),
            errors=list(summary.errors),
            memory_ids=list(summary.memory_ids),
            succeeded=summary.succeeded,
        )


class TraceListResponse(ApiModel):
    """Recent traces, newest first."""

    count: int
    traces: list[TraceResponse] = Field(default_factory=list)


class TraceDetailResponse(ApiModel):
    """One trace with its summary."""

    trace: TraceResponse
    summary: TraceSummaryResponse
    event_count: int


class MetricsResponse(ApiModel):
    """Counters derived by counting recorded events.

    ``usage_reported`` is the number of provider calls that actually supplied
    usage. It is never inferred from a token estimate or a character count.
    """

    traces: dict[str, int] = Field(default_factory=dict)
    operations: dict[str, int] = Field(default_factory=dict)
    statuses: dict[str, int] = Field(default_factory=dict)
    memory: dict[str, int] = Field(default_factory=dict)
    context: dict[str, int] = Field(default_factory=dict)
    provider: dict[str, int] = Field(default_factory=dict)
    usage_reported: int
    failed_traces: int
    event_count: int = Field(description="Total recorded events these counters were derived from.")

    @classmethod
    def from_domain(cls, metrics: OperationMetrics, *, event_count: int) -> MetricsResponse:
        """Project derived counters plus the population they came from."""
        return cls(
            traces=dict(sorted(metrics.traces.items())),
            operations=dict(sorted(metrics.operations.items())),
            statuses=dict(sorted(metrics.statuses.items())),
            memory=dict(sorted(metrics.memory.items())),
            context=dict(sorted(metrics.context.items())),
            provider=dict(sorted(metrics.provider.items())),
            usage_reported=metrics.usage_reported,
            failed_traces=metrics.failed_traces,
            event_count=event_count,
        )


class MetricResultResponse(ApiModel):
    """One evaluated metric, with the counts it was computed from."""

    name: str
    definition: str = Field(description="The exact formula this score means.")
    numerator: int
    denominator: int
    score: float
    failures: list[str] = Field(default_factory=list)


class EvaluationCategoryResponse(ApiModel):
    """Pass/total for one category."""

    passed: int
    total: int


class EvaluationCaseResponse(ApiModel):
    """One evaluated case."""

    case_id: str
    category: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    numerators: dict[str, int] = Field(default_factory=dict)
    denominators: dict[str, int] = Field(default_factory=dict)
    measurements: dict[str, float] = Field(default_factory=dict)


class EvaluationResponse(ApiModel):
    """One complete evaluation run.

    Shaped so a dashboard can render it directly: categories, per-metric series
    with their counts, and the cases behind every failure.
    """

    dataset: str
    version: str
    reference: str = Field(description="``name@version``.")
    total_cases: int
    passed_cases: int
    failed_cases: int
    case_score: float
    metric_score: float
    overall: float
    threshold: float
    passed: bool
    categories: dict[str, EvaluationCategoryResponse] = Field(default_factory=dict)
    metrics: list[MetricResultResponse] = Field(default_factory=list)
    cases: list[EvaluationCaseResponse] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, summary: EvaluationSummary) -> EvaluationResponse:
        """Project an evaluated summary into its wire shape."""
        return cls(
            dataset=summary.dataset,
            version=summary.version,
            reference=f"{summary.dataset}@{summary.version}",
            total_cases=summary.total_cases,
            passed_cases=summary.passed_cases,
            failed_cases=summary.failed_cases,
            case_score=summary.case_score,
            metric_score=summary.metric_score,
            overall=summary.overall,
            threshold=summary.threshold,
            passed=summary.passed,
            categories={
                name: EvaluationCategoryResponse(passed=counts[0], total=counts[1])
                for name, counts in summary.categories.items()
            },
            metrics=[
                MetricResultResponse(
                    name=metric.name.value,
                    definition=METRIC_DEFINITIONS[metric.name],
                    numerator=metric.numerator,
                    denominator=metric.denominator,
                    score=metric.score,
                    failures=list(metric.failures),
                )
                for metric in summary.metrics
                if metric.applicable
            ],
            cases=[
                EvaluationCaseResponse(
                    case_id=case.case_id,
                    category=case.category.value,
                    passed=case.passed,
                    failures=list(case.failures),
                    numerators={name.value: value for name, value in case.numerators.items()},
                    denominators={name.value: value for name, value in case.denominators.items()},
                    measurements=dict(sorted(case.measurements.items())),
                )
                for case in summary.failures
            ],
        )
