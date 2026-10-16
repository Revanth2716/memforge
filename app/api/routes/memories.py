"""Memory CRUD + history endpoints (``/api/v1/memories``)."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Path, Query, status

from app.api.dependencies import MemoryServiceDependency
from app.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from app.api.schemas import (
    ConflictCandidateResponse,
    ConflictResolutionResponse,
    ConflictsResponse,
    DuplicateMatchResponse,
    DuplicatesResponse,
    IntelligenceResponse,
    M2FactorResponse,
    MemoryCreateRequest,
    MemoryHistoryResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
    MemoryVersionResponse,
    RelationshipResponse,
    RelationshipsResponse,
    ResolutionStepResponse,
)
from app.domain.models import MemoryStatus, MemoryType
from app.storage.base import MemoryQuery

router = APIRouter(prefix="/memories", tags=["memories"])

PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT, description="Page size.")]
PageOffset = Annotated[int, Query(ge=0, description="Number of results to skip.")]
MemoryIdPath = Annotated[UUID, Path(description="Memory identifier (UUID).")]


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a memory",
)
def create_memory(payload: MemoryCreateRequest, service: MemoryServiceDependency) -> MemoryResponse:
    """Store a new memory together with its first history version."""
    memory = service.create_memory(
        content=payload.content,
        memory_type=payload.memory_type,
        source=payload.source,
        confidence=payload.confidence,
        importance=payload.importance,
    )
    return MemoryResponse.from_domain(memory)


@router.get("", response_model=MemoryListResponse, summary="List memories")
def list_memories(
    service: MemoryServiceDependency,
    status: Annotated[MemoryStatus | None, Query(description="Filter by status.")] = None,
    memory_type: Annotated[MemoryType | None, Query(description="Filter by memory type.")] = None,
    limit: PageLimit = DEFAULT_PAGE_LIMIT,
    offset: PageOffset = 0,
) -> MemoryListResponse:
    """Return a page of memories, newest first.

    Without a ``status`` filter only ``ACTIVE`` memories are listed; pass
    ``status=ARCHIVED`` or ``status=DELETED`` to inspect those explicitly.
    """
    query = MemoryQuery(
        status=status if status is not None else MemoryStatus.ACTIVE,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
    )
    page = service.list_memories(query)
    return MemoryListResponse(
        items=[MemoryResponse.from_domain(memory) for memory in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Get a memory by id",
)
def get_memory(memory_id: MemoryIdPath, service: MemoryServiceDependency) -> MemoryResponse:
    """Return a single memory. Forgotten memories return 404."""
    memory = service.get_memory(str(memory_id))
    return MemoryResponse.from_domain(memory)


@router.get(
    "/{memory_id}/history",
    response_model=MemoryHistoryResponse,
    summary="Inspect the version history of a memory",
)
def get_memory_history(
    memory_id: MemoryIdPath, service: MemoryServiceDependency
) -> MemoryHistoryResponse:
    """Return every version of a memory, oldest first.

    History remains available after a memory is forgotten.
    """
    versions = service.get_history(str(memory_id))
    current = versions[-1].version if versions else 0
    return MemoryHistoryResponse(
        memory_id=str(memory_id),
        current_version=current,
        total=len(versions),
        items=[MemoryVersionResponse.from_domain(version) for version in versions],
    )


@router.patch(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Update a memory",
)
def update_memory(
    memory_id: MemoryIdPath,
    payload: MemoryUpdateRequest,
    service: MemoryServiceDependency,
) -> MemoryResponse:
    """Apply a partial update; a new history version is appended."""
    memory = service.update_memory(str(memory_id), payload.to_domain_update())
    return MemoryResponse.from_domain(memory)


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Forget a memory (soft delete)",
)
def forget_memory(memory_id: MemoryIdPath, service: MemoryServiceDependency) -> None:
    """Forget a memory: it disappears from reads but its history is kept.

    Repeat calls for the same id return 204 (idempotent).
    """
    service.forget_memory(str(memory_id))


@router.get(
    "/{memory_id}/intelligence",
    response_model=IntelligenceResponse,
    summary="Deterministic intelligence for a memory",
)
def get_memory_intelligence(
    memory_id: MemoryIdPath, service: MemoryServiceDependency
) -> IntelligenceResponse:
    """Importance breakdown, confidence and decayed effective priority.

    Every number is recomputed on request from the stored memory, the configured
    weights and the current time; nothing is cached or stored.
    """
    memory = service.get_memory(str(memory_id))
    importance = service.score_memory(str(memory_id))
    priority = service.get_effective_priority(str(memory_id))
    return IntelligenceResponse(
        memory_id=memory.id,
        memory_type=memory.memory_type,
        status=memory.status,
        version=memory.version,
        confidence=memory.confidence,
        stored_importance=memory.importance,
        importance_score=importance.score,
        importance_factors=[M2FactorResponse.from_domain(factor) for factor in importance.factors],
        importance_explanation=importance.explanation,
        effective_priority=priority.priority,
        base_priority=priority.base_priority,
        decay_multiplier=priority.decay_multiplier,
        age_days=priority.age_days,
        priority_explanation=priority.explanation,
        access_count=memory.access_count,
        last_accessed_at=memory.last_accessed_at,
        decay_half_life_days=service.intelligence_config.decay_half_life_days,
        similarity_threshold=service.intelligence_config.similarity_threshold,
        computed_at=priority.computed_at,
    )


@router.get(
    "/{memory_id}/duplicates",
    response_model=DuplicatesResponse,
    summary="Near-duplicate memories",
)
def get_memory_duplicates(
    memory_id: MemoryIdPath,
    service: MemoryServiceDependency,
    threshold: Annotated[
        float | None, Query(ge=0.0, le=1.0, description="Override the similarity threshold.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT, description="Maximum matches.")] = 20,
) -> DuplicatesResponse:
    """Active memories that say (nearly) the same thing, best match first.

    Detection is lexical and deterministic (Dice coefficient over normalised
    token sets). Similarity is reported for every match, and each match is
    recorded as a ``DUPLICATE_OF`` relationship. No memory is modified.
    """
    matches = service.find_duplicates(str(memory_id), threshold=threshold, limit=limit)
    effective_threshold = (
        threshold if threshold is not None else service.intelligence_config.similarity_threshold
    )
    return DuplicatesResponse(
        memory_id=str(memory_id),
        threshold=effective_threshold,
        total=len(matches),
        items=[
            DuplicateMatchResponse(
                memory_id=match.memory_id,
                similarity=match.similarity,
                content=match.content,
                memory_type=match.memory_type,
                shared_tokens=list(match.shared_tokens),
                explanation=match.explanation,
            )
            for match in matches
        ],
    )


@router.get(
    "/{memory_id}/conflicts",
    response_model=ConflictsResponse,
    summary="Conflicting memories",
)
def get_memory_conflicts(
    memory_id: MemoryIdPath, service: MemoryServiceDependency
) -> ConflictsResponse:
    """Unresolved conflicts: same attribute, different value.

    Pairs already resolved by a ``SUPERSEDES`` relationship are not reported.
    """
    candidates = service.detect_conflicts(str(memory_id))
    return ConflictsResponse(
        memory_id=str(memory_id),
        total=len(candidates),
        items=[
            ConflictCandidateResponse(
                target_memory_id=candidate.target_memory_id,
                attribute=candidate.attribute,
                source_value=candidate.source_value,
                target_value=candidate.target_value,
                value_similarity=candidate.value_similarity,
                source_content=candidate.source_content,
                target_content=candidate.target_content,
                explanation=candidate.explanation,
            )
            for candidate in candidates
        ],
    )


@router.post(
    "/{memory_id}/conflicts/{target_id}/resolve",
    response_model=ConflictResolutionResponse,
    summary="Resolve a conflict deterministically",
)
def resolve_memory_conflict(
    memory_id: MemoryIdPath,
    target_id: MemoryIdPath,
    service: MemoryServiceDependency,
) -> ConflictResolutionResponse:
    """Pick a winner with the documented policy and record ``SUPERSEDES``.

    The losing memory is archived (which appends a history version), so the
    decision is visible in history and nothing is overwritten.
    """
    resolution = service.resolve_conflict(str(memory_id), target_id=str(target_id))
    return ConflictResolutionResponse(
        winner_memory_id=resolution.winner.id,
        loser_memory_id=resolution.loser.id,
        decided_by=resolution.decided_by,
        explanation=resolution.explanation,
        steps=[
            ResolutionStepResponse(
                factor=step.factor,
                left_value=step.left_value,
                right_value=step.right_value,
                winner_id=step.winner_id,
            )
            for step in resolution.steps
        ],
        loser_archived=resolution.loser_archived,
        conflict=ConflictCandidateResponse(
            target_memory_id=resolution.conflict.target_memory_id,
            attribute=resolution.conflict.attribute,
            source_value=resolution.conflict.source_value,
            target_value=resolution.conflict.target_value,
            value_similarity=resolution.conflict.value_similarity,
            source_content=resolution.conflict.source_content,
            target_content=resolution.conflict.target_content,
            explanation=resolution.conflict.explanation,
        ),
    )


@router.get(
    "/{memory_id}/relationships",
    response_model=RelationshipsResponse,
    summary="Relationships recorded for a memory",
)
def get_memory_relationships(
    memory_id: MemoryIdPath, service: MemoryServiceDependency
) -> RelationshipsResponse:
    """Every ``DUPLICATE_OF`` / ``CONFLICTS_WITH`` / ``SUPERSEDES`` link."""
    relationships = service.get_relationships(str(memory_id))
    return RelationshipsResponse(
        memory_id=str(memory_id),
        total=len(relationships),
        items=[
            RelationshipResponse(
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                relationship=relationship.relationship.value,
                similarity=relationship.similarity,
                created_at=relationship.created_at,
            )
            for relationship in relationships
        ],
    )
