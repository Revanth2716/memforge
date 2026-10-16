"""Retrieval endpoints (``/api/v1/memories/search``).

The route layer only translates HTTP <-> domain: all ranking, filtering and
explanation logic lives in :class:`MemoryRetrievalService`.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Path, Query, status
from pydantic import StringConstraints

from app.api.dependencies import MemoryRetrievalServiceDependency
from app.api.schemas import (
    ConflictInfoResponse,
    FactorResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    RetrievalResultResponse,
    SuppressedCandidateResponse,
)
from app.domain.models import MemoryType
from app.domain.retrieval import RetrievalMode

router = APIRouter(prefix="/memories", tags=["retrieval"])

MemoryIdPath = Annotated[UUID, Path(description="Memory identifier (UUID).")]


def _to_response(result: object) -> RetrievalResultResponse:
    """Translate a domain retrieval result into its API representation."""
    from app.domain.retrieval import RetrievalResult

    assert isinstance(result, RetrievalResult)
    memory = result.memory
    factors = {factor.name: factor.value for factor in result.components}
    return RetrievalResultResponse(
        memory_id=memory.id,
        content=memory.content,
        memory_type=memory.memory_type,
        status=memory.status,
        confidence=memory.confidence,
        importance=memory.importance,
        effective_priority=factors.get("effective_priority", 0.0),
        freshness=factors.get("freshness", 0.0),
        lexical_score=result.lexical_score,
        semantic_score=result.semantic_score,
        relevance=result.relevance,
        final_score=result.final_score,
        factors=[FactorResponse.from_domain(factor) for factor in result.components],
        explanation=result.explanation,
        duplicate_of=result.duplicate_of,
        superseded_by=result.superseded_by,
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
            for conflict in result.conflicts
        ],
    )


@router.post(
    "/search",
    response_model=MemorySearchResponse,
    summary="Search memories (lexical, semantic or hybrid)",
)
def search_memories(
    payload: MemorySearchRequest,
    service: MemoryRetrievalServiceDependency,
) -> MemorySearchResponse:
    """Return the most useful memories for a query, with full explanations.

    ``retrieval_mode`` selects the strategy. When ``semantic`` or ``hybrid`` is
    requested but no embedding provider is available, the response says so
    (``semantic_available: false``, ``retrieval_mode: "lexical"``,
    ``degraded_reason``) instead of pretending embeddings were used.
    """
    response = service.search(payload.to_domain_query())
    return MemorySearchResponse(
        query=response.query,
        top_k=response.top_k,
        requested_mode=response.requested_mode.value,
        retrieval_mode=response.retrieval_mode.value,
        semantic_available=response.semantic_available,
        semantic_provider=response.semantic_provider,
        semantic_model=response.semantic_model,
        degraded_reason=response.degraded_reason,
        candidates_considered=response.candidates_considered,
        candidates_returned=response.candidates_returned,
        candidates_suppressed=response.candidates_suppressed,
        latency_ms=response.latency_ms,
        results=[_to_response(result) for result in response.results],
        suppressed=[
            SuppressedCandidateResponse(
                memory_id=candidate.memory_id,
                final_score=candidate.final_score,
                reason=candidate.reason,
            )
            for candidate in response.suppressed
        ],
    )


@router.get(
    "/{memory_id}/explain",
    response_model=RetrievalResultResponse,
    summary="Explain how one memory would rank for a query",
)
def explain_memory(
    memory_id: MemoryIdPath,
    service: MemoryRetrievalServiceDependency,
    query: Annotated[
        str,
        StringConstraints(min_length=1, max_length=1000, strip_whitespace=True),
        Query(description="Natural language query."),
    ],
    retrieval_mode: Annotated[RetrievalMode, Query(description="Requested retrieval mode.")] = (
        RetrievalMode.LEXICAL
    ),
) -> RetrievalResultResponse:
    """Score a single memory against a query and return the full breakdown."""
    from app.domain.retrieval import RetrievalQuery

    result = service.explain(
        str(memory_id),
        RetrievalQuery(text=query, mode=retrieval_mode, top_k=1),
    )
    return _to_response(result)


__all__ = ["MemoryType", "router", "search_memories", "status"]
