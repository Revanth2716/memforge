"""M5 endpoints: the memory-aware agent and the agent-facing ``remember`` verb.

Two routes, both thin translations of existing services. No retrieval, context
compilation or storage logic lives here.

* ``POST /api/v1/agent/respond`` -- one memory-aware turn, inside a trace (M6).
* ``POST /api/v1/memories/remember`` -- store a memory and report its duplicates.

Nothing in either response can contain a provider credential: the response
schemas simply have no field for one.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.api.dependencies import (
    AgentTracerDependency,
    MemoryOperationsServiceDependency,
    ObservabilityServiceDependency,
)
from app.api.schemas import (
    AgentRespondRequest,
    AgentRespondResponse,
    DuplicateMatchResponse,
    MemoryResponse,
    RememberRequest,
    RememberResponse,
)
from app.observability.instrumentation import RequestTracer

#: Response header carrying the id of the trace a request produced.
TRACE_HEADER = "X-MemForge-Trace-Id"

router = APIRouter(tags=["agent"])


@router.post(
    "/agent/respond",
    response_model=AgentRespondResponse,
    summary="Answer a request using compiled memory context",
)
def respond_to_request(
    payload: AgentRespondRequest,
    tracer: AgentTracerDependency,
    response: Response,
) -> AgentRespondResponse:
    """Run one memory-aware agent turn.

    The turn is: memory gate -> recall (M3) -> context compilation (M4) ->
    provider call -> answer, plus optional candidate extraction. Storing a
    candidate requires ``store_candidates: true``; without it the candidates are
    reported and discarded.

    When the provider fails, the endpoint returns an explicit error envelope
    (``LLM_PROVIDER_ERROR`` and friends). MemForge never substitutes its own
    text for an answer the model did not give.

    Every turn is recorded as one trace whose id is returned in the
    ``X-MemForge-Trace-Id`` response header, so the whole lifecycle can be read
    back from ``GET /api/v1/traces/{trace_id}/events``.
    """
    traced = tracer.respond(payload.to_domain_request())
    response.headers[TRACE_HEADER] = traced.trace.trace_id
    return AgentRespondResponse.from_domain(traced.result)


@router.post(
    "/memories/remember",
    response_model=RememberResponse,
    status_code=201,
    summary="Store a memory and report its near-duplicates",
)
def remember(
    payload: RememberRequest,
    service: MemoryOperationsServiceDependency,
    observability: ObservabilityServiceDependency,
) -> RememberResponse:
    """Store one memory (M1) and report the near-duplicates M2 finds for it.

    The write is recorded inside a short trace of its own, so a store can be
    audited without an agent turn around it.
    """
    tracer = RequestTracer(observability)
    with tracer.trace("memories.remember", {"check_duplicates": payload.check_duplicates}):
        memory = service.store(**payload.to_domain_call())  # type: ignore[arg-type]
        duplicates = (
            service.memory_service.find_duplicates(memory.id) if payload.check_duplicates else []
        )
    return RememberResponse(
        memory=MemoryResponse.from_domain(memory),
        duplicate_count=len(duplicates),
        duplicates=[
            DuplicateMatchResponse(
                memory_id=match.memory_id,
                similarity=match.similarity,
                content=match.content,
                memory_type=match.memory_type,
                shared_tokens=list(match.shared_tokens),
                explanation=match.explanation,
            )
            for match in duplicates
        ],
    )


__all__ = ["router"]
