"""Read-only observability endpoints (M6).

Four routes, all ``GET``, all projections of what already exists:

* ``GET /api/v1/traces``                   -- recent trace headers;
* ``GET /api/v1/traces/{trace_id}``        -- one trace and its summary;
* ``GET /api/v1/traces/{trace_id}/events`` -- its ordered, redacted events;
* ``GET /api/v1/metrics``                  -- counters derived from events.

Nothing here can start, mutate or delete a trace. There is deliberately no
"record a trace" endpoint: traces come from real operations, never from a caller
asserting that one happened.

No response schema has a field for a credential, a prompt or an answer, so no
secret can leave through this surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import ObservabilityServiceDependency
from app.api.errors import ObservabilityNotFoundError
from app.api.schemas import (
    MetricsResponse,
    TraceDetailResponse,
    TraceEventResponse,
    TraceListResponse,
    TraceResponse,
    TraceSummaryResponse,
)
from app.domain.errors import ObservabilityError
from app.domain.observability import TraceKind

router = APIRouter(tags=["observability"])


@router.get("/traces", response_model=TraceListResponse, summary="List recent traces")
def list_traces(
    observability: ObservabilityServiceDependency,
    limit: int = Query(default=20, ge=1, le=200, description="Maximum traces to return"),
    kind: str | None = Query(
        default=None, description="Filter by kind: agent_request, mcp_request, evaluation_case, cli"
    ),
) -> TraceListResponse:
    """Return recent trace headers, newest first."""
    traces = observability.list_traces(limit=limit, kind=_kind(kind))
    return TraceListResponse(
        count=len(traces), traces=[TraceResponse.from_domain(trace) for trace in traces]
    )


@router.get("/traces/{trace_id}", response_model=TraceDetailResponse, summary="Read one trace")
def get_trace(trace_id: str, observability: ObservabilityServiceDependency) -> TraceDetailResponse:
    """Read one trace with its attributes and its derived summary."""
    trace = observability.get_trace(trace_id)
    summary = observability.summarize_trace(trace_id)
    if trace is None or summary is None:
        raise ObservabilityNotFoundError(f"no trace with id {trace_id!r}", trace_id=trace_id)
    return TraceDetailResponse(
        trace=TraceResponse.from_domain(trace),
        summary=TraceSummaryResponse.from_domain(summary),
        event_count=summary.event_count,
    )


@router.get(
    "/traces/{trace_id}/events",
    response_model=list[TraceEventResponse],
    summary="Read one trace's events in order",
)
def get_trace_events(
    trace_id: str, observability: ObservabilityServiceDependency
) -> list[TraceEventResponse]:
    """Return every event of a trace, ordered by its recorded sequence.

    The order is the recorded sequence, not a timestamp comparison: two events
    can share an instant, and the sequence is what an operator can rely on.
    Durations are real measurements and are not comparable between runs.
    """
    if observability.get_trace(trace_id) is None:
        raise ObservabilityNotFoundError(f"no trace with id {trace_id!r}", trace_id=trace_id)
    return [TraceEventResponse.from_domain(event) for event in observability.get_events(trace_id)]


@router.get("/metrics", response_model=MetricsResponse, summary="Operational counters")
def get_metrics(observability: ObservabilityServiceDependency) -> MetricsResponse:
    """Return counters derived by counting recorded events.

    Nothing here is estimated: each value counts something that happened. The
    ``provider.usage_reported`` counter in particular stays ``0`` when no
    provider ever reported usage -- MemForge does not derive one.
    """
    metrics = observability.get_metrics()
    return MetricsResponse.from_domain(metrics, event_count=observability.store.count_events())


def _kind(value: str | None) -> TraceKind | None:
    """Parse a trace kind, or raise the shared validation error."""
    if value is None:
        return None
    try:
        return TraceKind(value)
    except ValueError:
        allowed = ", ".join(sorted(kind.value for kind in TraceKind))
        raise ObservabilityError(
            f"unknown trace kind {value!r}; expected one of {allowed}"
        ) from None


__all__ = ["router"]
