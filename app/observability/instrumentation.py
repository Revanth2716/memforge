"""Instrumentation (M6).

M1-M5 were written before traces existed and are deliberately left alone: no
memory service imports an observability concern. Instead, *decorators* wrap them
here. A decorator delegates first and reports afterwards, so a MemForge result is
never shaped by the fact that it is being watched.

::

    AgentTracer / McpTracer        start a trace, bind it, close it
        |
        +-- TracedLLMProvider          LLM_REQUEST / LLM_RESPONSE
        +-- TracedRetrievalService     RETRIEVAL_START / _COMPLETE
        +-- TracedContextCompiler      CONTEXT_COMPILE_START / _COMPLETE
        +-- TracedMemoryService        MEMORY_WRITE / _UPDATE / _FORGET
        |
    (bound via ObservabilityService.use_trace, so a wrapped call that happens
     outside a trace records nothing at all)

Two rules govern every decorator:

1. **Report, never re-compute.** Every number below is read off a value the real
   service already produced.
2. **Never record content.** Only counts, ids, modes, token figures and sizes.
   Usage is echoed only when the provider reported it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from app.domain.agent import AgentRequest, AgentResult
from app.domain.context import ContextRequest, ContextResult
from app.domain.errors import LLMError
from app.domain.llm import LLMProvider, LLMRequest, LLMResponse, message_text
from app.domain.observability import EventKind, EventStatus, Trace, TraceKind
from app.domain.retrieval import RetrievalQuery, RetrievalResponse
from app.observability.service import ObservabilityService
from app.services.context_service import ContextCompilerService
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService

#: Operations whose counters are only meaningful when a value was produced.
_SUPPRESSION_KEYWORDS = ("duplicate", "conflict", "redundant", "superseded")


def _size(payload: str | None) -> int:
    """Character count of a payload, or 0 when absent. The payload is never kept."""
    return len(payload) if payload else 0


# -- service decorators ------------------------------------------------------


class TracedMemoryService:
    """M1 with lifecycle events appended (delegating decorator)."""

    def __init__(self, inner: MemoryService, observability: ObservabilityService) -> None:
        self._inner = inner
        self._observability = observability

    @property
    def inner(self) -> MemoryService:
        """The wrapped M1 service."""
        return self._inner

    def __getattr__(self, name: str) -> Any:
        """Delegate everything the facade does not instrument."""
        return getattr(self._inner, name)

    def create_memory(self, **kwargs: Any) -> Any:
        """Store a memory, recording the write."""
        memory = self._inner.create_memory(**kwargs)
        self._emit(EventKind.MEMORY_WRITE, memory)
        return memory

    def update_memory(self, memory_id: str, update: Any) -> Any:
        """Apply a partial update, recording the new version."""
        memory = self._inner.update_memory(memory_id, update)
        self._emit(EventKind.MEMORY_UPDATE, memory)
        return memory

    def forget_memory(self, memory_id: str) -> Any:
        """Soft-delete a memory, recording the forget."""
        memory = self._inner.forget_memory(memory_id)
        self._emit(EventKind.MEMORY_FORGET, memory)
        return memory

    def _emit(self, kind: EventKind, memory: Any) -> None:
        """Record a memory lifecycle event, if a trace is in scope."""
        trace_id = self._observability.current_trace_id
        if trace_id is None:
            return
        self._observability.record_event(
            kind,
            trace_id=trace_id,
            metadata={
                "memory_ids": [memory.id],
                "memory_type": memory.memory_type.value,
                "version": memory.version,
                "status": memory.status.value,
            },
        )


class TracedRetrievalService:
    """M3 with retrieval events appended (delegating decorator)."""

    def __init__(self, inner: MemoryRetrievalService, observability: ObservabilityService) -> None:
        self._inner = inner
        self._observability = observability

    @property
    def inner(self) -> MemoryRetrievalService:
        """The wrapped M3 service."""
        return self._inner

    def __getattr__(self, name: str) -> Any:
        """Delegate everything the facade does not instrument.

        ``explain`` in particular must **not** be instrumented: the context
        compiler calls it to re-score memories M3 already suppressed, and
        counting it as another retrieval would double-count the work.
        """
        return getattr(self._inner, name)

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Run a search, recording its real mode, counts and latency."""
        trace_id = self._observability.current_trace_id
        if trace_id is None:
            return self._inner.search(query)
        span = self._observability.begin_operation(
            EventKind.RETRIEVAL_START,
            trace_id=trace_id,
            metadata={
                "requested_mode": query.mode.value,
                "top_k": query.top_k,
                "query_characters": _size(query.text),
                "semantic_available": self._inner.semantic_available,
            },
        )
        try:
            response = self._inner.search(query)
        except Exception as error:
            self._observability.complete_operation(
                span,
                EventKind.RETRIEVAL_COMPLETE,
                status=EventStatus.ERROR,
                metadata={"error_kind": getattr(error, "kind", type(error).__name__)},
                detail=type(error).__name__,
            )
            raise
        self._observability.complete_operation(
            span,
            EventKind.RETRIEVAL_COMPLETE,
            metadata={
                "retrieval_mode": response.retrieval_mode.value,
                "candidates": response.candidates_considered,
                "recalled": response.candidates_returned,
                "duplicate_candidates": len(response.suppressed),
                "degraded": response.degraded_reason is not None,
                "latency_ms": response.latency_ms,
            },
        )
        return response


class TracedContextCompilerService:
    """M4 with compilation events appended (delegating decorator)."""

    def __init__(self, inner: ContextCompilerService, observability: ObservabilityService) -> None:
        self._inner = inner
        self._observability = observability

    @property
    def inner(self) -> ContextCompilerService:
        """The wrapped M4 compiler."""
        return self._inner

    def __getattr__(self, name: str) -> Any:
        """Delegate everything the facade does not instrument."""
        return getattr(self._inner, name)

    def compile(self, request: ContextRequest) -> ContextResult:
        """Compile a context, recording selection and suppression counts."""
        trace_id = self._observability.current_trace_id
        if trace_id is None:
            return self._inner.compile(request)
        span = self._observability.begin_operation(
            EventKind.CONTEXT_COMPILE_START,
            trace_id=trace_id,
            metadata={
                "token_budget": request.token_budget,
                "top_k": request.top_k,
                "requested_mode": request.retrieval_mode.value,
                "compress": request.compress,
                "estimator": self._inner.estimator_config.estimator_id,
            },
        )
        try:
            result = self._inner.compile(request)
        except Exception as error:
            self._observability.complete_operation(
                span,
                EventKind.CONTEXT_COMPILE_COMPLETE,
                status=EventStatus.ERROR,
                metadata={"error_kind": getattr(error, "kind", type(error).__name__)},
                detail=type(error).__name__,
            )
            raise
        self._observability.complete_operation(
            span,
            EventKind.CONTEXT_COMPILE_COMPLETE,
            metadata={
                "token_budget": result.token_budget,
                "estimated_tokens": result.estimated_tokens,
                "remaining_tokens": result.remaining_tokens,
                "candidates": result.candidates_considered,
                "selected": result.selected_count,
                "suppressed": result.suppressed_count,
                "over_budget": result.estimated_tokens > result.token_budget,
                "retrieval_mode": result.retrieval_mode.value,
                **_suppression_breakdown(result),
            },
        )
        return result


class TracedLLMProvider:
    """A provider decorator that records the call and the *reported* usage.

    ``LLMResponse.usage`` is ``None`` unless the provider sent one. The trace
    mirrors that exactly: ``usage_reported`` is ``False`` and no token count is
    written when nothing was supplied. A character count is never converted into
    tokens and no cost is ever derived.
    """

    def __init__(self, inner: LLMProvider, observability: ObservabilityService) -> None:
        self._inner = inner
        self._observability = observability

    @property
    def inner(self) -> LLMProvider:
        """The wrapped provider."""
        return self._inner

    @property
    def name(self) -> str:
        """The wrapped provider's name."""
        return self._inner.name

    @property
    def model(self) -> str:
        """The wrapped provider's model."""
        return self._inner.model

    @property
    def is_configured(self) -> bool:
        """The wrapped provider's readiness."""
        return self._inner.is_configured

    def describe(self) -> str:
        """The wrapped provider's description (never a credential)."""
        return self._inner.describe()

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate, recording the call and its real usage if reported."""
        return self._call(request, self._inner.generate)

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate structured output, recorded the same way."""
        return self._call(request, self._inner.generate_structured)

    def _call(self, request: LLMRequest, run: Callable[[LLMRequest], LLMResponse]) -> LLMResponse:
        """Delegate one provider call inside a traced span."""
        trace_id = self._observability.current_trace_id
        if trace_id is None:
            return run(request)
        span = self._observability.begin_operation(
            EventKind.LLM_REQUEST,
            trace_id=trace_id,
            metadata={
                "provider": self._inner.name,
                "model": self._inner.model,
                "operation": request.operation,
                "input_characters": _size(message_text(request.messages)),
                "structured": request.wants_structured_output,
            },
        )
        try:
            response = run(request)
        except LLMError as error:
            self._observability.complete_operation(
                span,
                EventKind.LLM_RESPONSE,
                status=EventStatus.ERROR,
                metadata={"error_kind": getattr(error, "kind", "llm_error")},
                detail=type(error).__name__,
            )
            raise
        self._observability.complete_operation(
            span,
            EventKind.LLM_RESPONSE,
            metadata={
                "provider": self._inner.name,
                "model": self._inner.model,
                "operation": request.operation,
                "output_characters": _size(response.text),
                **_usage_metadata(response),
            },
        )
        return response


def _usage_metadata(response: LLMResponse) -> dict[str, Any]:
    """Echo provider-reported usage; omit every figure when none was sent."""
    if not response.has_usage or response.usage is None:
        return {"usage_reported": False}
    usage = response.usage
    return {
        "usage_reported": True,
        "usage_total_tokens": usage.total_tokens,
        "usage_prompt_tokens": usage.prompt_tokens,
        "usage_completion_tokens": usage.completion_tokens,
    }


def _suppression_breakdown(result: ContextResult) -> dict[str, int]:
    """Count suppression reasons by keyword, from the real selection trace."""
    breakdown: dict[str, int] = {}
    for entry in result.selection_trace:
        if entry.selected:
            continue
        reason = entry.reason.value
        for keyword in _SUPPRESSION_KEYWORDS:
            if keyword in reason:
                breakdown[keyword] = breakdown.get(keyword, 0) + 1
                break
    return {f"suppressed_{key}": value for key, value in sorted(breakdown.items())}


# -- assembly ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TracedStack:
    """The instrumented object graph handed to the callers."""

    operations: MemoryOperationsService
    agent: Any
    provider: LLMProvider


def build_traced_operations(
    memories: MemoryService,
    retrieval: MemoryRetrievalService,
    compiler: ContextCompilerService,
    observability: ObservabilityService,
) -> MemoryOperationsService:
    """Compose a facade whose M1/M3/M4 layers are traced decorators.

    The facade itself is the M5 class, untouched -- only the services it delegates
    to are wrapped, so every internal call it makes is instrumented too.

    One deliberate detail: the compiler keeps the *undecorated* retrieval service,
    because M4's internal re-run of retrieval is part of compiling a context, not
    a separate recall. Its work is therefore accounted for inside the
    ``context.compile`` span, whose ``candidates`` count is everything
    compilation considered.
    """
    return MemoryOperationsService(
        TracedMemoryService(memories, observability),  # type: ignore[arg-type]
        TracedRetrievalService(retrieval, observability),  # type: ignore[arg-type]
        TracedContextCompilerService(compiler, observability),  # type: ignore[arg-type]
    )


def trace_provider(provider: LLMProvider, observability: ObservabilityService) -> LLMProvider:
    """Wrap a provider so its calls are recorded."""
    if isinstance(provider, TracedLLMProvider):
        return provider
    traced = TracedLLMProvider(provider, observability)
    return cast("LLMProvider", traced)


# -- callers -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TracedTurn:
    """One instrumented agent turn: the real result plus its trace."""

    trace: Trace
    result: AgentResult


class AgentTracer:
    """Runs one agent turn inside a trace.

    The retrieval, compilation, provider and write stages are recorded by the
    decorators in this module. Only two things are reported here, because they
    exist nowhere else: the memory gate's decision (a heuristic, not a service
    call) and the candidates the provider proposed (which the turn extracts
    *after* the provider call, so no decorator can see them).
    """

    def __init__(self, agent: Any, observability: ObservabilityService) -> None:
        self._agent = agent
        self._observability = observability

    @property
    def agent(self) -> Any:
        """The wrapped agent service."""
        return self._agent

    def respond(self, request: AgentRequest, *, trace_id: str | None = None) -> TracedTurn:
        """Run a turn inside a trace and return both the result and the trace."""
        observability = self._observability
        trace = observability.start_trace(
            "agent.respond",
            kind=TraceKind.AGENT_REQUEST,
            trace_id=trace_id,
            attributes={
                "use_memory": request.use_memory,
                "context_only": request.context_only,
                "extract_candidates": request.extract_candidates,
                "store_candidates": request.store_candidates,
                "retrieval_mode": request.retrieval_mode.value,
                "top_k": request.top_k,
                "token_budget": request.token_budget,
                "request_characters": _size(request.request),
                "provider": self._agent.provider.name,
                "model": self._agent.provider.model,
            },
        )
        with observability.use_trace(trace.trace_id):
            try:
                result = self._agent.respond(request)
            except Exception as error:
                observability.record_event(
                    EventKind.ERROR,
                    trace_id=trace.trace_id,
                    status=EventStatus.ERROR,
                    metadata={"error_kind": getattr(error, "kind", type(error).__name__)},
                    detail=type(error).__name__,
                )
                observability.end_trace(trace.trace_id, status=EventStatus.ERROR)
                raise
        self._report_gate(trace.trace_id, request, result)
        observability.end_trace(
            trace.trace_id,
            status=EventStatus.OK,
            metadata={
                "answer_characters": _size(result.answer),
                "context_characters": _size(result.context_text),
                "context_blocks": len(result.context_blocks),
                "used_memory": result.used_memory,
            },
        )
        return TracedTurn(trace=trace, result=result)

    def _report_gate(self, trace_id: str, request: AgentRequest, result: AgentResult) -> None:
        """Record the memory gate and the candidate extraction outcome."""
        observability = self._observability
        observability.record_event(
            EventKind.MEMORY_GATE,
            trace_id=trace_id,
            status=EventStatus.SKIPPED if not result.gate.should_recall else EventStatus.OK,
            metadata={
                "use_memory": result.gate.use.value,
                "should_recall": result.gate.should_recall,
                "reason": result.gate.reason,
                "requested": request.use_memory,
            },
        )
        observability.record_event(
            EventKind.MEMORY_CANDIDATE,
            trace_id=trace_id,
            status=(EventStatus.SKIPPED if not request.extract_candidates else EventStatus.OK),
            metadata={
                "extracted": request.extract_candidates,
                "candidate_count": len(result.candidate_memories),
                "store_permitted": request.store_candidates,
                "rejected": sum(1 for item in result.candidate_memories if not item.is_valid),
            },
        )
        if not result.stored_memory_ids:
            observability.record_event(
                EventKind.MEMORY_WRITE,
                trace_id=trace_id,
                status=EventStatus.SKIPPED,
                metadata={
                    "memory_ids": [],
                    "reason": "storage was not permitted or no valid candidate was proposed",
                },
            )


class CallTracer:
    """Runs one tool or endpoint body inside a trace.

    The body itself is unchanged; this only opens and closes the trace so that
    whatever it did through the traced services lands inside it. The kind and the
    operation prefix are the only differences between an MCP call, a REST write
    and a CLI step.
    """

    def __init__(
        self,
        observability: ObservabilityService,
        *,
        kind: TraceKind = TraceKind.MCP_REQUEST,
        prefix: str = "mcp",
    ) -> None:
        """Args:
        observability: The service that owns the trace.
        kind: Trace kind to record (``mcp_request``, ``api_request``, ``cli``).
        prefix: Operation-name prefix, e.g. ``"mcp"`` or ``"api"``.
        """
        self._observability = observability
        self._kind = kind
        self._prefix = prefix

    @property
    def kind(self) -> TraceKind:
        """The trace kind this tracer records."""
        return self._kind

    def call(self, tool: str, arguments: Mapping[str, Any], run: Any) -> Any:
        """Run ``run`` inside a trace named after the call.

        A tool that reports a structured error (``isError: true``) is a *failed
        call*, not a successful one, so the trace is closed with an ``error``
        status rather than claiming the operation worked.
        """
        trace = self._start(tool, arguments)
        with self._observability.use_trace(trace.trace_id):
            try:
                result = run()
            except Exception as error:
                self._fail(trace.trace_id, error)
                raise
        failed = _is_error(result)
        self._observability.end_trace(
            trace.trace_id,
            status=EventStatus.ERROR if failed else EventStatus.OK,
            metadata={"tool": tool, "is_error": failed},
        )
        return result

    @contextmanager
    def trace(self, tool: str, arguments: Mapping[str, Any] | None = None) -> Iterator[Trace]:
        """Bind a trace around a ``with`` block.

        The closing event is recorded on the way out, whatever happened inside;
        a raised exception is recorded as an ``ERROR`` event and re-raised, so a
        trace never claims a call succeeded when it did not.
        """
        trace = self._start(tool, arguments or {})
        try:
            with self._observability.use_trace(trace.trace_id):
                yield trace
        except Exception as error:
            self._fail(trace.trace_id, error)
            raise
        self._observability.end_trace(
            trace.trace_id, status=EventStatus.OK, metadata={"tool": tool}
        )

    def _start(self, tool: str, arguments: Mapping[str, Any]) -> Trace:
        """Open a trace and record the opening request event."""
        return self._observability.start_trace(
            f"{self._prefix}.{tool}",
            kind=self._kind,
            attributes={
                "tool": tool,
                "arguments": sorted(arguments),
                "argument_count": len(arguments),
            },
        )

    def _fail(self, trace_id: str, error: Exception) -> None:
        """Record an unexpected failure and close the trace as failed."""
        observability = self._observability
        observability.record_event(
            EventKind.ERROR,
            trace_id=trace_id,
            status=EventStatus.ERROR,
            detail=type(error).__name__,
        )
        observability.end_trace(trace_id, status=EventStatus.ERROR)


class McpTracer(CallTracer):
    """A :class:`CallTracer` for one MCP tool call."""

    def __init__(self, observability: ObservabilityService) -> None:
        """Bind the tracer to MCP traces."""
        super().__init__(observability, kind=TraceKind.MCP_REQUEST, prefix="mcp")


class RequestTracer(CallTracer):
    """A :class:`CallTracer` for one REST write that MemForge owns."""

    def __init__(self, observability: ObservabilityService) -> None:
        """Bind the tracer to REST traces."""
        super().__init__(observability, kind=TraceKind.API_REQUEST, prefix="api")


def _is_error(result: Any) -> bool:
    """True when an MCP tool result carries ``isError: true``."""
    return bool(getattr(result, "is_error", False))


__all__ = [
    "AgentTracer",
    "CallTracer",
    "McpTracer",
    "RequestTracer",
    "TracedContextCompilerService",
    "TracedLLMProvider",
    "TracedMemoryService",
    "TracedRetrievalService",
    "TracedStack",
    "TracedTurn",
    "build_traced_operations",
    "trace_provider",
]
