"""Observability subsystem (M6).

Two layers, deliberately separate:

* :mod:`app.domain.observability` -- the trace/event *models* and the redaction
  helpers. No I/O, no services, no transport.
* :mod:`app.observability` -- the service that owns a trace lifecycle and the
  decorators that emit events into it.

Nothing here imports FastAPI, an LLM provider or the evaluation subsystem.
"""

from __future__ import annotations

from app.observability.instrumentation import (
    AgentTracer,
    CallTracer,
    McpTracer,
    RequestTracer,
    TracedContextCompilerService,
    TracedLLMProvider,
    TracedMemoryService,
    TracedRetrievalService,
    TracedTurn,
    build_traced_operations,
    trace_provider,
)
from app.observability.service import (
    NullTraceStore,
    ObservabilityService,
    OperationSpan,
    SequentialIdFactory,
)

__all__ = [
    "AgentTracer",
    "CallTracer",
    "McpTracer",
    "NullTraceStore",
    "ObservabilityService",
    "OperationSpan",
    "RequestTracer",
    "SequentialIdFactory",
    "TracedContextCompilerService",
    "TracedLLMProvider",
    "TracedMemoryService",
    "TracedRetrievalService",
    "TracedTurn",
    "build_traced_operations",
    "trace_provider",
]
