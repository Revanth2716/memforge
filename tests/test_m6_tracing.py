"""Tests for the M6 instrumentation: tracing real operations, not re-implementing them.

The central claim under test is that tracing is *invisible*: a traced turn returns
exactly what an untraced one does, and every event comes from a value a real
MemForge service produced.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

from app.domain.agent import AgentRequest
from app.domain.errors import LLMTransportError
from app.domain.llm import LLMMessage, LLMRequest, LLMResponse, LLMUsage
from app.domain.models import MemoryType
from app.domain.observability import EventKind, EventStatus, TraceKind
from app.domain.retrieval import RetrievalMode, RetrievalQuery
from app.domain.time import ManualClock
from app.observability.instrumentation import (
    CallTracer,
    McpTracer,
    RequestTracer,
    TracedContextCompilerService,
    TracedLLMProvider,
    TracedMemoryService,
    TracedRetrievalService,
    trace_provider,
)
from app.observability.service import ObservabilityService
from app.services.llm.mock_provider import MockLLMProvider
from tests.m6_support import Stack

QUESTION = "What do you know about my backend experience?"
STORING_QUESTION = "Note my setup.\nremember: [PREFERENCE] The user prefers a mechanical keyboard."


def _seed(stack: Stack) -> None:
    """Store a small corpus so recall and compilation have something to read."""
    stack.clock.advance(minutes=1)
    stack.memories.create_memory(
        content="The user writes Python and SQL for backend services.",
        memory_type=MemoryType.FACT,
        confidence=0.9,
        importance=0.8,
    )
    stack.clock.advance(minutes=1)
    stack.memories.create_memory(
        content="The user prefers the dark theme in every code editor.",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.8,
        importance=0.6,
    )


class TestAgentTracing:
    """One agent turn, one trace, the documented lifecycle."""

    def test_a_turn_produces_one_trace_with_the_documented_events(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION, extract_candidates=True))
        events = stack.observability.get_events(traced.trace.trace_id)
        kinds = [event.kind for event in events]
        assert kinds == [
            EventKind.REQUEST,
            EventKind.RETRIEVAL_START,
            EventKind.RETRIEVAL_COMPLETE,
            EventKind.CONTEXT_COMPILE_START,
            EventKind.CONTEXT_COMPILE_COMPLETE,
            EventKind.LLM_REQUEST,
            EventKind.LLM_RESPONSE,
            EventKind.LLM_REQUEST,
            EventKind.LLM_RESPONSE,
            EventKind.MEMORY_GATE,
            EventKind.MEMORY_CANDIDATE,
            EventKind.MEMORY_WRITE,
            EventKind.RESPONSE,
        ]
        assert [event.sequence for event in events] == list(range(len(events)))

    def test_the_trace_is_correlated_across_every_stage(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION))
        assert {
            event.trace_id for event in stack.observability.get_events(traced.trace.trace_id)
        } == {traced.trace.trace_id}
        summary = stack.observability.summarize_trace(traced.trace.trace_id)
        assert summary is not None
        assert summary.operations == {
            "agent.request": "ok",
            "agent.memory_gate": "ok",
            "memory.retrieval": "ok",
            "context.compile": "ok",
            "llm.generate": "ok",
            "agent.memory_candidate": "skipped",
            "memory.write": "skipped",
            "agent.response": "ok",
        }

    def test_spans_point_back_at_their_opening_event(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION))
        events = stack.observability.get_events(traced.trace.trace_id)
        opened = {event.event_id: event.kind for event in events}
        pairs = (
            (EventKind.RETRIEVAL_START, EventKind.RETRIEVAL_COMPLETE),
            (EventKind.CONTEXT_COMPILE_START, EventKind.CONTEXT_COMPILE_COMPLETE),
            (EventKind.LLM_REQUEST, EventKind.LLM_RESPONSE),
        )
        by_kind = {event.kind: event for event in events}
        for start, complete in pairs:
            assert by_kind[complete].parent_id == by_kind[start].event_id
        assert opened  # the opening events are recorded, not merely implied

    def test_retrieval_metadata_comes_from_the_real_response(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION))
        events = stack.observability.get_events(traced.trace.trace_id)
        complete = next(e for e in events if e.kind is EventKind.RETRIEVAL_COMPLETE)
        assert complete.metadata["retrieval_mode"] == RetrievalMode.LEXICAL.value
        assert complete.metadata["recalled"] == len(traced.result.context_blocks)
        assert complete.metadata["candidates"] >= complete.metadata["recalled"]
        assert complete.duration_ms is not None and complete.duration_ms >= 0.0

    def test_compilation_metadata_comes_from_the_real_context_result(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION, token_budget=200))
        events = stack.observability.get_events(traced.trace.trace_id)
        complete = next(e for e in events if e.kind is EventKind.CONTEXT_COMPILE_COMPLETE)
        result = traced.result
        assert complete.metadata["selected"] == len(result.context_blocks)
        assert complete.metadata["estimated_tokens"] == result.context_estimated_tokens
        assert complete.metadata["token_budget"] == result.context_token_budget
        assert complete.metadata["over_budget"] is False

    def test_usage_is_recorded_as_unreported_for_the_mock(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION, extract_candidates=True))
        responses = [
            event
            for event in stack.observability.get_events(traced.trace.trace_id)
            if event.kind is EventKind.LLM_RESPONSE
        ]
        assert len(responses) == 2  # the answer and the candidate extraction
        for event in responses:
            assert event.metadata["usage_reported"] is False
            for figure in ("usage_total_tokens", "usage_input_tokens", "usage_output_tokens"):
                assert figure not in event.metadata

    def test_a_write_without_permission_is_recorded_as_skipped(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(
            AgentRequest(request=STORING_QUESTION, extract_candidates=True)
        )
        assert traced.result.candidate_memories, "the mock proposed no candidate"
        assert traced.result.stored_memory_ids == ()
        write = next(
            event
            for event in stack.observability.get_events(traced.trace.trace_id)
            if event.kind is EventKind.MEMORY_WRITE
        )
        assert write.status is EventStatus.SKIPPED
        assert write.metadata["memory_ids"] == []

    def test_a_permitted_write_is_recorded_with_the_stored_ids(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(
            AgentRequest(request=STORING_QUESTION, extract_candidates=True, store_candidates=True)
        )
        stored = traced.result.stored_memory_ids
        assert stored, "the permitted candidate was not stored"
        write = next(
            event
            for event in stack.observability.get_events(traced.trace.trace_id)
            if event.kind is EventKind.MEMORY_WRITE
        )
        assert write.status is EventStatus.OK
        assert write.metadata["memory_ids"] == list(stored)

    def test_the_gate_decision_is_recorded(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request="hi"))
        gate = next(
            event
            for event in stack.observability.get_events(traced.trace.trace_id)
            if event.kind is EventKind.MEMORY_GATE
        )
        assert gate.status is EventStatus.SKIPPED
        assert gate.metadata["use_memory"] == "skipped"
        assert gate.metadata["should_recall"] is False
        # A gated turn performs no retrieval at all.
        assert not any(
            event.kind is EventKind.RETRIEVAL_START
            for event in stack.observability.get_events(traced.trace.trace_id)
        )

    def test_the_request_attributes_record_the_call_not_the_text(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION, top_k=3))
        trace = stack.observability.get_trace(traced.trace.trace_id)
        assert trace is not None
        assert trace.attributes["top_k"] == 3
        assert trace.attributes["provider"] == "mock"
        assert trace.attributes["request_characters"] == len(QUESTION)
        assert QUESTION not in json.dumps(trace.to_dict())

    def test_a_failing_turn_is_recorded_as_failed_and_reraised(self, stack: Stack) -> None:
        from app.domain.errors import AgentValidationError
        from app.observability.instrumentation import AgentTracer
        from app.services.agent_service import AgentService

        capped = AgentTracer(
            AgentService(stack.operations, stack.provider, max_token_budget=100),
            stack.observability,
        )
        with pytest.raises(AgentValidationError):
            capped.respond(AgentRequest(request="x", token_budget=999_999))
        trace = stack.observability.list_traces(limit=1)[0]
        summary = stack.observability.summarize_trace(trace.trace_id)
        assert summary is not None
        assert not summary.succeeded
        assert summary.operations["agent.response"] == "error"
        assert any(
            event.kind is EventKind.ERROR
            for event in stack.observability.get_events(trace.trace_id)
        )

    def test_an_explicit_trace_id_is_honoured(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.tracer.respond(AgentRequest(request=QUESTION), trace_id="trace-fixed")
        assert traced.trace.trace_id == "trace-fixed"
        assert stack.observability.get_trace("trace-fixed") is not None


class TestTracingIsTransparent:
    """The traced stack must behave exactly like the untraced one."""

    def test_a_traced_turn_equals_an_untraced_turn(self, db_path: Path, clock: ManualClock) -> None:
        from app.domain.time import ManualClock
        from app.observability.service import ObservabilityService, SequentialIdFactory
        from app.storage.sqlite.trace_store import SqliteTraceStore
        from tests.m6_support import build_m6_stack, step_timer

        other = db_path.with_name("other.sqlite")
        plain = build_m6_stack(
            other,
            ManualClock(clock.now()),
            ObservabilityService(clock=clock, timer=step_timer(), id_factory=SequentialIdFactory()),
        )
        traced_stack = build_m6_stack(
            db_path,
            clock,
            ObservabilityService(
                SqliteTraceStore(db_path),
                clock=clock,
                timer=step_timer(),
                id_factory=SequentialIdFactory(),
            ),
        )
        for target in (plain, traced_stack):
            target.clock.advance(minutes=1)
            target.memories.create_memory(
                content="The user writes Python and SQL for backend services.",
                memory_type=MemoryType.FACT,
                confidence=0.9,
                importance=0.8,
            )
        request = AgentRequest(request=QUESTION, token_budget=200, extract_candidates=True)
        untraced = plain.agent.respond(request)
        traced = traced_stack.tracer.respond(request)
        assert untraced.gate.to_dict() == traced.result.gate.to_dict()
        assert untraced.context_text == traced.result.context_text
        assert untraced.context_estimated_tokens == traced.result.context_estimated_tokens
        assert [b.memory_id for b in untraced.context_blocks] == [
            b.memory_id for b in traced.result.context_blocks
        ]
        assert untraced.stored_memory_ids == traced.result.stored_memory_ids
        assert [c.content for c in untraced.candidate_memories] == [
            c.content for c in traced.result.candidate_memories
        ]

    def test_a_wrapper_records_nothing_outside_a_trace(self, stack: Stack) -> None:
        _seed(stack)
        before = stack.observability.store.count_events()
        stack.operations.search("python backend")
        stack.operations.store(
            "An untraced memory.", MemoryType.FACT, confidence=0.5, importance=0.5
        )
        assert stack.observability.store.count_events() == before

    def test_the_wrapped_provider_is_transparent(self, stack: Stack) -> None:
        traced = cast(TracedLLMProvider, stack.provider)
        assert isinstance(traced.inner, MockLLMProvider)
        assert traced.name == "mock"
        assert traced.model == "mock-echo-v1"
        assert traced.is_configured is True
        assert traced.describe() == MockLLMProvider().describe()

    def test_wrapping_is_idempotent(self, observability: ObservabilityService) -> None:
        provider = MockLLMProvider()
        once = trace_provider(provider, observability)
        assert trace_provider(once, observability) is once


class TestServiceDecorators:
    """Each decorator reports a real result and delegates everything else."""

    def test_the_retrieval_decorator_passes_through_explain(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.operations.retrieval_service
        assert isinstance(traced, TracedRetrievalService)
        assert isinstance(traced.inner, type(stack.retrieval))
        response = traced.explain(
            stack.memories.list_memories().items[0].id,
            RetrievalQuery(text="python", top_k=1),
        )
        assert response.final_score > 0.0

    def test_the_compiler_decorator_delegates_configuration(self, stack: Stack) -> None:
        _seed(stack)
        compiler = stack.operations.compiler_service
        assert isinstance(compiler, TracedContextCompilerService)
        assert compiler.policy == stack.compiler.policy
        assert compiler.estimator_config == stack.compiler.estimator_config
        assert compiler.retrieval_service is stack.retrieval

    def test_the_memory_decorator_delegates_non_instrumented_calls(self, stack: Stack) -> None:
        _seed(stack)
        traced = stack.operations.memory_service
        assert isinstance(traced, TracedMemoryService)
        assert traced.list_memories().total == 2
        assert traced.repository is stack.memories.repository

    def test_a_failing_retrieval_is_recorded_as_an_error(
        self, stack: Stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.domain.errors import RetrievalValidationError

        def boom(*_args: object, **_kwargs: object) -> NoReturn:
            raise RetrievalValidationError("the query was rejected")

        monkeypatch.setattr(stack.retrieval, "search", boom)
        trace = stack.observability.start_trace("test")
        with (
            pytest.raises(RetrievalValidationError),
            stack.observability.use_trace(trace.trace_id),
        ):
            stack.operations.search("python")
        complete = next(
            event
            for event in stack.observability.get_events(trace.trace_id)
            if event.kind is EventKind.RETRIEVAL_COMPLETE
        )
        assert complete.status is EventStatus.ERROR
        assert complete.detail == "RetrievalValidationError"

    def test_a_failing_compilation_is_recorded_as_an_error(
        self, stack: Stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.domain.errors import ContextValidationError

        def boom(*_args: object, **_kwargs: object) -> NoReturn:
            raise ContextValidationError("the budget was rejected")

        monkeypatch.setattr(stack.compiler, "compile", boom)
        trace = stack.observability.start_trace("test")
        with (
            pytest.raises(ContextValidationError),
            stack.observability.use_trace(trace.trace_id),
        ):
            stack.operations.compile_context("python", token_budget=100)
        complete = next(
            event
            for event in stack.observability.get_events(trace.trace_id)
            if event.kind is EventKind.CONTEXT_COMPILE_COMPLETE
        )
        assert complete.status is EventStatus.ERROR
        assert complete.detail == "ContextValidationError"


class TestLifecycleDecorator:
    """M1 lifecycle operations are recorded against the active trace."""

    def test_a_store_update_and_forget_are_each_recorded(
        self, stack: Stack, observability: ObservabilityService
    ) -> None:
        _seed(stack)
        trace = observability.start_trace("mcp.memory.store", kind=TraceKind.MCP_REQUEST)
        memory_id = None
        with observability.use_trace(trace.trace_id):
            memory = stack.operations.store("A traced memory.", MemoryType.FACT)
            memory_id = memory.id
            stack.operations.update(memory_id, importance=0.9)
            stack.operations.forget(memory_id)
        kinds = [event.kind for event in observability.get_events(trace.trace_id)]
        assert kinds.count(EventKind.MEMORY_WRITE) == 1
        assert EventKind.MEMORY_UPDATE in kinds
        assert EventKind.MEMORY_FORGET in kinds
        summary = observability.summarize_trace(trace.trace_id)
        assert summary is not None
        assert summary.memory_ids == (memory_id,)
        metrics = observability.get_metrics()
        assert metrics.memory["written"] == 1
        assert metrics.memory["updated"] == 1
        assert metrics.memory["forgotten"] == 1

    def test_each_decorator_exposes_the_service_it_wraps(self, stack: Stack) -> None:
        traced_memory = cast(TracedMemoryService, stack.operations.memory_service)
        traced_retrieval = cast(TracedRetrievalService, stack.operations.retrieval_service)
        traced_compiler = cast(TracedContextCompilerService, stack.operations.compiler_service)
        assert traced_memory.inner is stack.memories
        assert traced_retrieval.inner is stack.retrieval
        assert traced_compiler.inner is stack.compiler


class TestProviderDecorator:
    """Provider instrumentation records usage only when it exists."""

    class _ReportingProvider(MockLLMProvider):
        """A mock that reports usage, standing in for a real API."""

        def generate(self, request: LLMRequest) -> LLMResponse:
            """Return a response that carries a usage report."""
            response = super().generate(request)
            return LLMResponse(
                text=response.text,
                provider=response.provider,
                model=response.model,
                usage=LLMUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
                structured=response.structured,
            )

    def test_reported_usage_is_echoed(self, observability: ObservabilityService) -> None:
        provider = TracedLLMProvider(self._ReportingProvider(), observability)
        trace = observability.start_trace("agent.respond")
        with observability.use_trace(trace.trace_id):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hello"),), operation="respond"))
        response = next(
            event
            for event in observability.get_events(trace.trace_id)
            if event.kind is EventKind.LLM_RESPONSE
        )
        assert response.metadata["usage_reported"] is True
        assert response.metadata["usage_total_tokens"] == 18
        assert response.metadata["usage_prompt_tokens"] == 11
        assert response.metadata["usage_completion_tokens"] == 7

    def test_a_provider_failure_is_recorded_and_reraised(
        self, observability: ObservabilityService
    ) -> None:

        class _Failing(MockLLMProvider):
            name = "custom"

            def generate(self, request: LLMRequest) -> LLMResponse:
                raise LLMTransportError("upstream refused", provider="custom")

        provider = TracedLLMProvider(_Failing(), observability)
        trace = observability.start_trace("agent.respond")
        with pytest.raises(LLMTransportError), observability.use_trace(trace.trace_id):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        response = next(
            event
            for event in observability.get_events(trace.trace_id)
            if event.kind is EventKind.LLM_RESPONSE
        )
        assert response.status is EventStatus.ERROR
        assert response.metadata["error_kind"] == "transport"
        assert observability.get_metrics().provider["failures"] == 1

    def test_a_structured_call_is_flagged_as_structured(
        self, observability: ObservabilityService
    ) -> None:
        provider = TracedLLMProvider(MockLLMProvider(), observability)
        trace = observability.start_trace("agent.respond")
        with observability.use_trace(trace.trace_id):
            provider.generate_structured(
                LLMRequest(
                    messages=(LLMMessage.user("remember: [FACT] something"),),
                    response_schema={"type": "object"},
                )
            )
        request = next(
            event
            for event in observability.get_events(trace.trace_id)
            if event.kind is EventKind.LLM_REQUEST
        )
        assert request.metadata["structured"] is True


class TestCallTracers:
    """MCP and REST call tracers."""

    def test_an_mcp_call_is_traced_with_its_nested_events(
        self, stack: Stack, observability: ObservabilityService
    ) -> None:
        _seed(stack)
        tracer = McpTracer(observability)
        memory_id = tracer.call(
            "memory.store",
            {"content_length": 20, "memory_type": "FACT"},
            lambda: stack.operations.store("A memory stored through MCP.", MemoryType.FACT).id,
        )
        trace = observability.list_traces(limit=1)[0]
        assert trace.kind is TraceKind.MCP_REQUEST
        assert trace.operation == "mcp.memory.store"
        assert trace.attributes["tool"] == "memory.store"
        kinds = [event.kind for event in observability.get_events(trace.trace_id)]
        assert kinds[0] is EventKind.MCP_REQUEST
        assert kinds[-1] is EventKind.MCP_RESPONSE
        assert EventKind.MEMORY_WRITE in kinds
        summary = observability.summarize_trace(trace.trace_id)
        assert summary is not None
        assert memory_id in summary.memory_ids

    def test_a_tool_error_result_closes_the_trace_as_failed(
        self, stack: Stack, observability: ObservabilityService
    ) -> None:
        class _ErrorResult:
            is_error = True

        trace_id_holder: list[str] = []

        class _Tracer(McpTracer):
            def call(
                self,
                tool: str,
                arguments: Mapping[str, Any],
                run: Callable[[], Any],
            ) -> Any:
                result = super().call(tool, arguments, run)
                trace_id_holder.append(observability.list_traces(limit=1)[0].trace_id)
                return result

        _Tracer(observability).call("memory.forget", {}, lambda: _ErrorResult())
        summary = observability.summarize_trace(trace_id_holder[0])
        assert summary is not None
        assert not summary.succeeded
        assert summary.operations["mcp.request"] == "error"

    def test_an_raised_tool_error_is_recorded_and_reraised(
        self, stack: Stack, observability: ObservabilityService
    ) -> None:
        def explode() -> None:
            raise RuntimeError("tool crashed")

        with pytest.raises(RuntimeError):
            McpTracer(observability).call("memory.store", {}, explode)
        trace = observability.list_traces(limit=1)[0]
        summary = observability.summarize_trace(trace.trace_id)
        assert summary is not None
        assert summary.errors

    def test_the_rest_tracer_records_an_api_request_kind(
        self, stack: Stack, observability: ObservabilityService
    ) -> None:
        _seed(stack)
        with RequestTracer(observability).trace("memories.remember", {"check_duplicates": True}):
            stack.operations.store("A REST-stored memory.", MemoryType.FACT)
        trace = observability.list_traces(limit=1)[0]
        assert trace.kind is TraceKind.API_REQUEST
        assert trace.operation == "api.memories.remember"
        kinds = [event.kind for event in observability.get_events(trace.trace_id)]
        assert kinds[-1] is EventKind.RESPONSE

    def test_a_request_trace_failure_is_recorded(self, observability: ObservabilityService) -> None:
        with pytest.raises(ValueError), RequestTracer(observability).trace("bad.call", {}):
            raise ValueError("nope")
        summary = observability.summarize_trace(observability.list_traces(limit=1)[0].trace_id)
        assert summary is not None
        assert not summary.succeeded

    def test_the_base_tracer_defaults_to_mcp(self, observability: ObservabilityService) -> None:
        assert CallTracer(observability).kind is TraceKind.MCP_REQUEST
        assert CallTracer(observability, kind=TraceKind.CLI, prefix="cli").kind is TraceKind.CLI
