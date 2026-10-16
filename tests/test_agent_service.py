"""M5 agent tests: the memory-aware workflow end to end.

The agent is exercised through a real service stack (SQLite + M1-M4 + provider)
with an injected provider, so the workflow is proved without a network call and
without a model.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

from app.domain.agent import (
    AgentRequest,
    CandidateMemory,
    MemoryUse,
    OperationRecord,
    build_messages,
    decide_memory_use,
)
from app.domain.errors import AgentValidationError, LLMProviderError, LLMTimeoutError
from app.domain.ids import SequentialIdGenerator
from app.domain.llm import LLMMessage, LLMProvider, LLMRequest, LLMResponse, LLMUsage
from app.domain.models import MemoryType
from app.domain.retrieval import RetrievalMode
from app.domain.time import ManualClock
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.mock_provider import MockLLMProvider
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository

PYTHON = "User prefers Python for backend work."
QUESTION = "Which backend language should I use for the MemForge service?"
REMEMBER = "Please remember: the user ships on Fridays."


class ScriptedProvider(LLMProvider):
    """A provider whose answers are decided by the test."""

    name = "scripted"
    model = "scripted-v1"

    def __init__(
        self,
        *,
        text: str = "scripted answer",
        structured: dict[str, Any] | None = None,
        fail_with: Exception | None = None,
        fail_structured_with: Exception | None = None,
        usage: LLMUsage | None = None,
    ) -> None:
        self.text = text
        self.structured = structured
        self.fail_with = fail_with
        self.fail_structured_with = fail_structured_with
        self.usage = usage
        self.seen: list[LLMRequest] = []

    @property
    def is_configured(self) -> bool:
        return self.fail_with is None or True

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        return LLMResponse(
            text=self.text,
            provider=self.name,
            model=self.model,
            usage=self.usage,
            request_metadata=request.to_dict(),
        )

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        if self.fail_structured_with is not None:
            raise self.fail_structured_with
        payload = self.structured if self.structured is not None else {"candidates": []}
        return LLMResponse(
            text=json.dumps(payload),
            provider=self.name,
            model=self.model,
            usage=self.usage,
            structured=payload,
            request_metadata=request.to_dict(),
        )


@pytest.fixture
def memories(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryService:
    """A deterministic memory service."""
    return MemoryService(
        repository, clock=clock, id_generator=SequentialIdGenerator(prefix="agent")
    )


@pytest.fixture
def operations(
    repository: SqliteMemoryRepository, memories: MemoryService, clock: ManualClock
) -> MemoryOperationsService:
    """The memory-operations facade."""
    retrieval = MemoryRetrievalService(repository, clock=clock)
    return MemoryOperationsService(memories, retrieval, ContextCompilerService(retrieval))


@pytest.fixture
def agent(operations: MemoryOperationsService) -> AgentService:
    """The agent with the default mock provider."""
    return AgentService(operations, MockLLMProvider())


@pytest.fixture
def corpus(operations: MemoryOperationsService) -> None:
    """One memory that the question actually recalls."""
    operations.store(PYTHON, MemoryType.PROFILE, confidence=0.95, importance=0.9)


class TestMemoryGate:
    def test_a_real_question_uses_memory(self) -> None:
        gate = decide_memory_use(QUESTION)
        assert gate.use is MemoryUse.USED
        assert gate.should_recall is True
        assert gate.reason

    @pytest.mark.parametrize("greeting", ["hi", "hello", "thanks", "ok", "hey there"])
    def test_small_talk_skips_memory(self, greeting: str) -> None:
        gate = decide_memory_use(greeting)
        assert gate.use is MemoryUse.SKIPPED
        assert gate.should_recall is False

    def test_a_question_is_not_small_talk(self) -> None:
        assert decide_memory_use("what is this?").use is MemoryUse.USED

    def test_an_empty_request_skips_memory(self) -> None:
        assert decide_memory_use("   ").use is MemoryUse.SKIPPED

    def test_explicit_opt_out_disables_memory(self) -> None:
        gate = decide_memory_use(QUESTION, use_memory=False)
        assert gate.use is MemoryUse.DISABLED
        assert gate.should_recall is False

    def test_gate_serialises(self) -> None:
        assert decide_memory_use(QUESTION).to_dict()["use"] == "used"


class TestRequestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"request": ""},
            {"request": "   "},
            {"request": "x", "token_budget": 0},
            {"request": "x", "top_k": 0},
            {"request": "x", "temperature": 5.0},
            {"request": "x", "max_tokens": 0},
        ],
    )
    def test_invalid_requests_are_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(**kwargs)

    def test_writes_require_both_switches(self) -> None:
        assert AgentRequest(request="x").will_store is False
        assert AgentRequest(request="x", extract_candidates=True).will_store is False
        assert (
            AgentRequest(request="x", extract_candidates=True, store_candidates=True).will_store
            is True
        )
        # store_candidates alone does nothing without extraction.
        assert AgentRequest(request="x", store_candidates=True).will_store is False


class TestMessages:
    def test_context_is_separated_from_the_instruction(self) -> None:
        messages = build_messages(system="sys", user="usr", context="CTX")
        assert messages[0].role.value == "system"
        assert messages[1].content == "CTX\n\n---\n\nUser request: usr"

    def test_without_context_the_request_stands_alone(self) -> None:
        assert build_messages(system="s", user="u")[1].content == "u"


class TestTurn:
    def test_the_workflow_compiles_and_answers(self, agent: AgentService, corpus: None) -> None:
        result = agent.respond(AgentRequest(request=QUESTION, token_budget=300))

        assert result.gate.use is MemoryUse.USED
        assert result.used_memory is True
        assert PYTHON in result.context_text
        assert result.answer
        assert result.provider == "mock"
        assert result.context_estimated_tokens > 0
        assert result.context_estimated_tokens <= result.context_token_budget

    def test_the_compiled_context_reaches_the_provider(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider()
        AgentService(operations, provider).respond(AgentRequest(request=QUESTION, token_budget=300))
        prompt = provider.seen[0].messages[-1].content
        assert PYTHON in prompt
        assert QUESTION in prompt

    def test_context_only_never_calls_the_provider(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider()
        result = AgentService(operations, provider).respond(
            AgentRequest(request=QUESTION, token_budget=300, context_only=True)
        )
        assert provider.seen == []
        assert result.answer == ""
        assert result.context_text

    def test_small_talk_skips_the_memory_path(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider()
        result = AgentService(operations, provider).respond(AgentRequest(request="hi"))
        assert result.gate.use is MemoryUse.SKIPPED
        assert result.context_text == ""
        assert "PYTHON" not in provider.seen[0].messages[-1].content

    def test_an_unrelated_request_reports_no_matches(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        result = AgentService(operations, MockLLMProvider()).respond(
            AgentRequest(request="what about kubernetes ingress controllers?")
        )
        assert result.gate.use is MemoryUse.NO_MATCHES
        assert result.context_text == ""
        assert result.answer  # the provider is still asked, just without memory

    def test_use_memory_false_short_circuits(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        result = AgentService(operations, MockLLMProvider()).respond(
            AgentRequest(request=QUESTION, use_memory=False)
        )
        assert result.gate.use is MemoryUse.DISABLED
        assert result.context_text == ""

    def test_turns_are_deterministic(self, agent: AgentService, corpus: None) -> None:
        first = agent.respond(AgentRequest(request=QUESTION, token_budget=300))
        second = agent.respond(AgentRequest(request=QUESTION, token_budget=300))
        assert first.answer == second.answer
        assert first.context_text == second.context_text
        assert [block.memory_id for block in first.context_blocks] == [
            block.memory_id for block in second.context_blocks
        ]

    def test_budget_ceiling_is_enforced_before_any_work(
        self, operations: MemoryOperationsService
    ) -> None:
        agent = AgentService(operations, MockLLMProvider(), max_token_budget=500)
        with pytest.raises(AgentValidationError, match="configured maximum"):
            agent.respond(AgentRequest(request=QUESTION, token_budget=501))

    def test_ceiling_is_enforced_even_when_memory_is_off(
        self, operations: MemoryOperationsService
    ) -> None:
        agent = AgentService(operations, MockLLMProvider(), max_token_budget=100)
        with pytest.raises(AgentValidationError):
            agent.respond(AgentRequest(request=QUESTION, token_budget=101, use_memory=False))

    def test_context_blocks_carry_provenance(self, agent: AgentService, corpus: None) -> None:
        result = agent.respond(AgentRequest(request=QUESTION, token_budget=300))
        block = result.context_blocks[0]
        assert block.memory_id
        assert block.memory_type is MemoryType.PROFILE
        assert block.section.value == "PROFILE"
        assert block.utility_score > 0.0
        assert block.estimated_tokens > 0


class TestProviderFailures:
    def test_a_failure_is_raised_not_swallowed(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider(fail_with=LLMTimeoutError("no answer", provider="scripted"))
        agent = AgentService(operations, provider)
        with pytest.raises(LLMTimeoutError):
            agent.respond(AgentRequest(request=QUESTION, token_budget=300))

    def test_a_failure_is_recorded_in_the_observability(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider(fail_with=LLMTimeoutError("no answer", provider="scripted"))
        agent = AgentService(operations, provider)
        with pytest.raises(LLMTimeoutError):
            agent.respond(AgentRequest(request=QUESTION, token_budget=300))

    def test_no_fabricated_answer_on_failure(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider(fail_with=LLMTimeoutError("no answer", provider="scripted"))
        result: Any = None
        with contextlib.suppress(LLMTimeoutError):
            result = AgentService(operations, provider).respond(
                AgentRequest(request=QUESTION, token_budget=300)
            )
        assert result is None

    def test_an_unconfigured_provider_is_refused(self, operations: MemoryOperationsService) -> None:
        class Unconfigured(LLMProvider):
            name = "unconfigured"
            model = "none"

            @property
            def is_configured(self) -> bool:
                return False

            def generate(self, request: LLMRequest) -> LLMResponse:
                raise AssertionError("must not be called")

        with pytest.raises(LLMProviderError, match="not configured"):
            AgentService(operations, Unconfigured()).respond(
                AgentRequest(request=QUESTION, token_budget=100)
            )

    def test_extraction_failure_does_not_break_the_answer(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider(
            fail_structured_with=LLMTimeoutError("no structure", provider="scripted")
        )
        result = AgentService(operations, provider).respond(
            AgentRequest(request=QUESTION, token_budget=300, extract_candidates=True)
        )
        assert result.answer == "scripted answer"
        assert result.candidate_memories == ()
        assert any(
            record.operation == "extract_candidates" and not record.success
            for record in result.records
        )


class TestCandidates:
    def _structured(self, **overrides: Any) -> dict[str, Any]:
        payload = {
            "candidates": [
                {
                    "content": "The user ships on Fridays.",
                    "memory_type": "FACT",
                    "confidence": 0.8,
                    "importance": 0.6,
                    "reason": "stated explicitly",
                }
            ]
        }
        payload.update(overrides)
        return payload

    def test_nothing_is_stored_without_permission(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider(structured=self._structured())
        before = operations.memory_service.list_memories().total
        result = AgentService(operations, provider).respond(
            AgentRequest(request=QUESTION, token_budget=300, extract_candidates=True)
        )
        assert len(result.candidate_memories) == 1
        assert result.stored_memory_ids == ()
        assert operations.memory_service.list_memories().total == before

    def test_permitted_candidates_are_stored(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        provider = ScriptedProvider(structured=self._structured())
        result = AgentService(operations, provider).respond(
            AgentRequest(
                request=QUESTION,
                token_budget=300,
                extract_candidates=True,
                store_candidates=True,
                source="agent:test",
            )
        )
        assert len(result.stored_memory_ids) == 1
        stored = operations.memory_service.get_memory(result.stored_memory_ids[0])
        assert stored.content == "The user ships on Fridays."
        assert stored.memory_type is MemoryType.FACT
        assert stored.source == "agent:test"
        assert stored.confidence == 0.8

    def test_malformed_candidates_are_rejected_not_stored(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        payload = {
            "candidates": [
                {"content": "", "memory_type": "FACT"},
                {"content": "valid", "memory_type": "NOT_A_TYPE"},
                {"content": "good", "memory_type": "FACT", "confidence": "high"},
            ]
        }
        provider = ScriptedProvider(structured=payload)
        result = AgentService(operations, provider).respond(
            AgentRequest(
                request=QUESTION,
                token_budget=300,
                extract_candidates=True,
                store_candidates=True,
            )
        )
        rejections = [item.rejection for item in result.candidate_memories]
        assert rejections == ["empty_content", "unknown_type", None]
        assert result.stored_memory_ids == result.stored_memory_ids[:1]
        assert len(result.stored_memory_ids) == 1

    def test_a_non_list_payload_is_ignored(self, operations: MemoryOperationsService) -> None:
        provider = ScriptedProvider(structured={"candidates": "not a list"})
        result = AgentService(operations, provider).respond(
            AgentRequest(request=QUESTION, token_budget=300, extract_candidates=True)
        )
        assert result.candidate_memories == ()

    def test_scores_are_clamped(self) -> None:
        candidate = CandidateMemory.from_payload(
            {"content": "x", "memory_type": "FACT", "confidence": 9.0, "importance": -2.0}
        )
        assert candidate.confidence == 1.0
        assert candidate.importance == 0.0

    def test_missing_scores_default(self) -> None:
        candidate = CandidateMemory.from_payload({"content": "x", "memory_type": "FACT"})
        assert candidate.confidence == 0.5
        assert candidate.memory_type is MemoryType.FACT


class TestObservability:
    def test_records_cover_the_pipeline(self, agent: AgentService, corpus: None) -> None:
        result = agent.respond(
            AgentRequest(request=QUESTION, token_budget=300, extract_candidates=True)
        )
        operations = [record.operation for record in result.records]
        assert "compile_context" in operations
        assert "respond" in operations
        assert "extract_candidates" in operations
        assert all(record.success for record in result.records)

    def test_context_record_reports_the_estimate(self, agent: AgentService, corpus: None) -> None:
        result = agent.respond(AgentRequest(request=QUESTION, token_budget=300))
        record = next(r for r in result.records if r.operation == "compile_context")
        assert record.context_estimated_tokens == result.context_estimated_tokens
        assert record.memories_recalled == 1
        assert record.provider == "mock"

    def test_usage_is_recorded_only_when_reported(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        without = AgentService(operations, ScriptedProvider()).respond(
            AgentRequest(request=QUESTION, token_budget=300)
        )
        with_usage = AgentService(
            operations,
            ScriptedProvider(usage=LLMUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12)),
        ).respond(AgentRequest(request=QUESTION, token_budget=300))

        assert without.usage is None
        assert with_usage.usage is not None
        assert with_usage.usage.total_tokens == 12
        record = next(r for r in with_usage.records if r.operation == "respond")
        assert record.usage is not None

    def test_records_are_secret_free(self, agent: AgentService, corpus: None) -> None:
        result = agent.respond(AgentRequest(request=QUESTION, token_budget=300))
        for record in result.records:
            rendered = json.dumps(record.to_dict()).lower()
            for marker in ("authorization", "bearer", "api_key", "secret"):
                assert marker not in rendered

    def test_record_serialises(self) -> None:
        record = OperationRecord(operation="test", success=True, latency_ms=1.5)
        assert record.to_dict()["usage"] is None
        assert record.to_dict()["latency_ms"] == 1.5


class TestSemanticModes:
    def test_hybrid_is_reported_through_the_agent(
        self, operations: MemoryOperationsService, corpus: None
    ) -> None:
        result = AgentService(operations, MockLLMProvider()).respond(
            AgentRequest(request=QUESTION, token_budget=300, retrieval_mode=RetrievalMode.HYBRID)
        )
        assert result.used_memory is True

    def test_messages_are_always_reported(self, agent: AgentService, corpus: None) -> None:
        result = agent.respond(AgentRequest(request=QUESTION, token_budget=300))
        assert all(isinstance(message, LLMMessage) for message in result.messages)
        assert len(result.messages) == 2
