"""M5 coverage tests for paths the main suites do not reach.

These are small, targeted checks for defensive branches -- error shapes,
optional helpers and the seams between the SDK and the domain. They exist to
keep the new modules fully covered, not to restate behaviour proved elsewhere.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.domain.agent import (
    AgentRequest,
    CandidateMemory,
    ContextBlockSummary,
    OperationRecord,
    messages_as_dicts,
)
from app.domain.errors import (
    AgentValidationError,
    LLMNotConfiguredError,
    LLMProtocolError,
    LLMProviderError,
)
from app.domain.ids import SequentialIdGenerator
from app.domain.llm import LLMMessage, LLMRequest, LLMUsage, redact_mapping, safe_endpoint
from app.domain.models import MemoryStatus, MemoryType
from app.domain.time import ManualClock
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.custom_provider import CustomLLMConfig, CustomLLMProvider
from app.services.llm.mock_provider import MockLLMProvider, candidate_payload
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.repository import SqliteMemoryRepository


@pytest.fixture
def operations(repository: SqliteMemoryRepository, clock: ManualClock) -> MemoryOperationsService:
    """A facade over a fresh, deterministic stack."""
    memories = MemoryService(
        repository, clock=clock, id_generator=SequentialIdGenerator(prefix="edge")
    )
    retrieval = MemoryRetrievalService(repository, clock=clock)
    return MemoryOperationsService(memories, retrieval, ContextCompilerService(retrieval))


class TestAgentDomainHelpers:
    def test_gate_serialises_every_use(self) -> None:
        from app.domain.agent import MemoryGate, MemoryUse

        for use in MemoryUse:
            assert MemoryGate(use, "because").to_dict() == {
                "use": use.value,
                "reason": "because",
            }

    def test_messages_as_dicts_round_trips(self) -> None:
        messages = (LLMMessage.system("s"), LLMMessage.user("u"))
        assert messages_as_dicts(messages) == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]

    def test_an_unknown_retrieval_mode_is_rejected(self) -> None:
        with pytest.raises(AgentValidationError, match="unknown retrieval mode"):
            AgentRequest(request="x", retrieval_mode="telepathy")  # type: ignore[arg-type]

    def test_block_summary_is_a_plain_value(self) -> None:
        summary = ContextBlockSummary(
            memory_id="m",
            memory_type=MemoryType.FACT,
            section=_facts_section(),
            order=0,
            utility_score=0.5,
            retrieval_score=0.4,
            estimated_tokens=12,
        )
        assert summary.memory_type is MemoryType.FACT
        assert summary.estimated_tokens == 12

    def test_operation_record_defaults_are_none(self) -> None:
        record = OperationRecord(operation="op", success=True, latency_ms=0.0)
        body = record.to_dict()
        assert body["usage"] is None
        assert body["memories_stored"] is None
        assert body["error_kind"] is None

    def test_candidate_to_dict_reports_a_rejection(self) -> None:
        candidate = CandidateMemory("x", MemoryType.FACT, 0.5, 0.5, "why", "unknown_type")
        assert candidate.to_dict()["rejection"] == "unknown_type"
        assert candidate.is_valid is False

    def test_candidate_rejects_non_boolean_confidence(self) -> None:
        candidate = CandidateMemory.from_payload(
            {"content": "x", "memory_type": "FACT", "confidence": True}
        )
        assert candidate.confidence == 0.5

    def test_candidate_defaults_the_memory_type(self) -> None:
        candidate = CandidateMemory.from_payload({"content": "x"})
        assert candidate.memory_type is MemoryType.OBSERVATION

    def test_a_blank_reason_becomes_empty(self) -> None:
        assert CandidateMemory.from_payload({"content": "x", "reason": 7}).reason == ""


def _facts_section() -> Any:
    from app.domain.context import ContextSection

    return ContextSection.FACTS


class TestLLMDomainHelpers:
    def test_non_string_values_survive_redaction(self) -> None:
        assert redact_mapping({"count": 3, "flag": True}) == {"count": 3, "flag": True}

    def test_nested_sequences_are_walked(self) -> None:
        assert redact_mapping({"values": [1, 2]}) == {"values": [1, 2]}

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("https://host/v1", "https://host/v1"),
            ("https://user:pw@host/v1", "https://host/v1"),
            ("host-without-scheme@x", "host-without-scheme@x"),
        ],
    )
    def test_safe_endpoint_handles_every_shape(self, base_url: str, expected: str) -> None:
        assert safe_endpoint(base_url) == expected

    def test_usage_from_mapping_keeps_the_provider_flag(self) -> None:
        assert LLMUsage.from_mapping({"total_tokens": 5}).reported_by == "provider"


class TestMockProviderHelpers:
    def test_candidate_payload_is_canonical(self) -> None:
        assert candidate_payload([{"content": "x"}]) == {"candidates": [{"content": "x"}]}

    def test_no_user_message_yields_an_empty_reply(self) -> None:
        response = MockLLMProvider().generate(
            LLMRequest(messages=(LLMMessage.system("only a system prompt"),))
        )
        assert response.text.endswith("last_user_message: ")

    def test_usage_report_is_always_none(self) -> None:
        assert MockLLMProvider().as_usage_report() is None


class TestCustomProviderHelpers:
    def test_negative_backoff_is_rejected(self) -> None:
        with pytest.raises(LLMNotConfiguredError, match="backoff_seconds"):
            CustomLLMConfig(backoff_seconds=-1.0)

    def test_endpoint_is_exposed(self) -> None:
        config = CustomLLMConfig(base_url="https://host/v1", model="m")
        assert config.endpoint == "https://host/v1/chat/completions"

    def test_a_non_mapping_choice_is_a_protocol_error(self) -> None:
        provider, _ = _provider({"choices": ["not a mapping"]})
        with pytest.raises(LLMProtocolError, match="malformed choice"):
            provider.generate(_request())

    def test_a_missing_message_is_a_protocol_error(self) -> None:
        provider, _ = _provider({"choices": [{"finish_reason": "stop"}]})
        with pytest.raises(LLMProtocolError, match="no message"):
            provider.generate(_request())

    def test_non_text_content_is_a_protocol_error(self) -> None:
        provider, _ = _provider({"choices": [{"message": {"content": 7}}]})
        with pytest.raises(LLMProtocolError, match="expected text"):
            provider.generate(_request())

    def test_a_finish_reason_of_the_wrong_type_is_ignored(self) -> None:
        provider, _ = _provider({"choices": [{"message": {"content": "hi"}, "finish_reason": 7}]})
        assert provider.generate(_request()).finish_reason is None

    def test_a_non_string_id_falls_back_to_empty(self) -> None:
        provider, _ = _provider({"id": 7, "choices": [{"message": {"content": "hi"}}]})
        assert provider.generate(_request()).response_id == ""

    def test_a_non_mapping_error_body_is_summarised(self) -> None:
        provider, _ = _provider_error([1, 2, 3], status=400)
        with pytest.raises(LLMProviderError):
            provider.generate(_request())

    def test_a_top_level_message_is_used_as_detail(self) -> None:
        provider, _ = _provider_error({"message": "flat detail"}, status=400)
        with pytest.raises(LLMProviderError, match="flat detail"):
            provider.generate(_request())

    def test_a_non_json_error_body_is_used_as_detail(self) -> None:
        provider, _ = _provider_error_text("plain text failure")
        with pytest.raises(LLMProviderError, match="plain text failure"):
            provider.generate(_request())

    def test_closing_an_owned_client_is_safe(self) -> None:
        provider = CustomLLMProvider(CustomLLMConfig(base_url="https://host/v1", model="m"))
        provider.close()
        provider.close()


def _request() -> LLMRequest:
    return LLMRequest(messages=(LLMMessage.user("hi"),))


def _provider(body: dict[str, Any]) -> tuple[CustomLLMProvider, httpx.Client]:
    return _respond(httpx.Response(200, json=body))


def _provider_error(body: Any, *, status: int) -> tuple[CustomLLMProvider, httpx.Client]:
    return _respond(httpx.Response(status, json=body))


def _provider_error_text(text: str) -> tuple[CustomLLMProvider, httpx.Client]:
    return _respond(httpx.Response(400, text=text))


def _respond(response: httpx.Response) -> tuple[CustomLLMProvider, httpx.Client]:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: response))
    config = CustomLLMConfig(base_url="https://host/v1", model="m", api_key="k", max_retries=0)
    return CustomLLMProvider(config, client=client, sleep=lambda _: None), client


class _NonDictStructuredProvider(MockLLMProvider):
    """A provider whose structured payload is not a mapping at all."""

    def generate_structured(self, request: LLMRequest) -> Any:
        from app.domain.llm import LLMResponse

        return LLMResponse(
            text="[]",
            provider=self.name,
            model=self.model,
            structured=[1, 2],  # type: ignore[arg-type]
        )


class TestAgentServiceHelpers:
    def test_the_provider_is_exposed(self, operations: MemoryOperationsService) -> None:
        provider = MockLLMProvider()
        agent = AgentService(operations, provider)
        assert agent.provider is provider
        assert agent.operations is operations

    def test_a_non_dict_payload_yields_no_candidates(
        self, operations: MemoryOperationsService
    ) -> None:
        operations.store("User prefers Python for backend work.", MemoryType.PROFILE)
        result = AgentService(operations, _NonDictStructuredProvider()).respond(
            AgentRequest(
                request="Which backend language should I use?",
                token_budget=200,
                extract_candidates=True,
            )
        )
        assert result.answer
        assert result.candidate_memories == ()


class TestMemoryOperationsHelpers:
    def test_an_empty_corpus_is_handled(self, operations: MemoryOperationsService) -> None:
        # Nothing stored yet: the search is empty and the listing is empty, so
        # the deny-list branch is skipped without ever being reached.
        result = operations.search("python", top_k=1)
        assert result.candidates_considered == 0
        assert operations.memory_service.list_memories().total == 0

    def test_update_rejects_an_illegal_transition(
        self, operations: MemoryOperationsService
    ) -> None:
        from app.domain.errors import InvalidMemoryStateError

        stored = operations.store("User prefers Python.", MemoryType.FACT)
        with pytest.raises(InvalidMemoryStateError):
            operations.update(stored.id, status=MemoryStatus.DELETED)


class TestDemoJsonHelpers:
    def test_candidate_payload_json_is_stable(self) -> None:
        assert json.dumps(candidate_payload([])) == '{"candidates": []}'
