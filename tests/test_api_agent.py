"""M5 REST tests: the agent endpoint, ``remember``, errors and secret safety."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.errors import LLMTimeoutError
from app.main import create_app
from app.observability.instrumentation import TracedLLMProvider, TracedMemoryService
from app.services.agent_service import AgentService
from app.services.llm.mock_provider import MockLLMProvider

AGENT_URL = "/api/v1/agent/respond"
REMEMBER_URL = "/api/v1/memories/remember"

QUESTION = "Which backend language should I use for the MemForge service?"
SECRET = "sk-api-key-must-never-appear-1234567890"


def seed(client: TestClient, count: int = 1) -> list[str]:
    """Create memories that the question can recall."""
    created: list[str] = []
    for index in range(count):
        response = client.post(
            "/api/v1/memories",
            json={
                "content": f"User prefers Python for backend work (note {index})."
                if index
                else "User prefers Python for backend work.",
                "memory_type": "PROFILE",
                "source": "user:test",
                "confidence": 0.95,
                "importance": 0.9,
            },
        )
        assert response.status_code == 201, response.text
        created.append(response.json()["id"])
    return created


class TestRememberEndpoint:
    def test_stores_a_memory(self, client: TestClient) -> None:
        response = client.post(
            REMEMBER_URL,
            json={"content": "User prefers Python.", "memory_type": "PREFERENCE"},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["memory"]["content"] == "User prefers Python."
        assert body["memory"]["version"] == 1
        assert body["duplicate_count"] == 0

    def test_reports_near_duplicates(self, client: TestClient) -> None:
        client.post(
            REMEMBER_URL,
            json={"content": "User prefers Python.", "memory_type": "PREFERENCE"},
        )
        response = client.post(
            REMEMBER_URL,
            json={
                "content": "The user prefers python for backend development work",
                "memory_type": "OBSERVATION",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["duplicate_count"] >= 1
        assert body["duplicates"][0]["similarity"] > 0.0

    def test_duplicate_detection_can_be_skipped(self, client: TestClient) -> None:
        client.post(
            REMEMBER_URL,
            json={"content": "User prefers Python.", "memory_type": "PREFERENCE"},
        )
        response = client.post(
            REMEMBER_URL,
            json={
                "content": "The user prefers python for backend development work",
                "memory_type": "OBSERVATION",
                "check_duplicates": False,
            },
        )
        assert response.json()["duplicate_count"] == 0
        assert response.json()["duplicates"] == []

    def test_provenance_is_kept(self, client: TestClient) -> None:
        response = client.post(
            REMEMBER_URL,
            json={
                "content": "User ships on Fridays.",
                "memory_type": "FACT",
                "source": "agent:test",
                "confidence": 0.8,
                "importance": 0.4,
            },
        )
        memory = response.json()["memory"]
        assert memory["source"] == "agent:test"
        assert memory["confidence"] == 0.8
        assert memory["importance"] == 0.4

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"content": ""},
            {"content": "x"},
            {"content": "x", "memory_type": "NOPE"},
            {"content": "x", "memory_type": "FACT", "confidence": 2.0},
            {"content": "x", "memory_type": "FACT", "unexpected": True},
        ],
    )
    def test_malformed_requests_use_the_error_envelope(
        self, client: TestClient, body: dict[str, Any]
    ) -> None:
        response = client.post(REMEMBER_URL, json=body)
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


class TestAgentRespond:
    def test_answers_with_compiled_memory(self, client: TestClient) -> None:
        seed(client)
        response = client.post(AGENT_URL, json={"request": QUESTION, "token_budget": 300})
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["provider"] == "mock"
        assert body["memory_used"] is True
        assert body["memory_gate"]["use"] == "used"
        assert "PROFILE" in body["context_text"]
        assert body["answer"]
        assert body["context_estimated_tokens"] <= body["context_token_budget"]
        assert body["context_blocks"][0]["memory_type"] == "PROFILE"
        assert body["messages"][0]["role"] == "system"

    def test_usage_is_null_for_the_mock(self, client: TestClient) -> None:
        seed(client)
        body = client.post(AGENT_URL, json={"request": QUESTION}).json()
        assert body["usage"] is None

    def test_context_blocks_carry_provenance(self, client: TestClient) -> None:
        created = seed(client)
        body = client.post(AGENT_URL, json={"request": QUESTION, "token_budget": 300}).json()
        block = body["context_blocks"][0]
        assert block["memory_id"] == created[0]
        assert block["section"] == "PROFILE"
        assert block["utility_score"] > 0.0
        assert block["estimated_tokens"] > 0

    def test_operations_are_reported(self, client: TestClient) -> None:
        seed(client)
        body = client.post(AGENT_URL, json={"request": QUESTION, "token_budget": 300}).json()
        operations = {record["operation"] for record in body["operations"]}
        assert "compile_context" in operations
        assert "respond" in operations
        assert all(record["success"] for record in body["operations"])
        record = next(r for r in body["operations"] if r["operation"] == "respond")
        assert record["provider"] == "mock"
        assert record["usage"] is None

    def test_context_only_skips_the_provider(self, client: TestClient) -> None:
        seed(client)
        body = client.post(AGENT_URL, json={"request": QUESTION, "context_only": True}).json()
        assert body["answer"] == ""
        assert body["context_text"].strip()
        assert all(record["operation"] != "respond" for record in body["operations"])

    def test_small_talk_skips_memory(self, client: TestClient) -> None:
        seed(client)
        body = client.post(AGENT_URL, json={"request": "hi"}).json()
        assert body["memory_gate"]["use"] == "skipped"
        assert body["context_text"] == ""
        assert body["answer"]

    def test_use_memory_false(self, client: TestClient) -> None:
        seed(client)
        body = client.post(AGENT_URL, json={"request": QUESTION, "use_memory": False}).json()
        assert body["memory_gate"]["use"] == "disabled"
        assert body["context_text"] == ""

    def test_unrelated_request_reports_no_matches(self, client: TestClient) -> None:
        seed(client)
        body = client.post(
            AGENT_URL, json={"request": "what about kubernetes ingress controllers?"}
        ).json()
        assert body["memory_gate"]["use"] == "no_matches"
        assert body["memory_used"] is False

    def test_empty_corpus_still_answers(self, client: TestClient) -> None:
        body = client.post(AGENT_URL, json={"request": QUESTION}).json()
        assert body["answer"]
        assert body["context_text"] == ""
        assert body["memory_gate"]["use"] == "no_matches"


class TestCandidateHandling:
    REQUEST = "What should I do?\nremember: [FACT] The user ships on Fridays."

    def test_candidates_are_reported_but_not_stored(self, client: TestClient) -> None:
        seed(client)
        before = client.get("/api/v1/memories", params={"limit": 100}).json()["total"]
        body = client.post(
            AGENT_URL,
            json={
                "request": self.REQUEST,
                "extract_candidates": True,
                "store_candidates": False,
            },
        ).json()
        after = client.get("/api/v1/memories", params={"limit": 100}).json()["total"]

        assert body["candidate_memories"][0]["content"] == "The user ships on Fridays."
        assert body["candidate_memories"][0]["memory_type"] == "FACT"
        assert body["stored_memory_ids"] == []
        assert after == before

    def test_storage_requires_explicit_permission(self, client: TestClient) -> None:
        body = client.post(
            AGENT_URL,
            json={
                "request": self.REQUEST,
                "extract_candidates": True,
                "store_candidates": True,
                "source": "agent:api",
            },
        ).json()
        assert len(body["stored_memory_ids"]) == 1
        stored = client.get(f"/api/v1/memories/{body['stored_memory_ids'][0]}").json()
        assert stored["source"] == "agent:api"
        assert stored["content"] == "The user ships on Fridays."

    def test_candidates_without_extraction_are_not_returned(self, client: TestClient) -> None:
        seed(client)
        body = client.post(AGENT_URL, json={"request": self.REQUEST}).json()
        assert body["candidate_memories"] == []


class TestBudgetAndValidation:
    @pytest.mark.parametrize("budget", [1, 20, 80, 300, 2000])
    def test_the_budget_is_never_exceeded(self, client: TestClient, budget: int) -> None:
        seed(client, count=2)
        body = client.post(AGENT_URL, json={"request": QUESTION, "token_budget": budget}).json()
        assert body["context_estimated_tokens"] <= budget

    def test_a_budget_above_the_ceiling_is_refused(self, client: TestClient) -> None:
        response = client.post(AGENT_URL, json={"request": QUESTION, "token_budget": 16001})
        assert response.status_code == 400
        envelope = response.json()["error"]
        assert envelope["code"] == "AGENT_VALIDATION_ERROR"
        assert "16000" in envelope["message"]

    @pytest.mark.parametrize(
        ("body", "status", "code"),
        [
            ({}, 422, "VALIDATION_ERROR"),
            ({"request": ""}, 422, "VALIDATION_ERROR"),
            ({"request": "x", "token_budget": 0}, 422, "VALIDATION_ERROR"),
            ({"request": "x", "token_budget": -1}, 422, "VALIDATION_ERROR"),
            ({"request": "x", "top_k": 0}, 422, "VALIDATION_ERROR"),
            ({"request": "x", "retrieval_mode": "telepathy"}, 422, "VALIDATION_ERROR"),
            ({"request": "x", "temperature": 9.0}, 422, "VALIDATION_ERROR"),
            ({"request": "x", "unknown": True}, 422, "VALIDATION_ERROR"),
        ],
    )
    def test_malformed_requests_use_the_error_envelope(
        self, client: TestClient, body: dict[str, Any], status: int, code: str
    ) -> None:
        response = client.post(AGENT_URL, json=body)
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code


class TestProviderFailures:
    def test_a_provider_timeout_is_an_explicit_failure(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(self: AgentService, request: object) -> object:
            raise LLMTimeoutError("the provider did not answer", provider="custom")

        monkeypatch.setattr(AgentService, "respond", explode)
        with TestClient(create_app(settings)) as client:
            response = client.post(AGENT_URL, json={"request": QUESTION})

        assert response.status_code == 504
        envelope = response.json()["error"]
        assert envelope["code"] == "LLM_TIMEOUT"
        assert envelope["details"]["kind"] == "timeout"
        assert envelope["details"]["retryable"] is True

    def test_no_answer_is_invented_on_failure(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(self: AgentService, request: object) -> object:
            raise LLMTimeoutError("no answer", provider="custom")

        monkeypatch.setattr(AgentService, "respond", explode)
        with TestClient(create_app(settings)) as client:
            response = client.post(AGENT_URL, json={"request": QUESTION})

        body = response.json()
        # The envelope carries an error and nothing else: no answer field, no
        # fabricated text, no usage numbers.
        assert set(body) == {"error"}
        assert set(body["error"]["details"]) == {"kind", "provider", "retryable"}
        assert SECRET not in response.text
        assert "mock" not in json.dumps(body).lower()


class TestSecretSafety:
    def test_the_response_schema_exposes_no_credential_field(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        fields = " ".join(schema["components"]["schemas"]["AgentRespondResponse"]["properties"])
        for forbidden in ("api_key", "authorization", "credential", "secret"):
            assert forbidden not in fields.lower()

    def test_a_configured_key_never_appears_in_a_response(self, tmp_path: Any) -> None:
        settings = Settings(
            database_path=tmp_path / "secrets.sqlite",
            llm_provider="custom",
            llm_base_url="https://llm.example.invalid/v1",
            llm_model="test-model",
            llm_api_key=SECRET,
        )
        with TestClient(create_app(settings)) as client:
            client.post(
                "/api/v1/memories",
                json={"content": "User prefers Python.", "memory_type": "PROFILE"},
            )
            agent = client.post(AGENT_URL, json={"request": QUESTION, "token_budget": 300})
            openapi = client.get("/openapi.json")

        for payload in (agent.text, openapi.text):
            assert SECRET not in payload

    def test_settings_repr_hides_the_key(self) -> None:
        settings = Settings(llm_api_key=SECRET)
        assert SECRET not in repr(settings)
        assert "llm_api_key" not in repr(settings)


class TestDeterminism:
    def test_repeated_turns_are_identical(self, client: TestClient) -> None:
        seed(client)
        payload = {"request": QUESTION, "token_budget": 300}
        first = client.post(AGENT_URL, json=payload)
        second = client.post(AGENT_URL, json=payload)
        first_body = first.json()
        second_body = second.json()
        # Latency is a measurement, not a contract.
        for body in (first_body, second_body):
            for record in body["operations"]:
                record["latency_ms"] = 0.0
        assert first_body == second_body

    def test_the_whole_payload_is_reproducible(self, client: TestClient) -> None:
        seed(client)
        payload = {"request": QUESTION, "token_budget": 300}
        first = client.post(AGENT_URL, json=payload).text
        second = client.post(AGENT_URL, json=payload).text
        first_body = json.loads(first)
        second_body = json.loads(second)
        for body in (first_body, second_body):
            for record in body["operations"]:
                record["latency_ms"] = 0.0
        assert json.dumps(first_body, sort_keys=True) == json.dumps(second_body, sort_keys=True)


class TestCompatibility:
    def test_m1_to_m4_endpoints_are_untouched(self, client: TestClient) -> None:
        created = seed(client)
        assert client.get("/health").status_code == 200
        assert client.get(f"/api/v1/memories/{created[0]}").status_code == 200
        assert client.get(f"/api/v1/memories/{created[0]}/history").status_code == 200
        assert client.get(f"/api/v1/memories/{created[0]}/intelligence").status_code == 200
        assert client.post("/api/v1/memories/search", json={"query": "python"}).status_code == 200
        assert (
            client.post(
                "/api/v1/context/compile", json={"query": "python", "token_budget": 200}
            ).status_code
            == 200
        )

    def test_the_agent_turn_does_not_mutate_memories(self, client: TestClient) -> None:
        created = seed(client)
        before = client.get(f"/api/v1/memories/{created[0]}").json()
        client.post(AGENT_URL, json={"request": QUESTION, "token_budget": 300})
        after = client.get(f"/api/v1/memories/{created[0]}").json()
        assert before == after

    def test_openapi_advertises_both_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert AGENT_URL in paths
        assert REMEMBER_URL in paths
        assert paths[AGENT_URL]["post"]["tags"] == ["agent"]

    def test_services_are_shared(self, settings: Settings) -> None:
        app = create_app(settings)
        with TestClient(app) as client:
            assert isinstance(app.state.agent_service, AgentService)
            # M6 wraps the provider and the memory layer in tracing decorators.
            # Each wrapper is a transparent decorator around exactly one shared
            # instance, so the "one instance per app" guarantee is unchanged.
            assert isinstance(app.state.llm_provider, TracedLLMProvider)
            assert isinstance(app.state.llm_provider.inner, MockLLMProvider)
            assert app.state.agent_service.operations is app.state.memory_operations
            memories = cast(TracedMemoryService, app.state.memory_operations.memory_service)
            assert memories.inner is app.state.memory_service
            assert client.post(AGENT_URL, json={"request": QUESTION}).status_code == 200

    def test_app_without_lifespan_serves_the_agent(self, settings: Settings) -> None:
        app = create_app(settings)
        client = TestClient(app)
        try:
            response = client.post(AGENT_URL, json={"request": QUESTION})
            assert response.status_code == 200
            assert response.json()["provider"] == "mock"
        finally:
            client.close()
