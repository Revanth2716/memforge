"""REST contract tests for ``POST /api/v1/context/compile`` (M4)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.errors import InvalidContextConfigError
from app.domain.models import MemoryType
from app.main import create_app
from app.services.context_service import ContextCompilerService
from tests.support import make_memory

COMPILE_URL = "/api/v1/context/compile"

PROFILE = "User prefers Python for backend work."
DUPLICATE = "The user prefers python for backend development work"


def seed(client: TestClient) -> dict[str, str]:
    """Create a small corpus with a recorded near-duplicate pair."""
    created: dict[str, str] = {}
    for key, payload in (
        (
            "profile",
            {
                "content": PROFILE,
                "memory_type": "PROFILE",
                "source": "user:demo",
                "confidence": 0.95,
                "importance": 0.9,
            },
        ),
        (
            "duplicate",
            {
                "content": DUPLICATE,
                "memory_type": "OBSERVATION",
                "source": "agent:note",
                "confidence": 0.5,
                "importance": 0.3,
            },
        ),
        (
            "task",
            {
                "content": "Task: ship the backend service in Python.",
                "memory_type": "TASK",
                "source": "agent:planner",
                "confidence": 0.9,
                "importance": 0.8,
            },
        ),
    ):
        response = client.post("/api/v1/memories", json=payload)
        assert response.status_code == 201, response.text
        created[key] = response.json()["id"]
    duplicates = client.get(f"/api/v1/memories/{created['profile']}/duplicates")
    assert duplicates.status_code == 200
    assert duplicates.json()["items"], "the near-duplicate pair must be detected"
    return created


def compile_body(**overrides: object) -> dict[str, object]:
    """A valid compile request, overridable field by field."""
    body: dict[str, object] = {"query": "python backend", "token_budget": 400}
    body.update(overrides)
    return body


class TestValidCompilation:
    def test_returns_the_documented_payload(self, client: TestClient) -> None:
        seed(client)
        response = client.post(COMPILE_URL, json=compile_body())
        assert response.status_code == 200, response.text
        payload = response.json()

        assert payload["query"] == "python backend"
        assert payload["token_budget"] == 400
        assert payload["within_budget"] is True
        assert payload["estimated_tokens"] <= payload["token_budget"]
        assert payload["remaining_tokens"] == payload["token_budget"] - payload["estimated_tokens"]
        assert payload["selected_count"] == len(payload["blocks"])
        assert payload["suppressed_count"] == len(payload["suppressed_candidates"])
        assert payload["candidates_considered"] == len(payload["selection_trace"])
        assert payload["retrieval_mode"] == "lexical"
        assert payload["context_text"].strip()

    def test_blocks_carry_provenance(self, client: TestClient) -> None:
        created = seed(client)
        payload = client.post(COMPILE_URL, json=compile_body()).json()

        profile_block = next(
            block for block in payload["blocks"] if block["memory_id"] == created["profile"]
        )
        assert profile_block["memory_type"] == MemoryType.PROFILE
        assert profile_block["source"] == "user:demo"
        assert profile_block["confidence"] == 0.95
        assert profile_block["importance"] == 0.9
        assert profile_block["content"] == PROFILE
        assert profile_block["retrieval_score"] > 0.0
        assert profile_block["utility_score"] > 0.0
        assert profile_block["section"] == "PROFILE"
        assert profile_block["compression"]["original_text"] == PROFILE

    def test_every_line_of_the_context_belongs_to_a_block(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json=compile_body()).json()
        rendered = {block["rendered_content"] for block in payload["blocks"]}

        for line in payload["context_text"].splitlines():
            if line.startswith("- "):
                assert line[2:] in rendered

    def test_duplicate_is_reported_as_suppressed(self, client: TestClient) -> None:
        created = seed(client)
        payload = client.post(COMPILE_URL, json=compile_body(token_budget=2000)).json()

        entry = next(
            item
            for item in payload["suppressed_candidates"]
            if item["memory_id"] == created["duplicate"]
        )
        assert entry["selected"] is False
        assert entry["reason"] == "duplicate_of"
        assert entry["related_memory_id"] == created["profile"]
        assert "near-duplicate of" in entry["detail"]

    def test_explanation_is_machine_readable(self, client: TestClient) -> None:
        seed(client)
        explanation = client.post(COMPILE_URL, json=compile_body()).json()["explanation"]

        assert explanation["estimator"] == "memforge-char-word/v1"
        assert explanation["ordering_policy"] == "section-then-utility/v1"
        assert explanation["section_order"][0] == "PROFILE"
        assert explanation["selection_counts"]["selected"] >= 1
        assert set(explanation["utility_weights"]) == {
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
        }
        assert any("not provider tokenisations" in note for note in explanation["notes"])

    def test_default_budget_is_used_when_omitted(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json={"query": "python backend"}).json()
        assert payload["token_budget"] == 1200

    def test_empty_corpus_returns_an_empty_context(self, client: TestClient) -> None:
        payload = client.post(COMPILE_URL, json=compile_body()).json()
        assert payload["blocks"] == []
        assert payload["context_text"] == ""
        assert payload["estimated_tokens"] == 0
        assert payload["within_budget"] is True

    def test_unknown_query_returns_an_empty_context(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json=compile_body(query="kubernetes")).json()
        assert payload["candidates_considered"] == 0
        assert payload["blocks"] == []

    def test_response_is_deterministic(self, client: TestClient) -> None:
        seed(client)
        first = client.post(COMPILE_URL, json=compile_body(token_budget=60))
        second = client.post(COMPILE_URL, json=compile_body(token_budget=60))
        assert first.json() == second.json()

    def test_optional_filters_are_applied(self, client: TestClient) -> None:
        created = seed(client)
        payload = client.post(
            COMPILE_URL,
            json=compile_body(
                token_budget=2000,
                exclude_ids=[created["profile"]],
                min_confidence=0.0,
            ),
        ).json()

        assert all(block["memory_id"] != created["profile"] for block in payload["blocks"])
        assert any(item["reason"] == "excluded_by_id" for item in payload["suppressed_candidates"])

    def test_preferred_types_are_honoured(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(
            COMPILE_URL,
            json=compile_body(token_budget=2000, preferred_memory_types=["TASK"]),
        ).json()
        assert all(block["memory_type"] == "TASK" for block in payload["blocks"])

    def test_hybrid_request_without_embeddings_is_reported(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json=compile_body(retrieval_mode="hybrid")).json()

        assert payload["requested_retrieval_mode"] == "hybrid"
        assert payload["retrieval_mode"] == "lexical"
        assert payload["semantic_available"] is False
        assert payload["degraded_reason"] is not None

    def test_compression_can_be_disabled(self, client: TestClient) -> None:
        created = seed(client)
        payload = client.post(
            COMPILE_URL, json=compile_body(compress=False, token_budget=2000)
        ).json()
        block = next(item for item in payload["blocks"] if item["memory_id"] == created["profile"])
        assert block["rendered_content"] == block["content"]


class TestBudgetValidation:
    @pytest.mark.parametrize("budget", [0, -1, -100])
    def test_non_positive_budget_is_rejected(self, client: TestClient, budget: int) -> None:
        response = client.post(COMPILE_URL, json=compile_body(token_budget=budget))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_budget_above_the_configured_maximum_is_rejected(self, client: TestClient) -> None:
        response = client.post(COMPILE_URL, json=compile_body(token_budget=16001))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "CONTEXT_VALIDATION_ERROR"
        assert "16000" in error["message"]

    def test_budget_at_the_configured_maximum_is_accepted(self, client: TestClient) -> None:
        response = client.post(COMPILE_URL, json=compile_body(token_budget=16000))
        assert response.status_code == 200

    @pytest.mark.parametrize("budget", [1, 2, 7, 15, 40, 120, 399, 400])
    def test_the_hard_limit_holds_for_every_budget(self, client: TestClient, budget: int) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json=compile_body(token_budget=budget)).json()
        assert payload["estimated_tokens"] <= budget
        assert payload["remaining_tokens"] >= 0
        assert payload["within_budget"] is True


class TestRequestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"query": ""},
            {"query": "   "},
            {"query": "python", "token_budget": "many"},
            {"query": "python", "top_k": 0},
            {"query": "python", "top_k": 500},
            {"query": "python", "retrieval_mode": "telepathy"},
            {"query": "python", "min_confidence": 1.5},
            {"query": "python", "min_candidate_score": -0.1},
            {"query": "python", "max_block_tokens": 0},
            {"query": "python", "preferred_memory_types": []},
            {"query": "python", "preferred_memory_types": ["NOPE"]},
            {"query": "python", "include_ids": [" "], "exclude_ids": []},
            {"query": "python", "include_ids": ["a"], "exclude_ids": ["a"]},
            {"query": "python", "semantic_weight": 2.0},
            {"query": "python", "status": "GONE"},
        ],
    )
    def test_malformed_requests_use_the_error_envelope(
        self, client: TestClient, body: dict[str, object]
    ) -> None:
        response = client.post(COMPILE_URL, json=body)
        assert response.status_code in (400, 422), response.text
        envelope = response.json()["error"]
        assert envelope["code"] in ("VALIDATION_ERROR", "CONTEXT_VALIDATION_ERROR")
        assert envelope["message"]

    def test_unknown_fields_are_forbidden(self, client: TestClient) -> None:
        response = client.post(COMPILE_URL, json=compile_body(unexpected_field=True))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_unknown_memory_id_in_include_list_is_not_an_error(self, client: TestClient) -> None:
        # An allow list naming a memory that does not exist yields an empty
        # context, not a 404: the allow list filters candidates, it does not
        # fetch them.
        seed(client)
        response = client.post(
            COMPILE_URL,
            json=compile_body(
                include_ids=["00000000-0000-5000-8000-000000000000"],
                token_budget=2000,
            ),
        )
        assert response.status_code == 200
        assert response.json()["blocks"] == []

    def test_validation_details_are_returned(self, client: TestClient) -> None:
        response = client.post(COMPILE_URL, json={"query": ""})
        details = response.json()["error"]["details"]
        assert details["errors"]
        assert details["errors"][0]["location"][0] == "body"


class TestOpenApi:
    def test_endpoint_is_documented(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert COMPILE_URL in schema["paths"]
        operation = schema["paths"][COMPILE_URL]["post"]
        assert operation["tags"] == ["context"]
        assert "compile" in operation["summary"].lower()

    def test_endpoint_is_added_next_to_the_milestone_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/memories/search" in paths
        assert "/api/v1/context/compile" in paths
        assert len([path for path in paths if path.startswith("/api/v1/context")]) == 1


class TestCompatibility:
    def test_milestone_endpoints_are_untouched(self, client: TestClient) -> None:
        seed(client)
        assert client.get("/health").status_code == 200
        assert client.post("/api/v1/memories/search", json={"query": "python"}).status_code == 200

    def test_compilation_never_mutates_a_memory(self, client: TestClient) -> None:
        created = seed(client)
        before = client.get(f"/api/v1/memories/{created['profile']}").json()

        client.post(COMPILE_URL, json=compile_body(token_budget=5))
        after = client.get(f"/api/v1/memories/{created['profile']}").json()

        assert before == after
        assert after["version"] == 1


class TestDomainSeparation:
    def test_memory_content_is_never_rewritten(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json=compile_body(token_budget=2000)).json()

        for block in payload["blocks"]:
            stored = client.get(f"/api/v1/memories/{block['memory_id']}").json()
            assert block["content"] == stored["content"]

    def test_suppressed_memories_are_still_readable(self, client: TestClient) -> None:
        seed(client)
        payload = client.post(COMPILE_URL, json=compile_body(token_budget=2000)).json()
        assert payload["suppressed_count"] > 0
        for entry in payload["suppressed_candidates"]:
            assert client.get(f"/api/v1/memories/{entry['memory_id']}").status_code == 200

    def test_unicode_content_survives_compilation(self, client: TestClient) -> None:
        memory = make_memory(id="x", content="café ☕ python backend")
        assert memory.content == "café ☕ python backend"
        client.post(
            "/api/v1/memories",
            json={"content": "café ☕ preferred python backend", "memory_type": "FACT"},
        )
        payload = client.post(COMPILE_URL, json=compile_body(token_budget=500)).json()

        assert payload["blocks"]
        assert "café ☕" in payload["blocks"][0]["rendered_content"]


class TestWiring:
    """The context compiler is wired like every other service."""

    def test_the_compiler_is_shared_with_the_application(self, settings: Settings) -> None:
        app = create_app(settings)
        with TestClient(app) as client:
            service = app.state.context_service
            assert isinstance(service, ContextCompilerService)
            assert client.post(COMPILE_URL, json=compile_body()).status_code == 200
            assert app.state.context_service is service

    def test_the_compiler_is_created_from_settings(self, settings: Settings) -> None:
        app = create_app(settings)
        with TestClient(app):
            service = app.state.context_service
            assert service.policy == settings.context_policy()
            assert service.estimator_config == settings.token_estimator_config()

    def test_app_without_lifespan_still_compiles(self, settings: Settings) -> None:
        app = create_app(settings)
        client = TestClient(app)
        try:
            client.post(
                "/api/v1/memories",
                json={"content": "alpha beta python", "memory_type": "FACT"},
            )
            response = client.post(COMPILE_URL, json=compile_body())
            assert response.status_code == 200
            assert response.json()["candidates_considered"] == 1
        finally:
            client.close()

    def test_invalid_context_configuration_is_mapped(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(self: ContextCompilerService, request: object) -> object:
            raise InvalidContextConfigError("utility weights must sum to 1.0")

        monkeypatch.setattr(ContextCompilerService, "compile", explode)
        response = client.post(COMPILE_URL, json=compile_body())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INVALID_CONFIGURATION"
