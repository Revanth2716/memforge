"""REST API tests for the M3 retrieval endpoints."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

API = "/api/v1/memories"
SEARCH = f"{API}/search"
UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"

PYTHON = "User prefers Python for backend work"
NEAR_DUPLICATE = "The user prefers python for backend development work"
CONFLICT_A = "preferred backend language = Python"
CONFLICT_B = "preferred backend language = TypeScript"


def create(client: TestClient, **payload: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"content": "a memory", "memory_type": "FACT"}
    body.update(payload)
    response = client.post(API, json=body)
    assert response.status_code == 201, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


def search(client: TestClient, **payload: Any) -> tuple[int, dict[str, Any]]:
    body: dict[str, Any] = {"query": "python backend work"}
    body.update(payload)
    response = client.post(SEARCH, json=body)
    return response.status_code, (response.json() if response.content else {})


def assert_error(response: Any, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    body: dict[str, Any] = response.json()
    error: dict[str, Any] = body["error"]
    assert error["code"] == code
    return body


class TestSearch:
    def test_returns_ranked_results_with_explanations(self, client: TestClient) -> None:
        first = create(client, content=PYTHON, memory_type="PREFERENCE", confidence=0.9)
        create(client, content="User is learning TypeScript", memory_type="OBSERVATION")

        status_code, body = search(client, top_k=5)
        assert status_code == 200
        assert body["query"] == "python backend work"
        assert body["retrieval_mode"] == "lexical"
        assert body["requested_mode"] == "lexical"
        assert body["semantic_available"] is False
        assert body["degraded_reason"] is None
        # Only the Python memory shares terms with this query.
        assert body["candidates_considered"] == 1
        assert body["candidates_returned"] >= 1

        top = body["results"][0]
        assert top["memory_id"] == first["id"]
        assert top["content"] == PYTHON
        assert top["memory_type"] == "PREFERENCE"
        assert top["status"] == "ACTIVE"
        assert top["confidence"] == 0.9
        assert top["importance"] == 0.5
        assert 0.0 <= top["final_score"] <= 1.0
        assert 0.0 <= top["relevance"] <= 1.0
        assert top["semantic_score"] is None
        assert "final " in top["explanation"]
        assert len(top["factors"]) == 5
        assert top["conflicts"] == []
        assert top["duplicate_of"] is None
        assert top["superseded_by"] is None

    def test_factors_are_documented(self, client: TestClient) -> None:
        create(client, content=PYTHON, memory_type="PREFERENCE")
        _, body = search(client)
        names = [factor["name"] for factor in body["results"][0]["factors"]]
        assert names == [
            "relevance",
            "effective_priority",
            "confidence",
            "freshness",
            "type_preference",
        ]
        assert sum(factor["weight"] for factor in body["results"][0]["factors"]) == pytest.approx(
            1.0
        )
        for factor in body["results"][0]["factors"]:
            assert 0.0 <= factor["value"] <= 1.0
            assert factor["explanation"]

    def test_query_metadata_is_reported(self, client: TestClient) -> None:
        create(client, content=PYTHON)
        _, body = search(client, top_k=1)
        assert body["top_k"] == 1
        assert body["candidates_returned"] == 1
        assert body["latency_ms"] >= 0.0
        assert body["semantic_provider"] is None
        assert body["semantic_model"] is None
        assert body["suppressed"] == []

    def test_no_match_returns_an_empty_result_set(self, client: TestClient) -> None:
        create(client, content=PYTHON)
        _, body = search(client, query="kubernetes")
        assert body["results"] == []
        assert body["candidates_considered"] == 0

    def test_top_k_limits_results(self, client: TestClient) -> None:
        for index in range(4):
            create(client, content=f"python backend work {index}", memory_type="TASK")
        _, body = search(client, top_k=2)
        assert body["candidates_considered"] == 4
        assert body["candidates_returned"] == 2

    def test_filters(self, client: TestClient) -> None:
        create(client, content="python backend", memory_type="TASK")
        create(client, content="python backend", memory_type="PREFERENCE")

        _, filtered = search(client, memory_type="TASK")
        assert filtered["candidates_considered"] == 1
        assert filtered["results"][0]["memory_type"] == "TASK"

        _, confident = search(client, min_confidence=0.9)
        assert confident["candidates_considered"] == 0

    def test_include_archived(self, client: TestClient) -> None:
        created = create(client, content=PYTHON)
        client.patch(f"{API}/{created['id']}", json={"status": "ARCHIVED"})

        _, default = search(client)
        assert default["candidates_considered"] == 0

        _, archived = search(client, status="ARCHIVED")
        assert archived["candidates_considered"] == 1

        _, everything = search(client, status=None)
        assert everything["candidates_considered"] == 1

    def test_creation_window(self, client: TestClient) -> None:
        create(client, content="python backend")
        _, body = search(client, created_after="2030-01-01T00:00:00+00:00")
        assert body["candidates_considered"] == 0

    def test_duplicate_suppression_is_reported(self, client: TestClient) -> None:
        first = create(client, content=PYTHON)
        second = create(client, content=NEAR_DUPLICATE)
        # Record the M2 duplicate relationship that suppression relies on.
        client.get(f"{API}/{first['id']}/duplicates")

        _, body = search(client, top_k=5)
        returned = [result["memory_id"] for result in body["results"]]

        # Exactly one representative survives; which one is decided by the
        # documented deterministic tie-break (score descending, then id).
        assert body["candidates_suppressed"] == 1
        suppressed = body["suppressed"][0]
        assert suppressed["memory_id"] in {first["id"], second["id"]}
        assert set(returned) == {first["id"], second["id"]} - {suppressed["memory_id"]}
        representative = first["id"] if suppressed["memory_id"] == second["id"] else second["id"]
        assert representative in suppressed["reason"]
        assert "duplicate_of" in suppressed["reason"]

    def test_conflict_metadata_is_reported(self, client: TestClient) -> None:
        loser = create(client, content=CONFLICT_A, importance=0.2)
        winner = create(client, content=CONFLICT_B, importance=0.9)
        client.get(f"{API}/{winner['id']}/conflicts")
        assert (
            client.post(f"{API}/{winner['id']}/conflicts/{loser['id']}/resolve").status_code == 200
        )

        _, body = search(client, query="preferred backend language", top_k=5)
        top = body["results"][0]
        assert top["memory_id"] == winner["id"]
        assert top["conflicts"][0]["target_memory_id"] == loser["id"]
        assert top["conflicts"][0]["resolution"] == "supersedes_other"
        assert top["superseded_by"] is None


class TestSemanticRequests:
    def test_semantic_request_without_a_provider_degrades(self, client: TestClient) -> None:
        create(client, content=PYTHON)
        status_code, body = search(client, retrieval_mode="semantic")

        assert status_code == 200
        assert body["requested_mode"] == "semantic"
        assert body["retrieval_mode"] == "lexical"
        assert body["semantic_available"] is False
        assert body["degraded_reason"] is not None
        assert body["results"][0]["semantic_score"] is None

    def test_hybrid_request_without_a_provider_degrades(self, client: TestClient) -> None:
        create(client, content=PYTHON)
        _, body = search(client, retrieval_mode="hybrid")
        assert body["retrieval_mode"] == "lexical"
        assert "no embedding provider" in body["degraded_reason"]

    def test_semantic_weight_is_accepted(self, client: TestClient) -> None:
        create(client, content=PYTHON)
        _, body = search(client, retrieval_mode="hybrid", semantic_weight=0.25)
        assert body["retrieval_mode"] == "lexical"


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"query": ""},
            {"query": "   "},
            {"query": "x", "top_k": 0},
            {"query": "x", "top_k": 101},
            {"query": "x", "retrieval_mode": "vector"},
            {"query": "x", "semantic_weight": 1.5},
            {"query": "x", "min_confidence": -0.1},
            {"query": "x", "min_importance": 2.0},
            {"query": "x", "memory_type": "NOT_A_TYPE"},
            {"query": "x", "status": "NOPE"},
            {
                "query": "x",
                "created_after": "2030-01-01T00:00:00+00:00",
                "created_before": "2020-01-01T00:00:00+00:00",
            },
            {"query": "x", "unknown_field": 1},
            {"query": "x" * 2000},
        ],
    )
    def test_invalid_requests(self, client: TestClient, payload: dict[str, Any]) -> None:
        response = client.post(SEARCH, json=payload)
        assert_error(response, 422, "VALIDATION_ERROR")

    def test_malformed_json(self, client: TestClient) -> None:
        response = client.post(
            SEARCH, content=b"{not json", headers={"content-type": "application/json"}
        )
        assert response.status_code in (400, 422)
        assert "error" in response.json()

    def test_search_is_post_only(self, client: TestClient) -> None:
        # ``GET /memories/{id}`` is a sibling route, so a GET on /search is
        # not the search contract (405, or a UUID validation 422).
        response = client.get(SEARCH)
        assert response.status_code in (405, 422)


class TestExplain:
    def test_explain_returns_one_result(self, client: TestClient) -> None:
        created = create(client, content=PYTHON, memory_type="PREFERENCE", confidence=0.9)
        response = client.get(f"{API}/{created['id']}/explain", params={"query": "python backend"})

        assert response.status_code == 200
        body = response.json()
        assert body["memory_id"] == created["id"]
        assert len(body["factors"]) == 5
        assert "python backend" in body["explanation"]
        assert body["final_score"] > 0.0

    def test_explain_unknown_memory(self, client: TestClient) -> None:
        response = client.get(f"{API}/{UNKNOWN_ID}/explain", params={"query": "python"})
        assert_error(response, 404, "MEMORY_NOT_FOUND")

    def test_explain_requires_a_query(self, client: TestClient) -> None:
        created = create(client, content=PYTHON)
        response = client.get(f"{API}/{created['id']}/explain")
        assert_error(response, 422, "VALIDATION_ERROR")

    def test_explain_rejects_an_empty_query(self, client: TestClient) -> None:
        created = create(client, content=PYTHON)
        response = client.get(f"{API}/{created['id']}/explain", params={"query": "  "})
        assert_error(response, 422, "VALIDATION_ERROR")

    def test_explain_rejects_a_bad_mode(self, client: TestClient) -> None:
        created = create(client, content=PYTHON)
        response = client.get(
            f"{API}/{created['id']}/explain", params={"query": "python", "retrieval_mode": "vector"}
        )
        assert_error(response, 422, "VALIDATION_ERROR")


class TestConfiguration:
    def test_embedding_provider_can_be_enabled(self, settings: object) -> None:
        from app.config import Settings

        assert isinstance(settings, Settings)
        provider = settings.build_embedding_provider()
        assert provider is None  # default configuration is lexical-only

        embedded = Settings(embedding_provider="hashed", database_path=settings.database_path)
        provider = embedded.build_embedding_provider()
        assert provider is not None
        assert provider.is_available() is True

    def test_unknown_provider_is_rejected(self, settings: object) -> None:
        from app.config import Settings
        from app.domain.embeddings import EmbeddingProviderError

        assert isinstance(settings, Settings)
        broken = Settings(embedding_provider="openai", database_path=settings.database_path)
        with pytest.raises(EmbeddingProviderError):
            broken.build_embedding_provider()


class TestCompatibility:
    """M3 must not change M1 or M2 behaviour."""

    def test_m1_endpoints_still_work(self, client: TestClient) -> None:
        created = create(client, content=PYTHON)
        assert client.get(f"{API}/{created['id']}").status_code == 200
        assert client.get(f"{API}/{created['id']}/history").status_code == 200
        assert client.get(f"{API}/{created['id']}/intelligence").status_code == 200
        assert client.get(f"{API}/{created['id']}/duplicates").status_code == 200
        assert client.get(f"{API}/{created['id']}/conflicts").status_code == 200
        assert client.get(f"{API}/{created['id']}/relationships").status_code == 200

    def test_search_does_not_mutate_memories(self, client: TestClient) -> None:
        created = create(client, content=PYTHON)
        before = client.get(f"{API}/{created['id']}").json()
        search(client, top_k=5)
        assert client.get(f"{API}/{created['id']}").json() == before
        assert len(client.get(f"{API}/{created['id']}/history").json()["items"]) == 1

    def test_openapi_documents_the_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/memories/search" in paths
        assert "/api/v1/memories/{memory_id}/explain" in paths
