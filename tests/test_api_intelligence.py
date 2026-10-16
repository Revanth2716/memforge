"""REST API tests for the M2 intelligence endpoints."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.errors import NoConflictError
from app.services.memory_service import MemoryService

API = "/api/v1/memories"
UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"

DUPLICATE_A = "User prefers Python"
DUPLICATE_B = "Python is the user's preferred language"
CONFLICT_A = "preferred backend language = Python"
CONFLICT_B = "preferred backend language = TypeScript"


def create(client: TestClient, **payload: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"content": "a memory", "memory_type": "FACT"}
    body.update(payload)
    response = client.post(API, json=body)
    assert response.status_code == 201, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


def assert_error(response: Any, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    body: dict[str, Any] = response.json()
    error: dict[str, Any] = body["error"]
    assert error["code"] == code
    return body


def service_of(client: TestClient) -> MemoryService:
    """The service instance attached to the running application."""
    app = cast(FastAPI, client.app)
    service: MemoryService = app.state.memory_service
    return service


class TestIntelligenceEndpoint:
    def test_returns_full_breakdown(self, client: TestClient) -> None:
        memory = create(
            client,
            content="User prefers Python",
            memory_type="PREFERENCE",
            confidence=0.8,
            importance=0.6,
        )
        response = client.get(f"{API}/{memory['id']}/intelligence")

        assert response.status_code == 200
        body = response.json()
        assert body["memory_id"] == memory["id"]
        assert body["confidence"] == 0.8
        assert body["stored_importance"] == 0.6
        assert 0.0 <= body["importance_score"] <= 1.0
        assert 0.0 <= body["effective_priority"] <= 1.0
        assert len(body["importance_factors"]) == 5
        assert body["importance_explanation"].startswith("importance ")
        assert body["priority_explanation"].startswith("priority ")
        assert body["decay_multiplier"] == 1.0
        assert body["access_count"] == 0
        assert body["last_accessed_at"] is None
        assert body["decay_half_life_days"] == 30.0
        assert body["similarity_threshold"] == 0.6
        assert body["computed_at"]

    def test_factors_are_documented(self, client: TestClient) -> None:
        memory = create(client)
        factors = client.get(f"{API}/{memory['id']}/intelligence").json()["importance_factors"]
        assert [factor["name"] for factor in factors] == [
            "memory_type",
            "explicit_importance",
            "recurrence",
            "recency",
            "access_frequency",
        ]
        assert sum(factor["weight"] for factor in factors) == pytest.approx(1.0)
        for factor in factors:
            assert 0.0 <= factor["raw_value"] <= 1.0
            assert factor["contribution"] == pytest.approx(
                factor["weight"] * factor["raw_value"], abs=1e-6
            )
            assert factor["explanation"]

    def test_recurrence_grows_the_score(self, client: TestClient) -> None:
        first = create(client, content=DUPLICATE_A)
        before = client.get(f"{API}/{first['id']}/intelligence").json()["importance_score"]

        create(client, content=DUPLICATE_B)
        client.get(f"{API}/{first['id']}/duplicates")

        after = client.get(f"{API}/{first['id']}/intelligence").json()["importance_score"]
        assert after > before

    def test_unknown_memory_returns_404(self, client: TestClient) -> None:
        assert_error(client.get(f"{API}/{UNKNOWN_ID}/intelligence"), 404, "MEMORY_NOT_FOUND")

    def test_malformed_id_returns_422(self, client: TestClient) -> None:
        assert_error(client.get(f"{API}/not-a-uuid/intelligence"), 422, "VALIDATION_ERROR")


class TestDuplicatesEndpoint:
    def test_returns_matches_with_evidence(self, client: TestClient) -> None:
        first = create(client, content=DUPLICATE_A)
        second = create(client, content=DUPLICATE_B)

        body = client.get(f"{API}/{first['id']}/duplicates").json()
        assert body["memory_id"] == first["id"]
        assert body["threshold"] == 0.6
        assert body["total"] == 1
        match = body["items"][0]
        assert match["memory_id"] == second["id"]
        assert match["similarity"] == pytest.approx(6 / 7)
        assert set(match["shared_tokens"]) == {"user", "prefer", "python"}
        assert "Dice" in match["explanation"]

    def test_no_duplicates_returns_empty_list(self, client: TestClient) -> None:
        memory = create(client, content="the server is in hyderabad")
        body = client.get(f"{API}/{memory['id']}/duplicates").json()
        assert body == {
            "memory_id": memory["id"],
            "threshold": 0.6,
            "total": 0,
            "items": [],
        }

    def test_threshold_query_parameter(self, client: TestClient) -> None:
        first = create(client, content="alpha beta")
        create(client, content="alpha gamma")

        strict = client.get(f"{API}/{first['id']}/duplicates").json()
        loose = client.get(f"{API}/{first['id']}/duplicates", params={"threshold": 0.5}).json()
        assert strict["total"] == 0
        assert loose["total"] == 1
        assert loose["threshold"] == 0.5

    def test_limit_query_parameter(self, client: TestClient) -> None:
        anchor = create(client, content="alpha beta gamma delta")
        for index in range(4):
            create(client, content=f"alpha beta gamma {index}")
        body = client.get(f"{API}/{anchor['id']}/duplicates", params={"limit": 3}).json()
        assert body["total"] == 3

    def test_detection_records_a_relationship(self, client: TestClient) -> None:
        first = create(client, content=DUPLICATE_A)
        second = create(client, content=DUPLICATE_B)
        client.get(f"{API}/{first['id']}/duplicates")

        relationships = client.get(f"{API}/{first['id']}/relationships").json()
        assert relationships["total"] == 1
        assert relationships["items"][0]["relationship"] == "DUPLICATE_OF"
        assert relationships["items"][0]["target_id"] == second["id"]
        assert relationships["items"][0]["similarity"] == pytest.approx(6 / 7)
        assert relationships["items"][0]["created_at"]

    def test_invalid_query_parameters(self, client: TestClient) -> None:
        memory = create(client)
        for params in ({"threshold": 1.5}, {"threshold": -1}, {"limit": 0}, {"limit": "abc"}):
            assert_error(
                client.get(f"{API}/{memory['id']}/duplicates", params=params),
                422,
                "VALIDATION_ERROR",
            )

    def test_unknown_memory_returns_404(self, client: TestClient) -> None:
        assert_error(client.get(f"{API}/{UNKNOWN_ID}/duplicates"), 404, "MEMORY_NOT_FOUND")


class TestConflictsEndpoint:
    def test_lists_unresolved_conflicts(self, client: TestClient) -> None:
        first = create(client, content=CONFLICT_A)
        second = create(client, content=CONFLICT_B)

        body = client.get(f"{API}/{second['id']}/conflicts").json()
        assert body["memory_id"] == second["id"]
        assert body["total"] == 1
        conflict = body["items"][0]
        assert conflict["target_memory_id"] == first["id"]
        assert conflict["attribute"] == "prefer backend language"
        assert conflict["source_value"] == "typescript"
        assert conflict["target_value"] == "python"
        assert conflict["value_similarity"] == 0.0
        assert "both memories state" in conflict["explanation"]

    def test_no_conflicts_returns_empty_list(self, client: TestClient) -> None:
        memory = create(client, content="User prefers Python")
        create(client, content="User works from Hyderabad")
        body = client.get(f"{API}/{memory['id']}/conflicts").json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_conflict_detection_records_a_relationship(self, client: TestClient) -> None:
        first = create(client, content=CONFLICT_A)
        second = create(client, content=CONFLICT_B)
        client.get(f"{API}/{second['id']}/conflicts")

        relationships = client.get(f"{API}/{second['id']}/relationships").json()
        assert [item["relationship"] for item in relationships["items"]] == ["CONFLICTS_WITH"]
        assert relationships["items"][0]["target_id"] == first["id"]
        assert relationships["items"][0]["similarity"] is None

    def test_unknown_memory_returns_404(self, client: TestClient) -> None:
        assert_error(client.get(f"{API}/{UNKNOWN_ID}/conflicts"), 404, "MEMORY_NOT_FOUND")


class TestResolveEndpoint:
    def test_resolution_returns_explanation(self, client: TestClient) -> None:
        python = create(client, content=CONFLICT_A, importance=0.4)
        typescript = create(client, content=CONFLICT_B, importance=0.9)

        response = client.post(f"{API}/{typescript['id']}/conflicts/{python['id']}/resolve")
        assert response.status_code == 200
        body = response.json()
        assert body["winner_memory_id"] == typescript["id"]
        assert body["loser_memory_id"] == python["id"]
        assert body["decided_by"] == "explicit_importance"
        assert body["loser_archived"] is True
        assert body["steps"][0]["factor"] == "explicit_importance"
        assert body["steps"][0]["winner_id"] == typescript["id"]
        assert "explicit_importance" in body["explanation"]
        assert body["conflict"]["target_memory_id"] == python["id"]

    def test_resolution_archives_the_loser(self, client: TestClient) -> None:
        python = create(client, content=CONFLICT_A, importance=0.4)
        typescript = create(client, content=CONFLICT_B, importance=0.9)
        client.post(f"{API}/{typescript['id']}/conflicts/{python['id']}/resolve")

        assert client.get(f"{API}/{python['id']}").json()["status"] == "ARCHIVED"
        assert client.get(f"{API}/{typescript['id']}").json()["status"] == "ACTIVE"
        # The conflict is gone from the unresolved list.
        assert client.get(f"{API}/{typescript['id']}/conflicts").json()["total"] == 0

    def test_history_of_the_loser_is_preserved(self, client: TestClient) -> None:
        python = create(client, content=CONFLICT_A, importance=0.4)
        typescript = create(client, content=CONFLICT_B, importance=0.9)
        client.post(f"{API}/{typescript['id']}/conflicts/{python['id']}/resolve")

        history = client.get(f"{API}/{python['id']}/history").json()
        assert [version["change"] for version in history["items"]] == ["CREATED", "UPDATED"]
        assert history["items"][-1]["status"] == "ARCHIVED"

    def test_resolution_requires_an_existing_conflict(self, client: TestClient) -> None:
        memory = create(client, content=CONFLICT_A)
        other = create(client, content="User works from Hyderabad")
        assert_error(
            client.post(f"{API}/{memory['id']}/conflicts/{other['id']}/resolve"),
            409,
            "NO_CONFLICT",
        )

    def test_resolution_is_not_repeatable(self, client: TestClient) -> None:
        python = create(client, content=CONFLICT_A, importance=0.4)
        typescript = create(client, content=CONFLICT_B, importance=0.9)
        assert (
            client.post(f"{API}/{typescript['id']}/conflicts/{python['id']}/resolve").status_code
            == 200
        )
        assert_error(
            client.post(f"{API}/{typescript['id']}/conflicts/{python['id']}/resolve"),
            409,
            "NO_CONFLICT",
        )

    def test_unknown_memory_returns_404(self, client: TestClient) -> None:
        assert_error(
            client.post(f"{API}/{UNKNOWN_ID}/conflicts/{UNKNOWN_ID}/resolve"),
            404,
            "MEMORY_NOT_FOUND",
        )

    def test_malformed_ids_return_422(self, client: TestClient) -> None:
        memory = create(client, content=CONFLICT_A)
        assert_error(
            client.post(f"{API}/{memory['id']}/conflicts/not-a-uuid/resolve"),
            422,
            "VALIDATION_ERROR",
        )

    def test_service_error_is_mapped(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory = create(client, content=CONFLICT_A)
        other = create(client, content=CONFLICT_B)
        service = service_of(client)

        def boom(*_: object, **__: object) -> None:
            raise NoConflictError(memory["id"], other["id"])

        monkeypatch.setattr(service, "resolve_conflict", boom)
        assert_error(
            client.post(f"{API}/{memory['id']}/conflicts/{other['id']}/resolve"),
            409,
            "NO_CONFLICT",
        )


class TestRelationshipsEndpoint:
    def test_returns_every_relationship(self, client: TestClient) -> None:
        first = create(client, content=CONFLICT_A, importance=0.4)
        second = create(client, content=CONFLICT_B, importance=0.9)

        client.get(f"{API}/{second['id']}/duplicates")
        client.get(f"{API}/{second['id']}/conflicts")
        client.post(f"{API}/{second['id']}/conflicts/{first['id']}/resolve")

        body = client.get(f"{API}/{second['id']}/relationships").json()
        kinds = {item["relationship"] for item in body["items"]}
        # A conflicting pair is never also reported as a duplicate.
        assert kinds == {"CONFLICTS_WITH", "SUPERSEDES"}
        for item in body["items"]:
            assert item["source_id"] and item["target_id"]
            assert item["created_at"]

    def test_empty_relationships(self, client: TestClient) -> None:
        memory = create(client)
        body = client.get(f"{API}/{memory['id']}/relationships").json()
        assert body == {"memory_id": memory["id"], "total": 0, "items": []}

    def test_unknown_memory_returns_404(self, client: TestClient) -> None:
        assert_error(client.get(f"{API}/{UNKNOWN_ID}/relationships"), 404, "MEMORY_NOT_FOUND")

    def test_forgotten_memory_still_exposes_relationships(self, client: TestClient) -> None:
        first = create(client, content=CONFLICT_A, importance=0.4)
        second = create(client, content=CONFLICT_B, importance=0.9)
        client.post(f"{API}/{second['id']}/conflicts/{first['id']}/resolve")
        client.delete(f"{API}/{second['id']}")

        # The memory is invisible for reads, but its audit trail survives.
        assert client.get(f"{API}/{second['id']}").status_code == 404
        assert client.get(f"{API}/{second['id']}/relationships").status_code == 200


class TestOpenApi:
    def test_new_paths_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/api/v1/memories/{memory_id}/intelligence",
            "/api/v1/memories/{memory_id}/duplicates",
            "/api/v1/memories/{memory_id}/conflicts",
            "/api/v1/memories/{memory_id}/conflicts/{target_id}/resolve",
            "/api/v1/memories/{memory_id}/relationships",
        ):
            assert path in paths, path
