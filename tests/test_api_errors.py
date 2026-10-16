"""API error-handling and validation tests (status codes + error envelope)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.errors import (
    InvalidIntelligenceConfigError,
    InvalidRelationshipError,
    MemforgeError,
    MemoryAlreadyExistsError,
    MemoryValidationError,
    RepositoryError,
)
from app.services.memory_service import MemoryService

API = "/api/v1/memories"
UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"


def create(client: TestClient, **payload: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"content": "a memory", "memory_type": "FACT"}
    body.update(payload)
    response = client.post(API, json=body)
    assert response.status_code == 201, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


def assert_error(response: Any, status_code: int, code: str) -> dict[str, Any]:
    """Assert the standard error envelope shape and return it."""
    assert response.status_code == status_code
    body: dict[str, Any] = response.json()
    assert set(body) == {"error"}
    error: dict[str, Any] = body["error"]
    assert set(error) >= {"code", "message"}
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    return body


def service_of(client: TestClient) -> MemoryService:
    """The service instance attached to the running application."""
    app = cast(FastAPI, client.app)
    service: MemoryService = app.state.memory_service
    return service


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"memory_type": "FACT"},  # missing content
            {"content": "", "memory_type": "FACT"},
            {"content": "   ", "memory_type": "FACT"},
            {"content": "x"},  # missing memory_type
            {"content": "x", "memory_type": "NOT_A_TYPE"},
            {"content": "x", "memory_type": "FACT", "confidence": 1.5},
            {"content": "x", "memory_type": "FACT", "confidence": -0.1},
            {"content": "x", "memory_type": "FACT", "importance": 2.0},
            {"content": "x", "memory_type": "FACT", "source": "s" * 201},
            {"content": "x" * 4001, "memory_type": "FACT"},
            {"content": "x", "memory_type": "FACT", "unknown_field": 1},
            {"content": 5, "memory_type": "FACT"},
        ],
    )
    def test_create_rejects_invalid_payloads(
        self, client: TestClient, payload: dict[str, Any]
    ) -> None:
        response = client.post(API, json=payload)
        assert_error(response, 422, "VALIDATION_ERROR")

    def test_create_creates_nothing_when_validation_fails(self, client: TestClient) -> None:
        client.post(API, json={"content": "", "memory_type": "FACT"})
        assert client.get(API).json()["total"] == 0

    @pytest.mark.parametrize(
        "params",
        [{"limit": 0}, {"limit": 101}, {"limit": "abc"}, {"offset": -1}, {"status": "NOPE"}],
    )
    def test_list_rejects_invalid_query_params(
        self, client: TestClient, params: dict[str, Any]
    ) -> None:
        assert_error(client.get(API, params=params), 422, "VALIDATION_ERROR")

    def test_patch_requires_at_least_one_field(self, client: TestClient) -> None:
        created = create(client)
        assert_error(client.patch(f"{API}/{created['id']}", json={}), 422, "VALIDATION_ERROR")
        assert client.get(f"{API}/{created['id']}").json()["version"] == 1

    @pytest.mark.parametrize("field", ["content", "memory_type", "confidence", "status"])
    def test_patch_rejects_null_for_non_nullable_fields(
        self, client: TestClient, field: str
    ) -> None:
        created = create(client)
        response = client.patch(f"{API}/{created['id']}", json={field: None})
        assert_error(response, 422, "VALIDATION_ERROR")

    @pytest.mark.parametrize("field", ["confidence", "importance"])
    def test_patch_rejects_out_of_range_scores(self, client: TestClient, field: str) -> None:
        created = create(client)
        assert_error(
            client.patch(f"{API}/{created['id']}", json={field: 5.0}), 422, "VALIDATION_ERROR"
        )

    def test_patch_rejects_unknown_fields(self, client: TestClient) -> None:
        created = create(client)
        assert_error(
            client.patch(f"{API}/{created['id']}", json={"nope": 1}), 422, "VALIDATION_ERROR"
        )

    @pytest.mark.parametrize(
        "path",
        [
            f"{API}/not-a-uuid",
            f"{API}/not-a-uuid/history",
        ],
    )
    def test_invalid_identifier_format(self, client: TestClient, path: str) -> None:
        assert_error(client.get(path), 422, "VALIDATION_ERROR")

    def test_malformed_json_body(self, client: TestClient) -> None:
        response = client.post(
            API, content=b"{not json", headers={"content-type": "application/json"}
        )
        assert_error(response, 422, "VALIDATION_ERROR")


class TestErrorEnvelope:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", f"{API}/{UNKNOWN_ID}"),
            ("patch", f"{API}/{UNKNOWN_ID}"),
            ("delete", f"{API}/{UNKNOWN_ID}"),
            ("get", f"{API}/{UNKNOWN_ID}/history"),
        ],
    )
    def test_unknown_memory_returns_404_envelope(
        self, client: TestClient, method: str, path: str
    ) -> None:
        kwargs = {"json": {"content": "x"}} if method == "patch" else {}
        response = getattr(client, method)(path, **kwargs)
        assert_error(response, 404, "MEMORY_NOT_FOUND")

    def test_update_forgotten_memory_returns_404(self, client: TestClient) -> None:
        created = create(client)
        client.delete(f"{API}/{created['id']}")
        response = client.patch(f"{API}/{created['id']}", json={"content": "zombie"})
        assert_error(response, 404, "MEMORY_NOT_FOUND")

    def test_update_cannot_forget_a_memory(self, client: TestClient) -> None:
        created = create(client)
        response = client.patch(f"{API}/{created['id']}", json={"status": "DELETED"})
        assert_error(response, 409, "INVALID_MEMORY_STATE")

        # The rejected patch must not have changed the stored memory.
        body = client.get(f"{API}/{created['id']}").json()
        assert body["status"] == "ACTIVE"
        assert body["version"] == 1
        history = client.get(f"{API}/{created['id']}/history").json()
        assert history["total"] == 1

    def test_unknown_route_returns_404_envelope(self, client: TestClient) -> None:
        assert_error(client.get("/does-not-exist"), 404, "HTTP_ERROR")

    def test_unsupported_method_returns_405_envelope(self, client: TestClient) -> None:
        assert_error(client.put(API), 405, "HTTP_ERROR")

    def test_unhandled_server_error_returns_500_envelope(
        self, client_no_raise: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            service_of(client_no_raise),
            "create_memory",
            _raise(RuntimeError("kaboom")),
        )
        response = client_no_raise.post(API, json={"content": "x", "memory_type": "FACT"})
        assert_error(response, 500, "INTERNAL_ERROR")

    def test_health_reports_storage_unavailable(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_of(client), "health_check", lambda: False)
        assert_error(client.get("/health"), 503, "STORAGE_UNAVAILABLE")

    def test_openapi_documents_every_endpoint(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/health",
            "/api/v1/memories",
            "/api/v1/memories/{memory_id}",
            "/api/v1/memories/{memory_id}/history",
        ):
            assert path in paths, path


def _raise(error: Exception) -> Callable[..., None]:
    def raiser(*_: object, **__: object) -> None:
        raise error

    return raiser


class TestDomainErrorMapping:
    """The API layer is the only place that maps domain errors to HTTP codes."""

    def test_memory_validation_error_maps_to_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = create(client)
        service = service_of(client)
        monkeypatch.setattr(
            service,
            "update_memory",
            _raise(MemoryValidationError("the domain rejected this")),
        )

        response = client.patch(f"{API}/{created['id']}", json={"content": "x"})
        assert_error(response, 400, "MEMORY_VALIDATION_ERROR")

    def test_memory_already_exists_maps_to_409(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = service_of(client)
        monkeypatch.setattr(
            service, "create_memory", _raise(MemoryAlreadyExistsError("duplicate id"))
        )

        response = client.post(API, json={"content": "x", "memory_type": "FACT"})
        assert_error(response, 409, "MEMORY_ALREADY_EXISTS")

    def test_repository_error_maps_to_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = service_of(client)
        monkeypatch.setattr(service, "list_memories", _raise(RepositoryError("disk on fire")))

        assert_error(client.get(API), 500, "REPOSITORY_ERROR")

    def test_unmapped_memforge_error_maps_to_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = service_of(client)
        monkeypatch.setattr(service, "get_history", _raise(MemforgeError("unknown failure")))

        created = create(client)
        assert_error(client.get(f"{API}/{created['id']}/history"), 500, "INTERNAL_ERROR")

    def test_invalid_relationship_maps_to_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = create(client)
        monkeypatch.setattr(
            service_of(client),
            "get_relationships",
            _raise(InvalidRelationshipError("a memory cannot be related to itself")),
        )
        assert_error(
            client.get(f"{API}/{created['id']}/relationships"), 400, "INVALID_RELATIONSHIP"
        )

    def test_invalid_intelligence_config_maps_to_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = create(client)
        monkeypatch.setattr(
            service_of(client),
            "score_memory",
            _raise(InvalidIntelligenceConfigError("weights must sum to 1.0")),
        )
        assert_error(
            client.get(f"{API}/{created['id']}/intelligence"), 500, "INVALID_CONFIGURATION"
        )
