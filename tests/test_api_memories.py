"""REST API happy-path tests (``/api/v1/memories``)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

API = "/api/v1/memories"


def create(client: TestClient, **payload: Any) -> dict[str, Any]:
    """POST a memory and return the parsed body."""
    body: dict[str, Any] = {"content": "a memory", "memory_type": "FACT"}
    body.update(payload)
    response = client.post(API, json=body)
    assert response.status_code == 201, response.text
    return dict(response.json())


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "service": "memforge",
            "version": "1.0.0",
            "storage": "sqlite",
        }

    def test_app_also_works_without_the_lifespan_hook(self, settings: Settings) -> None:
        """A plain TestClient (no ``with``) must still serve requests."""
        with TestClient(create_app(settings)) as warm_client:
            warm_client.post(API, json={"content": "warm up", "memory_type": "FACT"})

        cold_client = TestClient(create_app(settings))
        try:
            response = cold_client.get(API)
            assert response.status_code == 200
            assert response.json()["total"] == 1
        finally:
            cold_client.close()


class TestCreate:
    def test_create_returns_201_with_full_representation(self, client: TestClient) -> None:
        response = client.post(
            API,
            json={
                "content": "User prefers dark mode.",
                "memory_type": "PREFERENCE",
                "source": "user:revanth",
                "confidence": 0.9,
                "importance": 0.7,
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["content"] == "User prefers dark mode."
        assert body["memory_type"] == "PREFERENCE"
        assert body["source"] == "user:revanth"
        assert body["confidence"] == 0.9
        assert body["importance"] == 0.7
        assert body["status"] == "ACTIVE"
        assert body["version"] == 1
        assert body["created_at"] == body["updated_at"]

    def test_create_applies_documented_defaults(self, client: TestClient) -> None:
        body = create(client)
        assert body["source"] is None
        assert body["confidence"] == 0.5
        assert body["importance"] == 0.5

    def test_created_memory_is_readable(self, client: TestClient) -> None:
        body = create(client, content="readable", memory_type="TASK")
        response = client.get(f"{API}/{body['id']}")
        assert response.status_code == 200
        assert response.json() == body

    def test_ids_are_unique(self, client: TestClient) -> None:
        ids = {create(client)["id"] for _ in range(3)}
        assert len(ids) == 3


class TestList:
    def test_list_empty(self, client: TestClient) -> None:
        response = client.get(API)
        assert response.status_code == 200
        assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}

    def test_list_returns_newest_first(self, client: TestClient) -> None:
        first = create(client, content="first")
        second = create(client, content="second")

        body = client.get(API).json()
        assert body["total"] == 2
        assert [item["content"] for item in body["items"]] == ["second", "first"]
        assert [item["id"] for item in body["items"]] == [second["id"], first["id"]]

    def test_list_filters_by_type(self, client: TestClient) -> None:
        create(client, content="a", memory_type="TASK")
        create(client, content="b", memory_type="PREFERENCE")

        body = client.get(API, params={"memory_type": "TASK"}).json()
        assert body["total"] == 1
        assert body["items"][0]["content"] == "a"

    def test_list_paginates_with_stable_total(self, client: TestClient) -> None:
        for index in range(3):
            create(client, content=f"memory {index}")

        page_one = client.get(API, params={"limit": 2, "offset": 0}).json()
        page_two = client.get(API, params={"limit": 2, "offset": 2}).json()

        assert page_one["total"] == 3
        assert len(page_one["items"]) == 2
        assert [item["content"] for item in page_one["items"]] == ["memory 2", "memory 1"]
        assert page_two["total"] == 3
        assert [item["content"] for item in page_two["items"]] == ["memory 0"]

    def test_list_default_hides_archived_and_deleted(self, client: TestClient) -> None:
        active = create(client, content="active")
        archived = create(client, content="archived")
        deleted = create(client, content="deleted")

        client.patch(f"{API}/{archived['id']}", json={"status": "ARCHIVED"})
        client.delete(f"{API}/{deleted['id']}")

        body = client.get(API).json()
        assert [item["id"] for item in body["items"]] == [active["id"]]
        assert client.get(API, params={"status": "ARCHIVED"}).json()["total"] == 1
        assert client.get(API, params={"status": "DELETED"}).json()["total"] == 1


class TestGet:
    def test_get_by_id(self, client: TestClient) -> None:
        created = create(client, content="find me")
        response = client.get(f"{API}/{created['id']}")
        assert response.status_code == 200
        assert response.json()["content"] == "find me"

    def test_get_deleted_memory_returns_404(self, client: TestClient) -> None:
        created = create(client)
        client.delete(f"{API}/{created['id']}")
        assert client.get(f"{API}/{created['id']}").status_code == 404


class TestPatch:
    def test_partial_update_keeps_other_fields(self, client: TestClient) -> None:
        created = create(
            client,
            content="original",
            memory_type="TASK",
            source="agent:planner",
            confidence=0.8,
            importance=0.4,
        )
        response = client.patch(f"{API}/{created['id']}", json={"content": "revised"})

        assert response.status_code == 200
        body = response.json()
        assert body["content"] == "revised"
        assert body["memory_type"] == "TASK"
        assert body["source"] == "agent:planner"
        assert body["confidence"] == 0.8
        assert body["importance"] == 0.4
        assert body["status"] == "ACTIVE"
        assert body["version"] == 2
        assert body["updated_at"] > body["created_at"]
        assert body["created_at"] == created["created_at"]

    def test_update_can_clear_source_with_null(self, client: TestClient) -> None:
        created = create(client, source="user:revanth")
        response = client.patch(f"{API}/{created['id']}", json={"source": None})
        assert response.status_code == 200
        assert response.json()["source"] is None

    def test_update_can_archive(self, client: TestClient) -> None:
        created = create(client)
        response = client.patch(f"{API}/{created['id']}", json={"status": "ARCHIVED"})
        assert response.json()["status"] == "ARCHIVED"
        assert client.get(API).json()["total"] == 0

    def test_update_after_delete_returns_404(self, client: TestClient) -> None:
        created = create(client)
        client.delete(f"{API}/{created['id']}")
        response = client.patch(f"{API}/{created['id']}", json={"content": "zombie"})
        assert response.status_code == 404


class TestDelete:
    def test_delete_returns_204_and_hides_memory(self, client: TestClient) -> None:
        created = create(client)
        response = client.delete(f"{API}/{created['id']}")

        assert response.status_code == 204
        assert response.content == b""
        assert client.get(f"{API}/{created['id']}").status_code == 404
        assert client.get(API).json()["total"] == 0

    def test_delete_is_idempotent(self, client: TestClient) -> None:
        created = create(client)
        assert client.delete(f"{API}/{created['id']}").status_code == 204
        assert client.delete(f"{API}/{created['id']}").status_code == 204
        assert client.get(f"{API}/{created['id']}/history").json()["total"] == 2

    def test_delete_keeps_history(self, client: TestClient) -> None:
        created = create(client, content="temporary")
        client.patch(f"{API}/{created['id']}", json={"content": "still temporary"})
        client.delete(f"{API}/{created['id']}")

        body = client.get(f"{API}/{created['id']}/history").json()
        assert body["total"] == 3
        assert [version["change"] for version in body["items"]] == [
            "CREATED",
            "UPDATED",
            "FORGOTTEN",
        ]
        assert body["items"][-1]["status"] == "DELETED"
        assert body["items"][0]["content"] == "temporary"


class TestHistory:
    def test_history_of_created_memory(self, client: TestClient) -> None:
        created = create(client, content="one")
        client.patch(f"{API}/{created['id']}", json={"content": "two"})

        body = client.get(f"{API}/{created['id']}/history").json()
        assert body["memory_id"] == created["id"]
        assert body["current_version"] == 2
        assert body["total"] == 2
        assert [version["version"] for version in body["items"]] == [1, 2]
        assert [version["change"] for version in body["items"]] == ["CREATED", "UPDATED"]
        assert [version["content"] for version in body["items"]] == ["one", "two"]

    def test_history_snapshot_contains_all_fields(self, client: TestClient) -> None:
        created = create(
            client,
            content="snapshot",
            memory_type="PROFILE",
            source="user:revanth",
            confidence=0.75,
            importance=0.25,
        )
        version = client.get(f"{API}/{created['id']}/history").json()["items"][0]
        assert version["content"] == "snapshot"
        assert version["memory_type"] == "PROFILE"
        assert version["source"] == "user:revanth"
        assert version["confidence"] == 0.75
        assert version["importance"] == 0.25
        assert version["status"] == "ACTIVE"
        assert version["recorded_at"]
