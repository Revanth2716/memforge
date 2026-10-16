"""Tests for the M6 REST endpoints: traces, events, metrics and evaluations.

Every route is read-only and returns only projections of what was recorded.
These tests assert the shape, the error contract and -- above all -- that
nothing credential-shaped can leave through them.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.agent import TRACE_HEADER
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.observability import router as observability_router

API = "/api/v1"
TRACES = f"{API}/traces"
METRICS = f"{API}/metrics"
EVALUATIONS = f"{API}/evaluations"

#: Substrings that must never appear in a trace or evaluation payload.
SECRET_MARKERS: tuple[str, ...] = (
    "authorization",
    "bearer ",
    "api_key",
    "api-key",
    "password",
    "secret",
)

QUESTION = "What do you know about my backend experience?\nremember: [FACT] The user likes SQLite."


def _seed(client: TestClient) -> None:
    """Store a small corpus through the public API."""
    for content, memory_type in (
        ("The user writes Python and SQL for backend services.", "FACT"),
        ("The user prefers the dark theme in every code editor.", "PREFERENCE"),
    ):
        response = client.post(
            f"{API}/memories", json={"content": content, "memory_type": memory_type}
        )
        assert response.status_code == 201


def _turn(client: TestClient, **overrides: object) -> str:
    """Run one agent turn and return its trace id."""
    payload: dict[str, object] = {
        "request": QUESTION,
        "token_budget": 300,
        "extract_candidates": True,
    }
    payload.update(overrides)
    response = client.post(f"{API}/agent/respond", json=payload)
    assert response.status_code == 200, response.text
    return str(response.headers[TRACE_HEADER])


class TestAgentTraceHeader:
    """A turn returns its trace id."""

    def test_the_response_carries_a_trace_id(self, client: TestClient) -> None:
        _seed(client)
        trace_id = _turn(client)
        assert trace_id
        detail = client.get(f"{TRACES}/{trace_id}")
        assert detail.status_code == 200


class TestTraceEndpoints:
    """``/traces`` and its sub-resources."""

    def test_an_empty_install_lists_no_traces(self, client: TestClient) -> None:
        body = client.get(TRACES).json()
        assert body["count"] == 0
        assert body["traces"] == []

    def test_traces_are_listed_newest_first_with_their_attributes(self, client: TestClient) -> None:
        _seed(client)
        trace_id = _turn(client)
        listing = client.get(TRACES).json()
        assert listing["count"] >= 1
        listed = {trace["trace_id"]: trace for trace in listing["traces"]}
        assert trace_id in listed
        entry = listed[trace_id]
        assert entry["kind"] == "agent_request"
        assert entry["operation"] == "agent.respond"
        assert entry["attributes"]["provider"] == "mock"
        assert entry["attributes"]["request_characters"] == len(QUESTION)

    def test_a_trace_can_be_filtered_by_kind(self, client: TestClient) -> None:
        _seed(client)
        _turn(client)
        client.post(
            f"{API}/memories/remember",
            json={"content": "A remembered fact about keyboards.", "memory_type": "FACT"},
        )
        agents = client.get(TRACES, params={"kind": "agent_request"}).json()
        assert agents["count"] >= 1
        assert all(trace["kind"] == "agent_request" for trace in agents["traces"])
        rest = client.get(TRACES, params={"kind": "api_request"}).json()
        assert rest["count"] >= 1
        assert all(trace["kind"] == "api_request" for trace in rest["traces"])

    def test_an_unknown_kind_is_rejected(self, client: TestClient) -> None:
        response = client.get(TRACES, params={"kind": "telepathy"})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "OBSERVABILITY_ERROR"

    def test_the_limit_is_validated(self, client: TestClient) -> None:
        assert client.get(TRACES, params={"limit": 0}).status_code == 422
        assert client.get(TRACES, params={"limit": 500}).status_code == 422

    def test_one_trace_carries_its_summary(self, client: TestClient) -> None:
        _seed(client)
        trace_id = _turn(client)
        body = client.get(f"{TRACES}/{trace_id}").json()
        summary = body["summary"]
        assert summary["trace_id"] == trace_id
        assert summary["succeeded"] is True
        assert summary["errors"] == []
        assert summary["event_count"] >= 10
        for operation in ("agent.request", "memory.retrieval", "context.compile", "llm.generate"):
            assert summary["operations"][operation] == "ok"
        assert summary["operations"]["memory.write"] == "skipped"
        assert summary["total_duration_ms"] >= 0.0

    def test_an_unknown_trace_returns_the_error_envelope(self, client: TestClient) -> None:
        response = client.get(f"{TRACES}/nope")
        assert response.status_code == 404
        envelope = response.json()["error"]
        assert envelope["code"] == "TRACE_NOT_FOUND"
        assert "nope" in envelope["message"]

    def test_events_are_returned_in_sequence_order(self, client: TestClient) -> None:
        _seed(client)
        trace_id = _turn(client)
        events = client.get(f"{TRACES}/{trace_id}/events").json()
        assert [event["sequence"] for event in events] == list(range(len(events)))
        assert events[0]["kind"] == "request"
        assert events[-1]["kind"] == "response"
        for event in events:
            assert event["trace_id"] == trace_id
            assert event["status"] in ("ok", "skipped", "error")
            if event["duration_ms"] is not None:
                assert event["duration_ms"] >= 0.0

    def test_events_of_an_unknown_trace_return_404(self, client: TestClient) -> None:
        response = client.get(f"{TRACES}/nope/events")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "TRACE_NOT_FOUND"

    def test_trace_payloads_carry_no_secret(self, client: TestClient) -> None:
        _seed(client)
        trace_id = _turn(client)
        rendered = json.dumps(
            [
                client.get(f"{TRACES}/{trace_id}").json(),
                client.get(f"{TRACES}/{trace_id}/events").json(),
            ]
        ).lower()
        for marker in SECRET_MARKERS:
            assert marker not in rendered, marker
        assert "remember:" not in rendered
        assert "backend experience" not in rendered


class TestMetricsEndpoint:
    """``/metrics``."""

    def test_an_empty_install_reports_zero_events(self, client: TestClient) -> None:
        body = client.get(METRICS).json()
        assert body["event_count"] == 0
        assert body["traces"] == {}
        assert body["usage_reported"] == 0

    def test_counters_come_from_the_recorded_events(self, client: TestClient) -> None:
        _seed(client)
        _turn(client)
        body = client.get(METRICS).json()
        assert body["event_count"] > 0
        assert body["traces"]["agent_request"] >= 1
        assert body["provider"]["calls"] >= 1
        assert body["context"]["selected"] >= 1
        assert body["memory"]["candidates"] >= 1
        assert sum(body["statuses"].values()) == body["event_count"]

    def test_usage_is_zero_when_no_provider_reported_any(self, client: TestClient) -> None:
        _seed(client)
        _turn(client)
        assert client.get(METRICS).json()["usage_reported"] == 0

    def test_the_openapi_schema_exposes_no_credential_field(self, client: TestClient) -> None:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        for name in ("MetricsResponse", "TraceEventResponse", "TraceSummaryResponse"):
            fields = " ".join(schemas[name]["properties"]).lower()
            for marker in SECRET_MARKERS:
                assert marker not in fields, f"{name} exposes {marker!r}"


class TestEvaluationEndpoints:
    """``/evaluations``."""

    def test_latest_reports_a_real_run(self, client: TestClient) -> None:
        body = client.get(f"{EVALUATIONS}/latest").json()
        assert body["reference"] == "memory-core@v1"
        assert body["total_cases"] >= 30
        assert body["passed"] is True
        assert body["failed_cases"] == 0
        assert body["cases"] == []
        assert set(body["categories"]) == {
            "agent",
            "conflict",
            "context",
            "duplicate",
            "lifecycle",
            "retrieval",
        }
        for category in body["categories"].values():
            assert category["passed"] == category["total"]

    def test_every_reported_metric_is_measured_and_defined(self, client: TestClient) -> None:
        body = client.get(f"{EVALUATIONS}/latest").json()
        assert body["metrics"]
        for metric in body["metrics"]:
            assert metric["denominator"] > 0
            assert metric["numerator"] <= metric["denominator"]
            assert 0.0 <= metric["score"] <= 1.0
            assert metric["definition"]

    def test_a_second_call_is_served_from_the_cache(self, client: TestClient) -> None:
        application = cast(FastAPI, client.app)
        first = client.get(f"{EVALUATIONS}/latest").json()
        runs = application.state.evaluation_registry.run_count
        second = client.get(f"{EVALUATIONS}/latest").json()
        assert application.state.evaluation_registry.run_count == runs
        assert first == second

    def test_the_reference_and_the_name_both_resolve(self, client: TestClient) -> None:
        latest = client.get(f"{EVALUATIONS}/latest").json()
        for reference in ("memory-core@v1", "memory-core", "memory-core-v1"):
            body = client.get(f"{EVALUATIONS}/{reference}").json()
            assert body["reference"] == latest["reference"]

    def test_a_threshold_is_accepted(self, client: TestClient) -> None:
        body = client.get(f"{EVALUATIONS}/latest", params={"threshold": 0.5}).json()
        assert body["threshold"] == 0.5
        assert body["passed"] is True

    def test_an_out_of_range_threshold_is_rejected(self, client: TestClient) -> None:
        assert client.get(f"{EVALUATIONS}/latest", params={"threshold": 2}).status_code == 422

    def test_an_unknown_dataset_returns_the_error_envelope(self, client: TestClient) -> None:
        response = client.get(f"{EVALUATIONS}/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "EVALUATION_NOT_FOUND"

    def test_the_evaluation_route_is_read_only(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path, operations in paths.items():
            if path.startswith(f"{API}/evaluations") or path.startswith(f"{API}/traces"):
                assert set(operations) <= {"get"}, f"{path} exposes {set(operations)}"


class TestRouting:
    """The new routers are mounted where the documentation says."""

    def test_both_routers_are_registered(self) -> None:
        assert {str(getattr(route, "path", "")) for route in observability_router.routes} >= {
            "/traces",
            "/traces/{trace_id}",
            "/traces/{trace_id}/events",
            "/metrics",
        }
        assert {str(getattr(route, "path", "")) for route in evaluation_router.routes} == {
            "/evaluations/latest",
            "/evaluations/{reference}",
        }

    def test_the_paths_are_tagged_and_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert paths[f"{API}/metrics"]["get"]["tags"] == ["observability"]
        assert paths[f"{API}/evaluations/latest"]["get"]["tags"] == ["evaluation"]
        for path, operations in paths.items():
            if path.startswith(f"{API}/traces"):
                for operation in operations.values():
                    assert operation["summary"]


class TestNoTraceEndpoint:
    """There is deliberately no way to fabricate a trace."""

    def test_writing_to_the_observability_surface_is_refused(self, client: TestClient) -> None:
        for method, url in (
            ("post", TRACES),
            ("delete", TRACES),
            ("post", f"{TRACES}/anything"),
            ("post", METRICS),
        ):
            response = getattr(client, method)(url)
            assert response.status_code == 405, f"{method.upper()} {url} was accepted"


@pytest.mark.parametrize("marker", SECRET_MARKERS)
def test_no_secret_marker_appears_in_any_m6_response(client: TestClient, marker: str) -> None:
    """Every M6 response, taken together, is free of credential-shaped text."""
    _seed(client)
    trace_id = _turn(client)
    payloads = [
        client.get(TRACES).text,
        client.get(f"{TRACES}/{trace_id}").text,
        client.get(f"{TRACES}/{trace_id}/events").text,
        client.get(METRICS).text,
    ]
    for payload in payloads:
        assert marker not in payload.lower(), marker
