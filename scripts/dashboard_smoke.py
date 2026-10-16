"""Real-data dashboard smoke test (M7).

Serves the **production build** of the dashboard over HTTP, in front of a **real
uvicorn process**, and walks the mandatory demo flow against real data:

    Dashboard -> Memories -> Retrieval -> Context Compiler -> Agent -> Trace
              -> Evaluation

It is HTTP-backed rather than browser-driven on purpose: MemForge's project
already checks real integration at the transport level (``smoke_test.py`` for
REST, ``mcp_selftest.py`` for MCP), and this script extends the same idea to the
presentation layer without adding a browser stack. What it proves:

* the production build is servable, and every client-side route gets the SPA
  shell (a deep link must not 404);
* the bundle references the same-origin ``/api/v1`` prefix, so no secret and no
  absolute backend origin is baked in;
* every backend endpoint the dashboard consumes answers over real HTTP with the
  exact request bodies the dashboard sends;
* the flow a recruiter walks produces real data at each step: a ranked
  retrieval, a budgeted context, an agent turn, its trace, and the evaluation.

Every check is mandatory; the script exits non-zero if any fails.

    python scripts/dashboard_smoke.py

**Zero network.** The backend binds to ``127.0.0.1``, the dashboard is served
from the local build output, and nothing else is contacted.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST = REPO_ROOT / "dashboard" / "dist"
SOURCES = REPO_ROOT / "dashboard" / "src"
API = "/api/v1"
STARTUP_TIMEOUT_SECONDS = 30.0

#: The corpus. The same shape the REST smoke test seeds, so both agree on what a
#: realistic MemForge database contains.
MEMORY_SEED: tuple[dict[str, Any], ...] = (
    {
        "content": "User prefers Python for backend work.",
        "memory_type": "PROFILE",
        "source": "user:demo",
        "confidence": 0.95,
        "importance": 0.9,
    },
    {
        "content": "The user prefers python for backend development work",
        "memory_type": "OBSERVATION",
        "source": "agent:note",
        "confidence": 0.5,
        "importance": 0.3,
    },
    {
        "content": "preferred backend language = Python",
        "memory_type": "FACT",
        "confidence": 0.7,
        "importance": 0.5,
    },
    {
        "content": "preferred backend language = TypeScript",
        "memory_type": "FACT",
        "confidence": 0.5,
        "importance": 0.9,
    },
    {
        "content": "User recently started the MemForge context compiler.",
        "memory_type": "EVENT",
        "source": "agent:planner",
        "confidence": 0.9,
        "importance": 0.7,
    },
)

AGENT_REQUEST = (
    "Which backend language should I use for the MemForge service?\n"
    "remember: [FACT] MemForge uses Python and FastAPI."
)

#: The dashboard routes in demo order. ``/`` plus the seven other areas; the
#: memory-detail and trace-detail pages need an id and are covered below.
DASHBOARD_ROUTES: tuple[str, ...] = (
    "/",
    "/memories",
    "/retrieval",
    "/context",
    "/agent",
    "/traces",
    "/evaluation",
    "/metrics",
)

#: Patterns that would mean a credential reached a shipped browser bundle. The
#: *words* ``authorization`` and ``sk-`` legitimately appear in the dashboard's
#: redaction deny-list, so the markers below are shaped like a header being
#: attached or a key being read, not like the bare word.
BUNDLE_SECRET_MARKERS: tuple[str, ...] = (
    '"authorization":',
    "'authorization':",
    "authorization: ",
    "x-api-key",
    "memforge_llm_api_key",
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "Bearer ${",
)

#: Same idea for anything the dashboard can read back from the API. Here the bare
#: word is fine: no backend response may contain it at all.
RESPONSE_SECRET_MARKERS: tuple[str, ...] = (
    "authorization",
    "api_key",
    "bearer ",
    "sk-",
    "password",
    "chain_of_thought",
)


class CheckFailed(AssertionError):
    """A mandatory dashboard smoke check did not hold."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`CheckFailed` with a readable message."""
    if not condition:
        raise CheckFailed(message)


def free_port() -> int:
    """Ask the OS for an unused localhost port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Backend:
    """A real uvicorn process on a throwaway SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Backend:
        environment = {
            **os.environ,
            "MEMFORGE_DATABASE_PATH": str(self.db_path),
            "MEMFORGE_LOG_LEVEL": "WARNING",
            "MEMFORGE_EMBEDDING_PROVIDER": "none",
            "PYTHONPATH": str(REPO_ROOT),
        }
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._await_health()
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            self.process.kill()
            self.process.wait(timeout=10)

    def _await_health(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                output = (self.process.stdout.read() if self.process.stdout else b"").decode()
                raise CheckFailed(f"uvicorn exited during startup:\n{output}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.0) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        raise CheckFailed("the backend did not become healthy in time")


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves ``dashboard/dist`` with an SPA fallback and a backend proxy.

    This is what any static host plus reverse proxy does in production, and it is
    what lets the real build talk to the real backend without CORS.
    """

    backend_base = ""

    def log_message(self, *_: Any) -> None:  # pragma: no cover - quiet server
        """Silence the default stderr access log."""

    def _resolve(self) -> Path:
        """Map a request path onto a built file, falling back to the shell."""
        relative = self.path.split("?", 1)[0].lstrip("/")
        candidate = (DIST / relative).resolve() if relative else DIST / "index.html"
        if candidate.is_file():
            return candidate
        index = DIST / "index.html"
        return index if index.is_file() else candidate

    def do_GET(self) -> None:  # stdlib handler naming
        path = self.path.split("?", 1)[0]
        if path == "/health" or path.startswith(f"{API}/"):
            self._proxy("GET", None)
            return
        target = self._resolve()
        if not target.is_file():
            self.send_error(404, "asset not found")
            return
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(target)))
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # stdlib handler naming
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else None
        self._proxy("POST", body)

    def do_PATCH(self) -> None:  # stdlib handler naming
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else None
        self._proxy("PATCH", body)

    def do_DELETE(self) -> None:  # stdlib handler naming
        self._proxy("DELETE", None)

    def _proxy(self, method: str, body: bytes | None) -> None:
        url = f"{self.backend_base}{self.path}"
        request = urllib.request.Request(url, data=body, method=method)
        content_type = self.headers.get("Content-Type")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                self._relay(response.status, response.headers, response.read())
        except urllib.error.HTTPError as error:
            self._relay(error.code, error.headers, error.read())
        except (urllib.error.URLError, OSError) as error:
            payload = json.dumps({"error": {"code": "PROXY", "message": str(error)}}).encode()
            self._relay(502, {"Content-Type": "application/json"}, payload)

    def _relay(self, status: int, headers: Any, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        trace_id = headers.get("X-MemForge-Trace-Id")
        if trace_id:
            self.send_header("X-MemForge-Trace-Id", trace_id)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Dashboard:
    """A real static server for the production build."""

    def __init__(self, backend: Backend) -> None:
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        handler = type(
            "BoundDashboardHandler", (DashboardHandler,), {"backend_base": backend.base_url}
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> Dashboard:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()


class Response:
    """One HTTP response, decoded when it is JSON."""

    def __init__(self, status: int, text: str, headers: dict[str, str]) -> None:
        self.status = status
        self.text = text
        self.headers = headers
        try:
            self.json: Any = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            self.json = {}


class Session:
    """A tiny JSON client over urllib, so the script adds no dependency."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Response:
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=payload, method=method)
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8", errors="replace")
                headers = {key.lower(): value for key, value in response.headers.items()}
                return Response(response.status, raw, headers)
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            headers = {key.lower(): value for key, value in error.headers.items()}
            return Response(error.code, raw, headers)

    def get(self, path: str) -> Response:
        return self.request("GET", path)

    def post(self, path: str, body: dict[str, Any] | None = None) -> Response:
        return self.request("POST", path, body)

    def patch(self, path: str, body: dict[str, Any]) -> Response:
        return self.request("PATCH", path, body)

    def delete(self, path: str) -> Response:
        return self.request("DELETE", path)


class State:
    """Values the later checks need, produced by the earlier ones."""

    def __init__(self) -> None:
        self.active = 0
        self.archived = 0
        self.deleted = 0
        self.metrics: dict[str, Any] = {}
        self.memory_id = ""
        self.forgotten_id = ""
        self.search: dict[str, Any] = {}
        self.compile: dict[str, Any] = {}
        self.agent: dict[str, Any] = {}
        self.trace_id = ""
        self.evaluation: dict[str, Any] = {}


# -- the mandatory checks -----------------------------------------------------


def check_01_build_exists(client: Session, state: State) -> None:
    """The production build is present and servable."""
    require(DIST.is_dir(), "dashboard/dist is missing; run `npm run build` in dashboard/")
    index = DIST / "index.html"
    require(index.is_file(), "dashboard/dist/index.html is missing")
    assets = list((DIST / "assets").glob("*.js"))
    require(len(assets) > 0, "the build produced no JavaScript bundle")
    require(len(list((DIST / "assets").glob("*.css"))) > 0, "the build produced no stylesheet")

    response = client.get("/")
    require(response.status == 200, f"the dashboard did not answer at / (HTTP {response.status})")
    require('<div id="root">' in response.text, "the served shell has no React root element")
    print(f"    shell served, {len(assets)} bundle(s)")


def check_02_every_route_serves_the_spa(client: Session, state: State) -> None:
    """A deep link on every page returns the shell, not a 404."""
    for route in DASHBOARD_ROUTES:
        response = client.get(route)
        require(response.status == 200, f"route {route} returned {response.status}, expected 200")
        require(
            "text/html" in response.headers.get("content-type", ""),
            f"route {route} did not return HTML",
        )
        require('<div id="root">' in response.text, f"route {route} served something else")
    print(f"    {len(DASHBOARD_ROUTES)} client-side routes served")


def check_03_bundle_is_safe(client: Session, state: State) -> None:
    """No credential and no baked backend origin in the shipped build."""
    built = [*(DIST / "assets").glob("*.js"), *(DIST / "assets").glob("*.css"), DIST / "index.html"]
    for bundle in built:
        text = bundle.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        for marker in BUNDLE_SECRET_MARKERS:
            require(
                marker.lower() not in lowered,
                f"{bundle.name} contains {marker!r}: a credential reached the browser bundle",
            )
        require(
            "127.0.0.1:8000" not in lowered,
            f"{bundle.name} hard-codes a backend origin; use VITE_MEMFORGE_API_BASE",
        )
    sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").lower()
        for path in sorted(SOURCES.rglob("*.ts*"))
    )
    require(
        "memforge_llm_api_key" not in sources,
        "a dashboard source file names the API key environment variable",
    )
    require(
        '"/health"' in sources,
        "the client lost the out-of-prefix /health route the sidebar depends on",
    )
    print(f"    {len(built)} built file(s) free of credential-shaped content")


def check_04_health_through_the_proxy(client: Session, state: State) -> None:
    """``/health`` is served outside the API prefix, as the client expects."""
    response = client.get("/health")
    require(response.status == 200, f"the proxied health check returned {response.status}")
    require(response.json.get("status") == "ok", f"unexpected health {response.json!r}")
    require(response.json.get("storage") == "sqlite", f"unexpected storage {response.json!r}")


def check_05_dashboard_counters(client: Session, state: State) -> None:
    """Dashboard home: every counter it shows is backed by a real response."""
    counts = {}
    for status_filter in ("ACTIVE", "ARCHIVED", "DELETED"):
        response = client.get(f"{API}/memories?status={status_filter}&limit=1&offset=0")
        require(
            response.status == 200, f"memory count for {status_filter} returned {response.status}"
        )
        require(
            isinstance(response.json.get("total"), int),
            f"{status_filter} total is not a count",
        )
        counts[status_filter.lower()] = response.json["total"]
    state.active, state.archived, state.deleted = (
        counts["active"],
        counts["archived"],
        counts["deleted"],
    )

    response = client.get(f"{API}/metrics")
    require(response.status == 200, f"metrics returned {response.status}")
    require(response.json.get("event_count", 0) >= 0, "metrics has no event count")
    state.metrics = response.json
    print(f"    active={state.active} archived={state.archived} deleted={state.deleted}")


def check_06_memory_explorer(client: Session, state: State) -> None:
    """Memories page: a real list and a real CREATED/UPDATED/FORGOTTEN chain."""
    created: list[str] = []
    for payload in MEMORY_SEED:
        response = client.post(f"{API}/memories", payload)
        require(response.status == 201, f"a seeded memory was not created (HTTP {response.status})")
        created.append(response.json["id"])
    # The profile memory is the one with near-duplicates; the event memory is the
    # one whose history this check drives to a FORGOTTEN version.
    state.memory_id = created[0]
    state.forgotten_id = created[4]
    require(bool(state.memory_id and state.forgotten_id), "no memory was created")

    response = client.get(f"{API}/memories?limit=25&offset=0&status=ACTIVE")
    require(response.status == 200, "the memory list did not answer")
    require(
        response.json["total"] == len(MEMORY_SEED),
        f"expected {len(MEMORY_SEED)} active memories, got {response.json['total']}",
    )

    response = client.patch(
        f"{API}/memories/{state.forgotten_id}", {"importance": 1.0, "source": "user:revanth"}
    )
    require(response.status == 200, f"the update failed (HTTP {response.status})")
    require(
        response.json["version"] == 2,
        f"the update did not append a version: v{response.json['version']}",
    )
    response = client.delete(f"{API}/memories/{state.forgotten_id}")
    require(response.status == 204, f"forgetting failed with HTTP {response.status}")

    response = client.get(f"{API}/memories/{state.forgotten_id}/history")
    require(response.status == 200, "history did not answer after the forget")
    require(response.json["total"] == 3, f"expected 3 versions, got {response.json['total']}")
    changes = [item["change"] for item in response.json["items"]]
    require(
        changes == ["CREATED", "UPDATED", "FORGOTTEN"],
        f"unexpected lifecycle: {changes}",
    )

    # The dashboard surfaces the forgotten memory's history even though the
    # memory itself is gone from reads.
    response = client.get(f"{API}/memories/{state.forgotten_id}")
    require(
        response.status == 404 and response.json["error"]["code"] == "MEMORY_NOT_FOUND",
        "a forgotten memory still answers GET /memories/{id}",
    )
    print(f"    {len(MEMORY_SEED)} memories, history {changes}, forgotten read -> 404")


def check_07_memory_detail_panels(client: Session, state: State) -> None:
    """Memory detail: the six panels the page renders, all answering."""
    profile = state.memory_id
    for suffix in ("", "/intelligence", "/duplicates", "/conflicts", "/relationships"):
        response = client.get(f"{API}/memories/{profile}{suffix}")
        require(response.status == 200, f"memory panel {suffix or '/'} returned {response.status}")
        require(response.json, f"memory panel {suffix or '/'} returned no payload")

    response = client.get(f"{API}/memories/{profile}/history")
    require(response.status == 200, f"history did not answer: {response.status}")

    response = client.get(f"{API}/memories/{profile}/intelligence")
    intelligence = response.json
    for field in ("importance_score", "effective_priority", "decay_multiplier", "access_count"):
        require(field in intelligence, f"intelligence is missing {field}")
    require(intelligence["importance_factors"], "importance has no factor breakdown")

    response = client.get(f"{API}/memories/{profile}/duplicates")
    require(response.json["items"], "the seeded near-duplicate pair was not detected")
    print(
        f"    importance={intelligence['importance_score']:.4f} "
        f"priority={intelligence['effective_priority']:.4f} "
        f"duplicates={response.json['total']}"
    )


def check_08_retrieval_lab(client: Session, state: State) -> None:
    """Retrieval Lab: a real ranking with the explanation that produced it."""
    response = client.post(
        f"{API}/memories/search", {"query": "python backend language", "top_k": 5}
    )
    require(response.status == 200, f"search failed: {response.text}")
    body = response.json
    require(body["results"], "the search returned nothing for a seeded corpus")
    first = body["results"][0]
    for field in (
        "final_score",
        "relevance",
        "effective_priority",
        "freshness",
        "confidence",
        "importance",
    ):
        require(field in first, f"the top result is missing {field}")
    require(first["final_score"] > 0.0, "the top result has no score")
    require(first["explanation"], "the top result carries no explanation")
    require(first["factors"], "the top result has no factor breakdown")
    require(body["latency_ms"] >= 0.0, "the search reported a negative latency")
    state.search = body
    print(f"    {len(body['results'])} result(s), top score {first['final_score']:.4f}")


def check_09_context_compiler(client: Session, state: State) -> None:
    """Context Compiler: the hard budget holds and every candidate is decided."""
    response = client.post(
        f"{API}/context/compile",
        {"query": "python backend language", "token_budget": 160, "top_k": 10},
    )
    require(response.status == 200, f"compilation failed: {response.text}")
    body = response.json
    require(body["within_budget"] is True, "the compiler reported a budget violation")
    require(
        body["estimated_tokens"] <= body["token_budget"],
        f"{body['estimated_tokens']} estimated tokens exceed a {body['token_budget']} budget",
    )
    require(body["remaining_tokens"] >= 0, "the remaining budget is negative")
    require(
        body["candidates_considered"] == len(body["selection_trace"]),
        "not every candidate has exactly one decision",
    )
    require(body["selected_count"] == len(body["blocks"]), "selected_count disagrees with blocks")
    require(body["context_text"].strip(), "the compiled context is empty")
    require(body["explanation"]["estimator"], "the explanation names no estimator")
    state.compile = body
    print(
        f"    budget={body['token_budget']} used={body['estimated_tokens']} "
        f"selected={body['selected_count']} suppressed={body['suppressed_count']}"
    )


def check_10_agent_playground(client: Session, state: State) -> None:
    """Agent Playground: one real turn, with the trace id it produced."""
    response = client.post(
        f"{API}/agent/respond",
        {
            "request": AGENT_REQUEST,
            "token_budget": 300,
            "top_k": 10,
            "extract_candidates": True,
            "store_candidates": False,
        },
    )
    require(response.status == 200, f"the agent turn failed: {response.text}")
    body = response.json
    require(body["memory_used"] is True, "the turn used no memory")
    require(body["memory_gate"]["use"] == "used", f"the gate did not fire: {body['memory_gate']}")
    require(body["answer"].strip(), "the turn produced no answer")
    require(
        body["context_estimated_tokens"] <= body["context_token_budget"],
        "the agent context broke its budget",
    )
    require(body["stored_memory_ids"] == [], "candidates were stored without permission")
    require(body["usage"] is None, "the mock provider reported usage; it must not")
    trace_id = response.headers.get("x-memforge-trace-id", "")
    require(bool(trace_id), "the agent response carried no trace id")
    state.agent = body
    state.trace_id = trace_id
    print(f"    provider={body['provider']} blocks={len(body['context_blocks'])} trace={trace_id}")


def check_11_trace_viewer(client: Session, state: State) -> None:
    """Trace Viewer: the recorded lifecycle, read back."""
    response = client.get(f"{API}/traces?limit=25")
    require(response.status == 200, "the trace list did not answer")
    require(response.json["traces"], "no trace was listed")
    require(
        state.trace_id in {item["trace_id"] for item in response.json["traces"]},
        "the turn's trace is missing from the list",
    )

    response = client.get(f"{API}/traces/{state.trace_id}")
    require(response.status == 200, f"the trace is not readable (HTTP {response.status})")
    require(response.json["summary"]["succeeded"] is True, "a successful turn summarised as failed")

    response = client.get(f"{API}/traces/{state.trace_id}/events")
    require(response.status == 200, "the trace events are not readable")
    events = response.json
    require(events, "the trace recorded no events")
    sequences = [event["sequence"] for event in events]
    require(
        sequences == list(range(len(sequences))), f"sequence numbers are not total: {sequences}"
    )

    operations = {event["operation"] for event in events}
    for expected in (
        "agent.request",
        "memory.retrieval",
        "context.compile",
        "llm.generate",
        "agent.response",
    ):
        require(
            expected in operations, f"the trace does not report {expected}: {sorted(operations)}"
        )

    rendered = json.dumps(events).lower()
    for marker in RESPONSE_SECRET_MARKERS:
        require(marker not in rendered, f"a trace payload leaked {marker!r}")
    print(f"    {len(events)} events, operations={sorted(operations)}")


def check_12_evaluation_page(client: Session, state: State) -> None:
    """Evaluation: a real, measured run — reported exactly as it came."""
    response = client.get(f"{API}/evaluations/latest")
    require(response.status == 200, f"the evaluation endpoint failed (HTTP {response.status})")
    body = response.json
    require(body["reference"] == "memory-core@v1", f"unexpected dataset {body['reference']!r}")
    require(body["total_cases"] > 0, "the evaluation ran no case")
    require(
        body["passed_cases"] + body["failed_cases"] == body["total_cases"],
        "case counts do not add up",
    )
    require(0.0 <= body["overall"] <= 1.0, f"the overall score {body['overall']} is outside [0, 1]")
    require(body["metrics"], "no metric was reported")
    require(len(body["metrics"]) == 12, f"expected 12 metrics, got {len(body['metrics'])}")
    for metric in body["metrics"]:
        require(metric["definition"], f"metric {metric['name']} carries no definition")
        require(metric["denominator"] > 0, f"metric {metric['name']} has no measured denominator")
    require(
        len(body["cases"]) == body["failed_cases"],
        "the failing-case list does not match the failed count",
    )
    state.evaluation = body
    precision = next(m for m in body["metrics"] if m["name"] == "retrieval_precision")
    print(
        f"    {body['reference']} cases={body['total_cases']} overall={body['overall']:.5f} "
        f"precision={precision['score']:.5f} "
        f"({precision['numerator']}/{precision['denominator']})"
    )


def check_13_metrics_page(client: Session, state: State) -> None:
    """Metrics page: aggregate counters, derived from real events."""
    response = client.get(f"{API}/metrics")
    require(response.status == 200, "the metrics endpoint did not answer")
    metrics = response.json
    for group in ("traces", "operations", "statuses", "memory", "context", "provider"):
        require(group in metrics, f"metrics is missing the {group} group")
        require(
            all(isinstance(value, int) for value in metrics[group].values()),
            f"{group} holds a non-count value",
        )
    require(metrics["usage_reported"] == 0, "the mock provider must never report usage")
    require(
        sum(metrics["statuses"].values()) == metrics["event_count"],
        "status counts do not add up to the event count",
    )
    print(
        f"    events={metrics['event_count']} operations={len(metrics['operations'])} "
        f"usage_reported={metrics['usage_reported']}"
    )


def check_14_full_demo_flow(client: Session, state: State) -> None:
    """The recruiter path, over real HTTP, in demo order."""
    flow: list[tuple[str, bool]] = []

    response = client.get(f"{API}/memories?status=ACTIVE&limit=1&offset=0")
    flow.append(("Dashboard", response.status == 200 and "total" in response.json))

    response = client.get(f"{API}/memories?limit=1&offset=0")
    flow.append(("Memories", response.status == 200 and "items" in response.json))

    response = client.post(
        f"{API}/memories/search", {"query": "python backend language", "top_k": 3}
    )
    flow.append(("Retrieval", response.status == 200 and "candidates_returned" in response.json))

    response = client.post(
        f"{API}/context/compile", {"query": "python backend language", "token_budget": 200}
    )
    flow.append(
        ("Context Compiler", response.status == 200 and response.json["within_budget"] is True)
    )

    response = client.post(f"{API}/agent/respond", {"request": AGENT_REQUEST, "token_budget": 300})
    flow.append(("Agent", response.status == 200 and bool(response.json["answer"])))

    response = client.get(f"{API}/traces/{state.trace_id}")
    flow.append(("Trace", response.status == 200 and response.json["event_count"] > 0))

    response = client.get(f"{API}/evaluations/latest")
    flow.append(("Evaluation", response.status == 200 and response.json["total_cases"] > 0))

    broken = [name for name, ok in flow if not ok]
    require(not broken, f"the demo flow broke at: {', '.join(broken)}")
    print("    " + " -> ".join(name for name, _ in flow))


def check_15_no_secrets_in_any_response(client: Session, state: State) -> None:
    """Nothing the dashboard can read back carries a credential."""
    endpoints = (
        ("GET", "/health"),
        ("GET", f"{API}/memories?limit=5"),
        ("GET", f"{API}/memories/{state.memory_id}/history"),
        ("GET", f"{API}/memories/{state.memory_id}/intelligence"),
        ("GET", f"{API}/metrics"),
        ("GET", f"{API}/traces?limit=10"),
        ("GET", f"{API}/evaluations/latest"),
    )
    for method, path in endpoints:
        response = client.request(method, path)
        require(response.status == 200, f"{path} returned {response.status}")
        lowered = response.text.lower()
        for marker in RESPONSE_SECRET_MARKERS:
            require(marker not in lowered, f"{path} leaked {marker!r}")
    print(f"    {len(endpoints)} endpoint response(s) free of credential-shaped content")


def check_16_response_shapes_match_the_client(client: Session, state: State) -> None:
    """Every envelope the dashboard reads has the keys its page reads.

    The frontend validates each response against a declared shape, and a shape
    that disagrees with the backend produces a "malformed API response" state
    rather than data. The frontend's own fixtures cannot catch that drift, so the
    keys each page depends on are asserted here against the **real** responses.
    """
    expectations: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
        (
            "GET",
            f"{API}/memories?limit=1&offset=0",
            ("items", "total", "limit", "offset"),
            "Memory explorer page",
        ),
        (
            "GET",
            f"{API}/memories/{state.memory_id}",
            (
                "id",
                "content",
                "memory_type",
                "source",
                "confidence",
                "importance",
                "status",
                "created_at",
                "updated_at",
                "version",
            ),
            "Memory detail page",
        ),
        (
            "GET",
            f"{API}/memories/{state.memory_id}/history",
            ("memory_id", "current_version", "total", "items"),
            "Version history panel",
        ),
        (
            "GET",
            f"{API}/memories/{state.memory_id}/intelligence",
            (
                "importance_score",
                "importance_factors",
                "effective_priority",
                "base_priority",
                "decay_multiplier",
                "age_days",
                "access_count",
                "priority_explanation",
            ),
            "Intelligence panel",
        ),
        (
            "GET",
            f"{API}/memories/{state.memory_id}/duplicates",
            ("memory_id", "threshold", "total", "items"),
            "Duplicates panel",
        ),
        (
            "GET",
            f"{API}/memories/{state.memory_id}/conflicts",
            ("memory_id", "total", "items"),
            "Conflicts panel",
        ),
        (
            "GET",
            f"{API}/memories/{state.memory_id}/relationships",
            ("memory_id", "total", "items"),
            "Relationships panel",
        ),
        ("GET", f"{API}/traces?limit=5", ("count", "traces"), "Traces page"),
        (
            "GET",
            f"{API}/traces/{state.trace_id}",
            ("trace", "summary", "event_count"),
            "Trace page",
        ),
        (
            "GET",
            f"{API}/traces/{state.trace_id}/events",
            ("event_id", "sequence", "kind", "operation", "status", "metadata"),
            "Trace timeline",
        ),
        (
            "GET",
            f"{API}/metrics",
            ("traces", "operations", "statuses", "memory", "context", "provider", "event_count"),
            "Metrics page",
        ),
        (
            "GET",
            f"{API}/evaluations/latest",
            ("dataset", "reference", "total_cases", "overall", "categories", "metrics", "cases"),
            "Evaluation page",
        ),
        ("GET", "/health", ("status", "service", "version", "storage"), "Sidebar connection badge"),
    )

    for method, path, keys, consumer in expectations:
        response = client.request(method, path)
        require(response.status == 200, f"{path} returned {response.status}")
        payload = response.json
        if isinstance(payload, list):
            require(len(payload) > 0, f"{path} returned an empty list; {consumer} needs one record")
            payload = payload[0]
        require(isinstance(payload, dict), f"{path} did not return an object for {consumer}")
        missing = [key for key in keys if key not in payload]
        require(not missing, f"{path} is missing {missing}, which {consumer} reads")
    print(f"    {len(expectations)} response envelope(s) match what the pages read")


CHECKS: tuple[tuple[str, Callable[[Session, State], None]], ...] = (
    ("the production build exists", check_01_build_exists),
    ("every dashboard route serves the SPA", check_02_every_route_serves_the_spa),
    ("the bundle carries no secret or origin", check_03_bundle_is_safe),
    ("health answers through the proxy", check_04_health_through_the_proxy),
    ("memories page has a real version chain", check_06_memory_explorer),
    ("memory detail panels all answer", check_07_memory_detail_panels),
    ("dashboard counters are real", check_05_dashboard_counters),
    ("retrieval lab returns an explained ranking", check_08_retrieval_lab),
    ("context compiler respects its budget", check_09_context_compiler),
    ("agent playground runs a real turn", check_10_agent_playground),
    ("trace viewer reads the lifecycle back", check_11_trace_viewer),
    ("evaluation page reports a measured run", check_12_evaluation_page),
    ("metrics page reports real counters", check_13_metrics_page),
    ("the full demo flow walks end to end", check_14_full_demo_flow),
    ("no endpoint leaks a credential", check_15_no_secrets_in_any_response),
    ("every response envelope matches the client", check_16_response_shapes_match_the_client),
)


def run() -> int:
    """Run every mandatory check against the real backend and build."""
    if not DIST.is_dir():
        print("dashboard/dist is missing.")
        print("Build it first:  cd dashboard && npm install && npm run build")
        return 1

    state = State()
    failures: list[tuple[int, str, str]] = []

    with tempfile.TemporaryDirectory(prefix="memforge-dashboard-") as directory:
        db_path = Path(directory) / "dashboard.sqlite"
        with Backend(db_path) as backend, Dashboard(backend) as dashboard:
            client = Session(dashboard.base_url)
            print(f"MemForge dashboard smoke: {dashboard.base_url} -> {backend.base_url}")
            print(f"build={DIST} db={db_path.name}")
            print("=" * 78)
            for index, (title, check) in enumerate(CHECKS, start=1):
                try:
                    check(client, state)
                except CheckFailed as failure:
                    failures.append((index, title, str(failure)))
                    print(f"[{index:2}/{len(CHECKS)}] FAIL  {title}")
                    print(f"          {failure}")
                    continue
                print(f"[{index:2}/{len(CHECKS)}] PASS  {title}")

    print("=" * 78)
    passed = len(CHECKS) - len(failures)
    print(f"{passed}/{len(CHECKS)} mandatory dashboard checks passed")
    if failures:
        print("\nFailures:")
        for index, title, message in failures:
            print(f"  {index:2}. {title}: {message}")
        return 1
    print("Dashboard smoke test finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
