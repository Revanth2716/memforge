"""Live HTTP smoke test for MemForge (M1 -> M6).

Starts a **real uvicorn server** on a throwaway SQLite file and exercises the
public API over HTTP with ``httpx`` -- no in-process test client, no mocks.

    python scripts/smoke_test.py

Every check is mandatory: the script exits non-zero if any of them fails.

**Zero network.** The server binds to ``127.0.0.1`` and the script talks to
nothing else. The default configuration has no embedding provider, so no
outbound call is possible; the checks that exercise hybrid mode assert that the
degradation is *reported* rather than faked.

M6 adds the traceable end-to-end flow (agent turn -> trace -> ordered events ->
metrics) and the evaluation endpoint. Latency values are read back and checked
for being *present and non-negative*, never compared against a fixed number:
they are measured at runtime and are not a benchmark.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
API = "/api/v1"
STARTUP_TIMEOUT_SECONDS = 30.0

#: The corpus: a profile, a near-duplicate, two facts and an unrelated note.
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
    {
        "content": "The staging cluster runs on Kubernetes with three worker nodes.",
        "memory_type": "OBSERVATION",
        "source": "agent:note",
        "confidence": 0.5,
        "importance": 0.2,
    },
)

QUERY = "python backend language memforge"
COMPILE_URL = f"{API}/context/compile"


class CheckFailed(AssertionError):
    """A mandatory smoke check did not hold."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`CheckFailed` with a readable message."""
    if not condition:
        raise CheckFailed(message)


def free_port() -> int:
    """Ask the OS for an unused localhost port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Server:
    """A real uvicorn process, started and stopped by this script."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Server:
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
        """Block until the server answers, or fail with its captured output."""
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                output = (self.process.stdout.read() if self.process.stdout else b"").decode()
                raise CheckFailed(f"uvicorn exited during startup:\n{output}")
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.1)
        raise CheckFailed("the server did not become healthy in time")


class Corpus:
    """Created memory ids, keyed by the role they play in the smoke test."""

    def __init__(self, client: httpx.Client) -> None:
        self.ids: dict[str, str] = {}
        for index, payload in enumerate(MEMORY_SEED):
            response = client.post(f"{API}/memories", json=payload)
            require(response.status_code == 201, f"memory {index} was not created")
            self.ids[f"memory_{index}"] = response.json()["id"]

    @property
    def profile(self) -> str:
        """The profile memory."""
        return self.ids["memory_0"]

    @property
    def near_duplicate(self) -> str:
        """The near-duplicate of the profile."""
        return self.ids["memory_1"]

    @property
    def conflict_winner(self) -> str:
        """The memory that wins the conflict resolution."""
        return self.ids["memory_3"]


# -- the mandatory checks -----------------------------------------------------


def check_01_health(client: httpx.Client) -> None:
    """The API is up and reports its storage."""
    response = client.get("/health")
    require(response.status_code == 200, "health check failed")
    body = response.json()
    require(body["status"] == "ok", f"unexpected health status {body['status']!r}")
    require(body["storage"] == "sqlite", f"unexpected storage {body['storage']!r}")


def check_02_memory_set(client: httpx.Client) -> Corpus:
    """A realistic memory set can be created."""
    corpus = Corpus(client)
    require(len(corpus.ids) == len(MEMORY_SEED), "not every memory was created")
    listing = client.get(f"{API}/memories", params={"limit": 50}).json()
    require(
        listing["total"] == len(MEMORY_SEED),
        f"expected {len(MEMORY_SEED)} active memories, got {listing['total']}",
    )
    return corpus


def check_03_relationships(client: httpx.Client, corpus: Corpus) -> None:
    """M2 records the duplicate pair and resolves the conflict in place."""
    duplicates = client.get(f"{API}/memories/{corpus.profile}/duplicates")
    require(duplicates.status_code == 200, "duplicate detection failed")
    matches = duplicates.json()["items"]
    require(matches, "the near-duplicate pair was not detected")
    require(
        any(match["memory_id"] == corpus.near_duplicate for match in matches),
        "the expected near-duplicate was not reported",
    )

    conflicts = client.get(f"{API}/memories/{corpus.conflict_winner}/conflicts")
    require(conflicts.status_code == 200, "conflict detection failed")
    require(conflicts.json()["items"], "the conflicting pair was not detected")

    resolution = client.post(
        f"{API}/memories/{corpus.conflict_winner}/conflicts/{corpus.ids['memory_2']}/resolve"
    )
    require(resolution.status_code == 200, f"resolution failed: {resolution.text}")
    body = resolution.json()
    require(
        body["winner_memory_id"] == corpus.conflict_winner,
        "the deterministic policy picked a different winner",
    )
    require(body["loser_memory_id"] == corpus.ids["memory_2"], "an unexpected loser")
    links = client.get(f"{API}/memories/{corpus.conflict_winner}/relationships").json()
    require(
        any(item["relationship"] == "SUPERSEDES" for item in links["items"]),
        "the SUPERSEDES relationship was not recorded",
    )


def check_04_retrieval(client: httpx.Client, corpus: Corpus) -> None:
    """M3 retrieval still works and is the compiler's only source of candidates."""
    response = client.post(f"{API}/memories/search", json={"query": QUERY, "top_k": 10})
    require(response.status_code == 200, f"search failed: {response.text}")
    body = response.json()
    require(body["candidates_considered"] > 0, "retrieval considered no candidate")
    require(body["retrieval_mode"] == "lexical", "the default path must stay lexical")
    first = body["results"][0]
    require(first["final_score"] > 0.0, "the top result has no score")
    require(first["explanation"], "the top result carries no explanation")
    require(corpus.profile in {item["memory_id"] for item in body["results"]}, "profile missing")


def check_05_compile(client: httpx.Client) -> dict[str, Any]:
    """Context compilation returns the documented contract."""
    response = client.post(COMPILE_URL, json={"query": QUERY, "token_budget": 160, "top_k": 10})
    require(response.status_code == 200, f"compilation failed: {response.text}")
    body = response.json()
    for field in (
        "query",
        "token_budget",
        "estimated_tokens",
        "remaining_tokens",
        "candidates_considered",
        "selected_count",
        "suppressed_count",
        "retrieval_mode",
        "blocks",
        "selection_trace",
        "suppressed_candidates",
        "explanation",
    ):
        require(field in body, f"the response is missing {field!r}")
    require(body["selected_count"] == len(body["blocks"]), "selected_count disagrees with blocks")
    require(
        body["suppressed_count"] == len(body["suppressed_candidates"]),
        "suppressed_count disagrees with suppressed_candidates",
    )
    require(
        body["candidates_considered"] == len(body["selection_trace"]),
        "every candidate must have exactly one trace entry",
    )
    require(body["estimated_tokens"] > 0, "the compiled context is empty")
    return body


def check_06_selected(client: httpx.Client, body: dict[str, Any]) -> None:
    """The selected memories are listed with their utility."""
    require(body["blocks"], "nothing was selected")
    for block in body["blocks"]:
        require(block["utility_score"] > 0.0, f"block {block['memory_id']} has no utility")
        require(block["rendered_content"].strip(), f"block {block['memory_id']} has no content")
    sections = [block["section"] for block in body["blocks"]]
    require(len(set(sections)) == len(sections), "two blocks claim the same section")
    print(f"    selected {len(body['blocks'])} block(s):")
    for block in body["blocks"]:
        print(
            f"      [{block['section']:<14}] utility={block['utility_score']:.4f} "
            f"tokens={block['estimated_tokens']:>3} {block['rendered_content'][:48]}"
        )


def check_07_duplicate_suppression(
    client: httpx.Client, body: dict[str, Any], corpus: Corpus
) -> None:
    """A near-duplicate is suppressed, and the representative is kept."""
    entry = next(
        (
            item
            for item in body["suppressed_candidates"]
            if item["memory_id"] == corpus.near_duplicate
        ),
        None,
    )
    require(entry is not None, "the near-duplicate was not suppressed")
    require(entry["selected"] is False, "a suppressed candidate is marked selected")
    require(entry["reason"] == "duplicate_of", f"unexpected reason {entry['reason']!r}")
    require(
        entry["related_memory_id"] == corpus.profile,
        "the suppression does not name the representative",
    )
    require(
        corpus.near_duplicate not in {block["memory_id"] for block in body["blocks"]},
        "the duplicate reached the context",
    )
    require(
        corpus.profile in {block["memory_id"] for block in body["blocks"]},
        "the representative is missing from the context",
    )
    # Suppression is a context decision, never a deletion.
    stored = client.get(f"{API}/memories/{corpus.near_duplicate}")
    require(stored.status_code == 200, "the suppressed memory was destroyed")
    print(f"    duplicate_of {entry['related_memory_id']} <- {entry['memory_id']}")


def check_08_conflict_suppression(client: httpx.Client, corpus: Corpus) -> None:
    """A superseded memory never reaches the context as authoritative.

    M2 archived the loser, so the default (ACTIVE) compilation simply never
    sees it. Asking explicitly for archived memories proves the stronger
    property: even when the loser *is* a candidate, the compiler suppresses it
    and names the winner.
    """
    loser = corpus.ids["memory_2"]
    default = client.post(
        COMPILE_URL, json={"query": QUERY, "token_budget": 160, "top_k": 10}
    ).json()
    require(
        loser not in {block["memory_id"] for block in default["blocks"]},
        "the superseded memory reached the context",
    )
    require(
        loser not in {entry["memory_id"] for entry in default["selection_trace"]},
        "the archived loser should not even be a candidate",
    )

    archived = client.post(
        COMPILE_URL,
        json={"query": QUERY, "token_budget": 160, "top_k": 10, "status": "ARCHIVED"},
    ).json()
    entry = next(
        (item for item in archived["suppressed_candidates"] if item["memory_id"] == loser),
        None,
    )
    require(entry is not None, "the superseded memory was not reported")
    require(entry["reason"] == "conflict_superseded", f"unexpected reason {entry['reason']!r}")
    require(
        entry["related_memory_id"] == corpus.conflict_winner,
        "the suppression does not name the conflict winner",
    )
    require(
        loser not in {block["memory_id"] for block in archived["blocks"]},
        "the superseded memory reached the context",
    )
    require(
        client.get(f"{API}/memories/{loser}").status_code == 200,
        "the superseded memory was destroyed",
    )
    print(f"    conflict_superseded by {entry['related_memory_id']} <- {loser}")


def check_09_budget_and_remaining(client: httpx.Client) -> dict[str, Any]:
    """The hard budget holds and the remaining allowance is reported."""
    response = client.post(COMPILE_URL, json={"query": QUERY, "token_budget": 160, "top_k": 10})
    body = response.json()
    require(body["within_budget"] is True, "the compiler reported a budget violation")
    require(
        body["estimated_tokens"] <= body["token_budget"],
        f"{body['estimated_tokens']} estimated tokens exceed a {body['token_budget']} budget",
    )
    require(
        body["remaining_tokens"] == body["token_budget"] - body["estimated_tokens"],
        "remaining_tokens is inconsistent",
    )
    require(body["remaining_tokens"] >= 0, "the remaining budget is negative")
    require(
        0.0 <= body["utilization"] <= 1.0,
        f"utilization {body['utilization']} is outside [0, 1]",
    )
    print(
        f"    budget={body['token_budget']} used={body['estimated_tokens']} "
        f"remaining={body['remaining_tokens']}"
    )
    return body


def check_10_budget_sweep(client: httpx.Client) -> None:
    """Across a range of budgets the limit is never exceeded."""
    for budget in (1, 5, 10, 25, 60, 120, 160, 400, 1200, 16000):
        response = client.post(
            COMPILE_URL, json={"query": QUERY, "token_budget": budget, "top_k": 10}
        )
        require(response.status_code == 200, f"budget {budget} was rejected")
        body = response.json()
        require(
            body["estimated_tokens"] <= budget,
            f"budget {budget}: {body['estimated_tokens']} estimated tokens overflowed it",
        )
        require(body["remaining_tokens"] >= 0, f"budget {budget}: negative remainder")
        skipped = [
            item
            for item in body["suppressed_candidates"]
            if item["reason"] == "insufficient_budget"
        ]
        for item in skipped:
            require(
                item["detail"],
                f"budget {budget}: a skipped memory was reported without a reason",
            )
    print("    budgets 1..16000 all respected")


def check_11_provenance(client: httpx.Client, body: dict[str, Any]) -> None:
    """Every block traces back to a stored, unmodified memory."""
    rendered = set()
    for block in body["blocks"]:
        stored = client.get(f"{API}/memories/{block['memory_id']}")
        require(stored.status_code == 200, "a block references an unknown memory")
        memory = stored.json()
        require(block["content"] == memory["content"], "the original content was altered")
        require(block["source"] == memory["source"], "the source was altered")
        require(block["memory_type"] == memory["memory_type"], "the memory type was altered")
        require(block["confidence"] == memory["confidence"], "confidence was altered")
        require(block["retrieval_score"] > 0.0, "no retrieval score")
        require(block["utility_score"] > 0.0, "no selection score")
        require("applied" in block["compression"], "no compression record")
        require(
            block["compression"]["original_text"] == memory["content"],
            "the compression record lost the original text",
        )
        rendered.add(f"- {block['rendered_content']}")
    for line in body["context_text"].splitlines():
        if line.startswith("- "):
            require(line in rendered, f"context line without provenance: {line!r}")
    print(f"    {len(body['blocks'])} block(s) traced back to their stored memory")


def check_12_explanation(client: httpx.Client, body: dict[str, Any]) -> None:
    """The explanation is machine-readable and honest."""
    explanation = body["explanation"]
    require(explanation["estimator"] == "memforge-char-word/v1", "unexpected estimator")
    require("chars_per_token" in explanation["estimator_formula"], "the formula is not documented")
    require(
        "utility = value" in explanation["utility_formula"], "the utility model is undocumented"
    )
    require(explanation["ordering_policy"] == "section-then-utility/v1", "unexpected ordering")
    require(
        explanation["section_order"]
        == ["PROFILE", "FACTS", "RECENT_EVENTS", "TASKS", "OBSERVATIONS"],
        "unexpected section order",
    )
    require(
        sum(explanation["selection_counts"].values()) == body["candidates_considered"],
        "the selection counts do not add up to the candidate count",
    )
    require(
        any("not provider tokenisations" in note for note in explanation["notes"]),
        "the response does not state that token counts are estimates",
    )
    require(
        any("no LLM" in note for note in explanation["notes"]),
        "the response does not state that no LLM was used",
    )
    for entry in body["selection_trace"]:
        require(entry["detail"], f"memory {entry['memory_id']} has no explanation")
    print(
        "    reasons: "
        + ", ".join(f"{k}={v}" for k, v in explanation["selection_counts"].items() if v)
    )


def check_13_determinism(client: httpx.Client) -> None:
    """Repeated compilation returns byte-identical output."""
    payload = {"query": QUERY, "token_budget": 160, "top_k": 10}
    responses = [client.post(COMPILE_URL, json=payload) for _ in range(3)]
    bodies = [response.json() for response in responses]
    first = json.dumps(bodies[0], sort_keys=True)
    for index, body in enumerate(bodies[1:], start=2):
        require(
            json.dumps(body, sort_keys=True) == first,
            f"compilation {index} differed from compilation 1",
        )
    print("    3 identical compilations")


def check_14_validation(client: httpx.Client) -> None:
    """Invalid requests use the documented error envelope."""
    cases: tuple[tuple[dict[str, Any], int, str], ...] = (
        ({}, 422, "VALIDATION_ERROR"),
        ({"query": ""}, 422, "VALIDATION_ERROR"),
        ({"query": QUERY, "token_budget": 0}, 422, "VALIDATION_ERROR"),
        ({"query": QUERY, "token_budget": -5}, 422, "VALIDATION_ERROR"),
        ({"query": QUERY, "token_budget": 16001}, 400, "CONTEXT_VALIDATION_ERROR"),
        ({"query": QUERY, "unknown_field": 1}, 422, "VALIDATION_ERROR"),
    )
    for body, status, code in cases:
        response = client.post(COMPILE_URL, json=body)
        require(
            response.status_code == status,
            f"{body} returned {response.status_code}, expected {status}",
        )
        envelope = response.json()["error"]
        require(envelope["code"] == code, f"{body} returned {envelope['code']}, expected {code}")
        require(envelope["message"], "an error envelope without a message")
    print(f"    {len(cases)} malformed request(s) rejected with the error envelope")


def check_15_hybrid_degradation(client: httpx.Client) -> None:
    """Hybrid without a provider degrades loudly instead of pretending."""
    response = client.post(
        COMPILE_URL, json={"query": QUERY, "token_budget": 160, "retrieval_mode": "hybrid"}
    )
    body = response.json()
    require(body["requested_retrieval_mode"] == "hybrid", "the requested mode was lost")
    require(body["retrieval_mode"] == "lexical", "the degraded mode was not reported")
    require(body["semantic_available"] is False, "semantic availability was misreported")
    require(body["degraded_reason"], "no degradation reason was reported")
    require(body["within_budget"] is True, "the budget broke during degradation")
    print(f"    degraded: {body['degraded_reason']}")


def check_16_default_offline_path(client: httpx.Client) -> None:
    """The default path needs no provider, no key and no network."""
    response = client.post(COMPILE_URL, json={"query": QUERY})
    body = response.json()
    require(body["token_budget"] == 1200, "the default budget is not the configured one")
    require(body["semantic_available"] is False, "an embedding provider was configured")
    require(body["within_budget"] is True, "the default request broke the budget")
    print(f"    default budget={body['token_budget']} used={body['estimated_tokens']}")


# -- M5: agent + custom provider safety ---------------------------------------

AGENT_URL = "/api/v1/agent/respond"
REMEMBER_URL = "/api/v1/memories/remember"

AGENT_REQUEST = (
    "Which backend language should I use for the MemForge service?\n"
    "remember: [FACT] MemForge uses Python and FastAPI."
)


def check_17_remember_endpoint(client: httpx.Client) -> None:
    """``POST /memories/remember`` stores a memory and reports duplicates."""
    response = client.post(
        REMEMBER_URL,
        json={
            "content": "User prefers Python for backend work.",
            "memory_type": "PROFILE",
            "source": "user:smoke",
            "confidence": 0.95,
            "importance": 0.9,
        },
    )
    require(response.status_code == 201, f"remember failed: {response.text}")
    body = response.json()
    require(body["memory"]["memory_type"] == "PROFILE", "the memory type was lost")
    require("memory" in body and "duplicate_count" in body, "the response shape changed")

    duplicate = client.post(
        REMEMBER_URL,
        json={
            "content": "The user prefers python for backend development work",
            "memory_type": "OBSERVATION",
        },
    )
    require(duplicate.status_code == 201, "the near-duplicate was not stored")
    require(
        duplicate.json()["duplicate_count"] >= 1,
        "the near-duplicate pair was not detected through remember()",
    )
    print(
        f"    stored a profile memory and detected "
        f"{duplicate.json()['duplicate_count']} near-duplicate(s)"
    )


def check_18_agent_turn(client: httpx.Client) -> dict[str, Any]:
    """A memory-aware turn answers using compiled context, offline."""
    response = client.post(
        AGENT_URL,
        json={
            "request": AGENT_REQUEST,
            "token_budget": 300,
            "top_k": 10,
            "extract_candidates": True,
            "store_candidates": False,
        },
    )
    require(response.status_code == 200, f"agent turn failed: {response.text}")
    body = response.json()
    require(body["provider"] == "mock", f"the default provider is {body['provider']!r}")
    require(body["memory_used"] is True, "the turn used no memory")
    require(body["memory_gate"]["use"] == "used", "the memory gate did not fire")
    require(body["context_text"].strip(), "no compiled context reached the provider")
    require(
        body["context_estimated_tokens"] <= body["context_token_budget"],
        "the agent context broke its budget",
    )
    require(body["context_blocks"], "no block provenance was reported")
    require(body["answer"].strip(), "the turn produced no answer")
    require(body["usage"] is None, "the mock provider reported usage; it must not")
    print(
        f"    provider={body['provider']} model={body['model']} "
        f"context={body['context_estimated_tokens']}/{body['context_token_budget']} "
        f"blocks={len(body['context_blocks'])}"
    )
    return body


def check_19_candidates_are_not_stored_by_default(
    client: httpx.Client, body: dict[str, Any]
) -> None:
    """Candidates are reported, and nothing is written without permission."""
    require(body["candidate_memories"], "the mock extracted no candidate")
    require(body["stored_memory_ids"] == [], "candidates were stored without permission")
    require(
        all(item["rejection"] is None for item in body["candidate_memories"]),
        "a candidate was rejected without a reason",
    )
    listing = client.get("/api/v1/memories", params={"limit": 100}).json()["total"]
    require(listing == len(MEMORY_SEED) + 1, f"the corpus changed unexpectedly ({listing})")
    print("    1 candidate reported, 0 stored (permission is opt-in)")


def check_20_storage_requires_permission(client: httpx.Client) -> None:
    """With ``store_candidates: true`` the candidate is persisted."""
    response = client.post(
        AGENT_URL,
        json={
            "request": "remember: [FACT] The user ships on Fridays.",
            "token_budget": 200,
            "extract_candidates": True,
            "store_candidates": True,
            "source": "agent:smoke",
        },
    )
    require(response.status_code == 200, f"agent turn failed: {response.text}")
    body = response.json()
    require(len(body["stored_memory_ids"]) == 1, "the permitted candidate was not stored")
    stored_id = body["stored_memory_ids"][0]
    stored = client.get(f"/api/v1/memories/{stored_id}")
    require(stored.status_code == 200, "the stored memory is not readable")
    require(
        stored.json()["source"] == "agent:smoke",
        "the stored memory has the wrong provenance",
    )
    print(f"    1 candidate stored with explicit permission ({stored_id[:8]})")


def check_21_agent_context_only(client: httpx.Client) -> None:
    """``context_only`` compiles the context and never calls the provider."""
    body = client.post(
        AGENT_URL,
        json={"request": AGENT_REQUEST, "token_budget": 200, "context_only": True},
    ).json()
    require(body["answer"] == "", "context_only produced an answer")
    require(body["context_text"].strip(), "context_only compiled no context")
    require(
        body["operations"] == []
        or all(record["operation"] != "respond" for record in body["operations"]),
        "context_only called the provider",
    )
    print(f"    compiled {body['context_estimated_tokens']} tokens without a provider call")


def check_22_agent_memory_gate(client: httpx.Client) -> None:
    """Small talk skips memory; an explicit opt-out disables it."""
    small_talk = client.post(AGENT_URL, json={"request": "hi"}).json()
    require(
        small_talk["memory_gate"]["use"] == "skipped",
        f"small talk was not gated: {small_talk['memory_gate']}",
    )
    require(small_talk["context_text"] == "", "small talk still compiled a context")

    disabled = client.post(AGENT_URL, json={"request": AGENT_REQUEST, "use_memory": False}).json()
    require(
        disabled["memory_gate"]["use"] == "disabled",
        "use_memory=false was ignored",
    )
    require(disabled["context_text"] == "", "use_memory=false still compiled a context")
    print("    small talk skipped; use_memory=false disabled memory")


def check_23_agent_validation(client: httpx.Client) -> None:
    """Invalid agent requests use the documented error envelope."""
    cases: tuple[tuple[dict[str, Any], int, str], ...] = (
        ({}, 422, "VALIDATION_ERROR"),
        ({"request": ""}, 422, "VALIDATION_ERROR"),
        ({"request": "x", "token_budget": 0}, 422, "VALIDATION_ERROR"),
        ({"request": "x", "top_k": 0}, 422, "VALIDATION_ERROR"),
        ({"request": "x", "retrieval_mode": "telepathy"}, 422, "VALIDATION_ERROR"),
        ({"request": "x", "temperature": 9.0}, 422, "VALIDATION_ERROR"),
        ({"request": "x", "unknown_field": True}, 422, "VALIDATION_ERROR"),
        ({"request": "x", "token_budget": 999999}, 400, "AGENT_VALIDATION_ERROR"),
    )
    for body, status, code in cases:
        response = client.post(AGENT_URL, json=body)
        require(
            response.status_code == status,
            f"{body} returned {response.status_code}, expected {status}",
        )
        require(
            response.json()["error"]["code"] == code,
            f"{body} returned {response.json()['error']['code']}, expected {code}",
        )
    print(f"    {len(cases)} malformed agent request(s) rejected with the error envelope")


def check_24_no_secret_leaks(client: httpx.Client, body: dict[str, Any]) -> None:
    """No agent or remember response can carry provider configuration."""
    settings = client.get("/openapi.json").json()
    agent_schema = settings["components"]["schemas"]["AgentRespondResponse"]
    field_names = " ".join(agent_schema["properties"]).lower()
    # ``*_token*`` fields are budget/estimate figures, not credentials; the
    # forbidden set is what would actually leak a secret.
    for forbidden in ("api_key", "authorization", "credential", "secret", "bearer"):
        require(
            forbidden not in field_names,
            f"AgentRespondResponse exposes a {forbidden!r} field",
        )
    rendered = json.dumps(body).lower()
    for forbidden in ("authorization", "bearer ", "api_key", "api-key"):
        require(forbidden not in rendered, f"the agent response leaked {forbidden!r}")
    print("    no credential-shaped field or value in the agent contract or response")


# -- M6: evaluation + observability -------------------------------------------

TRACES_URL = "/api/v1/traces"
METRICS_URL = "/api/v1/metrics"
EVALUATIONS_URL = "/api/v1/evaluations/latest"

#: Substrings that must never appear anywhere in a trace payload.
SECRET_MARKERS: tuple[str, ...] = (
    "authorization",
    "bearer ",
    "api_key",
    "api-key",
    "password",
    "secret",
    "sk-",
)


def check_25_agent_turn_is_traceable(client: httpx.Client) -> str:
    """The mandatory M6 flow, end to end over real HTTP.

    agent request -> memory gate -> retrieval -> context compilation ->
    mock provider -> trace created -> events persisted -> readable back.
    """
    response = client.post(
        AGENT_URL,
        json={
            "request": AGENT_REQUEST,
            "token_budget": 300,
            "extract_candidates": True,
            "store_candidates": False,
        },
    )
    require(response.status_code == 200, f"agent turn failed: {response.text}")
    trace_id = response.headers.get("X-MemForge-Trace-Id")
    require(bool(trace_id), "the agent response carried no trace id")

    detail = client.get(f"{TRACES_URL}/{trace_id}")
    require(detail.status_code == 200, f"the trace is not readable: {detail.text}")
    summary = detail.json()["summary"]
    require(summary["succeeded"] is True, "a successful turn was summarised as failed")
    for operation in (
        "agent.request",
        "agent.memory_gate",
        "memory.retrieval",
        "context.compile",
        "llm.generate",
        "agent.memory_candidate",
        "agent.response",
    ):
        require(
            summary["operations"].get(operation) == "ok",
            f"the trace does not report {operation}: {summary['operations']}",
        )
    require(
        summary["operations"].get("memory.write") == "skipped",
        "a turn without permission reported a memory write",
    )
    print(
        f"    trace={trace_id} events={summary['event_count']} operations={len(summary['operations'])}"
    )
    return trace_id


def check_26_trace_event_sequence(client: httpx.Client, trace_id: str) -> None:
    """The recorded event order is the documented one, and is total."""
    response = client.get(f"{TRACES_URL}/{trace_id}/events")
    require(response.status_code == 200, f"events are not readable: {response.text}")
    events = response.json()
    require(events, "the trace recorded no events")

    sequences = [event["sequence"] for event in events]
    require(sequences == sorted(sequences), f"events are out of order: {sequences}")
    require(
        sequences == list(range(len(sequences))), f"sequence numbers are not total: {sequences}"
    )

    kinds = [event["kind"] for event in events]
    expected_order = [
        "request",
        "retrieval_start",
        "retrieval_complete",
        "context_compile_start",
        "context_compile_complete",
        "llm_request",
        "llm_response",
        "llm_request",
        "llm_response",
        "memory_gate",
        "memory_candidate",
        "memory_write",
        "response",
    ]
    require(kinds == expected_order, f"unexpected lifecycle: {kinds}")

    started = {event["kind"]: event["event_id"] for event in events}
    require(
        events[2]["parent_id"] == started["retrieval_start"],
        "the retrieval completion does not point at its opening event",
    )
    require(
        events[4]["parent_id"] == started["context_compile_start"],
        "the compilation completion does not point at its opening event",
    )
    for event in events:
        if event["kind"].endswith("_start") or event["kind"].endswith("_request"):
            continue
        if event["duration_ms"] is not None:
            require(event["duration_ms"] >= 0.0, f"a negative duration was recorded: {event}")
    print(f"    {len(events)} events in the documented order, nested by parent_id")


def check_27_trace_contains_no_secrets(client: httpx.Client, trace_id: str) -> None:
    """Nothing recorded in a trace can be a credential or a prompt."""
    response = client.get(f"{TRACES_URL}/{trace_id}/events")
    rendered = json.dumps(response.json()).lower()
    for marker in SECRET_MARKERS:
        require(marker not in rendered, f"a trace payload leaked {marker!r}")
    # The request text itself must not be stored, only its size.
    require("backend language should" not in rendered, "the raw request text was recorded")
    for event in response.json():
        for key in ("prompt", "system_prompt", "answer", "content", "messages"):
            require(key not in event["metadata"], f"event metadata carries {key!r}")
        if event["kind"] == "llm_response":
            require(
                event["metadata"]["usage_reported"] is False,
                "the mock provider was recorded as reporting usage",
            )
            for figure in (
                "usage_total_tokens",
                "usage_prompt_tokens",
                "usage_completion_tokens",
            ):
                require(
                    figure not in event["metadata"],
                    f"a token figure was recorded although no usage was reported ({figure})",
                )
    print("    no credential, prompt or answer in the recorded payloads")


def check_28_traces_are_listed_and_persisted(client: httpx.Client, trace_id: str) -> None:
    """Traces are listable, filterable and readable after the fact."""
    listing = client.get(TRACES_URL, params={"limit": 50})
    require(listing.status_code == 200, f"traces could not be listed: {listing.text}")
    body = listing.json()
    require(body["count"] > 0, "no traces were listed")
    require(
        trace_id in {item["trace_id"] for item in body["traces"]}, "the turn's trace is missing"
    )

    agents = client.get(TRACES_URL, params={"kind": "agent_request"}).json()
    require(agents["count"] > 0, "the agent_request filter returned nothing")
    require(
        all(item["kind"] == "agent_request" for item in agents["traces"]),
        "the kind filter leaked other kinds",
    )
    missing = client.get(f"{TRACES_URL}/does-not-exist")
    require(missing.status_code == 404, "an unknown trace did not return 404")
    require(
        missing.json()["error"]["code"] == "TRACE_NOT_FOUND",
        "an unknown trace did not use the error envelope",
    )
    bad_kind = client.get(TRACES_URL, params={"kind": "telepathy"})
    require(bad_kind.status_code == 500, "an unknown trace kind was accepted")
    print(f"    {body['count']} trace(s) listed, filtered and 404 on an unknown id")


def check_29_metrics(client: httpx.Client) -> None:
    """Counters are derived from events, and usage is never invented."""
    response = client.get(METRICS_URL)
    require(response.status_code == 200, f"metrics failed: {response.text}")
    body = response.json()
    require(body["event_count"] > 0, "no events were counted")
    require(
        body["traces"].get("agent_request", 0) > 0,
        "no agent trace was counted",
    )
    require(body["provider"].get("calls", 0) > 0, "no provider call was counted")
    require(
        body["memory"].get("written", 0) > 0,
        "a permitted memory write was never counted",
    )
    require(
        body["context"].get("selected", 0) > 0,
        "no context selection was counted",
    )
    # The mock reports no usage, so this must be exactly 0 -- not an estimate.
    require(
        body["usage_reported"] == 0,
        f"usage was reported without a provider sending it: {body['usage_reported']}",
    )
    # ``failed_traces`` is a derived counter, so cross-check it against the raw
    # events rather than against an expectation: one earlier check deliberately
    # sends an invalid request, so a non-zero count here is correct as long as
    # it agrees with the traces.
    traces = client.get(TRACES_URL, params={"limit": 200}).json()["traces"]
    observed_failures = 0
    for trace in traces:
        events = client.get(f"{TRACES_URL}/{trace['trace_id']}/events").json()
        broken = [
            f"{event['kind']}({event['detail']})" for event in events if event["status"] == "error"
        ]
        if broken:
            observed_failures += 1
            print(f"      note: {trace['operation']} failed as expected at {', '.join(broken)}")
    require(
        body["failed_traces"] == observed_failures,
        f"failed_traces={body['failed_traces']} but {observed_failures} trace(s) hold an error",
    )
    require(
        sum(body["statuses"].values()) == body["event_count"],
        "status counts do not add up to the event count",
    )
    print(
        f"    events={body['event_count']} provider_calls={body['provider']['calls']} "
        f"usage_reported={body['usage_reported']}"
    )


def check_30_evaluation_endpoint(client: httpx.Client) -> None:
    """The evaluation endpoint reports a real, measured run."""
    response = client.get(EVALUATIONS_URL)
    require(response.status_code == 200, f"the evaluation endpoint failed: {response.text}")
    body = response.json()
    for field in (
        "dataset",
        "version",
        "reference",
        "total_cases",
        "passed_cases",
        "failed_cases",
        "case_score",
        "metric_score",
        "overall",
        "threshold",
        "passed",
        "categories",
        "metrics",
        "cases",
    ):
        require(field in body, f"the evaluation response is missing {field!r}")
    require(body["reference"] == "memory-core@v1", f"unexpected dataset {body['reference']!r}")
    require(body["total_cases"] > 0, "the evaluation ran no case")
    require(
        body["passed_cases"] + body["failed_cases"] == body["total_cases"],
        "case counts do not add up",
    )
    require(body["passed"] is True, f"the shipped evaluation does not pass: {body['cases']}")
    require(body["metrics"], "no metric was reported")
    for metric in body["metrics"]:
        require(0.0 <= metric["score"] <= 1.0, f"{metric['name']} scored {metric['score']}")
        require(
            metric["denominator"] > 0,
            f"{metric['name']} was reported without a measured denominator",
        )
        require(metric["definition"], f"{metric['name']} carries no definition")
    categories = dict(body["categories"])
    require(
        set(categories) == {"agent", "conflict", "context", "duplicate", "lifecycle", "retrieval"},
        f"unexpected categories: {sorted(categories)}",
    )

    by_reference = client.get("/api/v1/evaluations/memory-core@v1")
    require(by_reference.status_code == 200, "the dataset reference did not resolve")
    require(
        by_reference.json()["reference"] == body["reference"],
        "the reference lookup returned a different dataset",
    )
    unknown = client.get("/api/v1/evaluations/does-not-exist")
    require(unknown.status_code == 404, "an unknown dataset did not return 404")
    print(
        f"    {body['reference']} cases={body['total_cases']} "
        f"overall={body['overall']:.3f} metrics={len(body['metrics'])}"
    )


CHECKS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("health endpoint answers", check_01_health),
    ("realistic memory set created", check_02_memory_set),
    ("M2 duplicates and conflicts recorded", check_03_relationships),
    ("M3 retrieval still works", check_04_retrieval),
    ("context compilation returns the contract", check_05_compile),
    ("selected memories are listed", check_06_selected),
    ("duplicates are suppressed", check_07_duplicate_suppression),
    ("conflicts are suppressed", check_08_conflict_suppression),
    ("budget accounting is consistent", check_09_budget_and_remaining),
    ("hard budget holds across budgets", check_10_budget_sweep),
    ("every block has provenance", check_11_provenance),
    ("the explanation is honest", check_12_explanation),
    ("compilation is deterministic", check_13_determinism),
    ("malformed requests are rejected", check_14_validation),
    ("hybrid degrades honestly", check_15_hybrid_degradation),
    ("the default path is offline", check_16_default_offline_path),
    ("memories/remember stores and dedupes", check_17_remember_endpoint),
    ("agent turn uses compiled memory", check_18_agent_turn),
    ("candidates are not stored by default", check_19_candidates_are_not_stored_by_default),
    ("storage requires explicit permission", check_20_storage_requires_permission),
    ("context_only skips the provider", check_21_agent_context_only),
    ("the memory gate is respected", check_22_agent_memory_gate),
    ("agent validation is strict", check_23_agent_validation),
    ("no secrets in the agent contract", check_24_no_secret_leaks),
    ("agent turns produce a trace", check_25_agent_turn_is_traceable),
    ("trace events are ordered", check_26_trace_event_sequence),
    ("traces carry no secrets", check_27_trace_contains_no_secrets),
    ("traces are listed and persisted", check_28_traces_are_listed_and_persisted),
    ("metrics are derived from events", check_29_metrics),
    ("the evaluation endpoint reports a run", check_30_evaluation_endpoint),
)


def run() -> int:
    """Run every mandatory check; return a process exit code."""
    corpus: Corpus | None = None
    context_body: dict[str, Any] = {}
    trace_id = ""
    failures: list[tuple[int, str, str]] = []

    with tempfile.TemporaryDirectory(prefix="memforge-smoke-") as directory:
        db_path = Path(directory) / "smoke.sqlite"
        with (
            Server(db_path) as server,
            httpx.Client(base_url=server.base_url, timeout=120.0) as client,
        ):
            print(f"MemForge smoke test against {server.base_url} (db={db_path.name})")
            print("=" * 78)
            for index, (title, check) in enumerate(CHECKS, start=1):
                try:
                    produced = _dispatch(check, client, corpus, context_body, trace_id)
                except CheckFailed as failure:
                    failures.append((index, title, str(failure)))
                    print(f"[{index:2}/{len(CHECKS)}] FAIL  {title}")
                    print(f"          {failure}")
                    continue
                if isinstance(produced, Corpus):
                    corpus = produced
                elif isinstance(produced, str) and not trace_id:
                    trace_id = produced
                elif isinstance(produced, dict):
                    context_body = produced
                print(f"[{index:2}/{len(CHECKS)}] PASS  {title}")

    print("=" * 78)
    passed = len(CHECKS) - len(failures)
    print(f"{passed}/{len(CHECKS)} mandatory checks passed")
    if failures:
        print("\nFailures:")
        for index, title, message in failures:
            print(f"  {index:2}. {title}: {message}")
        return 1
    print("Smoke test finished successfully.")
    return 0


def _dispatch(
    check: Callable[..., Any],
    client: httpx.Client,
    corpus: Corpus | None,
    body: dict[str, Any],
    trace_id: str = "",
) -> Any:
    """Call the check with exactly the state it needs."""
    name = check.__name__
    if name == "check_25_agent_turn_is_traceable":
        return check(client)
    if name in ("check_26_trace_event_sequence", "check_27_trace_contains_no_secrets"):
        require(bool(trace_id), "a trace must be produced before this check")
        return check(client, trace_id)
    if name == "check_28_traces_are_listed_and_persisted":
        require(bool(trace_id), "a trace must be produced before this check")
        return check(client, trace_id)
    if name in ("check_29_metrics", "check_30_evaluation_endpoint"):
        return check(client)
    if name == "check_05_compile":
        return check(client)
    if name in (
        "check_01_health",
        "check_02_memory_set",
        "check_10_budget_sweep",
        "check_17_remember_endpoint",
        "check_20_storage_requires_permission",
        "check_21_agent_context_only",
        "check_22_agent_memory_gate",
        "check_23_agent_validation",
    ):
        return check(client)
    if name == "check_18_agent_turn":
        return check(client)
    if name in ("check_19_candidates_are_not_stored_by_default", "check_24_no_secret_leaks"):
        require(bool(body), "the agent turn must run before this check")
        return check(client, body)
    if name in ("check_03_relationships", "check_04_retrieval"):
        require(corpus is not None, "the corpus must exist before this check")
        return check(client, corpus)
    if name == "check_06_selected":
        return check(client, body)
    if name == "check_07_duplicate_suppression":
        require(corpus is not None, "the corpus must exist before this check")
        return check(client, body, corpus)
    if name == "check_08_conflict_suppression":
        require(corpus is not None, "the corpus must exist before this check")
        return check(client, corpus)
    if name in ("check_11_provenance", "check_12_explanation"):
        return check(client, body)
    return check(client)


if __name__ == "__main__":
    raise SystemExit(run())
