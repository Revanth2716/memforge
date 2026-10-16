"""``python -m app.observability.demo`` -- the observability demo (M6).

::

    TRACE trace-0001
    ---------
    agent.request                 PASS
    memory.gate                   PASS
    retrieval                     PASS
    context.compile               PASS
    llm.generate                  PASS
    memory.candidate              PASS
    memory.write                  SKIPPED
    response                      PASS

Every row is derived from a real agent turn over a real SQLite file. The rows are
per **operation** (the trace's own event list is longer than this), so a
compensation call that ran twice appears once, with the worst status of the two.

Determinism
-----------
Identifiers come from a sequential factory and timestamps from a manual clock, so
two runs print byte-identical output. Durations are measured with an injected
deterministic timer: they are *real measurements of a supplied clock*, not
hard-coded numbers, and they are labelled as such. In the server the same service
uses :func:`time.perf_counter`, and nothing asserts on the values either way.

What is never printed
---------------------
No prompt, no answer, no credential, no chain-of-thought. The trace holds counts,
ids, modes, sizes and provider-reported usage only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

from app.config import PROJECT_ROOT, Settings
from app.domain.agent import AgentRequest
from app.domain.errors import MemforgeError
from app.domain.ids import SequentialIdGenerator
from app.domain.models import MemoryType
from app.domain.observability import EventStatus
from app.domain.time import ManualClock
from app.observability.instrumentation import (
    AgentTracer,
    build_traced_operations,
    trace_provider,
)
from app.observability.service import ObservabilityService, SequentialIdFactory
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.factory import build_llm_provider
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository
from app.storage.sqlite.trace_store import SqliteTraceStore

#: Throwaway database used when the caller names no path.
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "observability-demo.sqlite"

WIDTH = 78

#: Operations shown in the trace tree, in the order a turn emits them.
_OPERATION_ORDER: tuple[str, ...] = (
    "agent.request",
    "agent.memory_gate",
    "memory.retrieval",
    "context.compile",
    "llm.generate",
    "agent.memory_candidate",
    "memory.write",
    "agent.response",
)

#: Human labels for those operations.
_OPERATION_LABELS: dict[str, str] = {
    "agent.request": "agent.request",
    "agent.memory_gate": "memory.gate",
    "memory.retrieval": "retrieval",
    "context.compile": "context.compile",
    "llm.generate": "llm.generate",
    "agent.memory_candidate": "memory.candidate",
    "memory.write": "memory.write",
    "agent.response": "response",
}

#: Status text per terminal status.
_STATUS_TEXT: dict[str, str] = {
    EventStatus.OK.value: "PASS",
    EventStatus.SKIPPED.value: "SKIPPED",
    EventStatus.ERROR.value: "FAIL",
}

#: A write that happened is reported as allowed, not merely "ok".
_WRITE_STATUS_TEXT: dict[str, str] = {
    EventStatus.OK.value: "ALLOWED",
    EventStatus.SKIPPED.value: "SKIPPED",
    EventStatus.ERROR.value: "FAIL",
}


class DeterministicTimer:
    """A monotonic clock, shaped like :func:`time.perf_counter`, with known steps.

    Used so the demo's durations are reproducible. It is still a *measurement*:
    the service reads it exactly the way it reads ``perf_counter`` -- in seconds,
    monotonic, advancing -- it is simply a clock whose readings are known in
    advance. Real deployments never inject it.
    """

    def __init__(self, step_seconds: float = 0.0015) -> None:
        self._step = step_seconds
        self._elapsed = 0.0

    def __call__(self) -> float:
        """Return the current reading in seconds and advance by one step."""
        current = self._elapsed
        self._elapsed += self._step
        return current


class DemoStack(NamedTuple):
    """Everything the demo needs to drive a traced turn."""

    tracer: AgentTracer
    observability: ObservabilityService
    memories: MemoryService
    clock: ManualClock


def build_demo_stack(db_path: Path, settings: Settings | None = None) -> DemoStack:
    """Build a deterministic, traced MemForge stack over a fresh database."""
    if db_path.exists():
        db_path.unlink()
    resolved = settings or Settings(database_path=db_path, log_level="WARNING")
    clock = ManualClock()
    repository = SqliteMemoryRepository(db_path)
    memories = MemoryService(
        repository,
        clock=clock,
        id_generator=SequentialIdGenerator(prefix="obs"),
        intelligence=resolved.intelligence_config(),
    )
    memories.initialize()
    retrieval = MemoryRetrievalService(
        repository,
        clock=clock,
        ranking=resolved.ranking_config(),
        intelligence=resolved.intelligence_config(),
        embedding_store=SqliteEmbeddingStore(db_path),
    )
    compiler = ContextCompilerService(
        retrieval,
        policy=resolved.context_policy(),
        estimator=resolved.token_estimator_config(),
        max_token_budget=resolved.max_context_token_budget,
    )
    observability = ObservabilityService(
        SqliteTraceStore(db_path),
        clock=clock,
        timer=DeterministicTimer(),
        id_factory=SequentialIdFactory(),
    )
    operations = build_traced_operations(memories, retrieval, compiler, observability)
    provider = trace_provider(build_llm_provider(resolved), observability)
    agent = AgentService(operations, provider, max_token_budget=resolved.max_context_token_budget)
    return DemoStack(
        tracer=AgentTracer(agent, observability),
        observability=observability,
        memories=memories,
        clock=clock,
    )


def _rule(char: str = "=") -> None:
    """Print a full-width rule."""
    print(char * WIDTH)


def _seed(memories: MemoryService, clock: ManualClock) -> None:
    """Store a small corpus so recall and compilation have something to work on."""
    corpus = (
        ("The user writes Python and SQL for backend services.", MemoryType.FACT, 0.9, 0.8),
        ("The user prefers the dark theme in every code editor.", MemoryType.PREFERENCE, 0.8, 0.7),
        ("The user drinks oat milk with breakfast.", MemoryType.OBSERVATION, 0.4, 0.2),
    )
    for content, memory_type, confidence, importance in corpus:
        clock.advance(minutes=1)
        memories.create_memory(
            content=content,
            memory_type=memory_type,
            source="user:demo",
            confidence=confidence,
            importance=importance,
        )


def _print_trace(observability: ObservabilityService, trace_id: str) -> None:
    """Print one trace as a per-operation status table."""
    summary = observability.summarize_trace(trace_id)
    if summary is None:  # pragma: no cover - the demo always records first
        return
    print(f"TRACE {summary.trace_id}  ({summary.kind.value} / {summary.operation})")
    print("-" * (WIDTH // 2))
    for operation in _OPERATION_ORDER:
        status = summary.operations.get(operation)
        if status is None:
            continue
        label = _OPERATION_LABELS.get(operation, operation)
        vocabulary = _WRITE_STATUS_TEXT if operation == "memory.write" else _STATUS_TEXT
        print(f"{label:<32}{vocabulary.get(status, status):<10}{status}")
    print("-" * (WIDTH // 2))
    print(f"events recorded : {summary.event_count}")
    print(f"memories touched: {', '.join(summary.memory_ids) or 'none'}")
    measured = ", ".join(
        f"{name}={value:.3f}ms" for name, value in sorted(summary.durations_ms.items())
    )
    print(f"measured        : {measured or 'none'}")
    print()


def _print_metrics(observability: ObservabilityService) -> None:
    """Print the derived counters."""
    metrics = observability.get_metrics()
    print("METRICS")
    print("-" * (WIDTH // 2))
    for title, counters in (
        ("traces", metrics.traces),
        ("operations", metrics.operations),
        ("statuses", metrics.statuses),
        ("memory", metrics.memory),
        ("context", metrics.context),
        ("provider", metrics.provider),
    ):
        rendered = ", ".join(f"{name}={value}" for name, value in counters.items())
        print(f"{title:<12} {rendered or 'none'}")
    print(
        f"{'usage':<12} reported_by_provider={metrics.usage_reported} "
        f"(0 means no provider supplied usage; nothing is inferred)"
    )
    print()


def run_demo(db_path: Path = DEFAULT_DB_PATH, *, store_candidates: bool = False) -> int:
    """Run the observability walkthrough. Returns a process exit code."""
    tracer, observability, memories, clock = build_demo_stack(db_path)
    print("=" * WIDTH)
    print("MemForge observability (M6) -- traces, events and metrics".center(WIDTH))
    print(f"database: {db_path}".center(WIDTH))
    print("=" * WIDTH)
    print()

    _seed(memories, clock)

    print("TURN 1 -- candidate proposed, storage not permitted")
    print("-" * (WIDTH // 2))
    first = _run_turn(
        tracer,
        "Summarise my backend experience.\nremember: [FACT] The user ships Python services.",
        store_candidates=store_candidates,
    )
    _print_trace(observability, first)

    print("TURN 2 -- candidate proposed, storage explicitly permitted")
    print("-" * (WIDTH // 2))
    second = _run_turn(
        tracer,
        "Note my setup.\nremember: [PREFERENCE] The user prefers a mechanical keyboard.",
        store_candidates=True,
    )
    _print_trace(observability, second)

    _print_metrics(observability)

    print("SAFETY")
    print("-" * (WIDTH // 2))
    print("trace payloads exclude prompts, answers, credentials and reasoning;")
    print("only counts, ids, modes, sizes and provider-reported usage are stored.")
    return 0


def _run_turn(tracer: AgentTracer, request: str, *, store_candidates: bool) -> str:
    """Run one traced agent turn, printing a line about what it did."""
    traced = tracer.respond(
        AgentRequest(
            request=request,
            extract_candidates=True,
            store_candidates=store_candidates,
        )
    )
    result = traced.result
    print(
        f"gate={result.gate.use.value} "
        f"blocks={len(result.context_blocks)} "
        f"candidates={len(result.candidate_memories)} "
        f"stored={len(result.stored_memory_ids)} "
        f"usage_reported={result.usage is not None}"
    )
    return traced.trace.trace_id


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m app.observability.demo",
        description="Print a real MemForge trace, event list and metric summary.",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="throwaway SQLite path (recreated from scratch)",
    )
    parser.add_argument(
        "--store-candidates",
        action="store_true",
        help="let the first turn persist candidates too",
    )
    args = parser.parse_args(argv)
    try:
        return run_demo(Path(args.db), store_candidates=args.store_candidates)
    except MemforgeError as error:  # pragma: no cover - defensive
        print(f"observability demo failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
