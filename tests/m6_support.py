"""Builders shared by the M6 tests.

Fixtures live in ``tests/conftest.py`` (pytest only discovers fixtures from
``conftest.py``); the helpers that build a real, traced MemForge stack live here
so both the fixtures and the tests can reach them.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.ids import SequentialIdGenerator
from app.domain.llm import LLMProvider
from app.domain.time import ManualClock
from app.observability.instrumentation import (
    AgentTracer,
    build_traced_operations,
    trace_provider,
)
from app.observability.service import ObservabilityService
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.mock_provider import MockLLMProvider
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository

#: Seconds each reading of the deterministic timer advances by.
TIMER_STEP = 0.002


class StepTimer:
    """A ``perf_counter``-shaped clock whose readings advance by a fixed step.

    Injected so durations are reproducible. Tests still treat the values as
    *measured*: they assert non-negativity and ordering, never a magic number.
    """

    def __init__(self, step: float = TIMER_STEP) -> None:
        self._step = step
        self._elapsed = 0.0
        self.readings = 0

    def __call__(self) -> float:
        """Return the current reading in seconds and advance by one step."""
        current = self._elapsed
        self._elapsed += self._step
        self.readings += 1
        return current


def step_timer(step: float = TIMER_STEP) -> StepTimer:
    """Build a fresh deterministic timer."""
    return StepTimer(step)


class Stack:
    """The instrumented object graph a test drives.

    ``operations`` is the traced facade the application uses; ``plain`` is the
    same M5 class built from undecorated services, which is what a test needs to
    assert that tracing does not change behaviour.
    """

    def __init__(
        self,
        *,
        clock: ManualClock,
        memories: MemoryService,
        retrieval: MemoryRetrievalService,
        compiler: ContextCompilerService,
        operations: MemoryOperationsService,
        provider: LLMProvider,
        agent: AgentService,
        tracer: AgentTracer,
        plain: MemoryOperationsService,
        observability: ObservabilityService,
    ) -> None:
        self.clock = clock
        self.memories = memories
        self.retrieval = retrieval
        self.compiler = compiler
        self.operations = operations
        self.provider = provider
        self.agent = agent
        self.tracer = tracer
        self.plain = plain
        self.observability = observability


def build_m6_stack(db_path: Path, clock: ManualClock, observability: ObservabilityService) -> Stack:
    """Assemble a deterministic, fully traced MemForge stack."""
    repository = SqliteMemoryRepository(db_path)
    memories = MemoryService(
        repository, clock=clock, id_generator=SequentialIdGenerator(prefix="m6")
    )
    memories.initialize()
    retrieval = MemoryRetrievalService(
        repository, clock=clock, embedding_store=SqliteEmbeddingStore(db_path)
    )
    compiler = ContextCompilerService(retrieval)
    operations = build_traced_operations(memories, retrieval, compiler, observability)
    provider = trace_provider(MockLLMProvider(), observability)
    agent = AgentService(operations, provider)
    return Stack(
        clock=clock,
        memories=memories,
        retrieval=retrieval,
        compiler=compiler,
        operations=operations,
        provider=provider,
        agent=agent,
        tracer=AgentTracer(agent, observability),
        plain=MemoryOperationsService(memories, retrieval, compiler),
        observability=observability,
    )


__all__ = ["TIMER_STEP", "Stack", "StepTimer", "build_m6_stack", "step_timer"]
