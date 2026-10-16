"""Shared pytest fixtures.

Every test runs against a throwaway SQLite file inside pytest's ``tmp_path``;
nothing here touches the repository's ``data/`` directory or the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.ids import SequentialIdGenerator
from app.domain.time import ManualClock
from app.evaluation.service import EvaluationService
from app.main import create_app
from app.observability.service import ObservabilityService, SequentialIdFactory
from app.services.memory_service import MemoryService
from app.storage.sqlite.repository import SqliteMemoryRepository
from app.storage.sqlite.trace_store import SqliteTraceStore
from tests.m6_support import Stack, StepTimer, build_m6_stack, step_timer
from tests.support import FIXED_START


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Nested path, so directory creation is exercised by every test."""
    return tmp_path / "state" / "memforge-test.sqlite"


@pytest.fixture
def repository(db_path: Path) -> SqliteMemoryRepository:
    repo = SqliteMemoryRepository(db_path)
    repo.initialize()
    return repo


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(FIXED_START)


#: Fixture type aliases so tests can annotate the injected ports.
ClockFixture = ManualClock
IdGeneratorFixture = SequentialIdGenerator


@pytest.fixture
def id_generator() -> SequentialIdGenerator:
    return SequentialIdGenerator(prefix="test")


@pytest.fixture
def service(
    repository: SqliteMemoryRepository, clock: ManualClock, id_generator: SequentialIdGenerator
) -> MemoryService:
    """A deterministic service: fixed clock, reproducible ids."""
    return MemoryService(repository, clock=clock, id_generator=id_generator)


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(database_path=db_path, log_level="INFO")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """TestClient that runs the application lifespan (startup/shutdown hooks)."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_raise(settings: Settings) -> Iterator[TestClient]:
    """TestClient that lets error responses (e.g. 500) be inspected."""
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# -- M6: observability fixtures ------------------------------------------------
# These live here rather than in ``m6_support`` because pytest only discovers
# fixtures from ``conftest.py``; the builders they call live in ``m6_support``.


@pytest.fixture
def timer() -> StepTimer:
    """A deterministic monotonic timer (seconds, like ``perf_counter``)."""
    return step_timer()


@pytest.fixture
def observability(db_path: Path, clock: ManualClock, timer: StepTimer) -> ObservabilityService:
    """An observability service persisting to the throwaway database."""
    return ObservabilityService(
        SqliteTraceStore(db_path),
        clock=clock,
        timer=timer,
        id_factory=SequentialIdFactory(prefix="trace"),
    )


@pytest.fixture
def in_memory_observability(clock: ManualClock, timer: StepTimer) -> ObservabilityService:
    """An observability service that keeps nothing on disk."""
    return ObservabilityService(clock=clock, timer=timer, id_factory=SequentialIdFactory())


@pytest.fixture
def stack(db_path: Path, clock: ManualClock, observability: ObservabilityService) -> Stack:
    """A real, traced MemForge stack over the throwaway database."""
    return build_m6_stack(db_path, clock, observability)


@pytest.fixture(scope="module")
def evaluation_service() -> EvaluationService:
    """One evaluation service; every case is isolated anyway."""
    return EvaluationService()
