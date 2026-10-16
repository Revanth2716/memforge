"""Application factory and ASGI entrypoint.

Run locally with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.routes import agent, context, evaluation, health, memories, observability, retrieval
from app.config import Settings
from app.evaluation.registry import EvaluationRegistry
from app.observability.instrumentation import (
    AgentTracer,
    build_traced_operations,
    trace_provider,
)
from app.observability.service import ObservabilityService
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.factory import build_llm_provider, describe_provider
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository
from app.storage.sqlite.trace_store import SqliteTraceStore

logger = logging.getLogger("memforge")

API_DESCRIPTION = """
MemForge is a **local-first persistent memory layer for AI agents**.

Milestone 1 ships the deterministic memory core: memories are created, read,
updated and forgotten through a versioned REST API backed by SQLite. Every
change appends an immutable history version, so previous states can always be
inspected.

Milestone 4 adds the **context compiler**: `POST /api/v1/context/compile` turns
retrieved memories into a budgeted, ordered, fully explained agent context.

Milestone 5 adds the **agent integration**: `POST /api/v1/agent/respond` answers
a request using compiled memory context. The language-model provider defaults to
a deterministic offline mock; a custom OpenAI-compatible API is opt-in through
`MEMFORGE_*` environment variables.

Milestone 6 adds **evaluation and observability**: `GET /api/v1/traces`,
`GET /api/v1/traces/{id}/events` and `GET /api/v1/metrics` expose the
correlated, append-only record of what a request actually did, and
`GET /api/v1/evaluations/latest` reports the dataset's measured behaviour.

No network access, API keys or LLM calls are required for any endpoint in the
default configuration. Token figures reported by the compiler are documented
estimates, not provider tokenisations, and usage reported by a provider is only
ever echoed when the provider actually sent it. Traces contain counts, ids, modes
and sizes -- never prompts, answers, credentials or chain-of-thought.
"""


def configure_logging(level: str) -> None:
    """Configure root logging once (idempotent)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the storage schema and the shared service on startup."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    database_path = settings.resolved_database_path()
    repository = SqliteMemoryRepository(database_path)
    repository.initialize()
    app.state.repository = repository
    app.state.memory_service = MemoryService(
        repository, intelligence=settings.intelligence_config()
    )
    app.state.retrieval_service = MemoryRetrievalService(
        repository,
        ranking=settings.ranking_config(),
        intelligence=settings.intelligence_config(),
        embedding_provider=settings.build_embedding_provider(),
        embedding_store=SqliteEmbeddingStore(database_path),
    )
    context_service = ContextCompilerService(
        app.state.retrieval_service,
        policy=settings.context_policy(),
        estimator=settings.token_estimator_config(),
        max_token_budget=settings.max_context_token_budget,
    )
    app.state.context_service = context_service

    # M6: traces live in the same SQLite file as everything else. The
    # observability service is built first so the traced decorators below can
    # be wired in from the start, rather than being added to a live graph.
    observability = ObservabilityService(SqliteTraceStore(database_path))
    if settings.llm_api_key:
        # The key stays in the environment; the service only learns the literal
        # so it can redact it out of anything that reaches a trace.
        observability.register_secret(settings.llm_api_key)
    app.state.observability = observability
    app.state.evaluation_registry = EvaluationRegistry()

    operations = build_traced_operations(
        app.state.memory_service, app.state.retrieval_service, context_service, observability
    )
    app.state.memory_operations = operations
    llm_provider = trace_provider(build_llm_provider(settings), observability)
    app.state.llm_provider = llm_provider
    agent_service = AgentService(
        operations, llm_provider, max_token_budget=settings.max_context_token_budget
    )
    app.state.agent_service = agent_service
    app.state.agent_tracer = AgentTracer(agent_service, observability)

    logger.info(
        "MemForge %s ready (db=%s, api=%s, retrieval=%s, embeddings=%s, context=%s, llm=%s)",
        __version__,
        database_path,
        settings.api_prefix,
        settings.default_retrieval_mode,
        "none" if settings.embedding_provider is None else settings.embedding_provider,
        context_service.describe_policy(),
        # describe_provider never includes a credential.
        describe_provider(llm_provider),
    )
    try:
        yield
    finally:
        logger.info("MemForge stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional settings override (used by tests). When omitted,
            settings are loaded from the environment / ``.env``.
    """
    resolved_settings = settings or Settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved_settings

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(memories.router, prefix=resolved_settings.api_prefix)
    app.include_router(retrieval.router, prefix=resolved_settings.api_prefix)
    app.include_router(context.router, prefix=resolved_settings.api_prefix)
    app.include_router(agent.router, prefix=resolved_settings.api_prefix)
    app.include_router(observability.router, prefix=resolved_settings.api_prefix)
    app.include_router(evaluation.router, prefix=resolved_settings.api_prefix)

    return app


#: ASGI application object (``uvicorn app.main:app``).
app = create_app()
