"""FastAPI dependency wiring.

The service is created once per application (in the lifespan hook) and shared:
the SQLite repository opens a short-lived connection per operation, so a single
instance is safe for concurrent requests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.domain.llm import LLMProvider
from app.evaluation.registry import EvaluationRegistry
from app.observability.instrumentation import AgentTracer, trace_provider
from app.observability.service import ObservabilityService
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.factory import build_llm_provider
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository
from app.storage.sqlite.trace_store import SqliteTraceStore


def get_settings(request: Request) -> Settings:
    """Return the settings attached to the running application."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:  # pragma: no cover - create_app always sets this
        raise RuntimeError("Application settings are not configured")
    return settings


def get_memory_service(request: Request) -> MemoryService:
    """Return the shared memory service, creating it on first use."""
    service: MemoryService | None = getattr(request.app.state, "memory_service", None)
    if service is None:
        # The lifespan hook normally does this; the fallback keeps the app
        # usable with frameworks/tests that construct it without a lifespan.
        repository = SqliteMemoryRepository(get_settings(request).resolved_database_path())
        repository.initialize()
        service = MemoryService(
            repository, intelligence=get_settings(request).intelligence_config()
        )
        request.app.state.memory_service = service
    return service


SettingsDependency = Annotated[Settings, Depends(get_settings)]
MemoryServiceDependency = Annotated[MemoryService, Depends(get_memory_service)]


def get_retrieval_service(request: Request) -> MemoryRetrievalService:
    """Return the shared retrieval service, creating it on first use."""
    service: MemoryRetrievalService | None = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        # The lifespan hook normally does this; the fallback keeps the app
        # usable with frameworks/tests that construct it without a lifespan.
        settings = get_settings(request)
        repository = SqliteMemoryRepository(settings.resolved_database_path())
        repository.initialize()
        service = MemoryRetrievalService(
            repository,
            ranking=settings.ranking_config(),
            intelligence=settings.intelligence_config(),
            embedding_provider=settings.build_embedding_provider(),
            embedding_store=SqliteEmbeddingStore(settings.resolved_database_path()),
        )
        request.app.state.retrieval_service = service
    return service


MemoryRetrievalServiceDependency = Annotated[MemoryRetrievalService, Depends(get_retrieval_service)]


def get_context_service(request: Request) -> ContextCompilerService:
    """Return the shared context compiler, creating it on first use."""
    service: ContextCompilerService | None = getattr(request.app.state, "context_service", None)
    if service is None:
        settings = get_settings(request)
        service = ContextCompilerService(
            get_retrieval_service(request),
            policy=settings.context_policy(),
            estimator=settings.token_estimator_config(),
            max_token_budget=settings.max_context_token_budget,
        )
        request.app.state.context_service = service
    return service


ContextCompilerServiceDependency = Annotated[ContextCompilerService, Depends(get_context_service)]


def get_llm_provider(request: Request) -> LLMProvider:
    """Return the shared LLM provider, wrapped for tracing.

    The wrapper records the call, the real latency and the usage the provider
    actually reported. It is transparent: every property a caller reads is the
    wrapped provider's own.
    """
    provider: LLMProvider | None = getattr(request.app.state, "llm_provider", None)
    if provider is None:
        provider = trace_provider(
            build_llm_provider(get_settings(request)), get_observability(request)
        )
        request.app.state.llm_provider = provider
    return provider


def get_memory_operations(request: Request) -> MemoryOperationsService:
    """Return the shared memory-operations facade over M1-M4."""
    service: MemoryOperationsService | None = getattr(request.app.state, "memory_operations", None)
    if service is None:
        service = MemoryOperationsService(
            get_memory_service(request),
            get_retrieval_service(request),
            get_context_service(request),
        )
        request.app.state.memory_operations = service
    return service


MemoryOperationsServiceDependency = Annotated[
    MemoryOperationsService, Depends(get_memory_operations)
]


def get_observability(request: Request) -> ObservabilityService:
    """Return the shared observability service, creating it on first use.

    The service is created with the real timer and the system clock; only the
    demo and the tests inject deterministic ones. Its store is the same SQLite
    file as everything else, so traces survive a restart.
    """
    service: ObservabilityService | None = getattr(request.app.state, "observability", None)
    if service is None:
        service = ObservabilityService(
            SqliteTraceStore(get_settings(request).resolved_database_path())
        )
        _register_secret(service, get_settings(request))
        request.app.state.observability = service
    return service


def _register_secret(service: ObservabilityService, settings: Settings) -> None:
    """Teach the observability service which literal value must never be stored.

    The key itself never leaves the environment; the service only learns the
    literal so it can redact it out of any string that reaches a trace.
    """
    if settings.llm_api_key:
        service.register_secret(settings.llm_api_key)


ObservabilityServiceDependency = Annotated[ObservabilityService, Depends(get_observability)]


def get_evaluation_registry(request: Request) -> EvaluationRegistry:
    """Return the shared evaluation registry, creating it on first use."""
    registry: EvaluationRegistry | None = getattr(request.app.state, "evaluation_registry", None)
    if registry is None:
        registry = EvaluationRegistry()
        request.app.state.evaluation_registry = registry
    return registry


EvaluationRegistryDependency = Annotated[EvaluationRegistry, Depends(get_evaluation_registry)]


def get_agent_service(request: Request) -> AgentService:
    """Return the shared agent service.

    The provider is wrapped so every call is recorded. The memory layer is *not*
    wrapped here: the agent receives the traced facade the lifespan hook built,
    so a turn's retrieval, compilation and writes all land in the same trace.
    """
    service: AgentService | None = getattr(request.app.state, "agent_service", None)
    if service is None:
        service = AgentService(
            get_memory_operations(request),
            get_llm_provider(request),
            max_token_budget=get_settings(request).max_context_token_budget,
        )
        request.app.state.agent_service = service
    return service


def get_agent_tracer(request: Request) -> AgentTracer:
    """Return a tracer that runs one agent turn inside a trace."""
    tracer: AgentTracer | None = getattr(request.app.state, "agent_tracer", None)
    if tracer is None:
        tracer = AgentTracer(get_agent_service(request), get_observability(request))
        request.app.state.agent_tracer = tracer
    return tracer


AgentServiceDependency = Annotated[AgentService, Depends(get_agent_service)]
AgentTracerDependency = Annotated[AgentTracer, Depends(get_agent_tracer)]
