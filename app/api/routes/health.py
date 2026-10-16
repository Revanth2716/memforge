"""Liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.dependencies import MemoryServiceDependency
from app.api.schemas import HealthResponse
from app.domain.errors import StorageUnavailableError

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and storage health check")
def health(service: MemoryServiceDependency) -> HealthResponse:
    """Return service metadata after verifying that storage answers."""
    if not service.health_check():
        raise StorageUnavailableError("SQLite storage did not answer a health query")
    return HealthResponse(
        status="ok",
        service="memforge",
        version=__version__,
        storage="sqlite",
    )
