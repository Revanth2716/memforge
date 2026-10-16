"""REST API package."""

from app.api.dependencies import (
    MemoryServiceDependency,
    SettingsDependency,
    get_memory_service,
    get_settings,
)
from app.api.errors import error_envelope, register_exception_handlers
from app.api.routes import health, memories

__all__ = [
    "MemoryServiceDependency",
    "SettingsDependency",
    "error_envelope",
    "get_memory_service",
    "get_settings",
    "health",
    "memories",
    "register_exception_handlers",
]
