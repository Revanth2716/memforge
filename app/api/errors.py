"""Consistent API error handling.

Every error response uses the same envelope::

    {"error": {"code": "MEMORY_NOT_FOUND", "message": "...", "details": {...}}}

The mapping from domain error to HTTP status lives here, and only here, so the
domain layer never needs to know about HTTP.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import (
    AgentValidationError,
    ContextValidationError,
    EvaluationError,
    InvalidContextConfigError,
    InvalidIntelligenceConfigError,
    InvalidMemoryStateError,
    InvalidRelationshipError,
    InvalidRetrievalConfigError,
    LLMProviderError,
    MemforgeError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryValidationError,
    NoConflictError,
    ObservabilityError,
    RepositoryError,
    RetrievalValidationError,
    StorageUnavailableError,
)

logger = logging.getLogger("memforge.api")

#: A trace that was never recorded.
ObservabilityNotFoundError = ObservabilityError

#: An evaluation dataset that could not be loaded.
EvaluationNotFoundError = EvaluationError


def error_envelope(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the standard error payload."""
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return {"error": body}


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_envelope(code, message, details))


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the MemForge error envelope to the application."""

    @app.exception_handler(MemoryNotFoundError)
    async def _memory_not_found(request: Request, exc: MemoryNotFoundError) -> JSONResponse:
        return _error_response(404, "MEMORY_NOT_FOUND", str(exc))

    @app.exception_handler(MemoryValidationError)
    async def _memory_validation(request: Request, exc: MemoryValidationError) -> JSONResponse:
        return _error_response(400, "MEMORY_VALIDATION_ERROR", str(exc))

    @app.exception_handler(MemoryAlreadyExistsError)
    async def _memory_exists(request: Request, exc: MemoryAlreadyExistsError) -> JSONResponse:
        return _error_response(409, "MEMORY_ALREADY_EXISTS", str(exc))

    @app.exception_handler(InvalidMemoryStateError)
    async def _invalid_state(request: Request, exc: InvalidMemoryStateError) -> JSONResponse:
        return _error_response(409, "INVALID_MEMORY_STATE", str(exc))

    @app.exception_handler(InvalidRelationshipError)
    async def _invalid_relationship(
        request: Request, exc: InvalidRelationshipError
    ) -> JSONResponse:
        return _error_response(400, "INVALID_RELATIONSHIP", str(exc))

    @app.exception_handler(NoConflictError)
    async def _no_conflict(request: Request, exc: NoConflictError) -> JSONResponse:
        return _error_response(409, "NO_CONFLICT", str(exc))

    @app.exception_handler(RetrievalValidationError)
    async def _retrieval_validation(
        request: Request, exc: RetrievalValidationError
    ) -> JSONResponse:
        return _error_response(400, "RETRIEVAL_VALIDATION_ERROR", str(exc))

    @app.exception_handler(InvalidRetrievalConfigError)
    async def _invalid_retrieval_config(
        request: Request, exc: InvalidRetrievalConfigError
    ) -> JSONResponse:
        logger.error("Invalid retrieval configuration: %s", exc)
        return _error_response(500, "INVALID_CONFIGURATION", "Retrieval configuration is invalid")

    @app.exception_handler(ContextValidationError)
    async def _context_validation(request: Request, exc: ContextValidationError) -> JSONResponse:
        return _error_response(400, "CONTEXT_VALIDATION_ERROR", str(exc))

    @app.exception_handler(InvalidContextConfigError)
    async def _invalid_context_config(
        request: Request, exc: InvalidContextConfigError
    ) -> JSONResponse:
        logger.error("Invalid context configuration: %s", exc)
        return _error_response(500, "INVALID_CONFIGURATION", "Context configuration is invalid")

    @app.exception_handler(StorageUnavailableError)
    async def _storage_unavailable(request: Request, exc: StorageUnavailableError) -> JSONResponse:
        return _error_response(503, "STORAGE_UNAVAILABLE", str(exc))

    @app.exception_handler(InvalidIntelligenceConfigError)
    async def _invalid_config(
        request: Request, exc: InvalidIntelligenceConfigError
    ) -> JSONResponse:
        logger.error("Invalid intelligence configuration: %s", exc)
        return _error_response(
            500, "INVALID_CONFIGURATION", "Intelligence configuration is invalid"
        )

    @app.exception_handler(AgentValidationError)
    async def _agent_validation(request: Request, exc: AgentValidationError) -> JSONResponse:
        return _error_response(400, "AGENT_VALIDATION_ERROR", str(exc))

    @app.exception_handler(LLMProviderError)
    async def _llm_provider_error(request: Request, exc: LLMProviderError) -> JSONResponse:
        # The provider layer already normalised the failure and removed any
        # credential from the message, so this only maps kind -> status.
        logger.warning(
            "LLM provider %s failed (%s) on %s %s",
            exc.provider or "unknown",
            exc.kind,
            request.method,
            request.url.path,
        )
        status_code = {
            "not_configured": 503,
            "timeout": 504,
            "authentication": 502,
            "rate_limit": 429,
            "transport": 502,
            "protocol": 502,
        }.get(exc.kind, 502)
        details = {"kind": exc.kind, "provider": exc.provider, "retryable": exc.retryable}
        return _error_response(status_code, f"LLM_{exc.kind.upper()}", str(exc), details)

    @app.exception_handler(ObservabilityError)
    async def _observability_error(request: Request, exc: ObservabilityError) -> JSONResponse:
        # An unknown trace id is a client mistake; a corrupt row is ours. The
        # message distinguishes them without exposing any recorded content.
        unknown = "no trace with id" in str(exc) or "no evaluation" in str(exc).lower()
        logger.info("Observability error on %s %s: %s", request.method, request.url.path, exc)
        return _error_response(
            404 if unknown else 500,
            "TRACE_NOT_FOUND" if unknown else "OBSERVABILITY_ERROR",
            str(exc),
        )

    @app.exception_handler(EvaluationError)
    async def _evaluation_error(request: Request, exc: EvaluationError) -> JSONResponse:
        # A dataset that will not parse is a configuration problem, not a
        # request problem: the client named something that is not there.
        logger.warning("Evaluation error on %s %s: %s", request.method, request.url.path, exc)
        return _error_response(404, "EVALUATION_NOT_FOUND", str(exc))

    @app.exception_handler(RepositoryError)
    async def _repository_error(request: Request, exc: RepositoryError) -> JSONResponse:
        logger.error("Repository error on %s %s: %s", request.method, request.url.path, exc)
        return _error_response(500, "REPOSITORY_ERROR", "Storage operation failed")

    @app.exception_handler(MemforgeError)
    async def _memforge_error(request: Request, exc: MemforgeError) -> JSONResponse:
        logger.error("Unmapped MemForge error on %s %s: %s", request.method, request.url.path, exc)
        return _error_response(500, "INTERNAL_ERROR", "Unexpected error")

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = {
            "errors": [
                {
                    "location": [str(part) for part in error.get("loc", ())],
                    "message": error.get("msg", ""),
                    "type": error.get("type", ""),
                }
                for error in exc.errors()
            ]
        }
        return _error_response(422, "VALIDATION_ERROR", "Request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return _error_response(exc.status_code, "HTTP_ERROR", message)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error_response(500, "INTERNAL_ERROR", "Unexpected server error")
