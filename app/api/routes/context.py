"""Context compilation endpoint (``POST /api/v1/context/compile``).

The route only translates HTTP to domain objects. Budget enforcement, selection,
ordering, compression, provenance and explanation all live in
:class:`~app.services.context_service.ContextCompilerService` and
:mod:`app.domain.context`, so the same behaviour is reachable from the CLI and
from a test without a web server.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ContextCompilerServiceDependency, SettingsDependency
from app.api.schemas import ContextCompileRequest, ContextCompileResponse

router = APIRouter(prefix="/context", tags=["context"])


@router.post(
    "/compile",
    response_model=ContextCompileResponse,
    summary="Compile retrieved memories into a budgeted agent context",
)
def compile_context(
    payload: ContextCompileRequest,
    service: ContextCompilerServiceDependency,
    settings: SettingsDependency,
) -> ContextCompileResponse:
    """Retrieve, then compile the best context that fits the token budget.

    The response carries the rendered ``context_text`` the agent would receive,
    the structured ``blocks`` with provenance back to each memory, the full
    ``selection_trace`` (one entry per candidate, with the reason it was admitted
    or rejected), the suppressed candidates and a machine-readable
    ``explanation``.

    Token figures are **estimates** produced by MemForge's documented
    character/word estimator, never provider tokenisations. The hard guarantee is
    ``estimated_tokens <= token_budget``, reported as ``within_budget``.
    """
    request = payload.to_domain_request(default_token_budget=settings.default_context_token_budget)
    return ContextCompileResponse.from_domain(service.compile(request))
