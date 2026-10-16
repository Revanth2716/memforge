"""Read-only evaluation endpoints (M6).

Two routes:

* ``GET /api/v1/evaluations/latest``  -- the most recent run of the shipped
  dataset, running it on first call and caching the result;
* ``GET /api/v1/evaluations/{ref}``   -- one run by its ``name@version``
  reference (``memory-core@v1``), with the same caching.

There is no "run the evaluation" endpoint. Evaluation is a regression gate, not
something a browser should be able to trigger on demand, and keeping it ``GET``
only means nothing here can change MemForge's state.

What the response means
-----------------------
The numbers are counts over real executions of the version-controlled dataset,
under the default configuration. They prove the behaviour encoded in those cases
and nothing beyond it. The response carries each metric's own definition so a
consumer never has to guess what a score means.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import EvaluationRegistryDependency
from app.api.errors import EvaluationNotFoundError
from app.api.schemas import EvaluationResponse
from app.domain.errors import EvaluationError
from app.evaluation.datasets import DEFAULT_DATASET

router = APIRouter(prefix="/evaluations", tags=["evaluation"])


@router.get(
    "/latest",
    response_model=EvaluationResponse,
    summary="The most recent evaluation of the shipped dataset",
)
def latest_evaluation(
    registry: EvaluationRegistryDependency,
    threshold: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum case pass rate required to pass (defaults to 1.0).",
    ),
) -> EvaluationResponse:
    """Return the latest summary of the default dataset.

    The dataset is executed on the first call and the summary is cached, so this
    is cheap to poll. Re-running is the CLI's job.
    """
    summary = registry.run(DEFAULT_DATASET, threshold=1.0 if threshold is None else threshold)
    return EvaluationResponse.from_domain(summary)


@router.get(
    "/{reference}",
    response_model=EvaluationResponse,
    summary="One evaluation run by dataset reference",
)
def get_evaluation(
    reference: str,
    registry: EvaluationRegistryDependency,
    threshold: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum case pass rate required to pass (defaults to 1.0).",
    ),
) -> EvaluationResponse:
    """Return one run, addressed as ``name`` or ``name@version``.

    ``memory-core`` and ``memory-core@v1`` both resolve to the shipped
    dataset; the version lives inside the file, not in its name.
    """
    resolved = 1.0 if threshold is None else threshold
    cached = registry.get(reference)
    if cached is not None:
        return EvaluationResponse.from_domain(cached)
    try:
        summary = registry.run(reference, threshold=resolved)
    except EvaluationError as error:
        raise EvaluationNotFoundError(str(error), dataset=reference) from error
    return EvaluationResponse.from_domain(summary)


__all__ = ["router"]
