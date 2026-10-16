"""An in-process cache of the most recent evaluation runs (M6).

Evaluation is expensive enough that running it on every request would be
absurd, and it is deterministic enough that its result only changes when the
code or the dataset changes. So one registry keeps the last summary per dataset
reference (``name@version``) and the HTTP layer reads from it.

The cache is deliberately in-process and deliberately not persisted: a summary
is a *reading* of a run, not a record of one. The authoritative history of what
actually happened is the trace/event store, which is append-only and on disk.

The registry is not a global singleton. The application builds one in its
lifespan hook and hands it to the dependency, which keeps tests isolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.evaluation import EvaluationSummary
from app.evaluation.datasets import load_dataset
from app.evaluation.service import DEFAULT_THRESHOLD, EvaluationService


@dataclass(slots=True)
class EvaluationRegistry:
    """The latest evaluation summary per dataset reference."""

    #: ``"memory-core@v1" -> summary``.
    _summaries: dict[str, EvaluationSummary] = field(default_factory=dict)
    #: Monotonic run counter, so a client can tell two runs apart.
    _runs: int = 0

    def latest(self) -> EvaluationSummary | None:
        """The most recently produced summary, or ``None``."""
        if not self._summaries:
            return None
        return self._summaries[self.reference_of(self._summaries)]

    def reference_of(self, summaries: dict[str, EvaluationSummary]) -> str:
        """The reference of the most recently stored summary."""
        return next(reversed(summaries))

    def get(self, reference: str) -> EvaluationSummary | None:
        """The cached summary for one ``name@version``, or ``None``."""
        return self._summaries.get(reference)

    def run(
        self,
        name: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        settings: object | None = None,
        force: bool = False,
    ) -> EvaluationSummary:
        """Return the cached summary for ``name``, running the dataset if needed.

        ``force`` re-runs even when a summary is cached, which is what the CLI
        does on every invocation.
        """
        dataset = load_dataset(name)
        reference = dataset.reference
        cached = self._summaries.get(reference)
        if cached is not None and not force:
            return cached
        service = EvaluationService(settings)  # type: ignore[arg-type]
        summary = service.evaluate(dataset, threshold=threshold)
        self._runs += 1
        self._summaries[reference] = summary
        return summary

    @property
    def run_count(self) -> int:
        """How many evaluation runs this registry has produced."""
        return self._runs


__all__ = ["EvaluationRegistry"]
