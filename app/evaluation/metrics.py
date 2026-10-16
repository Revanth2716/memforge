"""MemForge's evaluation metrics (M6).

Every metric is a ratio of two integers that the evaluation itself produced by
running real services. Nothing here estimates, samples or weights: the formulas
are in :data:`app.domain.evaluation.METRIC_DEFINITIONS` and are reproduced in
the report so a reader never has to trust a name.

The aggregation in this module is deliberately dumb: it sums numerators and
denominators across cases and divides once. That keeps every metric an
interpretable micro-average over real counts, and it means a metric can only
move when a real decision changed.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.domain.evaluation import (
    TOP_K,
    CaseResult,
    MetricName,
    MetricResult,
)

__all__ = ["TOP_K", "compute_metrics"]


def compute_metrics(results: Iterable[CaseResult]) -> tuple[MetricResult, ...]:
    """Aggregate case results into one :class:`MetricResult` per metric.

    A metric that no case applied is still returned, with a zero denominator, so
    a report always shows the same metric set. :attr:`MetricResult.applicable`
    separates "measured and perfect" from "never exercised".
    """
    totals: dict[MetricName, int] = {}
    counts: dict[MetricName, int] = {}
    failures: dict[MetricName, list[str]] = {}
    for result in results:
        for name, total in result.denominators.items():
            counts[name] = counts.get(name, 0) + total
            totals[name] = totals.get(name, 0) + result.numerators.get(name, 0)
        if result.passed:
            continue
        for name, total in result.denominators.items():
            if result.numerators.get(name, 0) < total:
                failures.setdefault(name, []).append(result.case_id)
    return tuple(
        MetricResult(
            name=name,
            numerator=totals.get(name, 0),
            denominator=counts.get(name, 0),
            failures=tuple(failures.get(name, ())),
        )
        for name in sorted(MetricName, key=lambda item: item.value)
    )


def credit(metric: MetricName, earned: int, total: int) -> tuple[int, int]:
    """Record one metric's contribution from one case.

    A zero total is normalised to zero credit rather than skipped, so a case
    that declared a metric but measured nothing cannot silently inflate it.
    """
    return (0, 0) if total <= 0 else (max(0, min(earned, total)), total)


__all__ = ["TOP_K", "compute_metrics"]
