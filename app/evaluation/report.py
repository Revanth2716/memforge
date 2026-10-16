"""The deterministic evaluation report (M6).

Plain text, fixed column widths, numbers rounded to three decimals and every
value derived from a real run. Two runs of the same dataset against the same
MemForge build produce byte-identical output -- which is the property that makes
this usable as a regression gate rather than a slide.

The report deliberately shows its own limits. It prints what was measured, what
was configured and what the cases prove, next to each other, so a reader is never
left guessing which is which.
"""

from __future__ import annotations

from app.domain.evaluation import METRIC_DEFINITIONS, EvaluationSummary, MetricResult

#: Total report width.
WIDTH = 78

#: Column width of a metric label.
LABEL_WIDTH = 34

#: Decimal places for a reported score.
SCORE_PRECISION = 3


def render(summary: EvaluationSummary, *, verbose: bool = False) -> str:
    """Render one evaluation summary as deterministic text."""
    lines: list[str] = []
    lines.extend(_header(summary))
    lines.extend(_categories(summary))
    lines.extend(_metrics(summary))
    lines.extend(_failures(summary))
    lines.extend(_limits())
    if verbose:
        lines.extend(_metric_definitions(summary))
    lines.extend(_verdict(summary))
    return "\n".join(lines)


def _rule(char: str = "-") -> str:
    """A full-width rule."""
    return char * WIDTH


def _header(summary: EvaluationSummary) -> list[str]:
    """The dataset identity and the case counts."""
    return [
        "",
        _rule("="),
        "EVALUATION REPORT".center(WIDTH),
        _rule("="),
        f"  Dataset      : {summary.dataset}@{summary.version}",
        f"  Cases        : {summary.total_cases}",
        f"  Passed       : {summary.passed_cases}",
        f"  Failed       : {summary.failed_cases}",
        f"  Case score   : {_score(summary.case_score)}",
        f"  Metric score : {_score(summary.metric_score)}",
        f"  Overall      : {_score(summary.overall)}",
        f"  Threshold    : {_score(summary.threshold)} (minimum case pass rate)",
        _rule(),
    ]


def _categories(summary: EvaluationSummary) -> list[str]:
    """Per-category pass counts."""
    lines = ["  Categories:"]
    for name, (passed, total) in summary.categories.items():
        lines.append(f"    {name:<12} {passed}/{total}")
    return [*lines, ""]


def _metrics(summary: EvaluationSummary) -> list[str]:
    """Per-metric scores, with the counts they were computed from."""
    lines = ["  Metrics (measured):"]
    if not summary.metrics:
        lines.append("    (none -- no case declared an applicable metric)")
    for metric in summary.metrics:
        lines.append(f"    {_metric_line(metric)}")
    lines.append("")
    lines.append("  Recall, top-k, suppression and compliance are pass/fail checks;")
    lines.append("  precision is graded, so it can sit below 1.0 on a healthy build.")
    return [*lines, ""]


def _metric_line(metric: MetricResult) -> str:
    """One metric row: label, score, then the raw counts."""
    label = metric.name.value[: LABEL_WIDTH - 1].ljust(LABEL_WIDTH)
    counts = f"{metric.numerator}/{metric.denominator}"
    return f"{label} {_score(metric.score)}  ({counts})"


def _failures(summary: EvaluationSummary) -> list[str]:
    """Every failed case, with the checks that did not hold."""
    if not summary.failures:
        return ["  Failed cases: none", ""]
    lines = ["  Failed cases:"]
    for case in summary.failures:
        lines.append(f"    {case.case_id} [{case.category.value}]")
        for failure in case.failures:
            lines.append(f"      - {failure}")
    return [*lines, ""]


def _limits() -> list[str]:
    """What this report does and does not establish.

    Printed in every report, not only in verbose mode: a score without its
    boundary is easy to over-read.
    """
    return [
        _rule(),
        "  Scope of these measurements:",
        "    * Every number above is a count over real MemForge services.",
        "    * Each case proves only the behaviour its own fixture encodes, on",
        "      that fixture, with the configured defaults.",
        "    * A pass is not a claim about memory intelligence in general, and",
        "      no LLM-as-a-judge was involved in producing any number.",
        "    * Scores are not comparable across datasets or configurations.",
        _rule(),
    ]


def _metric_definitions(summary: EvaluationSummary) -> list[str]:
    """The exact formula behind every applicable metric."""
    lines = ["  Metric definitions:"]
    for metric in summary.metrics:
        lines.append(f"    {metric.name.value}: {METRIC_DEFINITIONS[metric.name]}")
    return [*lines, ""]


def _verdict(summary: EvaluationSummary) -> list[str]:
    """The pass/fail line the regression gate exits on."""
    verdict = "PASS" if summary.passed else "FAIL"
    if summary.failed_cases:
        verdict = f"FAIL ({summary.failed_cases} case(s) did not hold their expectations)"
    return ["", f"  VERDICT: {verdict}", ""]


def _score(value: float) -> str:
    """Format a score with a fixed number of decimals."""
    return f"{value:.{SCORE_PRECISION}f}"


__all__ = ["LABEL_WIDTH", "SCORE_PRECISION", "WIDTH", "render"]
