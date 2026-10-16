"""Evaluation domain for MemForge (M6).

**This is not an LLM evaluation framework.** There is no judge model, no
reference-answer scoring of generated text and no general "is this smart?"
question. An evaluation here asks one narrow thing:

    given a corpus and a query, did the memory system make the decision its own
    documented rules say it should?

Every metric is a **count over a real execution**: the services run, the real
results come back, and a metric counts them. Nothing is sampled, nothing is
estimated, and no number appears in a report unless it was computed from a case
that actually ran.

What a pass means
-----------------
A passing case means *the behaviour encoded in that case held on that corpus*.
It does not mean the memory is "intelligent", that the ranking is optimal, or
that the system would behave the same way on a different corpus. The dataset is
small, hand-written and version-controlled precisely so that its claims stay
narrow.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from app.domain.errors import EvaluationError

#: Decimal places kept in reported ratios.
_PRECISION = 6

#: Rank depth used by the top-k hit rate. Fixed and documented so the metric
#: stays comparable between runs; it is deliberately *not* a case's ``top_k``.
TOP_K = 3


class CaseCategory(enum.StrEnum):
    """The MemForge subsystem a case exercises."""

    RETRIEVAL = "retrieval"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    CONTEXT = "context"
    LIFECYCLE = "lifecycle"
    AGENT = "agent"


class MetricName(enum.StrEnum):
    """Every metric MemForge computes.

    Each one is defined once, below, in terms of real execution output. A metric
    is added here only when it answers a question about the memory system that
    a count can settle.
    """

    # -- retrieval --------------------------------------------------------
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_PRECISION = "retrieval_precision"
    TOP_K_HIT_RATE = "top_k_hit_rate"
    # -- duplicates -------------------------------------------------------
    DUPLICATE_SUPPRESSION_ACCURACY = "duplicate_suppression_accuracy"
    # -- conflicts --------------------------------------------------------
    CONFLICT_RESOLUTION_ACCURACY = "conflict_resolution_accuracy"
    STALE_MEMORY_AVOIDANCE = "stale_memory_avoidance"
    # -- context ----------------------------------------------------------
    CONTEXT_BUDGET_COMPLIANCE = "context_budget_compliance"
    CONTEXT_SELECTION_ACCURACY = "context_selection_accuracy"
    CONTEXT_PROVENANCE_COMPLETENESS = "context_provenance_completeness"
    # -- lifecycle ---------------------------------------------------------
    HISTORY_PRESERVATION = "history_preservation"
    # -- agent ------------------------------------------------------------
    MEMORY_WRITE_POLICY_COMPLIANCE = "memory_write_policy_compliance"
    RECALL_RELEVANCE = "recall_relevance"


#: Documented definition of every metric, rendered in the report so a reader
#: never has to guess what a number means.
METRIC_DEFINITIONS: Mapping[MetricName, str] = MappingProxyType(
    {
        MetricName.RETRIEVAL_RECALL: (
            "share of required target memories that appeared in the retrieval results"
        ),
        MetricName.RETRIEVAL_PRECISION: (
            "share of returned memories that were declared relevant; a case that "
            "returned nothing is not counted (recall covers that failure)"
        ),
        MetricName.TOP_K_HIT_RATE: (
            "share of cases where at least one required memory appeared within the "
            f"first {TOP_K} ranked results"
        ),
        MetricName.DUPLICATE_SUPPRESSION_ACCURACY: (
            "share of expected redundant memories that were actually suppressed"
        ),
        MetricName.CONFLICT_RESOLUTION_ACCURACY: (
            "share of conflict cases where the expected winner was the one selected"
        ),
        MetricName.STALE_MEMORY_AVOIDANCE: (
            "share of cases where a superseded memory did not enter the final context"
        ),
        MetricName.CONTEXT_BUDGET_COMPLIANCE: (
            "share of compilations whose estimated tokens were <= the requested budget"
        ),
        MetricName.CONTEXT_SELECTION_ACCURACY: (
            "share of required memories selected into the compiled context"
        ),
        MetricName.CONTEXT_PROVENANCE_COMPLETENESS: (
            "share of selected blocks carrying complete source provenance"
        ),
        MetricName.HISTORY_PRESERVATION: (
            "share of lifecycle operations that preserved the required history"
        ),
        MetricName.MEMORY_WRITE_POLICY_COMPLIANCE: (
            "share of agent turns that stored candidates only when explicitly permitted"
        ),
        MetricName.RECALL_RELEVANCE: (
            "share of cases that returned no declared-irrelevant memory (and, when "
            "a case declares required memories, returned every one of them)"
        ),
    }
)


class CaseAction(enum.StrEnum):
    """What a case does with its fixture before it is evaluated.

    The evaluation never re-implements retrieval, conflict resolution or context
    compilation: it calls the same services any caller would. A case only
    describes the *corpus* and the *call*, and what should have happened.
    """

    #: Store the fixture memories, then run the query.
    SEARCH = "search"
    #: Store the fixture memories, detect duplicates, then run the query.
    SEARCH_WITH_DUPLICATES = "search_with_duplicates"
    #: Store the fixture memories, detect + optionally resolve a conflict, then
    #: compile and search.
    SEARCH_WITH_CONFLICT = "search_with_conflict"
    #: Store the fixture memories and compile a context under a budget.
    COMPILE = "compile"
    #: Apply the declared lifecycle steps to the first fixture, then check history.
    LIFECYCLE = "lifecycle"
    #: Store the fixture memories and run one agent turn.
    AGENT = "agent"


class LifecycleStep(enum.StrEnum):
    """One M1 operation a ``LIFECYCLE`` case applies to its memory.

    The steps are the documented, auditable lifecycle operations. Each one that
    succeeds appends exactly one history version, except ``FORGET_AGAIN`` which
    is a documented no-op -- that difference is the point of the cases.
    """

    UPDATE = "update"
    ARCHIVE = "archive"
    FORGET = "forget"
    #: Forget an already forgotten memory: succeeds, changes nothing.
    FORGET_AGAIN = "forget_again"


@dataclass(frozen=True, slots=True)
class MemoryFixture:
    """One memory to store while building a case's corpus."""

    #: Stable label used by the expectations instead of a generated id.
    label: str
    content: str
    memory_type: str
    confidence: float = 0.5
    importance: float = 0.5
    source: str | None = None
    #: Minutes the manual clock advances before this memory is stored, so a
    #: "recent vs old" case is reproducible instead of depending on run time.
    age_minutes: int = 1

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise EvaluationError("a fixture label must be a non-empty string")
        if not self.content or not self.content.strip():
            raise EvaluationError(f"fixture {self.label!r} has no content")
        if self.age_minutes < 0:
            raise EvaluationError(f"fixture {self.label!r} age_minutes must be >= 0")
        for field_name, value in (
            ("confidence", self.confidence),
            ("importance", self.importance),
        ):
            if not 0.0 <= value <= 1.0:
                raise EvaluationError(f"fixture {self.label!r} {field_name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ExpectedBehavior:
    """What the case asserts, expressed against fixture *labels*.

    Labels rather than ids keep a case readable and keep it stable across runs:
    the ids are generated at execution time, the expectations are not.
    """

    #: Labels that must appear in the result.
    required: tuple[str, ...] = ()
    #: Labels that must appear in the result.
    forbidden: tuple[str, ...] = ()
    #: Labels that must be absent because they are duplicates.
    duplicates_suppressed: tuple[str, ...] = ()
    #: Labels that must be absent because they are superseded.
    stale: tuple[str, ...] = ()
    #: Labels that must be selected into the compiled context.
    selected: tuple[str, ...] = ()
    #: Exact number of memories expected to be stored by an agent turn.
    expected_stored: int | None = None
    #: Label that must win a conflict.
    winner: str | None = None
    #: Number of history versions expected after a lifecycle operation.
    expected_versions: int | None = None
    #: Token budget for a context case.
    token_budget: int | None = None
    #: Metrics this case contributes to.
    metrics: tuple[MetricName, ...] = ()

    def __post_init__(self) -> None:
        if not self.metrics:
            raise EvaluationError("a case must declare at least one applicable metric")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One deterministic, version-controlled evaluation case."""

    id: str
    title: str
    category: CaseCategory
    action: CaseAction
    #: The query (or task) handed to MemForge.
    query: str
    #: The corpus to build before evaluating.
    fixture: tuple[MemoryFixture, ...]
    expected: ExpectedBehavior
    #: How many results to ask for (``top_k`` for search/recall cases).
    top_k: int = 10
    #: Resolve the conflict M2 detects, and score the winner it chose.
    resolve_conflict: bool = True
    #: Archive the losing memory when resolving. When ``False`` the loser stays
    #: ACTIVE but carries a ``SUPERSEDES`` link, which is how M4 gets to prove
    #: it suppresses a superseded memory rather than merely losing a status filter.
    archive_conflict_loser: bool = True
    #: Store the agent's candidate memories for ``AGENT`` cases.
    store_candidates: bool = False
    #: The M1 operations a ``LIFECYCLE`` case applies, in order.
    lifecycle_steps: tuple[LifecycleStep, ...] = ()
    #: What the case is for (shown in the report).
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise EvaluationError("a case id must be a non-empty string")
        if not self.query or not self.query.strip():
            raise EvaluationError(f"case {self.id!r} has no query")
        if self.top_k < 1:
            raise EvaluationError(f"case {self.id!r} top_k must be >= 1")
        if self.action is CaseAction.LIFECYCLE and not self.lifecycle_steps:
            raise EvaluationError(f"case {self.id!r} declares no lifecycle_steps")
        if self.action is not CaseAction.LIFECYCLE and self.lifecycle_steps:
            raise EvaluationError(
                f"case {self.id!r} declares lifecycle_steps but does not run a lifecycle"
            )
        labels = [memory.label for memory in self.fixture]
        if len(labels) != len(set(labels)):
            raise EvaluationError(f"case {self.id!r} has duplicate fixture labels")
        if not self.fixture:
            raise EvaluationError(f"case {self.id!r} needs at least one fixture memory")
        for label in (
            self.expected.required
            + self.expected.forbidden
            + self.expected.duplicates_suppressed
            + self.expected.stale
            + self.expected.selected
        ):
            if label not in labels:
                raise EvaluationError(
                    f"case {self.id!r} expects label {label!r}, which is not a fixture"
                )
        if self.expected.winner is not None and self.expected.winner not in labels:
            raise EvaluationError(
                f"case {self.id!r} names a winner {self.expected.winner!r} that is not a fixture"
            )


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A versioned, deterministic set of cases."""

    name: str
    version: str
    cases: tuple[EvaluationCase, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise EvaluationError("a dataset needs a name")
        if not self.version or not self.version.strip():
            raise EvaluationError("a dataset needs a version")
        if not self.cases:
            raise EvaluationError("a dataset needs at least one case")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise EvaluationError("case ids must be unique within a dataset")

    @property
    def reference(self) -> str:
        """The ``name@version`` reference reported in every summary."""
        return f"{self.name}@{self.version}"

    def cases_for(self, category: CaseCategory) -> tuple[EvaluationCase, ...]:
        """The cases of one category, in dataset order."""
        return tuple(case for case in self.cases if case.category is category)

    @property
    def categories(self) -> tuple[CaseCategory, ...]:
        """The categories present, in declaration order."""
        seen: list[CaseCategory] = []
        for case in self.cases:
            if case.category not in seen:
                seen.append(case.category)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One metric computed over the cases that applied it."""

    name: MetricName
    #: Numerator and denominator, so a ratio is always inspectable.
    numerator: int
    denominator: int
    #: Cases that contributed zero to the numerator.
    failures: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """The metric value in ``[0, 1]``; ``1.0`` when nothing applied."""
        if self.denominator == 0:
            return 1.0
        return round(self.numerator / self.denominator, _PRECISION)

    @property
    def applicable(self) -> bool:
        """False when no case applied this metric (it is not counted)."""
        return self.denominator > 0

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "name": self.name.value,
            "definition": METRIC_DEFINITIONS[self.name],
            "numerator": self.numerator,
            "denominator": self.denominator,
            "score": self.score,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The outcome of running one case, in raw counts.

    ``numerators``/``denominators`` are what makes every reported metric
    inspectable: a ratio is always a number over a number, and both numbers come
    from the execution. ``measurements`` carries the context figures (tokens,
    counts) the report and the dashboard show.
    """

    case_id: str
    category: CaseCategory
    passed: bool
    #: The checks that failed, as ``"<check>: <detail>"`` (empty when passing).
    failures: tuple[str, ...] = ()
    #: Per-metric credit this execution earned.
    numerators: Mapping[MetricName, int] = field(default_factory=dict)
    #: Per-metric total this execution was judged against.
    denominators: Mapping[MetricName, int] = field(default_factory=dict)
    #: Context figures, reported for the dashboard.
    measurements: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "case_id": self.case_id,
            "category": self.category.value,
            "passed": self.passed,
            "failures": list(self.failures),
            "numerators": {
                name.value: value for name, value in sorted(self.numerators.items(), key=_by_value)
            },
            "denominators": {
                name.value: value
                for name, value in sorted(self.denominators.items(), key=_by_value)
            },
            "measurements": dict(sorted(self.measurements.items())),
        }


def _by_value(item: tuple[MetricName, int]) -> str:
    """Sort key for metric-keyed mappings."""
    return item[0].value


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Every case result, in dataset order."""

    dataset: str
    version: str
    cases: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        """How many cases ran."""
        return len(self.cases)

    @property
    def passed(self) -> int:
        """How many cases held their expectations."""
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed(self) -> int:
        """How many cases did not."""
        return self.total - self.passed

    @property
    def failed_cases(self) -> tuple[CaseResult, ...]:
        """The failing cases, in dataset order."""
        return tuple(case for case in self.cases if not case.passed)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "dataset": self.dataset,
            "version": self.version,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """The complete, deterministic result of one evaluation run."""

    dataset: str
    version: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    #: Per-category pass counts, in dataset category order.
    categories: Mapping[str, tuple[int, int]]
    #: Every metric that at least one case applied.
    metrics: tuple[MetricResult, ...]
    #: The per-case failures behind the score.
    failures: tuple[CaseResult, ...]
    #: The threshold this run was judged against: the minimum **case pass rate**
    #: required. The default of ``1.0`` means every case must hold its
    #: expectations. :attr:`overall` folds in graded metrics such as retrieval
    #: precision and is reported for comparison, not gated, so that a partly
    #: useful result set cannot silently fail an otherwise healthy build.
    threshold: float

    @property
    def case_score(self) -> float:
        """Passed cases over total cases."""
        if self.total_cases == 0:
            return 1.0
        return round(self.passed_cases / self.total_cases, _PRECISION)

    @property
    def metric_score(self) -> float:
        """The unweighted mean of every applicable metric."""
        applicable = [metric.score for metric in self.metrics if metric.applicable]
        if not applicable:
            return 1.0
        return round(sum(applicable) / len(applicable), _PRECISION)

    @property
    def overall(self) -> float:
        """The mean of the case score and the metric score."""
        return round((self.case_score + self.metric_score) / 2, _PRECISION)

    @property
    def passed(self) -> bool:
        """True when the case pass rate meets the threshold.

        The gate is per-case. Graded metrics (retrieval precision, for one)
        move :attr:`overall` without failing a case, so they stay visible
        without making the gate sensitive to a partly useful result set.
        """
        return self.case_score >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "dataset": self.dataset,
            "version": self.version,
            "reference": f"{self.dataset}@{self.version}",
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "case_score": self.case_score,
            "metric_score": self.metric_score,
            "overall": self.overall,
            "threshold": self.threshold,
            "passed": self.passed,
            "categories": {
                name: {"passed": counts[0], "total": counts[1]}
                for name, counts in self.categories.items()
            },
            "metrics": [metric.to_dict() for metric in self.metrics],
            "failures": [case.to_dict() for case in self.failures],
        }


def summarise(
    dataset: EvaluationDataset,
    result: EvaluationResult,
    metrics: Sequence[MetricResult],
    *,
    threshold: float,
) -> EvaluationSummary:
    """Combine a result and its metrics into the reported summary."""
    if not 0.0 <= threshold <= 1.0:
        raise EvaluationError("threshold must be in [0, 1]")
    categories: dict[str, tuple[int, int]] = {}
    for case in result.cases:
        passed, total = categories.get(case.category.value, (0, 0))
        categories[case.category.value] = (passed + (1 if case.passed else 0), total + 1)
    return EvaluationSummary(
        dataset=dataset.name,
        version=dataset.version,
        total_cases=result.total,
        passed_cases=result.passed,
        failed_cases=result.failed,
        categories=MappingProxyType(dict(sorted(categories.items()))),
        metrics=tuple(sorted(metrics, key=lambda item: item.name.value)),
        failures=result.failed_cases,
        threshold=threshold,
    )


__all__ = [
    "METRIC_DEFINITIONS",
    "TOP_K",
    "CaseAction",
    "CaseCategory",
    "CaseResult",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationError",
    "EvaluationResult",
    "EvaluationSummary",
    "ExpectedBehavior",
    "LifecycleStep",
    "MemoryFixture",
    "MetricName",
    "MetricResult",
    "summarise",
]
