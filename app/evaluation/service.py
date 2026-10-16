"""The evaluation service (M6).

    EvaluationService
        ↓  builds a real, isolated MemForge stack per case
    MemoryService / MemoryRetrievalService / ContextCompilerService / AgentService
        ↓  produces real results
    a handful of counts
        ↓
    CaseResult -> MetricResult -> EvaluationSummary

Three rules make the numbers trustworthy:

1. **Nothing is re-implemented.** A case never scores memories itself. It asks
   M3 to search, M4 to compile, M2 to detect and M1 to store -- then it counts
   what came back. If MemForge's behaviour changes, the scores change; there is
   no second scoring path that could agree with it by accident.
2. **Every case is isolated.** Each case gets a fresh SQLite file, a fresh
   manual clock and its own deterministic id namespace, so cases cannot leak
   state into each other and two runs are identical.
3. **A metric never appears without a count.** A case that declares a metric it
   could not measure *fails* rather than contributing a silent zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.domain.agent import AgentRequest
from app.domain.context import ContextBlock, ContextRequest, ContextResult
from app.domain.errors import EvaluationError
from app.domain.evaluation import (
    CaseAction,
    CaseResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationSummary,
    LifecycleStep,
    MetricName,
    summarise,
)
from app.domain.ids import SequentialIdGenerator
from app.domain.models import MemoryStatus, MemoryType
from app.domain.retrieval import RetrievalMode, RetrievalQuery, RetrievalResponse
from app.domain.time import ManualClock
from app.evaluation.metrics import TOP_K, compute_metrics
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.factory import build_llm_provider
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository

#: Score a run must reach when no threshold is configured (i.e. all cases pass).
DEFAULT_THRESHOLD = 1.0

#: Namespace prefix of the per-case deterministic id generator.
ID_NAMESPACE = "memforge-eval"

#: Token budget used when a case does not name one.
FALLBACK_BUDGET = 1200


@dataclass(frozen=True, slots=True)
class CaseStack:
    """A real MemForge stack over one throwaway SQLite file."""

    clock: ManualClock
    memories: MemoryService
    retrieval: MemoryRetrievalService
    compiler: ContextCompilerService
    operations: MemoryOperationsService
    agent: AgentService

    def store_fixture(self, case: EvaluationCase) -> dict[str, str]:
        """Store every fixture memory; return ``label -> memory id``.

        The clock advances one minute per memory, so a "recent vs old" case is
        reproducible instead of depending on when the suite happened to run.
        """
        ids: dict[str, str] = {}
        for memory in case.fixture:
            self.clock.advance(minutes=memory.age_minutes)
            stored = self.memories.create_memory(
                content=memory.content,
                memory_type=MemoryType(memory.memory_type),
                source=memory.source,
                confidence=memory.confidence,
                importance=memory.importance,
            )
            ids[memory.label] = stored.id
        return ids


class EvaluationService:
    """Runs a dataset against real MemForge services and counts the results."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings(log_level="WARNING")

    # -- running ----------------------------------------------------------

    def run_dataset(self, dataset: EvaluationDataset) -> EvaluationResult:
        """Run every case, in dataset order."""
        return EvaluationResult(
            dataset=dataset.name,
            version=dataset.version,
            cases=tuple(self.run_case(case) for case in dataset.cases),
        )

    def evaluate(
        self, dataset: EvaluationDataset, *, threshold: float = DEFAULT_THRESHOLD
    ) -> EvaluationSummary:
        """Run the dataset and reduce it to the reported summary."""
        result = self.run_dataset(dataset)
        return summarise(dataset, result, compute_metrics(result.cases), threshold=threshold)

    def run_case(self, case: EvaluationCase) -> CaseResult:
        """Run one case against its own isolated stack."""
        with TemporaryDirectory(prefix="memforge-eval-") as directory:
            stack = self._build(Path(directory) / "eval.sqlite", case.id)
            ids = stack.store_fixture(case)
            return _run(case, stack, ids)

    # -- construction -----------------------------------------------------

    def _build(self, db_path: Path, case_id: str) -> CaseStack:
        """Assemble a deterministic MemForge stack for one case."""
        settings = self._settings
        clock = ManualClock()
        repository = SqliteMemoryRepository(db_path)
        memories = MemoryService(
            repository,
            clock=clock,
            id_generator=SequentialIdGenerator(namespace=f"{ID_NAMESPACE}-{case_id}"),
        )
        memories.initialize()
        retrieval = MemoryRetrievalService(
            repository,
            clock=clock,
            ranking=settings.ranking_config(),
            intelligence=settings.intelligence_config(),
            embedding_store=SqliteEmbeddingStore(db_path),
        )
        compiler = ContextCompilerService(
            retrieval,
            policy=settings.context_policy(),
            estimator=settings.token_estimator_config(),
            max_token_budget=settings.max_context_token_budget,
        )
        operations = MemoryOperationsService(memories, retrieval, compiler)
        agent = AgentService(
            operations,
            build_llm_provider(settings),
            max_token_budget=settings.max_context_token_budget,
        )
        return CaseStack(
            clock=clock,
            memories=memories,
            retrieval=retrieval,
            compiler=compiler,
            operations=operations,
            agent=agent,
        )


# -- measurement accumulator -------------------------------------------------


@dataclass(slots=True)
class _Measurement:
    """Accumulates one case's metric credits, reported figures and failures."""

    declared: frozenset[MetricName]
    numerators: dict[MetricName, int] = field(default_factory=dict)
    denominators: dict[MetricName, int] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def add(self, metric: MetricName, earned: int, total: int) -> None:
        """Record one metric's credit, clamped to the total that was measured."""
        if metric not in self.declared or total <= 0:
            return
        self.numerators[metric] = self.numerators.get(metric, 0) + min(max(earned, 0), total)
        self.denominators[metric] = self.denominators.get(metric, 0) + total

    def fail(self, check: str, detail: str) -> None:
        """Record a failed expectation as ``"<check>: <detail>"``."""
        self.failures.append(f"{check}: {detail}")

    def figure(self, name: str, value: float) -> None:
        """Record a reported figure (tokens, counts) for the dashboard."""
        self.measurements[name] = float(value)

    def result(self, case: EvaluationCase) -> CaseResult:
        """Freeze the accumulated counts into a case result."""
        return CaseResult(
            case_id=case.id,
            category=case.category,
            passed=not self.failures,
            failures=tuple(self.failures),
            numerators=dict(self.numerators),
            denominators=dict(self.denominators),
            measurements=dict(self.measurements),
        )


def _run(case: EvaluationCase, stack: CaseStack, ids: dict[str, str]) -> CaseResult:
    """Dispatch one case to the runner for its action and freeze the outcome."""
    outcome = _Measurement(declared=frozenset(case.expected.metrics))
    _RUNNERS[case.action](case, stack, ids, outcome)
    measured = set(outcome.denominators)
    for metric in sorted(set(case.expected.metrics) - measured, key=lambda item: item.value):
        outcome.fail(
            "measurement",
            f"the case declared the '{metric.value}' metric but measured nothing for it",
        )
    return outcome.result(case)


# -- case runners ------------------------------------------------------------


def _run_search(
    case: EvaluationCase, stack: CaseStack, ids: dict[str, str], out: _Measurement
) -> None:
    """Store the corpus and run one M3 search."""
    response = _search(stack, case)
    _score_search(case, response, _labels_of(response, ids), out)


def _run_search_with_duplicates(
    case: EvaluationCase, stack: CaseStack, ids: dict[str, str], out: _Measurement
) -> None:
    """Let M2 detect duplicates of the anchor memory, then search again.

    Detection starts from the first fixture only, which is how M2 is actually
    used (a memory arrives, its near-duplicates are found). It also produces a
    clean star of ``DUPLICATE_OF`` links instead of a chain, so each copy has one
    canonical memory to be suppressed against.
    """
    stack.memories.find_duplicates(ids[case.fixture[0].label], persist=True)
    _run_search(case, stack, ids, out)


def _run_search_with_conflict(
    case: EvaluationCase, stack: CaseStack, ids: dict[str, str], out: _Measurement
) -> None:
    """Detect a conflict, let M2 resolve it, then compile and search."""
    winner_id = _detect_and_resolve(stack, case, ids[case.fixture[0].label])
    _score_conflict(case, ids, winner_id, out)
    context = _compile(stack, case, case.expected.token_budget or FALLBACK_BUDGET)
    _score_context(case, context, [block.memory_id for block in context.blocks], ids, out)
    response = _search(stack, case)
    _score_search(case, response, _labels_of(response, ids), out)


def _run_compile(
    case: EvaluationCase, stack: CaseStack, ids: dict[str, str], out: _Measurement
) -> None:
    """Store the corpus and compile one budgeted context."""
    if case.expected.duplicates_suppressed:
        stack.memories.find_duplicates(ids[case.fixture[0].label], persist=True)
    if case.resolve_conflict:
        winner_id = _detect_and_resolve(
            stack,
            case,
            ids[case.fixture[0].label],
            archive_loser=case.archive_conflict_loser,
        )
        _score_conflict(case, ids, winner_id, out)
    budget = _require_budget(case)
    context = _compile(stack, case, budget)
    _score_context(case, context, [block.memory_id for block in context.blocks], ids, out)


def _run_lifecycle(
    case: EvaluationCase, stack: CaseStack, ids: dict[str, str], out: _Measurement
) -> None:
    """Apply the case's declared M1 steps and check the history survived."""
    target = ids[case.fixture[0].label]
    versions = [len(stack.memories.get_history(target))]
    for step in case.lifecycle_steps:
        _apply_lifecycle_step(stack, target, step)
        versions.append(len(stack.memories.get_history(target)))
    out.figure("versions_after_create", versions[0])
    out.figure("versions_after_lifecycle", versions[-1])
    out.figure("history_steps", len(versions) - 1)

    expected = _require_expectation(case, "expected_versions")
    out.add(MetricName.HISTORY_PRESERVATION, int(versions[-1] == expected), 1)
    if versions[-1] != expected:
        out.fail("history", f"expected {expected} versions, found {versions[-1]} ({versions})")
    regressed = [
        (step.value, before, after)
        for step, before, after in zip(case.lifecycle_steps, versions, versions[1:], strict=False)
        # FORGET_AGAIN is documented to change nothing, so it is the one step
        # that is *expected* not to append a version.
        if step is not LifecycleStep.FORGET_AGAIN and after <= before
    ]
    if regressed:
        out.fail(
            "history",
            "a lifecycle step did not append a version: "
            + ", ".join(f"{name} {before}->{after}" for name, before, after in regressed),
        )


def _apply_lifecycle_step(stack: CaseStack, target: str, step: LifecycleStep) -> None:
    """Apply one declared lifecycle operation through the real facade.

    ``FORGET_AGAIN`` repeats a forget, which M1 documents as a successful no-op:
    it must not append a version. Asserting an exact version count is therefore a
    real check of that guarantee rather than a tautology.
    """
    if step is LifecycleStep.UPDATE:
        current = stack.operations.memory_service.get_memory(target)
        stack.operations.update(target, content=f"{current.content} (reviewed)")
    elif step is LifecycleStep.ARCHIVE:
        stack.operations.update(target, status=MemoryStatus.ARCHIVED)
    elif step is LifecycleStep.FORGET:
        stack.operations.forget(target)
    else:
        stack.operations.forget(target)


def _run_agent(
    case: EvaluationCase, stack: CaseStack, ids: dict[str, str], out: _Measurement
) -> None:
    """Run one agent turn and judge its write policy and its recall."""
    result = stack.agent.respond(
        AgentRequest(
            request=case.query,
            token_budget=case.expected.token_budget or FALLBACK_BUDGET,
            top_k=case.top_k,
            extract_candidates=True,
            store_candidates=case.store_candidates,
        )
    )
    out.figure("gate_recalled", 1.0 if result.gate.should_recall else 0.0)
    out.figure("context_blocks", len(result.context_blocks))
    out.figure("context_estimated_tokens", result.context_estimated_tokens)
    out.figure("candidates", len(result.candidate_memories))
    out.figure("stored", len(result.stored_memory_ids))

    expected_stored = _require_expectation(case, "expected_stored")
    out.add(
        MetricName.MEMORY_WRITE_POLICY_COMPLIANCE,
        int(len(result.stored_memory_ids) == expected_stored),
        1,
    )
    if len(result.stored_memory_ids) != expected_stored:
        out.fail(
            "write_policy",
            f"expected {expected_stored} stored memories, found {len(result.stored_memory_ids)}",
        )

    recalled = {block.memory_id for block in result.context_blocks}
    missing = sorted(label for label in case.expected.required if ids[label] not in recalled)
    noisy = sorted(label for label in case.expected.forbidden if ids[label] in recalled)
    if case.expected.required or case.expected.forbidden:
        out.add(MetricName.RECALL_RELEVANCE, int(not missing and not noisy), 1)
    for label in missing:
        out.fail("recall", f"{label!r} should have been recalled but was not")
    for label in noisy:
        out.fail("irrelevant", f"{label!r} was declared irrelevant but was recalled")


_RUNNERS = {
    CaseAction.SEARCH: _run_search,
    CaseAction.SEARCH_WITH_DUPLICATES: _run_search_with_duplicates,
    CaseAction.SEARCH_WITH_CONFLICT: _run_search_with_conflict,
    CaseAction.COMPILE: _run_compile,
    CaseAction.LIFECYCLE: _run_lifecycle,
    CaseAction.AGENT: _run_agent,
}


# -- shared helpers ----------------------------------------------------------


def _search(stack: CaseStack, case: EvaluationCase) -> RetrievalResponse:
    """Run one lexical search through the real M3 service."""
    return stack.retrieval.search(
        RetrievalQuery(text=case.query, top_k=case.top_k, mode=RetrievalMode.LEXICAL)
    )


def _compile(stack: CaseStack, case: EvaluationCase, budget: int) -> ContextResult:
    """Compile one context through the real M4 compiler."""
    return stack.compiler.compile(
        ContextRequest(
            query=case.query,
            token_budget=budget,
            top_k=case.top_k,
            retrieval_mode=RetrievalMode.LEXICAL,
        )
    )


def _detect_and_resolve(
    stack: CaseStack, case: EvaluationCase, anchor: str, *, archive_loser: bool = True
) -> str | None:
    """Record a conflict with M2 and apply its deterministic resolution.

    Returns the id M2 actually chose as the winner, or ``None`` when no
    resolution was requested. The winner is read back from M2's own return
    value -- it is never recomputed here. A case that names no winner only asks
    M2 to *detect* conflicts, so "there is no conflict" stays a passing result
    rather than an error.
    """
    stack.memories.detect_conflicts(anchor, persist=True)
    if not case.resolve_conflict or case.expected.winner is None:
        return None
    resolution = stack.memories.resolve_conflict(anchor, archive_loser=archive_loser)
    return resolution.winner.id


def _score_conflict(
    case: EvaluationCase, ids: dict[str, str], winner_id: str | None, out: _Measurement
) -> None:
    """Score conflict resolution against the winner M2 actually chose."""
    expected = case.expected.winner
    if expected is None:
        return
    out.add(
        MetricName.CONFLICT_RESOLUTION_ACCURACY,
        int(winner_id is not None and winner_id == ids[expected]),
        1,
    )
    if winner_id != ids[expected]:
        out.fail(
            "conflict_resolution",
            f"expected {expected!r} to win the conflict, but "
            f"{'nothing' if winner_id is None else 'another memory'} did",
        )


def _require_budget(case: EvaluationCase) -> int:
    """Return the case's token budget, or refuse the case."""
    budget = case.expected.token_budget
    if budget is None:
        raise EvaluationError(
            f"case {case.id!r} compiles a context but declares no expected.token_budget"
        )
    return budget


def _require_expectation(case: EvaluationCase, field_name: str) -> int:
    """Return an expected count the case must declare, or refuse the case."""
    value: int | None = getattr(case.expected, field_name)
    if value is None:
        raise EvaluationError(
            f"case {case.id!r} runs action {case.action.value!r} "
            f"but declares no expected.{field_name}"
        )
    return value


def _labels_of(response: RetrievalResponse, ids: dict[str, str]) -> list[str]:
    """Map returned memory ids back to fixture labels, in rank order."""
    reverse = {memory_id: label for label, memory_id in ids.items()}
    return [reverse[result.memory.id] for result in response.results if result.memory.id in reverse]


# -- scoring -----------------------------------------------------------------


def _score_search(
    case: EvaluationCase,
    response: RetrievalResponse,
    returned: list[str],
    out: _Measurement,
) -> None:
    """Score the retrieval and duplicate metrics from one real search.

    Recall, top-k and the declared-absence checks are **hard** expectations: a
    case that misses one fails, because its fixture says what should happen.
    Precision is **graded**: an extra returned memory lowers the metric instead
    of failing the case, because "how much of what we returned was useful" is a
    quality signal, not a binary rule.
    """
    expected = case.expected
    out.figure("candidates_considered", response.candidates_considered)
    out.figure("results_returned", len(response.results))
    out.figure("results_suppressed", len(response.suppressed))

    if expected.required:
        hits = [label for label in expected.required if label in returned]
        out.add(MetricName.RETRIEVAL_RECALL, len(hits), len(expected.required))
        for label in expected.required:
            if label not in returned:
                out.fail("recall", f"{label!r} was required but was not returned")
        top = set(returned[:TOP_K])
        out.add(MetricName.TOP_K_HIT_RATE, int(bool(set(expected.required) & top)), 1)
        if not set(expected.required) & top:
            out.fail(
                "top_k",
                f"no required memory appeared in the first {TOP_K} results "
                f"(ranked: {returned[:TOP_K] or 'none'})",
            )
        if returned:
            precise = sum(1 for label in returned if label in expected.required)
            out.add(MetricName.RETRIEVAL_PRECISION, precise, len(returned))

    if expected.forbidden:
        leaked = [label for label in expected.forbidden if label in returned]
        satisfied = not leaked and all(label in returned for label in expected.required)
        out.add(MetricName.RECALL_RELEVANCE, int(satisfied), 1)
        for label in leaked:
            out.fail("irrelevant", f"{label!r} was declared irrelevant but was returned")

    if expected.duplicates_suppressed:
        leaked = [label for label in expected.duplicates_suppressed if label in returned]
        total = len(expected.duplicates_suppressed)
        out.add(MetricName.DUPLICATE_SUPPRESSION_ACCURACY, total - len(leaked), total)
        for label in leaked:
            out.fail("duplicate", f"{label!r} was a duplicate but was still returned")


def _score_context(
    case: EvaluationCase,
    context: ContextResult,
    selected: list[str],
    ids: dict[str, str],
    out: _Measurement,
) -> None:
    """Score the context metrics from one real compilation."""
    expected = case.expected
    out.figure("token_budget", context.token_budget)
    out.figure("estimated_tokens", context.estimated_tokens)
    out.figure("remaining_tokens", context.remaining_tokens)
    out.figure("selected_count", len(selected))
    out.figure("suppressed_count", context.suppressed_count)

    within = context.estimated_tokens <= context.token_budget
    out.add(MetricName.CONTEXT_BUDGET_COMPLIANCE, int(within), 1)
    if not within:
        out.fail(
            "budget",
            f"estimated {context.estimated_tokens} tokens against a budget of "
            f"{context.token_budget}",
        )

    if expected.selected:
        chosen = set(selected)
        missing = [label for label in expected.selected if ids[label] not in chosen]
        out.add(
            MetricName.CONTEXT_SELECTION_ACCURACY,
            len(expected.selected) - len(missing),
            len(expected.selected),
        )
        for label in missing:
            out.fail("selection", f"{label!r} should have been selected but was not")

    if expected.stale:
        leaked = sorted(label for label in expected.stale if ids[label] in set(selected))
        total = len(expected.stale)
        out.add(MetricName.STALE_MEMORY_AVOIDANCE, total - len(leaked), total)
        for label in leaked:
            out.fail("stale", f"superseded memory {label!r} entered the final context")

    if expected.duplicates_suppressed:
        chosen = set(selected)
        leaked = [label for label in expected.duplicates_suppressed if ids[label] in chosen]
        total = len(expected.duplicates_suppressed)
        out.add(MetricName.DUPLICATE_SUPPRESSION_ACCURACY, total - len(leaked), total)
        for label in leaked:
            out.fail("duplicate", f"{label!r} duplicated another memory but was selected")

    complete = sum(1 for block in context.blocks if _has_provenance(block))
    out.add(MetricName.CONTEXT_PROVENANCE_COMPLETENESS, complete, len(context.blocks))
    for block in context.blocks:
        if not _has_provenance(block):
            out.fail("provenance", f"block for {block.memory_id!r} lacks complete provenance")


def _has_provenance(block: ContextBlock) -> bool:
    """True when a compiled block names one stored memory and its content.

    This is the check behind "Context Provenance Completeness": a reader must be
    able to take any line of a rendered context back to a single stored memory.
    """
    return bool(
        block.memory_id.strip()
        and block.rendered_content.strip()
        and block.memory_type is not None
        and block.section is not None
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "FALLBACK_BUDGET",
    "ID_NAMESPACE",
    "CaseStack",
    "EvaluationService",
]
