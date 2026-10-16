"""Deterministic command-line demo for milestones 1, 2, 3 and 4.

The demo walks through the whole lifecycle on a throwaway SQLite file:

M1: create -> list -> update -> archive -> history -> forget -> history
M2: score importance -> detect near-duplicate -> detect conflict ->
    resolve conflict -> show explanation -> show history remains intact
M3: retrieve for a query (lexical) -> retrieve again (hybrid, explained)
M4: build a memory pool -> compile context under a token budget ->
    show selection, suppression, provenance and the remaining budget

Identifiers come from :class:`SequentialIdGenerator` and timestamps from
:class:`ManualClock`, so running the demo twice produces byte-identical output.
No LLM, no network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings
from app.domain.agent import AgentRequest
from app.domain.context import ContextRequest
from app.domain.embeddings import HashedLexicalEmbeddingProvider
from app.domain.errors import MemforgeError, MemoryNotFoundError
from app.domain.ids import SequentialIdGenerator
from app.domain.models import Memory, MemoryStatus, MemoryType, MemoryVersion
from app.domain.retrieval import RetrievalMode, RetrievalQuery
from app.domain.time import ManualClock
from app.domain.updates import MemoryUpdate
from app.evaluation.datasets import load_dataset
from app.evaluation.report import render
from app.evaluation.service import EvaluationService
from app.observability.demo import DeterministicTimer
from app.observability.instrumentation import (
    AgentTracer,
    build_traced_operations,
    trace_provider,
)
from app.observability.service import ObservabilityService, SequentialIdFactory
from app.services.agent_service import AgentService
from app.services.context_service import ContextCompilerService
from app.services.llm.factory import build_llm_provider
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.base import MemoryQuery
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository
from app.storage.sqlite.trace_store import SqliteTraceStore

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "demo.sqlite"
WIDTH = 78
CONTENT_WIDTH = 42


def _demo_settings(db_path: Path, llm_provider: str) -> Settings:
    """Settings for a demo run: a throwaway database and a chosen LLM provider.

    The custom provider is read from the environment exactly as the server would
    read it, so the demo never has a credential of its own and never writes one.
    """
    return Settings(
        database_path=db_path,
        llm_provider=llm_provider,
        log_level="WARNING",
    )


def _rule(char: str = "=") -> None:
    print(char * WIDTH)


def _step(number: int, title: str) -> None:
    print()
    _rule()
    print(f"STEP {number} -- {title}")
    _rule()


def _short(text: str, width: int = CONTENT_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "~"


def _print_memory(memory: Memory) -> None:
    print(
        f"  {memory.id}  {memory.memory_type.value:<11} {memory.status.value:<8} "
        f"v{memory.version}  conf={memory.confidence:.2f} imp={memory.importance:.2f}"
    )
    print(f"      {_short(memory.content)}")


def _print_version(version: MemoryVersion) -> None:
    print(
        f"  v{version.version:<2} {version.change.value:<9} {version.recorded_at.isoformat()} "
        f"status={version.status.value:<8} imp={version.importance:.2f}"
    )
    print(f"      {_short(version.content)}")


def build_demo_service(
    db_path: Path,
    settings: Settings | None = None,
) -> tuple[MemoryService, ManualClock, MemoryRetrievalService, MemoryRetrievalService]:
    """Create a deterministic service backed by a fresh SQLite database.

    Any existing file at ``db_path`` is removed first, so the demo always starts
    from an empty database (and therefore always produces the same output).
    """
    if db_path.exists():
        db_path.unlink()
    clock = ManualClock()
    repository = SqliteMemoryRepository(db_path)
    service = MemoryService(
        repository, clock=clock, id_generator=SequentialIdGenerator(prefix="demo")
    )
    service.initialize()
    retriever = MemoryRetrievalService(
        repository,
        clock=clock,
        embedding_store=SqliteEmbeddingStore(db_path),
        ranking=(settings or Settings()).ranking_config(),
        intelligence=(settings or Settings()).intelligence_config(),
    )
    semantic_retriever = MemoryRetrievalService(
        repository,
        clock=clock,
        embedding_provider=HashedLexicalEmbeddingProvider(dimensions=64),
        embedding_store=SqliteEmbeddingStore(db_path),
    )
    return service, clock, retriever, semantic_retriever


def run_demo(db_path: Path, *, llm_provider: str = "mock", store_candidates: bool = False) -> int:
    """Run the full M1-M5 walkthrough. Returns a process exit code.

    Args:
        db_path: Throwaway SQLite file, recreated from scratch.
        llm_provider: ``mock`` (default, deterministic and offline) or ``custom``
            (opt-in: requires ``MEMFORGE_LLM_*`` and network access).
        store_candidates: Whether the M5 step may persist extracted candidates.
    """
    settings = _demo_settings(db_path, llm_provider)
    service, clock, retriever, semantic_retriever = build_demo_service(db_path, settings)
    provider = build_llm_provider(settings)
    print("=" * WIDTH)
    print(
        "MemForge -- memory core (M1) + intelligence (M2) + retrieval (M3) "
        "+ context (M4) + agent (M5) demo".center(WIDTH)
    )
    print(f"database: {db_path}".center(WIDTH))
    print(f"llm provider: {provider.describe()}".center(WIDTH))
    print("=" * WIDTH)

    # ---------------------------------------------------------------- create
    _step(1, "CREATE three memories")
    clock.advance(minutes=1)
    profile = service.create_memory(
        content="User is a fresher targeting AI Engineer roles.",
        memory_type=MemoryType.PROFILE,
        source="user:revanth",
        confidence=1.0,
        importance=0.9,
    )
    clock.advance(minutes=1)
    preference = service.create_memory(
        content="Prefers concise answers with runnable code examples.",
        memory_type=MemoryType.PREFERENCE,
        source="user:revanth",
        confidence=0.9,
        importance=0.7,
    )
    clock.advance(minutes=1)
    task = service.create_memory(
        content="Finish MemForge milestone 1 and run the full test suite.",
        memory_type=MemoryType.TASK,
        source="agent:planner",
        confidence=1.0,
        importance=0.8,
    )
    for memory in (profile, preference, task):
        _print_memory(memory)
    print(f"  -> created {len([profile, preference, task])} memories (version 1 each)")

    # ------------------------------------------------------------------ list
    _step(2, "LIST active memories (newest first)")
    page = service.list_memories()
    for memory in page.items:
        _print_memory(memory)
    print(f"  -> {page.total} active memories")

    # ---------------------------------------------------------------- update
    _step(3, "UPDATE the task memory, then ARCHIVE the preference")
    clock.advance(minutes=5)
    task = service.update_memory(
        task.id,
        MemoryUpdate(
            content="Ship MemForge milestone 1 with tests, docs and a demo.", importance=1.0
        ),
    )
    _print_memory(task)
    print("  -> version bumped, new history entry appended")
    clock.advance(minutes=5)
    preference = service.update_memory(preference.id, MemoryUpdate(status=MemoryStatus.ARCHIVED))
    _print_memory(preference)

    # --------------------------------------------------------------- history
    _step(4, "HISTORY of the updated task memory")
    task_history = service.get_history(task.id)
    for version in task_history:
        _print_version(version)
    print(f"  -> {len(task_history)} versions retained")

    # ---------------------------------------------------------------- forget
    _step(5, "FORGET the archived preference and list what remains")
    clock.advance(minutes=5)
    preference = service.forget_memory(preference.id)
    _print_memory(preference)
    page = service.list_memories()
    for memory in page.items:
        _print_memory(memory)
    print(f"  -> {page.total} active memories left")
    try:
        service.get_memory(preference.id)
    except MemoryNotFoundError as exc:
        print(f"  -> get_memory() now raises {type(exc).__name__}")

    # ------------------------------------------------------ history survives
    _step(6, "HISTORY of the forgotten memory is still available")
    for version in service.get_history(preference.id):
        _print_version(version)

    # --------------------------------------------------------------- summary
    _step(7, "SUMMARY (computed from the database, not hard-coded)")
    for status in MemoryStatus:
        count = service.list_memories(MemoryQuery(status=status, limit=100)).total
        print(f"  {status.value:<9}: {count}")

    # --------------------------------------------------------- M2: scoring
    _step(8, "SCORE importance (deterministic, never stored)")
    for _ in range(3):
        clock.advance(minutes=1)
        service.record_access(task.id)
    clock.advance(hours=6)
    breakdown = service.score_memory(task.id)
    for factor in breakdown.factors:
        print(
            f"  {factor.name:<20} weight={factor.weight:.2f} value={factor.raw_value:.3f} "
            f"contribution={factor.contribution:.4f}"
        )
        print(f"      {factor.explanation}")
    print(f"  -> {breakdown.explanation}")
    print(f"  -> {service.get_effective_priority(task.id).explanation}")
    stored = service.get_memory(task.id)
    print(
        f"  -> stored importance is still {task.importance:.2f}, version {stored.version}, "
        f"access_count {stored.access_count}"
    )

    # -------------------------------------------------- M2: near duplicates
    _step(9, "DETECT a near-duplicate (lexical, no embeddings)")
    clock.advance(minutes=1)
    python_one = service.create_memory(
        content="User prefers Python",
        memory_type=MemoryType.PREFERENCE,
        source="user:revanth",
        importance=0.6,
    )
    clock.advance(minutes=1)
    service.create_memory(
        content="Python is the user's preferred language",
        memory_type=MemoryType.PREFERENCE,
        source="user:revanth",
        importance=0.4,
    )
    matches = service.find_duplicates(python_one.id)
    for match in matches:
        print(f"  {match.similarity:.4f}  {match.memory_id}")
        print(f"      {match.explanation}")
    print(f"  -> {len(matches)} near-duplicate(s), recorded as DUPLICATE_OF")

    # ------------------------------------------------------- M2: conflicts
    _step(10, "CREATE a conflicting memory, DETECT and RESOLVE it")
    clock.advance(minutes=1)
    service.create_memory(
        content="preferred backend language = Python",
        memory_type=MemoryType.FACT,
        importance=0.4,
        confidence=0.6,
    )
    clock.advance(minutes=1)
    conflicting_two = service.create_memory(
        content="preferred backend language = TypeScript",
        memory_type=MemoryType.FACT,
        importance=0.9,
        confidence=0.5,
    )
    conflicts = service.detect_conflicts(conflicting_two.id)
    for candidate in conflicts:
        print(f"  {candidate.target_memory_id}  attribute={candidate.attribute!r}")
        print(
            f"      {candidate.source_value!r} vs {candidate.target_value!r} "
            f"(value similarity {candidate.value_similarity:.2f})"
        )
        print(f"      {candidate.explanation}")
    print(f"  -> {len(conflicts)} unresolved conflict(s)")

    resolution = service.resolve_conflict(conflicting_two.id)
    print(f"  winner : {resolution.winner.id}  {resolution.winner.content}")
    print(f"  loser  : {resolution.loser.id}  {resolution.loser.content}")
    print(f"  decided by: {resolution.decided_by}")
    for step in resolution.steps:
        print(f"      {step.factor}: {step.left_value} vs {step.right_value} -> {step.winner_id}")
    print(f"  -> {resolution.explanation}")
    print(
        f"  -> loser archived: {resolution.loser_archived}; "
        f"unresolved conflicts left: {len(service.detect_conflicts(conflicting_two.id))}"
    )
    print("  loser history after resolution:")
    for version in service.get_history(resolution.loser.id):
        _print_version(version)

    # ------------------------------------------------------- M3: retrieval
    _step(11, "RETRIEVE memories for a query (lexical, explainable)")
    _run_retrieval(
        service,
        retriever,
        "python backend work",
        RetrievalMode.LEXICAL,
        semantic_weight=None,
    )

    # ------------------------------------------------- M3: hybrid retrieval
    _step(12, "RETRIEVE again in hybrid mode (deterministic local embeddings)")
    print(f"  semantic_available: {semantic_retriever.semantic_available}")
    print(f"  provider: {semantic_retriever.describe_provider()}")
    _run_retrieval(
        service,
        semantic_retriever,
        "python backend work",
        RetrievalMode.HYBRID,
        semantic_weight=0.5,
    )

    # ------------------------------------------------------ M4: context pool
    clock.advance(minutes=1)
    context_pool = service.create_memory(
        content="User is building AI applications with FastAPI.",
        memory_type=MemoryType.FACT,
        source="user:revanth",
        confidence=0.95,
        importance=0.85,
    )
    clock.advance(minutes=1)
    service.create_memory(
        content="User is building AI apps with FastAPI.",
        memory_type=MemoryType.OBSERVATION,
        source="agent:note",
        confidence=0.6,
        importance=0.4,
    )
    clock.advance(minutes=1)
    service.create_memory(
        content="User recently started the MemForge context compiler.",
        memory_type=MemoryType.EVENT,
        source="agent:planner",
        confidence=0.9,
        importance=0.7,
    )
    clock.advance(minutes=1)
    service.create_memory(
        content="The staging cluster runs on Kubernetes with three worker nodes.",
        memory_type=MemoryType.OBSERVATION,
        source="agent:note",
        confidence=0.5,
        importance=0.2,
    )
    service.find_duplicates(context_pool.id)

    # ---------------------------------------------------------- M4: compile
    _step(13, "COMPILE context for an agent task (M4, hard token budget)")
    compiler = ContextCompilerService(retriever)
    print(f"  policy: {compiler.describe_policy()}")
    _run_context(
        compiler,
        query="python backend language fastapi memforge",
        token_budget=160,
    )

    # ------------------------------------------------------ M4: tight budget
    _step(14, "COMPILE again with a tight budget (the limit is never exceeded)")
    _run_context(
        compiler,
        query="python backend language fastapi memforge",
        token_budget=40,
    )

    # ------------------------------------------------------ M4: summary
    # ------------------------------------------------- M5: memory-aware agent
    _step(15, "AGENT turn (M5, memory-aware, deterministic mock provider)")
    print("  USER REQUEST  ->  MEMORY GATE  ->  RECALL (M3)  ->  CONTEXT (M4)")
    print("              ->  PROVIDER  ->  ANSWER  ->  CANDIDATES  ->  STORE (M1)")
    observability = ObservabilityService(
        SqliteTraceStore(db_path),
        clock=clock,
        timer=DeterministicTimer(),
        id_factory=SequentialIdFactory(prefix="demo-trace"),
    )
    traced_operations = build_traced_operations(service, retriever, compiler, observability)
    traced_agent = AgentTracer(
        AgentService(traced_operations, trace_provider(provider, observability)),
        observability,
    )
    trace_id = _run_agent(
        traced_agent,
        store_candidates=store_candidates,
    )

    # ------------------------------------------------------ M6: trace record
    _step(16, "TRACE of that turn (M6, what actually happened)")
    _print_trace(observability, trace_id)

    # ------------------------------------------------------ M6: evaluation
    _step(17, "EVALUATION of the shipped dataset (M6, measured)")
    summary = EvaluationService(settings).evaluate(load_dataset())
    for line in render(summary).splitlines():
        print(f"  {line}" if line else "")

    # ------------------------------------------------------ M6: metrics
    _step(18, "METRICS derived from the recorded events (M6)")
    _print_metrics(observability)

    # ---------------------------------------------------------- final summary
    _step(19, "FINAL SUMMARY (nothing was destroyed)")
    for status in MemoryStatus:
        count = service.list_memories(MemoryQuery(status=status, limit=100)).total
        print(f"  {status.value:<9}: {count}")
    for relationship in service.get_relationships(conflicting_two.id):
        print(f"  relationship: {relationship.relationship.value} -> {relationship.target_id}")
    print(f"  traces:       {len(observability.list_traces(limit=100))}")
    _rule()
    print("Demo finished successfully.")
    return 0


def _print_trace(observability: ObservabilityService, trace_id: str) -> None:
    """Print one recorded trace as an ordered, inspectable event list."""
    summary = observability.summarize_trace(trace_id)
    if summary is None:  # pragma: no cover - the turn always records one
        return
    print(f"  TRACE {summary.trace_id}  ({summary.kind.value} / {summary.operation})")
    print(f"  events: {summary.event_count}")
    for event in observability.get_events(trace_id):
        parent = f" <- {event.parent_id.rsplit('-', 1)[-1]}" if event.parent_id else ""
        timing = f"  {event.duration_ms:.3f} ms" if event.duration_ms is not None else ""
        print(
            f"    {event.sequence:>2}  {event.kind.value:<24} {event.status.value:<8}{timing}{parent}"
        )
    print("  operations:")
    for operation, status in sorted(summary.operations.items()):
        print(f"    {operation:<24} {status}")
    print(
        "  measured (deterministic clock): "
        + ", ".join(f"{name}={value:.3f}ms" for name, value in sorted(summary.durations_ms.items()))
    )
    print("  payloads hold counts, ids, modes and sizes -- never text or secrets.")


def _print_metrics(observability: ObservabilityService) -> None:
    """Print the counters derived from the recorded events."""
    metrics = observability.get_metrics()
    for title, counters in (
        ("traces", metrics.traces),
        ("operations", metrics.operations),
        ("statuses", metrics.statuses),
        ("memory", metrics.memory),
        ("context", metrics.context),
        ("provider", metrics.provider),
    ):
        rendered = ", ".join(f"{name}={value}" for name, value in counters.items())
        print(f"  {title:<11} {rendered or 'none'}")
    print(f"  usage      reported_by_provider={metrics.usage_reported} (nothing is inferred)")


def _run_agent(tracer: AgentTracer, *, store_candidates: bool) -> str:
    """Run one traced memory-aware agent turn and print every step of it."""
    provider = tracer.agent.provider
    request = AgentRequest(
        request=(
            "What backend language should I use for the MemForge service?\n"
            "remember: [FACT] MemForge uses Python and FastAPI for its backend."
        ),
        token_budget=200,
        top_k=5,
        extract_candidates=True,
        store_candidates=store_candidates,
    )

    print()
    print(f"  provider: {provider.describe()}")
    traced = tracer.respond(request)
    result = traced.result

    print(f"  request:   {_short(result.request, 60)!r}")
    print(f"  memory:    {result.gate.use.value} ({result.gate.reason})")
    print(
        f"  context:   {result.context_estimated_tokens} estimated tokens of a "
        f"{result.context_token_budget} budget"
    )
    for line in result.context_text.splitlines():
        print(f"      {line}")
    print()
    print("  answer:")
    for line in result.answer.splitlines():
        print(f"      {line}")
    print()
    print("  candidate memories:")
    for candidate in result.candidate_memories:
        print(
            f"      [{candidate.memory_type.value}] confidence={candidate.confidence:.2f} "
            f"importance={candidate.importance:.2f} {candidate.content}"
        )
        print(f"          reason: {candidate.reason}")
    print(
        f"  stored: {len(result.stored_memory_ids)} "
        f"({'permitted' if store_candidates else 'not permitted -- nothing was written'})"
    )
    print()
    print("  observability:")
    for record in result.records:
        usage = "not reported" if record.usage is None else "reported by provider"
        print(
            f"      {record.operation:<20} success={record.success} "
            f"provider={record.provider} usage={usage}"
        )
    if not store_candidates:
        print("  (run with --store-candidates to let the agent persist them)")
    print(f"  trace:    {traced.trace.trace_id}")
    return traced.trace.trace_id


def _run_context(compiler: ContextCompilerService, *, query: str, token_budget: int) -> None:
    """Compile one context and print the budgeted result and its explanation."""
    print()
    print("  MEMORY POOL  ->  RETRIEVAL  ->  CONTEXT COMPILER  ->  BUDGETED CONTEXT")
    result = compiler.compile(ContextRequest(query=query, token_budget=token_budget))

    print()
    print("  Context Compiler")
    print("  ----------------")
    print(f"  Query:              {result.query!r}")
    print(f"  Budget:             {result.token_budget}")
    print(f"  Candidates:         {result.candidates_considered}")
    print(f"  Selected:           {result.selected_count}")
    print(f"  Suppressed:         {result.suppressed_count}")
    print(f"  Remaining:          {result.remaining_tokens}")

    print()
    print("  Selected:")
    for block in result.blocks:
        print(
            f"    [{block.section.heading}] utility={block.utility_score:.4f} "
            f"tokens={block.estimated_tokens} {block.memory_id}"
        )
        print(f"        {_short(block.rendered_content, 60)}")
        print(
            f"        provenance: type={block.memory_type.value} source={block.source} "
            f"confidence={block.confidence:.2f} importance={block.importance:.2f} "
            f"retrieval={block.retrieval_score:.4f} selection={block.utility_score:.4f}"
        )

    print()
    print("  Suppressed:")
    for entry in result.suppressed_candidates:
        print(
            f"    {entry.reason.value:22} {entry.memory_id} "
            f"utility={entry.utility_score:.4f} cost={entry.estimated_tokens}"
        )
        print(f"        {entry.detail}")

    print()
    print("  Final context:")
    for line in result.context_text.splitlines() or ["    (empty)"]:
        print(f"    {line}")

    print()
    print(f"  Final context tokens: {result.estimated_tokens} / {result.token_budget}")
    print(f"  Within budget:        {result.within_budget}")
    print(f"  Estimator:            {result.explanation.estimator}")
    print("  Token counts are MemForge estimates, not provider tokenisations.")


def _run_retrieval(
    service: MemoryService,
    retriever: MemoryRetrievalService,
    query: str,
    mode: RetrievalMode,
    *,
    semantic_weight: float | None,
) -> None:
    """Run one retrieval query and print the explained ranking."""
    response = retriever.search(
        RetrievalQuery(text=query, top_k=5, mode=mode, semantic_weight=semantic_weight)
    )
    print(f"  query: {query!r}  mode={response.retrieval_mode.value}  top_k={response.top_k}")
    print(
        f"  considered={response.candidates_considered} returned={response.candidates_returned} "
        f"suppressed={response.candidates_suppressed} semantic={response.semantic_available}"
    )
    if response.degraded_reason:
        print(f"  degraded: {response.degraded_reason}")
    for rank, result in enumerate(response.results, start=1):
        semantic = "n/a" if result.semantic_score is None else f"{result.semantic_score:.4f}"
        print(
            f"  #{rank} {result.memory.id}  final={result.final_score:.4f} "
            f"relevance={result.relevance:.4f} lexical={result.lexical_score:.4f} "
            f"semantic={semantic}"
        )
        print(f"      {result.explanation}")
    for candidate in response.suppressed:
        print(
            f"  suppressed {candidate.memory_id} score={candidate.final_score:.4f} "
            f":: {candidate.reason}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="memforge-demo",
        description=(
            "Deterministic walkthrough of the MemForge memory core, intelligence, "
            "retrieval, context compilation and the memory-aware agent."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite file to use (default: {DEFAULT_DB_PATH}). It is recreated from scratch.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("mock", "custom"),
        default="mock",
        help=(
            "Language-model provider for the M5 step. 'mock' (default) is offline "
            "and deterministic; 'custom' is opt-in and requires MEMFORGE_LLM_BASE_URL, "
            "MEMFORGE_LLM_MODEL and MEMFORGE_LLM_API_KEY plus network access."
        ),
    )
    parser.add_argument(
        "--store-candidates",
        action="store_true",
        help="Let the M5 step persist the candidate memories it extracted (opt-in).",
    )
    args = parser.parse_args(argv)

    try:
        return run_demo(
            args.db, llm_provider=args.llm_provider, store_candidates=args.store_candidates
        )
    except MemforgeError as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        # Selecting a provider without configuring it is an explicit failure --
        # never a silent fallback that would pretend a model answered.
        print(f"demo failed: invalid configuration: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
