"""Agent-facing memory operations (M5).

One thin facade over the M1-M4 services so that an agent, the REST API and the
MCP server all speak the same six verbs:

=================  ==========================================================
verb               delegates to
=================  ==========================================================
``store``          :class:`MemoryService` (M1)
``search``         :meth:`MemoryRetrievalService.search` (M3)
``recall``         :meth:`MemoryRetrievalService.search` + a relevance gate
``update``         :meth:`MemoryService.update_memory` (M1)
``forget``         :meth:`MemoryService.forget_memory` (M1)
``compile_context``:meth:`ContextCompilerService.compile` (M4)
=================  ==========================================================

There is deliberately **no business logic here**. Every validation rule, every
error type and every explanation already exists one layer down; re-implementing
any of it would give the agent a second, divergent memory system. This module
only chooses *which* service answers and normalises the arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.context import ContextRequest, ContextResult
from app.domain.models import Memory, MemoryStatus, MemoryType
from app.domain.retrieval import RetrievalMode, RetrievalQuery, RetrievalResponse, RetrievalResult
from app.domain.updates import UNSET, MemoryUpdate, Unset
from app.services.context_service import ContextCompilerService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService


@dataclass(frozen=True, slots=True)
class RecallPolicy:
    """The agent-facing policy for "is this result good enough to recall?".

    M3 returns a ranked list; an agent usually wants a *short list of things
    worth putting in front of a model*. This policy applies two documented
    filters and reports how many candidates each one removed:

    * ``min_score`` -- the M3 ``final_score`` floor;
    * ``require_match`` -- drop candidates whose lexical/semantic relevance is
      exactly zero, i.e. memories that matched no query term at all.
    """

    min_score: float = 0.0
    require_match: bool = True
    max_results: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if self.max_results < 1:
            raise ValueError("max_results must be >= 1")


class MemoryOperationsService:
    """The six memory verbs, composed from the M1-M4 services."""

    def __init__(
        self,
        memories: MemoryService,
        retrieval: MemoryRetrievalService,
        compiler: ContextCompilerService,
    ) -> None:
        self._memories = memories
        self._retrieval = retrieval
        self._compiler = compiler

    # -- introspection ---------------------------------------------------

    @property
    def memory_service(self) -> MemoryService:
        """The composed M1 service."""
        return self._memories

    @property
    def retrieval_service(self) -> MemoryRetrievalService:
        """The composed M3 service."""
        return self._retrieval

    @property
    def compiler_service(self) -> ContextCompilerService:
        """The composed M4 compiler."""
        return self._compiler

    # -- verbs -----------------------------------------------------------

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        *,
        source: str | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
    ) -> Memory:
        """Create a memory (M1)."""
        return self._memories.create_memory(
            content=content,
            memory_type=memory_type,
            source=source,
            confidence=confidence,
            importance=importance,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.LEXICAL,
        semantic_weight: float | None = None,
    ) -> RetrievalResponse:
        """Search memories (M3), unfiltered."""
        return self._retrieval.search(
            RetrievalQuery(text=query, top_k=top_k, mode=mode, semantic_weight=semantic_weight)
        )

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.LEXICAL,
        semantic_weight: float | None = None,
        policy: RecallPolicy | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Recall the memories worth putting in front of a model.

        Runs the M3 search and then applies :class:`RecallPolicy`. M3 already
        suppresses duplicates, so the result contains no two copies of the same
        fact by construction.
        """
        resolved = policy or RecallPolicy(max_results=top_k)
        response = self.search(query, top_k=top_k, mode=mode, semantic_weight=semantic_weight)
        kept: list[RetrievalResult] = []
        for result in response.results:
            if result.final_score < resolved.min_score:
                continue
            if resolved.require_match and result.relevance <= 0.0:
                continue
            kept.append(result)
            if len(kept) >= resolved.max_results:
                break
        return tuple(kept)

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        memory_type: MemoryType | None = None,
        source: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        status: MemoryStatus | None = None,
        clear_source: bool = False,
    ) -> Memory:
        """Apply a partial update (M1).

        ``None`` means "leave this field alone", which is what
        :class:`~app.domain.updates.MemoryUpdate`'s ``UNSET`` sentinel expresses;
        only ``clear_source=True`` sends an explicit null. Translating that here
        is the whole point of the facade -- M1's rules stay in M1.
        """
        patch = MemoryUpdate(
            content=_or_unset(content),
            memory_type=_or_unset(memory_type),
            source=None if clear_source else _or_unset(source),
            confidence=_or_unset(confidence),
            importance=_or_unset(importance),
            status=_or_unset(status),
        )
        return self._memories.update_memory(memory_id, patch)

    def forget(self, memory_id: str) -> Memory:
        """Soft-delete a memory (M1). Nothing is destroyed."""
        return self._memories.forget_memory(memory_id)

    def compile_context(
        self,
        query: str,
        *,
        token_budget: int,
        top_k: int = 10,
        retrieval_mode: RetrievalMode = RetrievalMode.LEXICAL,
        semantic_weight: float | None = None,
        preferred_memory_types: tuple[MemoryType, ...] | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
        exclude_ids: tuple[str, ...] = (),
        include_ids: tuple[str, ...] = (),
    ) -> ContextResult:
        """Compile a budgeted agent context (M4)."""
        return self._compiler.compile(
            ContextRequest(
                query=query,
                token_budget=token_budget,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                semantic_weight=semantic_weight,
                preferred_memory_types=preferred_memory_types,
                min_confidence=min_confidence,
                min_importance=min_importance,
                include_ids=include_ids,
                exclude_ids=exclude_ids,
            )
        )

    def explain(self, memory_id: str, query: str) -> RetrievalResult:
        """Explain how one memory ranks for a query (M3)."""
        return self._retrieval.explain(memory_id, RetrievalQuery(text=query, top_k=1))


def _or_unset[T: (MemoryType, MemoryStatus, str, float)](value: T | None) -> T | Unset:
    """Translate "not supplied" into the M1 sentinel, never into an explicit null."""
    return UNSET if value is None else value


__all__ = ["MemoryOperationsService", "RecallPolicy"]
