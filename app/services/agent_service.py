"""The memory-aware agent service (M5).

One bounded pipeline, no autonomy:

    AgentRequest
        -> memory gate            (pure heuristic, ``app.domain.agent``)
        -> recall                 (M3, via MemoryOperationsService)
        -> compile context        (M4, hard token budget)
        -> provider call          (LLMProvider; mock by default)
        -> answer
        -> candidate extraction   (provider, structured output)
        -> store                  (M1, only when explicitly permitted)

Properties this service guarantees
----------------------------------
* **Deterministic by default.** With the mock provider the whole turn is a pure
  function of (corpus, clock, request). Nothing is sampled, nothing is random.
* **No silent fabrication.** A provider failure is raised as a typed
  :class:`~app.domain.errors.LLMProviderError`; the caller gets an explicit
  failure, never a plausible-looking answer produced by MemForge itself.
* **Writes are opt-in.** ``store_candidates`` defaults to ``False``. Model output
  is treated as untrusted input: a candidate can be proposed, reported and
  rejected, but it is never written without explicit permission.
* **Nothing is measured that was not reported.** Usage is ``None`` unless the
  provider sent it; the context token figure is always M4's own *estimate* and is
  labelled as one.
* **No secrets in evidence.** Every record carries provider name, model and
  timings -- never a credential.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from app.domain.agent import (
    CANDIDATE_MEMORY_SCHEMA,
    CONTEXT_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    AgentRequest,
    AgentResult,
    CandidateMemory,
    ContextBlockSummary,
    MemoryGate,
    MemoryUse,
    OperationRecord,
    build_messages,
    decide_memory_use,
)
from app.domain.context import ContextResult
from app.domain.errors import AgentValidationError, LLMProviderError
from app.domain.llm import LLMMessage, LLMProvider, LLMRequest, LLMUsage
from app.services.memory_operations import MemoryOperationsService, RecallPolicy

#: Schema name sent with the candidate-extraction request.
CANDIDATE_SCHEMA_NAME = "candidate_memories"


class AgentService:
    """Turns an agent request into an answer, using memory where it helps."""

    def __init__(
        self,
        operations: MemoryOperationsService,
        provider: LLMProvider,
        *,
        max_token_budget: int | None = None,
    ) -> None:
        """Args:
        operations: The M1-M4 facade.
        provider: The language-model provider.
        max_token_budget: Hard ceiling for the context budget. It is validated
            **before** the memory gate, so an impossible budget is refused even
            when the turn would not have compiled a context at all.
        """
        self._operations = operations
        self._provider = provider
        self._max_token_budget = max_token_budget

    @property
    def provider(self) -> LLMProvider:
        """The provider in use."""
        return self._provider

    @property
    def operations(self) -> MemoryOperationsService:
        """The composed memory-operations service."""
        return self._operations

    def respond(self, request: AgentRequest) -> AgentResult:
        """Run one agent turn."""
        self._validate(request)
        if not self._provider.is_configured:
            raise LLMProviderError(
                f"the '{self._provider.name}' provider is not configured",
                provider=self._provider.name,
            )

        records: list[OperationRecord] = []
        gate = decide_memory_use(request.request, use_memory=request.use_memory)
        context_result: ContextResult | None = None

        if gate.should_recall:
            recalled = self._operations.recall(
                request.request,
                top_k=request.top_k,
                mode=request.retrieval_mode,
                semantic_weight=request.semantic_weight,
                policy=RecallPolicy(max_results=request.top_k),
            )
            if not recalled:
                # Nothing worth recalling: compiling would only re-run retrieval
                # to produce the same empty result, so it is skipped and reported.
                gate = MemoryGate(
                    MemoryUse.NO_MATCHES,
                    "recall returned no memory with a non-zero relevance",
                )
            else:
                context_result = self._operations.compile_context(
                    request.request,
                    token_budget=request.token_budget,
                    top_k=request.top_k,
                    retrieval_mode=request.retrieval_mode,
                    semantic_weight=request.semantic_weight,
                )
                records.append(
                    OperationRecord(
                        operation="compile_context",
                        success=True,
                        latency_ms=0.0,
                        provider=self._provider.name,
                        model=self._provider.model,
                        context_estimated_tokens=context_result.estimated_tokens,
                        memories_recalled=len(recalled),
                    )
                )

        context_text = context_result.context_text if context_result is not None else ""
        messages = build_messages(
            system=CONTEXT_SYSTEM_PROMPT, user=request.request, context=context_text
        )
        context_only = context_result
        estimated = context_result.estimated_tokens if context_result is not None else 0

        if request.context_only:
            return AgentResult(
                request=request.request,
                answer="",
                gate=gate,
                context_text=context_text,
                context_estimated_tokens=estimated,
                context_token_budget=request.token_budget,
                context_blocks=_blocks(context_only),
                provider=self._provider.name,
                model=self._provider.model,
                records=tuple(records),
                messages=messages,
            )

        answer, usage = self._call_model(request, messages, operation="respond", records=records)

        candidates: tuple[CandidateMemory, ...] = ()
        stored: tuple[str, ...] = ()
        if request.extract_candidates:
            candidates, stored = self._candidates(request, records=records)

        return AgentResult(
            request=request.request,
            answer=answer,
            gate=gate,
            context_text=context_text,
            context_estimated_tokens=estimated,
            context_token_budget=request.token_budget,
            context_blocks=_blocks(context_only),
            candidate_memories=candidates,
            stored_memory_ids=stored,
            provider=self._provider.name,
            model=self._provider.model,
            usage=usage,
            records=tuple(records),
            messages=messages,
        )

    # -- internals: validation --------------------------------------------

    def _validate(self, request: AgentRequest) -> None:
        """Refuse a turn MemForge cannot serve, before any work happens."""
        ceiling = self._max_token_budget
        if ceiling is not None and request.token_budget > ceiling:
            raise AgentValidationError(
                f"token_budget {request.token_budget} exceeds the configured maximum of "
                f"{ceiling} estimated tokens"
            )

    # -- internals: provider ---------------------------------------------

    def _call_model(
        self,
        request: AgentRequest,
        messages: tuple[LLMMessage, ...],
        *,
        operation: str,
        records: list[OperationRecord],
    ) -> tuple[str, LLMUsage | None]:
        """Call the provider, recording success or a normalised failure."""
        started = time.perf_counter()
        try:
            response = self._provider.generate(
                LLMRequest(
                    messages=messages,
                    model=self._provider.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    operation=operation,
                )
            )
        except LLMProviderError as error:
            records.append(
                OperationRecord(
                    operation=operation,
                    success=False,
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    provider=self._provider.name,
                    model=self._provider.model,
                    error_kind=error.kind,
                )
            )
            raise
        records.append(
            OperationRecord(
                operation=operation,
                success=True,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                provider=self._provider.name,
                model=self._provider.model,
                # ``None`` unless the provider actually reported usage.
                usage=response.usage,
            )
        )
        return response.text, response.usage

    # -- internals: candidates -------------------------------------------

    def _candidates(
        self,
        request: AgentRequest,
        *,
        records: list[OperationRecord],
    ) -> tuple[tuple[CandidateMemory, ...], tuple[str, ...]]:
        """Ask the provider for candidate memories, then optionally store them.

        Extraction failure is never fatal: a provider that cannot produce
        structured output still produced a perfectly good answer, so the failure
        is recorded and the turn continues with no candidates.
        """
        messages = build_messages(system=EXTRACTION_SYSTEM_PROMPT, user=request.request)
        started = time.perf_counter()
        try:
            response = self._provider.generate_structured(
                LLMRequest(
                    messages=messages,
                    model=self._provider.model,
                    temperature=request.temperature,
                    response_schema=CANDIDATE_MEMORY_SCHEMA,
                    schema_name=CANDIDATE_SCHEMA_NAME,
                    operation="extract_candidates",
                )
            )
        except LLMProviderError as error:
            records.append(
                OperationRecord(
                    operation="extract_candidates",
                    success=False,
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    provider=self._provider.name,
                    model=self._provider.model,
                    error_kind=error.kind,
                )
            )
            return (), ()

        candidates = _parse_candidates(response.structured)
        stored = self._store(request, candidates) if request.will_store else ()
        records.append(
            OperationRecord(
                operation="extract_candidates",
                success=True,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                provider=self._provider.name,
                model=self._provider.model,
                usage=response.usage,
                memories_stored=len(stored),
            )
        )
        return tuple(candidates), stored

    def _store(
        self, request: AgentRequest, candidates: Sequence[CandidateMemory]
    ) -> tuple[str, ...]:
        """Persist permitted candidates through M1, skipping invalid ones."""
        stored: list[str] = []
        for candidate in candidates:
            if not candidate.is_valid:
                continue
            memory = self._operations.store(
                candidate.content,
                candidate.memory_type,
                source=request.source,
                confidence=candidate.confidence,
                importance=candidate.importance,
            )
            stored.append(memory.id)
        return tuple(stored)


def _parse_candidates(payload: Mapping[str, Any] | None) -> list[CandidateMemory]:
    """Normalise a provider payload into candidates, dropping malformed items."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("candidates")
    if not isinstance(raw, list):
        return []
    candidates: list[CandidateMemory] = []
    for item in raw:
        if isinstance(item, dict):
            candidates.append(CandidateMemory.from_payload(item))
    return candidates


def _blocks(context_result: ContextResult | None) -> tuple[ContextBlockSummary, ...]:
    """Condense M4 blocks into evidence-safe provenance summaries."""
    if context_result is None:
        return ()
    return tuple(
        ContextBlockSummary(
            memory_id=block.memory_id,
            memory_type=block.memory_type,
            section=block.section,
            order=block.order,
            utility_score=block.utility_score,
            retrieval_score=block.retrieval_score,
            estimated_tokens=block.estimated_tokens,
        )
        for block in context_result.blocks
    )


__all__ = ["CANDIDATE_SCHEMA_NAME", "AgentService"]
