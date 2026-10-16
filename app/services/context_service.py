"""Context compilation service (M4).

This is the only component that knows how *both* halves fit together::

    API
     |
     v
    ContextCompilerService      <-- this module
     |
     v
    MemoryRetrievalService      <-- M3, unchanged: candidates, ranking, relevance
     |
     v
    MemoryRepository / EmbeddingStore

The compiler **composes** retrieval, it never re-implements it: candidate
generation, lexical/semantic relevance, ranking, conflict and duplicate metadata
all come from M3. Everything M4 adds on top is budgeted *selection*, deterministic
*compression*, prompt *ordering*, provenance and explanation -- all of it pure
functions from ``app.domain.context``.

Candidates the compiler considers
---------------------------------
M3 applies its own duplicate suppression, so some memories are already dropped
before the compiler sees them. They are still *candidates* as far as the context
is concerned ("why is this memory missing?" is a context question), so the
compiler re-scores them through M3's public ``explain()`` API and runs them
through the same selection pipeline. The result is one explained decision for
every memory retrieval considered -- which is what makes
``suppressed_candidates`` complete rather than "whatever happened to reach M4".

Invariants this service guarantees
----------------------------------
* the compiled context never exceeds the requested budget in estimated tokens;
* every candidate considered appears exactly once in ``selection_trace``, with
  the reason it was admitted or rejected;
* every selected block traces back to a stored memory, unmodified content
  included -- compression is recorded next to it, never in place of it;
* the same corpus, clock and request always produce the same result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.domain.context import (
    CONTEXT_NOTES,
    ContextBlock,
    ContextExplanation,
    ContextPolicy,
    ContextRequest,
    ContextResult,
    ContextSelection,
    SelectionReason,
    apply_admission_filters,
    order_blocks,
    render_context,
    score_candidate,
    select_within_budget,
    suppress_redundant,
)
from app.domain.errors import ContextValidationError, MemoryNotFoundError
from app.domain.retrieval import RetrievalQuery, RetrievalResponse, RetrievalResult
from app.domain.tokens import TokenEstimatorConfig, estimate_tokens
from app.services.retrieval_service import MemoryRetrievalService

#: Ordering policy name reported in every explanation.
ORDERING_POLICY = "section-then-utility/v1"


class ContextCompilerService:
    """Compiles retrieved memories into a budgeted, ordered agent context."""

    def __init__(
        self,
        retrieval: MemoryRetrievalService,
        *,
        policy: ContextPolicy | None = None,
        estimator: TokenEstimatorConfig | None = None,
        max_token_budget: int | None = None,
    ) -> None:
        """Build a compiler.

        Args:
            retrieval: The M3 retrieval service (composed, never re-implemented).
            policy: Utility, ordering and compression policy.
            estimator: Token estimator configuration.
            max_token_budget: Hard ceiling for a single request. ``None`` means
                no ceiling beyond the request's own budget.
        """
        self._retrieval = retrieval
        self._policy = policy or ContextPolicy()
        self._estimator = estimator or TokenEstimatorConfig()
        self._max_token_budget = max_token_budget

    # -- configuration ---------------------------------------------------

    @property
    def policy(self) -> ContextPolicy:
        """The context policy in use."""
        return self._policy

    @property
    def estimator_config(self) -> TokenEstimatorConfig:
        """The token estimator configuration in use."""
        return self._estimator

    @property
    def retrieval_service(self) -> MemoryRetrievalService:
        """The composed M3 retrieval service."""
        return self._retrieval

    def describe_policy(self) -> str:
        """One-line, deterministic description of the active policy."""
        sections = " > ".join(section.value for section in self._policy.section_order)
        return (
            f"{self._estimator.estimator_id}; {self._policy.formula()}; "
            f"ordering={ORDERING_POLICY}; sections={sections}"
        )

    # -- compilation -----------------------------------------------------

    def compile(self, request: ContextRequest) -> ContextResult:
        """Compile a budgeted, ordered, fully explained context."""
        self._validate_budget(request)

        retrieval_query = request.to_retrieval_query()
        response = self._retrieval.search(retrieval_query)

        candidates = [
            score_candidate(
                result,
                policy=self._policy,
                estimator=self._estimator,
                query_terms=retrieval_query.terms,
                enabled=request.compress,
                max_block_tokens=request.max_block_tokens,
            )
            for result in self._scored_candidates(response, retrieval_query)
        ]

        eligible, rejected = apply_admission_filters(
            candidates,
            request=request,
            policy=self._policy,
            token_budget=request.token_budget,
        )
        admissible, redundant = suppress_redundant(eligible, token_budget=request.token_budget)
        selected, admitted = select_within_budget(admissible, token_budget=request.token_budget)

        blocks = order_blocks(selected, policy=self._policy)
        blocks, context_text = self._enforce_rendered_budget(
            blocks, token_budget=request.token_budget
        )
        estimated_tokens = self._estimate(context_text)
        trace = self._finalise_trace(
            [*rejected, *redundant, *admitted],
            blocks=blocks,
            remaining_tokens=max(0, request.token_budget - estimated_tokens),
        )

        suppressed = tuple(entry for entry in trace if not entry.selected)
        return ContextResult(
            query=request.query,
            token_budget=request.token_budget,
            estimated_tokens=estimated_tokens,
            remaining_tokens=request.token_budget - estimated_tokens,
            candidates_considered=len(candidates),
            selected_count=len(blocks),
            suppressed_count=len(suppressed),
            requested_retrieval_mode=request.retrieval_mode,
            retrieval_mode=response.retrieval_mode,
            semantic_available=response.semantic_available,
            degraded_reason=response.degraded_reason,
            context_text=context_text,
            blocks=blocks,
            selection_trace=trace,
            suppressed_candidates=suppressed,
            explanation=self._explain(blocks, trace),
        )

    # -- internals: budget ------------------------------------------------

    def _scored_candidates(
        self,
        response: RetrievalResponse,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Every memory retrieval considered, including what it suppressed.

        M3 drops near-duplicates from its result set, so those memories never
        appear in ``results``. They are re-scored here through M3's own
        ``explain()`` entry point -- not by re-implementing scoring -- so the
        compiler can report them with the same explanation quality as the rest.
        A memory that vanished between the two calls is skipped rather than
        failing the whole compilation.
        """
        scored: list[RetrievalResult] = list(response.results)
        for suppressed in response.suppressed:
            try:
                scored.append(self._retrieval.explain(suppressed.memory_id, query))
            except MemoryNotFoundError:
                continue
        return scored

    def _validate_budget(self, request: ContextRequest) -> None:
        """Reject a budget this service is not configured to serve."""
        ceiling = self._max_token_budget
        if ceiling is not None and request.token_budget > ceiling:
            raise ContextValidationError(
                f"token_budget {request.token_budget} exceeds the configured maximum of "
                f"{ceiling} estimated tokens"
            )

    def _estimate(self, text: str) -> int:
        """Estimated token cost of a piece of rendered text."""
        return estimate_tokens(text, config=self._estimator).estimated_tokens

    def _enforce_rendered_budget(
        self,
        blocks: tuple[ContextBlock, ...],
        *,
        token_budget: int,
    ) -> tuple[tuple[ContextBlock, ...], str]:
        """Drop the lowest-priority block until the *rendered* text fits.

        ``select_within_budget`` already pays for every block's line, so this
        loop normally does not run. It exists because the rendered context also
        carries section headings, whose cost is only known once the surviving
        blocks are known. Dropping from the end of the prompt order removes the
        least important section first, and the decision is reported through
        :meth:`_finalise_trace`.
        """
        context_text = render_context(blocks, policy=self._policy)
        while blocks and self._estimate(context_text) > token_budget:
            blocks = blocks[:-1]
            context_text = render_context(blocks, policy=self._policy)
        return blocks, context_text

    # -- internals: trace -------------------------------------------------

    def _finalise_trace(
        self,
        trace: Sequence[ContextSelection],
        *,
        blocks: Sequence[ContextBlock],
        remaining_tokens: int,
    ) -> tuple[ContextSelection, ...]:
        """Reconcile the trace with the blocks that survived rendering.

        A block dropped by the last-resort render trim keeps exactly one trace
        entry -- rewritten to ``insufficient_budget`` -- so every candidate
        still appears exactly once. The trace is then ordered by descending
        utility and ascending memory id, which makes it stable to diff.
        """
        surviving = {block.memory_id for block in blocks}
        finalised: list[ContextSelection] = []
        for entry in trace:
            if entry.selected and entry.memory_id not in surviving:
                finalised.append(
                    replace(
                        entry,
                        reason=SelectionReason.INSUFFICIENT_BUDGET,
                        detail=(
                            "admitted by the budget pass, then dropped: the rendered context "
                            "(section headings included) still exceeded the token budget"
                        ),
                        remaining_tokens=remaining_tokens,
                        selected=False,
                    )
                )
            else:
                finalised.append(entry)
        return tuple(sorted(finalised, key=ContextSelection.selection_key))

    # -- internals: explanation -------------------------------------------

    def _explain(
        self,
        blocks: Sequence[ContextBlock],
        trace: Sequence[ContextSelection],
    ) -> ContextExplanation:
        """Summarise the compilation: counts, ordering, compression, caveats."""
        counts: dict[str, int] = {reason.value: 0 for reason in SelectionReason}
        for entry in trace:
            counts[entry.reason.value] += 1
        compressed = [block for block in blocks if block.compression.changed]
        return ContextExplanation(
            estimator=self._estimator.estimator_id,
            estimator_formula=self._estimator.formula,
            utility_formula=self._policy.formula(),
            utility_weights=_sorted_weights(self._policy.utility_weights),
            ordering_policy=ORDERING_POLICY,
            section_order=self._policy.section_order,
            selection_counts=dict(sorted(counts.items())),
            compressed_blocks=len(compressed),
            tokens_saved=sum(block.compression.tokens_saved for block in compressed),
            notes=CONTEXT_NOTES,
        )


def _sorted_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """Stable, alphabetical weight mapping for the explanation payload."""
    return {name: weights[name] for name in sorted(weights)}


__all__ = ["ORDERING_POLICY", "ContextCompilerService"]
