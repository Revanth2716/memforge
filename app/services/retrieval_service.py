"""Memory-aware retrieval service.

This is the only place that combines query understanding, the inverted index,
the optional embedding layer and the M2 intelligence signals into a ranked,
explainable answer.

Pipeline
--------
1. **Candidate generation** — ``lexical`` mode uses the in-memory inverted index
   built from the active corpus (only matching memories are scored);
   ``semantic``/``hybrid`` modes must score the whole active corpus because
   embeddings can match without shared tokens.
2. **Filtering** — ``MemoryRepository.search_candidates`` applies the typed
   filters (memory type, status, confidence/importance bounds, creation window)
   in SQL, so ranking only ever sees admissible rows.
3. **Scoring** — relevance (lexical / semantic / hybrid) plus the M2 signals
   (effective priority, confidence, freshness) and the documented type
   preference, combined with the conflict and redundancy penalties defined in
   ``app.domain.retrieval``.
4. **Selection** — deterministic order (score descending, id ascending) with
   duplicate suppression: a candidate that duplicates an already selected memory
   is reported as suppressed instead of being deleted.
5. **Explanation** — every result carries the exact factor values, weights and
   contributions behind its score, plus its conflicts and duplicate links.

The in-memory inverted index is rebuilt whenever the active corpus changes. The
check that detects that (``active_corpus_signature``) is a cheap indexed
aggregate, so a query after a write costs one extra load and a query with no
writes costs none. The trade-off — an incremental index would avoid the reload
but would have to hook every write path — is documented in the README.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.domain.attributes import extract_attribute_values
from app.domain.embeddings import EmbeddingProvider, cosine_similarity
from app.domain.errors import MemoryNotFoundError
from app.domain.intelligence import (
    IntelligenceConfig,
    compute_effective_priority,
    compute_importance,
)
from app.domain.models import Memory
from app.domain.relationships import MemoryRelationship, RelationshipType
from app.domain.retrieval import (
    ConflictInfo,
    RankingConfig,
    RankingFactor,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    SuppressedCandidate,
    conflict_penalty_from,
    final_score,
    freshness,
    hybrid_relevance,
    lexical_relevance,
    ranking_explanation,
    semantic_relevance,
    sort_key,
    type_preference,
)
from app.domain.similarity import normalized_tokens
from app.domain.time import Clock, SystemClock
from app.storage.base import CandidateQuery, MemoryRepository
from app.storage.sqlite.embedding_store import EmbeddingStore, SqliteEmbeddingStore


@dataclass(slots=True)
class _InvertedIndex:
    """Token postings plus per-document term frequencies."""

    corpus_size: int = 0
    document_frequencies: dict[str, int] = field(default_factory=dict)
    term_frequencies: dict[str, dict[str, int]] = field(default_factory=dict)
    tokens_by_memory: dict[str, frozenset[str]] = field(default_factory=dict)

    def documents_for(self, terms: tuple[str, ...]) -> set[str]:
        """Union of the postings of every query term."""
        return {memory_id for term in terms for memory_id in self.term_frequencies.get(term, ())}

    def tokens_of(self, memory_id: str) -> frozenset[str]:
        """Normalised token set of an indexed memory."""
        return self.tokens_by_memory.get(memory_id, frozenset())


def _repository_path(repository: MemoryRepository) -> Path:
    """Locate the SQLite file backing a repository.

    The repository contract does not require a file location (an in-memory or
    remote backend would not have one), so the embedding store has to be
    injected in that case.
    """
    path = getattr(repository, "db_path", None)
    if path is None:
        raise TypeError(
            "the repository does not expose a db_path; inject an EmbeddingStore instead"
        )
    return Path(path)


class MemoryRetrievalService:
    """Explainable, memory-aware retrieval over the local corpus."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        ranking: RankingConfig | None = None,
        intelligence: IntelligenceConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_store: EmbeddingStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._repository = repository
        self._ranking = ranking or RankingConfig()
        self._intelligence = intelligence or IntelligenceConfig()
        self._provider = embedding_provider
        self._store = embedding_store or SqliteEmbeddingStore(_repository_path(repository))
        self._clock = clock or SystemClock()
        self._index = _InvertedIndex()
        self._signature: tuple[int, str] | None = None

    # -- configuration -----------------------------------------------------

    @property
    def ranking_config(self) -> RankingConfig:
        """The ranking configuration in use."""
        return self._ranking

    @property
    def semantic_available(self) -> bool:
        """True when an embedding provider is configured and usable."""
        return self._provider is not None and self._provider.is_available()

    @property
    def semantic_provider(self) -> str | None:
        """Provider name, or ``None`` when embeddings are not in play."""
        return None if self._provider is None else self._provider.name

    @property
    def semantic_model(self) -> str | None:
        """Model name, or ``None`` when embeddings are not in play."""
        return None if self._provider is None else self._provider.model

    def describe_provider(self) -> str:
        """Human readable description of the embedding layer (or its absence)."""
        if self._provider is None:
            return "none (lexical only)"
        return f"{self._provider.name}/{self._provider.model} ({self._provider.dimensions}d)"

    # -- index management --------------------------------------------------

    def rebuild_index(self) -> None:
        """Rebuild the inverted index from the searchable corpus.

        The corpus is "everything that is not forgotten": the status filter is
        applied by the candidate query itself, so an explicit request for
        archived memories still works.
        """
        memories = self._repository.search_candidates(CandidateQuery(status=None))
        index = _InvertedIndex(corpus_size=len(memories))
        for memory in memories:
            tokens = normalized_tokens(memory.content)
            index.tokens_by_memory[memory.id] = frozenset(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                index.term_frequencies.setdefault(token, {})[memory.id] = count
        for token, postings in index.term_frequencies.items():
            index.document_frequencies[token] = len(postings)
        self._index = index
        self._signature = self._repository.searchable_corpus_signature()

    def _ensure_index(self) -> None:
        """Rebuild the index when the active corpus has changed."""
        signature = self._repository.searchable_corpus_signature()
        if signature != self._signature:
            self.rebuild_index()

    # -- retrieval ---------------------------------------------------------

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Run a retrieval query and return ranked, explainable results."""
        started = time.perf_counter()
        self._ensure_index()
        now = self._clock.now()
        terms = query.terms

        semantic_available = self.semantic_available
        effective_mode, degraded_reason = self._resolve_mode(query.mode, semantic_available)

        candidates = self._candidates(query, terms, effective_mode)
        query_embedding = self._query_embedding(query.text, effective_mode, semantic_available)
        scored = [
            self._score_one(
                memory,
                terms=terms,
                query=query,
                mode=effective_mode,
                semantic_available=semantic_available,
                now=now,
                redundancy_penalty=0.0,
                query_embedding=query_embedding,
            )
            for memory in candidates
        ]
        selected, suppressed = self._select(query, scored)

        return RetrievalResponse(
            query=query.text,
            top_k=query.top_k,
            requested_mode=query.mode,
            retrieval_mode=effective_mode,
            semantic_available=semantic_available,
            semantic_provider=self.semantic_provider,
            semantic_model=self.semantic_model,
            degraded_reason=degraded_reason,
            candidates_considered=len(candidates),
            candidates_returned=len(selected),
            candidates_suppressed=len(suppressed),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            results=tuple(selected),
            suppressed=tuple(suppressed),
        )

    def explain(self, memory_id: str, query: RetrievalQuery) -> RetrievalResult:
        """Score one memory against a query, with the full explanation."""
        self._ensure_index()
        memory = self._require_memory(memory_id)
        now = self._clock.now()
        semantic_available = self.semantic_available
        effective_mode, _ = self._resolve_mode(query.mode, semantic_available)
        query_embedding = self._query_embedding(query.text, effective_mode, semantic_available)
        return self._score_one(
            memory,
            terms=query.terms,
            query=query,
            mode=effective_mode,
            semantic_available=semantic_available,
            now=now,
            redundancy_penalty=0.0,
            query_embedding=query_embedding,
        )

    # -- internals: candidates ---------------------------------------------

    def _require_memory(self, memory_id: str) -> Memory:
        memory = self._repository.get(memory_id)
        if memory is None:
            raise MemoryNotFoundError(memory_id)
        return memory

    @staticmethod
    def _resolve_mode(
        requested: RetrievalMode, semantic_available: bool
    ) -> tuple[RetrievalMode, str | None]:
        """Downgrade a request that needs embeddings when none are available."""
        if requested is RetrievalMode.LEXICAL or semantic_available:
            return requested, None
        return (
            RetrievalMode.LEXICAL,
            f"{requested.value} retrieval was requested but no embedding provider is "
            "available, so lexical retrieval was used",
        )

    def _candidates(
        self, query: RetrievalQuery, terms: tuple[str, ...], mode: RetrievalMode
    ) -> list[Memory]:
        """Candidate generation: index postings, or the whole active corpus."""
        candidate_query = CandidateQuery(
            ids=None
            if mode is not RetrievalMode.LEXICAL
            else tuple(sorted(self._index.documents_for(terms))),
            status=query.filters.status,
            memory_type=query.filters.memory_type,
            min_confidence=query.filters.min_confidence,
            min_importance=query.filters.min_importance,
            created_after=query.filters.created_after,
            created_before=query.filters.created_before,
        )
        return self._repository.search_candidates(candidate_query)

    def _query_embedding(
        self, text: str, mode: RetrievalMode, semantic_available: bool
    ) -> tuple[float, ...] | None:
        """Embed the query once when the semantic path is in play."""
        if mode not in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID) or not semantic_available:
            return None
        if self._provider is None:
            return None
        try:
            return self._provider.embed(text).values
        except Exception:  # a provider failure degrades to lexical, it never breaks search
            return None

    # -- internals: scoring -------------------------------------------------

    def _score_one(
        self,
        memory: Memory,
        *,
        terms: tuple[str, ...],
        query: RetrievalQuery,
        mode: RetrievalMode,
        semantic_available: bool,
        now: datetime,
        redundancy_penalty: float,
        query_embedding: tuple[float, ...] | None,
    ) -> RetrievalResult:
        """Score one candidate and build its explanation."""
        lexical = lexical_relevance(
            terms,
            self._index.tokens_of(memory.id),
            corpus_size=max(1, self._index.corpus_size),
            document_frequencies=self._index.document_frequencies,
        )
        semantic_score = (
            self._semantic_for(memory, query.text, query_embedding)
            if mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID)
            else None
        )
        relevance = self._relevance(mode, lexical, semantic_score, query.semantic_weight)
        priority, importance_score = self._priority_for(memory, now)
        freshness_value = freshness(
            memory, now=now, half_life_days=self._ranking.decay_half_life_days
        )
        type_value = type_preference(memory, self._ranking)

        conflicts, resolutions, superseded_by = self._conflicts_for(memory)
        conflict_penalty = conflict_penalty_from(resolutions)

        score = final_score(
            relevance=relevance,
            priority=priority,
            confidence=memory.confidence,
            freshness_value=freshness_value,
            type_value=type_value,
            conflict_penalty_value=conflict_penalty,
            redundancy_penalty_value=redundancy_penalty,
            config=self._ranking,
        )
        factors = self._factors(
            memory=memory,
            relevance=relevance,
            lexical=lexical,
            semantic_score=semantic_score,
            mode=mode,
            semantic_weight=self._ranking.semantic_weight_for(query.semantic_weight),
            priority=priority,
            importance_score=importance_score,
            freshness_value=freshness_value,
            type_value=type_value,
        )
        explanation = ranking_explanation(
            query=query.text,
            factors=factors,
            conflict_penalty_value=conflict_penalty,
            redundancy_penalty_value=redundancy_penalty,
            config=self._ranking,
            final=score,
        )
        return RetrievalResult(
            memory=memory,
            lexical_score=lexical,
            semantic_score=semantic_score,
            relevance=relevance,
            final_score=score,
            components=factors,
            explanation=explanation,
            conflicts=tuple(conflicts),
            superseded_by=superseded_by,
            duplicate_of=self._duplicate_of(memory),
            suppressed=redundancy_penalty > 0.0,
            suppression_reason=None,
        )

    def _relevance(
        self,
        mode: RetrievalMode,
        lexical: float,
        semantic_score: float | None,
        requested_weight: float | None,
    ) -> float:
        """Documented blend: lexical only, semantic only, or a weighted mix."""
        if mode is RetrievalMode.SEMANTIC:
            return semantic_score if semantic_score is not None else 0.0
        if mode is RetrievalMode.HYBRID and semantic_score is not None:
            weight = self._ranking.semantic_weight_for(requested_weight)
            return hybrid_relevance(lexical, semantic_score, semantic_weight=weight)
        return lexical

    def _factors(
        self,
        *,
        memory: Memory,
        relevance: float,
        lexical: float,
        semantic_score: float | None,
        mode: RetrievalMode,
        semantic_weight: float,
        priority: float,
        importance_score: float,
        freshness_value: float,
        type_value: float,
    ) -> tuple[RankingFactor, ...]:
        weights = self._ranking.component_weights
        if mode is RetrievalMode.SEMANTIC:
            relevance_explanation = (
                f"relevance {relevance:.4f} = semantic similarity only "
                f"({semantic_score if semantic_score is not None else 0.0:.4f})"
            )
        elif mode is RetrievalMode.HYBRID and semantic_score is not None:
            relevance_explanation = (
                f"relevance {relevance:.4f} = (1 - {semantic_weight:.2f}) * lexical "
                f"{lexical:.4f} + {semantic_weight:.2f} * semantic {semantic_score:.4f}"
            )
        else:
            relevance_explanation = f"relevance {relevance:.4f} = lexical coverage only"

        specs = (
            ("relevance", relevance, relevance_explanation),
            (
                "effective_priority",
                priority,
                f"M2 effective priority {priority:.4f} (importance {importance_score:.4f}, "
                f"confidence {memory.confidence:.2f}, decay applied)",
            ),
            ("confidence", memory.confidence, f"stored confidence {memory.confidence:.2f}"),
            (
                "freshness",
                freshness_value,
                f"decay factor {freshness_value:.4f} "
                f"(half-life {self._ranking.decay_half_life_days:g} days)",
            ),
            (
                "type_preference",
                type_value,
                f"{memory.memory_type.value} has retrieval type weight {type_value:.2f}",
            ),
        )
        return tuple(
            RankingFactor(
                name=name,
                weight=weights[name],
                value=round(value, 6),
                contribution=round(weights[name] * value, 6),
                explanation=explanation,
            )
            for name, value, explanation in specs
        )

    def _priority_for(self, memory: Memory, now: datetime) -> tuple[float, float]:
        """M2 priority, with recurrence taken from stored relationships."""
        duplicate_count = sum(
            1
            for relationship in self._repository.relationships(memory.id)
            if relationship.relationship is RelationshipType.DUPLICATE_OF
        )
        breakdown = compute_importance(
            memory,
            config=self._intelligence,
            now=now,
            duplicate_count=duplicate_count,
            access_count=memory.access_count,
        )
        priority = compute_effective_priority(
            memory,
            config=self._intelligence,
            now=now,
            importance_score=breakdown.score,
        )
        return priority.priority, breakdown.score

    def _conflicts_for(self, memory: Memory) -> tuple[list[ConflictInfo], list[str], str | None]:
        """Conflict metadata and the M2-derived conflict penalty."""
        relationships = self._repository.relationships(memory.id)
        source_pairs = extract_attribute_values(memory.content)
        infos: list[ConflictInfo] = []
        resolutions: list[str] = []
        superseded_by: str | None = None

        for relationship in relationships:
            if relationship.relationship is not RelationshipType.CONFLICTS_WITH:
                continue
            other_id = (
                relationship.target_id
                if relationship.source_id == memory.id
                else relationship.source_id
            )
            resolution, reason = self._conflict_resolution(memory.id, other_id, relationships)
            resolutions.append(resolution)
            if resolution == "superseded_by_other" and superseded_by is None:
                superseded_by = other_id
            other = self._repository.get(other_id)
            target_value = ""
            if other is not None:
                attribute = source_pairs[0].raw_attribute if source_pairs else ""
                for pair in extract_attribute_values(other.content):
                    if pair.raw_attribute == attribute:
                        target_value = pair.raw_value
                        break
            infos.append(
                ConflictInfo(
                    target_memory_id=other_id,
                    relationship=relationship.relationship.value,
                    attribute=source_pairs[0].raw_attribute if source_pairs else "",
                    source_value=source_pairs[0].raw_value if source_pairs else "",
                    target_value=target_value,
                    resolution=resolution,
                    resolution_reason=reason,
                )
            )
        return infos, resolutions, superseded_by

    @staticmethod
    def _conflict_resolution(
        memory_id: str, other_id: str, relationships: Sequence[MemoryRelationship]
    ) -> tuple[str, str | None]:
        """Resolve the state of one conflict from the SUPERSEDES links."""
        pair = frozenset({memory_id, other_id})
        for relationship in relationships:
            if relationship.relationship is not RelationshipType.SUPERSEDES:
                continue
            if frozenset({relationship.source_id, relationship.target_id}) != pair:
                continue
            if relationship.source_id == memory_id:
                return "supersedes_other", "this memory superseded the conflicting one"
            return "superseded_by_other", "this memory was superseded by the conflicting one"
        return "unresolved", None

    def _duplicate_of(self, memory: Memory) -> str | None:
        """A duplicate partner recorded by M2 detection, if any."""
        for relationship in self._repository.relationships(memory.id):
            if relationship.relationship is not RelationshipType.DUPLICATE_OF:
                continue
            other_id = (
                relationship.target_id
                if relationship.source_id == memory.id
                else relationship.source_id
            )
            if other_id != memory.id:
                return other_id
        return None

    def _semantic_for(
        self,
        memory: Memory,
        query_text: str,
        query_embedding: tuple[float, ...] | None,
    ) -> float:
        """Cosine similarity between the query and the memory's embedding."""
        if self._provider is None or not self._provider.is_available():
            return 0.0
        if query_embedding is None:
            return 0.0
        embedding = self._store.get_or_embed(
            memory.id, memory.content, self._provider, at=self._clock.now()
        )
        if embedding is None or embedding.dimensions != len(query_embedding):
            return 0.0
        return semantic_relevance(cosine_similarity(query_embedding, embedding.values))

    # -- internals: selection -----------------------------------------------

    def _select(
        self, query: RetrievalQuery, scored: list[RetrievalResult]
    ) -> tuple[list[RetrievalResult], list[SuppressedCandidate]]:
        """Deterministic ordering plus duplicate suppression."""
        ordered = sorted(scored, key=sort_key)
        selected: list[RetrievalResult] = []
        suppressed: list[SuppressedCandidate] = []
        selected_ids: set[str] = set()

        for result in ordered:
            duplicate_of = self._duplicate_of(result.memory)
            if duplicate_of is not None and duplicate_of in selected_ids:
                penalised = self._with_redundancy_penalty(result)
                suppressed.append(
                    SuppressedCandidate(
                        memory_id=result.memory.id,
                        final_score=penalised.final_score,
                        reason=(
                            f"duplicate_of {duplicate_of}, which ranks higher in this result set"
                        ),
                    )
                )
                continue
            if len(selected) >= query.top_k:
                continue
            selected.append(result)
            selected_ids.add(result.memory.id)

        return selected, suppressed

    def _with_redundancy_penalty(self, result: RetrievalResult) -> RetrievalResult:
        """Re-apply the formula with a full redundancy penalty (for reporting)."""
        conflict_penalty = conflict_penalty_from(
            [conflict.resolution for conflict in result.conflicts]
        )
        score = final_score(
            relevance=result.relevance,
            priority=self._factor_value(result, "effective_priority"),
            confidence=self._factor_value(result, "confidence"),
            freshness_value=self._factor_value(result, "freshness"),
            type_value=self._factor_value(result, "type_preference"),
            conflict_penalty_value=conflict_penalty,
            redundancy_penalty_value=1.0,
            config=self._ranking,
        )
        factors = tuple(
            RankingFactor(
                name=factor.name,
                weight=factor.weight,
                value=factor.value,
                contribution=factor.contribution,
                explanation=factor.explanation,
            )
            for factor in result.components
        )
        explanation = ranking_explanation(
            query="",
            factors=factors,
            conflict_penalty_value=conflict_penalty,
            redundancy_penalty_value=1.0,
            config=self._ranking,
            final=score,
        )
        return RetrievalResult(
            memory=result.memory,
            lexical_score=result.lexical_score,
            semantic_score=result.semantic_score,
            relevance=result.relevance,
            final_score=score,
            components=factors,
            explanation=explanation,
            conflicts=result.conflicts,
            superseded_by=result.superseded_by,
            duplicate_of=result.duplicate_of,
            suppressed=True,
            suppression_reason=None,
        )

    @staticmethod
    def _factor_value(result: RetrievalResult, name: str) -> float:
        for factor in result.components:
            if factor.name == name:
                return factor.value
        return 0.0
