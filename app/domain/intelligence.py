"""Deterministic memory intelligence: importance, decay and resolution.

Nothing in this module looks at the wall clock, touches storage or calls a model.
Every function takes its inputs (memory, configuration, reference time) and
returns an explainable result, which is what makes M2 reproducible.

Scoring model (all values bounded in ``[0, 1]``)
-----------------------------------------------
**Importance** is a weighted sum of five documented factors::

    importance = w_type      * type_weight(memory_type)      # fixed table
               + w_explicit  * stored importance             # caller supplied
               + w_recurrence* min(1, duplicates / saturation)
               + w_recency   * 2 ** (-age_days / half_life)
               + w_access    * min(1, accesses / saturation)

**Effective priority** combines the computed importance with confidence and then
applies a decay multiplier, so a memory's priority falls as it ages *without the
stored memory ever being mutated*::

    base    = pw_importance * importance_score + pw_confidence * confidence
    decay   = 2 ** (-age_days / half_life)
    priority = base * decay
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from app.domain.errors import InvalidIntelligenceConfigError
from app.domain.models import Memory, MemoryStatus, MemoryType

#: Weight of each memory type in the importance score. These are explicit,
#: documented heuristics (not measurements): profile and fact memories are
#: treated as more load-bearing than observations.
DEFAULT_TYPE_WEIGHTS: Mapping[MemoryType, float] = MappingProxyType(
    {
        MemoryType.PROFILE: 1.0,
        MemoryType.FACT: 0.8,
        MemoryType.PREFERENCE: 0.7,
        MemoryType.TASK: 0.7,
        MemoryType.EVENT: 0.5,
        MemoryType.OBSERVATION: 0.3,
    }
)

#: Weights of the five importance factors. They must sum to 1.0 so the score is
#: a convex combination and therefore bounded in ``[0, 1]``.
DEFAULT_IMPORTANCE_WEIGHTS = MappingProxyType(
    {
        "memory_type": 0.20,
        "explicit_importance": 0.40,
        "recurrence": 0.15,
        "recency": 0.15,
        "access_frequency": 0.10,
    }
)

#: Weights combining importance and confidence into the base priority.
DEFAULT_PRIORITY_WEIGHTS = MappingProxyType({"importance": 0.70, "confidence": 0.30})

#: Decimal places kept in reported scores (keeps output stable and readable).
_SCORE_PRECISION = 6


def _validate_weights(weights: Mapping[str, float], label: str) -> None:
    for name, value in weights.items():
        if not 0.0 <= value <= 1.0:
            raise InvalidIntelligenceConfigError(f"{label}.{name} must be in [0, 1] (got {value})")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise InvalidIntelligenceConfigError(f"{label} must sum to 1.0 (got {total})")


@dataclass(frozen=True, slots=True)
class IntelligenceConfig:
    """Knobs for every deterministic M2 algorithm.

    Defaults are documented in the README; all of them are injectable, and the
    operational ones can also be overridden through ``MEMFORGE_*`` settings.
    """

    #: Dice similarity at/above which two memories count as near-duplicates.
    similarity_threshold: float = 0.60
    #: Two values of the *same* attribute whose similarity is below this are
    #: considered different values (i.e. a conflict).
    value_similarity_threshold: float = 0.50
    #: Days for a memory's recency/decay factor to halve.
    decay_half_life_days: float = 30.0
    #: Number of near-duplicates that saturates the recurrence factor.
    recurrence_saturation: int = 3
    #: Number of recorded accesses that saturates the access factor.
    access_saturation: int = 5
    importance_weights: Mapping[str, float] = field(default=DEFAULT_IMPORTANCE_WEIGHTS)
    priority_weights: Mapping[str, float] = field(default=DEFAULT_PRIORITY_WEIGHTS)
    type_weights: Mapping[MemoryType, float] = field(default=DEFAULT_TYPE_WEIGHTS)

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise InvalidIntelligenceConfigError("similarity_threshold must be in [0, 1]")
        if not 0.0 <= self.value_similarity_threshold <= 1.0:
            raise InvalidIntelligenceConfigError("value_similarity_threshold must be in [0, 1]")
        if self.decay_half_life_days <= 0.0:
            raise InvalidIntelligenceConfigError("decay_half_life_days must be > 0")
        if self.recurrence_saturation < 1:
            raise InvalidIntelligenceConfigError("recurrence_saturation must be >= 1")
        if self.access_saturation < 1:
            raise InvalidIntelligenceConfigError("access_saturation must be >= 1")
        _validate_weights(self.importance_weights, "importance_weights")
        _validate_weights(self.priority_weights, "priority_weights")
        for memory_type, weight in self.type_weights.items():
            if not 0.0 <= weight <= 1.0:
                raise InvalidIntelligenceConfigError(
                    f"type_weights[{memory_type}] must be in [0, 1]"
                )

    def type_weight(self, memory_type: MemoryType) -> float:
        """Documented weight of a memory type."""
        return float(self.type_weights[memory_type])


# -- pure factor functions ----------------------------------------------------


def decay_multiplier(age_days: float, half_life_days: float) -> float:
    """``2 ** (-age / half_life)``: 1.0 now, 0.5 after one half-life."""
    if age_days < 0.0:
        return 1.0
    multiplier = 2.0 ** (-age_days / half_life_days)
    return float(multiplier)


def saturation_factor(count: int, saturation: int) -> float:
    """``min(1, count / saturation)`` -- a bounded, saturating ramp."""
    if count <= 0:
        return 0.0
    return min(1.0, count / float(saturation))


def age_in_days(now: datetime, then: datetime) -> float:
    """Whole/partial days between two timezone-aware timestamps (>= 0)."""
    delta = now - then
    return max(0.0, delta.total_seconds() / 86400.0)


# -- importance ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorScore:
    """One factor of the importance score, with its contribution."""

    name: str
    weight: float
    raw_value: float
    contribution: float
    explanation: str


@dataclass(frozen=True, slots=True)
class ImportanceBreakdown:
    """Importance score plus the factor breakdown that produced it."""

    memory_id: str
    score: float
    factors: tuple[FactorScore, ...]
    explanation: str
    computed_at: datetime


def compute_importance(
    memory: Memory,
    *,
    config: IntelligenceConfig,
    now: datetime,
    duplicate_count: int = 0,
    access_count: int = 0,
) -> ImportanceBreakdown:
    """Compute the deterministic importance score of ``memory``.

    Args:
        memory: The memory to score (uses its stored ``importance`` field).
        config: Weights and saturation knobs.
        now: Reference time from the injected clock (never read implicitly).
        duplicate_count: Number of *other* active memories detected as
            near-duplicates -- the recurrence signal.
        access_count: Number of recorded retrieval events.
    """
    weights = config.importance_weights
    type_raw = config.type_weight(memory.memory_type)
    explicit_raw = float(memory.importance)
    recurrence_raw = saturation_factor(duplicate_count, config.recurrence_saturation)
    recency_raw = decay_multiplier(age_in_days(now, memory.updated_at), config.decay_half_life_days)
    access_raw = saturation_factor(access_count, config.access_saturation)

    raws: tuple[tuple[str, float, str], ...] = (
        (
            "memory_type",
            type_raw,
            f"{memory.memory_type.value} has type weight {type_raw:.2f}",
        ),
        (
            "explicit_importance",
            explicit_raw,
            f"caller supplied importance {explicit_raw:.2f}",
        ),
        (
            "recurrence",
            recurrence_raw,
            f"{duplicate_count} near-duplicate active memor"
            f"{'y' if duplicate_count == 1 else 'ies'} "
            f"(saturation {config.recurrence_saturation})",
        ),
        (
            "recency",
            recency_raw,
            f"updated {age_in_days(now, memory.updated_at):.2f} days ago, "
            f"half-life {config.decay_half_life_days:g} days",
        ),
        (
            "access_frequency",
            access_raw,
            f"{access_count} recorded access(es) (saturation {config.access_saturation})",
        ),
    )

    factors = tuple(
        FactorScore(
            name=name,
            weight=float(weights[name]),
            raw_value=round(raw, _SCORE_PRECISION),
            contribution=round(float(weights[name]) * raw, _SCORE_PRECISION),
            explanation=explanation,
        )
        for name, raw, explanation in raws
    )
    score = round(sum(factor.contribution for factor in factors), _SCORE_PRECISION)
    formula = " + ".join(
        f"{factor.weight:.2f}*{factor.name}={factor.contribution:.4f}" for factor in factors
    )
    return ImportanceBreakdown(
        memory_id=memory.id,
        score=score,
        factors=factors,
        explanation=f"importance {score:.4f} = {formula}",
        computed_at=now,
    )


# -- effective priority -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PriorityBreakdown:
    """Effective priority: importance x confidence, decayed over time."""

    memory_id: str
    priority: float
    base_priority: float
    decay_multiplier: float
    importance_score: float
    confidence: float
    age_days: float
    explanation: str
    computed_at: datetime


def compute_effective_priority(
    memory: Memory,
    *,
    config: IntelligenceConfig,
    now: datetime,
    importance_score: float,
) -> PriorityBreakdown:
    """Compute the decayed priority of a memory (never mutates it)."""
    weights = config.priority_weights
    base = (
        float(weights["importance"]) * importance_score
        + float(weights["confidence"]) * memory.confidence
    )
    age_days = age_in_days(now, memory.updated_at)
    decay = decay_multiplier(age_days, config.decay_half_life_days)
    priority = round(base * decay, _SCORE_PRECISION)
    return PriorityBreakdown(
        memory_id=memory.id,
        priority=priority,
        base_priority=round(base, _SCORE_PRECISION),
        decay_multiplier=round(decay, _SCORE_PRECISION),
        importance_score=round(importance_score, _SCORE_PRECISION),
        confidence=memory.confidence,
        age_days=round(age_days, 6),
        explanation=(
            f"priority {priority:.4f} = (0.70*importance {importance_score:.4f} + "
            f"0.30*confidence {memory.confidence:.2f}) * decay {decay:.4f} "
            f"after {age_days:.2f} days (half-life {config.decay_half_life_days:g} days)"
        ),
        computed_at=now,
    )


# -- conflict resolution ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolutionStep:
    """One comparison of the deterministic resolution policy."""

    factor: str
    left_value: str
    right_value: str
    winner_id: str


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    """Result of resolving one conflicting pair of memories."""

    winner: Memory
    loser: Memory
    steps: tuple[ResolutionStep, ...]
    decided_by: str
    explanation: str


def _format_value(value: float) -> str:
    return f"{value:.4f}"


def decide_conflict(
    left: Memory,
    right: Memory,
    *,
    config: IntelligenceConfig,
    now: datetime,
    left_importance: float,
    right_importance: float,
) -> ResolutionDecision:
    """Deterministically pick the winner of a conflict.

    The policy compares, in order, and stops at the first decisive factor:

    1. **explicit importance** -- the caller's stated importance wins;
    2. **confidence** -- the more accurate memory wins;
    3. **recency** -- the more recently updated memory wins;
    4. **effective priority** -- the decayed composite priority wins (this also
       catches "referenced more often" and "has more near-duplicates");
    5. **identifier** -- the lexicographically smaller id wins (guarantees a
       total order, so the outcome never depends on iteration order).

    Ties are broken by the *next* factor, so the only way to reach the id rule
    is when the two memories are identical on every other axis.
    """
    candidates = (
        ("explicit_importance", left.importance, right.importance),
        ("confidence", left.confidence, right.confidence),
        ("recency", left.updated_at.timestamp(), right.updated_at.timestamp()),
    )
    steps: list[ResolutionStep] = []
    winner: Memory | None = None
    decided_by = ""

    for factor, left_value, right_value in candidates:
        winner, decided_by = _compare(factor, left, right, left_value, right_value, steps)
        if winner is not None:
            break

    if winner is None:
        left_priority = compute_effective_priority(
            left, config=config, now=now, importance_score=left_importance
        ).priority
        right_priority = compute_effective_priority(
            right, config=config, now=now, importance_score=right_importance
        ).priority
        winner, decided_by = _compare(
            "effective_priority", left, right, left_priority, right_priority, steps
        )

    if winner is None:
        # Final deterministic tie-break: identifier order.
        if left.id <= right.id:
            winner, loser = left, right
        else:
            winner, loser = right, left
        steps.append(
            ResolutionStep(
                factor="identifier",
                left_value=left.id,
                right_value=right.id,
                winner_id=winner.id,
            )
        )
        decided_by = "identifier"
    else:
        loser = right if winner.id == left.id else left

    explanation = "; ".join(
        f"{step.factor}: {step.winner_id} wins ({step.left_value} vs {step.right_value})"
        for step in steps
    )
    return ResolutionDecision(
        winner=winner,
        loser=loser,
        steps=tuple(steps),
        decided_by=decided_by,
        explanation=f"resolved by {decided_by}: {explanation}",
    )


def _compare(
    factor: str,
    left: Memory,
    right: Memory,
    left_value: float,
    right_value: float,
    steps: list[ResolutionStep],
) -> tuple[Memory | None, str]:
    """Compare one factor; append a step when it decides (or ties)."""
    if left_value > right_value:
        steps.append(
            ResolutionStep(
                factor=factor,
                left_value=_format_value(left_value),
                right_value=_format_value(right_value),
                winner_id=left.id,
            )
        )
        return left, factor
    if right_value > left_value:
        steps.append(
            ResolutionStep(
                factor=factor,
                left_value=_format_value(left_value),
                right_value=_format_value(right_value),
                winner_id=right.id,
            )
        )
        return right, factor
    steps.append(
        ResolutionStep(
            factor=factor,
            left_value=_format_value(left_value),
            right_value=_format_value(right_value),
            winner_id="tie",
        )
    )
    return None, ""


def is_active(memory: Memory) -> bool:
    """True when a memory may take part in detection scans."""
    return memory.status is MemoryStatus.ACTIVE
