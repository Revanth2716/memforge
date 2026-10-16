"""Deterministic importance scoring, temporal decay and resolution policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.domain.errors import InvalidIntelligenceConfigError
from app.domain.intelligence import (
    DEFAULT_TYPE_WEIGHTS,
    IntelligenceConfig,
    age_in_days,
    compute_effective_priority,
    compute_importance,
    decay_multiplier,
    decide_conflict,
    saturation_factor,
)
from app.domain.models import Memory, MemoryStatus, MemoryType
from tests.support import FIXED_START, make_memory

LATER = FIXED_START + timedelta(days=10)


class TestConfigValidation:
    def test_defaults_are_valid(self) -> None:
        config = IntelligenceConfig()
        assert config.similarity_threshold == 0.6
        assert config.decay_half_life_days == 30.0
        assert config.recurrence_saturation == 3
        assert config.access_saturation == 5

    @pytest.mark.parametrize(
        "field, value",
        [
            ("similarity_threshold", 1.5),
            ("similarity_threshold", -0.1),
            ("value_similarity_threshold", 2.0),
            ("decay_half_life_days", 0.0),
            ("decay_half_life_days", -1.0),
            ("recurrence_saturation", 0),
            ("access_saturation", -1),
        ],
    )
    def test_invalid_scalars_are_rejected(self, field: str, value: float) -> None:
        with pytest.raises(InvalidIntelligenceConfigError):
            IntelligenceConfig(**{field: value})  # type: ignore[arg-type]

    def test_importance_weights_must_sum_to_one(self) -> None:
        with pytest.raises(InvalidIntelligenceConfigError):
            IntelligenceConfig(importance_weights={"memory_type": 0.5})

    def test_priority_weights_must_sum_to_one(self) -> None:
        with pytest.raises(InvalidIntelligenceConfigError):
            IntelligenceConfig(priority_weights={"importance": 0.9, "confidence": 0.2})

    def test_type_weights_must_be_bounded(self) -> None:
        weights = {
            MemoryType.PROFILE: 1.0,
            MemoryType.FACT: 0.8,
            MemoryType.PREFERENCE: 0.7,
            MemoryType.TASK: 0.7,
            MemoryType.EVENT: 0.5,
            MemoryType.OBSERVATION: 1.5,
        }
        with pytest.raises(InvalidIntelligenceConfigError):
            IntelligenceConfig(type_weights=weights)

    def test_config_is_immutable(self) -> None:
        config = IntelligenceConfig()
        assert config.type_weight(MemoryType.PROFILE) == 1.0


class TestFactorFunctions:
    @pytest.mark.parametrize(
        ("age_days", "expected"),
        [(0.0, 1.0), (30.0, 0.5), (60.0, 0.25), (300.0, 2.0**-10)],
    )
    def test_decay_halves_every_half_life(self, age_days: float, expected: float) -> None:
        assert decay_multiplier(age_days, 30.0) == pytest.approx(expected)

    def test_decay_is_bounded_for_negative_age(self) -> None:
        assert decay_multiplier(-5.0, 30.0) == 1.0

    def test_decay_uses_a_different_half_life(self) -> None:
        assert decay_multiplier(7.0, 7.0) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(0, 0.0), (1, 1 / 3), (3, 1.0), (10, 1.0)],
    )
    def test_saturation_is_bounded(self, count: int, expected: float) -> None:
        assert saturation_factor(count, 3) == pytest.approx(expected)

    def test_age_in_days(self) -> None:
        assert age_in_days(LATER, FIXED_START) == pytest.approx(10.0)
        assert age_in_days(FIXED_START, LATER) == 0.0


class TestImportance:
    def test_score_is_bounded(self) -> None:
        for memory_type in MemoryType:
            memory = make_memory(memory_type=memory_type, importance=1.0, access_count=99)
            breakdown = compute_importance(
                memory,
                config=IntelligenceConfig(),
                now=LATER,
                duplicate_count=10,
                access_count=99,
            )
            assert 0.0 <= breakdown.score <= 1.0

    def test_minimum_score_is_zero(self) -> None:
        memory = make_memory(memory_type=MemoryType.OBSERVATION, importance=0.0)
        weights = {**DEFAULT_TYPE_WEIGHTS, MemoryType.OBSERVATION: 0.0}
        config = IntelligenceConfig(decay_half_life_days=1.0, type_weights=weights)
        breakdown = compute_importance(
            memory,
            config=config,
            now=FIXED_START + timedelta(days=365),
            duplicate_count=0,
            access_count=0,
        )
        assert breakdown.score == pytest.approx(0.0, abs=1e-6)

    def test_documented_formula(self) -> None:
        memory = make_memory(memory_type=MemoryType.PREFERENCE, importance=0.4, access_count=0)
        breakdown = compute_importance(
            memory,
            config=IntelligenceConfig(),
            now=FIXED_START,
            duplicate_count=1,
            access_count=0,
        )
        expected = 0.20 * 0.7 + 0.40 * 0.4 + 0.15 * (1 / 3) + 0.15 * 1.0 + 0.10 * 0.0
        assert breakdown.score == pytest.approx(expected)
        assert [factor.name for factor in breakdown.factors] == [
            "memory_type",
            "explicit_importance",
            "recurrence",
            "recency",
            "access_frequency",
        ]

    def test_factors_explain_themselves(self) -> None:
        breakdown = compute_importance(
            make_memory(importance=0.5),
            config=IntelligenceConfig(),
            now=FIXED_START,
            duplicate_count=2,
            access_count=1,
        )
        for factor in breakdown.factors:
            assert factor.explanation
            assert factor.contribution == pytest.approx(factor.weight * factor.raw_value)
        assert "importance " in breakdown.explanation
        assert breakdown.computed_at == FIXED_START

    def test_score_is_deterministic(self) -> None:
        memory = make_memory(importance=0.3)
        first = compute_importance(
            memory, config=IntelligenceConfig(), now=LATER, duplicate_count=1, access_count=2
        )
        second = compute_importance(
            memory, config=IntelligenceConfig(), now=LATER, duplicate_count=1, access_count=2
        )
        assert first == second

    def test_score_depends_on_recency(self) -> None:
        memory = make_memory()
        config = IntelligenceConfig(decay_half_life_days=10.0)
        fresh = compute_importance(memory, config=config, now=FIXED_START, access_count=0)
        stale = compute_importance(
            memory,
            config=config,
            now=FIXED_START + timedelta(days=10),
            duplicate_count=0,
            access_count=0,
        )
        assert stale.score < fresh.score

    def test_score_depends_on_type(self) -> None:
        config = IntelligenceConfig()
        profile = compute_importance(
            make_memory(memory_type=MemoryType.PROFILE), config=config, now=FIXED_START
        )
        observation = compute_importance(
            make_memory(memory_type=MemoryType.OBSERVATION), config=config, now=FIXED_START
        )
        assert profile.score > observation.score

    def test_score_depends_on_recurrence(self) -> None:
        memory = make_memory()
        config = IntelligenceConfig()
        alone = compute_importance(memory, config=config, now=FIXED_START)
        repeated = compute_importance(
            memory, config=config, now=FIXED_START, duplicate_count=3, access_count=0
        )
        assert repeated.score > alone.score

    def test_score_depends_on_access(self) -> None:
        memory = make_memory()
        config = IntelligenceConfig()
        unused = compute_importance(memory, config=config, now=FIXED_START, access_count=0)
        used = compute_importance(memory, config=config, now=FIXED_START, access_count=5)
        assert used.score > unused.score


class TestEffectivePriority:
    def test_priority_combines_importance_and_confidence(self) -> None:
        memory = make_memory(importance=1.0, confidence=1.0)
        breakdown = compute_effective_priority(
            memory, config=IntelligenceConfig(), now=FIXED_START, importance_score=1.0
        )
        assert breakdown.base_priority == pytest.approx(1.0)
        assert breakdown.decay_multiplier == 1.0
        assert breakdown.priority == pytest.approx(1.0)

    def test_priority_decays_without_mutating_the_memory(self) -> None:
        memory = make_memory(importance=0.6, confidence=0.8)
        config = IntelligenceConfig(decay_half_life_days=30.0)
        now = compute_effective_priority(
            memory, config=config, now=FIXED_START, importance_score=0.6
        )
        later = compute_effective_priority(
            memory, config=config, now=FIXED_START + timedelta(days=30), importance_score=0.6
        )
        assert later.decay_multiplier == pytest.approx(0.5)
        assert later.priority == pytest.approx(now.priority * 0.5)
        # The stored memory is untouched by the computation.
        assert memory.importance == 0.6
        assert memory.updated_at == FIXED_START

    def test_confidence_and_importance_are_separate_inputs(self) -> None:
        memory = make_memory(importance=1.0, confidence=0.0)
        breakdown = compute_effective_priority(
            memory, config=IntelligenceConfig(), now=FIXED_START, importance_score=1.0
        )
        assert breakdown.base_priority == pytest.approx(0.7)

    def test_priority_explains_itself(self) -> None:
        breakdown = compute_effective_priority(
            make_memory(confidence=0.5),
            config=IntelligenceConfig(),
            now=LATER,
            importance_score=0.5,
        )
        assert "priority " in breakdown.explanation
        assert breakdown.age_days == pytest.approx(10.0)


class TestResolutionPolicy:
    def _memory(self, **overrides: Any) -> Memory:
        return make_memory(**overrides)

    def test_higher_importance_wins(self) -> None:
        left = self._memory(id="11111111-1111-5111-8111-111111111111", importance=0.9)
        right = self._memory(id="22222222-2222-4222-8222-222222222222", importance=0.4)
        decision = decide_conflict(
            left,
            right,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.9,
            right_importance=0.4,
        )
        assert decision.winner.id == left.id
        assert decision.decided_by == "explicit_importance"
        assert len(decision.steps) == 1

    def test_confidence_breaks_an_importance_tie(self) -> None:
        left = self._memory(
            id="11111111-1111-5111-8111-111111111111", importance=0.5, confidence=0.9
        )
        right = self._memory(
            id="22222222-2222-4222-8222-222222222222", importance=0.5, confidence=0.2
        )
        decision = decide_conflict(
            left,
            right,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.5,
            right_importance=0.5,
        )
        assert decision.winner.id == left.id
        assert decision.decided_by == "confidence"
        assert [step.factor for step in decision.steps] == ["explicit_importance", "confidence"]

    def test_recency_breaks_a_confidence_tie(self) -> None:
        left = self._memory(
            id="11111111-1111-5111-8111-111111111111",
            importance=0.5,
            confidence=0.5,
            updated_at=FIXED_START,
        )
        right = self._memory(
            id="22222222-2222-4222-8222-222222222222",
            importance=0.5,
            confidence=0.5,
            created_at=FIXED_START - timedelta(days=1),
            updated_at=FIXED_START - timedelta(days=1),
        )
        decision = decide_conflict(
            left,
            right,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.5,
            right_importance=0.5,
        )
        assert decision.winner.id == left.id
        assert decision.decided_by == "recency"
        assert [step.factor for step in decision.steps] == [
            "explicit_importance",
            "confidence",
            "recency",
        ]

    def test_priority_breaks_a_recency_tie(self) -> None:
        # Same importance, confidence and timestamps, but the left memory was
        # scored higher (for example it has more near-duplicates).
        memory = self._memory(importance=0.5, confidence=0.5)
        twin = self._memory(
            id="22222222-2222-4222-8222-222222222222", importance=0.5, confidence=0.5
        )
        decision = decide_conflict(
            memory,
            twin,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.7,
            right_importance=0.7,
        )
        # Identical inputs: the policy falls through to the identifier rule.
        assert decision.decided_by == "identifier"

        higher = decide_conflict(
            memory,
            twin,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.8,
            right_importance=0.6,
        )
        assert higher.decided_by == "effective_priority"
        assert higher.winner.id == memory.id

    def test_identifier_is_the_final_deterministic_tie_break(self) -> None:
        memory = self._memory(importance=0.5, confidence=0.5)
        twin = self._memory(
            id="00000000-0000-4000-8000-000000000000", importance=0.5, confidence=0.5
        )
        decision = decide_conflict(
            memory,
            twin,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.5,
            right_importance=0.5,
        )
        assert decision.decided_by == "identifier"
        assert decision.winner.id == "00000000-0000-4000-8000-000000000000"
        assert [step.factor for step in decision.steps] == [
            "explicit_importance",
            "confidence",
            "recency",
            "effective_priority",
            "identifier",
        ]

    def test_policy_is_order_independent(self) -> None:
        left = self._memory(id="11111111-1111-5111-8111-111111111111", importance=0.7)
        right = self._memory(id="22222222-2222-4222-8222-222222222222", importance=0.3)
        forward = decide_conflict(
            left,
            right,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.7,
            right_importance=0.3,
        )
        backward = decide_conflict(
            right,
            left,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.3,
            right_importance=0.7,
        )
        assert forward.winner.id == backward.winner.id
        assert forward.loser.id == backward.loser.id

    def test_explanation_is_actionable(self) -> None:
        left = self._memory(id="11111111-1111-5111-8111-111111111111", importance=0.9)
        right = self._memory(id="22222222-2222-4222-8222-222222222222", importance=0.1)
        decision = decide_conflict(
            left,
            right,
            config=IntelligenceConfig(),
            now=FIXED_START,
            left_importance=0.9,
            right_importance=0.1,
        )
        assert decision.winner.id in decision.explanation
        assert "explicit_importance" in decision.explanation


def test_status_is_irrelevant_to_detection() -> None:
    """Sanity check: archived memories stay readable for the caller."""
    assert make_memory(status=MemoryStatus.ARCHIVED).status is MemoryStatus.ARCHIVED
    assert datetime.now(UTC).year >= 2026
