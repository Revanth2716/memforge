"""M4 configuration tests.

``tests/test_config.py`` covers the M1-M3 settings and is left untouched; the
context-compilation settings are proved here, including the fact that a bad
policy fails at startup rather than at the first compilation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestDefaults:
    def test_context_defaults_are_deterministic_and_offline(self) -> None:
        settings = Settings()
        assert settings.default_context_token_budget == 1200
        assert settings.max_context_token_budget == 16000
        assert settings.context_top_k == 20
        assert settings.embedding_provider is None

    def test_default_policy_uses_the_documented_weights(self) -> None:
        policy = Settings().context_policy()
        assert policy.utility_weights["relevance"] == 0.40
        assert policy.cost_weight == 0.35
        assert policy.cost_reference_tokens == 120
        assert policy.min_candidate_score == 0.0
        assert policy.max_block_tokens is None
        assert policy.compress is True

    def test_default_estimator_is_the_bundled_one(self) -> None:
        estimator = Settings().token_estimator_config()
        assert estimator.estimator_id == "memforge-char-word/v1"
        assert estimator.chars_per_token == 4.0
        assert estimator.words_per_token == 1.3
        assert estimator.non_ascii_tokens_per_char == 1.0


class TestValidation:
    def test_default_budget_may_not_exceed_the_maximum(self) -> None:
        with pytest.raises(ValidationError, match="max_context_token_budget"):
            Settings(default_context_token_budget=20000, max_context_token_budget=100)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"default_context_token_budget": 0},
            {"max_context_token_budget": 0},
            {"context_top_k": 0},
            {"context_cost_reference_tokens": 0},
            {"token_chars_per_token": 0.0},
            {"token_words_per_token": 0.0},
            {"token_non_ascii_tokens_per_char": 0.0},
            {"context_max_block_tokens": 0},
            {"context_cost_weight": 1.5},
            {"context_min_candidate_score": 2.0},
        ],
    )
    def test_invalid_context_settings_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            Settings(**kwargs)  # type: ignore[arg-type]

    def test_utility_weights_that_do_not_sum_to_one_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1"):
            Settings(
                context_relevance_weight=0.9,
                context_priority_weight=0.9,
                context_confidence_weight=0.9,
                context_freshness_weight=0.9,
                context_type_weight=0.9,
            )

    def test_a_valid_weight_split_is_accepted(self) -> None:
        settings = Settings(
            context_relevance_weight=0.50,
            context_priority_weight=0.20,
            context_confidence_weight=0.10,
            context_freshness_weight=0.10,
            context_type_weight=0.10,
        )
        assert settings.context_policy().relevance_weight == 0.50


class TestFactories:
    def test_context_settings_build_the_policy(self) -> None:
        settings = Settings(
            context_cost_weight=0.50,
            context_cost_reference_tokens=80,
            context_min_candidate_score=0.2,
            context_max_block_tokens=40,
        )
        policy = settings.context_policy()

        assert policy.cost_weight == 0.50
        assert policy.cost_reference_tokens == 80
        assert policy.min_candidate_score == 0.2
        assert policy.max_block_tokens == 40

    def test_estimator_settings_build_the_estimator(self) -> None:
        settings = Settings(
            token_chars_per_token=3.0,
            token_words_per_token=1.5,
            token_non_ascii_tokens_per_char=0.75,
        )
        estimator = settings.token_estimator_config()

        assert estimator.chars_per_token == 3.0
        assert estimator.words_per_token == 1.5
        assert estimator.non_ascii_tokens_per_char == 0.75


class TestEnvironment:
    def test_context_settings_come_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMFORGE_DEFAULT_CONTEXT_TOKEN_BUDGET", "256")
        monkeypatch.setenv("MEMFORGE_MAX_CONTEXT_TOKEN_BUDGET", "512")
        monkeypatch.setenv("MEMFORGE_CONTEXT_COST_REFERENCE_TOKENS", "64")
        settings = Settings()

        assert settings.default_context_token_budget == 256
        assert settings.max_context_token_budget == 512
        assert settings.context_policy().cost_reference_tokens == 64

    def test_estimator_settings_come_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMFORGE_TOKEN_CHARS_PER_TOKEN", "2.5")
        assert Settings().token_estimator_config().chars_per_token == 2.5

    def test_a_bad_environment_value_is_rejected_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMFORGE_CONTEXT_COST_REFERENCE_TOKENS", "0")
        with pytest.raises(ValidationError):
            Settings()
