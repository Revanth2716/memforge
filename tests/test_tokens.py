"""Deterministic token estimation tests (M4).

The estimator is MemForge's own heuristic, not a provider tokenizer. What is
proved here is the contract that matters for budgeting: determinism, stability,
conservatism, configurability and honest labelling -- never provider accuracy.
"""

from __future__ import annotations

import math

import pytest

from app.domain.errors import ContextValidationError, InvalidContextConfigError
from app.domain.tokens import (
    DEFAULT_ESTIMATOR_FORMULA,
    DEFAULT_ESTIMATOR_ID,
    TokenEstimatorConfig,
    count_characters,
    estimate_many,
    estimate_tokens,
    normalize_whitespace,
)


class TestDeterminism:
    def test_same_input_same_estimate(self) -> None:
        text = "User prefers Python for backend work."
        assert estimate_tokens(text) == estimate_tokens(text)

    def test_estimate_is_stable_across_configurations(self) -> None:
        config = TokenEstimatorConfig()
        text = "deterministic to the last character"
        assert estimate_tokens(text, config=config) == estimate_tokens(text, config=config)

    def test_never_claims_exact_tokenisation(self) -> None:
        estimate = estimate_tokens("anything at all")
        assert estimate.exact is False
        assert estimate.estimator == DEFAULT_ESTIMATOR_ID
        assert DEFAULT_ESTIMATOR_ID in estimate.explanation

    def test_estimator_formula_is_documented(self) -> None:
        assert "chars_per_token" in DEFAULT_ESTIMATOR_FORMULA
        assert "words_per_token" in DEFAULT_ESTIMATOR_FORMULA
        assert "non_ascii" in DEFAULT_ESTIMATOR_FORMULA


class TestBoundaries:
    @pytest.mark.parametrize("text", ["", "   ", "\t\n  ", "\n"])
    def test_empty_content_costs_nothing(self, text: str) -> None:
        estimate = estimate_tokens(text)
        assert estimate.estimated_tokens == 0
        assert estimate.characters == 0
        assert estimate.words == 0

    def test_single_character(self) -> None:
        estimate = estimate_tokens("a")
        assert estimate.estimated_tokens == 1
        assert estimate.characters == 1
        assert estimate.words == 1

    def test_floats_are_not_rounded_up_by_binary_error(self) -> None:
        # 4 ASCII characters / 4.0 chars_per_token == exactly 1.0 token; a naive
        # ceil() over a float like 1.0000000000000002 would report 2.
        assert estimate_tokens("abcd").estimated_tokens == 1
        assert estimate_tokens("abcde").estimated_tokens == 2

    def test_word_signal_can_dominate_the_character_signal(self) -> None:
        # Ten one-character words: 10 chars is cheap, 10 words is not.
        text = "a b c d e f g h i j"
        assert estimate_tokens(text).estimated_tokens == 8

    def test_estimate_is_never_below_one_for_content(self) -> None:
        assert estimate_tokens("!").estimated_tokens >= 1

    def test_estimate_is_monotonic_in_length(self) -> None:
        short = estimate_tokens("a short memory").estimated_tokens
        longer = estimate_tokens("a short memory about python").estimated_tokens
        assert longer > short


class TestUnicodeAndPunctuation:
    def test_non_ascii_characters_are_charged_a_full_token(self) -> None:
        estimate = estimate_tokens("héllo")
        assert estimate.non_ascii_characters == 1
        assert estimate.estimated_tokens == 2

    def test_cjk_costs_more_than_the_same_length_ascii(self) -> None:
        ascii_text = "abcd"
        cjk_text = "中文测试"
        assert (
            estimate_tokens(cjk_text).estimated_tokens
            > estimate_tokens(ascii_text).estimated_tokens
        )

    def test_emoji_are_not_ascii(self) -> None:
        estimate = estimate_tokens("hi 🙂")
        assert estimate.non_ascii_characters == 1

    def test_punctuation_runs_are_ordinary_words(self) -> None:
        # "---" and "???" are whitespace-separated runs, so they count as words.
        # They are cheap: the estimate stays far below a character-per-token count.
        text = "value --- ??? ..."
        estimate = estimate_tokens(text)
        assert estimate.words == 4
        assert estimate.estimated_tokens == 5
        assert estimate.estimated_tokens < len(text)

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert (
            estimate_tokens("   padded   ").estimated_tokens
            == estimate_tokens("padded").estimated_tokens
        )

    def test_character_counts_use_code_points(self) -> None:
        assert count_characters("héllo") == (5, 1, 1)
        assert count_characters("two words") == (9, 0, 2)


class TestConfiguredEstimator:
    def test_custom_ratios_change_the_estimate(self) -> None:
        text = "User prefers Python for backend work."
        conservative = TokenEstimatorConfig(chars_per_token=2.0, words_per_token=1.0)
        assert (
            estimate_tokens(text, config=conservative).estimated_tokens
            > estimate_tokens(text).estimated_tokens
        )

    def test_non_ascii_cost_is_configurable(self) -> None:
        text = "日本語"
        cheap = TokenEstimatorConfig(non_ascii_tokens_per_char=0.0 + 0.5)
        assert (
            estimate_tokens(text, config=cheap).estimated_tokens
            < estimate_tokens(text).estimated_tokens
        )

    def test_estimator_id_is_reported_back(self) -> None:
        config = TokenEstimatorConfig(estimator_id="custom/v9")
        assert estimate_tokens("x", config=config).estimator == "custom/v9"

    def test_estimate_many_preserves_order(self) -> None:
        assert estimate_many(["a", "abcd", ""]) == (1, 1, 0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"chars_per_token": 0.0},
            {"chars_per_token": -1.0},
            {"words_per_token": 0.0},
            {"non_ascii_tokens_per_char": 0.0},
            {"estimator_id": "  "},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(InvalidContextConfigError):
            TokenEstimatorConfig(**kwargs)  # type: ignore[arg-type]

    def test_configuration_formula_reports_the_ratios(self) -> None:
        config = TokenEstimatorConfig(chars_per_token=3.0, words_per_token=2.0)
        assert "chars_per_token(3)" in config.formula
        assert "words_per_token(2)" in config.formula
        assert "non_ascii_tokens_per_char(1)" in config.formula


class TestWhitespaceNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  padded  ", "padded"),
            ("line\nbreak", "line break"),
            ("tabs\t\tand   spaces", "tabs and spaces"),
            ("already clean", "already clean"),
        ],
    )
    def test_normalisation_is_meaning_preserving(self, raw: str, expected: str) -> None:
        assert normalize_whitespace(raw) == expected

    def test_normalisation_never_increases_the_estimate(self) -> None:
        spaced = "word   \n\n   word"
        assert (
            estimate_tokens(normalize_whitespace(spaced)).estimated_tokens
            <= estimate_tokens(spaced).estimated_tokens
        )


class TestContract:
    def test_non_string_input_is_rejected(self) -> None:
        with pytest.raises(ContextValidationError):
            estimate_tokens(42)  # type: ignore[arg-type]

    def test_estimates_are_integers(self) -> None:
        assert isinstance(estimate_tokens("some text").estimated_tokens, int)

    def test_estimate_never_exceeds_a_char_bound(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text).estimated_tokens == math.ceil(400 / 4.0)
