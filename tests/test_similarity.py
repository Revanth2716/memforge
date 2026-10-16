"""Deterministic text similarity: normalisation, stemming, Dice coefficient."""

from __future__ import annotations

import pytest

from app.domain.similarity import (
    STOPWORDS,
    SimilarityDetail,
    dice_similarity,
    normalize,
    normalized_tokens,
    similarity_details,
    stem,
    token_set,
    tokenize,
)


class TestTokenize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("User prefers Python", ("user", "prefers", "python")),
            ("user's", ("user", "s")),
            ("memforge-2 is local-first!", ("memforge", "2", "is", "local", "first")),
            ("Tabs\tand\nnewlines", ("tabs", "and", "newlines")),
            ("", ()),
            ("   ", ()),
            ("Ünïcödé", ("n", "c", "d")),  # non-ASCII letters are not tokens
        ],
    )
    def test_tokenize(self, text: str, expected: tuple[str, ...]) -> None:
        assert tokenize(text) == expected

    def test_tokenize_rejects_non_strings(self) -> None:
        with pytest.raises(TypeError):
            tokenize(5)  # type: ignore[arg-type]


class TestStem:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("prefers", "prefer"),
            ("preferred", "prefer"),
            ("preferring", "prefer"),
            ("prefer", "prefer"),
            ("studies", "study"),
            ("running", "run"),
            ("stopped", "stop"),
            ("types", "type"),
            ("boxes", "box"),
            ("is", "is"),  # short tokens are left alone
            ("python", "python"),
            ("typescript", "typescript"),
        ],
    )
    def test_documented_suffix_rules(self, token: str, expected: str) -> None:
        assert stem(token) == expected

    def test_stem_merges_inflections_of_the_same_word(self) -> None:
        assert len({stem(token) for token in ("prefers", "preferred", "preferring")}) == 1


class TestNormalization:
    def test_stopwords_are_removed(self) -> None:
        assert normalized_tokens("the user is a fresher") == ("user", "fresher")

    def test_stopword_list_has_no_negations(self) -> None:
        for negation in ("not", "no", "never", "nor"):
            assert negation not in STOPWORDS

    def test_tokens_are_de_duplicated_in_order(self) -> None:
        assert normalized_tokens("python python java") == ("python", "java")

    def test_punctuation_does_not_split_words(self) -> None:
        assert normalized_tokens("user's") == ("user",)

    def test_normalize_joins_tokens(self) -> None:
        assert normalize("The USER prefers  Python.") == "user prefer python"

    def test_empty_text_has_no_tokens(self) -> None:
        assert normalize("") == ""
        assert token_set("   ") == frozenset()


class TestDiceSimilarity:
    def test_identical_text_scores_one(self) -> None:
        assert dice_similarity("user prefers python", "user prefers python") == 1.0

    def test_disjoint_text_scores_zero(self) -> None:
        assert dice_similarity("alpha beta", "gamma delta") == 0.0

    def test_both_empty_scores_zero(self) -> None:
        assert dice_similarity("", "") == 0.0
        assert dice_similarity("the a of", "and or but") == 0.0

    def test_documented_example_from_the_requirements(self) -> None:
        # "User prefers Python" vs "Python is the user's preferred language"
        score = dice_similarity("User prefers Python", "Python is the user's preferred language")
        assert score == pytest.approx(6 / 7)

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("alpha beta", "alpha beta", 1.0),
            ("alpha beta", "alpha gamma", 0.5),
            ("alpha beta", "alpha beta gamma delta", 2 / 3),
            ("alpha beta delta", "alpha beta gamma epsilon", 4 / 7),
            ("alpha beta gamma delta", "alpha beta gamma epsilon zeta eta", 0.6),
        ],
    )
    def test_known_values(self, left: str, right: str, expected: float) -> None:
        assert dice_similarity(left, right) == pytest.approx(expected, abs=1e-6)

    def test_similarity_is_symmetric(self) -> None:
        left, right = "user prefers python", "python is preferred"
        assert dice_similarity(left, right) == dice_similarity(right, left)

    def test_similarity_is_bounded(self) -> None:
        for pair in (("a b", "a b c"), ("a b c d", "a b"), ("x", "y z")):
            score = dice_similarity(*pair)
            assert 0.0 <= score <= 1.0

    def test_details_expose_the_evidence(self) -> None:
        comparison = similarity_details("user prefers python", "user prefers java")
        assert isinstance(comparison, SimilarityDetail)
        assert comparison.shared_tokens == ("prefer", "user")
        assert comparison.left_tokens == ("prefer", "python", "user")
        assert comparison.right_tokens == ("java", "prefer", "user")
        assert "shared token" in comparison.explanation

    def test_scores_are_rounded(self) -> None:
        assert similarity_details("alpha beta", "alpha beta gamma").score == 0.8

    @pytest.mark.parametrize(
        "payload",
        ["'; DROP TABLE memories; --", '"quoted"', "100% ? & =", "line1\nline2"],
    )
    def test_hostile_text_is_handled(self, payload: str) -> None:
        assert 0.0 <= dice_similarity(payload, payload) <= 1.0
