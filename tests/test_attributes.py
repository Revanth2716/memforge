"""Attribute/value extraction and deterministic conflict detection."""

from __future__ import annotations

import pytest

from app.domain.attributes import (
    AttributeValue,
    compare_values,
    extract_attribute_values,
    find_conflicts,
)


class TestExtraction:
    def test_equals_pattern(self) -> None:
        pairs = extract_attribute_values("preferred backend language = Python")
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.pattern == "equals"
        assert pair.attribute == "prefer backend language"
        assert pair.value == "python"
        assert pair.raw_attribute == "preferred backend language"
        assert pair.raw_value == "Python"

    def test_colon_pattern(self) -> None:
        pairs = extract_attribute_values("preferred backend language: Python")
        assert len(pairs) == 1
        assert pairs[0].pattern == "colon"
        assert pairs[0].attribute == "prefer backend language"
        assert pairs[0].value == "python"

    def test_copula_pattern(self) -> None:
        pairs = extract_attribute_values("Python is the user's preferred language")
        assert len(pairs) == 1
        assert pairs[0].pattern == "copula"
        assert pairs[0].attribute == "python"
        assert pairs[0].value == "user prefer language"

    def test_text_without_a_pattern_yields_nothing(self) -> None:
        assert extract_attribute_values("User prefers concise answers.") == ()
        assert extract_attribute_values("") == ()
        assert extract_attribute_values("   ") == ()

    def test_empty_value_or_attribute_yields_nothing(self) -> None:
        assert extract_attribute_values("preferred = ") == ()
        assert extract_attribute_values("= Python") == ()
        assert extract_attribute_values("the = is") == ()

    def test_only_the_first_pattern_matches(self) -> None:
        # "=" wins over "is"; the copula form is not also extracted.
        pairs = extract_attribute_values("language is Python = preferred")
        assert len(pairs) == 1
        assert pairs[0].pattern == "equals"

    def test_first_line_only(self) -> None:
        pairs = extract_attribute_values("preferred backend language = Python\nother = thing")
        assert len(pairs) == 1
        assert pairs[0].value == "python"

    def test_invalid_pair_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            AttributeValue(
                attribute="", value="x", raw_attribute="", raw_value="x", pattern="equals"
            )


class TestValueComparison:
    def test_identical_values_are_not_a_conflict(self) -> None:
        left = extract_attribute_values("backend = Python")[0]
        right = extract_attribute_values("backend = python")[0]
        assert compare_values(left, right, threshold=0.5) is None

    def test_different_values_conflict(self) -> None:
        left = extract_attribute_values("backend = Python")[0]
        right = extract_attribute_values("backend = TypeScript")[0]
        comparison = compare_values(left, right, threshold=0.5)
        assert comparison is not None
        assert comparison.attribute == "backend"
        assert comparison.left_value == "python"
        assert comparison.right_value == "typescript"
        assert comparison.similarity == 0.0
        assert comparison.is_conflict is True

    def test_near_identical_values_are_not_a_conflict(self) -> None:
        left = extract_attribute_values("backend = Python 3")[0]
        right = extract_attribute_values("backend = Python 3.11")[0]
        assert compare_values(left, right, threshold=0.5) is None

    def test_different_attributes_never_conflict(self) -> None:
        left = extract_attribute_values("backend = Python")[0]
        right = extract_attribute_values("frontend = Angular")[0]
        assert compare_values(left, right, threshold=0.5) is None

    def test_threshold_controls_the_boundary(self) -> None:
        left = extract_attribute_values("tool = alpha beta")[0]
        right = extract_attribute_values("tool = alpha gamma")[0]
        # similarity of "alpha beta" vs "alpha gamma" is exactly 0.5
        assert compare_values(left, right, threshold=0.5) is None
        assert compare_values(left, right, threshold=0.6) is not None


class TestFindConflicts:
    def test_documented_conflict_pair(self) -> None:
        conflicts = find_conflicts(
            "preferred backend language = Python",
            "preferred backend language = TypeScript",
            value_similarity_threshold=0.5,
        )
        assert len(conflicts) == 1
        assert conflicts[0].attribute == "prefer backend language"
        assert conflicts[0].left_value == "python"
        assert conflicts[0].right_value == "typescript"

    def test_unrelated_texts_have_no_conflicts(self) -> None:
        assert find_conflicts("User likes Python", "User works from Hyderabad") == ()

    def test_same_value_is_not_a_conflict(self) -> None:
        assert (
            find_conflicts("backend = Python", "backend = python", value_similarity_threshold=0.5)
            == ()
        )

    def test_conflicting_and_agreed_attributes(self) -> None:
        conflicts = find_conflicts(
            "preferred backend language = Python",
            "preferred backend language = TypeScript",
            value_similarity_threshold=0.5,
        )
        assert len(conflicts) == 1
        assert conflicts[0].right_value == "typescript"

    def test_conflict_detection_is_symmetric(self) -> None:
        a = "preferred backend language = Python"
        b = "preferred backend language = TypeScript"
        forward = find_conflicts(a, b, value_similarity_threshold=0.5)
        backward = find_conflicts(b, a, value_similarity_threshold=0.5)
        assert len(forward) == len(backward) == 1
        assert forward[0].left_value == backward[0].right_value

    def test_hostile_text_is_handled(self) -> None:
        conflicts = find_conflicts("note = '; DROP TABLE memories; --", "note = safe")
        assert len(conflicts) == 1
