"""Deterministic attribute/value extraction for conflict detection.

M2 does not attempt unrestricted natural-language reasoning. Instead it looks for
three explicit, documented constructions and treats the rest of a memory as
opaque text:

``<attribute> = <value>``     e.g. ``preferred backend language = Python``
``<attribute>: <value>``     e.g. ``preferred backend language: Python``
``<attribute> is <value>``    e.g. ``Python is the user's preferred language``

Two memories are **conflict candidates** when they share a normalised attribute
and the normalised values are different (similarity below the configured value
similarity threshold). Everything about the comparison -- the attribute, both
values, and the value similarity -- is returned so the caller can judge it.

Known limitation: this grammar cannot detect contradictions expressed in free
prose ("likes X" versus "does not like X"). That requires semantic understanding
and is out of scope for M2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.similarity import dice_similarity, normalize

#: ``attribute = value``
_EQUALS_PATTERN = re.compile(r"^(?P<attribute>[^=]{1,120}?)\s*=\s*(?P<value>.+)$")
#: ``attribute: value``
_COLON_PATTERN = re.compile(r"^(?P<attribute>[^:]{1,120}?)\s*:\s*(?P<value>.+)$")
#: ``attribute is value``
_COPULA_PATTERN = re.compile(r"^(?P<attribute>[^:]{1,120}?)\s+is\s+(?P<value>.+)$", re.IGNORECASE)

#: Patterns are tried in this order; the first match wins.
PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EQUALS_PATTERN, "equals"),
    (_COLON_PATTERN, "colon"),
    (_COPULA_PATTERN, "copula"),
)


@dataclass(frozen=True, slots=True)
class AttributeValue:
    """A single ``attribute -> value`` pair extracted from a memory."""

    attribute: str
    value: str
    raw_attribute: str
    raw_value: str
    pattern: str

    def __post_init__(self) -> None:
        if not self.attribute or not self.value:
            raise ValueError("an attribute and a value are both required")


def _clean(raw: str) -> str:
    return raw.strip().strip("\"'").strip()


def extract_attribute_values(text: str) -> tuple[AttributeValue, ...]:
    """Extract every ``attribute = value`` pair found in ``text``.

    Only the first matching pattern is applied to a given text, and at most one
    pair is produced per text, which keeps the extraction deterministic.
    """
    if not isinstance(text, str) or not text.strip():
        return ()

    first_line = text.strip().splitlines()[0]
    for pattern, name in PATTERNS:
        match = pattern.match(first_line)
        if match is None:
            continue
        raw_attribute = _clean(match.group("attribute"))
        raw_value = _clean(match.group("value"))
        attribute = normalize(raw_attribute)
        value = normalize(raw_value)
        if not attribute or not value:
            return ()
        return (
            AttributeValue(
                attribute=attribute,
                value=value,
                raw_attribute=raw_attribute,
                raw_value=raw_value,
                pattern=name,
            ),
        )
    return ()


@dataclass(frozen=True, slots=True)
class ValueComparison:
    """Comparison of two values that share an attribute."""

    attribute: str
    left_value: str
    right_value: str
    similarity: float

    @property
    def is_conflict(self) -> bool:
        """True when the two values are different enough to be a conflict."""
        return self.similarity < 1.0


def compare_values(
    left: AttributeValue, right: AttributeValue, *, threshold: float
) -> ValueComparison | None:
    """Compare two attribute/value pairs that share the same attribute.

    Returns ``None`` when the pairs do not share an attribute, or when the
    values are similar enough to be considered the *same* value (``>= threshold``).
    """
    if left.attribute != right.attribute:
        return None
    similarity = dice_similarity(left.value, right.value)
    if similarity >= threshold:
        return None
    return ValueComparison(
        attribute=left.attribute,
        left_value=left.value,
        right_value=right.value,
        similarity=similarity,
    )


def find_conflicts(
    left_text: str,
    right_text: str,
    *,
    value_similarity_threshold: float = 0.5,
) -> tuple[ValueComparison, ...]:
    """Return every conflicting attribute/value pair between two texts."""
    left_pairs = extract_attribute_values(left_text)
    right_pairs = extract_attribute_values(right_text)
    conflicts: list[ValueComparison] = []
    for left in left_pairs:
        for right in right_pairs:
            comparison = compare_values(left, right, threshold=value_similarity_threshold)
            if comparison is not None:
                conflicts.append(comparison)
    return tuple(conflicts)
