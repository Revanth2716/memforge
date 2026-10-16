"""Deterministic lexical text similarity (no embeddings).

M2 does not use embeddings or an LLM. Near-duplicate detection is built from a
documented, inspectable pipeline:

1. **tokenise** -- lowercase, then extract ``[a-z0-9]+`` runs (this naturally
   drops punctuation and splits ``user's`` into ``user`` + ``s``);
2. **drop short tokens** -- single characters carry no meaning here;
3. **remove stop words** -- a fixed, grammatical list (no negations, so
   "prefers X" and "does not prefer X" are never collapsed);
4. **suffix stripping** -- a tiny deterministic stemmer for plural, past tense
   and progressive forms;
5. **Dice coefficient over the resulting token sets**:
   ``2 * |A n B| / (|A| + |B|)`` -- symmetric, bounded in ``[0, 1]``.

Every step is pure and unit-tested, and :func:`similarity_details` returns the
token sets so a score can always be explained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Tokens are runs of ASCII letters/digits. Applied after lower-casing.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

#: Grammatical stop words. Deliberately excludes negations and modals so that
#: removing them cannot invert the meaning of a memory.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "but",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "at",
        "by",
        "from",
        "into",
        "than",
        "then",
        "so",
        "if",
        "per",
        "via",
        "we",
        "you",
        "they",
        "them",
        "his",
        "her",
        "their",
        "our",
        "your",
        "my",
        "me",
        "us",
        "he",
        "she",
    }
)

#: Minimum token length kept after tokenisation.
MIN_TOKEN_LENGTH = 2


@dataclass(frozen=True, slots=True)
class SimilarityDetail:
    """Everything needed to explain a similarity score."""

    score: float
    left_tokens: tuple[str, ...]
    right_tokens: tuple[str, ...]
    shared_tokens: tuple[str, ...]

    @property
    def explanation(self) -> str:
        """Human readable summary of the comparison."""
        return (
            f"Dice={self.score:.4f} over {len(self.shared_tokens)} shared token(s) "
            f"({', '.join(self.shared_tokens) or 'none'}); "
            f"left={list(self.left_tokens)}, right={list(self.right_tokens)}"
        )


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into lowercase alphanumeric tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _undouble(stem: str) -> str:
    """``preferr`` -> ``prefer`` (strip a duplicated trailing consonant)."""
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1].isalpha():
        return stem[:-1]
    return stem


def stem(token: str) -> str:
    """Strip a documented set of suffixes.

    Rules, applied in order (short tokens are left alone):

    * ``ies`` -> ``y`` (``studies`` -> ``study``);
    * ``sses`` -> ``ss`` (``classes`` -> ``class``);
    * ``ing`` / ``ed`` -> removed, then a doubled trailing consonant is halved
      (``preferring`` and ``preferred`` both -> ``prefer``);
    * ``es`` -> removed for ``x``/``z``/``s``/``h`` stems (``boxes`` -> ``box``,
      ``watches`` -> ``watch``);
    * ``s`` -> removed unless the word ends in ``ss``/``us``/``is``
      (``prefers`` and ``types`` -> ``prefer``/``type``).

    This is a *suffix stripper*, not a linguistic stemmer: it does not handle
    irregular morphology (``typed`` stays ``typed``) and that limitation is
    documented in the README.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses") and len(token) > 5:
        return token[:-2]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) > 5:
            return _undouble(token[: -len(suffix)])
    if token.endswith("es") and len(token) > 4 and token[-3] in "xzsh":
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        return token[:-1]
    return token


def normalize(text: str) -> str:
    """Return the normalised token string (space joined, stop words removed)."""
    return " ".join(normalized_tokens(text))


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Return the ordered, de-duplicated normalised tokens of ``text``."""
    tokens = [
        stem(token)
        for token in tokenize(text)
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]
    ordered: list[str] = []
    for token in tokens:
        if token not in ordered:
            ordered.append(token)
    return tuple(ordered)


def token_set(text: str) -> frozenset[str]:
    """Return the normalised token set of ``text``."""
    return frozenset(normalized_tokens(text))


def dice_similarity(left: str, right: str) -> float:
    """Dice coefficient between the token sets of two strings (``[0, 1]``)."""
    return similarity_details(left, right).score


def similarity_details(left: str, right: str) -> SimilarityDetail:
    """Dice coefficient plus the token sets behind it, for explainability."""
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    shared = left_tokens & right_tokens
    total = len(left_tokens) + len(right_tokens)
    score = 0.0 if total == 0 else 2.0 * len(shared) / total
    return SimilarityDetail(
        score=round(score, 6),
        left_tokens=tuple(sorted(left_tokens)),
        right_tokens=tuple(sorted(right_tokens)),
        shared_tokens=tuple(sorted(shared)),
    )
