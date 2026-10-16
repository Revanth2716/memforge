"""Deterministic token *estimation* for context budgets (M4).

Why an estimate and not a tokenizer
------------------------------------
MemForge never claims to know how many tokens a specific provider will charge
for. Doing that would require that provider's tokenizer (an optional, network
downloadable artefact) and would silently tie the context compiler to one model
family. Instead the compiler works with an **estimate** produced by this module:

- the default estimator uses only the standard library, so it is always
  available, always offline and always reproducible;
- it is an explicit, documented function of the text, not a measurement;
- it is configured, validated and replaceable (:class:`TokenEstimatorConfig`).

Every number this module produces is flagged ``exact=False`` and carries the id
of the estimator that produced it, so a downstream consumer can never mistake an
estimate for provider billing. MemForge's contract is therefore:

    the compiled context never exceeds the *requested* budget
    in terms of MemForge's own estimator.

The default formula (``memforge-char-word/v1``)
-----------------------------------------------
::

    A = number of ASCII code points      (code point < 128)
    N = number of non-ASCII code points
    W = number of whitespace-separated words

    char_cost  = A / chars_per_token                 # default 4.0
    word_cost  = W / words_per_token                 # default 1.3
    glyph_cost = N * non_ascii_tokens_per_char       # default 1.0

    estimated = max(1, ceil(max(char_cost, word_cost) + glyph_cost))

Both a character-based and a word-based term are computed and the **larger** one
is used, so the estimate follows whichever signal dominates: dense code
(characters) or long natural language (words). Non-ASCII characters are charged a
full token each, which is deliberately conservative for CJK text and for
accented or symbol-heavy prose (where real BPE tokenizers usually need *more*
than one token per character, never fewer). Empty content costs nothing.

Nothing in MemForge depends on this estimator being *accurate* -- only on it
being deterministic, monotonic and conservative enough that the hard budget
guarantee holds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.domain.errors import ContextValidationError, InvalidContextConfigError

#: Identifier of the bundled estimator, reported with every token figure so a
#: reader always knows which numbers they are looking at.
DEFAULT_ESTIMATOR_ID = "memforge-char-word/v1"

#: Human readable formula, echoed by the API so the estimate is self-describing.
DEFAULT_ESTIMATOR_FORMULA = (
    "estimated = max(1, ceil(max(ascii_chars / chars_per_token, "
    "words / words_per_token) + non_ascii_chars * non_ascii_tokens_per_char))"
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class TokenEstimatorConfig:
    """Knobs of the deterministic token estimator.

    Every value must be strictly positive: a zero or negative ratio would make
    the estimate meaningless (or infinite), so it is rejected at construction
    rather than producing nonsense budgets later.
    """

    #: Average characters per token for dense text. 4.0 is the widely used
    #: English approximation; no provider is assumed to match it.
    chars_per_token: float = 4.0
    #: Average words per token for natural language.
    words_per_token: float = 1.3
    #: Tokens charged for a single non-ASCII code point (conservative default).
    non_ascii_tokens_per_char: float = 1.0
    #: Reported alongside every figure produced with this configuration.
    estimator_id: str = DEFAULT_ESTIMATOR_ID

    def __post_init__(self) -> None:
        for name in ("chars_per_token", "words_per_token", "non_ascii_tokens_per_char"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise InvalidContextConfigError(f"{name} must be a finite number > 0 (got {value})")
        if not isinstance(self.estimator_id, str) or not self.estimator_id.strip():
            raise InvalidContextConfigError("estimator_id must be a non-empty string")

    @property
    def formula(self) -> str:
        """The formula behind this configuration (never a provider claim)."""
        return (
            f"{self.estimator_id}: estimated = max(1, ceil(max("
            f"ascii_chars/chars_per_token({self.chars_per_token:g}), "
            f"words/words_per_token({self.words_per_token:g})) + "
            f"non_ascii_chars*non_ascii_tokens_per_char"
            f"({self.non_ascii_tokens_per_char:g})))"
        )


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """One deterministic token estimate.

    ``exact`` is a constant ``False``: it exists so the API payload states
    explicitly that these are estimates and not provider tokenisations.
    """

    estimated_tokens: int
    characters: int
    words: int
    non_ascii_characters: int
    estimator: str
    exact: bool
    explanation: str


def count_characters(text: str) -> tuple[int, int, int]:
    """Return ``(characters, non_ascii_characters, words)`` of ``text``.

    Characters are Unicode code points (not UTF-8 bytes), words are the
    whitespace-separated runs of the trimmed text.
    """
    stripped = _WHITESPACE_RE.sub(" ", text).strip()
    non_ascii = sum(1 for character in stripped if not character.isascii())
    return len(stripped), non_ascii, len(stripped.split()) if stripped else 0


def estimate_tokens(
    text: str,
    *,
    config: TokenEstimatorConfig | None = None,
) -> TokenEstimate:
    """Estimate how many tokens ``text`` would occupy.

    Pure and deterministic: the same text and configuration always produce the
    same estimate, on any machine, with no tokenizer library installed.
    """
    if not isinstance(text, str):
        raise ContextValidationError("text must be a string")
    resolved = config or TokenEstimatorConfig()
    characters, non_ascii, words = count_characters(text)

    if characters == 0:
        return TokenEstimate(
            estimated_tokens=0,
            characters=0,
            words=0,
            non_ascii_characters=0,
            estimator=resolved.estimator_id,
            exact=False,
            explanation="empty content is estimated at 0 tokens",
        )

    ascii_characters = characters - non_ascii
    char_cost = ascii_characters / resolved.chars_per_token
    word_cost = words / resolved.words_per_token
    glyph_cost = non_ascii * resolved.non_ascii_tokens_per_char
    # ``round`` before ``ceil`` keeps a float such as 3.0000000000000004 from
    # being rounded up to 4 and breaking the "same input, same output" promise.
    estimated = max(1, math.ceil(round(max(char_cost, word_cost) + glyph_cost, 6)))

    return TokenEstimate(
        estimated_tokens=estimated,
        characters=characters,
        words=words,
        non_ascii_characters=non_ascii,
        estimator=resolved.estimator_id,
        exact=False,
        explanation=(
            f"{estimated} estimated tokens via {resolved.estimator_id} "
            f"(max({ascii_characters}/{resolved.chars_per_token:g} chars, "
            f"{words}/{resolved.words_per_token:g} words) "
            f"+ {non_ascii} non-ascii char(s) "
            f"x {resolved.non_ascii_tokens_per_char:g})"
        ),
    )


def estimate_many(
    texts: tuple[str, ...] | list[str],
    *,
    config: TokenEstimatorConfig | None = None,
) -> tuple[int, ...]:
    """Estimate several texts at once, preserving order."""
    return tuple(estimate_tokens(text, config=config).estimated_tokens for text in texts)


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and strip the result.

    This is the only transformation M4 ever applies to memory *content* by
    default, and it is lossless in meaning: no character other than whitespace is
    added, removed or reordered.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


__all__ = [
    "DEFAULT_ESTIMATOR_FORMULA",
    "DEFAULT_ESTIMATOR_ID",
    "TokenEstimate",
    "TokenEstimatorConfig",
    "count_characters",
    "estimate_many",
    "estimate_tokens",
    "normalize_whitespace",
]
