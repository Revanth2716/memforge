"""Agent domain for MemForge (M5): the memory-aware workflow, as data.

This module holds no I/O. It defines what a request looks like, how the agent
decides whether memory is relevant at all, what a candidate memory is, and what
safe metadata is captured about an operation.

Deliberately **not** here
-------------------------
There is no planner, no tool loop, no queue, no autonomous behaviour and no
model-driven control flow. The agent in MemForge is one bounded pipeline:

    request -> memory gate -> recall (M3) -> compile context (M4)
            -> provider call -> answer -> optional candidate extraction
            -> optional, explicitly permitted storage (M1)

Every step is deterministic given the same corpus, clock and provider. The only
non-deterministic component in the whole chain is the language model, and its
output is treated as *untrusted input* to the memory path: it can propose a
memory, but it can never write one without explicit permission.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from app.domain.context import ContextSection
from app.domain.errors import AgentValidationError
from app.domain.llm import LLMMessage, LLMUsage
from app.domain.models import MemoryType
from app.domain.retrieval import RetrievalMode

#: Requests shorter than this are treated as small talk unless memory is forced.
SMALL_TALK_CHARACTER_LIMIT = 24

#: Requests whose normalised text matches one of these need no stored memory.
SMALL_TALK_PREFIXES: tuple[str, ...] = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "yes",
    "no",
    "bye",
    "good morning",
    "good evening",
)

#: The JSON schema the provider is asked to satisfy when extracting candidates.
CANDIDATE_MEMORY_SCHEMA: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "memory_type": {
                            "type": "string",
                            "enum": [memory_type.value for memory_type in MemoryType],
                        },
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                    "required": ["content", "memory_type", "confidence", "importance"],
                },
            }
        },
        "required": ["candidates"],
    }
)

#: The instruction sent with a context-compilation request.
CONTEXT_SYSTEM_PROMPT = (
    "You are MemForge's memory-aware assistant. Answer using only the supplied "
    "memory context. If the context does not contain the answer, say so plainly. "
    "Never invent facts, identifiers or stored memories."
)

#: The instruction sent when asking the provider to propose new memories.
EXTRACTION_SYSTEM_PROMPT = (
    "Extract durable facts about the user from the conversation. Only propose "
    "what was stated explicitly. Return an empty list when nothing is durable."
)


class MemoryUse(enum.StrEnum):
    """Whether the agent consulted stored memory, and why."""

    #: Memory was recalled and compiled into the context.
    USED = "used"
    #: The gate decided memory was irrelevant (small talk).
    SKIPPED = "skipped"
    #: The caller asked for a context-free answer.
    DISABLED = "disabled"
    #: Recall produced no admissible candidate.
    NO_MATCHES = "no_matches"


@dataclass(frozen=True, slots=True)
class MemoryGate:
    """The deterministic decision about whether memory is relevant.

    This is a **documented heuristic**, not a classifier: it answers a narrow,
    cheap question ("is this the kind of turn that could possibly benefit from
    stored memory?") and it is bypassed entirely when the caller says so.
    """

    use: MemoryUse
    reason: str

    @property
    def should_recall(self) -> bool:
        """True when the gate decided memory is relevant.

        Only :attr:`MemoryUse.USED` proceeds to recall; ``skipped`` (small talk)
        and ``disabled`` (the caller opted out) short-circuit the memory path
        before any retrieval happens.
        """
        return self.use is MemoryUse.USED

    def to_dict(self) -> dict[str, str]:
        """Serializable form."""
        return {"use": self.use.value, "reason": self.reason}


def decide_memory_use(request: str, *, use_memory: bool = True) -> MemoryGate:
    """Decide whether stored memory is relevant for ``request``.

    Order of the rules (all deterministic, all cheap):

    1. ``use_memory=False`` -> :attr:`MemoryUse.DISABLED`;
    2. an empty turn, or a short greeting with no question in it ->
       :attr:`MemoryUse.SKIPPED`;
    3. everything else -> :attr:`MemoryUse.USED`.
    """
    if not use_memory:
        return MemoryGate(
            MemoryUse.DISABLED, "the caller explicitly disabled memory for this request"
        )
    text = " ".join(request.lower().split())
    if not text:
        return MemoryGate(MemoryUse.SKIPPED, "the request is empty")
    is_short = len(text) < SMALL_TALK_CHARACTER_LIMIT
    is_question = text.endswith(("!", "?", "."))
    if is_short and not is_question and _is_small_talk(text):
        return MemoryGate(
            MemoryUse.SKIPPED, "the request is small talk and cannot benefit from memory"
        )
    return MemoryGate(MemoryUse.USED, "the request may depend on stored memory")


def _is_small_talk(text: str) -> bool:
    """True when a short request is one of the documented greetings."""
    return any(text == prefix or text.startswith(f"{prefix} ") for prefix in SMALL_TALK_PREFIXES)


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    """A memory the provider *proposes*. Never written without permission."""

    content: str
    memory_type: MemoryType
    confidence: float
    importance: float
    reason: str = ""
    #: Set when the candidate was rejected instead of stored.
    rejection: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CandidateMemory:
        """Build a candidate from a provider payload, normalising every field.

        Unknown types, missing scores and empty content are *not* raised here: the
        candidate is returned with a rejection reason so the caller can report
        exactly what the provider proposed and what was done with it.
        """
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return cls("", MemoryType.OBSERVATION, 0.0, 0.0, rejection="empty_content")
        raw_type = payload.get("memory_type")
        try:
            memory_type = MemoryType(str(raw_type).upper()) if raw_type else MemoryType.OBSERVATION
        except ValueError:
            return cls(content.strip(), MemoryType.OBSERVATION, 0.0, 0.0, rejection="unknown_type")
        confidence = _unit(payload.get("confidence"), 0.5)
        importance = _unit(payload.get("importance"), 0.5)
        reason = payload.get("reason")
        return cls(
            content=content.strip(),
            memory_type=memory_type,
            confidence=confidence,
            importance=importance,
            reason=reason if isinstance(reason, str) else "",
        )

    @property
    def is_valid(self) -> bool:
        """True when the candidate may be stored."""
        return self.rejection is None

    def to_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "reason": self.reason,
            "rejection": self.rejection,
        }


def _unit(value: object, default: float) -> float:
    """Clamp a provider-supplied score into ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """One agent turn."""

    #: What the user asked.
    request: str
    #: Hard token budget for the compiled context (M4).
    token_budget: int = 800
    top_k: int = 10
    retrieval_mode: RetrievalMode = RetrievalMode.LEXICAL
    semantic_weight: float | None = None
    #: Set to False to answer without consulting memory at all.
    use_memory: bool = True
    #: Set to True to compile a context but never call a provider.
    context_only: bool = False
    #: Ask the provider to propose candidate memories.
    extract_candidates: bool = False
    #: Persist accepted candidates. **Always opt-in.**
    store_candidates: bool = False
    source: str = "agent:m5"
    temperature: float = 0.0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, str) or not self.request.strip():
            raise AgentValidationError("request must be a non-empty string")
        if isinstance(self.token_budget, bool) or self.token_budget < 1:
            raise AgentValidationError("token_budget must be >= 1 estimated token")
        if isinstance(self.top_k, bool) or self.top_k < 1:
            raise AgentValidationError("top_k must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise AgentValidationError("temperature must be in [0, 2]")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise AgentValidationError("max_tokens must be >= 1")
        if not isinstance(self.retrieval_mode, RetrievalMode):
            raise AgentValidationError(f"unknown retrieval mode {self.retrieval_mode!r}")

    @property
    def will_store(self) -> bool:
        """True when this turn may write to storage."""
        return self.extract_candidates and self.store_candidates


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """Safe, loggable metadata about one step of an agent turn.

    Deliberately small and deliberately boring: it exists so an operator can
    answer "what happened, which provider answered, how long did it take, and did
    it fail?" without ever seeing a credential or a fabricated number.
    """

    operation: str
    success: bool
    latency_ms: float
    provider: str | None = None
    model: str | None = None
    #: Usage **only** when the provider reported it.
    usage: LLMUsage | None = None
    #: M4's own estimate, always labelled as an estimate.
    context_estimated_tokens: int | None = None
    memories_recalled: int | None = None
    memories_stored: int | None = None
    error_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serializable, secret-free form."""
        return {
            "operation": self.operation,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "model": self.model,
            # ``None`` means "the provider did not report usage".
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "context_estimated_tokens": self.context_estimated_tokens,
            "memories_recalled": self.memories_recalled,
            "memories_stored": self.memories_stored,
            "error_kind": self.error_kind,
        }


@dataclass(frozen=True, slots=True)
class ContextBlockSummary:
    """Provenance of one compiled block, as an agent turn reports it."""

    memory_id: str
    memory_type: MemoryType
    section: ContextSection
    order: int
    utility_score: float
    retrieval_score: float
    estimated_tokens: int


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Everything one agent turn produced."""

    request: str
    answer: str
    #: The memory gate's decision.
    gate: MemoryGate
    #: The rendered context, when memory was used.
    context_text: str = ""
    context_estimated_tokens: int = 0
    context_token_budget: int = 0
    context_blocks: tuple[ContextBlockSummary, ...] = ()
    candidate_memories: tuple[CandidateMemory, ...] = ()
    stored_memory_ids: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    #: ``None`` unless the provider reported usage.
    usage: LLMUsage | None = None
    records: tuple[OperationRecord, ...] = ()
    messages: tuple[LLMMessage, ...] = field(default_factory=tuple)

    @property
    def used_memory(self) -> bool:
        """True when stored memory reached the provider."""
        return bool(self.context_text)


def build_messages(*, system: str, user: str, context: str = "") -> tuple[LLMMessage, ...]:
    """Build the provider message list for one turn.

    The compiled context is separated from the user request by a horizontal
    rule, so a model can tell stored memory from the live instruction.
    """
    body = f"{context}\n\n---\n\nUser request: {user}" if context else user
    return (LLMMessage.system(system), LLMMessage.user(body))


def messages_as_dicts(messages: Sequence[LLMMessage]) -> list[dict[str, str]]:
    """Serialize messages for an evidence payload."""
    return [message.to_dict() for message in messages]


__all__ = [
    "CANDIDATE_MEMORY_SCHEMA",
    "CONTEXT_SYSTEM_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT",
    "SMALL_TALK_CHARACTER_LIMIT",
    "SMALL_TALK_PREFIXES",
    "AgentRequest",
    "AgentResult",
    "CandidateMemory",
    "ContextBlockSummary",
    "MemoryGate",
    "MemoryUse",
    "OperationRecord",
    "build_messages",
    "decide_memory_use",
    "messages_as_dicts",
]
