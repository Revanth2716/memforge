"""The default LLM provider: a deterministic, offline stand-in.

The mock provider exists so that everything above it — the agent workflow, the
memory operations, the MCP tools, the API and the demo — is reproducible on a
laptop with no network and no credential. It is a **script**, not a model, and it
says so: its ``name`` is ``"mock"`` and every response it produces is a
deterministic function of the request.

Text mode
---------
The reply is a fixed, documented rendering of the request: the last user message,
the number of messages, and the schema name if one was requested. Identical input
always produces identical output.

Structured mode
---------------
``generate_structured`` understands one explicit convention, so the demo and the
tests can exercise the *real* extraction-and-storage path without a model: lines
of the last user message of the form::

    remember: [FACT] The user prefers Python.

are returned as candidate memories. ``[FACT]`` is a :class:`MemoryType` name; a
missing tag defaults to ``OBSERVATION``. Lines that do not match the convention
are ignored, so a prompt that never uses it yields an empty candidate list
rather than a fabricated memory.

Nothing here fabricates usage: the mock reports ``usage=None``, because it has no
tokens to report.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.domain.llm import (
    LLMMessage,
    LLMMessageRole,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

#: ``remember: [TYPE] content`` -- the mock's documented extraction convention.
_REMEMBER_LINE = re.compile(
    r"^\s*remember\s*:\s*(?:\[(?P<type>[A-Z]+)\]\s*)?(?P<content>\S.*?)\s*$"
)


@dataclass(frozen=True, slots=True)
class MockCandidate:
    """One candidate memory recognised in the prompt."""

    content: str
    memory_type: str
    confidence: float
    importance: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "importance": self.importance,
            "reason": self.reason,
        }


class MockLLMProvider(LLMProvider):
    """Deterministic offline provider (the default in every environment)."""

    name = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-echo-v1",
        candidates: Sequence[MockCandidate] | None = None,
    ) -> None:
        """Args:
        model: Reported model name (never implies a real model was used).
        candidates: Optional fixed candidates for structured requests. When
            omitted, the prompt convention is used instead.
        """
        self.model = model
        self._candidates = tuple(candidates) if candidates is not None else None

    @property
    def is_configured(self) -> bool:
        """The mock always is: it needs no credential and no network."""
        return True

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Render a deterministic reply describing the request itself."""
        last_user = _last_user_message(request.messages)
        body = "\n".join(
            [
                "[mock provider: deterministic offline stand-in, no model was called]",
                f"operation: {request.operation}",
                f"messages: {len(request.messages)}",
                f"characters: {sum(len(message.content) for message in request.messages)}",
                f"structured_output: {request.wants_structured_output}",
                f"last_user_message: {last_user}",
            ]
        )
        return LLMResponse(
            text=body,
            provider=self.name,
            model=self.model,
            # The mock has no tokens to report, so it reports none.
            usage=None,
            latency_ms=0.0,
            finish_reason="stop",
            response_id=None,
            request_metadata=request.to_dict(),
        )

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Return candidate memories using the documented prompt convention."""
        if request.response_schema is None:
            return super().generate_structured(request)
        candidates = (
            list(self._candidates) if self._candidates is not None else _parse_candidates(request)
        )
        payload = {"candidates": [candidate.to_dict() for candidate in candidates]}
        return LLMResponse(
            text=json.dumps(payload, sort_keys=True),
            provider=self.name,
            model=self.model,
            usage=None,
            latency_ms=0.0,
            finish_reason="stop",
            response_id=None,
            structured=payload,
            request_metadata=request.to_dict(),
        )

    def as_usage_report(self) -> LLMUsage | None:
        """Always ``None``: the mock never reports usage."""
        return None


def _last_user_message(messages: Sequence[LLMMessage]) -> str:
    """The final user message, or an empty string when there is none."""
    for message in reversed(messages):
        if message.role is LLMMessageRole.USER:
            return message.content
    return ""


def _parse_candidates(request: LLMRequest) -> list[MockCandidate]:
    """Extract ``remember:`` lines from the final user message."""
    candidates: list[MockCandidate] = []
    for line in _last_user_message(request.messages).splitlines():
        match = _REMEMBER_LINE.match(line)
        if match is None:
            continue
        candidates.append(
            MockCandidate(
                content=match.group("content"),
                memory_type=match.group("type") or "OBSERVATION",
                confidence=0.7,
                importance=0.5,
                reason="explicit 'remember:' instruction in the request",
            )
        )
    return candidates


def candidate_payload(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the canonical structured payload for candidate memories."""
    return {"candidates": [dict(candidate) for candidate in candidates]}


__all__ = ["MockCandidate", "MockLLMProvider", "candidate_payload"]
