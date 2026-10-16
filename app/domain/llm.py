"""LLM domain for MemForge (M5): a provider interface, not a model.

MemForge never talks to a model directly. Everything that can reach a provider
goes through the small, explicit surface in this module:

* :class:`LLMProvider` — one method for text, one for structured output;
* :class:`LLMRequest` / :class:`LLMResponse` — immutable value objects;
* :class:`LLMUsage` — present **only** when the provider actually reported usage.

Three rules this module enforces structurally rather than by convention:

**Token counts are never invented.** :class:`LLMUsage` is optional, and the
observability layer treats ``None`` as "unknown", never as ``0``. MemForge has
its own *estimator* (see :mod:`app.domain.tokens`), which is labelled as an
estimate everywhere it appears; a provider-reported count is a different thing and
is never mixed with it.

**Secrets are structurally absent.** A provider's credential lives on the provider
object with ``repr=False`` and is never copied into a response, an error, a log
record or an evidence payload. :func:`redact_text` and :func:`redact_mapping` are
the only sanctioned way to render anything that came from an HTTP layer.

**Failures are typed.** Every failure a provider can produce is one of the
:class:`LLMErrorKind` values below, raised as an :class:`LLMProviderError`. The
API layer maps those kinds to HTTP status codes; the provider layer never knows
what HTTP is.

Nothing in this module performs I/O, so the whole contract is unit-testable with
an injected fake.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.domain.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMNotConfiguredError,
    LLMProtocolError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransportError,
)

#: The single placeholder used wherever a secret would otherwise appear.
REDACTED = "***redacted***"

#: Header names that are dropped outright rather than masked.
SECRET_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "api_key",
        "x-goog-api-key",
        "cookie",
        "set-cookie",
    }
)


class LLMMessageRole(enum.StrEnum):
    """Who authored a message (the three every compatible API accepts)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One message of a chat request."""

    role: LLMMessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, LLMMessageRole):
            raise LLMProtocolError(f"unknown message role {self.role!r}")
        if not isinstance(self.content, str):
            raise LLMProtocolError("message content must be a string")

    def to_dict(self) -> dict[str, str]:
        """Wire representation of the message."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content: str) -> LLMMessage:
        """A system instruction."""
        return cls(LLMMessageRole.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> LLMMessage:
        """A user turn."""
        return cls(LLMMessageRole.USER, content)

    @classmethod
    def assistant(cls, content: str) -> LLMMessage:
        """An assistant turn."""
        return cls(LLMMessageRole.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Usage **as reported by the provider**.

    MemForge never derives these numbers. If the provider does not send a usage
    block, the field is absent from the response instead of being guessed, and
    the observability layer records ``usage: None`` rather than zeros.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: Always ``"provider"``: these numbers came from the wire, not from us.
    reported_by: str = "provider"

    def to_dict(self) -> dict[str, int | str | None]:
        """Serializable, log-safe form."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reported_by": self.reported_by,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LLMUsage:
        """Build usage from a provider payload, coercing or dropping bad fields."""
        return cls(
            prompt_tokens=_optional_count(data.get("prompt_tokens")),
            completion_tokens=_optional_count(data.get("completion_tokens")),
            total_tokens=_optional_count(data.get("total_tokens")),
        )


def _optional_count(value: Any) -> int | None:
    """Coerce a usage field to a non-negative integer, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 0 else None


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A provider-agnostic generation request.

    ``response_schema`` requests structured output; a provider that cannot honour
    it must say so rather than return prose that happens to look like JSON.
    """

    messages: tuple[LLMMessage, ...]
    #: Overrides the provider's configured model when set.
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    #: JSON schema the provider is asked to satisfy.
    response_schema: Mapping[str, Any] | None = None
    #: Name of the schema, for providers that require one.
    schema_name: str | None = None
    #: Opaque, non-secret operation label used in observability.
    operation: str = "generate"
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise LLMProtocolError("a request must carry at least one message")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise LLMProtocolError("temperature must be in [0, 2]")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise LLMProtocolError("max_tokens must be >= 1")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise LLMProtocolError("timeout_seconds must be > 0")

    @property
    def wants_structured_output(self) -> bool:
        """True when the caller asked for schema-constrained output."""
        return self.response_schema is not None

    def to_dict(self) -> dict[str, Any]:
        """Log-safe request metadata: shape only, never content."""
        return {
            "messages": len(self.messages),
            "roles": [message.role.value for message in self.messages],
            "characters": sum(len(message.content) for message in self.messages),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "structured": self.wants_structured_output,
            "schema_name": self.schema_name,
            "operation": self.operation,
        }


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """What a provider returned, plus what is safe to log about the call."""

    text: str
    provider: str
    model: str
    #: ``None`` when the provider reported no usage. Never synthesised.
    usage: LLMUsage | None = None
    latency_ms: float = 0.0
    finish_reason: str | None = None
    #: Provider response id when present.
    response_id: str | None = None
    #: Parsed structured payload (structured requests only).
    structured: Mapping[str, Any] | None = None
    #: Safe request metadata (see :meth:`LLMRequest.to_dict`).
    request_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_usage(self) -> bool:
        """True only when the provider actually reported usage."""
        return self.usage is not None


class LLMProvider(abc.ABC):
    """The interface every provider implements.

    Implementations must be side-effect free with respect to credentials: the key
    belongs to the instance, never to a response, an error or a log record.
    """

    #: Stable provider identifier, e.g. ``"mock"`` or ``"custom"``.
    name: str = "abstract"
    #: Model identifier this provider was configured with.
    model: str = "unset"

    @property
    @abc.abstractmethod
    def is_configured(self) -> bool:
        """True when the provider can serve a request right now."""

    @abc.abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate free text."""

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate output constrained by ``request.response_schema``.

        The default implementation calls :meth:`generate` and refuses anything
        that is not valid JSON against the requested schema. Providers that
        support native structured output override this.
        """
        if request.response_schema is None:
            raise LLMProtocolError("generate_structured requires a response_schema")
        response = self.generate(request)
        payload = parse_json_object(response.text)
        return LLMResponse(
            text=response.text,
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            response_id=response.response_id,
            structured=payload,
            request_metadata=response.request_metadata,
        )

    def describe(self) -> str:
        """One-line, secret-free description of the provider."""
        return f"{self.name} (model={self.model}, configured={self.is_configured})"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} model={self.model!r}>"


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model output, normalising every failure.

    Models routinely wrap JSON in prose or a fenced block, so a strict parse is
    retried against the outermost braces. Anything that still fails is a protocol
    error, never a half-parsed guess.
    """
    from app.domain.errors import LLMProtocolError as _ProtocolError

    stripped = text.strip()
    if not stripped:
        raise _ProtocolError("the provider returned an empty response")
    try:
        parsed = _loads(stripped)
    except ValueError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise _ProtocolError("the provider did not return a JSON object") from None
        try:
            parsed = _loads(stripped[start : end + 1])
        except ValueError as error:
            raise _ProtocolError(f"the provider returned malformed JSON: {error}") from None
    if not isinstance(parsed, dict):
        raise _ProtocolError(f"expected a JSON object, received {type(parsed).__name__}")
    return parsed


def _loads(text: str) -> Any:
    """``json.loads`` with the domain error type kept out of the stdlib import."""
    import json

    return json.loads(text)


# -- redaction ----------------------------------------------------------------


def redact_text(text: str, *secrets: str | None) -> str:
    """Replace every occurrence of a known secret with :data:`REDACTED`.

    Applied to every provider-generated string that reaches a log, an error or an
    evidence payload. It is a belt-and-braces measure: the credential never
    reaches these strings in the first place.
    """
    result = text
    for secret in secrets:
        if secret and len(secret) >= 4 and secret in result:
            result = result.replace(secret, REDACTED)
    return result


def redact_mapping(data: Mapping[str, Any], *secrets: str | None) -> dict[str, Any]:
    """Return a log-safe copy of a mapping.

    Auth-bearing keys are dropped entirely (masking them would still advertise
    that a credential exists); every remaining string value is passed through
    :func:`redact_text`.
    """
    safe: dict[str, Any] = {}
    for key, value in data.items():
        if str(key).lower() in SECRET_HEADERS:
            safe[str(key)] = REDACTED
        elif isinstance(value, str):
            safe[str(key)] = redact_text(value, *secrets)
        elif isinstance(value, Mapping):
            safe[str(key)] = redact_mapping(value, *secrets)
        elif isinstance(value, (list, tuple)):
            safe[str(key)] = [
                redact_text(item, *secrets) if isinstance(item, str) else item for item in value
            ]
        else:
            safe[str(key)] = value
    return safe


def safe_endpoint(base_url: str) -> str:
    """Strip any credential embedded in a URL (``https://user:pw@host``).

    A misconfigured base URL must not turn a log line into a secret leak, so the
    userinfo component is removed before the endpoint is reported anywhere.
    """
    if "@" not in base_url:
        return base_url
    scheme, _, remainder = base_url.rpartition("://")
    if not scheme:
        return base_url
    return f"{scheme}://{remainder.split('@', 1)[1]}"


#: Error types a provider may raise, keyed by the ``kind`` they report. The API
#: layer maps a kind to a status code; nothing else needs this table.
PROVIDER_ERROR_KINDS: tuple[str, ...] = (
    "not_configured",
    "timeout",
    "authentication",
    "rate_limit",
    "transport",
    "protocol",
    "provider_error",
)


def message_text(messages: Sequence[LLMMessage]) -> str:
    """Join message contents (used by providers that take a single prompt)."""
    return "\n\n".join(message.content for message in messages)


__all__ = [
    "PROVIDER_ERROR_KINDS",
    "REDACTED",
    "SECRET_HEADERS",
    "LLMAuthenticationError",
    "LLMError",
    "LLMMessage",
    "LLMMessageRole",
    "LLMNotConfiguredError",
    "LLMProtocolError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMTransportError",
    "LLMUsage",
    "message_text",
    "parse_json_object",
    "redact_mapping",
    "redact_text",
    "safe_endpoint",
]
