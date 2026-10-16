"""Custom LLM provider: an OpenAI-compatible HTTP chat-completions client.

This is the only component in MemForge that can reach the network, and it is
**never** on the default path: :func:`build_llm_provider` returns the mock unless
``MEMFORGE_LLM_PROVIDER=custom`` *and* a base URL and model are configured.

Wire contract (the widely adopted "OpenAI-compatible" shape)
------------------------------------------------------------
Request — ``POST {base_url}{path}`` with ``Authorization: Bearer <key>``::

    {"model": "...",
     "messages": [{"role": "system|user|assistant", "content": "..."}],
     "temperature": 0.0,                 # omitted when unset
     "max_tokens": 512,                  # omitted when unset
     "response_format": {"type": "json_schema",
                         "json_schema": {"name": "...", "schema": {...}, "strict": true}}}

Response — the first choice's message content is used::

    {"id": "...", "model": "...",
     "choices": [{"message": {"content": "..."}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}

``usage`` is optional and is passed through **only when present**. Everything is
configurable: the endpoint path, the model, the timeout, the retry budget and the
backoff.

Safety
------
* the API key lives on the config object with ``repr=False`` and is never copied
  into a response, an error message, a log record or a header dump;
* every error message is passed through :func:`~app.domain.llm.redact_text` with
  the key as a secret, so a server that echoes the ``Authorization`` header
  cannot leak it through an error;
* ``repr`` of both the config and the provider omits the key;
* userinfo embedded in ``base_url`` is stripped from anything reported.

Failure normalisation
---------------------
===========================  =====================  ==========================
condition                    raised                 retryable
===========================  =====================  ==========================
missing base URL / model     ``LLMNotConfigured``   no
missing API key              ``LLMNotConfigured``   no
timeout                      ``LLMTimeout``         yes
connection failure           ``LLMTransport``       yes
401 / 403                     ``LLMAuthentication``  no
429                          ``LLMRateLimit``       yes
5xx                          ``LLMProviderError``   yes
other 4xx                    ``LLMProviderError``   no
body is not JSON / no choice ``LLMProtocol``        no
===========================  =====================  ==========================

Retries are **bounded** by ``max_retries`` (0 disables them) with a fixed,
configurable backoff, and only for retryable kinds. Nothing is retried forever
and nothing is retried forever silently.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from app.domain.errors import (
    LLMAuthenticationError,
    LLMNotConfiguredError,
    LLMProtocolError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransportError,
)
from app.domain.llm import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    parse_json_object,
    redact_mapping,
    redact_text,
    safe_endpoint,
)

#: Default path of the compatible chat-completions contract.
DEFAULT_CHAT_PATH = "/chat/completions"

#: Status codes that are safe to retry within a bounded budget.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class CustomLLMConfig:
    """Configuration of the custom provider.

    Built from the environment by :meth:`app.config.Settings.llm_config`; it never
    reads a credential from anywhere else, and the credential is never persisted.
    """

    base_url: str = ""
    #: Never rendered, logged or included in a repr.
    api_key: str = field(default="", repr=False)
    model: str = ""
    path: str = DEFAULT_CHAT_PATH
    timeout_seconds: float = 30.0
    #: Additional attempts after the first one. 0 disables retrying.
    max_retries: int = 2
    backoff_seconds: float = 0.5
    #: Send ``Authorization: Bearer`` (the compatible default) or nothing.
    send_authorization: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise LLMNotConfiguredError("timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise LLMNotConfiguredError("max_retries must be >= 0")
        if self.backoff_seconds < 0:
            raise LLMNotConfiguredError("backoff_seconds must be >= 0")
        if self.path and not self.path.startswith("/"):
            raise LLMNotConfiguredError("path must start with '/'")

    @property
    def endpoint(self) -> str:
        """Absolute endpoint, built from the base URL and the path."""
        return f"{self.base_url.rstrip('/')}{self.path}"

    @property
    def safe_base_url(self) -> str:
        """Base URL with any embedded userinfo removed."""
        return safe_endpoint(self.base_url)

    def is_configured(self) -> bool:
        """True when base URL, model and (when required) key are present."""
        if not self.base_url or not self.model:
            return False
        return not self.send_authorization or bool(self.api_key)

    def missing(self) -> tuple[str, ...]:
        """Names of the settings that block a request (safe to report)."""
        gaps: list[str] = []
        if not self.base_url:
            gaps.append("base_url")
        if not self.model:
            gaps.append("model")
        if self.send_authorization and not self.api_key:
            gaps.append("api_key")
        return tuple(gaps)

    def headers(self) -> dict[str, str]:
        """Request headers. The key is added here and nowhere else."""
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.send_authorization and self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def safe_headers(self) -> dict[str, str]:
        """Headers with the credential masked, for evidence payloads."""
        return redact_mapping(self.headers(), self.api_key)

    def describe(self) -> dict[str, Any]:
        """Secret-free description of the provider configuration.

        The endpoint goes through :func:`safe_endpoint`, so a base URL that
        carries ``user:password@`` cannot leak it into a log line.
        """
        return {
            "endpoint": safe_endpoint(self.endpoint),
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "authentication": "bearer" if self.send_authorization else "none",
            "api_key_present": bool(self.api_key),
        }


class CustomLLMProvider(LLMProvider):
    """OpenAI-compatible provider reached over HTTP."""

    name = "custom"

    def __init__(
        self,
        config: CustomLLMConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
        config: Provider configuration (holds the credential).
        client: Injected HTTP client. Tests pass one built on
            ``httpx.MockTransport``; production passes ``None`` so a client is
            created from the configured timeout.
        sleep: Injected sleep used between retries (tests pass a no-op).
        """
        if not isinstance(config, CustomLLMConfig):  # pragma: no cover - defensive
            raise TypeError("config must be a CustomLLMConfig")
        self._config = config
        self.model = config.model
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._sleep = sleep

    @property
    def config(self) -> CustomLLMConfig:
        """The configuration (its ``api_key`` is excluded from ``repr``)."""
        return self._config

    @property
    def endpoint(self) -> str:
        """The endpoint this provider calls."""
        return self._config.endpoint

    @property
    def is_configured(self) -> bool:
        """True when the configuration can serve a request."""
        return self._config.is_configured()

    def close(self) -> None:
        """Close the HTTP client if this provider created it."""
        if self._owns_client:
            self._client.close()

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the provider and normalise the answer."""
        payload = self._require_configured(request)
        body = self._post(payload, request)
        text = _extract_text(body, request)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=_string_or(body.get("model"), request.model or self._config.model),
            usage=_extract_usage(body),
            latency_ms=0.0,
            finish_reason=_finish_reason(body),
            response_id=_string_or(body.get("id"), ""),
            structured=None,
            request_metadata=request.to_dict(),
        )

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Ask for schema-constrained output and parse it.

        A provider that ignores ``response_format`` still works: the text is
        parsed, and prose around the JSON object is tolerated. A response that is
        not a JSON object is a protocol error -- never a silently accepted guess.
        """
        if request.response_schema is None:
            raise LLMProtocolError("generate_structured requires a response_schema")
        payload = self._require_configured(request)
        body = self._post(payload, request)
        text = _extract_text(body, request)
        structured = parse_json_object(text)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=_string_or(body.get("model"), request.model or self._config.model),
            usage=_extract_usage(body),
            latency_ms=0.0,
            finish_reason=_finish_reason(body),
            response_id=_string_or(body.get("id"), ""),
            structured=structured,
            request_metadata=request.to_dict(),
        )

    # -- internals -------------------------------------------------------

    def _require_configured(self, request: LLMRequest) -> dict[str, Any]:
        """Validate configuration and build the request body."""
        missing = self._config.missing()
        if missing:
            raise LLMNotConfiguredError(
                f"the custom provider is missing {', '.join(missing)}",
                provider=self.name,
            )
        payload: dict[str, Any] = {
            "model": request.model or self._config.model,
            "messages": [message.to_dict() for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name or "response",
                    "schema": dict(request.response_schema),
                    "strict": True,
                },
            }
        return payload

    def _post(self, payload: Mapping[str, Any], request: LLMRequest) -> dict[str, Any]:
        """POST the payload with bounded retries, normalising every failure."""
        timeout = request.timeout_seconds or self._config.timeout_seconds
        attempts = self._config.max_retries + 1
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.post(
                    self._config.endpoint,
                    json=dict(payload),
                    headers=self._config.headers(),
                    timeout=timeout,
                )
            except httpx.TimeoutException as error:
                last = self._fail(
                    LLMTimeoutError(
                        self._safe(f"the provider did not answer within {timeout:g}s"),
                        provider=self.name,
                    )
                )
                del error
            except httpx.HTTPError as error:
                last = self._fail(
                    LLMTransportError(
                        self._safe(f"the provider could not be reached: {type(error).__name__}"),
                        provider=self.name,
                    )
                )
            else:
                last = self._handle_status(response)
                if last is None:
                    return _decode_body(response, self._safe)
            if attempt >= attempts or not isinstance(last, LLMProviderError) or not last.retryable:
                raise last
            self._sleep(self._config.backoff_seconds * attempt)
        raise last or LLMTransportError("the provider could not be reached", provider=self.name)

    def _handle_status(self, response: httpx.Response) -> Exception | None:
        """Map an HTTP status to a normalised error (``None`` when it is fine)."""
        status = response.status_code
        if 200 <= status < 300:
            return None
        detail = self._safe(_error_detail(response))
        if status in (401, 403):
            return self._fail(
                LLMAuthenticationError(
                    f"the provider rejected the credentials (HTTP {status}): {detail}",
                    provider=self.name,
                )
            )
        if status == 429:
            return self._fail(
                LLMRateLimitError(
                    f"the provider rate limited the request (HTTP 429): {detail}",
                    provider=self.name,
                )
            )
        error = LLMProviderError(
            f"the provider returned HTTP {status}: {detail}",
            provider=self.name,
            status_code=status,
        )
        # 5xx answers are worth one more bounded attempt; 4xx answers are not.
        object.__setattr__(error, "retryable", status in _RETRYABLE_STATUS)
        return self._fail(error)

    def _fail(self, error: LLMProviderError) -> LLMProviderError:
        """Attach the provider name to a normalised failure."""
        error.provider = self.name
        return error

    def _safe(self, text: str) -> str:
        """Redact the credential from anything that came from the wire."""
        return redact_text(text, self._config.api_key)


def _decode_body(response: httpx.Response, safe: Callable[[str], str]) -> dict[str, Any]:
    """Decode a success body, turning every malformation into a protocol error."""
    try:
        payload = response.json()
    except ValueError:
        raise LLMProtocolError(
            f"the provider returned a non-JSON body: {safe(response.text[:200])}",
            provider="custom",
        ) from None
    if not isinstance(payload, dict):
        raise LLMProtocolError(
            f"the provider returned {type(payload).__name__}, expected a JSON object",
            provider="custom",
        )
    return payload


def _extract_text(body: Mapping[str, Any], request: LLMRequest) -> str:
    """Pull the assistant text out of the first choice."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProtocolError("the provider returned no choices", provider="custom")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LLMProtocolError("the provider returned a malformed choice", provider="custom")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise LLMProtocolError("the provider returned no message", provider="custom")
    content = message.get("content")
    if content is None:
        raise LLMProtocolError("the provider returned no content", provider="custom")
    if isinstance(content, list):
        # Some compatible servers return content parts instead of a string.
        parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
        content = "".join(part for part in parts if isinstance(part, str))
    if not isinstance(content, str):
        raise LLMProtocolError(
            f"the provider returned {type(content).__name__} content, expected text",
            provider="custom",
        )
    del request
    return content


def _extract_usage(body: Mapping[str, Any]) -> LLMUsage | None:
    """Return provider-reported usage, or ``None`` when it was not sent.

    A missing block means "unknown", never "zero": inventing zeros would be a
    fabricated measurement.
    """
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None
    reported = LLMUsage.from_mapping(usage)
    if (
        reported.prompt_tokens is None
        and reported.completion_tokens is None
        and reported.total_tokens is None
    ):
        return None
    return reported


def _finish_reason(body: Mapping[str, Any]) -> str | None:
    """The finish reason of the first choice, when present."""
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None


def _string_or(value: Any, default: str) -> str:
    """Return ``value`` when it is a non-empty string, else ``default``."""
    return value if isinstance(value, str) and value else default


def _error_detail(response: httpx.Response) -> str:
    """A short, useful excerpt of an error body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str):
                return message[:200]
        message = payload.get("message")
        if isinstance(message, str):
            return message[:200]
    return str(payload)[:200]


def with_overrides(config: CustomLLMConfig, **changes: Any) -> CustomLLMConfig:
    """Return a copy of a configuration with fields replaced (tests, demos)."""
    return replace(config, **changes)


__all__ = [
    "DEFAULT_CHAT_PATH",
    "CustomLLMConfig",
    "CustomLLMProvider",
    "with_overrides",
]
