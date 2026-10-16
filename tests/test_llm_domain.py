"""M5 LLM domain tests: interface, values, parsing and secret redaction.

Everything here is pure, so the whole provider contract is provable without an
HTTP client, a server or a credential.
"""

from __future__ import annotations

import pytest

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
from app.domain.llm import (
    PROVIDER_ERROR_KINDS,
    REDACTED,
    SECRET_HEADERS,
    LLMMessage,
    LLMMessageRole,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    message_text,
    parse_json_object,
    redact_mapping,
    redact_text,
    safe_endpoint,
)
from app.services.llm.mock_provider import MockLLMProvider

SECRET = "sk-super-secret-value-1234"


class TestMessages:
    def test_roles_serialise_to_the_wire_shape(self) -> None:
        assert LLMMessage.system("a").to_dict() == {"role": "system", "content": "a"}
        assert LLMMessage.user("b").to_dict() == {"role": "user", "content": "b"}
        assert LLMMessage.assistant("c").to_dict() == {"role": "assistant", "content": "c"}

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(LLMProtocolError, match="role"):
            LLMMessage("root", "content")  # type: ignore[arg-type]

    def test_non_string_content_is_rejected(self) -> None:
        with pytest.raises(LLMProtocolError, match="string"):
            LLMMessage(LLMMessageRole.USER, 42)  # type: ignore[arg-type]

    def test_message_text_joins_contents(self) -> None:
        joined = message_text([LLMMessage.system("one"), LLMMessage.user("two")])
        assert joined == "one\n\ntwo"


class TestRequest:
    def test_metadata_is_shape_only(self) -> None:
        request = LLMRequest(messages=(LLMMessage.user(SECRET),), temperature=0.0, operation="test")
        metadata = request.to_dict()

        assert metadata["messages"] == 1
        assert metadata["characters"] == len(SECRET)
        assert metadata["structured"] is False
        assert metadata["operation"] == "test"
        # The metadata is for logging: it must not carry the prompt content.
        assert SECRET not in str(metadata)

    def test_empty_request_is_rejected(self) -> None:
        with pytest.raises(LLMProtocolError, match="at least one message"):
            LLMRequest(messages=())

    @pytest.mark.parametrize(
        ("field", "value"),
        [("temperature", 2.5), ("temperature", -0.1), ("max_tokens", 0), ("timeout_seconds", 0)],
    )
    def test_bounds_are_validated(self, field: str, value: float) -> None:
        with pytest.raises(LLMProtocolError):
            LLMRequest(messages=(LLMMessage.user("x"),), **{field: value})  # type: ignore[arg-type]

    def test_structured_intent(self) -> None:
        plain = LLMRequest(messages=(LLMMessage.user("x"),))
        structured = LLMRequest(
            messages=(LLMMessage.user("x"),), response_schema={"type": "object"}
        )
        assert plain.wants_structured_output is False
        assert structured.wants_structured_output is True


class TestUsage:
    def test_absent_usage_stays_none(self) -> None:
        assert LLMResponse(text="hi", provider="mock", model="m").has_usage is False

    def test_reported_usage_is_kept_verbatim(self) -> None:
        usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert usage.to_dict()["total_tokens"] == 15
        assert usage.reported_by == "provider"

    def test_malformed_fields_become_none_not_zero(self) -> None:
        usage = LLMUsage.from_mapping({"prompt_tokens": "many", "total_tokens": -4})
        assert usage.prompt_tokens is None
        assert usage.total_tokens is None

    def test_float_counts_are_coerced(self) -> None:
        assert LLMUsage.from_mapping({"prompt_tokens": 12.0}).prompt_tokens == 12


class TestJsonParsing:
    def test_plain_object(self) -> None:
        assert parse_json_object('{"ok": true}') == {"ok": True}

    def test_fenced_and_wrapped_objects_are_tolerated(self) -> None:
        assert parse_json_object('Sure!\n```json\n{"ok": true}\n```') == {"ok": True}

    def test_empty_output_is_a_protocol_error(self) -> None:
        with pytest.raises(LLMProtocolError, match="empty"):
            parse_json_object("   ")

    def test_non_json_is_a_protocol_error(self) -> None:
        with pytest.raises(LLMProtocolError, match="malformed JSON"):
            parse_json_object("{not json at all}")

    def test_prose_without_an_object_is_rejected(self) -> None:
        with pytest.raises(LLMProtocolError, match="did not return a JSON object"):
            parse_json_object("I cannot answer that.")

    def test_json_without_an_object_is_rejected(self) -> None:
        with pytest.raises(LLMProtocolError, match="JSON object"):
            parse_json_object("[1, 2, 3]")

    def test_top_level_array_is_rejected(self) -> None:
        with pytest.raises(LLMProtocolError):
            parse_json_object("[]")


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        "error_type",
        [
            LLMNotConfiguredError,
            LLMTimeoutError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTransportError,
            LLMProtocolError,
            LLMProviderError,
        ],
    )
    def test_every_provider_failure_is_one_normalised_type(self, error_type: type) -> None:
        error = error_type("boom", provider="custom")
        assert isinstance(error, LLMProviderError)
        assert isinstance(error, LLMError)
        assert error.provider == "custom"
        assert error.kind in PROVIDER_ERROR_KINDS
        assert isinstance(error.retryable, bool)

    @pytest.mark.parametrize(
        ("error_type", "retryable"),
        [
            (LLMTimeoutError, True),
            (LLMTransportError, True),
            (LLMRateLimitError, True),
            (LLMAuthenticationError, False),
            (LLMNotConfiguredError, False),
            (LLMProtocolError, False),
        ],
    )
    def test_only_transient_failures_are_retryable(
        self, error_type: type[LLMProviderError], retryable: bool
    ) -> None:
        assert error_type("x").retryable is retryable

    def test_status_error_records_the_status(self) -> None:
        error = LLMProviderError("bad", provider="custom", status_code=500)
        assert error.status_code == 500


class TestRedaction:
    def test_secrets_are_replaced_in_text(self) -> None:
        redacted = redact_text(f"Authorization: Bearer {SECRET}", SECRET)
        assert SECRET not in redacted
        assert REDACTED in redacted

    def test_short_values_are_not_redacted(self) -> None:
        # Redacting a 1-character "secret" would mangle unrelated text.
        assert redact_text("abc", "a") == "abc"

    def test_auth_headers_are_dropped(self) -> None:
        safe = redact_mapping({"Authorization": f"Bearer {SECRET}", "content-type": "json"}, SECRET)
        assert safe["Authorization"] == REDACTED
        assert safe["content-type"] == "json"

    def test_nested_structures_are_redacted(self) -> None:
        safe = redact_mapping(
            {
                "headers": {"x-api-key": SECRET, "accept": "json"},
                "body": {"prompt": f"use {SECRET}"},
                "messages": [f"key={SECRET}", 7],
            },
            SECRET,
        )
        rendered = str(safe)
        assert SECRET not in rendered
        assert safe["messages"][1] == 7

    def test_secret_header_names_are_known(self) -> None:
        assert "authorization" in SECRET_HEADERS
        assert "x-api-key" in SECRET_HEADERS
        assert "cookie" in SECRET_HEADERS

    def test_userinfo_is_stripped_from_endpoints(self) -> None:
        assert safe_endpoint("https://user:pw@host/v1") == "https://host/v1"
        assert safe_endpoint("https://host/v1") == "https://host/v1"


class TestProviderInterface:
    def test_default_structured_generation_parses_text(self) -> None:
        response = _FixedProvider().generate_structured(
            LLMRequest(messages=(LLMMessage.user("x"),), response_schema={"type": "object"})
        )
        assert response.structured == {"ok": True}

    def test_default_structured_generation_needs_a_schema(self) -> None:
        with pytest.raises(LLMProtocolError, match="response_schema"):
            _FixedProvider().generate_structured(LLMRequest(messages=(LLMMessage.user("x"),)))

    def test_abstract_provider_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_repr_never_shows_internals(self) -> None:
        assert "MockLLMProvider" in repr(MockLLMProvider())


class _FixedProvider(LLMProvider):
    """A provider that only implements the abstract surface."""

    name = "fixed"
    model = "fixed-v1"

    @property
    def is_configured(self) -> bool:
        return True

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text='{"ok": true}', provider=self.name, model=self.model)
