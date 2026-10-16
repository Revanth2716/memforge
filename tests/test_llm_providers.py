"""M5 provider tests: the mock provider and the custom HTTP provider.

Every HTTP test uses an injected ``httpx.MockTransport`` -- **no live external
call is ever made**, and no credential ever leaves the process.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest

from app.config import Settings
from app.domain.errors import (
    LLMAuthenticationError,
    LLMNotConfiguredError,
    LLMProtocolError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransportError,
)
from app.domain.llm import LLMMessage, LLMRequest
from app.services.llm.custom_provider import (
    DEFAULT_CHAT_PATH,
    CustomLLMConfig,
    CustomLLMProvider,
    with_overrides,
)
from app.services.llm.factory import build_llm_provider, describe_provider
from app.services.llm.mock_provider import MockCandidate, MockLLMProvider

SECRET = "sk-do-not-log-me-9876543210"
BASE_URL = "https://llm.example.invalid/v1"


def config(**overrides: Any) -> CustomLLMConfig:
    """A fully configured custom-provider config."""
    base = {
        "base_url": BASE_URL,
        "api_key": SECRET,
        "model": "test-model",
        "max_retries": 0,
        "backoff_seconds": 0.0,
    }
    base.update(overrides)
    return CustomLLMConfig(**base)  # type: ignore[arg-type]


def completion(content: str = "hello", **extra: Any) -> dict[str, Any]:
    """A minimal compatible chat-completions response."""
    body: dict[str, Any] = {
        "id": "resp-1",
        "model": "test-model",
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }
    body.update(extra)
    return body


class Recorder:
    """Captures the requests a provider makes, for assertions."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def transport(self, handler: Any) -> httpx.MockTransport:
        """Build a mock transport that records and delegates to ``handler``."""

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return cast("httpx.Response", handler(request))

        return httpx.MockTransport(wrapped)

    @property
    def last(self) -> httpx.Request:
        """The most recent request."""
        return self.requests[-1]

    def body(self) -> dict[str, Any]:
        """The JSON body of the most recent request."""
        return cast("dict[str, Any]", json.loads(self.last.content.decode()))


def provider_with(
    handler: Any, *, recorder: Recorder | None = None, **overrides: Any
) -> tuple[CustomLLMProvider, Recorder]:
    """Build a custom provider over a mock transport."""
    recorder = recorder or Recorder()
    client = httpx.Client(transport=recorder.transport(handler))
    return CustomLLMProvider(config(**overrides), client=client, sleep=lambda _: None), recorder


# -- the mock provider --------------------------------------------------------


class TestMockProvider:
    def test_is_always_configured(self) -> None:
        assert MockLLMProvider().is_configured is True

    def test_is_deterministic(self) -> None:
        provider = MockLLMProvider()
        request = LLMRequest(messages=(LLMMessage.user("hello there"),))
        assert provider.generate(request).text == provider.generate(request).text

    def test_reports_no_usage(self) -> None:
        response = MockLLMProvider().generate(LLMRequest(messages=(LLMMessage.user("x"),)))
        assert response.usage is None
        assert response.has_usage is False

    def test_identifies_itself_as_a_stand_in(self) -> None:
        text = MockLLMProvider().generate(LLMRequest(messages=(LLMMessage.user("x"),))).text
        assert "no model was called" in text
        assert MockLLMProvider().name == "mock"

    def test_candidates_from_the_documented_convention(self) -> None:
        response = MockLLMProvider().generate_structured(
            LLMRequest(
                messages=(
                    LLMMessage.user(
                        "Please help.\nremember: [FACT] The user ships on Fridays.\n"
                        "remember: The user likes tea."
                    ),
                ),
                response_schema={"type": "object"},
            )
        )
        candidates = response.structured or {}
        assert [item["content"] for item in candidates["candidates"]] == [
            "The user ships on Fridays.",
            "The user likes tea.",
        ]
        assert candidates["candidates"][0]["memory_type"] == "FACT"
        assert candidates["candidates"][1]["memory_type"] == "OBSERVATION"

    def test_no_convention_means_no_candidates(self) -> None:
        response = MockLLMProvider().generate_structured(
            LLMRequest(
                messages=(LLMMessage.user("just a question"),),
                response_schema={"type": "object"},
            )
        )
        assert response.structured == {"candidates": []}

    def test_fixed_candidates_can_be_injected(self) -> None:
        provider = MockLLMProvider(candidates=[MockCandidate("injected", "FACT", 0.9, 0.8, "test")])
        response = provider.generate_structured(
            LLMRequest(messages=(LLMMessage.user("x"),), response_schema={"type": "object"})
        )
        payload = response.structured or {}
        assert payload["candidates"][0]["content"] == "injected"

    def test_structured_without_a_schema_is_refused(self) -> None:
        from app.domain.errors import LLMProtocolError

        with pytest.raises(LLMProtocolError, match="response_schema"):
            MockLLMProvider().generate_structured(LLMRequest(messages=(LLMMessage.user("x"),)))

    def test_describe_mentions_no_credentials(self) -> None:
        assert "mock" in MockLLMProvider().describe()


# -- the custom provider: request formation -----------------------------------


class TestRequestFormation:
    def test_posts_to_the_configured_endpoint(self) -> None:
        provider, recorder = provider_with(lambda request: httpx.Response(200, json=completion()))
        provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

        assert str(recorder.last.url) == f"{BASE_URL}{DEFAULT_CHAT_PATH}"
        assert recorder.last.method == "POST"

    def test_path_is_configurable(self) -> None:
        provider, recorder = provider_with(
            lambda request: httpx.Response(200, json=completion()), path="/v2/generate"
        )
        provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert str(recorder.last.url) == f"{BASE_URL}/v2/generate"

    def test_body_matches_the_compatible_contract(self) -> None:
        provider, recorder = provider_with(lambda request: httpx.Response(200, json=completion()))
        provider.generate(
            LLMRequest(
                messages=(LLMMessage.system("be brief"), LLMMessage.user("hi")),
                temperature=0.0,
                max_tokens=64,
            )
        )
        body = recorder.body()
        assert body["model"] == "test-model"
        assert body["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 64

    def test_unset_options_are_omitted(self) -> None:
        provider, recorder = provider_with(lambda request: httpx.Response(200, json=completion()))
        provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        body = recorder.body()
        assert "temperature" not in body
        assert "max_tokens" not in body
        assert "response_format" not in body

    def test_structured_request_carries_the_schema(self) -> None:
        provider, recorder = provider_with(
            lambda request: httpx.Response(200, json=completion('{"ok": true}'))
        )
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        provider.generate_structured(
            LLMRequest(
                messages=(LLMMessage.user("hi"),),
                response_schema=schema,
                schema_name="status",
            )
        )
        response_format = recorder.body()["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "status"
        assert response_format["json_schema"]["schema"] == schema
        assert response_format["json_schema"]["strict"] is True

    def test_model_override_is_honoured(self) -> None:
        provider, recorder = provider_with(lambda request: httpx.Response(200, json=completion()))
        provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),), model="other-model"))
        assert recorder.body()["model"] == "other-model"


# -- authentication -----------------------------------------------------------


class TestAuthentication:
    def test_bearer_token_is_sent(self) -> None:
        provider, recorder = provider_with(lambda request: httpx.Response(200, json=completion()))
        provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert recorder.last.headers["authorization"] == f"Bearer {SECRET}"
        assert recorder.last.headers["content-type"] == "application/json"

    def test_authentication_can_be_disabled(self) -> None:
        provider, recorder = provider_with(
            lambda request: httpx.Response(200, json=completion()), send_authorization=False
        )
        provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert "authorization" not in recorder.last.headers

    def test_missing_key_is_a_not_configured_error(self) -> None:
        provider = CustomLLMProvider(config(api_key=""))
        assert provider.is_configured is False
        with pytest.raises(LLMNotConfiguredError, match="api_key"):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_rejected_credentials_are_normalised(self) -> None:
        provider, _ = provider_with(
            lambda request: httpx.Response(401, json={"error": {"message": "bad token"}})
        )
        with pytest.raises(LLMAuthenticationError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert raised.value.retryable is False
        assert raised.value.kind == "authentication"

    def test_forbidden_is_also_an_authentication_failure(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(403, json={}))
        with pytest.raises(LLMAuthenticationError):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_the_key_is_absent_from_reprs_and_descriptions(self) -> None:
        settings = CustomLLMConfig(base_url=BASE_URL, api_key=SECRET, model="m")
        assert SECRET not in repr(settings)
        assert SECRET not in repr(CustomLLMProvider(settings))
        assert SECRET not in str(settings.describe())
        assert settings.describe()["api_key_present"] is True
        assert settings.safe_headers()["authorization"] != f"Bearer {SECRET}"

    def test_a_key_echoed_by_the_server_never_reaches_an_error(self) -> None:
        provider, _ = provider_with(
            lambda request: httpx.Response(
                400, json={"error": {"message": f"token {SECRET} is invalid"}}
            )
        )
        with pytest.raises(LLMProviderError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert SECRET not in str(raised.value)

    def test_a_key_in_a_url_is_stripped_from_the_endpoint(self) -> None:
        settings = CustomLLMConfig(
            base_url="https://user:password@llm.example.invalid/v1", model="m"
        )
        assert "password" not in settings.safe_base_url


# -- failures -----------------------------------------------------------------


class TestFailures:
    def test_timeout_is_normalised(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        provider, _ = provider_with(timeout)
        with pytest.raises(LLMTimeoutError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert raised.value.retryable is True
        assert raised.value.kind == "timeout"

    def test_connection_failure_is_normalised(self) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        provider, _ = provider_with(fail)
        with pytest.raises(LLMTransportError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert raised.value.retryable is True

    def test_rate_limit_is_normalised(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(429, json={}))
        with pytest.raises(LLMRateLimitError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert raised.value.retryable is True

    def test_server_error_is_normalised_and_carries_the_status(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(503, json={}))
        with pytest.raises(LLMProviderError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert raised.value.status_code == 503
        assert raised.value.retryable is True

    def test_client_error_is_not_retryable(self) -> None:
        provider, _ = provider_with(
            lambda request: httpx.Response(400, json={"error": {"message": "bad request"}})
        )
        with pytest.raises(LLMProviderError) as raised:
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert raised.value.status_code == 400
        assert raised.value.retryable is False

    def test_non_json_body_is_a_protocol_error(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, text="<html>nope</html>"))
        with pytest.raises(LLMProtocolError, match="non-JSON"):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_json_array_body_is_a_protocol_error(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json=[1, 2, 3]))
        with pytest.raises(LLMProtocolError, match="expected a JSON object"):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_missing_choices_is_a_protocol_error(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json={"id": "x"}))
        with pytest.raises(LLMProtocolError, match="no choices"):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_missing_content_is_a_protocol_error(self) -> None:
        provider, _ = provider_with(
            lambda request: httpx.Response(200, json={"choices": [{"message": {}}]})
        )
        with pytest.raises(LLMProtocolError, match="no content"):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_content_parts_are_joined(self) -> None:
        body = {
            "choices": [
                {
                    "message": {"content": [{"text": "one "}, {"text": "two"}]},
                    "finish_reason": "stop",
                }
            ]
        }
        provider, _ = provider_with(lambda request: httpx.Response(200, json=body))
        assert provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),))).text == "one two"

    def test_structured_output_that_is_not_json_is_a_protocol_error(self) -> None:
        provider, _ = provider_with(
            lambda request: httpx.Response(200, json=completion("I cannot do that"))
        )
        with pytest.raises(LLMProtocolError):
            provider.generate_structured(
                LLMRequest(messages=(LLMMessage.user("hi"),), response_schema={"type": "object"})
            )

    def test_structured_output_requires_a_schema(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json=completion()))
        with pytest.raises(LLMProtocolError, match="response_schema"):
            provider.generate_structured(LLMRequest(messages=(LLMMessage.user("hi"),)))

    def test_incomplete_configuration_is_reported(self) -> None:
        provider = CustomLLMProvider(CustomLLMConfig(model="m"))
        with pytest.raises(LLMNotConfiguredError, match="base_url"):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))


# -- bounded retries ----------------------------------------------------------


class TestRetries:
    def test_retryable_failure_is_retried_within_the_budget(self) -> None:
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503, json={})
            return httpx.Response(200, json=completion())

        provider, _ = provider_with(flaky, max_retries=2)
        response = provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert response.text == "hello"
        assert calls["n"] == 3

    def test_retries_are_bounded(self) -> None:
        calls = {"n": 0}

        def always_down(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={})

        provider, _ = provider_with(always_down, max_retries=2)
        with pytest.raises(LLMProviderError):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert calls["n"] == 3  # one attempt + two retries, never more

    def test_zero_retries_means_one_attempt(self) -> None:
        calls = {"n": 0}

        def always_down(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={})

        provider, _ = provider_with(always_down, max_retries=0)
        with pytest.raises(LLMProviderError):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert calls["n"] == 1

    def test_client_errors_are_never_retried(self) -> None:
        calls = {"n": 0}

        def rejected(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={})

        provider, _ = provider_with(rejected, max_retries=3)
        with pytest.raises(LLMProviderError):
            provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert calls["n"] == 1

    def test_timeouts_are_retried(self) -> None:
        calls = {"n": 0}

        def slow(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(200, json=completion())

        provider, _ = provider_with(slow, max_retries=1)
        assert provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),))).text == "hello"

    def test_backoff_is_injected_not_slept(self) -> None:
        slept: list[float] = []
        provider, _ = provider_with(
            lambda request: httpx.Response(503, json={}),
            recorder=None,
            max_retries=2,
            backoff_seconds=0.5,
        )
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503, json={}))
        )
        retrying = CustomLLMProvider(
            config(max_retries=2, backoff_seconds=0.5),
            client=client,
            sleep=slept.append,
        )
        del provider
        with pytest.raises(LLMProviderError):
            retrying.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert slept == [0.5, 1.0]

    def test_negative_retry_configuration_is_rejected(self) -> None:
        with pytest.raises(LLMNotConfiguredError, match="max_retries"):
            config(max_retries=-1)


# -- usage --------------------------------------------------------------------


class TestUsage:
    def test_reported_usage_is_echoed(self) -> None:
        body = completion(usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})
        provider, _ = provider_with(lambda request: httpx.Response(200, json=body))
        usage = provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),))).usage

        assert usage is not None
        assert usage.prompt_tokens == 11
        assert usage.total_tokens == 15
        assert usage.reported_by == "provider"

    def test_absent_usage_is_none_not_zero(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json=completion()))
        assert provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),))).usage is None

    def test_empty_usage_block_is_none(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json=completion(usage={})))
        assert provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),))).usage is None

    def test_estimates_are_never_substituted(self) -> None:
        from app.domain.tokens import estimate_tokens

        provider, _ = provider_with(lambda request: httpx.Response(200, json=completion()))
        response = provider.generate(
            LLMRequest(messages=(LLMMessage.user("a much longer prompt to estimate"),))
        )
        # The provider said nothing about usage, so MemForge does not guess --
        # it has its own estimator, but that is a different, labelled figure.
        assert response.usage is None
        assert estimate_tokens("x").estimated_tokens >= 1


class TestResponseMetadata:
    def test_response_id_model_and_finish_reason_are_read(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json=completion()))
        response = provider.generate(LLMRequest(messages=(LLMMessage.user("hi"),)))
        assert response.response_id == "resp-1"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"

    def test_request_metadata_carries_no_content(self) -> None:
        provider, _ = provider_with(lambda request: httpx.Response(200, json=completion()))
        response = provider.generate(LLMRequest(messages=(LLMMessage.user(SECRET),)))
        assert SECRET not in str(response.request_metadata)
        assert response.request_metadata["messages"] == 1


# -- configuration and factory ------------------------------------------------


class TestConfig:
    def test_endpoint_joins_base_url_and_path(self) -> None:
        assert config(base_url="https://host/v1/").endpoint == "https://host/v1/chat/completions"

    def test_path_must_be_absolute(self) -> None:
        with pytest.raises(LLMNotConfiguredError, match="must start with"):
            config(path="chat/completions")

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(LLMNotConfiguredError, match="timeout_seconds"):
            config(timeout_seconds=0)

    def test_missing_lists_every_gap(self) -> None:
        assert CustomLLMConfig().missing() == ("base_url", "model", "api_key")

    def test_overrides_produce_a_copy(self) -> None:
        original = config()
        updated = with_overrides(original, model="other")
        assert original.model == "test-model"
        assert updated.model == "other"


class TestFactory:
    def test_mock_is_the_default(self) -> None:
        provider = build_llm_provider(Settings())
        assert isinstance(provider, MockLLMProvider)
        assert provider.name == "mock"

    def test_custom_requires_complete_configuration(self) -> None:
        settings = Settings(llm_provider="custom", llm_base_url=BASE_URL)
        # No model and no key -> falls back to the offline default.
        assert isinstance(build_llm_provider(settings), MockLLMProvider)

    def test_custom_is_built_when_complete(self) -> None:
        settings = Settings(
            llm_provider="custom",
            llm_base_url=BASE_URL,
            llm_model="test-model",
            llm_api_key=SECRET,
        )
        provider = build_llm_provider(settings)
        assert isinstance(provider, CustomLLMProvider)
        assert provider.model == "test-model"
        provider.close()

    def test_describe_provider_is_secret_free(self) -> None:
        settings = Settings(
            llm_provider="custom",
            llm_base_url=BASE_URL,
            llm_model="test-model",
            llm_api_key=SECRET,
        )
        provider = build_llm_provider(settings)
        assert isinstance(provider, CustomLLMProvider)
        described = describe_provider(provider)
        provider.close()

        assert described["provider"] == "custom"
        assert described["model"] == "test-model"
        assert SECRET not in str(described)

    def test_a_key_in_the_base_url_is_stripped_from_the_description(self) -> None:
        settings = Settings(
            llm_provider="custom",
            llm_base_url="https://user:pw@llm.example.invalid/v1",
            llm_model="m",
            llm_api_key=SECRET,
        )
        provider = build_llm_provider(settings)
        assert isinstance(provider, CustomLLMProvider)
        described = describe_provider(provider)
        provider.close()
        assert "pw" not in str(described["endpoint"])


@pytest.fixture
def no_network() -> Iterator[None]:
    """Fail loudly if any code tries to open a real socket during the suite."""
    original = httpx.Client.send

    def guarded(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        raise AssertionError(f"a live HTTP call was attempted: {request.url}")

    httpx.Client.send = guarded  # type: ignore[method-assign]
    try:
        yield
    finally:
        httpx.Client.send = original  # type: ignore[method-assign]
