"""M5 configuration tests.

``tests/test_config.py`` and ``tests/test_m4_config.py`` cover earlier
milestones and are left untouched; the provider settings are proved here,
including the rule that a credential comes from the environment and nothing else.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import LLM_PROVIDERS, Settings
from app.domain.errors import LLMNotConfiguredError
from app.services.llm.custom_provider import DEFAULT_CHAT_PATH, CustomLLMConfig

SECRET = "sk-config-test-key-abcdef123456"


class TestDefaults:
    def test_mock_is_the_default_provider(self) -> None:
        settings = Settings()
        assert settings.llm_provider == "mock"
        assert settings.llm_base_url == ""
        assert settings.llm_api_key == ""
        assert settings.llm_model == ""
        assert settings.mock_llm_model == "mock-echo-v1"

    def test_the_default_path_needs_no_network(self) -> None:
        settings = Settings()
        assert settings.custom_llm_config().is_configured() is False
        assert settings.llm_chat_path == DEFAULT_CHAT_PATH

    def test_provider_names_are_closed(self) -> None:
        assert {"mock", "custom"} == LLM_PROVIDERS


class TestValidation:
    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="llm_provider"):
            Settings(llm_provider="openai")

    def test_custom_without_a_base_url_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="llm_base_url"):
            Settings(llm_provider="custom")

    def test_custom_with_a_base_url_is_accepted(self) -> None:
        settings = Settings(llm_provider="custom", llm_base_url="https://host/v1")
        assert settings.custom_llm_config().base_url == "https://host/v1"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"llm_timeout_seconds": 0},
            {"llm_max_retries": -1},
            {"llm_max_retries": 99},
            {"llm_retry_backoff_seconds": -1.0},
            {"llm_chat_path": "chat/completions"},
        ],
    )
    def test_invalid_provider_settings_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            Settings(**kwargs)  # type: ignore[arg-type]


class TestFactory:
    def test_environment_produces_a_custom_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMFORGE_LLM_PROVIDER", "custom")
        monkeypatch.setenv("MEMFORGE_LLM_BASE_URL", "https://host/v1")
        monkeypatch.setenv("MEMFORGE_LLM_MODEL", "a-model")
        monkeypatch.setenv("MEMFORGE_LLM_API_KEY", SECRET)
        monkeypatch.setenv("MEMFORGE_LLM_TIMEOUT_SECONDS", "12.5")
        monkeypatch.setenv("MEMFORGE_LLM_MAX_RETRIES", "1")

        settings = Settings()
        config = settings.custom_llm_config()
        assert config.is_configured() is True
        assert config.model == "a-model"
        assert config.timeout_seconds == 12.5
        assert config.max_retries == 1
        assert config.endpoint == "https://host/v1/chat/completions"

    def test_a_partial_environment_degrades_to_the_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMFORGE_LLM_PROVIDER", "custom")
        monkeypatch.setenv("MEMFORGE_LLM_BASE_URL", "https://host/v1")
        monkeypatch.setenv("MEMFORGE_LLM_MODEL", "a-model")
        monkeypatch.delenv("MEMFORGE_LLM_API_KEY", raising=False)

        settings = Settings()
        config = settings.custom_llm_config()
        assert config.is_configured() is False
        assert config.missing() == ("api_key",)

    def test_the_key_is_excluded_from_the_settings_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMFORGE_LLM_API_KEY", SECRET)
        settings = Settings()
        assert SECRET not in repr(settings)

    def test_the_key_is_excluded_from_the_config_repr(self) -> None:
        config = CustomLLMConfig(base_url="https://host/v1", model="m", api_key=SECRET)
        assert SECRET not in repr(config)

    def test_authentication_can_be_disabled(self) -> None:
        config = CustomLLMConfig(base_url="https://host/v1", model="m", send_authorization=False)
        assert config.is_configured() is True
        assert "authorization" not in config.headers()


class TestConfigValidation:
    def test_invalid_config_is_rejected_at_construction(self) -> None:
        with pytest.raises(LLMNotConfiguredError):
            CustomLLMConfig(base_url="https://host/v1", model="m", timeout_seconds=0)

    def test_describe_never_includes_the_key(self) -> None:
        config = CustomLLMConfig(base_url="https://user:pw@host/v1", model="m", api_key=SECRET)
        described = str(config.describe())
        assert SECRET not in described
        assert "user:pw" not in described
        assert described != ""
