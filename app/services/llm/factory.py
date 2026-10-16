"""Provider selection: the mock is the default, always.

Nothing here reaches the network, and nothing here reads a credential from a
file, a database or a request. The provider is chosen from
:class:`app.config.Settings`, i.e. from the environment, and the custom provider
is only constructed when it is explicitly selected *and* fully configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.llm import LLMProvider
from app.services.llm.custom_provider import CustomLLMProvider
from app.services.llm.mock_provider import MockLLMProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.config import Settings


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Build the provider described by ``settings``.

    The mock provider is returned for ``llm_provider`` in ``{None, "", "mock"}``
    and whenever the custom provider is selected but incompletely configured --
    a misconfiguration therefore degrades to the offline default instead of
    failing at startup with an opaque error.
    """
    if settings.llm_provider == "custom" and settings.custom_llm_config().is_configured():
        return CustomLLMProvider(settings.custom_llm_config())
    return MockLLMProvider(model=settings.mock_llm_model)


def describe_provider(provider: LLMProvider) -> dict[str, str | bool | None]:
    """Secret-free evidence about a provider instance."""
    described: dict[str, str | bool | None] = {
        "provider": provider.name,
        "model": provider.model,
        "configured": provider.is_configured,
    }
    if isinstance(provider, CustomLLMProvider):
        described["endpoint"] = provider.config.safe_base_url
    return described


__all__ = ["build_llm_provider", "describe_provider"]
