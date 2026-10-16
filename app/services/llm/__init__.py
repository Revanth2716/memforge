"""LLM providers (M5).

Two implementations, one interface:

* :class:`~app.services.llm.mock_provider.MockLLMProvider` — the default. Fully
  offline, deterministic, no credential, no network.
* :class:`~app.services.llm.custom_provider.CustomLLMProvider` — an HTTP client
  for an OpenAI-compatible chat-completions API, configured entirely through
  environment variables.

No vendor name is hard-coded anywhere in this package: the base URL, the path,
the model and the header shape are all configuration, and the default wire format
is the widely adopted ``/chat/completions`` contract rather than a product.
"""

from __future__ import annotations

from app.services.llm.custom_provider import CustomLLMConfig, CustomLLMProvider
from app.services.llm.factory import build_llm_provider
from app.services.llm.mock_provider import MockLLMProvider

__all__ = [
    "CustomLLMConfig",
    "CustomLLMProvider",
    "MockLLMProvider",
    "build_llm_provider",
]
