"""Application configuration.

Configuration is environment driven (``MEMFORGE_`` prefix) and validated by
pydantic-settings. The defaults keep MemForge fully local: a SQLite file under
``<repo>/data/``.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.context import ContextPolicy
from app.domain.embeddings import EmbeddingProvider, build_provider
from app.domain.intelligence import IntelligenceConfig
from app.domain.retrieval import RankingConfig
from app.domain.tokens import TokenEstimatorConfig
from app.services.llm.custom_provider import DEFAULT_CHAT_PATH, CustomLLMConfig

#: Repository root (``<repo>``) derived from this file's location.
#: This file lives at ``<repo>/app/config.py`` -> parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Selectable providers. ``mock`` is the default everywhere.
LLM_PROVIDERS: frozenset[str] = frozenset({"mock", "custom"})


class Settings(BaseSettings):
    """Runtime settings for the MemForge API and CLI."""

    model_config = SettingsConfigDict(
        env_prefix="MEMFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MemForge"
    #: Root path for the REST API. Health is always served at ``/health``.
    api_prefix: str = "/api/v1"
    #: SQLite database file. Relative paths are resolved from the current
    #: working directory; the default lives inside the repository.
    database_path: Path = PROJECT_ROOT / "data" / "memforge.sqlite"
    log_level: str = "INFO"

    # -- M2: deterministic intelligence knobs ------------------------------
    #: Each of these feeds a documented, testable algorithm (see
    #: ``app/domain/intelligence.py`` and the README). The scoring weights
    #: themselves live in ``IntelligenceConfig``.

    #: Days for the recency/decay factor of a memory to halve.
    decay_half_life_days: float = Field(default=30.0, gt=0.0)
    #: Dice similarity at/above which two memories are near-duplicates.
    similarity_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    #: Value similarity below which two values of the same attribute conflict.
    value_similarity_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    #: Near-duplicates that saturate the recurrence factor.
    recurrence_saturation: int = Field(default=3, ge=1)
    #: Recorded accesses that saturate the access factor.
    access_saturation: int = Field(default=5, ge=1)

    # -- M3: retrieval -----------------------------------------------------

    #: Retrieval strategy used when a request does not say otherwise.
    default_retrieval_mode: str = "lexical"
    #: Number of results returned by default.
    default_top_k: int = Field(default=10, ge=1, le=100)
    #: Blend weight of the semantic component in ``hybrid`` mode.
    default_semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Optional embedding provider: none (default), ``hashed`` or ``ollama``.
    embedding_provider: str | None = None
    #: Dimensions used by the model-free ``hashed`` provider.
    embedding_dimensions: int = Field(default=256, ge=8)
    #: Ollama settings (only used when ``embedding_provider=ollama``).
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_endpoint: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = Field(default=5.0, gt=0)

    # -- M4: context engineering ------------------------------------------

    #: Estimated tokens a caller gets when the request does not name a budget.
    default_context_token_budget: int = Field(default=1200, ge=1)
    #: Largest estimated budget a single compilation may request.
    max_context_token_budget: int = Field(default=16000, ge=1)
    #: Retrieval candidates handed to the compiler by default.
    context_top_k: int = Field(default=20, ge=1, le=100)

    # Utility weights: they must sum to 1.0 (validated by ``ContextPolicy``).
    context_relevance_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    context_priority_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    context_confidence_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    context_freshness_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    context_type_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Strength of the token-cost penalty in the utility model.
    context_cost_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Token cost at which that penalty reaches full strength.
    context_cost_reference_tokens: int = Field(default=120, ge=1)
    #: Candidates below this utility are never selected.
    context_min_candidate_score: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Soft per-block ceiling for deterministic sentence selection (``None``
    #: keeps every sentence).
    context_max_block_tokens: int | None = Field(default=None, ge=1)

    # Token estimator (documented heuristic, never a provider tokenizer).
    token_chars_per_token: float = Field(default=4.0, gt=0.0)
    token_words_per_token: float = Field(default=1.3, gt=0.0)
    token_non_ascii_tokens_per_char: float = Field(default=1.0, gt=0.0)

    # -- M5: language model -----------------------------------------------
    # Everything about the provider comes from the environment. The API key is
    # read from ``MEMFORGE_LLM_API_KEY`` and is never written to the database,
    # a log record, a response body or a repr.

    #: ``mock`` (the default, fully offline) or ``custom``.
    llm_provider: str = "mock"
    #: Base URL of an OpenAI-compatible API, e.g. ``https://host/v1``.
    llm_base_url: str = ""
    #: Bearer credential. Environment only -- never persisted, never logged.
    llm_api_key: str = Field(default="", repr=False)
    llm_model: str = ""
    #: Path of the compatible chat-completions contract.
    llm_chat_path: str = "/chat/completions"
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0)
    #: Extra attempts after the first one; 0 disables retrying.
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    #: Set to False for an endpoint that needs no credential.
    llm_send_authorization: bool = True
    #: Model name reported by the offline provider.
    mock_llm_model: str = "mock-echo-v1"

    @model_validator(mode="after")
    def _validate_llm_settings(self) -> Settings:
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {sorted(LLM_PROVIDERS)}")
        if self.llm_chat_path and not self.llm_chat_path.startswith("/"):
            raise ValueError("llm_chat_path must start with '/'")
        if self.llm_provider == "custom" and not self.llm_base_url:
            raise ValueError(
                "llm_provider=custom requires llm_base_url "
                "(set MEMFORGE_LLM_BASE_URL, or use the default 'mock' provider)"
            )
        return self

    @model_validator(mode="after")
    def _validate_context_settings(self) -> Settings:
        if self.default_context_token_budget > self.max_context_token_budget:
            raise ValueError(
                "default_context_token_budget must not exceed max_context_token_budget"
            )
        weights = (
            self.context_relevance_weight,
            self.context_priority_weight,
            self.context_confidence_weight,
            self.context_freshness_weight,
            self.context_type_weight,
        )
        if abs(math.fsum(weights) - 1.0) > 1e-9:
            raise ValueError(f"context utility weights must sum to 1.0 (got {math.fsum(weights)})")
        return self

    def intelligence_config(self) -> IntelligenceConfig:
        """Build the M2 intelligence configuration from these settings."""
        return IntelligenceConfig(
            similarity_threshold=self.similarity_threshold,
            value_similarity_threshold=self.value_similarity_threshold,
            decay_half_life_days=self.decay_half_life_days,
            recurrence_saturation=self.recurrence_saturation,
            access_saturation=self.access_saturation,
        )

    def ranking_config(self) -> RankingConfig:
        """Build the M3 ranking configuration (weights stay code-level)."""
        return RankingConfig(
            default_semantic_weight=self.default_semantic_weight,
            decay_half_life_days=self.decay_half_life_days,
        )

    def token_estimator_config(self) -> TokenEstimatorConfig:
        """Build the M4 token estimator configuration."""
        return TokenEstimatorConfig(
            chars_per_token=self.token_chars_per_token,
            words_per_token=self.token_words_per_token,
            non_ascii_tokens_per_char=self.token_non_ascii_tokens_per_char,
        )

    def context_policy(self) -> ContextPolicy:
        """Build the M4 context policy (utility weights + ordering + compression)."""
        return ContextPolicy(
            relevance_weight=self.context_relevance_weight,
            priority_weight=self.context_priority_weight,
            confidence_weight=self.context_confidence_weight,
            freshness_weight=self.context_freshness_weight,
            type_weight=self.context_type_weight,
            cost_weight=self.context_cost_weight,
            cost_reference_tokens=self.context_cost_reference_tokens,
            min_candidate_score=self.context_min_candidate_score,
            max_block_tokens=self.context_max_block_tokens,
        )

    def custom_llm_config(self) -> CustomLLMConfig:
        """Build the M5 custom-provider configuration from the environment.

        The API key lives on the returned config object (excluded from its
        ``repr``) and is never written to the database or to any response.
        """
        return CustomLLMConfig(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
            path=self.llm_chat_path or DEFAULT_CHAT_PATH,
            timeout_seconds=self.llm_timeout_seconds,
            max_retries=self.llm_max_retries,
            backoff_seconds=self.llm_retry_backoff_seconds,
            send_authorization=self.llm_send_authorization,
        )

    def build_embedding_provider(self) -> EmbeddingProvider | None:
        """Build the optional embedding provider, or ``None`` for lexical-only."""
        return build_provider(
            self.embedding_provider,
            dimensions=self.embedding_dimensions,
            model=self.ollama_embedding_model,
            endpoint=self.ollama_endpoint,
            timeout_seconds=self.ollama_timeout_seconds,
        )

    def resolved_database_path(self) -> Path:
        """Return the database path, resolved against the working directory."""
        if self.database_path.is_absolute():
            return self.database_path
        return Path.cwd() / self.database_path
