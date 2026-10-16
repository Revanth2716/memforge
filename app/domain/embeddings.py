"""Local embedding providers (optional by design).

No embedding provider is required for the default installation: the retrieval
layer works in pure ``lexical`` mode without any of this. Providers are only
constructed when explicitly configured.

Two implementations ship here:

* :class:`HashedLexicalEmbeddingProvider` — deterministic, offline, model-free.
  Normalised tokens (the M2 normaliser) are hashed into a fixed-dimension,
  non-negative, L2-normalised vector. It exercises the whole semantic code path
  on a laptop with no download and no server, but it can only capture token
  overlap — it cannot capture synonymy or paraphrase.
* :class:`OllamaEmbeddingProvider` — optional local embeddings from a running
  Ollama server. Configurable model and endpoint, short timeouts, explicit
  availability probing, and never mandatory.

Both implement the same :class:`EmbeddingProvider` protocol, so the retrieval and
ranking code never knows which one is in use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from app.domain.errors import EmbeddingProviderError
from app.domain.similarity import normalized_tokens

logger = logging.getLogger("memforge.embeddings")

#: Default dimensionality of the model-free hashed provider.
DEFAULT_HASHED_DIMENSIONS = 256

#: Default local Ollama endpoint.
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"

#: Default embedding model served by Ollama.
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"

#: How long an availability probe result is trusted (seconds).
AVAILABILITY_CACHE_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Embedding:
    """A dense vector plus the provenance needed to interpret it."""

    values: tuple[float, ...]
    provider: str
    model: str
    dimensions: int
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.values:
            raise EmbeddingProviderError("an embedding must have at least one value")
        if len(self.values) != self.dimensions:
            raise EmbeddingProviderError(
                f"embedding has {len(self.values)} values but declares {self.dimensions} dimensions"
            )
        for value in self.values:
            if not math.isfinite(value):
                raise EmbeddingProviderError("embedding values must be finite numbers")
        if not self.provider or not self.model:
            raise EmbeddingProviderError("embedding provenance (provider/model) is required")


def l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    """Return a unit-length copy of ``values`` (a zero vector stays zero)."""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(float(value) / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity clamped to ``[0, 1]``.

    Vectors are compared as-is; callers are expected to store L2-normalised
    vectors. The clamp keeps the score bounded for providers whose vectors can
    contain negative components (real neural embeddings), and is documented in
    the README.
    """
    if len(left) != len(right):
        raise EmbeddingProviderError(
            f"cannot compare embeddings of different dimensions ({len(left)} vs {len(right)})"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return max(0.0, min(1.0, dot))


class EmbeddingProvider:
    """Interface every embedding backend implements."""

    #: Stable identifier recorded in the embedding store.
    name: str
    #: Model identifier recorded in the embedding store.
    model: str

    @property
    def dimensions(self) -> int:
        """Vector width produced by this provider."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """True when this provider can embed text right now."""
        raise NotImplementedError

    def embed(self, text: str) -> Embedding:
        """Embed ``text`` into a vector."""
        raise NotImplementedError

    def describe(self) -> str:
        """Human readable provenance, used in API responses."""
        return f"{self.name}/{self.model} ({self.dimensions}d)"


class HashedLexicalEmbeddingProvider(EmbeddingProvider):
    """Model-free, deterministic embeddings built from normalised tokens.

    Each token is hashed to a bucket; the bucket is incremented, and the vector
    is L2-normalised. Cosine similarity between two such vectors approximates
    token-set similarity, including the morphological variants handled by the
    M2 normaliser. It is *not* a neural semantic embedding.
    """

    name = "hashed-lexical"
    model = "hashed-lexical-v1"

    def __init__(self, dimensions: int = DEFAULT_HASHED_DIMENSIONS) -> None:
        if dimensions < 8:
            raise EmbeddingProviderError("hashed embeddings need at least 8 dimensions")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Width of the vectors produced by this provider."""
        return self._dimensions

    def is_available(self) -> bool:
        """Always available: no server, no model, no network."""
        return True

    def embed(self, text: str) -> Embedding:
        if not isinstance(text, str):
            raise EmbeddingProviderError("text must be a string")
        buckets = [0.0] * self.dimensions
        for token in normalized_tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            buckets[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        return Embedding(
            values=l2_normalize(buckets),
            provider=self.name,
            model=self.model,
            dimensions=self.dimensions,
            created_at=datetime.now(UTC),
        )


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Optional local embeddings served by an Ollama instance.

    Ollama is never required: if the server is unreachable, ``is_available()``
    returns ``False`` and the retrieval layer stays in lexical mode instead of
    pretending that embeddings exist.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        timeout_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise EmbeddingProviderError("timeout_seconds must be > 0")
        self.model = model
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._available_until = 0.0
        self._available = False
        self._dimensions = 0

    @property
    def dimensions(self) -> int:
        """Vector width, discovered on the first successful embed."""
        return self._dimensions

    def is_available(self) -> bool:
        """Probe the local server, caching the answer briefly."""
        now = time.monotonic()
        if now < self._available_until:
            return self._available
        self._available = self._probe()
        self._available_until = now + AVAILABILITY_CACHE_SECONDS
        return self._available

    def _probe(self) -> bool:
        request = urllib.request.Request(
            f"{self._endpoint}/api/tags", headers={"accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.debug("Ollama embedding provider unavailable: %s", exc)
            return False
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return False
        return any(str(entry.get("name", "")).startswith(self.model) for entry in models)

    def embed(self, text: str) -> Embedding:
        if not self.is_available():
            raise EmbeddingProviderError(
                f"Ollama is not reachable at {self._endpoint} or model {self.model!r} is missing"
            )
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}/api/embeddings",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise EmbeddingProviderError(f"Ollama embedding request failed: {exc}") from exc

        values = payload.get("embedding") if isinstance(payload, dict) else None
        if not isinstance(values, list) or not values:
            raise EmbeddingProviderError("Ollama returned no embedding vector")
        vector = tuple(float(value) for value in values)
        self._dimensions = len(vector)
        return Embedding(
            values=vector,
            provider=self.name,
            model=self.model,
            dimensions=len(vector),
            created_at=datetime.now(UTC),
        )


def build_provider(
    name: str | None, *, dimensions: int = DEFAULT_HASHED_DIMENSIONS, **options: object
) -> EmbeddingProvider | None:
    """Build a provider from a configuration name, or ``None`` for lexical-only.

    Unknown names are rejected rather than silently ignored, so a typo can never
    degrade a deployment to a slower mode without an error.
    """
    if name is None or name == "":
        return None
    key = name.strip().lower()
    if key in {"none", "off", "lexical"}:
        return None
    if key in {HashedLexicalEmbeddingProvider.name, "hashed"}:
        return HashedLexicalEmbeddingProvider(dimensions=dimensions)
    if key == OllamaEmbeddingProvider.name:
        return OllamaEmbeddingProvider(**options)  # type: ignore[arg-type]
    raise EmbeddingProviderError(f"unknown embedding provider {name!r}")


__all__ = [
    "AVAILABILITY_CACHE_SECONDS",
    "DEFAULT_HASHED_DIMENSIONS",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OLLAMA_MODEL",
    "Embedding",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "HashedLexicalEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "build_provider",
    "cosine_similarity",
    "l2_normalize",
]
