"""Transport-agnostic errors raised by MemForge.

Every error carries enough context for the API layer to build a consistent
error envelope. The API layer is the only place that maps these to HTTP status
codes; the domain and storage layers never import a web framework.
"""

from __future__ import annotations


class MemforgeError(Exception):
    """Base class for every error raised by MemForge."""


class MemoryValidationError(MemforgeError):
    """A memory payload violates a domain invariant."""


class MemoryNotFoundError(MemforgeError):
    """The requested memory does not exist (or is forgotten and hidden)."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory {memory_id!r} does not exist")


class MemoryAlreadyExistsError(MemforgeError):
    """A memory with the same identifier already exists."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory {memory_id!r} already exists")


class InvalidMemoryStateError(MemforgeError):
    """The operation is not valid for the memory's current state."""

    def __init__(self, message: str, *, memory_id: str | None = None) -> None:
        self.memory_id = memory_id
        super().__init__(message)


class RepositoryError(MemforgeError):
    """The storage backend failed to complete an operation."""


class StorageUnavailableError(MemforgeError):
    """The storage backend is configured but not usable."""


class InvalidRelationshipError(MemforgeError):
    """A relationship between memories violates a domain invariant."""


class RelationshipExistsError(MemforgeError):
    """The same relationship between the same pair already exists."""

    def __init__(self, source_id: str, target_id: str, relationship: str) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.relationship = relationship
        super().__init__(
            f"Relationship {relationship} from {source_id!r} to {target_id!r} already exists"
        )


class NoConflictError(MemforgeError):
    """A conflict resolution was requested but no unresolved conflict exists."""

    def __init__(self, memory_id: str, target_id: str | None = None) -> None:
        self.memory_id = memory_id
        self.target_id = target_id
        if target_id is None:
            message = f"Memory {memory_id!r} has no unresolved conflicts"
        else:
            message = f"Memory {memory_id!r} has no unresolved conflict with {target_id!r}"
        super().__init__(message)


class InvalidIntelligenceConfigError(MemforgeError):
    """The intelligence configuration is inconsistent."""


class InvalidRetrievalConfigError(MemforgeError):
    """The retrieval ranking configuration is inconsistent."""


class RetrievalValidationError(MemforgeError):
    """A retrieval request is invalid."""


class EmbeddingProviderError(MemforgeError):
    """An embedding provider could not produce a usable vector."""


class InvalidContextConfigError(MemforgeError):
    """The context compilation configuration is inconsistent."""


class ContextValidationError(MemforgeError):
    """A context compilation request is invalid."""


class LLMError(MemforgeError):
    """Base class for every language-model failure.

    Carries the provider name and a machine-readable ``kind`` so the API layer can
    map a failure to a status code without the provider layer knowing that HTTP
    exists. Messages reaching this class must already be free of credentials.
    """

    #: Machine-readable failure category.
    kind: str = "unknown"
    #: Whether a bounded retry could plausibly succeed.
    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(message)


class LLMProviderError(LLMError):
    """A normalised language-model failure.

    The umbrella for everything a provider can go wrong with, so a caller that
    only wants "the provider failed" catches exactly one type. The concrete
    subclasses carry the meaning; ``kind`` is what the API layer maps.
    """

    kind = "provider_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message, provider=provider)


class LLMNotConfiguredError(LLMProviderError):
    """The provider is missing configuration required to serve a request."""

    kind = "not_configured"
    retryable = False


class LLMTimeoutError(LLMProviderError):
    """The provider did not answer within the configured timeout."""

    kind = "timeout"
    retryable = True


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the supplied credentials."""

    kind = "authentication"
    retryable = False


class LLMRateLimitError(LLMProviderError):
    """The provider refused the request because of rate limiting."""

    kind = "rate_limit"
    retryable = True


class LLMTransportError(LLMProviderError):
    """The request never produced an HTTP response."""

    kind = "transport"
    retryable = True


class LLMProtocolError(LLMProviderError):
    """The provider answered with something MemForge cannot interpret."""

    kind = "protocol"
    retryable = False


class AgentValidationError(MemforgeError):
    """An agent request is invalid."""


class ObservabilityError(MemforgeError):
    """A trace or an event could not be recorded or read."""

    def __init__(self, message: str, *, trace_id: str | None = None) -> None:
        self.trace_id = trace_id
        super().__init__(message)


class EvaluationError(MemforgeError):
    """An evaluation dataset, case or result is invalid."""

    def __init__(self, message: str, *, dataset: str | None = None) -> None:
        self.dataset = dataset
        super().__init__(message)
