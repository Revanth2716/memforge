"""Identifier generation.

Identifiers are generated locally (no network, no external service). The
generator is behind the :class:`IdGenerator` protocol so tests and demos can
inject a deterministic implementation.
"""

from __future__ import annotations

import uuid
from typing import Protocol


class IdGenerator(Protocol):
    """Produces memory identifiers."""

    def next_id(self) -> str:
        """Return a new unique identifier string."""
        ...


class UuidIdGenerator:
    """Random UUID4 identifiers (the default, offline and dependency-free)."""

    def next_id(self) -> str:
        return str(uuid.uuid4())


class SequentialIdGenerator:
    """Deterministic identifiers derived from a namespace plus a counter.

    Uses UUID5 (name based, SHA-1) so identifiers are valid UUID strings *and*
    byte-for-byte reproducible, which keeps demo output and tests deterministic.
    """

    def __init__(self, namespace: str = "memforge", prefix: str = "mem") -> None:
        self._namespace = uuid.uuid5(uuid.NAMESPACE_DNS, namespace)
        self._prefix = prefix
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return str(uuid.uuid5(self._namespace, f"{self._prefix}-{self._counter}"))
