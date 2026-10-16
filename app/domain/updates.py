"""Partial-update semantics for memories.

A patch must be able to distinguish three cases:

1. *field not provided* -> leave the stored value untouched;
2. *field provided with a value* -> replace the stored value;
3. *field explicitly set to null* -> clear the stored value (only ``source`` is
   nullable).

The :data:`UNSET` sentinel makes that explicit instead of overloading ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from app.domain.errors import MemoryValidationError
from app.domain.models import Memory, MemoryStatus, MemoryType


@dataclass(frozen=True, slots=True)
class Unset:
    """Sentinel type marking "this field is not part of the update"."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "UNSET"


#: Singleton sentinel instance.
UNSET = Unset()

#: Fields that can never be set to ``None`` by a patch.
_NON_NULLABLE = ("content", "memory_type", "confidence", "importance", "status")


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    """A set of changes to apply to an existing memory.

    Every field defaults to :data:`UNSET` ("not provided"). Assigning ``None``
    to ``source`` clears the provenance; assigning ``None`` to a non-nullable
    field raises :class:`MemoryValidationError`.
    """

    content: str | Unset | None = UNSET
    memory_type: MemoryType | Unset | None = UNSET
    source: str | Unset | None = UNSET
    confidence: float | Unset | None = UNSET
    importance: float | Unset | None = UNSET
    status: MemoryStatus | Unset | None = UNSET

    def provided_fields(self) -> tuple[str, ...]:
        """Names of the fields that are actually part of this update."""
        return tuple(f.name for f in fields(self) if not isinstance(getattr(self, f.name), Unset))

    def is_empty(self) -> bool:
        """True when the update would not change anything."""
        return not self.provided_fields()

    def apply_to(self, memory: Memory) -> Memory:
        """Return a new :class:`Memory` with the update applied.

        ``id``, ``created_at``, ``version`` and ``updated_at`` are preserved --
        the service layer decides when to bump the version and timestamp.
        """
        changes: dict[str, Any] = {}
        for name in self.provided_fields():
            value = getattr(self, name)
            if value is None and name in _NON_NULLABLE:
                raise MemoryValidationError(f"Field {name!r} cannot be null")
            changes[name] = value
        if not changes:
            raise MemoryValidationError("At least one field must be provided to update a memory")
        return replace(memory, **changes)
