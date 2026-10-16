"""Coverage for the M2 validation and error-mapping branches."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.errors import (
    InvalidIntelligenceConfigError,
    InvalidRelationshipError,
    MemoryValidationError,
)
from app.domain.intelligence import (
    DEFAULT_IMPORTANCE_WEIGHTS,
    DEFAULT_PRIORITY_WEIGHTS,
    IntelligenceConfig,
    is_active,
)
from app.domain.models import MemoryStatus, MemoryType
from app.domain.relationships import MemoryRelationship, RelationshipType
from app.domain.similarity import stem
from app.storage.sqlite.repository import SqliteMemoryRepository
from tests.support import FIXED_START, make_memory

OTHER_ID = "33333333-3333-4333-8333-333333333333"


class TestConfigValidationEdges:
    def test_weights_out_of_range(self) -> None:
        with pytest.raises(InvalidIntelligenceConfigError):
            IntelligenceConfig(importance_weights={"memory_type": -0.5, "x": 1.5})
        with pytest.raises(InvalidIntelligenceConfigError):
            IntelligenceConfig(priority_weights={"importance": 1.5, "confidence": 1.5})

    def test_default_weights_are_valid(self) -> None:
        assert sum(DEFAULT_IMPORTANCE_WEIGHTS.values()) == pytest.approx(1.0)
        assert sum(DEFAULT_PRIORITY_WEIGHTS.values()) == pytest.approx(1.0)


class TestMemoryValidationEdges:
    @pytest.mark.parametrize("access_count", [-1, True, "3"])
    def test_invalid_access_counts(self, access_count: object) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(access_count=access_count)

    def test_last_accessed_before_creation_is_rejected(self) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(last_accessed_at=FIXED_START - timedelta(days=1))

    def test_valid_telemetry_is_accepted(self) -> None:
        memory = make_memory(access_count=2, last_accessed_at=FIXED_START + timedelta(hours=1))
        assert memory.access_count == 2


class TestRelationshipValidationEdges:
    def _base(self) -> dict[str, object]:
        return {
            "source_id": "11111111-1111-5111-8111-111111111111",
            "target_id": OTHER_ID,
            "relationship": RelationshipType.DUPLICATE_OF,
            "created_at": FIXED_START,
        }

    def test_blank_ids_are_rejected(self) -> None:
        with pytest.raises(InvalidRelationshipError):
            MemoryRelationship(**{**self._base(), "source_id": "  "})  # type: ignore[arg-type]
        with pytest.raises(InvalidRelationshipError):
            MemoryRelationship(**{**self._base(), "target_id": ""})  # type: ignore[arg-type]

    def test_unknown_relationship_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationshipError):
            MemoryRelationship(
                **{**self._base(), "relationship": "BEST_FRIENDS"}  # type: ignore[arg-type]
            )

    def test_self_relationship_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationshipError):
            MemoryRelationship(
                **{**self._base(), "target_id": "11111111-1111-5111-8111-111111111111"}  # type: ignore[arg-type]
            )

    def test_is_active_helper(self) -> None:
        assert is_active(make_memory(status=MemoryStatus.ACTIVE)) is True
        assert is_active(make_memory(status=MemoryStatus.ARCHIVED)) is False


class TestStemmerEdges:
    def test_es_stripping_for_plural_consonants(self) -> None:
        assert stem("boxes") == "box"
        assert stem("watches") == "watch"
        assert stem("buzzes") == "buzz"
        assert stem("classes") == "class"
        assert stem("buses") == "bus"  # "s" rule
        assert stem("types") == "type"


class TestRelationshipIntegrityEdges:
    def test_unknown_constraint_is_wrapped(self, repository: SqliteMemoryRepository) -> None:
        """A rejected write that is not a duplicate or a missing memory."""
        import sqlite3

        error = SqliteMemoryRepository._relationship_integrity_error(
            sqlite3.IntegrityError("CHECK constraint failed: chk_relationship_kind"),
            MemoryRelationship(
                source_id="11111111-1111-5111-8111-111111111111",
                target_id=OTHER_ID,
                relationship=RelationshipType.SUPERSEDES,
                created_at=FIXED_START,
            ),
        )
        assert isinstance(error, InvalidRelationshipError)

    def test_missing_memory_is_reported(self, repository: SqliteMemoryRepository) -> None:
        import sqlite3

        error = SqliteMemoryRepository._relationship_integrity_error(
            sqlite3.IntegrityError("FOREIGN KEY constraint failed"),
            MemoryRelationship(
                source_id="11111111-1111-5111-8111-111111111111",
                target_id=OTHER_ID,
                relationship=RelationshipType.SUPERSEDES,
                created_at=FIXED_START,
            ),
        )
        from app.domain.errors import MemoryNotFoundError

        assert isinstance(error, MemoryNotFoundError)


def test_type_weight_lookup_uses_the_configuration() -> None:
    weights = {
        MemoryType.PROFILE: 0.1,
        MemoryType.FACT: 0.8,
        MemoryType.PREFERENCE: 0.7,
        MemoryType.TASK: 0.7,
        MemoryType.EVENT: 0.5,
        MemoryType.OBSERVATION: 0.3,
    }
    config = IntelligenceConfig(type_weights=weights)
    assert config.type_weight(MemoryType.PROFILE) == 0.1
