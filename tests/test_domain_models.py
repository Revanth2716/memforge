"""Domain model validation tests."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from app.domain.errors import MemoryValidationError
from app.domain.models import (
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    VersionChange,
    build_version,
)
from app.domain.updates import UNSET, MemoryUpdate
from tests.support import FIXED_START, make_memory


class TestMemoryEntity:
    def test_valid_memory(self) -> None:
        memory = make_memory()
        assert memory.memory_type is MemoryType.PREFERENCE
        assert memory.status is MemoryStatus.ACTIVE
        assert memory.version == 1

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(content="")

    def test_whitespace_only_content_rejected(self) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(content="   \n\t ")

    def test_blank_id_rejected(self) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(id="  ")

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
    def test_confidence_out_of_range_rejected(self, confidence: float) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(confidence=confidence)

    @pytest.mark.parametrize("importance", [-0.01, 1.01])
    def test_importance_out_of_range_rejected(self, importance: float) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(importance=importance)

    @pytest.mark.parametrize("field", ["confidence", "importance"])
    def test_bool_is_not_a_valid_score(self, field: str) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(**{field: True})

    def test_version_must_be_positive(self) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(version=0)

    def test_updated_before_created_rejected(self) -> None:
        with pytest.raises(MemoryValidationError):
            make_memory(updated_at=FIXED_START - timedelta(seconds=1))

    def test_memory_is_immutable(self) -> None:
        memory = make_memory()
        with pytest.raises(dataclasses.FrozenInstanceError):
            memory.content = "changed"  # type: ignore[misc]

    def test_memory_equality_is_value_based(self) -> None:
        assert make_memory() == make_memory()


class TestEnums:
    def test_memory_type_members(self) -> None:
        assert [member.value for member in MemoryType] == [
            "FACT",
            "PREFERENCE",
            "PROFILE",
            "EVENT",
            "TASK",
            "OBSERVATION",
        ]

    def test_memory_status_members(self) -> None:
        assert [member.value for member in MemoryStatus] == ["ACTIVE", "ARCHIVED", "DELETED"]

    def test_version_change_members(self) -> None:
        assert [member.value for member in VersionChange] == ["CREATED", "UPDATED", "FORGOTTEN"]

    def test_enum_lookup_from_value(self) -> None:
        assert MemoryType("TASK") is MemoryType.TASK
        with pytest.raises(ValueError):
            MemoryType("NOT_A_TYPE")


class TestBuildVersion:
    def test_snapshot_copies_current_state(self) -> None:
        memory = make_memory(version=3, confidence=0.25, importance=0.75)
        recorded_at = FIXED_START + timedelta(minutes=10)
        version = build_version(memory, change=VersionChange.UPDATED, recorded_at=recorded_at)

        assert version.memory_id == memory.id
        assert version.version == 3
        assert version.change is VersionChange.UPDATED
        assert version.content == memory.content
        assert version.memory_type is memory.memory_type
        assert version.source == memory.source
        assert version.confidence == 0.25
        assert version.importance == 0.75
        assert version.status is memory.status
        assert version.recorded_at == recorded_at

    def test_snapshot_rejects_invalid_version_number(self) -> None:
        memory = make_memory()
        with pytest.raises(MemoryValidationError):
            MemoryVersion(
                memory_id=memory.id,
                version=0,
                change=VersionChange.CREATED,
                content=memory.content,
                memory_type=memory.memory_type,
                source=memory.source,
                confidence=memory.confidence,
                importance=memory.importance,
                status=memory.status,
                recorded_at=FIXED_START,
            )

    def test_snapshot_rejects_invalid_memory_id(self) -> None:
        with pytest.raises(MemoryValidationError):
            MemoryVersion(
                memory_id=" ",
                version=1,
                change=VersionChange.CREATED,
                content="x",
                memory_type=MemoryType.FACT,
                source=None,
                confidence=0.5,
                importance=0.5,
                status=MemoryStatus.ACTIVE,
                recorded_at=FIXED_START,
            )


class TestMemoryUpdate:
    def test_empty_update(self) -> None:
        assert MemoryUpdate().is_empty()
        assert MemoryUpdate().provided_fields() == ()

    def test_provided_fields(self) -> None:
        update = MemoryUpdate(content="new", importance=0.1)
        assert update.provided_fields() == ("content", "importance")

    def test_apply_only_provided_fields(self) -> None:
        memory = make_memory(content="old", memory_type=MemoryType.TASK, source="user:test")
        updated = MemoryUpdate(content="new").apply_to(memory)

        assert updated.content == "new"
        assert updated.memory_type is MemoryType.TASK
        assert updated.source == "user:test"
        assert updated.created_at == memory.created_at
        assert updated.updated_at == memory.updated_at
        assert updated.version == memory.version
        assert memory.content == "old"  # original untouched

    def test_apply_clears_source_with_none(self) -> None:
        memory = make_memory(source="user:test")
        updated = MemoryUpdate(source=None).apply_to(memory)
        assert updated.source is None

    def test_apply_to_empty_update_raises(self) -> None:
        memory = make_memory()
        with pytest.raises(MemoryValidationError):
            MemoryUpdate().apply_to(memory)

    @pytest.mark.parametrize(
        "field", ["content", "memory_type", "confidence", "importance", "status"]
    )
    def test_null_rejected_for_non_nullable_fields(self, field: str) -> None:
        memory = make_memory()
        update = MemoryUpdate(**{field: None})
        with pytest.raises(MemoryValidationError):
            update.apply_to(memory)

    def test_unset_is_a_singleton_sentinel(self) -> None:
        assert isinstance(UNSET, object)
        assert repr(UNSET) == "UNSET"

    def test_memory_update_is_immutable(self) -> None:
        update = MemoryUpdate(content="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            update.content = "y"  # type: ignore[misc]


def test_timestamps_are_timezone_aware() -> None:
    now = datetime.now(UTC)
    memory = make_memory(created_at=now, updated_at=now)
    assert memory.created_at.tzinfo is not None
    assert memory.updated_at.tzinfo is not None
