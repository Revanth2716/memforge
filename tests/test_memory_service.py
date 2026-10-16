"""Memory service tests (lifecycle rules, deterministic clock and ids)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.errors import (
    InvalidMemoryStateError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from app.domain.models import MemoryStatus, MemoryType, VersionChange
from app.domain.time import ManualClock
from app.domain.updates import MemoryUpdate
from app.services.memory_service import MemoryService
from app.storage.base import MemoryQuery

UNKNOWN_ID = "00000000-0000-4000-8000-000000000000"


class TestCreate:
    def test_create_assigns_identity_and_defaults(self, service: MemoryService) -> None:
        memory = service.create_memory(
            content="User prefers pytest over unittest.",
            memory_type=MemoryType.PREFERENCE,
        )

        assert memory.id
        assert memory.status is MemoryStatus.ACTIVE
        assert memory.version == 1
        assert memory.created_at == memory.updated_at
        assert memory.source is None
        assert memory.confidence == 0.5
        assert memory.importance == 0.5

    def test_create_is_immediately_readable(self, service: MemoryService) -> None:
        created = service.create_memory(content="hello world", memory_type=MemoryType.FACT)
        assert service.get_memory(created.id) == created
        assert service.list_memories().total == 1

    def test_create_writes_first_history_version(self, service: MemoryService) -> None:
        created = service.create_memory(content="hello world", memory_type=MemoryType.FACT)
        history = service.get_history(created.id)

        assert len(history) == 1
        assert history[0].version == 1
        assert history[0].change is VersionChange.CREATED
        assert history[0].content == "hello world"

    def test_ids_are_unique_per_memory(self, service: MemoryService) -> None:
        ids = {
            service.create_memory(content=f"memory {i}", memory_type=MemoryType.TASK).id
            for i in range(5)
        }
        assert len(ids) == 5

    @pytest.mark.parametrize("content", ["", "   ", "\n\t"])
    def test_create_rejects_blank_content(self, service: MemoryService, content: str) -> None:
        with pytest.raises(MemoryValidationError):
            service.create_memory(content=content, memory_type=MemoryType.FACT)

    def test_create_rejects_out_of_range_scores(self, service: MemoryService) -> None:
        with pytest.raises(MemoryValidationError):
            service.create_memory(content="x", memory_type=MemoryType.FACT, confidence=1.5)
        with pytest.raises(MemoryValidationError):
            service.create_memory(content="x", memory_type=MemoryType.FACT, importance=-0.2)


class TestRead:
    def test_get_unknown_memory_raises(self, service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            service.get_memory(UNKNOWN_ID)

    def test_list_returns_newest_first(self, service: MemoryService, clock: ManualClock) -> None:
        first = service.create_memory(content="first", memory_type=MemoryType.TASK)
        clock.advance(minutes=1)
        second = service.create_memory(content="second", memory_type=MemoryType.TASK)

        page = service.list_memories()
        assert page.total == 2
        assert [memory.id for memory in page.items] == [second.id, first.id]
        assert page.limit == 50
        assert page.offset == 0

    def test_list_filters_by_type_and_status(self, service: MemoryService) -> None:
        service.create_memory(content="a", memory_type=MemoryType.TASK)
        service.create_memory(content="b", memory_type=MemoryType.PREFERENCE)

        page = service.list_memories(MemoryQuery(memory_type=MemoryType.TASK))
        assert page.total == 1
        assert page.items[0].content == "a"

        page = service.list_memories(MemoryQuery(status=MemoryStatus.ARCHIVED))
        assert page.total == 0

    def test_list_paginates(self, service: MemoryService, clock: ManualClock) -> None:
        for index in range(3):
            service.create_memory(content=f"memory {index}", memory_type=MemoryType.TASK)
            clock.advance(seconds=1)

        page = service.list_memories(MemoryQuery(limit=2, offset=0))
        assert page.total == 3
        assert len(page.items) == 2
        assert [memory.content for memory in page.items] == ["memory 2", "memory 1"]

        page = service.list_memories(MemoryQuery(limit=2, offset=2))
        assert [memory.content for memory in page.items] == ["memory 0"]


class TestUpdate:
    def test_update_changes_only_provided_fields(
        self, service: MemoryService, clock: ManualClock
    ) -> None:
        created = service.create_memory(
            content="original",
            memory_type=MemoryType.TASK,
            source="agent:planner",
            confidence=0.8,
            importance=0.4,
        )
        clock.advance(minutes=10)

        updated = service.update_memory(created.id, MemoryUpdate(content="revised", importance=0.9))

        assert updated.content == "revised"
        assert updated.importance == 0.9
        assert updated.memory_type is MemoryType.TASK
        assert updated.source == "agent:planner"
        assert updated.confidence == 0.8
        assert updated.status is MemoryStatus.ACTIVE
        assert updated.version == 2
        assert updated.created_at == created.created_at
        assert updated.updated_at - created.updated_at == timedelta(minutes=10)

    def test_update_keeps_previous_version_in_history(
        self, service: MemoryService, clock: ManualClock
    ) -> None:
        created = service.create_memory(content="original", memory_type=MemoryType.TASK)
        clock.advance(minutes=1)
        service.update_memory(created.id, MemoryUpdate(content="revised"))

        history = service.get_history(created.id)
        assert [version.version for version in history] == [1, 2]
        assert history[0].content == "original"
        assert history[0].change is VersionChange.CREATED
        assert history[1].content == "revised"
        assert history[1].change is VersionChange.UPDATED

    def test_update_can_change_status(self, service: MemoryService) -> None:
        created = service.create_memory(content="note", memory_type=MemoryType.OBSERVATION)
        updated = service.update_memory(created.id, MemoryUpdate(status=MemoryStatus.ARCHIVED))

        assert updated.status is MemoryStatus.ARCHIVED
        assert service.get_memory(created.id).status is MemoryStatus.ARCHIVED
        assert service.list_memories().total == 0  # default list only shows ACTIVE
        assert service.list_memories(MemoryQuery(status=MemoryStatus.ARCHIVED)).total == 1

    def test_update_can_clear_source(self, service: MemoryService) -> None:
        created = service.create_memory(
            content="note", memory_type=MemoryType.FACT, source="user:revanth"
        )
        updated = service.update_memory(created.id, MemoryUpdate(source=None))
        assert updated.source is None

    def test_empty_update_is_rejected(self, service: MemoryService) -> None:
        created = service.create_memory(content="note", memory_type=MemoryType.FACT)
        with pytest.raises(MemoryValidationError):
            service.update_memory(created.id, MemoryUpdate())
        assert service.get_memory(created.id).version == 1

    def test_update_unknown_memory_raises(self, service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            service.update_memory(UNKNOWN_ID, MemoryUpdate(content="x"))

    def test_update_forgotten_memory_raises_not_found(
        self, service: MemoryService, clock: ManualClock
    ) -> None:
        created = service.create_memory(content="note", memory_type=MemoryType.FACT)
        clock.advance(minutes=1)
        service.forget_memory(created.id)

        with pytest.raises(MemoryNotFoundError):
            service.update_memory(created.id, MemoryUpdate(content="zombie"))

    def test_update_cannot_forget_a_memory(self, service: MemoryService) -> None:
        created = service.create_memory(content="note", memory_type=MemoryType.FACT)
        with pytest.raises(InvalidMemoryStateError):
            service.update_memory(created.id, MemoryUpdate(status=MemoryStatus.DELETED))

        # The rejected update must not have touched the stored memory.
        stored = service.get_memory(created.id)
        assert stored.status is MemoryStatus.ACTIVE
        assert stored.version == 1
        assert [version.change for version in service.get_history(created.id)] == [
            VersionChange.CREATED
        ]


class TestForget:
    def test_forget_soft_deletes_and_records_history(
        self, service: MemoryService, clock: ManualClock
    ) -> None:
        created = service.create_memory(content="temporary", memory_type=MemoryType.TASK)
        clock.advance(minutes=2)

        forgotten = service.forget_memory(created.id)

        assert forgotten.status is MemoryStatus.DELETED
        assert forgotten.version == 2
        assert forgotten.content == "temporary"  # content is preserved, not destroyed
        assert forgotten.updated_at - created.updated_at == timedelta(minutes=2)

        with pytest.raises(MemoryNotFoundError):
            service.get_memory(created.id)
        assert service.list_memories().total == 0

        history = service.get_history(created.id)
        assert [version.change for version in history] == [
            VersionChange.CREATED,
            VersionChange.FORGOTTEN,
        ]

    def test_forget_is_idempotent(self, service: MemoryService, clock: ManualClock) -> None:
        created = service.create_memory(content="temporary", memory_type=MemoryType.TASK)
        clock.advance(minutes=1)
        service.forget_memory(created.id)

        again = service.forget_memory(created.id)
        assert again.version == 2
        assert len(service.get_history(created.id)) == 2

    def test_forget_unknown_memory_raises(self, service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            service.forget_memory(UNKNOWN_ID)


class TestHistory:
    def test_history_of_unknown_memory_raises(self, service: MemoryService) -> None:
        with pytest.raises(MemoryNotFoundError):
            service.get_history(UNKNOWN_ID)

    def test_history_is_append_only_over_many_updates(
        self, service: MemoryService, clock: ManualClock
    ) -> None:
        created = service.create_memory(content="v1", memory_type=MemoryType.TASK)
        for number in range(2, 6):
            clock.advance(minutes=1)
            service.update_memory(created.id, MemoryUpdate(content=f"v{number}"))

        clock.advance(minutes=1)
        service.forget_memory(created.id)

        history = service.get_history(created.id)
        assert [version.version for version in history] == [1, 2, 3, 4, 5, 6]
        assert [version.content for version in history] == ["v1", "v2", "v3", "v4", "v5", "v5"]
        assert [version.change for version in history] == [
            VersionChange.CREATED,
            *[VersionChange.UPDATED] * 4,
            VersionChange.FORGOTTEN,
        ]

    def test_history_available_after_forgetting(self, service: MemoryService) -> None:
        created = service.create_memory(content="keep me", memory_type=MemoryType.PREFERENCE)
        service.forget_memory(created.id)

        history = service.get_history(created.id)
        assert len(history) == 2
        assert history[-1].status is MemoryStatus.DELETED


class TestHealth:
    def test_health_check(self, service: MemoryService) -> None:
        assert service.health_check() is True

    def test_initialize_is_idempotent(self, service: MemoryService) -> None:
        service.initialize()
        service.initialize()
        assert service.health_check() is True
