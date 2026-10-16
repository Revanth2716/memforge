"""Persistence layer.

Exposes the backend-agnostic :class:`~app.storage.base.MemoryRepository`
contract plus its SQLite implementation.
"""

from app.storage.base import MemoryQuery, MemoryRepository
from app.storage.sqlite.repository import SqliteMemoryRepository

__all__ = ["MemoryQuery", "MemoryRepository", "SqliteMemoryRepository"]
