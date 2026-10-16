"""SQLite connection helpers.

Every repository operation opens its own short-lived connection. That keeps the
backend safe to share between the API, the CLI and background jobs, and it makes
"persistence across process restarts" a property of the file, not of an
in-memory pool.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def open_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection with MemForge's pragmas applied.

    * ``foreign_keys = ON``  -> version rows cannot outlive their memory,
    * ``row_factory``        -> rows behave like mappings,
    * ``isolation_level=None`` -> transactions are managed explicitly by
      :func:`transaction` instead of sqlite3's implicit legacy behaviour.
    """
    path = Path(db_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside a single ``BEGIN IMMEDIATE`` transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
