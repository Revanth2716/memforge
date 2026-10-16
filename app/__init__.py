"""MemForge -- local-first persistent memory and context engineering platform.

Milestone 1 (memory core) ships:

* a validated, framework-free memory domain model (``app.domain``),
* append-only version history,
* SQLite persistence behind a repository interface (``app.storage``),
* a service layer with the memory CRUD + history operations (``app.services``),
* a versioned REST API (``app.api``),
* a deterministic CLI demo (``app.cli``).

No network access, no API keys and no LLM are required for any of it.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
