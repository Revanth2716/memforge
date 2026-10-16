"""MCP surface for MemForge.

The server is a thin adapter: every tool validates its input, calls one
:class:`~app.services.memory_operations.MemoryOperationsService` verb, and returns
a strictly typed payload. There is no retrieval, ranking, context-compilation or
storage logic in this package -- if a tool could answer a question without MemForge
M1-M4, that would be a bug.

Tools
-----
=====================  ==========================================================
``memory.store``       store a memory (M1)
``memory.search``      ranked search (M3)
``memory.recall``      recall for an agent, relevance-gated (M3)
``memory.update``      partial update (M1)
``memory.forget``      soft delete (M1)
``memory.compile_context``  budgeted context compilation (M4)
``memory.explain``     per-memory ranking explanation (M3)
=====================  ==========================================================

Errors are normalised into ``{"error": {"code", "message", "details"}}`` -- the
same envelope the REST API uses -- and returned with ``isError`` set, so a client
sees a structured failure instead of a stack trace. No tool response can contain
a provider credential; none of them touch the provider layer at all.
"""

from __future__ import annotations

from app.mcp.server import MEMFORGE_MCP_INSTRUCTIONS, build_mcp_server, run_mcp_server

__all__ = ["MEMFORGE_MCP_INSTRUCTIONS", "build_mcp_server", "run_mcp_server"]
