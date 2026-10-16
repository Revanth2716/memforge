"""``memforge-mcp`` entry point: run the MemForge MCP server over stdio.

MCP clients launch a server as a subprocess and speak the protocol over its
stdin/stdout, so this command is the whole integration story::

    {
      "mcpServers": {
        "memforge": {
          "command": "memforge-mcp",
          "env": {"MEMFORGE_DATABASE_PATH": "~/.memforge/memforge.sqlite"}
        }
      }
    }

The server reads the same environment as the REST API (prefix ``MEMFORGE_``) and
exposes the same services. **Nothing it does requires a network or an API key**:
the memory tools never touch a language-model provider.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, Settings
from app.domain.errors import MemforgeError
from app.mcp.server import run_mcp_server
from app.observability.instrumentation import build_traced_operations
from app.observability.service import ObservabilityService
from app.services.context_service import ContextCompilerService
from app.services.memory_operations import MemoryOperationsService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import MemoryRetrievalService
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore
from app.storage.sqlite.repository import SqliteMemoryRepository
from app.storage.sqlite.trace_store import SqliteTraceStore

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memforge.sqlite"


def build_operations(
    settings: Settings, observability: ObservabilityService | None = None
) -> MemoryOperationsService:
    """Build the service stack the MCP tools are bound to.

    When an observability service is supplied, M1 and M3 are wrapped so the
    tools' own writes and retrievals are recorded into each call's trace. The
    facade itself is unchanged, so a tool behaves identically either way.
    """
    db_path = settings.resolved_database_path()
    repository = SqliteMemoryRepository(db_path)
    repository.initialize()
    memories = MemoryService(repository, intelligence=settings.intelligence_config())
    retrieval = MemoryRetrievalService(
        repository,
        ranking=settings.ranking_config(),
        intelligence=settings.intelligence_config(),
        embedding_provider=settings.build_embedding_provider(),
        embedding_store=SqliteEmbeddingStore(db_path),
    )
    compiler = ContextCompilerService(retrieval)
    if observability is None:
        return MemoryOperationsService(memories, retrieval, compiler)
    return build_traced_operations(memories, retrieval, compiler, observability)


def build_observability(settings: Settings) -> ObservabilityService:
    """Build the trace store the MCP server records into."""
    service = ObservabilityService(SqliteTraceStore(settings.resolved_database_path()))
    if settings.llm_api_key:
        # Traces must never carry a credential, even if a tool echoed one back.
        service.register_secret(settings.llm_api_key)
    return service


def main(argv: list[str] | None = None) -> int:
    """Run the MemForge MCP server on stdio."""
    parser = argparse.ArgumentParser(
        prog="memforge-mcp",
        description="Serve the MemForge memory layer over the Model Context Protocol.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            f"SQLite file to use (default: MEMFORGE_DATABASE_PATH, otherwise {DEFAULT_DB_PATH})."
        ),
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings(database_path=args.db) if args.db else Settings()
        observability = build_observability(settings)
        run_mcp_server(build_operations(settings, observability), observability=observability)
    except MemforgeError as exc:
        # Never print a credential, even in a startup failure.
        print(f"memforge-mcp failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
