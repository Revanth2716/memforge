"""OPTIONAL live smoke test for a custom OpenAI-compatible API (M5).

This script is **never** part of the test suite. It only runs when the operator
provides real credentials, and it refuses to start otherwise:

    $env:MEMFORGE_LLM_BASE_URL = "https://your-host/v1"
    $env:MEMFORGE_LLM_MODEL    = "your-model"
    $env:MEMFORGE_LLM_API_KEY  = "your-key"
    python scripts/llm_smoke_test.py

What it verifies
----------------
1. the provider is configured and can reach the endpoint;
2. a plain text generation round-trips;
3. a structured (schema-constrained) generation round-trips;
4. a memory-aware turn compiles context (M4) and sends it to the provider;
5. usage is reported **only** when the provider actually sent it.

Safety
------
* the API key is read from the environment only and is **never printed** -- not
  in the output, not in an error message, not in a repr;
* the script refuses to run without the required variables;
* every figure it prints is either measured locally (latency) or copied from the
  provider's own response (usage). Nothing is inferred.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.domain.agent import (  # noqa: E402
    AgentRequest,
)
from app.domain.errors import LLMProviderError  # noqa: E402
from app.domain.ids import SequentialIdGenerator  # noqa: E402
from app.domain.llm import LLMMessage, LLMRequest, redact_text  # noqa: E402
from app.domain.models import MemoryType  # noqa: E402
from app.domain.time import ManualClock  # noqa: E402
from app.services.agent_service import AgentService  # noqa: E402
from app.services.context_service import ContextCompilerService  # noqa: E402
from app.services.llm.factory import build_llm_provider  # noqa: E402
from app.services.memory_operations import MemoryOperationsService  # noqa: E402
from app.services.memory_service import MemoryService  # noqa: E402
from app.services.retrieval_service import MemoryRetrievalService  # noqa: E402
from app.storage.sqlite.embedding_store import SqliteEmbeddingStore  # noqa: E402
from app.storage.sqlite.repository import SqliteMemoryRepository  # noqa: E402

#: Environment variables required before the script will run.
REQUIRED_ENV = ("MEMFORGE_LLM_BASE_URL", "MEMFORGE_LLM_MODEL", "MEMFORGE_LLM_API_KEY")


class CheckFailed(AssertionError):
    """A mandatory live check did not hold."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`CheckFailed` with a readable message."""
    if not condition:
        raise CheckFailed(message)


def safe(text: str, secret: str) -> str:
    """Redact the credential from anything about to be printed."""
    return redact_text(text, secret)


def build_stack(directory: Path, settings: Settings):
    """Build the full service stack over a throwaway database."""
    db_path = directory / "llm-smoke.sqlite"
    clock = ManualClock()
    repository = SqliteMemoryRepository(db_path)
    repository.initialize()
    memories = MemoryService(
        repository, clock=clock, id_generator=SequentialIdGenerator(prefix="llm")
    )
    retrieval = MemoryRetrievalService(
        repository,
        clock=clock,
        embedding_store=SqliteEmbeddingStore(db_path),
        ranking=settings.ranking_config(),
    )
    operations = MemoryOperationsService(memories, retrieval, ContextCompilerService(retrieval))
    return operations


async def check_connectivity(provider, secret: str) -> None:
    """The provider is configured and answers a trivial request."""
    require(provider.is_configured, "the custom provider is not fully configured")
    response = provider.generate(
        LLMRequest(
            messages=(LLMMessage.user("Reply with the single word: ready"),),
            max_tokens=16,
            temperature=0.0,
            operation="smoke_connectivity",
        )
    )
    require(response.text.strip(), "the provider returned an empty answer")
    require(response.provider == "custom", "the provider reported a different identity")
    require(response.model, "the provider did not report a model name")
    print(f"    model reported by provider: {response.model}")
    print(f"    finish_reason: {response.finish_reason or 'not reported'}")
    print(f"    answer: {safe(response.text[:120], secret)!r}")


async def check_structured(provider, secret: str) -> None:
    """A schema-constrained generation round-trips into a typed payload."""
    response = provider.generate_structured(
        LLMRequest(
            messages=(
                LLMMessage.system("Answer with JSON only."),
                LLMMessage.user('Report {"ok": true} as JSON.'),
            ),
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            schema_name="smoke_status",
            temperature=0.0,
            operation="smoke_structured",
        )
    )
    require(response.structured is not None, "no structured payload was parsed")
    require(isinstance(response.structured.get("ok"), bool), "the payload is not typed")
    print(f"    structured payload keys: {sorted(response.structured)}")


async def check_memory_aware(provider, operations, secret: str) -> None:
    """A memory-aware turn compiles context (M4) and reaches the provider."""
    operations.store(
        "User prefers Python for backend work.",
        MemoryType.PROFILE,
        source="user:smoke",
        confidence=0.95,
        importance=0.9,
    )
    agent = AgentService(operations, provider)
    result = await asyncio.to_thread(
        agent.respond,
        AgentRequest(
            request="Which backend language does the user prefer?",
            token_budget=300,
            top_k=5,
            temperature=0.0,
            max_tokens=200,
        ),
    )
    require(result.context_text, "no memory context was compiled")
    require(result.used_memory, "the turn reports no memory use")
    require(result.context_estimated_tokens > 0, "no context token estimate was produced")
    require(result.answer.strip(), "the provider returned an empty answer")
    print(f"    memory used: {result.gate.use.value}")
    print(f"    context: {result.context_estimated_tokens} estimated tokens")
    print(f"    blocks: {[block.memory_id[:8] for block in result.context_blocks]}")
    print(f"    answer: {safe(result.answer[:160], secret)!r}")


def check_usage(reported: object) -> None:
    """Usage is echoed only when the provider supplied it."""
    if reported is None:
        print("    usage: not reported by this provider (recorded as null, not zero)")
    else:
        print(f"    usage (as reported): {reported}")


async def run() -> int:
    """Run the live checks; return a process exit code."""
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print("This smoke test is OPTIONAL and needs real credentials.")
        print(f"Missing environment variables: {', '.join(missing)}")
        print("\nSet them and re-run, for example:")
        print("  $env:MEMFORGE_LLM_BASE_URL = 'https://your-host/v1'")
        print("  $env:MEMFORGE_LLM_MODEL    = 'your-model'")
        print("  $env:MEMFORGE_LLM_API_KEY  = 'your-key'")
        print("\nThe API key is never printed by this script.")
        return 2

    settings = Settings(llm_provider="custom", log_level="WARNING")
    provider = build_llm_provider(settings)
    secret = settings.llm_api_key
    if not isinstance(provider, object) or provider.name != "custom":
        print(f"expected the custom provider, got {provider.name!r}")
        return 1

    failures: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="memforge-llm-") as directory:
        operations = build_stack(Path(directory), settings)
        checks = (
            ("provider connectivity", lambda: check_connectivity(provider, secret)),
            ("structured response", lambda: check_structured(provider, secret)),
            (
                "memory-aware request",
                lambda: check_memory_aware(provider, operations, secret),
            ),
        )
        print(f"MemForge live LLM smoke test -> {settings.llm_base_url}")
        print("(the API key is never printed)")
        print("=" * 60)
        usage: object = None
        for index, (title, check) in enumerate(checks, start=1):
            try:
                await check()
            except CheckFailed as failure:
                failures.append((title, str(failure)))
                print(f"[{index}/{len(checks)}] FAIL  {title}")
                print(f"          {safe(str(failure), secret)}")
                continue
            except LLMProviderError as error:
                # A normalised provider failure is a legitimate, explicit result.
                failures.append((title, f"LLM_{error.kind}: {error}"))
                print(f"[{index}/{len(checks)}] FAIL  {title}")
                print(f"          provider error ({error.kind}): {safe(str(error), secret)}")
                print("          MemForge returned an explicit failure; no answer was invented.")
                continue
            print(f"[{index}/{len(checks)}] PASS  {title}")
        try:
            probe = provider.generate(
                LLMRequest(messages=(LLMMessage.user("Reply with: ok"),), max_tokens=8)
            )
            usage = probe.usage
        except LLMProviderError as error:
            print(f"    usage probe failed ({error.kind}); usage stays null")

    print("=" * 60)
    check_usage(usage)
    passed = len(checks) - len(failures)
    print(f"{passed}/{len(checks)} live checks passed")
    if failures:
        print("\nFailures:")
        for title, message in failures:
            print(f"  - {title}: {safe(message, secret)}")
        return 1
    print("Live smoke test finished successfully.")
    return 0


def main() -> int:
    """Entry point."""
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
