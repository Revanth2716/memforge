"""M5 CLI demo tests: the agent step and byte-for-byte determinism.

The M1-M4 demo tests live in ``tests/test_cli_demo.py`` / ``test_m4_cli_demo.py``
and are left untouched; these cover the M5 section on the same harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli.demo import main
from app.domain.errors import LLMTransportError
from app.services.llm.custom_provider import CustomLLMProvider

M5_STEPS = (
    "STEP 15 -- AGENT turn (M5, memory-aware, deterministic mock provider)",
    # The M6 trace/evaluation/metrics steps sit between the agent turn and the
    # final summary, so the summary is now step 19.
    "STEP 19 -- FINAL SUMMARY",
)

#: A credential that must never reach stdout, stderr or a log line.
SECRET = "sk-test-key-not-printed-123456"


def run_demo(db_path: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> str:
    """Run the demo and return its output."""
    exit_code = main(["--db", str(db_path), *extra])
    assert exit_code == 0
    return capsys.readouterr().out


class TestDemoAgentStep:
    @pytest.fixture(autouse=True)
    def _output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.output: str = run_demo(tmp_path / "demo.sqlite", capsys)

    def test_the_m5_step_runs(self) -> None:
        for marker in M5_STEPS:
            assert marker in self.output, marker
        assert "Demo finished successfully." in self.output

    def test_the_pipeline_is_printed(self) -> None:
        assert "USER REQUEST  ->  MEMORY GATE  ->  RECALL (M3)  ->  CONTEXT (M4)" in (self.output)
        assert "PROVIDER  ->  ANSWER  ->  CANDIDATES  ->  STORE (M1)" in self.output

    def test_the_provider_is_the_offline_default(self) -> None:
        assert "llm provider: mock (model=mock-echo-v1, configured=True)" in self.output
        assert "no model was called" in self.output

    def test_the_memory_gate_and_context_are_shown(self) -> None:
        assert "memory:    used (" in self.output
        assert "context:   " in self.output
        assert "estimated tokens of a" in self.output
        assert "PROFILE\n      -------" in self.output

    def test_the_answer_is_printed(self) -> None:
        assert "  answer:" in self.output
        assert "last_user_message:" in self.output

    def test_candidates_are_extracted(self) -> None:
        assert "  candidate memories:" in self.output
        assert "[FACT] confidence=0.70 importance=0.50" in self.output
        assert "MemForge uses Python and FastAPI for its backend." in self.output
        assert "reason: explicit 'remember:' instruction in the request" in self.output

    def test_nothing_is_stored_without_permission(self) -> None:
        assert "stored: 0 (not permitted -- nothing was written)" in self.output
        assert "--store-candidates to let the agent persist them" in self.output
        assert "  ACTIVE   : 9" in self.output

    def test_observability_is_reported(self) -> None:
        assert "  observability:" in self.output
        assert "respond              success=True provider=mock usage=not reported" in (self.output)
        assert "extract_candidates   success=True" in self.output

    def test_no_wall_clock_or_latency_leaks(self) -> None:
        section = self.output.split("STEP 15")[1].split("STEP 16")[0]
        assert "latency_ms" not in section
        assert "Traceback" not in section


class TestDemoStorageOptIn:
    def test_storage_happens_only_with_the_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = run_demo(tmp_path / "demo.sqlite", capsys, "--store-candidates")
        assert "stored: 1 (permitted)" in output
        assert "--store-candidates to let the agent persist them" not in output
        assert "  ACTIVE   : 10" in output

    def test_the_flag_changes_only_the_storage_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        without = run_demo(tmp_path / "one.sqlite", capsys)
        with_flag = run_demo(tmp_path / "two.sqlite", capsys, "--store-candidates")
        agent_without = without.split("STEP 15")[1]
        agent_with = with_flag.split("STEP 15")[1]
        assert "stored: 0" in agent_without
        assert "stored: 1" in agent_with


class TestDemoDeterminism:
    def test_two_runs_on_two_databases_are_byte_identical(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = run_demo(tmp_path / "one.sqlite", capsys)
        second = run_demo(tmp_path / "two.sqlite", capsys)
        assert first.replace(str(tmp_path / "one.sqlite"), "<db>") == second.replace(
            str(tmp_path / "two.sqlite"), "<db>"
        )

    def test_two_runs_on_one_database_are_byte_identical(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db_path = tmp_path / "demo.sqlite"
        assert run_demo(db_path, capsys) == run_demo(db_path, capsys)

    def test_the_agent_section_is_reproducible_on_its_own(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = run_demo(tmp_path / "one.sqlite", capsys).split("STEP 15")[1]
        second = run_demo(tmp_path / "two.sqlite", capsys).split("STEP 15")[1]
        assert first == second


class TestDemoCustomProviderRejection:
    def test_an_unconfigured_custom_provider_fails_loudly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Selecting the custom provider without a base URL is an explicit
        # failure: the demo must never silently fall back and pretend that a
        # model answered.
        monkeypatch.delenv("MEMFORGE_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("MEMFORGE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MEMFORGE_LLM_MODEL", raising=False)

        exit_code = main(["--db", str(tmp_path / "demo.sqlite"), "--llm-provider", "custom"])
        stderr = capsys.readouterr().err

        assert exit_code == 1
        assert "demo failed: invalid configuration" in stderr
        assert "llm_base_url" in stderr

    def test_a_fully_configured_custom_provider_is_selected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The provider call is stubbed, so this test opens no socket: it proves
        # the selection, the failure handling and the secret hygiene, not the
        # network. A live check lives in scripts/llm_smoke_test.py and is opt-in.
        monkeypatch.setenv("MEMFORGE_LLM_BASE_URL", "https://llm.example.invalid/v1")
        monkeypatch.setenv("MEMFORGE_LLM_MODEL", "a-model")
        monkeypatch.setenv("MEMFORGE_LLM_API_KEY", SECRET)
        monkeypatch.setattr(
            CustomLLMProvider,
            "generate",
            lambda self, request: (_ for _ in ()).throw(
                LLMTransportError("the provider could not be reached", provider="custom")
            ),
        )

        exit_code = main(["--db", str(tmp_path / "demo.sqlite"), "--llm-provider", "custom"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "llm provider: custom (model=a-model, configured=True)" in captured.out
        # The failure is explicit and normalised, never a fabricated answer.
        assert "demo failed: the provider could not be reached" in captured.err
        assert SECRET not in captured.out
        assert SECRET not in captured.err
