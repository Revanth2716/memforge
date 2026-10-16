"""M4 CLI demo tests: context compilation output and byte-for-byte determinism.

The M1-M3 demo tests live in ``tests/test_cli_demo.py`` and are left untouched;
these add the M4 section on top of the same deterministic harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli.demo import main

M4_STEPS = (
    "STEP 13 -- COMPILE context for an agent task (M4, hard token budget)",
    "STEP 14 -- COMPILE again with a tight budget (the limit is never exceeded)",
    # The M5 agent step and the M6 trace/evaluation/metrics steps were inserted
    # before the summary, so it is now step 19.
    "STEP 19 -- FINAL SUMMARY",
)


def run_demo(db_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    exit_code = main(["--db", str(db_path)])
    assert exit_code == 0
    return capsys.readouterr().out


class TestDemoContextSection:
    @pytest.fixture(autouse=True)
    def _output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.output: str = run_demo(tmp_path / "demo.sqlite", capsys)

    def test_every_milestone_still_runs(self) -> None:
        for marker in M4_STEPS:
            assert marker in self.output, marker
        assert "Demo finished successfully." in self.output
        # M1-M3 behaviour is unchanged.
        assert "Dice=0.8571" in self.output
        assert "resolved by explicit_importance" in self.output
        assert "mode=hybrid" in self.output

    def test_pipeline_is_shown(self) -> None:
        assert "MEMORY POOL  ->  RETRIEVAL  ->  CONTEXT COMPILER  ->  BUDGETED CONTEXT" in (
            self.output
        )

    def test_summary_reports_the_budget(self) -> None:
        assert "  Context Compiler\n  ----------------" in self.output
        assert "  Budget:             160" in self.output
        assert "  Candidates:         7" in self.output
        assert "  Selected:           5" in self.output
        assert "  Suppressed:         2" in self.output
        assert "  Remaining:          82" in self.output

    def test_selected_memories_show_their_section_and_utility(self) -> None:
        assert "  Selected:" in self.output
        assert "[PROFILE] utility=" in self.output
        assert "[FACTS] utility=" in self.output
        assert "[RECENT EVENTS] utility=" in self.output
        assert "[TASKS] utility=" in self.output

    def test_provenance_is_printed_for_every_block(self) -> None:
        assert "        provenance: type=PREFERENCE source=user:revanth" in self.output
        assert "type=FACT source=user:revanth confidence=0.95 importance=0.85" in self.output
        assert "confidence=1.00 importance=1.00" in self.output
        assert "retrieval=" in self.output
        assert "selection=" in self.output

    def test_duplicates_are_suppressed_with_a_reason(self) -> None:
        assert "  Suppressed:" in self.output
        assert "    duplicate_of          " in self.output
        assert "near-duplicate of" in self.output
        assert "represents this duplicate group" in self.output

    def test_the_final_context_is_rendered(self) -> None:
        assert "  Final context:" in self.output
        assert "    PROFILE\n    -------\n    - Python is the user's preferred language" in (
            self.output
        )
        assert "    FACTS\n    -----" in self.output

    def test_budget_compliance_is_printed(self) -> None:
        assert "  Final context tokens: 78 / 160" in self.output
        assert "  Within budget:        True" in self.output
        assert "  Estimator:            memforge-char-word/v1" in self.output

    def test_estimates_are_labelled_as_estimates(self) -> None:
        assert "Token counts are MemForge estimates, not provider tokenisations." in self.output

    def test_tight_budget_reports_insufficient_budget(self) -> None:
        assert "    insufficient_budget    " in self.output
        assert "estimated cost 11 tokens exceeds the remaining budget of 3 tokens" in self.output
        assert "  Final context tokens: 26 / 40" in self.output
        assert self.output.count("  Within budget:        True") == 2

    def test_compilation_did_not_destroy_anything(self) -> None:
        assert "  ACTIVE   : 9" in self.output
        assert "  ARCHIVED : 1" in self.output
        assert "  DELETED  : 1" in self.output


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

    def test_no_wall_clock_leaks_into_the_context_section(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Latency would make the output non-reproducible, so it is never printed.
        output = run_demo(tmp_path / "demo.sqlite", capsys)
        context_section = output.split("STEP 13")[1].split("STEP 15")[0]
        assert "latency" not in context_section.lower()
        assert "ms)" not in context_section
