"""CLI demo tests: correctness and byte-for-byte determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.cli.demo as demo_module
from app.cli.demo import main
from app.domain.errors import RepositoryError
from app.domain.ids import SequentialIdGenerator

M1_STEPS = (
    "STEP 1 -- CREATE three memories",
    "STEP 2 -- LIST active memories (newest first)",
    "STEP 3 -- UPDATE the task memory, then ARCHIVE the preference",
    "STEP 4 -- HISTORY of the updated task memory",
    "STEP 5 -- FORGET the archived preference and list what remains",
    "STEP 6 -- HISTORY of the forgotten memory is still available",
    "STEP 7 -- SUMMARY",
)

M2_STEPS = (
    "STEP 8 -- SCORE importance (deterministic, never stored)",
    "STEP 9 -- DETECT a near-duplicate (lexical, no embeddings)",
    "STEP 10 -- CREATE a conflicting memory, DETECT and RESOLVE it",
)

M3_STEPS = (
    "STEP 11 -- RETRIEVE memories for a query (lexical, explainable)",
    "STEP 12 -- RETRIEVE again in hybrid mode (deterministic local embeddings)",
)


def run_demo(db_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    exit_code = main(["--db", str(db_path)])
    assert exit_code == 0
    return capsys.readouterr().out


def test_demo_runs_the_full_walkthrough(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = run_demo(tmp_path / "demo.sqlite", capsys)

    for marker in M1_STEPS + M2_STEPS + M3_STEPS + ("Demo finished successfully.",):
        assert marker in output, marker

    # M1 behaviour is still demonstrated.
    assert "CREATED" in output and "UPDATED" in output and "FORGOTTEN" in output
    assert "-> 2 active memories" in output
    assert "ACTIVE   : 2" in output and "ARCHIVED : 0" in output and "DELETED  : 1" in output

    # M2 behaviour: scoring, detection, resolution, explanation, history.
    assert "explicit_importance" in output
    assert "Dice=0.8571" in output
    assert "value similarity 0.00" in output
    assert "resolved by explicit_importance" in output
    assert "relationship: SUPERSEDES ->" in output
    assert "stored importance is still 1.00, version 2, access_count 3" in output
    assert "v2  UPDATED" in output

    # M3 behaviour: explained retrieval in both modes.
    assert "mode=lexical" in output
    assert "mode=hybrid" in output
    assert "semantic=False" in output and "semantic=True" in output
    assert "relevance=" in output
    assert "duplicate_of " in output
    assert "provider: hashed-lexical/hashed-lexical-v1" in output


def test_demo_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = run_demo(tmp_path / "one.sqlite", capsys)
    second = run_demo(tmp_path / "two.sqlite", capsys)

    # Only the database file location may differ between runs.
    assert first.replace(str(tmp_path / "one.sqlite"), "<db>") == second.replace(
        str(tmp_path / "two.sqlite"), "<db>"
    )
    # Identifiers come from the deterministic generator, so they are predictable.
    assert SequentialIdGenerator(prefix="demo").next_id() in first


def test_demo_writes_a_database_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "state" / "demo.sqlite"
    run_demo(db_path, capsys)
    assert db_path.exists()
    assert db_path.stat().st_size > 0


def test_demo_can_run_twice_on_the_same_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "demo.sqlite"
    first = run_demo(db_path, capsys)
    second = run_demo(db_path, capsys)
    assert first == second


def test_demo_reports_memforge_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(_: Path, **options: object) -> int:
        raise RepositoryError("simulated storage failure")

    monkeypatch.setattr(demo_module, "run_demo", explode)
    exit_code = main(["--db", str(tmp_path / "unused.sqlite")])

    assert exit_code == 1
    assert "demo failed: simulated storage failure" in capsys.readouterr().err
