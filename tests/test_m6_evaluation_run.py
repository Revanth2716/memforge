"""Tests for the shipped evaluation dataset, its CLI gate and the M6 demos.

The shipped dataset is asserted once, here, so a regression anywhere in M1-M5
shows up as a named case rather than as a mysteriously lower number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.evaluation import CaseCategory, EvaluationSummary, MetricName
from app.evaluation.datasets import DEFAULT_DATASET, load_dataset
from app.evaluation.report import render
from app.evaluation.run import EXIT_ERROR, EXIT_FAILED, EXIT_PASSED, main
from app.evaluation.service import EvaluationService
from app.observability.demo import main as demo_main
from app.observability.demo import run_demo as run_observability_demo


@pytest.fixture(scope="module")
def summary() -> EvaluationSummary:
    """Run the shipped dataset once for the whole module."""
    return EvaluationService().evaluate(load_dataset(DEFAULT_DATASET))


class TestShippedDataset:
    """The version-controlled dataset itself."""

    def test_every_case_holds_its_expectations(self, summary) -> None:  # type: ignore[no-untyped-def]
        assert summary.failed_cases == 0, [case.to_dict() for case in summary.failures]
        assert summary.passed
        assert summary.total_cases >= 30

    def test_every_category_is_exercised(self, summary) -> None:  # type: ignore[no-untyped-def]
        assert set(summary.categories) == set(CaseCategory)
        for name, (passed, total) in summary.categories.items():
            assert total >= 4, f"{name} has only {total} case(s)"
            assert passed == total

    def test_every_metric_is_measured_at_least_once(self, summary) -> None:  # type: ignore[no-untyped-def]
        measured = {metric.name for metric in summary.metrics if metric.applicable}
        assert measured == set(MetricName)

    def test_every_measured_metric_has_a_real_denominator(self, summary) -> None:  # type: ignore[no-untyped-def]
        for metric in summary.metrics:
            if metric.applicable:
                assert metric.denominator > 0
                assert 0 <= metric.numerator <= metric.denominator
                assert 0.0 <= metric.score <= 1.0

    def test_only_the_graded_metric_may_sit_below_one(self, summary) -> None:  # type: ignore[no-untyped-def]
        below = {
            metric.name for metric in summary.metrics if metric.applicable and metric.score < 1.0
        }
        assert below <= {MetricName.RETRIEVAL_PRECISION}, below

    def test_running_twice_is_identical(self) -> None:
        service = EvaluationService()
        dataset = load_dataset()
        assert service.evaluate(dataset).to_dict() == service.evaluate(dataset).to_dict()

    def test_the_report_is_byte_identical_between_runs(self) -> None:
        service = EvaluationService()
        dataset = load_dataset()
        first = render(service.evaluate(dataset))
        second = render(service.evaluate(dataset))
        assert first == second


class TestEvaluationCli:
    """``python -m app.evaluation.run``."""

    def test_the_gate_passes_on_the_shipped_dataset(self, capsys) -> None:  # type: ignore[no-untyped-def]
        assert main([]) == EXIT_PASSED
        out = capsys.readouterr().out
        assert "VERDICT: PASS" in out
        assert "memory-core@v1" in out

    def test_verbose_mode_prints_every_definition(self, capsys) -> None:  # type: ignore[no-untyped-def]
        assert main(["--verbose"]) == EXIT_PASSED
        out = capsys.readouterr().out
        assert "Metric definitions:" in out
        assert "retrieval_recall:" in out

    def test_json_mode_emits_a_machine_readable_summary(self, capsys) -> None:  # type: ignore[no-untyped-def]
        assert main(["--json"]) == EXIT_PASSED
        out = capsys.readouterr().out
        start = out.index("{", out.index("VERDICT"))
        payload = json.loads(out[start:])
        assert payload["reference"] == "memory-core@v1"
        assert payload["passed"] is True
        assert payload["total_cases"] == payload["passed_cases"]
        assert len(payload["metrics"]) == len(MetricName)

    def test_a_failing_dataset_exits_non_zero(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        broken = tmp_path / "broken.json"
        broken.write_text(
            json.dumps(
                {
                    "name": "broken",
                    "version": "v1",
                    "cases": [
                        {
                            "id": "impossible-1",
                            "title": "A case that cannot hold",
                            "category": "retrieval",
                            "action": "search",
                            "query": "kubernetes ingress",
                            "fixture": [
                                {
                                    "label": "unrelated",
                                    "content": "The user repairs bicycles at the weekend.",
                                    "memory_type": "OBSERVATION",
                                }
                            ],
                            "expected": {
                                "required": ["unrelated"],
                                "metrics": ["retrieval_recall"],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert main(["--dataset", str(broken)]) == EXIT_FAILED
        out = capsys.readouterr().out
        assert "VERDICT: FAIL" in out
        assert "impossible-1" in out
        assert "recall:" in out

    def test_a_relaxed_threshold_accepts_a_known_failure(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        partial = tmp_path / "partial.json"
        partial.write_text(
            json.dumps(
                {
                    "name": "partial",
                    "version": "v1",
                    "cases": [
                        {
                            "id": "reachable-1",
                            "title": "A case that holds",
                            "category": "retrieval",
                            "action": "search",
                            "query": "bicycle repair",
                            "fixture": [
                                {
                                    "label": "match",
                                    "content": "The user repairs bicycles at the weekend.",
                                    "memory_type": "OBSERVATION",
                                }
                            ],
                            "expected": {
                                "required": ["match"],
                                "metrics": ["retrieval_recall"],
                            },
                        },
                        {
                            "id": "impossible-1",
                            "title": "A case that cannot hold",
                            "category": "retrieval",
                            "action": "search",
                            "query": "kubernetes ingress",
                            "fixture": [
                                {
                                    "label": "unrelated",
                                    "content": "The user drinks oat milk with breakfast.",
                                    "memory_type": "OBSERVATION",
                                }
                            ],
                            "expected": {
                                "required": ["unrelated"],
                                "metrics": ["retrieval_recall"],
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert main(["--dataset", str(partial)]) == EXIT_FAILED
        assert main(["--dataset", str(partial), "--threshold", "0.5"]) == EXIT_PASSED
        assert "impossible-1" in capsys.readouterr().out

    def test_an_unknown_dataset_is_refused(self, capsys) -> None:  # type: ignore[no-untyped-def]
        assert main(["--dataset", "does-not-exist"]) == EXIT_ERROR
        assert "no evaluation dataset matches" in capsys.readouterr().err


class TestObservabilityDemo:
    """``python -m app.observability.demo``."""

    def test_the_demo_runs_and_reports_a_real_trace(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        code = run_observability_demo(tmp_path / "demo.sqlite")
        assert code == 0
        out = capsys.readouterr().out
        for marker in (
            "TRACE trace-0001",
            "agent.request",
            "memory.gate",
            "retrieval",
            "context.compile",
            "llm.generate",
            "memory.candidate",
            "memory.write",
            "response",
            "SKIPPED",
            "ALLOWED",
            "METRICS",
            "reported_by_provider=0",
            "SAFETY",
        ):
            assert marker in out, marker

    def test_the_demo_never_prints_a_prompt_or_an_answer(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        run_observability_demo(tmp_path / "demo.sqlite")
        out = capsys.readouterr().out.lower()
        for marker in ("authorization", "bearer ", "api_key", "api-key", "remember:"):
            assert marker not in out, marker

    def test_the_cli_entry_point_runs(self, tmp_path: Path) -> None:
        assert demo_main(["--db", str(tmp_path / "demo.sqlite")]) == 0

    def test_the_first_turn_can_be_given_write_permission(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both turns then report ALLOWED, which is the interesting contrast."""
        assert demo_main(["--db", str(tmp_path / "demo.sqlite"), "--store-candidates"]) == 0
        out = capsys.readouterr().out
        first_turn = out.split("TURN 2")[0]
        assert "ALLOWED" in first_turn
        assert "stored=1" in first_turn

    def test_two_runs_print_identical_output(self, tmp_path: Path) -> None:
        import io
        from contextlib import redirect_stdout

        # The same path twice: the demo recreates the database from scratch, so
        # any residual state would show up as a difference.
        target = tmp_path / "demo.sqlite"
        outputs = []
        for _ in range(2):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_observability_demo(target)
            outputs.append(buffer.getvalue())
        assert outputs[0] == outputs[1]
