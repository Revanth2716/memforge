"""Boundary and error-path tests for the M6 evaluation and observability layers.

Everything here exercises a branch a healthy run never takes: a malformed
dataset, a case that cannot measure what it promised, a threshold outside the
unit range, a store read before anything was written. They exist because a
silent success on bad input is the failure mode a regression gate must never
have.
"""

from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import IntegrityError

import pytest

from app.domain.errors import EvaluationError, ObservabilityError
from app.domain.evaluation import (
    CaseAction,
    CaseCategory,
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    ExpectedBehavior,
    MemoryFixture,
    MetricName,
    summarise,
)
from app.domain.observability import (
    EventKind,
    EventStatus,
    Trace,
    TraceEvent,
    TraceKind,
    redact_secrets_in,
)
from app.evaluation.datasets import (
    build_case,
    build_dataset,
    build_memory,
    dataset_path,
    parse_dataset,
    read_dataset,
)
from app.evaluation.metrics import compute_metrics, credit
from app.evaluation.registry import EvaluationRegistry
from app.evaluation.report import render
from app.evaluation.run import EXIT_ERROR, main
from app.evaluation.service import EvaluationService
from app.storage.sqlite.connection import open_connection
from app.storage.sqlite.trace_store import SqliteTraceStore

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _document(**case_overrides: object) -> dict[str, object]:
    """A minimal, valid dataset document holding one case."""
    case: dict[str, object] = {
        "id": "c1",
        "title": "A case",
        "category": "retrieval",
        "action": "search",
        "query": "q",
        "fixture": [{"label": "a", "content": "c", "memory_type": "FACT"}],
        "expected": {"required": ["a"], "metrics": ["retrieval_recall"]},
    }
    case.update(case_overrides)
    return {"name": "x", "version": "v1", "cases": [case]}


def _case(**overrides: object) -> EvaluationCase:
    """A minimal, valid case with one fixture and one required label."""
    fields: dict[str, object] = {
        "id": "c1",
        "title": "A case",
        "category": CaseCategory.RETRIEVAL,
        "action": CaseAction.SEARCH,
        "query": "q",
        "fixture": (
            MemoryFixture(label="a", content="The user writes Python.", memory_type="FACT"),
        ),
        "expected": ExpectedBehavior(metrics=(MetricName.RETRIEVAL_RECALL,)),
    }
    fields.update(overrides)
    return EvaluationCase(**fields)  # type: ignore[arg-type]


def _dataset(*cases: EvaluationCase) -> EvaluationDataset:
    """Wrap cases in a dataset (at least one is always required)."""
    return EvaluationDataset(name="x", version="v1", cases=tuple(cases) or (_case(),))


def _result(*results: object) -> EvaluationResult:
    """Wrap case results in an evaluation result."""
    return EvaluationResult(dataset="x", version="v1", cases=tuple(results))  # type: ignore[arg-type]


class TestStrictDatasetValidation:
    """Every validation helper names what it rejected."""

    def test_a_top_level_array_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="JSON object"):
            parse_dataset("[]")

    def test_an_unreadable_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match=r"could not read"):
            read_dataset(tmp_path / "missing.json")

    def test_a_non_list_fixture_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="non-empty list"):
            parse_dataset(json.dumps(_document(fixture="a string")))

    def test_a_non_object_fixture_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="fixture"):
            parse_dataset(json.dumps(_document(fixture=["a string"])))

    def test_an_empty_case_id_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="must not be empty"):
            parse_dataset(json.dumps(_document(id="   ")))

    def test_a_non_object_expectation_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="expected"):
            parse_dataset(json.dumps(_document(expected="required")))

    def test_a_mistyped_id_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="id"):
            parse_dataset(json.dumps(_document(id=7)))

    def test_a_blank_title_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="title"):
            parse_dataset(json.dumps(_document(title="   ")))

    def test_a_non_numeric_confidence_is_refused(self) -> None:
        document = _document()
        document["cases"][0]["fixture"][0]["confidence"] = "high"  # type: ignore[index]
        with pytest.raises(EvaluationError, match="confidence"):
            parse_dataset(json.dumps(document))

    def test_a_confidence_outside_the_unit_range_is_refused(self) -> None:
        document = _document()
        document["cases"][0]["fixture"][0]["confidence"] = 1.5  # type: ignore[index]
        with pytest.raises(EvaluationError, match="confidence"):
            parse_dataset(json.dumps(document))

    def test_a_negative_age_is_refused(self) -> None:
        document = _document()
        document["cases"][0]["fixture"][0]["age_minutes"] = -5  # type: ignore[index]
        with pytest.raises(EvaluationError, match="age_minutes"):
            parse_dataset(json.dumps(document))

    def test_a_negative_top_k_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="top_k"):
            parse_dataset(json.dumps(_document(top_k=-1)))

    def test_a_float_top_k_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="top_k"):
            parse_dataset(json.dumps(_document(top_k=2.5)))

    def test_a_boolean_flag_must_be_a_boolean(self) -> None:
        with pytest.raises(EvaluationError, match="resolve_conflict"):
            parse_dataset(json.dumps(_document(resolve_conflict="yes")))

    def test_a_blank_source_is_refused(self) -> None:
        document = _document()
        document["cases"][0]["fixture"][0]["source"] = "  "  # type: ignore[index]
        with pytest.raises(EvaluationError, match="source"):
            parse_dataset(json.dumps(document))

    def test_a_non_string_label_list_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="required"):
            parse_dataset(
                json.dumps(_document(expected={"required": [1], "metrics": ["retrieval_recall"]}))
            )

    def test_a_blank_label_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="blank labels"):
            parse_dataset(
                json.dumps(
                    _document(expected={"required": ["  "], "metrics": ["retrieval_recall"]})
                )
            )

    def test_an_unknown_action_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="action"):
            parse_dataset(json.dumps(_document(action="teleport")))

    def test_a_non_integer_expected_count_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="expected_stored"):
            parse_dataset(
                json.dumps(
                    _document(
                        category="agent",
                        action="agent",
                        expected={
                            "expected_stored": 1.5,
                            "metrics": ["memory_write_policy_compliance"],
                        },
                    )
                )
            )

    def test_the_origin_is_reported_in_every_message(self) -> None:
        with pytest.raises(EvaluationError, match=r"custom\.json"):
            parse_dataset(json.dumps(_document(id="")), origin="custom.json")

    def test_builders_can_be_used_directly(self) -> None:
        document = _document()
        cases = document["cases"]
        assert isinstance(cases, list)
        first = cases[0]
        assert isinstance(first, dict)
        built = build_case(first, origin="inline", index=0)
        assert built.id == "c1"
        memory = build_memory(
            {"label": "a", "content": "c", "memory_type": "fact"}, origin="inline", where="f"
        )
        assert memory.memory_type == "FACT"
        assert build_dataset(document, origin="inline").reference == "x@v1"

    def test_a_reference_never_matches_a_different_version(self) -> None:
        with pytest.raises(EvaluationError, match=r"no evaluation dataset"):
            dataset_path("memory-core@v9")


class TestUnmeasurableCases:
    """A case that cannot measure what it promised fails loudly."""

    def test_a_declared_but_unmeasured_metric_fails_the_case(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(expected=ExpectedBehavior(metrics=(MetricName.CONTEXT_BUDGET_COMPLIANCE,)))
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert result.failures == (
            "measurement: the case declared the 'context_budget_compliance' metric "
            "but measured nothing for it",
        )

    def test_zero_denominators_never_inflate_a_metric(self) -> None:
        assert credit(MetricName.RETRIEVAL_RECALL, 3, 0) == (0, 0)
        assert credit(MetricName.RETRIEVAL_RECALL, 3, 5) == (3, 5)
        assert credit(MetricName.RETRIEVAL_RECALL, 9, 5) == (5, 5)
        assert credit(MetricName.RETRIEVAL_RECALL, -1, 5) == (0, 5)

    def test_a_summary_with_no_metrics_scores_one_but_says_so(self) -> None:
        from app.domain.evaluation import CaseResult

        results = _result(CaseResult(case_id="c1", category=CaseCategory.RETRIEVAL, passed=True))
        summary = summarise(_dataset(), results, [], threshold=1.0)
        assert summary.metric_score == 1.0
        assert summary.passed
        text = render(summary)
        assert "no case declared an applicable metric" in text
        assert "VERDICT: PASS" in text


class TestSummaryBoundaries:
    """Score arithmetic at its edges."""

    def test_an_empty_case_set_scores_one(self) -> None:
        summary = summarise(_dataset(), _result(), [], threshold=1.0)
        assert summary.case_score == 1.0
        assert summary.total_cases == 0
        assert summary.passed

    def test_scores_are_rounded_to_a_fixed_precision(self) -> None:
        from app.domain.evaluation import CaseResult

        results = _result(
            *(
                CaseResult(
                    case_id=str(index),
                    category=CaseCategory.RETRIEVAL,
                    passed=True,
                    numerators={MetricName.RETRIEVAL_RECALL: 1},
                    denominators={MetricName.RETRIEVAL_RECALL: 3},
                )
                for index in range(3)
            )
        )
        summary = summarise(_dataset(), results, compute_metrics(results.cases), threshold=1.0)
        assert summary.metric_score == 0.333333
        assert summary.overall == 0.666667


class TestCliBoundaries:
    """The CLI's own error paths."""

    def test_a_threshold_above_one_is_refused(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--threshold", "2.0"]) == EXIT_ERROR
        assert "between 0.0 and 1.0" in capsys.readouterr().err

    def test_a_negative_threshold_is_refused(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--threshold", "-0.5"]) == EXIT_ERROR
        assert "between 0.0 and 1.0" in capsys.readouterr().err


class TestRegistryBoundaries:
    """Two registries never share a cache."""

    def test_two_registries_do_not_share_a_cache(self) -> None:
        assert EvaluationRegistry().run_count == 0
        assert EvaluationRegistry().latest() is None


class TestRedactionBoundaries:
    """Nested payload redaction."""

    def test_a_list_is_redacted_item_by_item(self) -> None:
        payload = redact_secrets_in(["safe", "sk-live-1"], "sk-live-1")
        assert isinstance(payload, list)
        assert "sk-live-1" not in json.dumps(payload)

    def test_a_number_passes_through_untouched(self) -> None:
        assert redact_secrets_in(42, "sk-1") == 42
        assert redact_secrets_in(None, "sk-1") is None


class TestTraceStoreBoundaries:
    """Persistence edges, including reads before the first write."""

    def test_reading_an_untouched_database_returns_nothing(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        assert store.list_traces(limit=10) == ()
        assert store.all_events() == ()
        assert store.count_events() == 0
        assert store.get_trace("nope") is None
        assert store.get_events("nope") == ()

    def test_all_events_can_be_limited(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(Trace(trace_id="t1", kind=TraceKind.CLI, operation="op", started_at=NOW))
        for index in range(3):
            store.append_event(
                TraceEvent(
                    trace_id="t1",
                    event_id=f"t1-e{index}",
                    sequence=index,
                    kind=EventKind.MEMORY_GATE,
                    operation="agent.memory_gate",
                    status=EventStatus.OK,
                    timestamp=NOW,
                )
            )
        assert len(store.all_events(limit=2)) == 2
        assert len(store.all_events()) == 3

    def test_initialising_twice_is_idempotent(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.initialize()
        store.initialize()
        assert store.list_traces(limit=1) == ()

    def test_the_schema_refuses_an_unknown_trace_kind(self, db_path: Path) -> None:
        """The CHECK constraint is the first line of defence."""
        store = SqliteTraceStore(db_path)
        store.put_trace(Trace(trace_id="t1", kind=TraceKind.CLI, operation="op", started_at=NOW))
        with closing(open_connection(db_path)) as conn, pytest.raises(IntegrityError):
            conn.execute("UPDATE traces SET kind = 'teleportation'")

    def test_a_corrupt_timestamp_is_reported(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(Trace(trace_id="t1", kind=TraceKind.CLI, operation="op", started_at=NOW))
        with closing(open_connection(db_path)) as conn:
            conn.execute("UPDATE traces SET started_at = 'not a timestamp'")
        with pytest.raises(ObservabilityError, match="corrupt trace row"):
            store.get_trace("t1")

    def test_a_corrupt_metadata_column_reads_as_empty(self, db_path: Path) -> None:
        store = SqliteTraceStore(db_path)
        store.put_trace(Trace(trace_id="t1", kind=TraceKind.CLI, operation="op", started_at=NOW))
        store.append_event(
            TraceEvent(
                trace_id="t1",
                event_id="t1-e0",
                sequence=0,
                kind=EventKind.MEMORY_GATE,
                operation="agent.memory_gate",
                status=EventStatus.OK,
                timestamp=NOW,
                metadata={"k": 1},
            )
        )
        with closing(open_connection(db_path)) as conn:
            conn.execute("UPDATE trace_events SET metadata = 'not json'")
        events = store.get_events("t1")
        assert events[0].metadata == {}
