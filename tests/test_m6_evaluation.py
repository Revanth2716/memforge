"""Tests for the M6 evaluation subsystem.

These cover the dataset contract, the metric arithmetic and the CLI gate -- not
the behaviour of the shipped dataset, which is asserted separately (and once) in
:mod:`tests.test_m6_evaluation_run`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.errors import EvaluationError
from app.domain.evaluation import (
    METRIC_DEFINITIONS,
    TOP_K,
    CaseAction,
    CaseCategory,
    CaseResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    ExpectedBehavior,
    LifecycleStep,
    MemoryFixture,
    MetricName,
    MetricResult,
    summarise,
)
from app.evaluation.datasets import (
    DATASET_DIR,
    DEFAULT_DATASET,
    dataset_path,
    load_dataset,
    parse_dataset,
    read_dataset,
)
from app.evaluation.metrics import compute_metrics
from app.evaluation.registry import EvaluationRegistry
from app.evaluation.report import render

# -- builders used by several tests ------------------------------------------


def _fixture(label: str, content: str = "A memory about the user's setup.") -> MemoryFixture:
    """One fixture memory."""
    return MemoryFixture(label=label, content=content, memory_type="FACT")


def _case(**overrides: object) -> EvaluationCase:
    """A minimal, valid evaluation case."""
    fields: dict[str, object] = {
        "id": "case-1",
        "title": "A case",
        "category": CaseCategory.RETRIEVAL,
        "action": CaseAction.SEARCH,
        "query": "user setup",
        "fixture": (_fixture("a"),),
        "expected": ExpectedBehavior(required=("a",), metrics=(MetricName.RETRIEVAL_RECALL,)),
    }
    fields.update(overrides)
    return EvaluationCase(**fields)  # type: ignore[arg-type]


def _dataset(*cases: EvaluationCase, name: str = "test", version: str = "v1") -> EvaluationDataset:
    """Wrap cases in a dataset."""
    return EvaluationDataset(name=name, version=version, cases=tuple(cases) or (_case(),))


class TestDomainModels:
    """Validation of the typed models."""

    def test_a_fixture_needs_a_label_content_and_bounded_scores(self) -> None:
        with pytest.raises(EvaluationError):
            MemoryFixture(label="", content="x", memory_type="FACT")
        with pytest.raises(EvaluationError):
            MemoryFixture(label="a", content="   ", memory_type="FACT")
        with pytest.raises(EvaluationError):
            MemoryFixture(label="a", content="x", memory_type="FACT", confidence=1.5)
        with pytest.raises(EvaluationError):
            MemoryFixture(label="a", content="x", memory_type="FACT", age_minutes=-1)

    def test_an_expectation_needs_at_least_one_metric(self) -> None:
        with pytest.raises(EvaluationError):
            ExpectedBehavior(required=("a",))

    def test_a_case_must_name_only_fixtures_it_stores(self) -> None:
        with pytest.raises(EvaluationError):
            _case(
                expected=ExpectedBehavior(
                    required=("ghost",), metrics=(MetricName.RETRIEVAL_RECALL,)
                )
            )
        with pytest.raises(EvaluationError):
            _case(
                expected=ExpectedBehavior(
                    winner="ghost", metrics=(MetricName.CONFLICT_RESOLUTION_ACCURACY,)
                )
            )

    def test_a_case_needs_a_query_fixtures_and_unique_labels(self) -> None:
        with pytest.raises(EvaluationError):
            _case(query="   ")
        with pytest.raises(EvaluationError):
            _case(fixture=())
        with pytest.raises(EvaluationError):
            _case(fixture=(_fixture("a"), _fixture("a")))
        with pytest.raises(EvaluationError):
            _case(top_k=0)

    def test_lifecycle_cases_must_declare_steps_and_only_them(self) -> None:
        with pytest.raises(EvaluationError):
            _case(action=CaseAction.LIFECYCLE)
        with pytest.raises(EvaluationError):
            _case(lifecycle_steps=(LifecycleStep.UPDATE,))
        case = _case(
            action=CaseAction.LIFECYCLE,
            lifecycle_steps=(LifecycleStep.UPDATE, LifecycleStep.FORGET),
            expected=ExpectedBehavior(
                expected_versions=3, metrics=(MetricName.HISTORY_PRESERVATION,)
            ),
        )
        assert case.lifecycle_steps == (LifecycleStep.UPDATE, LifecycleStep.FORGET)

    def test_a_dataset_needs_a_name_version_and_unique_case_ids(self) -> None:
        with pytest.raises(EvaluationError):
            EvaluationDataset(name="", version="v1", cases=(_case(),))
        with pytest.raises(EvaluationError):
            EvaluationDataset(name="x", version="", cases=(_case(),))
        with pytest.raises(EvaluationError):
            EvaluationDataset(name="x", version="v1", cases=())
        with pytest.raises(EvaluationError):
            EvaluationDataset(name="x", version="v1", cases=(_case(), _case()))

    def test_a_dataset_reports_its_reference_and_categories(self) -> None:
        dataset = _dataset(
            _case(),
            _case(id="case-2", category=CaseCategory.CONTEXT, action=CaseAction.COMPILE),
        )
        assert dataset.reference == "test@v1"
        assert dataset.categories == (CaseCategory.RETRIEVAL, CaseCategory.CONTEXT)
        assert len(dataset.cases_for(CaseCategory.RETRIEVAL)) == 1

    def test_every_metric_has_a_definition(self) -> None:
        assert set(METRIC_DEFINITIONS) == set(MetricName)
        for definition in METRIC_DEFINITIONS.values():
            assert definition.strip()
            assert "\n" not in definition

    def test_a_metric_result_is_a_ratio_of_real_counts(self) -> None:
        result = MetricResult(name=MetricName.RETRIEVAL_RECALL, numerator=3, denominator=4)
        assert result.score == 0.75
        assert result.applicable
        assert (
            MetricResult(name=MetricName.RETRIEVAL_RECALL, numerator=0, denominator=0).score == 1.0
        )
        assert not MetricResult(
            name=MetricName.RETRIEVAL_RECALL, numerator=0, denominator=0
        ).applicable


class TestSummary:
    """Pass/fail, scores and the threshold."""

    def _summary(self, **overrides: object) -> object:
        """Build a summary from a small, fully passing result."""
        case = CaseResult(
            case_id="case-1",
            category=CaseCategory.RETRIEVAL,
            passed=True,
            numerators={MetricName.RETRIEVAL_RECALL: 1},
            denominators={MetricName.RETRIEVAL_RECALL: 1},
        )
        result = EvaluationResult(dataset="test", version="v1", cases=(case,))
        fields: dict[str, object] = {"threshold": 1.0}
        fields.update(overrides)
        return summarise(_dataset(), result, compute_metrics(result.cases), **fields)  # type: ignore[arg-type]

    def test_a_fully_passing_run_meets_a_full_threshold(self) -> None:
        summary = self._summary()
        assert summary.passed  # type: ignore[attr-defined]
        assert summary.case_score == 1.0  # type: ignore[attr-defined]
        assert summary.overall == 1.0  # type: ignore[attr-defined]

    def test_a_graded_metric_below_one_does_not_fail_the_gate(self) -> None:
        case = CaseResult(
            case_id="case-1",
            category=CaseCategory.RETRIEVAL,
            passed=True,
            numerators={MetricName.RETRIEVAL_PRECISION: 3},
            denominators={MetricName.RETRIEVAL_PRECISION: 4},
        )
        result = EvaluationResult(dataset="test", version="v1", cases=(case,))
        summary = summarise(_dataset(), result, compute_metrics(result.cases), threshold=1.0)
        assert summary.passed
        assert summary.metric_score < 1.0
        assert summary.overall < 1.0

    def test_a_failed_case_fails_the_gate(self) -> None:
        failed = CaseResult(
            case_id="case-1", category=CaseCategory.RETRIEVAL, passed=False, failures=("recall: x",)
        )
        result = EvaluationResult(dataset="test", version="v1", cases=(failed,))
        summary = summarise(_dataset(), result, compute_metrics(result.cases), threshold=1.0)
        assert not summary.passed
        assert summary.failed_cases == 1
        assert summary.failures[0].case_id == "case-1"

    def test_a_relaxed_threshold_tolerates_a_known_failure(self) -> None:
        results = [
            CaseResult(case_id="a", category=CaseCategory.RETRIEVAL, passed=True),
            CaseResult(case_id="b", category=CaseCategory.RETRIEVAL, passed=False),
        ]
        result = EvaluationResult(dataset="test", version="v1", cases=tuple(results))
        metrics = compute_metrics(result.cases)
        assert summarise(_dataset(), result, metrics, threshold=1.0).passed is False
        assert summarise(_dataset(), result, metrics, threshold=0.5).passed is True

    def test_a_threshold_outside_the_unit_range_is_refused(self) -> None:
        with pytest.raises(EvaluationError):
            self._summary(threshold=1.5)
        with pytest.raises(EvaluationError):
            self._summary(threshold=-0.1)

    def test_the_wire_shape_is_complete_and_sorted(self) -> None:
        payload = self._summary().to_dict()  # type: ignore[attr-defined]
        assert payload["dataset"] == "test"
        assert payload["categories"]["retrieval"] == {"passed": 1, "total": 1}
        metric = payload["metrics"][0]
        assert set(metric) == {
            "name",
            "definition",
            "numerator",
            "denominator",
            "score",
            "failures",
        }


class TestMetricAggregation:
    """The arithmetic behind every reported number."""

    def test_every_metric_is_returned_even_when_unused(self) -> None:
        results = compute_metrics([])
        assert {metric.name for metric in results} == set(MetricName)
        assert all(not metric.applicable for metric in results)

    def test_counts_are_summed_not_averaged(self) -> None:
        results = compute_metrics(
            [
                CaseResult(
                    case_id="a",
                    category=CaseCategory.RETRIEVAL,
                    passed=True,
                    numerators={MetricName.RETRIEVAL_RECALL: 2},
                    denominators={MetricName.RETRIEVAL_RECALL: 3},
                ),
                CaseResult(
                    case_id="b",
                    category=CaseCategory.RETRIEVAL,
                    passed=True,
                    numerators={MetricName.RETRIEVAL_RECALL: 1},
                    denominators={MetricName.RETRIEVAL_RECALL: 1},
                ),
            ]
        )
        recall = next(m for m in results if m.name is MetricName.RETRIEVAL_RECALL)
        assert (recall.numerator, recall.denominator) == (3, 4)
        assert recall.score == 0.75

    def test_zero_credit_cases_are_counted_in_the_denominator(self) -> None:
        results = compute_metrics(
            [
                CaseResult(
                    case_id="a",
                    category=CaseCategory.RETRIEVAL,
                    passed=False,
                    numerators={MetricName.RETRIEVAL_RECALL: 0},
                    denominators={MetricName.RETRIEVAL_RECALL: 1},
                )
            ]
        )
        recall = next(m for m in results if m.name is MetricName.RETRIEVAL_RECALL)
        assert recall.score == 0.0
        assert recall.failures == ("a",)

    def test_a_partially_credited_case_is_listed_under_that_metric(self) -> None:
        results = compute_metrics(
            [
                CaseResult(
                    case_id="a",
                    category=CaseCategory.RETRIEVAL,
                    passed=False,
                    failures=("precision: 1 of 4 returned memories were not relevant",),
                    numerators={MetricName.RETRIEVAL_PRECISION: 3},
                    denominators={MetricName.RETRIEVAL_PRECISION: 4},
                )
            ]
        )
        precision = next(m for m in results if m.name is MetricName.RETRIEVAL_PRECISION)
        assert precision.failures == ("a",)
        assert precision.score == 0.75

    def test_a_fully_credited_case_is_not_listed_even_if_another_check_failed(self) -> None:
        results = compute_metrics(
            [
                CaseResult(
                    case_id="a",
                    category=CaseCategory.RETRIEVAL,
                    passed=False,
                    failures=("top_k: no required memory appeared in the first 3 results",),
                    numerators={MetricName.RETRIEVAL_RECALL: 2},
                    denominators={MetricName.RETRIEVAL_RECALL: 2},
                )
            ]
        )
        recall = next(m for m in results if m.name is MetricName.RETRIEVAL_RECALL)
        assert recall.failures == ()
        assert recall.score == 1.0

    def test_a_case_with_no_measurement_does_not_inflate_a_metric(self) -> None:
        results = compute_metrics(
            [
                CaseResult(
                    case_id="a",
                    category=CaseCategory.RETRIEVAL,
                    passed=True,
                    numerators={},
                    denominators={},
                )
            ]
        )
        assert all(not metric.applicable for metric in results)


class TestDatasetLoading:
    """Strict parsing of the version-controlled dataset."""

    def test_the_shipped_dataset_loads(self) -> None:
        dataset = load_dataset()
        assert dataset.name == "memory-core"
        assert dataset.version == "v1"
        assert dataset.reference == "memory-core@v1"
        assert len(dataset.cases) >= 30
        assert set(dataset.categories) == set(CaseCategory)

    def test_the_shipped_dataset_file_is_valid_json(self) -> None:
        payload = json.loads(read_dataset(dataset_path(DEFAULT_DATASET)))
        assert payload["name"] == "memory-core"
        assert isinstance(payload["cases"], list)

    def test_a_reference_resolves_to_the_shipped_file(self) -> None:
        assert dataset_path("memory-core").name == "memory-core-v1.json"
        assert dataset_path("memory-core@v1").name == "memory-core-v1.json"
        assert dataset_path("memory-core-v1").name == "memory-core-v1.json"

    def test_an_explicit_path_is_taken_as_given(self, tmp_path: Path) -> None:
        target = tmp_path / "custom.json"
        target.write_text(read_dataset(dataset_path(DEFAULT_DATASET)), encoding="utf-8")
        assert load_dataset(str(target)).reference == "memory-core@v1"

    def test_an_unknown_dataset_is_reported_with_the_available_names(self) -> None:
        with pytest.raises(EvaluationError) as failure:
            dataset_path("does-not-exist")
        assert "memory-core-v1" in str(failure.value)

    def test_malformed_documents_are_refused(self) -> None:
        with pytest.raises(EvaluationError):
            parse_dataset("not json")
        with pytest.raises(EvaluationError):
            parse_dataset("[]")
        with pytest.raises(EvaluationError):
            parse_dataset('{"name": "x", "version": "v1"}')
        with pytest.raises(EvaluationError):
            parse_dataset('{"name": "x", "version": "v1", "cases": []}')

    def test_missing_and_mistyped_fields_are_named(self) -> None:
        base = {
            "name": "x",
            "version": "v1",
            "cases": [
                {
                    "id": "c1",
                    "title": "T",
                    "category": "retrieval",
                    "action": "search",
                    "query": "q",
                    "fixture": [{"label": "a", "content": "c", "memory_type": "FACT"}],
                    "expected": {
                        "required": ["a"],
                        "metrics": ["retrieval_recall"],
                    },
                }
            ],
        }
        parse_dataset(json.dumps(base))

        no_id = json.loads(json.dumps(base))
        del no_id["cases"][0]["id"]
        with pytest.raises(EvaluationError) as failure:
            parse_dataset(json.dumps(no_id), origin="d.json")
        assert "id" in str(failure.value) and "d.json" in str(failure.value)

        bad_category = json.loads(json.dumps(base))
        bad_category["cases"][0]["category"] = "telepathy"
        with pytest.raises(EvaluationError) as failure:
            parse_dataset(json.dumps(bad_category))
        assert "retrieval" in str(failure.value)

    def test_an_unknown_metric_is_refused(self) -> None:
        document = json.dumps(
            {
                "name": "x",
                "version": "v1",
                "cases": [
                    {
                        "id": "c1",
                        "title": "T",
                        "category": "retrieval",
                        "action": "search",
                        "query": "q",
                        "fixture": [{"label": "a", "content": "c", "memory_type": "FACT"}],
                        "expected": {"metrics": ["vibes"]},
                    }
                ],
            }
        )
        with pytest.raises(EvaluationError) as failure:
            parse_dataset(document)
        assert "vibes" in str(failure.value)

    def test_a_case_needs_at_least_one_metric(self) -> None:
        with pytest.raises(EvaluationError):
            parse_dataset(
                json.dumps(
                    {
                        "name": "x",
                        "version": "v1",
                        "cases": [
                            {
                                "id": "c1",
                                "title": "T",
                                "category": "retrieval",
                                "action": "search",
                                "query": "q",
                                "fixture": [{"label": "a", "content": "c", "memory_type": "FACT"}],
                                "expected": {"metrics": []},
                            }
                        ],
                    }
                )
            )

    def test_an_unknown_lifecycle_step_is_refused(self) -> None:
        with pytest.raises(EvaluationError):
            parse_dataset(
                json.dumps(
                    {
                        "name": "x",
                        "version": "v1",
                        "cases": [
                            {
                                "id": "c1",
                                "title": "T",
                                "category": "lifecycle",
                                "action": "lifecycle",
                                "query": "q",
                                "lifecycle_steps": ["detonate"],
                                "fixture": [{"label": "a", "content": "c", "memory_type": "FACT"}],
                                "expected": {
                                    "expected_versions": 2,
                                    "metrics": ["history_preservation"],
                                },
                            }
                        ],
                    }
                )
            )

    def test_every_shipped_case_reference_is_a_fixture(self) -> None:
        for case in load_dataset().cases:
            labels = {memory.label for memory in case.fixture}
            referenced = (
                case.expected.required
                + case.expected.forbidden
                + case.expected.duplicates_suppressed
                + case.expected.stale
                + case.expected.selected
            )
            assert set(referenced) <= labels, case.id


class TestReport:
    """The rendered report."""

    def _summary(self, **overrides: object):  # type: ignore[no-untyped-def]
        """Build a small summary with one failing case."""
        results = [
            CaseResult(
                case_id="ok-case",
                category=CaseCategory.RETRIEVAL,
                passed=True,
                numerators={MetricName.RETRIEVAL_RECALL: 1},
                denominators={MetricName.RETRIEVAL_RECALL: 1},
            ),
            CaseResult(
                case_id="bad-case",
                category=CaseCategory.CONTEXT,
                passed=False,
                failures=("budget: estimated 200 tokens against a budget of 100",),
                numerators={MetricName.CONTEXT_BUDGET_COMPLIANCE: 0},
                denominators={MetricName.CONTEXT_BUDGET_COMPLIANCE: 1},
            ),
        ]
        dataset = _dataset()
        result = EvaluationResult(
            dataset=dataset.name, version=dataset.version, cases=tuple(results)
        )
        fields: dict[str, object] = {"threshold": 1.0}
        fields.update(overrides)
        return summarise(dataset, result, compute_metrics(result.cases), **fields)  # type: ignore[arg-type]

    def test_the_report_contains_every_required_section(self) -> None:
        text = render(self._summary())
        for marker in (
            "EVALUATION REPORT",
            "Dataset      : test@v1",
            "Cases        : 2",
            "Passed       : 1",
            "Failed       : 1",
            "Categories:",
            "Metrics (measured):",
            "retrieval_recall",
            "context_budget_compliance",
            "Failed cases:",
            "bad-case",
            "budget: estimated 200 tokens",
            "VERDICT: FAIL",
        ):
            assert marker in text, marker

    def test_the_report_states_its_own_limits(self) -> None:
        text = render(self._summary())
        assert "no LLM-as-a-judge was involved" in text
        assert "not a claim about memory intelligence" in text
        assert "Scores are not comparable across datasets" in text

    def test_verbose_mode_prints_every_definition(self) -> None:
        text = render(self._summary(), verbose=True)
        for metric in (MetricName.RETRIEVAL_RECALL, MetricName.CONTEXT_BUDGET_COMPLIANCE):
            assert f"{metric.value}:" in text
            assert METRIC_DEFINITIONS[metric] in text

    def test_the_report_is_deterministic(self) -> None:
        assert render(self._summary()) == render(self._summary())

    def test_a_clean_run_reports_pass(self) -> None:
        results = [
            CaseResult(
                case_id="ok",
                category=CaseCategory.RETRIEVAL,
                passed=True,
                numerators={MetricName.RETRIEVAL_RECALL: 1},
                denominators={MetricName.RETRIEVAL_RECALL: 1},
            )
        ]
        dataset = _dataset()
        result = EvaluationResult(
            dataset=dataset.name, version=dataset.version, cases=tuple(results)
        )
        summary = summarise(dataset, result, compute_metrics(result.cases), threshold=1.0)
        text = render(summary)
        assert "VERDICT: PASS" in text
        assert "Failed cases: none" in text

    def test_the_top_k_depth_is_documented_and_fixed(self) -> None:
        assert TOP_K == 3
        assert str(TOP_K) in METRIC_DEFINITIONS[MetricName.TOP_K_HIT_RATE]


class TestRegistry:
    """The in-process evaluation cache."""

    def test_a_run_is_cached_by_reference(self, tmp_path: Path) -> None:
        registry = EvaluationRegistry()
        first = registry.run(DEFAULT_DATASET)
        assert registry.run_count == 1
        second = registry.run(DEFAULT_DATASET)
        assert second is first
        assert registry.run_count == 1
        assert registry.get("memory-core@v1") is first
        assert registry.latest() is first

    def test_forcing_reruns(self) -> None:
        registry = EvaluationRegistry()
        registry.run(DEFAULT_DATASET)
        registry.run(DEFAULT_DATASET, force=True)
        assert registry.run_count == 2

    def test_an_empty_registry_has_no_latest(self) -> None:
        assert EvaluationRegistry().latest() is None
        assert EvaluationRegistry().get("memory-core@v1") is None

    def test_an_unknown_dataset_raises(self) -> None:
        with pytest.raises(EvaluationError):
            EvaluationRegistry().run("does-not-exist")

    def test_the_shipped_dataset_directory_holds_one_file(self) -> None:
        assert [path.stem for path in DATASET_DIR.glob("*.json")] == ["memory-core-v1"]
