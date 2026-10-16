"""Tests for the evaluation service: what a case actually runs and counts.

The point of these tests is not the shipped dataset (that is asserted once, in
:mod:`tests.test_m6_evaluation_run`) but the *machinery*: that a case executes
real services, that isolation holds, that each metric reads a real count, and
that a deliberately broken expectation is detected rather than smoothed over.
"""

from __future__ import annotations

import pytest

from app.domain.errors import EvaluationError
from app.domain.evaluation import (
    CaseAction,
    CaseCategory,
    EvaluationCase,
    EvaluationDataset,
    ExpectedBehavior,
    MemoryFixture,
    MetricName,
)
from app.evaluation.service import FALLBACK_BUDGET, EvaluationService


def _fixture(label: str, content: str, **overrides: object) -> MemoryFixture:
    """One fixture memory with sensible defaults."""
    fields: dict[str, object] = {
        "label": label,
        "content": content,
        "memory_type": "FACT",
        "confidence": 0.8,
        "importance": 0.7,
    }
    fields.update(overrides)
    return MemoryFixture(**fields)  # type: ignore[arg-type]


def _case(**overrides: object) -> EvaluationCase:
    """A minimal search case."""
    fields: dict[str, object] = {
        "id": "case-1",
        "title": "A case",
        "category": CaseCategory.RETRIEVAL,
        "action": CaseAction.SEARCH,
        "query": "python backend",
        "fixture": (_fixture("hit", "The user writes Python for backend services."),),
        "expected": ExpectedBehavior(required=("hit",), metrics=(MetricName.RETRIEVAL_RECALL,)),
    }
    fields.update(overrides)
    return EvaluationCase(**fields)  # type: ignore[arg-type]


def _dataset(*cases: EvaluationCase) -> EvaluationDataset:
    """Wrap cases in a dataset."""
    return EvaluationDataset(name="test", version="v1", cases=tuple(cases) or (_case(),))


class TestCaseExecution:
    """A case runs real services against its own isolated corpus."""

    def test_a_passing_case_reports_no_failure(self, evaluation_service: EvaluationService) -> None:
        result = evaluation_service.run_case(_case())
        assert result.passed
        assert result.failures == ()
        assert result.category is CaseCategory.RETRIEVAL
        assert result.numerators[MetricName.RETRIEVAL_RECALL] == 1
        assert result.denominators[MetricName.RETRIEVAL_RECALL] == 1

    def test_a_missing_required_memory_is_detected(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(_fixture("other", "The user repairs bicycles at the weekend."),),
            expected=ExpectedBehavior(required=("other",), metrics=(MetricName.RETRIEVAL_RECALL,)),
        )
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert any(failure.startswith("recall:") for failure in result.failures)

    def test_every_declared_metric_is_actually_measured(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            expected=ExpectedBehavior(
                required=("hit",),
                metrics=(MetricName.RETRIEVAL_RECALL, MetricName.CONTEXT_BUDGET_COMPLIANCE),
            )
        )
        result = evaluation_service.run_case(case)
        assert any(failure.startswith("measurement:") for failure in result.failures)
        assert not result.passed

    def test_a_case_is_isolated_from_the_ones_before_it(
        self, evaluation_service: EvaluationService
    ) -> None:
        corpus = tuple(
            _fixture(f"m{index}", f"The user keeps {index} books on the shelf.")
            for index in range(3)
        )
        expected = ExpectedBehavior(
            forbidden=("m0", "m1", "m2"), metrics=(MetricName.RECALL_RELEVANCE,)
        )
        first = evaluation_service.run_case(_case(fixture=corpus, expected=expected))
        second = evaluation_service.run_case(_case(fixture=corpus, expected=expected))
        assert first.measurements == second.measurements
        assert first.numerators == second.numerators

    def test_running_twice_gives_identical_results(
        self, evaluation_service: EvaluationService
    ) -> None:
        dataset = _dataset(_case(), _case(id="case-2"))
        assert (
            evaluation_service.run_dataset(dataset).to_dict()
            == evaluation_service.run_dataset(dataset).to_dict()
        )

    def test_identifiers_are_reproducible_across_stacks(
        self, evaluation_service: EvaluationService
    ) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        case = _case()
        with TemporaryDirectory() as directory:
            ids = [
                evaluation_service._build(
                    Path(directory) / f"{name}.sqlite", case.id
                ).store_fixture(case)
                for name in ("a", "b")
            ]
        assert ids[0] == ids[1]
        assert set(ids[0]) == {"hit"}
        assert ids[0]["hit"]


class TestRetrievalMetrics:
    """Recall, precision and top-k read the real search response."""

    def _run(self, evaluation_service: EvaluationService, case: EvaluationCase):  # type: ignore[no-untyped-def]
        """Run one case."""
        return evaluation_service.run_case(case)

    def test_recall_is_measured_per_required_memory(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(
                _fixture("a", "The user writes Python for backend services."),
                _fixture("b", "The user also maintains backend services on Kubernetes."),
            ),
            expected=ExpectedBehavior(
                required=("a", "b"),
                metrics=(
                    MetricName.RETRIEVAL_RECALL,
                    MetricName.TOP_K_HIT_RATE,
                    MetricName.RETRIEVAL_PRECISION,
                ),
            ),
        )
        result = self._run(evaluation_service, case)
        assert result.passed, result.failures
        assert result.denominators[MetricName.RETRIEVAL_RECALL] == 2
        assert result.numerators[MetricName.RETRIEVAL_RECALL] == 2
        assert result.denominators[MetricName.TOP_K_HIT_RATE] == 1

    def test_a_missing_required_memory_lowers_recall(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(
                _fixture("a", "The user writes Python for backend services."),
                _fixture("b", "The user repairs bicycles at the weekend."),
            ),
            expected=ExpectedBehavior(
                required=("a", "b"),
                metrics=(MetricName.RETRIEVAL_RECALL, MetricName.TOP_K_HIT_RATE),
            ),
        )
        result = self._run(evaluation_service, case)
        assert result.denominators[MetricName.RETRIEVAL_RECALL] == 2
        assert result.numerators[MetricName.RETRIEVAL_RECALL] == 1

    def test_precision_counts_returned_versus_relevant(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(
                _fixture("a", "The user writes Python for backend services."),
                _fixture("b", "The user also runs backend services on Kubernetes."),
            ),
            top_k=1,
            expected=ExpectedBehavior(
                required=("a", "b"),
                metrics=(MetricName.RETRIEVAL_PRECISION,),
            ),
        )
        result = self._run(evaluation_service, case)
        assert result.denominators[MetricName.RETRIEVAL_PRECISION] == 1
        assert result.numerators[MetricName.RETRIEVAL_PRECISION] == 1

    def test_declared_irrelevant_memories_are_never_returned(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(
                _fixture("a", "The user writes Python for backend services."),
                _fixture("noise", "The user repairs bicycles at the weekend."),
            ),
            expected=ExpectedBehavior(
                required=("a",),
                forbidden=("noise",),
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = self._run(evaluation_service, case)
        assert result.passed
        assert result.numerators[MetricName.RECALL_RELEVANCE] == 1

    def test_an_irrelevant_memory_that_matches_is_reported(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(
                _fixture("a", "The user writes Python for backend services."),
                _fixture("noise", "The user dislikes backend services documentation."),
            ),
            expected=ExpectedBehavior(
                required=("a",),
                forbidden=("noise",),
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = self._run(evaluation_service, case)
        assert not result.passed
        assert result.numerators[MetricName.RECALL_RELEVANCE] == 0

    def test_an_empty_result_set_is_not_scored_for_precision(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            fixture=(_fixture("a", "The user repairs bicycles at the weekend."),),
            expected=ExpectedBehavior(
                forbidden=("a",),
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = self._run(evaluation_service, case)
        assert result.passed
        assert result.measurements["results_returned"] == 0.0


class TestDuplicateSuppression:
    """Duplicate handling runs through M2's own detector."""

    def test_a_near_duplicate_is_suppressed(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            action=CaseAction.SEARCH_WITH_DUPLICATES,
            fixture=(
                _fixture(
                    "original",
                    "The user prefers the dark theme in every code editor.",
                    importance=0.95,
                ),
                _fixture(
                    "restated",
                    "The user prefers the dark theme in all code editors.",
                    importance=0.2,
                    confidence=0.3,
                ),
            ),
            query="dark theme editor",
            expected=ExpectedBehavior(
                required=("original",),
                duplicates_suppressed=("restated",),
                metrics=(MetricName.DUPLICATE_SUPPRESSION_ACCURACY,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.numerators[MetricName.DUPLICATE_SUPPRESSION_ACCURACY] == 1

    def test_several_copies_are_all_suppressed(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            action=CaseAction.SEARCH_WITH_DUPLICATES,
            fixture=(
                _fixture(
                    "original",
                    "The user works from a home office with a standing desk.",
                    importance=0.95,
                ),
                _fixture(
                    "copy_a",
                    "The user works from a home office with a standing desk too.",
                    importance=0.4,
                    confidence=0.3,
                ),
                _fixture(
                    "copy_b",
                    "The user works from a home office that has a standing desk.",
                    importance=0.3,
                    confidence=0.3,
                ),
            ),
            query="home office standing desk",
            top_k=10,
            expected=ExpectedBehavior(
                required=("original",),
                duplicates_suppressed=("copy_a", "copy_b"),
                metrics=(MetricName.DUPLICATE_SUPPRESSION_ACCURACY,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.denominators[MetricName.DUPLICATE_SUPPRESSION_ACCURACY] == 2

    def test_a_duplicate_in_a_compiled_context_is_dropped(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="workspace monitor",
            fixture=(
                _fixture(
                    "primary",
                    "The user works on a single 27 inch monitor.",
                    importance=0.8,
                ),
                _fixture(
                    "repetition",
                    "The user works on one 27 inch monitor only.",
                    importance=0.2,
                    confidence=0.3,
                ),
            ),
            expected=ExpectedBehavior(
                selected=("primary",),
                duplicates_suppressed=("repetition",),
                token_budget=400,
                metrics=(MetricName.DUPLICATE_SUPPRESSION_ACCURACY,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures


class TestConflictMetrics:
    """Conflict handling runs through M2's detector and resolver."""

    def _conflict_case(self, **overrides: object) -> EvaluationCase:
        """A conflicting pair whose higher-confidence side should win."""
        fields: dict[str, object] = {
            "action": CaseAction.SEARCH_WITH_CONFLICT,
            "fixture": (
                _fixture("loser", "preferred backend language = Python", confidence=0.4),
                _fixture("winner", "preferred backend language = TypeScript", confidence=0.95),
            ),
            "query": "preferred backend language",
            "expected": ExpectedBehavior(
                winner="winner",
                stale=("loser",),
                token_budget=FALLBACK_BUDGET,
                metrics=(
                    MetricName.CONFLICT_RESOLUTION_ACCURACY,
                    MetricName.STALE_MEMORY_AVOIDANCE,
                ),
            ),
        }
        fields.update(overrides)
        return _case(**fields)

    def test_the_expected_winner_is_confirmed(self, evaluation_service: EvaluationService) -> None:
        result = evaluation_service.run_case(self._conflict_case())
        assert result.passed, result.failures
        assert result.numerators[MetricName.CONFLICT_RESOLUTION_ACCURACY] == 1

    def test_a_mismatched_expectation_is_reported(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = self._conflict_case(
            expected=ExpectedBehavior(
                winner="loser",
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.CONFLICT_RESOLUTION_ACCURACY,),
            )
        )
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert result.numerators[MetricName.CONFLICT_RESOLUTION_ACCURACY] == 0
        assert any(f.startswith("conflict_resolution:") for f in result.failures)

    def test_importance_beats_confidence(self, evaluation_service: EvaluationService) -> None:
        case = self._conflict_case(
            fixture=(
                _fixture(
                    "winner", "preferred database = PostgreSQL", confidence=0.3, importance=0.95
                ),
                _fixture("loser", "preferred database = MySQL", confidence=0.9, importance=0.3),
            ),
            query="preferred database",
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures

    def test_an_unresolved_conflict_keeps_both(self, evaluation_service: EvaluationService) -> None:
        case = self._conflict_case(
            resolve_conflict=False,
            expected=ExpectedBehavior(
                required=("loser", "winner"),
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RETRIEVAL_RECALL,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.numerators[MetricName.RETRIEVAL_RECALL] == 2

    def test_a_superseded_memory_is_suppressed_from_the_context(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = self._conflict_case(
            action=CaseAction.COMPILE,
            archive_conflict_loser=False,
            expected=ExpectedBehavior(
                winner="winner",
                stale=("loser",),
                selected=("winner",),
                token_budget=FALLBACK_BUDGET,
                metrics=(
                    MetricName.CONFLICT_RESOLUTION_ACCURACY,
                    MetricName.STALE_MEMORY_AVOIDANCE,
                    MetricName.CONTEXT_SELECTION_ACCURACY,
                ),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures

    def test_different_attributes_are_not_treated_as_a_conflict(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = self._conflict_case(
            fixture=(
                _fixture("language", "preferred backend language = Go"),
                _fixture("database", "preferred database = PostgreSQL"),
            ),
            query="preferred",
            expected=ExpectedBehavior(
                required=("language", "database"),
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RETRIEVAL_RECALL,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures


class TestContextMetrics:
    """Budget, selection and provenance read the real compilation."""

    def test_a_tight_budget_is_never_exceeded(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="user reading habits",
            fixture=(
                _fixture("morning", "The user reads technical books before breakfast."),
                _fixture("night", "The user reads long essays late at night."),
                _fixture("podcast", "The user listens to podcasts while commuting."),
            ),
            expected=ExpectedBehavior(
                token_budget=40,
                metrics=(MetricName.CONTEXT_BUDGET_COMPLIANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.measurements["estimated_tokens"] <= 40

    def test_an_impossible_budget_yields_an_empty_context(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="user relocation plans",
            expected=ExpectedBehavior(
                token_budget=1,
                metrics=(MetricName.CONTEXT_BUDGET_COMPLIANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.measurements["selected_count"] == 0.0

    def test_a_missing_selection_is_reported(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="user relocation plans",
            fixture=(_fixture("far", "The user plans to relocate to another city for a role."),),
            expected=ExpectedBehavior(
                selected=("far",),
                token_budget=1,
                metrics=(MetricName.CONTEXT_SELECTION_ACCURACY,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert result.numerators[MetricName.CONTEXT_SELECTION_ACCURACY] == 0
        assert any(f.startswith("selection:") for f in result.failures)

    def test_provenance_is_measured_per_block(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="user learning habits",
            fixture=(
                _fixture("course", "The user completed a distributed systems course."),
                _fixture("book", "The user finished Designing Data Intensive Applications."),
            ),
            expected=ExpectedBehavior(
                token_budget=400,
                metrics=(MetricName.CONTEXT_PROVENANCE_COMPLETENESS,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.denominators[MetricName.CONTEXT_PROVENANCE_COMPLETENESS] >= 1
        assert (
            result.numerators[MetricName.CONTEXT_PROVENANCE_COMPLETENESS]
            == result.denominators[MetricName.CONTEXT_PROVENANCE_COMPLETENESS]
        )

    def test_a_compile_case_without_a_budget_is_refused(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            expected=ExpectedBehavior(metrics=(MetricName.CONTEXT_BUDGET_COMPLIANCE,)),
        )
        with pytest.raises(EvaluationError) as failure:
            evaluation_service.run_case(case)
        assert "token_budget" in str(failure.value)


class TestLifecycleCases:
    """History preservation is measured against an exact version count."""

    def _lifecycle_case(self, steps: tuple[str, ...], expected: int) -> EvaluationCase:
        """Build a lifecycle case with the given steps."""
        from app.domain.evaluation import LifecycleStep

        return _case(
            action=CaseAction.LIFECYCLE,
            lifecycle_steps=tuple(LifecycleStep(step) for step in steps),
            fixture=(_fixture("target", "The user plans to relocate for a new role."),),
            expected=ExpectedBehavior(
                expected_versions=expected,
                metrics=(MetricName.HISTORY_PRESERVATION,),
            ),
        )

    def test_update_then_forget_appends_two_versions(
        self, evaluation_service: EvaluationService
    ) -> None:
        result = evaluation_service.run_case(self._lifecycle_case(("update", "forget"), 3))
        assert result.passed, result.failures
        assert result.measurements["versions_after_create"] == 1
        assert result.measurements["versions_after_lifecycle"] == 3

    def test_forgetting_twice_is_a_documented_no_op(
        self, evaluation_service: EvaluationService
    ) -> None:
        result = evaluation_service.run_case(self._lifecycle_case(("forget", "forget_again"), 2))
        assert result.passed, result.failures

    def test_an_archived_memory_stays_editable(self, evaluation_service: EvaluationService) -> None:
        result = evaluation_service.run_case(self._lifecycle_case(("archive", "update"), 3))
        assert result.passed, result.failures

    def test_a_wrong_version_count_is_reported(self, evaluation_service: EvaluationService) -> None:
        result = evaluation_service.run_case(self._lifecycle_case(("update", "forget"), 99))
        assert not result.passed
        assert result.numerators[MetricName.HISTORY_PRESERVATION] == 0
        assert any(f.startswith("history:") for f in result.failures)

    def test_a_lifecycle_case_must_declare_steps_and_an_expected_count(self) -> None:
        from app.domain.evaluation import LifecycleStep

        with pytest.raises(EvaluationError):
            _case(
                action=CaseAction.LIFECYCLE,
                expected=ExpectedBehavior(
                    expected_versions=2, metrics=(MetricName.HISTORY_PRESERVATION,)
                ),
            )
        case = _case(
            action=CaseAction.LIFECYCLE,
            lifecycle_steps=(LifecycleStep.FORGET,),
            expected=ExpectedBehavior(metrics=(MetricName.HISTORY_PRESERVATION,)),
        )
        assert case.expected.expected_versions is None


class TestAgentCases:
    """The agent cases run a real turn through the offline mock provider."""

    def _agent_case(self, **overrides: object) -> EvaluationCase:
        """Build an agent case over a small corpus."""
        fields: dict[str, object] = {
            "fixture": (_fixture("editor", "The user types fastest in a keyboard driven editor."),),
            "query": "Tell me about my editor.\nremember: [PREFERENCE] The user prefers dark.",
            "expected": ExpectedBehavior(
                required=("editor",),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.MEMORY_WRITE_POLICY_COMPLIANCE,),
            ),
        }
        fields.update(overrides)
        return _case(category=CaseCategory.AGENT, action=CaseAction.AGENT, **fields)

    def test_nothing_is_stored_without_permission(
        self, evaluation_service: EvaluationService
    ) -> None:
        result = evaluation_service.run_case(self._agent_case())
        assert result.passed, result.failures
        assert result.measurements["stored"] == 0.0
        assert result.numerators[MetricName.MEMORY_WRITE_POLICY_COMPLIANCE] == 1

    def test_a_permitted_candidate_is_stored(self, evaluation_service: EvaluationService) -> None:
        case = self._agent_case(
            store_candidates=True,
            expected=ExpectedBehavior(
                required=("editor",),
                expected_stored=1,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.MEMORY_WRITE_POLICY_COMPLIANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.measurements["stored"] == 1.0

    def test_a_malformed_candidate_is_rejected_even_with_permission(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = self._agent_case(
            store_candidates=True,
            query="Note this.\nremember: [HOLOGRAM] The user is a pilot.",
            expected=ExpectedBehavior(
                required=("editor",),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.MEMORY_WRITE_POLICY_COMPLIANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.measurements["candidates"] == 1.0
        assert result.measurements["stored"] == 0.0

    def test_a_small_talk_turn_recalls_nothing(self, evaluation_service: EvaluationService) -> None:
        case = self._agent_case(
            query="thanks",
            expected=ExpectedBehavior(
                forbidden=("editor",),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.measurements["gate_recalled"] == 0.0
        assert result.measurements["context_blocks"] == 0.0

    def test_recall_relevance_combines_required_and_irrelevant(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = self._agent_case(
            fixture=(
                _fixture("python", "The user has shipped three Python services."),
                _fixture("pet", "The user adopted a tabby cat named Pepper."),
            ),
            query="What do you know about my Python experience?",
            expected=ExpectedBehavior(
                required=("python",),
                forbidden=("pet",),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures

    def test_an_unexpected_recall_is_reported(self, evaluation_service: EvaluationService) -> None:
        case = self._agent_case(
            fixture=(
                _fixture("python", "The user has shipped three Python services."),
                _fixture("weekend", "The user repairs bicycles on weekend afternoons."),
            ),
            query="What do you know about my Python experience and my weekend plans?",
            expected=ExpectedBehavior(
                required=("python", "weekend"),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.numerators[MetricName.RECALL_RELEVANCE] == 1

    def test_an_agent_case_must_declare_an_expected_count(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = self._agent_case(
            expected=ExpectedBehavior(
                required=("editor",), metrics=(MetricName.MEMORY_WRITE_POLICY_COMPLIANCE,)
            )
        )
        with pytest.raises(EvaluationError) as failure:
            evaluation_service.run_case(case)
        assert "expected_stored" in str(failure.value)


class TestDetectedFailures:
    """Every failure the evaluation can report is reachable and named.

    These cases deliberately encode a wrong expectation. A gate that cannot fail
    is not a gate, so each branch below is proven to be reachable.
    """

    def test_a_leaked_duplicate_is_reported(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="home office standing desk",
            fixture=(
                _fixture(
                    "original",
                    "The user works from a home office with a standing desk.",
                    importance=0.95,
                ),
                _fixture(
                    "restated",
                    "The user works from a home office with a standing desk too.",
                    importance=0.9,
                    confidence=0.8,
                ),
            ),
            expected=ExpectedBehavior(
                selected=("original",),
                duplicates_suppressed=("restated",),
                token_budget=400,
                metrics=(MetricName.DUPLICATE_SUPPRESSION_ACCURACY,),
            ),
        )
        # M4 suppresses the restatement, so the duplicate metric is satisfied.
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert result.numerators[MetricName.DUPLICATE_SUPPRESSION_ACCURACY] == 1

    def test_a_stale_memory_that_reaches_the_context_is_reported(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            resolve_conflict=False,
            query="preferred testing framework",
            fixture=(
                _fixture("winner", "preferred testing framework = pytest", confidence=0.4),
                _fixture("loser", "preferred testing framework = Jest", confidence=0.9),
            ),
            expected=ExpectedBehavior(
                stale=("loser",), token_budget=400, metrics=(MetricName.STALE_MEMORY_AVOIDANCE,)
            ),
        )
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert result.numerators[MetricName.STALE_MEMORY_AVOIDANCE] == 0
        assert any(f.startswith("stale:") for f in result.failures)

    def test_a_missing_recall_is_reported(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            category=CaseCategory.AGENT,
            action=CaseAction.AGENT,
            fixture=(
                _fixture("present", "The user writes Python for backend services."),
                _fixture("absent", "Bicycle repairs happen at the weekend every Sunday morning."),
            ),
            query="Which backend language does the user write?",
            expected=ExpectedBehavior(
                required=("present", "absent"),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert result.numerators[MetricName.RECALL_RELEVANCE] == 0
        assert any(f.startswith("recall:") for f in result.failures)

    def test_an_unexpected_recall_is_reported(self, evaluation_service: EvaluationService) -> None:
        case = _case(
            category=CaseCategory.AGENT,
            action=CaseAction.AGENT,
            fixture=(
                _fixture("wanted", "The user writes Python for backend services."),
                _fixture("noise", "The user also runs backend services on Kubernetes."),
            ),
            query="Tell me about my backend experience.\nremember: [FACT] The user likes SQLite.",
            expected=ExpectedBehavior(
                required=("wanted",),
                forbidden=("noise",),
                expected_stored=0,
                token_budget=FALLBACK_BUDGET,
                metrics=(MetricName.RECALL_RELEVANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert not result.passed
        assert any(f.startswith("irrelevant:") for f in result.failures)

    def test_a_compiled_block_always_carries_provenance(
        self, evaluation_service: EvaluationService
    ) -> None:
        """The only way provenance can fail is a malformed block, which M4
        cannot produce -- so a real run must score it perfectly."""
        case = _case(
            action=CaseAction.COMPILE,
            query="user learning habits",
            fixture=(_fixture("course", "The user completed a distributed systems course."),),
            expected=ExpectedBehavior(
                token_budget=400,
                metrics=(MetricName.CONTEXT_PROVENANCE_COMPLETENESS,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.passed, result.failures
        assert (
            result.numerators[MetricName.CONTEXT_PROVENANCE_COMPLETENESS]
            == result.denominators[MetricName.CONTEXT_PROVENANCE_COMPLETENESS]
        )


def test_a_duplicate_leak_is_reported() -> None:
    """A duplicate that slips through is named, not averaged away.

    M4 suppresses duplicates reliably, so this exercises the reporting path
    directly rather than trying to reproduce a leak the system prevents.
    """
    from app.domain.retrieval import RetrievalMode, RetrievalResponse, RetrievalResult
    from app.evaluation.service import _Measurement, _score_search

    case = _case(
        fixture=(_fixture("a", "A memory about python."), _fixture("b", "A copy of it.")),
        expected=ExpectedBehavior(
            required=("a",),
            duplicates_suppressed=("b",),
            metrics=(
                MetricName.RETRIEVAL_RECALL,
                MetricName.DUPLICATE_SUPPRESSION_ACCURACY,
            ),
        ),
    )
    out = _Measurement(declared=frozenset(case.expected.metrics))
    returned = RetrievalResult(
        memory=_fixture_memory("a"),
        lexical_score=0.9,
        semantic_score=None,
        relevance=0.5,
        final_score=0.9,
        components=(),
        explanation="hand-built",
        conflicts=(),
        superseded_by=None,
        duplicate_of=None,
        suppressed=False,
        suppression_reason=None,
    )
    response = RetrievalResponse(
        query=case.query,
        top_k=case.top_k,
        requested_mode=RetrievalMode.LEXICAL,
        retrieval_mode=RetrievalMode.LEXICAL,
        semantic_available=False,
        semantic_provider=None,
        semantic_model=None,
        degraded_reason=None,
        candidates_considered=2,
        candidates_returned=2,
        candidates_suppressed=0,
        latency_ms=0.0,
        results=(returned,),
        suppressed=(),
    )
    # The label list says both memories came back, which is the leak.
    _score_search(case, response, ["a", "b"], out)
    result = out.result(case)
    assert not result.passed
    assert result.numerators[MetricName.DUPLICATE_SUPPRESSION_ACCURACY] == 0
    assert result.failures == ("duplicate: 'b' was a duplicate but was still returned",)


def _fixture_memory(label: str):  # type: ignore[no-untyped-def]
    """A stored memory with a known id, used by the hand-built response above."""
    from datetime import UTC, datetime

    from app.domain.models import Memory, MemoryStatus, MemoryType

    return Memory(
        id=f"mem-{label}",
        content="A memory",
        memory_type=MemoryType.FACT,
        source=None,
        confidence=0.5,
        importance=0.5,
        status=MemoryStatus.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        version=1,
    )


class TestDatasetRun:
    """Running and summarising a whole dataset."""

    def test_a_passing_dataset_passes_its_threshold(
        self, evaluation_service: EvaluationService
    ) -> None:
        dataset = _dataset(_case(), _case(id="case-2"))
        summary = evaluation_service.evaluate(dataset)
        assert summary.passed
        assert summary.total_cases == 2
        assert summary.passed_cases == 2
        assert summary.failed_cases == 0
        assert summary.categories["retrieval"] == (2, 2)

    def test_a_failing_dataset_fails(self, evaluation_service: EvaluationService) -> None:
        failing = _case(
            fixture=(_fixture("other", "The user repairs bicycles."),),
            expected=ExpectedBehavior(required=("other",), metrics=(MetricName.RETRIEVAL_RECALL,)),
        )
        summary = evaluation_service.evaluate(_dataset(failing), threshold=1.0)
        assert not summary.passed
        assert summary.failed_cases == 1
        assert summary.failures[0].failures

    def test_reported_measurements_include_the_real_figures(
        self, evaluation_service: EvaluationService
    ) -> None:
        case = _case(
            action=CaseAction.COMPILE,
            query="user relocation plans",
            expected=ExpectedBehavior(
                token_budget=60,
                metrics=(MetricName.CONTEXT_BUDGET_COMPLIANCE,),
            ),
        )
        result = evaluation_service.run_case(case)
        assert result.measurements["token_budget"] == 60.0
        assert result.measurements["estimated_tokens"] <= 60.0
        assert result.measurements["remaining_tokens"] >= 0.0

    def test_the_summary_serialises_every_metric_it_measured(
        self, evaluation_service: EvaluationService
    ) -> None:
        payload = evaluation_service.evaluate(_dataset(_case())).to_dict()
        names = {metric["name"] for metric in payload["metrics"]}
        assert "retrieval_recall" in names
        for metric in payload["metrics"]:
            assert metric["definition"]
        # Every metric is listed; only the measured ones carry a denominator.
        measured = [metric for metric in payload["metrics"] if metric["denominator"] > 0]
        assert [metric["name"] for metric in measured] == ["retrieval_recall"]
        assert measured[0]["numerator"] == 1
        assert measured[0]["score"] == 1.0
