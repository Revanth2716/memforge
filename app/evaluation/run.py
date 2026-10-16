"""``python -m app.evaluation.run`` -- the MemForge evaluation gate (M6).

::

    python -m app.evaluation.run                 # the shipped dataset
    python -m app.evaluation.run --dataset mine  # another file in the folder
    python -m app.evaluation.run --threshold 0.95 --json
    python -m app.evaluation.run --verbose       # print every metric formula

Exit codes:

* ``0`` -- every case held its expectations and the overall score met the
  threshold;
* ``1`` -- the evaluation failed (a case failed, or the score was below the
  threshold);
* ``2`` -- the dataset could not be loaded or read.

Everything runs offline against the deterministic mock provider. No network, no
credential, no paid API.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.domain.errors import EvaluationError
from app.domain.evaluation import EvaluationSummary
from app.evaluation.datasets import DEFAULT_DATASET, load_dataset
from app.evaluation.report import render
from app.evaluation.service import DEFAULT_THRESHOLD, EvaluationService

#: Process exit codes.
EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.run",
        description="Run the MemForge memory evaluation against real services.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "dataset name inside app/evaluation/datasets, or a path to a .json "
            f"dataset (default: {DEFAULT_DATASET})"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(f"minimum overall score required to pass, in [0, 1] (default: {DEFAULT_THRESHOLD})"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="also print the full summary as deterministic JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the exact formula behind every applicable metric",
    )
    return parser


def run(
    *,
    dataset: str = DEFAULT_DATASET,
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = False,
) -> tuple[EvaluationSummary, int]:
    """Run one evaluation and return its summary plus the exit code."""
    if not 0.0 <= threshold <= 1.0:
        raise EvaluationError("--threshold must be between 0.0 and 1.0")
    loaded = load_dataset(dataset)
    summary = EvaluationService().evaluate(loaded, threshold=threshold)
    print(render(summary, verbose=verbose))
    return summary, (EXIT_PASSED if summary.passed else EXIT_FAILED)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        summary, code = run(dataset=args.dataset, threshold=args.threshold, verbose=args.verbose)
    except EvaluationError as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
