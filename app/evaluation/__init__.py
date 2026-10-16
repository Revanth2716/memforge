"""MemForge's own evaluation subsystem (M6).

**This is not EvalForge and not a generic LLM evaluation platform.** There is no
judge model, no scoring of generated prose and no attempt to measure "quality".
An evaluation here answers one narrow question:

    did MemForge make the memory decision its own documented rules say it should
    make, on this corpus?

Layers:

* :mod:`app.domain.evaluation` -- the typed dataset/case/result models.
* :mod:`app.evaluation.datasets` -- loading and strict validation of the
  version-controlled JSON dataset.
* :mod:`app.evaluation.service` -- runs each case against a real, isolated
  MemForge stack.
* :mod:`app.evaluation.metrics` -- the deterministic aggregation.
* :mod:`app.evaluation.report` -- the deterministic text report.
* :mod:`app.evaluation.run` -- the CLI regression gate.
"""

from __future__ import annotations

from app.evaluation.datasets import DATASET_DIR, DEFAULT_DATASET, load_dataset
from app.evaluation.metrics import TOP_K, compute_metrics
from app.evaluation.report import render
from app.evaluation.service import DEFAULT_THRESHOLD, EvaluationService

__all__ = [
    "DATASET_DIR",
    "DEFAULT_DATASET",
    "DEFAULT_THRESHOLD",
    "TOP_K",
    "EvaluationService",
    "compute_metrics",
    "load_dataset",
    "render",
]
