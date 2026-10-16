"""Loading and validating the version-controlled evaluation dataset (M6).

A dataset is a JSON file under :data:`DATASET_DIR`. It is plain text in the
repository so a change to a case is a reviewable diff, and it is parsed into the
strict typed models of :mod:`app.domain.evaluation` -- an unknown key, a missing
case field or a label that names no fixture is an error, never a silent default.

No database is involved. A dataset is an input, not a resource.
"""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Any

from app.domain.errors import EvaluationError
from app.domain.evaluation import (
    CaseAction,
    CaseCategory,
    EvaluationCase,
    EvaluationDataset,
    ExpectedBehavior,
    LifecycleStep,
    MemoryFixture,
    MetricName,
)

#: Directory holding the version-controlled datasets.
DATASET_DIR = Path(__file__).resolve().parent / "datasets"

#: The dataset MemForge ships and evaluates by default.
DEFAULT_DATASET = "memory-core-v1"


def dataset_path(name: str = DEFAULT_DATASET) -> Path:
    """Resolve a dataset argument to a file.

    Three forms are accepted, in order:

    1. a path ending in ``.json``;
    2. a filename stem inside :data:`DATASET_DIR` (``memory-core-v1``);
    3. a dataset *name* or ``name@version`` reference, matched against the
       headers of the files in :data:`DATASET_DIR`.

    The third form is what makes ``memory-core`` and ``memory-core@v1`` both
    resolve to ``memory-core-v1.json``: the version is in the file, not in its
    name.
    """
    candidate = Path(name)
    if candidate.suffix == ".json":
        return candidate
    direct = DATASET_DIR / f"{name}.json"
    if direct.exists():
        return direct
    return _by_header(name)


def _by_header(name: str) -> Path:
    """Find a dataset file by its declared ``name`` or ``name@version``."""
    wanted_name, _, wanted_version = name.partition("@")
    for path in sorted(DATASET_DIR.glob("*.json")):
        document = _load_header(path)
        if document is None:
            continue
        reference = f"{document.name}@{document.version}"
        if name == reference or (
            document.name == wanted_name
            and (not wanted_version or document.version == wanted_version)
        ):
            return path
    raise EvaluationError(
        f"no evaluation dataset matches {name!r}; available files: "
        f"{', '.join(sorted(path.stem for path in DATASET_DIR.glob('*.json'))) or 'none'}"
    )


def _load_header(path: Path) -> EvaluationDataset | None:
    """Parse a dataset file just far enough to read its name and version."""
    try:
        return parse_dataset(read_dataset(path), origin=str(path))
    except EvaluationError:
        return None


def load_dataset(name: str = DEFAULT_DATASET) -> EvaluationDataset:
    """Load and validate a dataset by name (default: the shipped one)."""
    return parse_dataset(read_dataset(dataset_path(name)), origin=str(dataset_path(name)))


def read_dataset(path: Path) -> str:
    """Read a dataset file, reporting its path on any failure."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationError(f"could not read the dataset at {path}: {error}") from error


def parse_dataset(raw: str, *, origin: str = "<string>") -> EvaluationDataset:
    """Parse and strictly validate a dataset document."""
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise EvaluationError(f"{origin} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise EvaluationError(f"{origin} must contain a JSON object at the top level")
    return build_dataset(_mapping(document, origin, "dataset"), origin=origin)


# -- strict field helpers ----------------------------------------------------


def _mapping(value: Any, origin: str, where: str) -> dict[str, Any]:
    """Require a JSON object."""
    if not isinstance(value, dict):
        raise EvaluationError(f"{origin}: {where} must be a JSON object")
    return value


def _required(document: dict[str, Any], key: str, origin: str, where: str) -> Any:
    """Require a present key."""
    if key not in document:
        raise EvaluationError(f"{origin}: {where} is missing the required key {key!r}")
    return document[key]


def _string(document: dict[str, Any], key: str, origin: str, where: str) -> str:
    """Require a string field."""
    value = _required(document, key, origin, where)
    if not isinstance(value, str):
        raise EvaluationError(f"{origin}: {where}.{key} must be a string")
    return value


def _text(document: dict[str, Any], key: str, origin: str, where: str) -> str:
    """Require a non-empty string field."""
    value = _string(document, key, origin, where)
    if not value.strip():
        raise EvaluationError(f"{origin}: {where}.{key} must not be empty")
    return value


def _float(document: dict[str, Any], key: str, origin: str, where: str, default: float) -> float:
    """Require a number in ``[0, 1]`` (optional)."""
    value = document.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{origin}: {where}.{key} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise EvaluationError(f"{origin}: {where}.{key} must be in [0, 1]")
    return float(value)


def _int(document: dict[str, Any], key: str, origin: str, where: str, default: int) -> int:
    """Require a non-negative integer (optional)."""
    value = document.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{origin}: {where}.{key} must be an integer")
    if value < 0:
        raise EvaluationError(f"{origin}: {where}.{key} must be >= 0")
    return value


def _optional_int(document: dict[str, Any], key: str, origin: str, where: str) -> int | None:
    """Require an integer when present, else ``None``."""
    if document.get(key) is None:
        return None
    return _int(document, key, origin, where, 0)


def _optional_text(document: dict[str, Any], key: str, origin: str, where: str) -> str | None:
    """Require a non-empty string when present, else ``None``."""
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{origin}: {where}.{key} must be a non-empty string")
    return value


def _bool(document: dict[str, Any], key: str, origin: str, where: str, default: bool) -> bool:
    """Require a boolean (optional)."""
    value = document.get(key, default)
    if not isinstance(value, bool):
        raise EvaluationError(f"{origin}: {where}.{key} must be true or false")
    return value


def _strings(
    document: dict[str, Any], key: str, origin: str, where: str, *, required: bool = False
) -> tuple[str, ...]:
    """Require a list of non-empty strings (optional unless ``required``)."""
    value = document.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"{origin}: {where}.{key} must be a list of strings")
    if required and not value:
        raise EvaluationError(f"{origin}: {where}.{key} must not be empty")
    for item in value:
        if not item.strip():
            raise EvaluationError(f"{origin}: {where}.{key} must not contain blank labels")
    return tuple(value)


def _one_of[E: enum.Enum](
    value: str, options: type[E], origin: str, where: str, name: str = "value"
) -> E:
    """Parse an enum value, reporting the allowed set on failure."""
    try:
        return options(value)
    except ValueError:
        allowed = ", ".join(sorted(str(item.value) for item in options))
        raise EvaluationError(
            f"{origin}: {where}.{name} is {value!r}; expected one of {allowed}"
        ) from None


# -- builders ----------------------------------------------------------------


def build_dataset(document: dict[str, Any], *, origin: str) -> EvaluationDataset:
    """Build a dataset from an already-parsed object."""
    cases = _required(document, "cases", origin, "dataset")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError(f"{origin}: dataset.cases must be a non-empty list")
    return EvaluationDataset(
        name=_text(document, "name", origin, "dataset"),
        version=_text(document, "version", origin, "dataset"),
        description=document.get("description", ""),
        cases=tuple(
            build_case(_mapping(case, origin, f"cases[{index}]"), origin=origin, index=index)
            for index, case in enumerate(cases)
        ),
    )


def build_case(document: dict[str, Any], *, origin: str, index: int) -> EvaluationCase:
    """Build one case from its JSON object."""
    where = f"cases[{index}]"
    fixture = _required(document, "fixture", origin, where)
    if not isinstance(fixture, list) or not fixture:
        raise EvaluationError(f"{origin}: {where}.fixture must be a non-empty list")
    expected = _mapping(_required(document, "expected", origin, where), origin, f"{where}.expected")
    metrics = _strings(expected, "metrics", origin, f"{where}.expected", required=True)
    steps = _strings(document, "lifecycle_steps", origin, where)
    return EvaluationCase(
        id=_text(document, "id", origin, where),
        title=_text(document, "title", origin, where),
        category=_one_of(
            _string(document, "category", origin, where), CaseCategory, origin, where, "category"
        ),
        action=_one_of(
            _string(document, "action", origin, where), CaseAction, origin, where, "action"
        ),
        query=_text(document, "query", origin, where),
        rationale=document.get("rationale", ""),
        fixture=tuple(
            build_memory(
                _mapping(item, origin, f"{where}.fixture[{position}]"),
                origin=origin,
                where=f"{where}.fixture[{position}]",
            )
            for position, item in enumerate(fixture)
        ),
        expected=build_expected(
            expected, origin=origin, where=f"{where}.expected", metrics=metrics
        ),
        top_k=_int(document, "top_k", origin, where, 10),
        resolve_conflict=_bool(document, "resolve_conflict", origin, where, True),
        archive_conflict_loser=_bool(document, "archive_conflict_loser", origin, where, True),
        store_candidates=_bool(document, "store_candidates", origin, where, False),
        lifecycle_steps=tuple(
            _one_of(step, LifecycleStep, origin, where, "lifecycle_steps") for step in steps
        ),
    )


def build_memory(document: dict[str, Any], *, origin: str, where: str) -> MemoryFixture:
    """Build one fixture memory."""
    return MemoryFixture(
        label=_text(document, "label", origin, where),
        content=_text(document, "content", origin, where),
        memory_type=_text(document, "memory_type", origin, where).upper(),
        confidence=_float(document, "confidence", origin, where, 0.5),
        importance=_float(document, "importance", origin, where, 0.5),
        source=_optional_text(document, "source", origin, where),
        age_minutes=_int(document, "age_minutes", origin, where, 1),
    )


def build_expected(
    document: dict[str, Any], *, origin: str, where: str, metrics: tuple[str, ...]
) -> ExpectedBehavior:
    """Build one case's expectations, resolving its metric names."""
    parsed: list[MetricName] = []
    for name in metrics:
        parsed.append(_one_of(name, MetricName, origin, where, "metrics"))
    return ExpectedBehavior(
        required=_strings(document, "required", origin, where),
        forbidden=_strings(document, "forbidden", origin, where),
        duplicates_suppressed=_strings(document, "duplicates_suppressed", origin, where),
        stale=_strings(document, "stale", origin, where),
        selected=_strings(document, "selected", origin, where),
        expected_stored=_optional_int(document, "expected_stored", origin, where),
        winner=_optional_text(document, "winner", origin, where),
        expected_versions=_optional_int(document, "expected_versions", origin, where),
        token_budget=_optional_int(document, "token_budget", origin, where),
        metrics=tuple(parsed),
    )


__all__ = [
    "DATASET_DIR",
    "DEFAULT_DATASET",
    "build_case",
    "build_dataset",
    "dataset_path",
    "load_dataset",
    "parse_dataset",
    "read_dataset",
]
