"""Configuration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings


def test_defaults_keep_everything_local() -> None:
    settings = Settings()
    assert settings.api_prefix == "/api/v1"
    assert settings.app_name == "MemForge"
    assert settings.database_path == PROJECT_ROOT / "data" / "memforge.sqlite"
    assert settings.database_path.is_absolute()


def test_database_path_can_be_overridden(tmp_path: Path) -> None:
    target = tmp_path / "custom.sqlite"
    assert Settings(database_path=target).database_path == target


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFORGE_API_PREFIX", "/api/v9")
    assert Settings().api_prefix == "/api/v9"


def test_resolved_database_path_relative_to_cwd() -> None:
    relative = Path("data") / "memforge.sqlite"
    settings = Settings(database_path=relative)
    assert settings.resolved_database_path() == Path.cwd() / relative


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFORGE_SOMETHING_ELSE", "ignored")
    assert Settings().app_name == "MemForge"

    monkeypatch.delenv("MEMFORGE_SOMETHING_ELSE", raising=False)
    assert os.environ.get("MEMFORGE_SOMETHING_ELSE") is None
