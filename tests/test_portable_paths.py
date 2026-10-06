"""Portable path resolution: source mode, frozen mode, and overrides."""

from __future__ import annotations

import sys
from pathlib import Path

from core import paths


def test_source_mode_keeps_repository_layout(monkeypatch) -> None:
    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    root = paths.bundle_root()
    assert root == paths.executable_dir()
    assert paths.data_dir() == root
    assert paths.artifacts_dir() == root / "ml" / "artifacts"
    assert paths.checkpoints_dir() == root / "ml" / "llm" / "checkpoints"


def test_data_dir_override_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "custom"))
    assert paths.data_dir() == (tmp_path / "custom").resolve()


def test_frozen_mode_writes_next_to_executable(monkeypatch, tmp_path: Path) -> None:
    exe_dir = tmp_path / "bundle"
    exe_dir.mkdir()
    meipass = tmp_path / "extract"
    meipass.mkdir()

    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "fayd-al-ruh"), raising=False)

    assert paths.is_frozen()
    assert paths.bundle_root() == meipass
    assert paths.executable_dir() == exe_dir
    assert paths.data_dir() == exe_dir / "data"


def test_default_checkpoint_prefers_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))

    bundled = paths.bundle_root() / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
    assert paths.default_checkpoint() == bundled

    trained = paths.checkpoints_dir() / "fayd-llm.npz"
    trained.parent.mkdir(parents=True)
    trained.write_bytes(b"trained")
    assert paths.default_checkpoint() == trained
