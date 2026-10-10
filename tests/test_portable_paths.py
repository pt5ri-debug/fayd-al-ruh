"""Portable path resolution: source mode, frozen mode, and overrides."""

from __future__ import annotations

import sys
from pathlib import Path

from core import paths


def test_source_mode_keeps_learned_data_under_git_ignored_dir(monkeypatch) -> None:
    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    root = paths.bundle_root()
    assert root == paths.executable_dir()
    assert paths.data_dir() == root / "data"
    assert paths.artifacts_dir() == root / "data" / "ml" / "artifacts"
    assert paths.checkpoints_dir() == root / "data" / "ml" / "llm" / "checkpoints"
    # The bundled, read-only copies remain the fallback.
    assert paths.bundled_artifacts_dir() == root / "ml" / "artifacts"
    assert paths.bundled_checkpoints_dir() == root / "ml" / "llm" / "checkpoints"


def test_source_mode_data_dir_is_git_ignored() -> None:
    """The source data folder must be ignored so updates never see it as dirty."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "data/anything"],
        cwd=str(paths.bundle_root()),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "data/ must be git-ignored"


def test_ensure_data_layout_copies_preexisting_data(monkeypatch, tmp_path: Path) -> None:
    """An upgrade migrates in-repo artifacts/checkpoints/kb into data/ once."""
    root = tmp_path / "checkout"
    (root / "ml" / "artifacts").mkdir(parents=True)
    (root / "ml" / "artifacts" / "release_manifest.json").write_text("{}", encoding="utf-8")
    (root / "ml" / "llm" / "checkpoints").mkdir(parents=True)
    (root / "ml" / "llm" / "checkpoints" / "fayd-llm.npz").write_bytes(b"ckpt")
    (root / "knowledge_base").mkdir()
    (root / "knowledge_base" / "shard.json").write_text("{}", encoding="utf-8")
    (root / "learned_cursor.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(paths, "_source_root", lambda: root)

    paths.ensure_data_layout()

    assert (root / "data" / "ml" / "artifacts" / "release_manifest.json").exists()
    assert (root / "data" / "ml" / "llm" / "checkpoints" / "fayd-llm.npz").read_bytes() == b"ckpt"
    assert (root / "data" / "knowledge_base" / "shard.json").exists()
    assert (root / "data" / "learned_cursor.json").exists()


def test_ensure_data_layout_is_a_no_op_once_data_exists(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "data").mkdir(parents=True)
    (root / "ml" / "artifacts").mkdir(parents=True)

    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(paths, "_source_root", lambda: root)

    paths.ensure_data_layout()

    assert not (root / "data" / "ml").exists()


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
