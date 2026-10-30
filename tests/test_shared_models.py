"""The shared language-model folder: one public catalog every character may run.

Besides each character's own model folder (``data/natasha``, ``data/hakeem``,
...), the project keeps one **public** folder of general GGUF models
(``data/shared_models``) that every GGUF mouth may run through the one in-process
runtime. These tests pin what the module owns:

* the catalog (three general Qwen models, ``qwen3.5-2b`` the default) and its
  summaries;
* the honest status report (an empty folder is reported, not hidden);
* the governed download (denied by default, run only with an approval) through an
  injected ``huggingface_hub`` seam, so no real fetch happens;
* the one-time migration of ``qwen3-1.7b`` out of Natasha's folder (a real move,
  never a delete it could not perform, and a no-op when there is nothing to do);
* the wiring: a shared model is offered to every GGUF mouth in the project
  catalog, and every character's model layer falls back to this folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core import shared_models
from core.shared_models import (
    CATALOG,
    DEFAULT_SPEC_KEY,
    SharedModelUnavailable,
    download_all,
    download_model,
    find_model_file,
    installed_spec_keys,
    list_models,
    migrate_from_natasha,
    model_is_cached,
    resolve_spec,
    shared_model_status,
    spec_names,
    spec_summaries,
)


# -- fixtures -------------------------------------------------------------
@pytest.fixture
def data_root(tmp_path: Path, monkeypatch) -> Path:
    """Point the whole program's data folder at a fresh temp directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# -- the catalog ----------------------------------------------------------
def test_catalog_holds_the_requested_models() -> None:
    """The three models asked for are present, and the default is one of them."""
    assert set(CATALOG) == {"qwen3.5-0.8b", "qwen3.5-2b", "qwen3-1.7b"}
    assert DEFAULT_SPEC_KEY in CATALOG
    assert DEFAULT_SPEC_KEY == "qwen3.5-2b"
    assert resolve_spec(None).key == DEFAULT_SPEC_KEY
    assert resolve_spec("qwen3.5-0.8b").filename == "Qwen3.5-0.8B-Q4_K_M.gguf"
    assert resolve_spec("qwen3-1.7b").filename == "Qwen3-1.7B-Q4_K_M.gguf"


def test_spec_names_and_summaries_are_sorted_and_complete() -> None:
    assert spec_names() == tuple(sorted(CATALOG))
    rows = spec_summaries()
    assert len(rows) == len(CATALOG)
    for key, title, size, summary in rows:
        assert key in CATALOG
        assert title and summary
        assert size > 0


def test_resolve_spec_refuses_an_unknown_name() -> None:
    with pytest.raises(SharedModelUnavailable):
        resolve_spec("not-a-model")


# -- the folder -----------------------------------------------------------
def test_an_empty_folder_is_reported_honestly(data_root: Path) -> None:
    status = shared_model_status()
    assert status.available is False
    assert status.models == ()
    assert "no model is downloaded yet" in status.summary()
    assert "shared-models download" in status.hint


def test_a_present_file_is_found_and_reported(data_root: Path) -> None:
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    (folder / "Some-Model.gguf").write_bytes(b"GGUF")
    assert model_is_cached() is True
    assert find_model_file() == folder / "Some-Model.gguf"
    status = shared_model_status()
    assert status.available is True
    assert status.models == ("Some-Model.gguf",)


def test_the_registry_names_the_file_to_load(data_root: Path) -> None:
    """The registry wins over the largest-file guess, so the choice is honoured."""
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    (folder / "small.gguf").write_bytes(b"x")
    (folder / "large.gguf").write_bytes(b"y" * 100)
    (folder / shared_models.REGISTRY_FILENAME).write_text(
        json.dumps({"file": "small.gguf", "spec": "qwen3.5-0.8b"}), encoding="utf-8"
    )
    assert find_model_file() == folder / "small.gguf"


def test_list_models_is_largest_first(data_root: Path) -> None:
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    (folder / "a.gguf").write_bytes(b"x")
    (folder / "b.gguf").write_bytes(b"y" * 100)
    names = [model.title for model in list_models()]
    assert names == ["b.gguf", "a.gguf"]


def test_installed_spec_keys_reports_the_catalog_hits(data_root: Path) -> None:
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    (folder / CATALOG["qwen3.5-2b"].filename).write_bytes(b"GGUF")
    assert installed_spec_keys() == ("qwen3.5-2b",)


# -- the governed download ------------------------------------------------
def test_download_is_denied_by_default(data_root: Path) -> None:
    with pytest.raises(SharedModelUnavailable):
        download_model("qwen3.5-2b")


def test_download_runs_through_the_gate_when_approved(data_root: Path, monkeypatch) -> None:
    """An approved fetch calls the hub seam and writes the registry."""
    seen: dict[str, object] = {}

    class FakeHub:
        @staticmethod
        def hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
            seen["repo"] = repo_id
            seen["filename"] = filename
            target = Path(local_dir) / filename
            target.write_bytes(b"GGUF")
            return str(target)

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", FakeHub)
    path = download_model("qwen3.5-2b", approved=True)
    assert path.name == CATALOG["qwen3.5-2b"].filename
    assert seen["repo"] == CATALOG["qwen3.5-2b"].repo
    registry = json.loads(
        (path.parent / shared_models.REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    assert registry["file"] == path.name
    assert registry["spec"] == "qwen3.5-2b"


def test_download_skips_a_file_already_on_disk(data_root: Path, monkeypatch) -> None:
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    target = folder / CATALOG["qwen3.5-0.8b"].filename
    target.write_bytes(b"GGUF")

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a cached model must not be re-fetched")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", explode)
    assert download_model("qwen3.5-0.8b", approved=True) == target


def test_download_all_fetches_every_model(data_root: Path, monkeypatch) -> None:
    class FakeHub:
        @staticmethod
        def hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
            target = Path(local_dir) / filename
            target.write_bytes(b"GGUF")
            return str(target)

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", FakeHub)
    paths = download_all(approved=True)
    assert len(paths) == len(CATALOG)


# -- the one-time migration ----------------------------------------------
def test_migration_moves_the_legacy_model_out_of_natashas_folder(tmp_path: Path) -> None:
    source = tmp_path / "natasha"
    source.mkdir()
    legacy = source / CATALOG["qwen3-1.7b"].filename
    legacy.write_bytes(b"GGUF")
    target_root = tmp_path / "shared_models"

    report = migrate_from_natasha(source=source, root=target_root)
    assert report.moved == (CATALOG["qwen3-1.7b"].filename,)
    assert report.changed is True
    assert not legacy.exists()
    assert (target_root / CATALOG["qwen3-1.7b"].filename).is_file()
    assert "Moved 1 model" in report.summary()


def test_migration_can_copy_instead_of_move(tmp_path: Path) -> None:
    source = tmp_path / "natasha"
    source.mkdir()
    legacy = source / CATALOG["qwen3-1.7b"].filename
    legacy.write_bytes(b"GGUF")
    target_root = tmp_path / "shared_models"

    migrate_from_natasha(source=source, root=target_root, copy=True)
    assert legacy.is_file()
    assert (target_root / legacy.name).is_file()


def test_migration_is_a_no_op_with_nothing_to_move(tmp_path: Path) -> None:
    source = tmp_path / "natasha"
    source.mkdir()
    report = migrate_from_natasha(source=source, root=tmp_path / "shared_models")
    assert report.moved == ()
    assert report.changed is False
    assert "Nothing to move" in report.summary() or report.detail


def test_migration_never_overwrites_an_existing_shared_copy(tmp_path: Path) -> None:
    source = tmp_path / "natasha"
    source.mkdir()
    legacy = source / CATALOG["qwen3-1.7b"].filename
    legacy.write_bytes(b"OLD")
    target_root = tmp_path / "shared_models"
    target_root.mkdir()
    kept = target_root / CATALOG["qwen3-1.7b"].filename
    kept.write_bytes(b"NEW")

    report = migrate_from_natasha(source=source, root=target_root)
    assert report.skipped == (CATALOG["qwen3-1.7b"].filename,)
    assert kept.read_bytes() == b"NEW"
    assert legacy.is_file()


def test_migration_finds_a_loosely_named_legacy_file(tmp_path: Path) -> None:
    """A user's own spelling (lower case, underscores) is still moved."""
    source = tmp_path / "natasha"
    source.mkdir()
    legacy = source / "qwen3-1.7b-q4_k_m.gguf"
    legacy.write_bytes(b"GGUF")
    target_root = tmp_path / "shared_models"

    report = migrate_from_natasha(source=source, root=target_root)
    assert report.moved == ("qwen3-1.7b-q4_k_m.gguf",)
    assert not legacy.exists()
    assert (target_root / CATALOG["qwen3-1.7b"].filename).is_file()


def test_migration_leaves_an_unrelated_model_alone(tmp_path: Path) -> None:
    """A model the catalog does not name is not swept into the shared folder."""
    source = tmp_path / "natasha"
    source.mkdir()
    other = source / "llama-3.2-3b-instruct-q4_k_m.gguf"
    other.write_bytes(b"GGUF")
    target_root = tmp_path / "shared_models"

    report = migrate_from_natasha(source=source, root=target_root)
    assert report.moved == ()
    assert other.is_file()
    assert not (target_root / "llama-3.2-3b-instruct-q4_k_m.gguf").exists()


# -- the wiring -----------------------------------------------------------
def test_a_shared_model_is_offered_to_every_gguf_mouth() -> None:
    """A model in the shared folder answers in every character's mouth."""
    from core import language_models as catalog

    model = catalog.find_language_model(catalog.MOUTH_HAKEEM, "qwen3.5-2b")
    assert model is not None
    assert catalog.MOUTH_SHARED in model.engines
    for mouth in catalog.GGUF_MOUTHS:
        assert mouth in model.engines


def test_hakeem_has_his_own_default_and_natasha_keeps_hers() -> None:
    """The two characters have different catalogs, so their defaults differ."""
    from companion.model import DEFAULT_SPEC_KEY as natasha_default
    from core import language_models as catalog
    from hakeem.model import DEFAULT_SPEC_KEY as hakeem_default

    assert hakeem_default != natasha_default
    assert catalog.default_language_model_key(catalog.MOUTH_HAKEEM) == hakeem_default
    assert catalog.default_language_model_key(catalog.MOUTH_NATASHA) == natasha_default


def test_the_shared_engine_can_download_its_own_models() -> None:
    from core import language_models as catalog

    model = catalog.find_language_model_anywhere("qwen3.5-2b")
    assert model is not None
    assert catalog.can_download(model[1]) is True
