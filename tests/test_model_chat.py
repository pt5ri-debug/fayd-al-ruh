"""The per-mouth chat windows: identity, voice, model, and the composer's row.

The feature this pins: every one of the seven model chips in the chat composer
opens a window of that model's *own*, holding, down the left side, its logo (with
the ability to change it), the reading voice exclusive to it, and the language
model it answers with -- plus a microphone in the composer.

The headless parts (the catalog, the profile store, the mouth factory, the chat
service, the CLI) run with no display. The window parts are skipped unless a
display (or Xvfb) is available, exactly like ``tests/test_gui.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from core import language_models as catalog
from core import mouth_profile as profiles

MOUTHS = (
    catalog.MOUTH_SOURCES,
    catalog.MOUTH_ULTRON,
    catalog.MOUTH_VISION,
    catalog.MOUTH_BANNER,
    catalog.MOUTH_NATASHA,
    catalog.MOUTH_HAKEEM,
    catalog.MOUTH_CLOUD,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point the whole program's data folder at a fresh temp directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _png(path: Path, size: int = 4) -> Path:
    """Write a tiny, real PNG so a logo copy has something to copy."""
    import struct
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(data)
    return path


# --- the unified language-model catalog -----------------------------------


def test_catalog_names_every_mouth() -> None:
    """Every answering mouth is in the catalog, in the composer's own order."""
    assert catalog.MOUTHS == MOUTHS
    for mouth in MOUTHS:
        assert mouth in catalog.ENGINE_LABELS


def test_gguf_mouths_share_one_catalog_so_any_downloaded_model_is_offered() -> None:
    """A dedicated GGUF window offers *every* GGUF model the project has."""
    models = catalog.models_for_choice(catalog.MOUTH_ULTRON)
    keys = {model.key for model in models}
    # Ultron's own model, a Vision-only model, a Banner-only model, a companion
    # model: all present, because they share the one in-process runtime.
    assert "qwen2.5-7b" in keys
    assert "ogai-reasoner" in keys
    assert "qwen3-8b" in keys
    assert "qwen3-4b" in keys
    # The trained checkpoint is offered too, through the same runtime.
    assert "fayd-llm" in keys
    # No duplicate: the same key is never listed twice.
    assert len(keys) == len(models)


def test_every_model_carries_an_honest_availability_and_next_step() -> None:
    """A model reports whether it is really here, and the real next step when not."""
    for model in catalog.models_for_choice(catalog.MOUTH_VISION):
        assert isinstance(model.available, bool)
        if not model.available:
            assert model.detail, model.key
        else:
            assert model.path is not None and model.path.is_file(), model.key


def test_availability_is_true_only_when_the_gguf_file_is_present(data_root) -> None:
    """A model becomes available exactly when its own file is on disk."""
    folder = data_root / "ultron"
    folder.mkdir(parents=True)
    from core.ultron import resolve_spec

    spec = resolve_spec("qwen2.5-7b")
    before = catalog.find_language_model(catalog.MOUTH_ULTRON, "qwen2.5-7b")
    assert before is not None and not before.available
    (folder / spec.filename).write_bytes(b"GGUF")
    after = catalog.find_language_model(catalog.MOUTH_ULTRON, "qwen2.5-7b")
    assert after is not None and after.available
    assert after.path == folder / spec.filename


def test_cloud_catalog_carries_provider_and_model() -> None:
    """A cloud entry names its provider, and its key keeps the two apart."""
    models = catalog.language_models(catalog.MOUTH_CLOUD)
    assert models
    for model in models:
        assert model.provider
        assert model.key.startswith(f"{model.provider}:")


def test_unknown_mouth_and_key_yield_nothing_rather_than_raising() -> None:
    """An unknown mouth or model is answered honestly, never with an exception."""
    assert catalog.language_models("not-a-mouth") == ()
    assert catalog.find_language_model(catalog.MOUTH_ULTRON, "not-a-model") is None
    assert catalog.resolve_choice(catalog.MOUTH_ULTRON, "not-a-model") is None
    assert catalog.resolve_choice(catalog.MOUTH_ULTRON, "") is None


def test_default_key_is_the_mouths_own_first_choice() -> None:
    """A mouth's default is the first model its catalog reports."""
    assert catalog.default_language_model_key(catalog.MOUTH_ULTRON) == "qwen2.5-7b"
    assert catalog.default_language_model_key("not-a-mouth") is None


# --- the per-mouth profile store -------------------------------------------


def test_profile_starts_at_the_defaults(data_root) -> None:
    profile = profiles.load_profile(catalog.MOUTH_ULTRON)
    assert profile.mouth == "ultron"
    assert profile.name == "Ultron"
    assert profile.logo == "" and profile.voice == "" and profile.model == ""


def test_profile_round_trips_each_field(data_root) -> None:
    profiles.set_name("natasha", "ناتاشا")
    profiles.set_voice("natasha", "woman-3")
    profiles.set_model("natasha", "qwen3-8b")
    loaded = profiles.load_profile("natasha")
    assert loaded.name == "ناتاشا"
    assert loaded.voice == "woman-3"
    assert loaded.model == "qwen3-8b"
    # And it survives a fresh read, which is what "remembers until I change it"
    # means across a restart.
    again = profiles.load_profile("natasha")
    assert again == loaded


def test_profiles_are_isolated_per_mouth(data_root) -> None:
    """A voice chosen for one mouth never changes another's."""
    profiles.set_voice("ultron", "man-2")
    profiles.set_voice("vision", "woman-4")
    assert profiles.load_profile("ultron").voice == "man-2"
    assert profiles.load_profile("vision").voice == "woman-4"


def test_a_saved_model_that_left_the_catalog_falls_back_to_the_default(data_root) -> None:
    """A stale saved model is refused rather than pointed at blindly."""
    profiles.set_model("ultron", "a-model-that-never-existed")
    assert profiles.resolve_model("ultron") == ""
    profiles.set_model("ultron", "qwen2.5-3b")
    assert profiles.resolve_model("ultron") == "qwen2.5-3b"


def test_saving_a_logo_copies_it_into_the_data_folder(data_root, tmp_path) -> None:
    """A chosen logo is copied in, named for its mouth, and returned."""
    source = _png(tmp_path / "logo.png")
    profile = profiles.save_logo("hakeem", source)
    assert profile.logo == "hakeem.png"
    stored = profile.logo_path()
    assert stored is not None and stored.is_file()
    assert stored.parent == data_root / "characters"
    # The original is untouched and the stored file is a real copy.
    assert source.is_file()
    assert stored.read_bytes() == source.read_bytes()


def test_a_replaced_logo_removes_the_old_one(data_root, tmp_path) -> None:
    """Only one logo file per mouth is kept, whatever extension it used."""
    profiles.save_logo("vision", _png(tmp_path / "one.png"))
    profiles.save_logo("vision", _png(tmp_path / "two.png"))
    folder = data_root / "characters"
    kept = sorted(p.name for p in folder.glob("vision.*"))
    assert kept == ["vision.png"]
    assert profiles.load_profile("vision").logo == "vision.png"


def test_a_bad_logo_choice_is_reported_not_stored(data_root, tmp_path) -> None:
    """An unsupported extension is refused with a clear message."""
    bad = tmp_path / "logo.bmp"
    bad.write_bytes(b"BM")
    with pytest.raises(ValueError, match="PNG"):
        profiles.save_logo("ultron", bad)
    assert profiles.load_profile("ultron").logo == ""


def test_a_deleted_logo_file_falls_back_to_the_default(data_root, tmp_path) -> None:
    """A saved logo whose file has gone is treated as no logo, not a crash."""
    profiles.save_logo("banner", _png(tmp_path / "logo.png"))
    stored = profiles.load_profile("banner").logo_path()
    assert stored is not None
    stored.unlink()
    assert profiles.load_profile("banner").logo_path() is None


def test_clearing_a_logo_removes_it(data_root, tmp_path) -> None:
    profiles.save_logo("cloud", _png(tmp_path / "logo.png"))
    profiles.clear_logo("cloud")
    assert profiles.load_profile("cloud").logo == ""
    assert not list((data_root / "characters").glob("cloud.*"))


def test_rename_refuses_an_empty_name(data_root) -> None:
    """A blank rename keeps the default rather than storing nothing."""
    profiles.set_name("hakeem", "   ")
    assert profiles.load_profile("hakeem").name == "Hakeem"


def test_a_corrupt_profile_falls_back_to_the_default(data_root) -> None:
    """A profile file of the wrong shape yields the default, never a crash."""
    from storage.json_store import JsonStore

    JsonStore(data_root).save(profiles.NAMESPACE, "ultron", ["not", "a", "mapping"])
    assert profiles.load_profile("ultron").name == "Ultron"


def test_all_profiles_lists_every_mouth(data_root) -> None:
    listed = profiles.all_profiles()
    assert tuple(profile.mouth for profile in listed) == MOUTHS


# --- the mouth factory ------------------------------------------------------


def test_the_factory_hands_the_loader_the_resolved_file(monkeypatch, data_root) -> None:
    """A chosen model reaches the mouth as its exact GGUF file."""
    from ai import mouths

    folder = data_root / "ultron"
    folder.mkdir(parents=True)
    from core.ultron import resolve_spec

    target = folder / resolve_spec("qwen2.5-3b").filename
    target.write_bytes(b"GGUF")

    seen: dict[str, Any] = {}

    def fake_loader(model_file=None, **kwargs):
        seen["model_file"] = model_file
        return "assistant"

    monkeypatch.setattr(mouths.ultron_mouth, "load_ultron_assistant", fake_loader)
    built = mouths.build_mouth(catalog.MOUTH_ULTRON, model_key="qwen2.5-3b")
    assert built == "assistant"
    assert seen["model_file"] == target


def test_the_factory_falls_back_to_the_default_for_an_unknown_key(monkeypatch, data_root) -> None:
    """An unknown key means "the mouth's own default", never a wrong file."""
    from ai import mouths

    seen: dict[str, Any] = {}

    def fake_loader(model_file=None, **kwargs):
        seen["model_file"] = model_file
        return "assistant"

    monkeypatch.setattr(mouths.ultron_mouth, "load_ultron_assistant", fake_loader)
    mouths.build_mouth(catalog.MOUTH_ULTRON, model_key="not-a-model")
    assert seen["model_file"] is None


def test_hakeem_mouth_is_handed_his_own_model(monkeypatch, data_root) -> None:
    """Hakeem has his own catalog, so his choice resolves through it."""
    from ai import mouths

    folder = data_root / "hakeem"
    folder.mkdir(parents=True)
    from hakeem.model import resolve_spec

    spec = resolve_spec("qwen3.5-4b")
    (folder / spec.filename).write_bytes(b"GGUF")

    seen: dict[str, Any] = {}

    def fake_loader(model=None, model_file=None, **kwargs):
        seen["model"] = model
        seen["model_file"] = model_file
        return "agent"

    monkeypatch.setattr(mouths.hakeem_mouth, "load_hakeem_assistant", fake_loader)
    mouths.build_mouth(catalog.MOUTH_HAKEEM, model_key="qwen3.5-4b")
    assert seen["model"] == spec.tag
    assert seen["model_file"] == folder / spec.filename


def test_a_character_mouth_may_run_a_model_from_another_catalog(monkeypatch, data_root) -> None:
    """The GGUF mouths share one runtime, so a key from another catalog resolves.

    Hakeem's window offers *every* GGUF model the project has; a key the
    companion's catalog names (``qwen3-4b``) is resolved through that catalog, and
    the file it found in Natasha's folder is handed over explicitly.
    """
    from ai import mouths

    folder = data_root / "natasha"
    folder.mkdir(parents=True)
    from companion.model import resolve_spec

    spec = resolve_spec("qwen3-4b")
    (folder / spec.filename).write_bytes(b"GGUF")

    seen: dict[str, Any] = {}

    def fake_loader(model=None, model_file=None, **kwargs):
        seen["model"] = model
        seen["model_file"] = model_file
        return "agent"

    monkeypatch.setattr(mouths.hakeem_mouth, "load_hakeem_assistant", fake_loader)
    mouths.build_mouth(catalog.MOUTH_HAKEEM, model_key="qwen3-4b")
    assert seen["model"] == spec.tag
    assert seen["model_file"] == folder / spec.filename


def test_the_factory_refuses_a_mouth_with_no_model() -> None:
    """A retrieval-only mouth has no model-backed assistant to build."""
    from ai import mouths

    with pytest.raises(ValueError):
        mouths.build_mouth(catalog.MOUTH_SOURCES)


# --- the per-mouth chat service --------------------------------------------


def test_chat_service_reports_the_mouth_identity(data_root) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    assert service.mouth == "ultron"
    assert service.name == "Ultron"
    assert service.blurb()


def test_chat_service_refuses_an_unknown_mouth(data_root) -> None:
    from gui.model_chat import ModelChatService

    with pytest.raises(ValueError):
        ModelChatService("not-a-mouth", data_root=data_root)


def test_chat_service_remembers_the_model_per_mouth(data_root) -> None:
    """Choosing a model here persists it, and only for this mouth."""
    from gui.model_chat import ModelChatService

    ultron = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    vision = ModelChatService(catalog.MOUTH_VISION, data_root=data_root)
    ultron.set_model("qwen2.5-3b")
    assert ultron.model_key() == "qwen2.5-3b"
    # A second service for the same mouth sees the saved choice...
    again = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    assert again.model_key() == "qwen2.5-3b"
    # ...and the other mouth is untouched.
    assert vision.model_key() == ""
    assert profiles.load_profile("vision").model == ""


def test_chat_service_remembers_the_voice_per_mouth(data_root) -> None:
    """The reading voice is exclusive to the mouth, and it persists."""
    from gui.model_chat import ModelChatService

    ultron = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    vision = ModelChatService(catalog.MOUTH_VISION, data_root=data_root)
    ultron.set_voice("man-2")
    assert ultron.voice_key() == "man-2"
    assert vision.voice_key() != "man-2"
    assert ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root).voice_key() == "man-2"


def test_chat_service_rejects_an_unknown_voice(data_root) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    with pytest.raises(ValueError):
        service.set_voice("not-a-voice")


def test_chat_service_applies_a_saved_voice_on_open(data_root) -> None:
    """A saved voice is in force the moment the window is built."""
    from gui.model_chat import ModelChatService

    profiles.set_voice("banner", "girl-2")
    service = ModelChatService(catalog.MOUTH_BANNER, data_root=data_root)
    assert service.voice_key() == "girl-2"


def test_chat_service_renames_and_reports_it(data_root) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_HAKEEM, data_root=data_root)
    assert service.set_name("الحكيم") == "الحكيم"
    assert ModelChatService(catalog.MOUTH_HAKEEM, data_root=data_root).name == "الحكيم"


def test_chat_service_batches_a_reading_into_few_engine_calls(data_root) -> None:
    """A long message is read in batches, so a local engine loads its model once.

    This is the hang fix for a per-mouth window: audio.cpp reloads its 1.35 GB model
    per process, so a paragraph-per-call read looked frozen. The service renders a
    whole batch through the reader's batched path.
    """

    class _Reader:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []
            self.voice = None
            self.mood = None

        def available(self) -> bool:
            return True

        def render(self, text: str):
            from types import SimpleNamespace

            return SimpleNamespace(silent=False, duration_seconds=0.0)

        def render_many(self, texts):
            from types import SimpleNamespace

            self.batches.append(list(texts))
            return [SimpleNamespace(silent=False, duration_seconds=0.0) for _ in texts]

        def play(self, clip):
            return None

    from gui.model_chat import ModelChatService

    reader = _Reader()
    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root, reader=reader)  # type: ignore[arg-type]
    text = "\n".join(f"Paragraph number {index}." for index in range(30))
    service.start_reading(text)
    assert reader.batches, "the batched path should have been used"
    assert len(reader.batches) < 30, "far fewer calls than paragraphs"
    from core.reading import BATCH_MAX_CHUNKS

    assert all(len(batch) <= BATCH_MAX_CHUNKS for batch in reader.batches)


def test_chat_service_reading_falls_back_without_batching(data_root) -> None:
    """A reader that only offers ``render`` is still read chunk by chunk."""

    class _Reader:
        def __init__(self) -> None:
            self.spoken: list[str] = []
            self.voice = None
            self.mood = None

        def available(self) -> bool:
            return True

        def render(self, text: str):
            from types import SimpleNamespace

            self.spoken.append(text)
            return SimpleNamespace(silent=False, duration_seconds=0.0)

        def play(self, clip):
            return None

    from gui.model_chat import ModelChatService

    reader = _Reader()
    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root, reader=reader)  # type: ignore[arg-type]
    service.start_reading("First one. Second one.")
    assert reader.spoken


def test_chat_service_sets_and_clears_the_logo(data_root, tmp_path) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_VISION, data_root=data_root)
    stored = service.set_logo(_png(tmp_path / "logo.png"))
    assert stored is not None and stored.is_file()
    assert service.logo_path() == stored
    service.clear_logo()
    assert service.logo_path() is None


def test_chat_service_model_status_is_honest(data_root) -> None:
    """A model that is not downloaded says so, and names the real next step."""
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    assert not service.model_ready()
    assert "download" in service.model_status()


def test_chat_service_model_status_is_ready_when_the_file_is_present(data_root) -> None:
    from gui.model_chat import ModelChatService

    folder = data_root / "ultron"
    folder.mkdir(parents=True)
    from core.ultron import resolve_spec

    (folder / resolve_spec("qwen2.5-7b").filename).write_bytes(b"GGUF")
    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    assert service.model_ready()
    assert "ready" in service.model_status()


def test_chat_service_offers_the_full_catalog(data_root) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    keys = {model.key for model in service.model_choices()}
    assert {"qwen2.5-7b", "ogai-reasoner", "qwen3-8b", "qwen3-4b"} <= keys


def test_chat_service_accepts_an_injected_service(data_root) -> None:
    """A caller's own service is used as-is, so the window is testable."""
    from types import SimpleNamespace

    from gui.model_chat import ModelChatService

    inner = SimpleNamespace(ask=lambda message: f"echo:{message}")
    service = ModelChatService(catalog.MOUTH_ULTRON, service=inner, data_root=data_root)  # type: ignore[arg-type]
    assert service.ask("hello") == "echo:hello"


# --- the custom-model registry ----------------------------------------------


def test_custom_registry_adds_a_gguf_and_offers_it_to_the_mouth(data_root, tmp_path) -> None:
    """A GGUF the user points a window at becomes a selectable model there."""
    from core import custom_models

    gguf = tmp_path / "My-Model.gguf"
    gguf.write_bytes(b"GGUF")
    entry = custom_models.add_model(gguf, title="My Model")
    assert entry.key == "custom:my-model"
    assert entry.available()
    assert entry.path == gguf.resolve()
    # It is offered to the GGUF mouths, marked as the user's own, with a real path.
    offered = catalog.find_language_model(catalog.MOUTH_ULTRON, entry.key)
    assert offered is not None and offered.custom and offered.available
    assert catalog.resolve_model_path(catalog.MOUTH_ULTRON, entry.key) == gguf.resolve()


def test_custom_registry_refuses_a_missing_or_non_gguf_file(data_root, tmp_path) -> None:
    """A bad choice is reported rather than stored."""
    from core import custom_models

    with pytest.raises(ValueError):
        custom_models.add_model(tmp_path / "absent.gguf")
    text = tmp_path / "notes.txt"
    text.write_text("not a model")
    with pytest.raises(ValueError):
        custom_models.add_model(text)


def test_custom_registry_persists_and_removes(data_root, tmp_path) -> None:
    """A remembered model survives a re-read, and can be forgotten."""
    from core import custom_models

    gguf = tmp_path / "Persist.gguf"
    gguf.write_bytes(b"GGUF")
    entry = custom_models.add_model(gguf)
    assert [m.key for m in custom_models.list_models()] == [entry.key]
    assert custom_models.remove_model(entry.key)
    assert custom_models.list_models() == ()
    assert not custom_models.remove_model(entry.key)


def test_custom_registry_reports_a_file_that_left(data_root, tmp_path) -> None:
    """A model whose file is gone is offered but reported as not available."""
    from core import custom_models

    gguf = tmp_path / "Gone.gguf"
    gguf.write_bytes(b"GGUF")
    entry = custom_models.add_model(gguf)
    gguf.unlink()
    offered = catalog.find_language_model(catalog.MOUTH_ULTRON, entry.key)
    assert offered is not None and not offered.available
    assert "not there any more" in offered.detail


def test_custom_registry_tolerates_a_corrupt_store(data_root) -> None:
    """A corrupt registry reads as empty rather than crashing a window."""
    from core import custom_models

    (data_root / custom_models.NAMESPACE).mkdir(parents=True, exist_ok=True)
    (data_root / custom_models.NAMESPACE / "registry.json").write_text("{ not json", encoding="utf-8")
    assert custom_models.list_models() == ()


def test_a_catalog_model_has_a_download_and_a_custom_one_does_not(data_root, tmp_path) -> None:
    """Only a catalog model the project knows how to fetch can be downloaded."""
    from core import custom_models

    known = catalog.find_language_model(catalog.MOUTH_ULTRON, "qwen2.5-7b")
    assert known is not None and catalog.can_download(known)
    assert "ultron download" in known.download_command

    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    entry = custom_models.add_model(gguf)
    mine = catalog.find_language_model(catalog.MOUTH_ULTRON, entry.key)
    assert mine is not None and not catalog.can_download(mine)


def test_download_model_dispatches_to_the_owning_engine(monkeypatch, data_root) -> None:
    """A chosen model's fetch runs through the engine that names it."""
    seen: dict[str, Any] = {}

    def fake_fetch(key, *, approved=False, progress=None):
        seen["key"] = key
        seen["approved"] = approved
        return Path("/tmp/model.gguf")

    monkeypatch.setattr("core.ultron.download_model", fake_fetch)
    path = catalog.download_model(catalog.MOUTH_ULTRON, "qwen2.5-7b", approved=True)
    assert path == Path("/tmp/model.gguf")
    assert seen == {"key": "qwen2.5-7b", "approved": True}


def test_download_model_refuses_a_model_with_nothing_to_fetch(data_root, tmp_path) -> None:
    from core import custom_models

    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    entry = custom_models.add_model(gguf)
    with pytest.raises(ValueError):
        catalog.download_model(catalog.MOUTH_ULTRON, entry.key, approved=True)


# --- the service activates its mouth and manages models ---------------------


def test_service_activates_its_mouth_on_open(data_root) -> None:
    """Opening the window makes this mouth the one answering its own session."""
    from gui.model_chat import ModelChatService

    calls: list[str] = []

    class _Inner:
        def __init__(self) -> None:
            self.mouth = ""

        def ask(self, message: str) -> str:
            return f"echo:{message}"

        def activate_mouth(self, mouth: str) -> str:
            calls.append(mouth)
            self.mouth = mouth
            return mouth

        def mouth_is_active(self, mouth: str) -> bool:
            return self.mouth == mouth

    inner = _Inner()
    service = ModelChatService(catalog.MOUTH_ULTRON, service=inner, data_root=data_root)  # type: ignore[arg-type]
    assert calls == [catalog.MOUTH_ULTRON]
    assert service.active()
    assert "answering here" in service.activation_status()
    # The window's own session answers with this mouth; a *separate* ChatService
    # (the main window's) is untouched by that.
    from gui.service import ChatService

    assert ChatService(data_root=data_root).switcher.mouth == catalog.MOUTH_MODEL


def test_service_activation_reports_the_real_next_step(data_root) -> None:
    """A mouth that cannot load still opens, and says what to do about it."""
    from gui.model_chat import ModelChatService

    # No model file and no runtime here, so Vision cannot come up: the window
    # records the real next step rather than failing to open.
    service = ModelChatService(catalog.MOUTH_VISION, data_root=data_root)
    assert not service.active()
    assert "cannot answer yet" in service.activation_status()
    # A separate ChatService is untouched by the failed activation.
    from gui.service import ChatService

    assert ChatService(data_root=data_root).switcher.mouth == catalog.MOUTH_MODEL


def test_service_add_model_makes_it_active(data_root, tmp_path) -> None:
    """Adding a GGUF file uses it here at once."""
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    key = service.add_model(gguf, title="Mine")
    assert key == "custom:mine"
    assert service.model_key() == key
    assert any(model.key == key and model.custom for model in service.model_choices())
    assert service.can_remove(key)
    assert not service.can_download(key)


def test_service_remove_model_falls_back_to_the_default(data_root, tmp_path) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    key = service.add_model(gguf)
    assert service.remove_model(key)
    assert service.model_key() == ""


def test_service_refuses_to_remove_a_catalog_model(data_root) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    assert not service.can_remove("qwen2.5-7b")
    assert not service.remove_model("qwen2.5-7b")


def test_service_offers_no_add_for_the_retrieval_mouth(data_root, tmp_path) -> None:
    """The sources mouth has no language model, so it refuses to add one."""
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_SOURCES, data_root=data_root)
    assert not service.can_add_model()
    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    with pytest.raises(ValueError):
        service.add_model(gguf)


def test_service_offers_add_for_a_model_mouth(data_root) -> None:
    from gui.model_chat import ModelChatService

    assert ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root).can_add_model()


def test_service_reports_the_download_for_the_chosen_model(data_root) -> None:
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    assert service.can_download()
    assert "ultron download" in service.download_command()


def test_service_saves_a_model_choice_even_when_it_cannot_load_yet(data_root) -> None:
    """Choosing a model whose runtime is missing is remembered, not lost."""
    from gui.model_chat import ModelChatService

    service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
    # No runtime here, so activation cannot succeed, but the choice must persist.
    assert service.set_model("qwen2.5-3b") == "qwen2.5-3b"
    assert profiles.load_profile("ultron").model == "qwen2.5-3b"
    assert ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root).model_key() == "qwen2.5-3b"


def test_set_mouth_model_drops_the_cached_assistant(data_root) -> None:
    """A model switch must rebuild the mouth, not reuse the cached assistant."""
    from gui.service import ChatService

    service = ChatService(data_root=data_root)
    dropped: list[str] = []
    service.switcher.invalidate_mouth = dropped.append  # type: ignore[assignment]
    service.set_mouth_model(catalog.MOUTH_ULTRON, "qwen2.5-3b")
    assert dropped == [catalog.MOUTH_ULTRON]
    assert service.mouth_model(catalog.MOUTH_ULTRON) == "qwen2.5-3b"


# --- the CLI ----------------------------------------------------------------


def _cli(argv: list[str], capsys) -> tuple[int, str]:
    from cli import model_chat

    code = model_chat.main(argv)
    return code, capsys.readouterr().out


def test_cli_lists_the_mouths(data_root, capsys) -> None:
    code, out = _cli(["mouths"], capsys)
    assert code == 0
    for mouth in MOUTHS:
        assert mouth in out


def test_cli_lists_a_mouths_models_and_marks_the_choice(data_root, capsys) -> None:
    code, out = _cli(["models", "ultron"], capsys)
    assert code == 0
    assert "Qwen2.5 7B" in out
    assert "OGAI Reasoner" in out


def test_cli_shows_a_profile_as_json(data_root, capsys) -> None:
    code, out = _cli(["profile", "natasha", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["mouth"] == "natasha"
    assert payload["name"] == "Natasha"
    assert "available" in payload


def test_cli_sets_and_reports_the_model(data_root, capsys) -> None:
    code, out = _cli(["set-model", "ultron", "qwen2.5-3b"], capsys)
    assert code == 0
    assert "qwen2.5-3b" in out
    assert profiles.load_profile("ultron").model == "qwen2.5-3b"


def test_cli_refuses_a_model_the_window_does_not_offer(data_root, capsys) -> None:
    code, out = _cli(["set-model", "ultron", "no-such-model"], capsys)
    assert code == 2
    assert "not one of the models" in out


def test_cli_sets_the_voice_and_keeps_it_exclusive(data_root, capsys) -> None:
    code, _ = _cli(["set-voice", "vision", "woman-4"], capsys)
    assert code == 0
    assert profiles.load_profile("vision").voice == "woman-4"
    assert profiles.load_profile("ultron").voice == ""


def test_cli_rejects_an_unknown_voice(data_root, capsys) -> None:
    code, out = _cli(["set-voice", "vision", "not-a-voice"], capsys)
    assert code == 2
    assert out.strip()


def test_cli_sets_the_name(data_root, capsys) -> None:
    code, _ = _cli(["set-name", "hakeem", "Scholar"], capsys)
    assert code == 0
    assert profiles.load_profile("hakeem").name == "Scholar"


def test_cli_sets_and_clears_the_logo(data_root, tmp_path, capsys) -> None:
    source = _png(tmp_path / "logo.png")
    code, _out = _cli(["set-logo", "banner", str(source)], capsys)
    assert code == 0
    assert profiles.load_profile("banner").logo == "banner.png"
    code, _ = _cli(["clear-logo", "banner"], capsys)
    assert code == 0
    assert profiles.load_profile("banner").logo == ""


def test_cli_reports_a_bad_logo(data_root, tmp_path, capsys) -> None:
    bad = tmp_path / "logo.bmp"
    bad.write_bytes(b"BM")
    code, out = _cli(["set-logo", "banner", str(bad)], capsys)
    assert code == 4
    assert "PNG" in out


def test_cli_rejects_an_unknown_mouth(data_root, capsys) -> None:
    code, out = _cli(["profile", "not-a-mouth"], capsys)
    assert code == 2
    assert "Unknown model" in out


def test_cli_adds_a_model_from_a_file(data_root, tmp_path, capsys) -> None:
    """A GGUF file is added, made active, and remembered."""
    from core import custom_models

    gguf = tmp_path / "My-Model.gguf"
    gguf.write_bytes(b"GGUF")
    code, out = _cli(["add-model", "ultron", str(gguf), "--title", "My Model"], capsys)
    assert code == 0
    assert "custom:my-model" in out
    entry = custom_models.find_model("custom:my-model")
    assert entry is not None and entry.title == "My Model"
    assert profiles.load_profile("ultron").model == "custom:my-model"


def test_cli_reports_a_bad_model_file(data_root, tmp_path, capsys) -> None:
    code, out = _cli(["add-model", "ultron", str(tmp_path / "absent.gguf")], capsys)
    assert code == 2
    assert out.strip()


def test_cli_removes_an_added_model(data_root, tmp_path, capsys) -> None:
    from core import custom_models

    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    _cli(["add-model", "ultron", str(gguf)], capsys)
    code, out = _cli(["remove-model", "ultron", "custom:mine"], capsys)
    assert code == 0
    assert "Forgot" in out
    assert custom_models.list_models() == ()


def test_cli_refuses_to_remove_a_catalog_model(data_root, capsys) -> None:
    code, out = _cli(["remove-model", "ultron", "qwen2.5-7b"], capsys)
    assert code == 2
    assert "added" in out


def test_cli_download_requires_approval(data_root, capsys) -> None:
    """A download is an external action, so it needs the explicit approval."""
    code, out = _cli(["download-model", "ultron", "qwen2.5-7b"], capsys)
    assert code == 2
    assert "--approve" in out


def test_cli_download_refuses_a_model_with_nothing_to_fetch(data_root, tmp_path, capsys) -> None:
    gguf = tmp_path / "Mine.gguf"
    gguf.write_bytes(b"GGUF")
    _cli(["add-model", "ultron", str(gguf)], capsys)
    code, out = _cli(["download-model", "ultron", "custom:mine", "--approve"], capsys)
    assert code == 2
    assert out.strip()


def test_cli_download_fetches_through_the_engine(monkeypatch, data_root, capsys) -> None:
    def fake_fetch(key, *, approved=False, progress=None):
        assert key == "qwen2.5-7b"
        assert approved is True
        return Path("/tmp/model.gguf")

    monkeypatch.setattr("core.ultron.download_model", fake_fetch)
    code, out = _cli(["download-model", "ultron", "qwen2.5-7b", "--approve"], capsys)
    assert code == 0
    assert "Model ready" in out


# --- the window -------------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_the_identity_the_voice_and_the_model(data_root) -> None:
    """The window's left side carries the logo, the voice menu, and the model list."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        # 1. The logo and the name.
        assert dialog._logo_canvas.winfo_ismapped()
        assert dialog._name_var.get() == "Ultron"
        # 2. The reading voice, exclusive to this mouth.
        assert dialog._voice_var.get() == service.voice_key()
        # 3. The language model list, with the mouth's default pre-selected.
        assert dialog._model_list.size() == len(service.model_choices())
        assert "qwen2.5-7b" in dialog._model_keys
        # And a microphone in the composer, when the machine has one.
        assert dialog._composer.mic_button is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_the_web_switch_only_for_the_sources_mouth(data_root) -> None:
    """The dedicated window carries the lookup switch, greyed for a model mouth.

    Web search belongs to the grounded *sources* mouth, so a language-model window
    (Ultron here) shows the switch honestly unavailable -- a click cannot turn on a
    lookup that mouth cannot perform.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        toggle = dialog._composer.web_toggle
        assert toggle is not None and toggle.winfo_ismapped()
        assert service.web_available() is False
        assert toggle._available is False
        # A click on an unavailable switch asks for nothing.
        toggle._on_click()
        assert service.web_enabled() is False
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_web_switch_turns_lookups_on_for_the_sources_mouth(data_root) -> None:
    """The sources window's switch really flips the session's lookup state."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_SOURCES, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        toggle = dialog._composer.web_toggle
        assert toggle is not None
        assert service.web_available() is True
        assert toggle._on is False
        # The window's own handler turns it on and the switch follows the truth.
        dialog._on_web_toggled(True)
        assert service.web_enabled() is True
        assert toggle._on is True
        dialog._on_web_toggled(False)
        assert service.web_enabled() is False
        assert toggle._on is False
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_opens_at_ninety_percent_height_and_keeps_both_scrollbars(data_root) -> None:
    """The window is shorter than before, and both areas always show a scrollbar.

    The height is 90% of the window's original 720px (648), and the left column
    and the conversation each carry a scrollbar from the first moment -- they are
    never a dead end on a short window.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import DEFAULT_HEIGHT, DEFAULT_WIDTH, ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        assert DEFAULT_HEIGHT == int(720 * 0.9)
        assert DEFAULT_WIDTH == 1180
        # The scrollbars are mapped as soon as the window is drawn, not only once
        # the content overflows.
        assert dialog._sidebar_scrollbar.winfo_ismapped()
        assert dialog._scrollbar.winfo_ismapped()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_initial_geometry_is_clamped_to_the_work_area(monkeypatch, data_root) -> None:
    """A small work area wins over the default, so the composer stays reachable."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        monkeypatch.setattr(
            "gui.window_controls.work_area", lambda: (0, 30, 1000, 600)
        )
        assert dialog._initial_geometry() == "1000x600"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_maximize_fills_the_work_area_and_restores(data_root) -> None:
    """The title bar's maximize grows to the usable desktop, then back."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        controls = dialog._window_controls
        controls.maximize()
        root.update()
        # Whatever path the window manager took, the window is grown now and the
        # button offers a restore.
        assert controls._is_open()
        controls.restore()
        root.update()
        assert not controls._is_open()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_rename_and_logo_change_persist(data_root, tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_VISION, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._name_var.set("البصيرة")
        dialog._on_rename()
        root.update()
        assert profiles.load_profile("vision").name == "البصيرة"
        assert dialog._header_title.cget("text") == "البصيرة"

        service.set_logo(_png(tmp_path / "logo.png"))
        dialog._refresh_identity()
        root.update()
        assert dialog._logo_photo is not None
        dialog._on_reset_logo()
        root.update()
        assert service.logo_path() is None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_model_choice_persists(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        index = dialog._model_keys.index("qwen2.5-3b")
        dialog._model_list.selection_clear(0, "end")
        dialog._model_list.selection_set(index)
        dialog._on_model_select(None)  # type: ignore[arg-type]
        root.update()
        assert service.model_key() == "qwen2.5-3b"
        assert profiles.load_profile("ultron").model == "qwen2.5-3b"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_voice_choice_is_remembered(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_NATASHA, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._on_voice_change("woman-2")
        root.update()
        assert profiles.load_profile("natasha").voice == "woman-2"
        assert service.voice_key() == "woman-2"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_offers_add_download_and_forget_for_the_model(data_root, tmp_path) -> None:
    """The window carries the add-a-model, download, and forget controls.

    The three controls are wired to the service: *Download* and *Forget* are enabled
    only when they apply to the chosen model, and adding a GGUF file uses it here.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        # The three controls exist, and Download is live for a catalog model.
        assert dialog._add_model_button.winfo_ismapped()
        assert dialog._download_model_button.winfo_ismapped()
        assert dialog._remove_model_button.winfo_ismapped()
        assert str(dialog._download_model_button.cget("state")) == "normal"
        assert str(dialog._remove_model_button.cget("state")) == "disabled"

        # Adding a GGUF file makes it the active model and enables Forget.
        gguf = tmp_path / "Mine.gguf"
        gguf.write_bytes(b"GGUF")
        key = service.add_model(gguf, title="Mine")
        dialog._populate_models()
        root.update()
        assert key in dialog._model_keys
        assert str(dialog._remove_model_button.cget("state")) == "normal"
        assert str(dialog._download_model_button.cget("state")) == "disabled"

        # Forgetting it falls back to the mouth's default.
        dialog._on_remove_model()
        root.update()
        assert key not in dialog._model_keys
        assert service.model_key() == ""
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_reports_its_activation(data_root) -> None:
    """The model note says whether this window's mouth is the one answering."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_VISION, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        note = dialog._model_note.cget("text")
        assert "cannot answer yet" in note
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_closes_cleanly_and_calls_back(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    root = tk.Tk()
    closed: list[bool] = []
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, data_root=data_root)
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono",
            on_close=lambda: closed.append(True),
        )
        root.update()
        dialog._close()
        root.update()
        assert closed == [True]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_message_has_a_read_aloud_toggle(data_root) -> None:
    """Every message carries the shared footer: Copy and a Read aloud / Stop toggle.

    The footer is the main window's own (``gui.message_footer``), so a per-model
    window reads exactly like the chat window: the control starts on ``Read aloud``,
    switches to ``Stop`` while that message is the one being spoken, and a second
    click stops the reading and returns it to ``Read aloud``.
    """
    import threading
    import tkinter as tk
    from types import SimpleNamespace

    from gui.app import DARK_THEME
    from gui.message_footer import READ_LABEL, STOP_LABEL
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    # The reader blocks in `play` until told to stop, so the reading stays in
    # progress long enough to observe the Stop state and to stop it.
    gate = threading.Event()
    stopped: list[bool] = []
    spoken: list[str] = []

    class _Voice:
        key = "woman-1"
        engine = "fake"

        def label(self) -> str:
            return self.key

        def category_label(self) -> str:
            return "Women"

    class _Playback:
        def wait(self) -> None:
            gate.wait(2)

        def stop(self) -> None:
            stopped.append(True)
            gate.set()

    class _Reader:
        """A reader with the small surface the window needs, no engine at all."""

        voice = _Voice()
        mood = None

        def available(self) -> bool:
            return True

        def set_voice(self, key: str) -> _Voice:
            self.voice = _Voice()
            self.voice.key = key
            return self.voice

        def render(self, text: str):
            spoken.append(text)
            return SimpleNamespace(silent=False, duration_seconds=0.0)

        def speak(self, text: str):
            spoken.append(text)
            return SimpleNamespace(silent=False, duration_seconds=0.0)

        def play(self, clip):
            return _Playback()

        def refresh(self) -> None:
            return None

    root = tk.Tk()
    try:
        service = ModelChatService(
            catalog.MOUTH_ULTRON, data_root=data_root, reader=_Reader()  # type: ignore[arg-type]
        )
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._add_message("assistant", "porosity is the fraction of pore space")
        root.update()
        assert dialog._footers, "the message must carry a footer"
        footer = dialog._footers[-1]
        assert footer.speaker is not None
        assert str(footer.speaker.cget("text")) == READ_LABEL
        # Start the reading: the control turns into Stop at once.
        dialog._toggle_footer_reading(footer)
        assert str(footer.speaker.cget("text")) == STOP_LABEL
        # Let the worker reach `play`, so the reading is genuinely in progress.
        for _ in range(300):
            root.update()
            time.sleep(0.01)
            if spoken:
                break
        assert spoken, "the message should have been read aloud"
        # A second click stops it, and the control returns to Read aloud.
        dialog._toggle_footer_reading(footer)
        assert stopped, "the reading in progress should have been stopped"
        assert str(footer.speaker.cget("text")) == READ_LABEL
    finally:
        gate.set()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_asks_through_its_own_service(data_root) -> None:
    """Sending a message asks the injected service and shows the reply."""
    import tkinter as tk
    from types import SimpleNamespace

    from gui.app import DARK_THEME
    from gui.model_chat import ModelChatService
    from gui.model_chat_dialog import ModelChatDialog

    inner = SimpleNamespace(
        ask=lambda message: SimpleNamespace(text=f"you said {message}", hint="", citations=()),
        mouth_model=lambda mouth: "",
        set_mouth_model=lambda mouth, key: None,
        microphone_available=lambda: False,
    )
    root = tk.Tk()
    try:
        service = ModelChatService(catalog.MOUTH_ULTRON, service=inner, data_root=data_root)  # type: ignore[arg-type]
        dialog = ModelChatDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._composer.set_text("hello there")
        dialog._on_send()
        # Wait for the worker thread's result to be drained by the UI thread.
        # A real (short) sleep is needed: the result crosses a thread boundary.
        for _ in range(300):
            root.update()
            time.sleep(0.01)
            if any(role == "assistant" for role, *_ in dialog._rows):
                break
        texts = [text for _role, text, *_ in dialog._rows]
        assert any("you said hello there" in text for text in texts)
    finally:
        root.destroy()
