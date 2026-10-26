"""The per-engine speech studios: one window per engine, its own voice and mood.

The feature this pins: each speech engine the project can speak through (the
voices the OS installed, VoiceTut-TTS over audio.cpp, Piper, and Qwen3-TTS) gets its
own studio, where a document is loaded and read aloud with three things exclusive
to that engine -- its logo/name, its reading voice (drawn only from the voices that
engine really reports), and its mood (applied at once and remembered).

The profile store, the service, the CLI, and the window all run headless with no
display, no model, and no audio hardware: a reader may be injected, and a fake one
records what was asked of it. The window parts are skipped unless a display (or
Xvfb) is available, exactly like ``tests/test_gui.py``.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from core import voice_studio_profile as profiles
from core.speech import AudioClip
from core.voices import (
    SOURCE_AUDIOCPP,
    SOURCE_PIPER,
    SOURCE_QWEN,
    SOURCE_SYSTEM,
    ModelVoice,
    SystemVoice,
)
from gui.voice_studio import VoiceStudioService


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
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", header)
    data += chunk(b"IDAT", zlib.compress(raw))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)
    return path


class FakeReader:
    """A reader that records what it was asked to speak, with no real engine."""

    def __init__(self, *, available: bool = True) -> None:
        self.spoken: list[str] = []
        self.batches: list[list[str]] = []
        self.engine_name = "fake-engine" if available else "silent"
        self._available = available
        self.voice = _FakeVoice()
        self.mood = None

    def available(self) -> bool:
        return self._available

    def describe(self) -> str:
        return "A fake reader for tests."

    def set_voice(self, key):
        self.voice = _FakeVoice(key)
        return self.voice

    def set_mood(self, mood):
        from core.voice_moods import resolve_mood

        self.mood = resolve_mood(mood) if not hasattr(mood, "key") else mood
        return self.mood

    def speak(self, text):
        self.spoken.append(text)
        if not self._available:
            return AudioClip.silence(0.1, sample_rate=16000)
        return AudioClip(tuple([1000] * 8), sample_rate=16000)

    def render(self, text):
        """Render a chunk (the stoppable path plays each chunk in turn)."""
        self.spoken.append(text)
        if not self._available:
            return AudioClip.silence(0.1, sample_rate=16000)
        return AudioClip(tuple([1000] * 8), sample_rate=16000)

    def render_many(self, texts):
        """Render a whole batch in one call, so a batching engine loads once."""
        self.batches.append(list(texts))
        return [self.render(text) for text in texts]

    def play(self, clip):
        """Play a rendered chunk; a recording reader returns an instant playback."""
        return _FakePlayback()

    def save(self, text, path):
        Path(path).write_bytes(b"RIFF")
        return Path(path)


class _FakePlayback:
    """A playback that finishes at once, so a reading test does not block."""

    def wait(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _FakeVoice:
    def __init__(self, key: str = "woman-1") -> None:
        self.key = key
        self.engine = "fake"

    def label(self) -> str:
        return self.key

    def category_label(self) -> str:
        return "Women"


# ------------------------------------------------------------------ profiles


def test_the_four_studios_are_declared_with_the_titles_asked_for() -> None:
    assert profiles.source_names() == (
        SOURCE_SYSTEM, SOURCE_AUDIOCPP, SOURCE_PIPER, SOURCE_QWEN
    )
    assert "Arabic" in profiles.studio_title(SOURCE_SYSTEM)
    assert "VoiceTut" in profiles.studio_title(SOURCE_AUDIOCPP)
    assert "Piper" in profiles.studio_title(SOURCE_PIPER)
    assert "Qwen3-TTS" in profiles.studio_title(SOURCE_QWEN)


def test_a_local_studio_starts_on_its_own_engines_voice(monkeypatch, data_root) -> None:
    """The regression this fixes: a local studio must not start on a catalog voice.

    A local engine cannot speak the synthetic catalog's ``woman-1`` -- audio.cpp has
    no reference clip to clone, Piper has no matching ONNX file, and Qwen3-TTS has no
    speaker by that name -- so before this fix every local studio handed its engine
    an unspeakable key, the engine failed, and the reading fell back to the system
    engine (the Windows default voice) instead of the voice the studio offered.
    """
    piper = ModelVoice(name="ar_JO-kareem-medium", engine=SOURCE_PIPER, title="Kareem")
    monkeypatch.setattr("gui.voice_studio.system_voices", lambda: ())
    monkeypatch.setattr("gui.voice_studio.model_voices", lambda engine=None: (piper,))
    reader = FakeReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    assert studio.voice_key() == "model:piper:ar_JO-kareem-medium"
    # The reader was handed the engine's own voice, not the catalog default.
    assert reader.voice.key == "model:piper:ar_JO-kareem-medium"


def test_each_local_studio_starts_on_its_own_engine(monkeypatch, data_root) -> None:
    """Each local engine is offered *its* voice, never another engine's."""
    voices = (
        ModelVoice(name="Sarah", engine=SOURCE_AUDIOCPP, title="Sarah"),
        ModelVoice(name="kareem", engine=SOURCE_PIPER, title="Kareem"),
        ModelVoice(name="Cherry", engine=SOURCE_QWEN, title="Cherry"),
    )
    monkeypatch.setattr("gui.voice_studio.system_voices", lambda: ())
    monkeypatch.setattr("gui.voice_studio.model_voices", lambda engine=None: voices)
    for source, expected in (
        (SOURCE_AUDIOCPP, "model:audio.cpp:Sarah"),
        (SOURCE_PIPER, "model:piper:kareem"),
        (SOURCE_QWEN, "model:qwen3-tts:Cherry"),
    ):
        studio = VoiceStudioService(source, reader=FakeReader())
        assert studio.voice_key() == expected


def test_a_saved_local_voice_is_kept_when_still_offered(monkeypatch, data_root) -> None:
    piper = ModelVoice(name="ar_JO-kareem-medium", engine=SOURCE_PIPER, title="Kareem")
    monkeypatch.setattr("gui.voice_studio.system_voices", lambda: ())
    monkeypatch.setattr("gui.voice_studio.model_voices", lambda engine=None: (piper,))
    first = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    first.set_voice("model:piper:ar_JO-kareem-medium")
    # A second open of the same studio keeps the saved voice.
    again = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    assert again.voice_key() == "model:piper:ar_JO-kareem-medium"


def test_the_system_studio_may_still_start_on_the_catalog(monkeypatch, data_root) -> None:
    """The synthetic catalog is settings for the *system* engine, so it stays its default."""
    monkeypatch.setattr("gui.voice_studio.system_voices", lambda: ())
    studio = VoiceStudioService(SOURCE_SYSTEM, reader=FakeReader())
    assert studio.voice_key() == "woman-1"


def test_setup_status_separates_the_engine_from_its_voices(monkeypatch, data_root) -> None:
    """The two halves are reported apart, each with its own next step."""
    from core.piper_speech import PiperStatus

    # Runtime present, voice missing: install is *not* offered, download is.
    monkeypatch.setattr(
        "core.piper_speech.piper_status",
        lambda root=None: PiperStatus(True, False, None, "Download a voice with: …"),
    )
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    status = studio.setup_status()
    assert status.runtime_ok and not status.voices_ok and not status.ready
    assert not status.can_install_runtime
    assert status.can_download_voices
    assert [key for key, _title in studio.fetchable_voices()]


def test_setup_status_offers_the_install_when_the_engine_is_absent(
    monkeypatch, data_root
) -> None:
    from core.piper_speech import PiperStatus

    monkeypatch.setattr(
        "core.piper_speech.piper_status",
        lambda root=None: PiperStatus(False, False, None, "Install the Piper runtime with: …"),
    )
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    status = studio.setup_status()
    assert not status.runtime_ok
    assert status.can_install_runtime
    # You cannot download a voice for an engine that cannot run it.
    assert not status.can_download_voices
    assert studio.fetchable_voices() == ()


def test_download_voices_refuses_before_the_engine_is_installed(
    monkeypatch, data_root
) -> None:
    from core.piper_speech import PiperStatus

    monkeypatch.setattr(
        "core.piper_speech.piper_status",
        lambda root=None: PiperStatus(False, False, None, "Install the Piper runtime with: …"),
    )
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    report = studio.download_voices("ar_JO-kareem-medium")
    assert not report.ok
    assert "install the engine first" in report.message.lower()


def test_a_governed_setup_failure_is_reported_not_raised(monkeypatch, data_root) -> None:
    """A download that fails must be reported, never crash the window."""
    from core.piper_speech import PiperStatus

    monkeypatch.setattr(
        "core.piper_speech.piper_status",
        lambda root=None: PiperStatus(True, False, None, "Download a voice with: …"),
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("the network is down")

    monkeypatch.setattr("core.piper_speech.download_model", boom)
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    report = studio.download_voices("ar_JO-kareem-medium")
    assert not report.ok
    assert "network is down" in report.message


def test_a_profile_round_trips_through_the_store(data_root) -> None:
    saved = profiles.save_profile(
        profiles.default_profile(SOURCE_PIPER).with_voice("man-1").with_mood("happy")
    )
    assert saved.voice == "man-1" and saved.mood == "happy"
    again = profiles.load_profile(SOURCE_PIPER)
    assert again.voice == "man-1"
    assert again.mood == "happy"


def test_voice_and_mood_are_exclusive_to_their_engine(data_root) -> None:
    profiles.set_voice(SOURCE_PIPER, "man-1")
    profiles.set_mood(SOURCE_PIPER, "sad")
    # Another engine keeps its own defaults.
    other = profiles.load_profile(SOURCE_QWEN)
    assert other.voice == ""
    assert other.mood == ""
    assert profiles.resolve_mood(SOURCE_QWEN) == "neutral"


def test_a_saved_choice_that_left_the_catalog_falls_back(data_root) -> None:
    profiles.save_profile(profiles.default_profile(SOURCE_PIPER).with_mood("gone"))
    assert profiles.resolve_mood(SOURCE_PIPER) == "neutral"
    # A voice no longer installed falls back to the first available.
    profiles.set_voice(SOURCE_PIPER, "gone")
    assert profiles.resolve_voice(SOURCE_PIPER, ("a", "b")) == "a"


def test_a_corrupt_profile_falls_back_rather_than_raising(data_root) -> None:
    from storage.json_store import JsonStore

    JsonStore(data_root).save(profiles.NAMESPACE, SOURCE_PIPER, "not-a-dict")
    profile = profiles.load_profile(SOURCE_PIPER)
    assert profile.source == SOURCE_PIPER
    assert profile.name == profiles.DEFAULT_NAMES[SOURCE_PIPER]


def test_the_logo_is_copied_in_and_can_be_cleared(data_root, tmp_path) -> None:
    image = _png(tmp_path / "logo.png")
    profile = profiles.save_logo(SOURCE_PIPER, image)
    assert profile.logo == f"voice-{SOURCE_PIPER}.png"
    path = profile.logo_path()
    assert path is not None and path.is_file()
    cleared = profiles.clear_logo(SOURCE_PIPER)
    assert cleared.logo == ""
    assert cleared.logo_path() is None


def test_a_bad_logo_is_refused(data_root, tmp_path) -> None:
    bad = tmp_path / "not-an-image.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        profiles.save_logo(SOURCE_PIPER, bad)
    with pytest.raises(ValueError):
        profiles.save_logo(SOURCE_PIPER, tmp_path / "missing.png")


def test_a_studio_logo_never_collides_with_a_mouth_logo(data_root, tmp_path) -> None:
    image = _png(tmp_path / "logo.png")
    profiles.save_logo(SOURCE_PIPER, image)
    assert profiles.logo_filename(SOURCE_PIPER, "x.png").startswith("voice-")


# ------------------------------------------------------------------ service


def test_the_service_lists_only_its_own_engines_voices(monkeypatch) -> None:
    system = (SystemVoice("Microsoft Hoda", "ar-SA", "Female"),)
    piper = (ModelVoice(name="ar_JO-kareem-medium", engine=SOURCE_PIPER),)
    monkeypatch.setattr("gui.voice_studio.system_voices", lambda: system)
    monkeypatch.setattr(
        "gui.voice_studio.model_voices", lambda: (*piper, ModelVoice(name="x", engine=SOURCE_QWEN))
    )
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    keys = studio.voice_keys()
    # Only the Piper voice (and the synthetic catalog, offered by the system studio
    # alone -- which this is not) is here.
    assert "model:piper:ar_JO-kareem-medium" in keys
    assert "model:qwen3-tts:x" not in keys
    assert not any(key.startswith("system:") for key in keys)


def test_the_system_studio_offers_the_installed_voices_and_the_catalog(monkeypatch) -> None:
    system = (SystemVoice("Microsoft Hoda", "ar-SA", "Female"),)
    monkeypatch.setattr("gui.voice_studio.system_voices", lambda: system)
    studio = VoiceStudioService(SOURCE_SYSTEM, reader=FakeReader())
    keys = studio.voice_keys()
    assert "system:system:Microsoft Hoda" in keys
    assert "woman-1" in keys  # the synthetic catalog belongs to this studio


def test_a_local_model_studio_is_never_offered_the_synthetic_catalog(monkeypatch) -> None:
    monkeypatch.setattr("gui.voice_studio.model_voices", lambda: ())
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    assert studio.catalog_voices() == ()


def test_set_voice_is_exclusive_and_persists(monkeypatch, data_root) -> None:
    monkeypatch.setattr(
        "gui.voice_studio.system_voices", lambda: (SystemVoice("Hoda", "ar-SA", "Female"),)
    )
    studio = VoiceStudioService(SOURCE_SYSTEM, reader=FakeReader())
    chosen = studio.set_voice("system:system:Hoda")
    assert chosen == "system:system:Hoda"
    assert profiles.load_profile(SOURCE_SYSTEM).voice == chosen
    # Another engine's profile is untouched.
    assert profiles.load_profile(SOURCE_PIPER).voice == ""


def test_set_voice_refuses_a_key_this_studio_does_not_offer(data_root) -> None:
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    with pytest.raises(ValueError):
        studio.set_voice("woman-1")  # a catalog voice, not this engine's


def test_set_mood_applies_at_once_and_persists(data_root) -> None:
    reader = FakeReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    chosen = studio.set_mood("happy")
    assert chosen == "happy"
    assert reader.mood is not None and reader.mood.key == "happy"
    assert profiles.load_profile(SOURCE_PIPER).mood == "happy"
    with pytest.raises(ValueError):
        studio.set_mood("nope")


def test_the_service_loads_a_document_and_reads_it(data_root, tmp_path) -> None:
    doc = tmp_path / "report.txt"
    doc.write_text("Porosity increases permeability. The reservoir is tight.\n")
    reader = FakeReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    loaded = studio.load_document(doc)
    assert loaded.filename == "report.txt"
    assert "Porosity" in studio.document_text()
    report = studio.read_aloud()
    assert report.ok
    assert report.chunks >= 1
    assert reader.spoken  # something was actually asked to be spoken


def test_reading_without_a_document_is_reported_honestly(data_root) -> None:
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    report = studio.read_aloud()
    assert not report.ok
    assert "no document" in report.message.lower()


def test_reading_with_no_engine_is_reported_honestly(data_root) -> None:
    from gui.voice_studio import LoadedDocument

    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader(available=False))
    studio.document = LoadedDocument(path=Path("x.txt"), text="hello there", kind="text")
    report = studio.read_aloud()
    assert not report.ok
    assert "engine" in report.message.lower()


def test_a_long_document_is_split_into_chunks(data_root, tmp_path) -> None:
    from gui.voice_studio import _split_for_speech

    text = " ".join(f"Sentence number {index}." for index in range(200))
    chunks = _split_for_speech(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_the_shared_split_helper_is_the_one_the_studio_uses() -> None:
    """The studio's splitter is the shared one, so every window chunks alike."""
    from core.reading import split_for_speech
    from gui.voice_studio import _split_for_speech

    text = "First one. Second one.\n\nThird one."
    assert _split_for_speech(text) == split_for_speech(text)


def test_clip_is_silent_reads_a_real_clip_and_a_caller_double() -> None:
    """The shared check sees silence whether the clip says ``is_silent`` or ``silent``.

    A real :class:`~core.speech.AudioClip` exposes ``is_silent``; the windows used
    to read only ``silent``, which a real clip does not have, so an empty render was
    reported as spoken. Both names, and a missing clip, are honoured here.
    """
    from types import SimpleNamespace

    from core.reading import clip_is_silent
    from core.speech import AudioClip

    assert clip_is_silent(AudioClip(tuple([1000] * 8), sample_rate=16000)) is False
    assert clip_is_silent(AudioClip.silence(0.1, sample_rate=16000)) is True
    # A caller's own double may name the flag ``silent`` instead.
    assert clip_is_silent(SimpleNamespace(silent=True)) is True
    assert clip_is_silent(SimpleNamespace(silent=False)) is False
    assert clip_is_silent(None) is True


def test_start_reading_plays_each_chunk_and_can_be_stopped(data_root, tmp_path) -> None:
    """The stoppable path plays the document and reports a stop honestly."""
    doc = tmp_path / "report.txt"
    doc.write_text("Porosity increases permeability. The reservoir is tight.\n")
    reader = FakeReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    studio.load_document(doc)
    report = studio.start_reading()
    assert report.ok
    assert report.chunks >= 1
    assert reader.spoken
    # With nothing running, a stop is a no-op that reports so.
    assert studio.stop_reading() is False


def test_stop_reading_ends_a_reading_in_progress(data_root, tmp_path) -> None:
    """A stop asked for mid-read ends the reading and is reported as stopped."""
    import threading

    doc = tmp_path / "report.txt"
    doc.write_text(" ".join(f"Sentence {index}." for index in range(60)))

    class _BlockingReader(FakeReader):
        def __init__(self) -> None:
            super().__init__()
            self.gate = threading.Event()
            self.playing = False

        def play(self, clip):
            self.playing = True
            outer = self

            class _Blocking:
                def wait(self) -> None:
                    outer.gate.wait(2)

                def stop(self) -> None:
                    outer.gate.set()

            return _Blocking()

    reader = _BlockingReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    studio.load_document(doc)
    result: list[Any] = []
    worker = threading.Thread(target=lambda: result.append(studio.start_reading()))
    worker.start()
    # Wait until the reading is genuinely in progress, then stop it.
    for _ in range(400):
        if studio.reading():
            break
        threading.Event().wait(0.005)
    assert studio.reading()
    assert studio.stop_reading() is True
    worker.join(timeout=3)
    assert result and result[0].ok
    assert "stopped" in result[0].message.lower()
    assert not studio.reading()


def test_a_paragraph_with_no_sentence_marks_is_still_bounded() -> None:
    from gui.voice_studio import _split_for_speech

    chunks = _split_for_speech("a" * 1300, max_chars=600)
    assert all(len(chunk) <= 600 for chunk in chunks)
    assert "".join(chunks) == "a" * 1300


def test_the_studio_batches_chunks_into_few_engine_calls(data_root, tmp_path) -> None:
    """A document is rendered in batches, not one engine call per paragraph.

    This is the hang fix: a local engine (audio.cpp) reloads its 1.35 GB model per
    process, so a paragraph-per-call read spent its time loading weights. The studio
    sends a whole batch at once through the reader's batched path.
    """
    doc = tmp_path / "long.txt"
    doc.write_text("\n".join(f"Paragraph number {index}." for index in range(30)))
    reader = FakeReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    studio.load_document(doc)
    report = studio.read_aloud()
    assert report.ok
    assert len(reader.batches) >= 1
    # Every chunk was rendered, and no batch exceeded the cap.
    assert sum(len(batch) for batch in reader.batches) == report.chunks
    from gui.voice_studio import BATCH_MAX_CHUNKS

    assert all(len(batch) <= BATCH_MAX_CHUNKS for batch in reader.batches)
    # Far fewer calls than chunks, which is the whole point.
    assert len(reader.batches) < report.chunks


def test_batch_chunks_keeps_order_and_bounds() -> None:
    from gui.voice_studio import BATCH_MAX_CHARS, BATCH_MAX_CHUNKS, _batch_chunks

    chunks = tuple(f"chunk-{index}-{'x' * 100}" for index in range(20))
    batches = _batch_chunks(chunks)
    assert [chunk for batch in batches for chunk in batch] == list(chunks)
    assert all(len(batch) <= BATCH_MAX_CHUNKS for batch in batches)
    assert all(sum(len(chunk) for chunk in batch) <= BATCH_MAX_CHARS + 200 for batch in batches)
    # A single oversized chunk becomes a batch of its own, never dropped.
    assert _batch_chunks(("z" * 9000,)) == (("z" * 9000,),)
    assert _batch_chunks(()) == ()


def test_render_group_uses_the_batched_path_when_available(data_root) -> None:
    reader = FakeReader()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    clips = studio._render_group(("one", "two", "three"))
    assert len(clips) == 3
    assert reader.batches == [["one", "two", "three"]]


def test_render_group_falls_back_for_a_reader_without_batching(data_root) -> None:
    """A reader double that only offers ``render`` is still read chunk by chunk."""

    class _NoBatch(FakeReader):
        render_many = None  # type: ignore[assignment]  # a reader with no batched path

    reader = _NoBatch()
    studio = VoiceStudioService(SOURCE_PIPER, reader=reader)
    clips = studio._render_group(("one", "two"))
    assert len(clips) == 2
    assert reader.spoken == ["one", "two"]


def test_identity_rename_and_logo_persist(data_root, tmp_path) -> None:
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    studio.set_name("My Reader")
    assert profiles.load_profile(SOURCE_PIPER).name == "My Reader"
    studio.set_logo(_png(tmp_path / "l.png"))
    assert studio.logo_path() is not None
    studio.clear_logo()
    assert studio.logo_path() is None


def test_describe_names_the_voice_and_the_mood(data_root) -> None:
    studio = VoiceStudioService(SOURCE_PIPER, reader=FakeReader())
    studio.set_mood("sad")
    line = studio.describe()
    assert "sad" in line.lower() or "Sad" in line


def test_an_unknown_engine_is_refused(data_root) -> None:
    with pytest.raises(ValueError):
        VoiceStudioService("not-an-engine")


def test_only_the_cloning_engine_offers_the_import_capability(data_root) -> None:
    """VoiceTut clones from a clip, so only its studio reports the capability."""
    assert VoiceStudioService(SOURCE_AUDIOCPP, reader=FakeReader()).can_import_voices() is True
    for source in (SOURCE_PIPER, SOURCE_QWEN, SOURCE_SYSTEM):
        assert VoiceStudioService(source, reader=FakeReader()).can_import_voices() is False
    # A non-cloning studio reports no imported voices even if some are registered.
    assert VoiceStudioService(SOURCE_PIPER, reader=FakeReader()).imported_voice_names() == ()


def test_the_studio_clones_and_removes_the_user_s_own_voice(data_root, monkeypatch, tmp_path) -> None:
    """The service is the one path the window and the CLI share for a new voice."""
    from core import audiocpp_speech as a

    monkeypatch.setattr(a, "voices_dir", lambda root=None: data_root / "audio.cpp" / "voices")
    monkeypatch.setattr(a, "imported_voices_path", lambda root=None: data_root / "audio.cpp" / "imported_voices.json")
    monkeypatch.setattr(a, "load_imported_voices", lambda root=None: _imported(data_root))
    monkeypatch.setattr(a, "_save_imported_voices", lambda voices, root=None: _save(data_root, voices))

    studio = VoiceStudioService(SOURCE_AUDIOCPP, reader=FakeReader())
    assert studio.imported_voice_names() == ()
    clip = AudioClip((1000, -1000, 2000, -2000), sample_rate=24_000)
    voice = studio.clone_voice(clip)
    assert voice.name == "My voice"
    assert studio.imported_voice_names() == ("My voice",)
    assert studio.remove_voice("My voice") is True
    assert studio.imported_voice_names() == ()
    # Removing a voice that is not there reports so rather than raising.
    assert studio.remove_voice("Nope") is False


def _imported(root: Path):
    """Read the imported-voice registry the test's fake folder holds."""
    import json

    from core.audiocpp_speech import ImportedVoice

    path = root / "audio.cpp" / "imported_voices.json"
    if not path.is_file():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        ImportedVoice.from_primitive(row) for row in data.get("voices", [])
    )


def _save(root: Path, voices) -> None:
    import json

    path = root / "audio.cpp" / "imported_voices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"voices": [v.to_primitive() for v in voices]}), encoding="utf-8"
    )
