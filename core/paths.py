"""Resolve where Fayd Al-Ruh reads bundled assets and writes its data.

Two modes matter:

- **Source checkout** (development): code and bundled assets stay in the
  repository, while learned data (artifacts, checkpoints, knowledge base,
  learned documents) is kept in a ``data/`` folder beside the code. That folder
  is git-ignored, so retraining never dirties the working tree and a program
  update can fast-forward without disturbing what the user has learned.
- **Frozen build** (portable): assets ship inside the bundle (PyInstaller
  ``_MEIPASS``) and data is written next to the executable, so the folder can be
  copied to a USB stick and keep its state.

``FAYD_DATA_DIR`` overrides the data location in either mode.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENV_DATA_DIR = "FAYD_DATA_DIR"


def is_frozen() -> bool:
    """True when running from a PyInstaller-style bundle rather than source."""
    return bool(getattr(sys, "frozen", False))


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundle_root() -> Path:
    """Read-only assets: the source tree, or the bundle's extraction directory."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return _source_root()


def executable_dir() -> Path:
    """Folder of the running program: the executable's folder, or the repo root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _source_root()


def data_dir() -> Path:
    """Writable folder for state, artifacts, and trained checkpoints.

    A frozen build is portable: data sits next to the executable. A source
    checkout keeps its learned data in a ``data/`` folder beside the code (which
    git ignores), so retraining never dirties the working tree and a program
    update can fast-forward without touching what the user has learned.
    ``FAYD_DATA_DIR`` overrides the location in either mode.
    """
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return executable_dir() / "data"
    return _source_root() / "data"


def artifacts_dir() -> Path:
    """Where ``python -m ml`` writes learned artifacts by default."""
    return data_dir() / "ml" / "artifacts"


def checkpoints_dir() -> Path:
    """Where ``python -m ml.llm`` writes the trained language-model checkpoint."""
    return data_dir() / "ml" / "llm" / "checkpoints"


def semantic_space_path() -> Path:
    """Where the trained distributional semantic space is cached.

    The space is learned from the project's own corpus (see
    :mod:`core.embeddings`) and is deterministic, so it is written once under the
    git-ignored data folder and reloaded on the next run rather than rebuilt.
    ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "ml" / "semantic_space.json"


def models_dir() -> Path:
    """Where optional learned models (e.g. a diffusion checkpoint) are cached.

    Kept under the writable data folder (git-ignored), never in the repository, so
    a downloaded model never dirties the working tree or bloats a program update.
    """
    return data_dir() / "models"


def ultron_models_dir() -> Path:
    """Where the Ultron general-language models (GGUF files) live.

    Ultron's model is *not* an in-project checkpoint: it is a general, advanced
    model the user drops into ``data/ultron`` (or downloads there with
    ``python -m cli ultron download --approve``). The folder is git-ignored, so a
    multi-gigabyte model never dirties the working tree or bloats a program
    update. ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "ultron"


def vision_models_dir() -> Path:
    """Where the Vision petroleum-engineering models (GGUF files) live.

    Vision is the project's *petroleum* local model (built on the OGAI Reasoner
    weights): like Ultron it is *not* an in-project checkpoint, but a GGUF file
    the user drops into ``data/vision`` (or downloads there with
    ``python -m cli vision download --approve``). The folder is git-ignored, so a
    multi-gigabyte model never dirties the working tree or bloats a program
    update. ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "vision"


def coder_models_dir() -> Path:
    """Where the Code Studio's in-process coding models (GGUF files) live.

    Like Ultron's models, these are *not* in-project checkpoints: they are GGUF
    files the user drops into ``data/coder`` (or downloads there with
    ``python -m cli studio download --approve``). Running the model in-process
    means no ``ollama serve`` daemon has to sit in the background holding RAM.
    The folder is git-ignored, so a multi-gigabyte model never dirties the
    working tree or bloats a program update.
    """
    return data_dir() / "coder"


def presentation_models_dir() -> Path:
    """Where the Presentation Studio's in-process authoring models live.

    The Studio drafts outlines and slide content with a GGUF file the user drops
    into ``data/presentation`` (or downloads there with
    ``python -m cli presentation model-download --approve``). Like Ultron's and
    the Code Studio's models these are *not* in-project checkpoints, so the
    folder is git-ignored and a multi-gigabyte model never dirties the working
    tree or bloats a program update. ``FAYD_DATA_DIR`` moves it with the rest of
    the data.
    """
    return data_dir() / "presentation"


def natasha_models_dir() -> Path:
    """Where Natasha's in-process GGUF models live.

    Natasha, the project's companion, speaks through a **GGUF file loaded into
    this process** with ``llama-cpp-python`` by default -- the same runtime
    Ultron, Vision, Banner, and the two Studios use -- so no ``ollama serve``
    daemon has to sit in the background holding RAM. The file lives under
    ``data/natasha`` (or is downloaded there with
    ``python -m cli natasha download --approve``); an existing Ollama setup is
    still honoured as a fallback. The folder is git-ignored, so a multi-gigabyte
    model never dirties the working tree or bloats a program update.
    ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "natasha"


def qwen_speech_models_dir() -> Path:
    """Where the Qwen3 speech models (ASR and TTS) live.

    Qwen3-ASR and Qwen3-TTS are *not* in-project checkpoints: they are
    ``transformers`` model folders the user drops into ``data/qwen-speech`` (or
    downloads there with ``python -m cli qwen-asr download --approve`` and
    ``python -m cli qwen-tts download --approve``). One folder holds both
    families, keyed by a registry, so a machine that installs one can install the
    other beside it. The folder is git-ignored, so a multi-gigabyte model never
    dirties the working tree or bloats a program update.
    """
    return data_dir() / "qwen-speech"


def piper_models_dir() -> Path:
    """Where the Piper text-to-speech voices live.

    Piper is a local neural TTS engine: each voice is an ONNX model the user drops
    into ``data/piper`` (or downloads there with ``python -m cli piper download
    --approve``). The folder is git-ignored, so a downloaded voice never dirties
    the working tree or bloats a program update.
    """
    return data_dir() / "piper"


def audiocpp_models_dir() -> Path:
    """Where the audio.cpp text-to-speech models live.

    audio.cpp is a local C++ (ggml) audio engine; each model is a GGUF file the
    user drops into ``data/audio.cpp`` (or downloads there with ``python -m cli
    audio download --approve``). The folder is git-ignored, so a multi-gigabyte
    model never dirties the working tree or bloats a program update.
    """
    return data_dir() / "audio.cpp"


def docs_models_dir() -> Path:
    """Where Doc's Studio's in-process vision-language models live.

    Doc's Studio reads documents -- including scanned ones -- with a local
    **Qwen3-VL** model: a GGUF file plus its ``mmproj`` projector, loaded
    straight into this process with ``llama-cpp-python``. Like Ultron's and the
    Presentation Studio's models these are *not* in-project checkpoints, so the
    folder is git-ignored and a multi-gigabyte model never dirties the working
    tree or bloats a program update. ``FAYD_DATA_DIR`` moves it with the rest of
    the data.
    """
    return data_dir() / "docs"


def docs_pages_dir() -> Path:
    """Where Doc's Studio keeps the page images it renders for a document.

    A scanned PDF page has no text layer, so the Studio rasterises it to a PNG
    and hands the *image* to the vision model. The rendered pages are a cache --
    reproducible from the source document at any time -- so they live under the
    git-ignored data folder rather than beside the document.
    """
    return docs_models_dir() / "pages"


def scene_studio_dir() -> Path:
    """Where the Scene Studio keeps its scenes, renders, and imported models.

    The Studio (the local DAZ-style 3D figure studio) writes everything it makes
    under this folder: saved scenes (``scenes/``), rendered PNGs (``renders/``),
    and any models the user imports (``library/``). It is git-ignored, so a saved
    scene or a multi-megabyte imported mesh never dirties the working tree or
    bloats a program update. ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "scene3d"


def scene_studio_scenes_dir() -> Path:
    """Where the Scene Studio saves its scenes."""
    return scene_studio_dir() / "scenes"


def scene_studio_renders_dir() -> Path:
    """Where the Scene Studio writes rendered images."""
    return scene_studio_dir() / "renders"


def scene_studio_library_dir() -> Path:
    """Where the Scene Studio keeps models the user imported or saved."""
    return scene_studio_dir() / "library"


def characters_dir() -> Path:
    """Where the per-mouth logos live (``data/characters``, git-ignored).

    Each answering mouth -- Jarvis, Ultron, Vision, Banner, Natasha, Hakeem, and
    Cloud -- has its own dedicated chat window (see :mod:`gui.model_chat`), and
    each window shows a logo the user may replace with any picture. The copies
    are kept here, beside the rest of the user's own data, so replacing a logo
    never dirties the working tree and a program update never overwrites it.
    ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "characters"


def character_learning_dir() -> Path:
    """Where each character's own learning data is written (``data/characters/learning``).

    The seven answering mouths (Jarvis, Ultron, Vision, Banner, Natasha, Hakeem,
    and Cloud) each keep a small, file-backed learning record here: the trait
    profile that shapes their replies, the durable facts they were told, and the
    exchanges they took part in. The folder is git-ignored, so a character's
    learning never dirties the working tree and a program update never overwrites
    it -- and the main Fayd Al-Ruh model trains on the same files, so what a
    character learns reaches the project's own model. ``FAYD_DATA_DIR`` moves it
    with the rest of the data. It is a subfolder of :func:`characters_dir` so a
    learning file can never collide with a character's logo image.
    """
    return characters_dir() / "learning"


def config_dir() -> Path:
    """Where the user's saved character configuration lives (``data/config``).

    The *shipped* defaults live in the repository's ``config/`` folder (read-only,
    tracked, so an update refreshes them); the configuration the user actually
    edited and applied lives here, under the git-ignored data folder, so a program
    update can fast-forward without overwriting a character's settings. The folder
    is git-ignored and ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "config"


def character_config_dir() -> Path:
    """Where each character's applied configuration is saved (``data/config/characters``)."""
    return config_dir() / "characters"


def bundled_config_dir() -> Path:
    """The shipped, read-only character configuration inside the code or bundle."""
    return bundle_root() / "config"


def learning_root_dir() -> Path:
    """The root of every independent learning store (``data/learning``).

    Inside it live the project model's own learning directory
    (``fayd_al_ruh/``) and one strictly isolated directory per character
    (``characters/<key>/``). It is git-ignored, so a character's learning never
    dirties the working tree and a program update never overwrites it.
    ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "learning"


def main_model_learning_dir() -> Path:
    """The **project model's** own independent learning directory.

    The main Fayd Al-Ruh model learns independently here -- from its own
    interactions, the language-model answers it produced, verified internet
    information, and (through the controlled sharing interface) the permitted
    knowledge of six characters. It is a *separate* directory from every
    character's, and every record it acquires from a character is written here as
    its **own** record, never into the source character's folder.
    """
    return learning_root_dir() / "fayd_al_ruh"


def character_learning_store_dir(key: str) -> Path:
    """One character's strictly isolated learning directory.

    ``data/learning/fayd_al_ruh/characters/<key>``. A
    :class:`~core.character_learning_store.CharacterLearningStore` is built for
    exactly one owner and resolves exactly this path, so a character can only ever
    read and write its own store -- the isolation is structural, not a flag.
    """
    return main_model_learning_dir() / "characters" / key


def learning_audit_dir() -> Path:
    """Where the append-only learning audit lives (``data/learning/fayd_al_ruh/audit``).

    Every learning operation and every *refused* cross-character access is written
    here, so the isolation rules are auditable rather than merely intended.
    """
    return main_model_learning_dir() / "audit"


def studio_dir() -> Path:
    """Where the Code Studio keeps its own state (git worktrees, run logs).

    The Studio edits the project through *git worktrees* under this folder: one
    branch per change, checked out beside the code so it can be tested in
    isolation and merged back with a button. The folder is git-ignored, so a
    half-finished change never dirties the working tree and a program update
    (which only fast-forwards tracked files) never touches it.
    """
    return data_dir() / "studio"


def studio_workspaces_dir() -> Path:
    """The parent folder of every Studio worktree (one directory per change)."""
    return studio_dir() / "workspaces"


def bundled_artifacts_dir() -> Path:
    """Read-only artifacts inside the code or bundle, used as a fallback."""
    return bundle_root() / "ml" / "artifacts"


def bundled_checkpoints_dir() -> Path:
    """Read-only language-model checkpoints inside the code or bundle."""
    return bundle_root() / "ml" / "llm" / "checkpoints"


def default_checkpoint() -> Path:
    """Prefer a user-retrained checkpoint, then fall back to the bundled one."""
    candidate = checkpoints_dir() / "fayd-llm.npz"
    if candidate.exists():
        return candidate
    return bundled_checkpoints_dir() / "fayd-llm.npz"


def ensure_data_layout() -> None:
    """One-time upgrade step: gather pre-existing source data under ``data/``.

    Before this, a source build wrote ML artifacts, the knowledge base, learned
    documents, and the learning cursor straight into the repository. Those still
    exist on an upgrading machine, so on first run we copy them into ``data/``
    once; git-updated copies of the tracked artifacts keep working, and nothing
    the user trained is lost. Best-effort and silent: any failure simply leaves
    the old locations in place.
    """
    if is_frozen() or os.environ.get(ENV_DATA_DIR):
        return
    root = _source_root()
    data = root / "data"
    if data.exists():
        return

    pairs = (
        (root / "ml" / "artifacts", data / "ml" / "artifacts"),
        (root / "ml" / "llm" / "checkpoints", data / "ml" / "llm" / "checkpoints"),
        (root / "knowledge_base", data / "knowledge_base"),
        (root / "documents", data / "documents"),
    )
    if not any(source.exists() for source, _target in pairs):
        return
    try:
        for source, target in pairs:
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
        cursor = root / "learned_cursor.json"
        if cursor.exists() and not (data / "learned_cursor.json").exists():
            data.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cursor, data / "learned_cursor.json")
    except OSError:
        pass


__all__ = [
    "ENV_DATA_DIR",
    "artifacts_dir",
    "audiocpp_models_dir",
    "bundle_root",
    "bundled_artifacts_dir",
    "bundled_checkpoints_dir",
    "bundled_config_dir",
    "character_config_dir",
    "character_learning_dir",
    "character_learning_store_dir",
    "characters_dir",
    "checkpoints_dir",
    "coder_models_dir",
    "config_dir",
    "data_dir",
    "default_checkpoint",
    "docs_models_dir",
    "docs_pages_dir",
    "ensure_data_layout",
    "executable_dir",
    "is_frozen",
    "learning_audit_dir",
    "learning_root_dir",
    "main_model_learning_dir",
    "models_dir",
    "natasha_models_dir",
    "piper_models_dir",
    "presentation_models_dir",
    "qwen_speech_models_dir",
    "studio_dir",
    "studio_workspaces_dir",
    "ultron_models_dir",
    "vision_models_dir",
]
