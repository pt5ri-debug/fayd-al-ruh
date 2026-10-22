"""Sidebar actions for the GUI, reusing the CLI menu's own functions.

Every action maps to one numbered CLI menu entry, so the GUI and the CLI share
behaviour instead of duplicating it. ``ACTIONS`` mirrors numbered items 5..11 --
the ones the GUI did not already offer through its own controls -- and carries
the same labels and default arguments the menu uses.

Nothing is listed twice: the four items above (the GUI window, the voice turn,
the tool agent, and the two chat modes) are reached from the window's own
controls and the composer, so they are deliberately absent here. The training
entries that used to sit in this sidebar are kept as well (they are hidden from
the numbered menu but remain harmless to expose).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cli.menu import (
    run_audio_speech,
    run_banner_chat,
    run_cloud_chat,
    run_code_studio,
    run_curriculum,
    run_demo,
    run_deps,
    run_evaluation,
    run_import_center,
    run_image,
    run_ingest,
    run_insight,
    run_internet_learner,
    run_knowledge,
    run_learn,
    run_llm,
    run_ml,
    run_natasha_chat,
    run_piper_speech,
    run_presentation,
    run_qwen_speech,
    run_release,
    run_selfcheck,
    run_ultron_chat,
    run_update,
    run_vision_chat,
    run_web,
)

Command = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class GuiAction:
    """A named, ready-to-run command shown as a sidebar button."""

    key: str
    label: str
    command: Command
    argv: tuple[str, ...] = ()
    #: When true, the composer's text is appended to ``argv`` (e.g. a web question).
    needs_input: bool = False
    #: When true, the action is kept in ``ACTIONS`` (so ``action_by_key`` and the
    #: CLI still resolve it) but is not drawn as a sidebar button. The mouth
    #: switches use this: the header's model selector already offers them.
    hidden: bool = False


@dataclass(frozen=True)
class SessionItem:
    """One saved conversation, ready to render as a sidebar button."""

    key: str
    label: str
    session_id: str
    is_current: bool = False


# Numbered CLI menu items 5..11, plus the training entries the sidebar already had.
# The first four entries are the "Answer from ..." mouth switches: they stay in the
# tuple (so ``action_by_key`` and the CLI keep resolving them) but are ``hidden``,
# because the header's model selector is the one visible way to switch mouths.
# Entries 11..14 (the Import Center, Code Studio, Presentation Studio, and image
# generation) are hidden too, because the composer's ＋ menu already opens them.
ACTIONS: tuple[GuiAction, ...] = (
    GuiAction(
        "ultron",
        "Answer from Ultron (general local model)",
        run_ultron_chat,
        (),
        hidden=True,
    ),
    GuiAction(
        "vision",
        "Answer from Vision (petroleum local model)",
        run_vision_chat,
        (),
        hidden=True,
    ),
    GuiAction(
        "banner",
        "Answer from Banner (presentation local model)",
        run_banner_chat,
        (),
        hidden=True,
    ),
    GuiAction(
        "cloud",
        "Answer from Cloud (remote model: ChatGPT, Claude, Gemini)",
        run_cloud_chat,
        (),
        hidden=True,
    ),
    GuiAction(
        "natasha",
        "Answer from Natasha (companion: memory, tone, relationship)",
        run_natasha_chat,
        (),
        hidden=True,
    ),
    GuiAction("learn", "Continuous learning and regression gate", run_learn, ()),
    GuiAction(
        "insight",
        "Ask the AI: what can it do and what has it learned",
        run_insight,
        ("--report",),
    ),
    GuiAction(
        "internet-learner",
        "Continuous Internet Learner",
        run_internet_learner,
        ("--approve", "--write", "--max-topics", "5", "--per-topic", "3"),
    ),
    GuiAction(
        "web",
        "Ask the web and verify the answer",
        run_web,
        ("--approve",),
        needs_input=True,
    ),
    GuiAction(
        "knowledge-build",
        "Build the bulk knowledge base (100 MB)",
        run_knowledge,
        ("build", "--target-mb", "100"),
    ),
    GuiAction("knowledge-stats", "Show the bulk knowledge base status", run_knowledge, ("stats",)),
    GuiAction(
        "import-center",
        "Import files and folders (Knowledge Import Center)",
        run_import_center,
        ("interactive",),
        hidden=True,
    ),
    GuiAction(
        "code-studio",
        "Code Studio: edit and merge the project's own code",
        run_code_studio,
        ("status",),
        hidden=True,
    ),
    GuiAction(
        "presentation",
        "Presentation Studio: build a PowerPoint deck",
        run_presentation,
        ("list",),
        hidden=True,
    ),
    GuiAction("image", "Generate an image from text", run_image, ("backends",), hidden=True),
    GuiAction(
        "qwen-speech",
        "Qwen3 speech: status of the local ASR + TTS models",
        run_qwen_speech,
        ("qwen-asr", "status"),
    ),
    GuiAction(
        "qwen-speech-asr",
        "Qwen3 speech: download the local ASR model (~3.4 GB)",
        run_qwen_speech,
        ("qwen-asr", "download", "--approve"),
    ),
    GuiAction(
        "qwen-speech-tts",
        "Qwen3 speech: download the local TTS model (~3.4 GB)",
        run_qwen_speech,
        ("qwen-tts", "download", "--approve"),
    ),
    GuiAction(
        "piper",
        "Piper: status of the natural local voices (Arabic + English)",
        run_piper_speech,
        ("status",),
    ),
    GuiAction(
        "piper-download",
        "Piper: download the natural local voice (~63 MB)",
        run_piper_speech,
        ("download", "--approve"),
    ),
    GuiAction(
        "audio-speech",
        "audio.cpp: status of the natural Arabic voice (VoiceTut-TTS)",
        run_audio_speech,
        ("status",),
    ),
    GuiAction(
        "audio-speech-voices",
        "audio.cpp: the local model's own voices",
        run_audio_speech,
        ("voices",),
    ),
    GuiAction(
        "audio-speech-download",
        "audio.cpp: download the natural Arabic voice model (~1.3 GB)",
        run_audio_speech,
        ("download", "--approve"),
    ),
    GuiAction(
        "deps",
        "Install optional PDF/OCR/image support",
        run_deps,
        ("--install", "pdf", "--install", "ocr"),
    ),
    GuiAction(
        "ingest-connectors",
        "Ingest sources: files, SQL, authorized web",
        run_ingest,
        ("list-connectors",),
    ),
    GuiAction(
        "ingest-stats",
        "Show the ingestion status (store + ledger)",
        run_ingest,
        ("stats",),
    ),
    GuiAction(
        "release",
        "Train everything for a release",
        run_release,
        ("--resume", "--episodes", "120", "--llm-steps", "4000", "--llm-shards", "4"),
    ),
    GuiAction(
        "release-sessions",
        "Train everything for a release (including the local mouths' chats)",
        run_release,
        ("--llm-sessions",),
    ),
    GuiAction("evaluation", "Run the evaluation suite", run_evaluation, ()),
    GuiAction("selfcheck", "Self-check the install", run_selfcheck, ()),
    GuiAction(
        "update",
        "Search for program updates (GitHub)",
        run_update,
        ("--apply",),
    ),
    GuiAction(
        "llm",
        "Train the language model",
        run_llm,
        ("--resume", "--steps", "200", "--kb-max-chunks", "2000", "--llm-shards", "4"),
    ),
    GuiAction(
        "llm-sessions",
        "Train the language model on the local mouths' own chats",
        run_llm,
        ("--llm-sessions", "--steps", "4000"),
    ),
    GuiAction("ml", "Train the ML pipeline", run_ml, ("--resume", "--episodes", "60")),
    GuiAction("curriculum", "Show the petroleum-engineering curriculum", run_curriculum, ()),
    GuiAction("demo", "Run the local cognitive demo", run_demo, ()),
)


def action_by_key(key: str) -> GuiAction:
    """Look up one action by its stable key. Raises ``KeyError`` if unknown."""
    for action in ACTIONS:
        if action.key == key:
            return action
    raise KeyError(key)


def visible_actions() -> tuple[GuiAction, ...]:
    """The actions the sidebar actually draws (``hidden`` ones left out)."""
    return tuple(action for action in ACTIONS if not action.hidden)


def saved_session_items(
    session_ids: Sequence[str], current: str | None = None
) -> tuple[SessionItem, ...]:
    """Build the sidebar's saved-chat buttons, marking the current session."""
    items = []
    for session_id in session_ids:
        is_current = session_id == current
        label = f"{session_id}   ✓ current" if is_current else session_id
        items.append(
            SessionItem(
                key=f"session:{session_id}",
                label=label,
                session_id=session_id,
                is_current=is_current,
            )
        )
    return tuple(items)


__all__ = [
    "ACTIONS",
    "Command",
    "GuiAction",
    "SessionItem",
    "action_by_key",
    "saved_session_items",
    "visible_actions",
]
