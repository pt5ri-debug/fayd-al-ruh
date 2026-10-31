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
    run_characters,
    run_cloud_chat,
    run_code_studio,
    run_cognition,
    run_curriculum,
    run_demo,
    run_deps,
    run_docs,
    run_evaluation,
    run_hakeem_chat,
    run_image,
    run_import_center,
    run_ingest,
    run_insight,
    run_internet_learner,
    run_knowledge,
    run_learn,
    run_llm,
    run_ml,
    run_models,
    run_natasha,
    run_natasha_chat,
    run_piper_speech,
    run_presentation,
    run_qwen_speech,
    run_release,
    run_scene3d,
    run_selfcheck,
    run_shared_models,
    run_ultron_chat,
    run_update,
    run_vision_chat,
    run_web,
)

Command = Callable[[Sequence[str]], int]

#: Concise sidebar labels for the visible actions, keyed by action key. The full
#: ``GuiAction.label`` stays the honest, complete name (the CLI reads it); this is
#: only what the narrow sidebar draws, so a grouped menu reads cleanly instead of
#: showing a sentence per row. An action with no entry here keeps its full label.
SIDEBAR_LABELS: dict[str, str] = {
    "insight": "✨   What the AI can do",
    "cognition": "Autonomous mind",
    "cognition-characters": "Evaluate the characters",
    "learn": "Learn from feedback",
    "internet-learner": "Learn from the internet",
    "web": "Ask the web",
    "knowledge-build": "Build the knowledge base",
    "knowledge-stats": "Knowledge base status",
    "ingest-connectors": "Ingest sources…",
    "ingest-stats": "Ingestion status",
    "natasha-traits": "Traits",
    "natasha-mood": "Mood",
    "natasha-initiative": "Initiative",
    "natasha-relationship-memory": "Relationship memory",
    "models": "Model Manager",
    "shared-models": "Shared models — status",
    "shared-models-download": "Shared models — download",
    "shared-models-migrate": "Shared models — move from Natasha",
    "characters": "Characters",
    "characters-shareable": "Who may share knowledge",
    "qwen-speech": "Qwen3 — status",
    "qwen-speech-asr": "Qwen3 — download ASR",
    "qwen-speech-tts": "Qwen3 — download TTS",
    "piper": "Piper — status",
    "piper-download": "Piper — download a voice",
    "audio-speech": "VoiceTut — status",
    "audio-speech-voices": "VoiceTut — voices",
    "audio-speech-download": "VoiceTut — download the model",
    "llm": "Train the language model",
    "llm-sessions": "Train on my chats",
    "llm-characters": "Train on the characters' learning",
    "ml": "Train the ML pipeline",
    "release": "Train a release",
    "release-sessions": "Train a release (with my chats)",
    "curriculum": "Curriculum",
    "demo": "Cognitive demo",
    "deps": "Install optional support…",
    "evaluation": "Run the evaluation",
    "selfcheck": "Self-check",
    "update": "Search for updates",
}


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

    @property
    def sidebar_label(self) -> str:
        """The short label the sidebar draws, else the full ``label``.

        The full label is a sentence (it has to explain the action on the CLI);
        in a 288-pixel column under a group header, that reads as a wall of text.
        A concise label is used where one is defined, so the sidebar stays tidy
        while ``label`` remains the honest, complete name for the CLI.
        """
        return SIDEBAR_LABELS.get(self.key, self.label)


@dataclass(frozen=True)
class SessionItem:
    """One saved conversation, ready to render as a sidebar button."""

    key: str
    label: str
    session_id: str
    is_current: bool = False


@dataclass(frozen=True)
class ActionGroup:
    """A named, collapsible section of the sidebar.

    The sidebar is drawn as a short list of these (a professional, grouped menu
    like the ones graphics and design programs use) instead of one long flat
    column. ``title`` is what the user reads, ``icon`` is a single glyph for the
    header, and ``keys`` are the :class:`GuiAction` keys the section holds, in
    the order they are drawn. A key that is not a *visible* action is skipped, so
    the groups can never resurrect a hidden row.
    """

    key: str
    title: str
    icon: str
    keys: tuple[str, ...]


#: The sidebar's sections, in the order a user meets them: the things used every
#: day first (chats, the reading voice, the characters), then the knowledge and
#: speech engines, then the power-user tools (training, system, preferences). The
#: grouping is the fix for the flat, repetitive column the window used to draw --
#: every action appears in exactly one section, and the sections mirror the CLI
#: menu's own neighbourhoods so the two never disagree about what belongs where.
ACTION_GROUPS: tuple[ActionGroup, ...] = (
    ActionGroup(
        "assistant",
        "Assistant",
        "✨",
        (
            "insight",
            "cognition",
        ),
    ),
    ActionGroup(
        "characters",
        "Characters",
        "🧑",
        (
            "characters",
            "characters-shareable",
        ),
    ),
    ActionGroup(
        "models",
        "Language models",
        "🧩",
        (
            "models",
            "shared-models",
            "shared-models-download",
            "shared-models-migrate",
        ),
    ),
    ActionGroup(
        "knowledge",
        "Knowledge & Web",
        "📚",
        (
            "learn",
            "internet-learner",
            "web",
            "knowledge-build",
            "knowledge-stats",
            "ingest-connectors",
            "ingest-stats",
            "import-center",
        ),
    ),
    ActionGroup(
        "companion",
        "Companion",
        "💬",
        (
            "natasha-traits",
            "natasha-mood",
            "natasha-initiative",
            "natasha-relationship-memory",
        ),
    ),
    ActionGroup(
        "speech",
        "Speech Models",
        "🔊",
        (
            "qwen-speech",
            "qwen-speech-asr",
            "qwen-speech-tts",
            "piper",
            "piper-download",
            "audio-speech",
            "audio-speech-voices",
            "audio-speech-download",
        ),
    ),
    ActionGroup(
        "training",
        "Training & Learning",
        "🧠",
        (
            "llm",
            "llm-sessions",
            "llm-characters",
            "ml",
            "release",
            "release-sessions",
            "curriculum",
            "demo",
        ),
    ),
    ActionGroup(
        "system",
        "System",
        "⚙",
        (
            "deps",
            "evaluation",
            "selfcheck",
            "update",
        ),
    ),
)


def action_group_by_key(key: str) -> ActionGroup:
    """Look up one group by its stable key. Raises ``KeyError`` if unknown."""
    for group in ACTION_GROUPS:
        if group.key == key:
            return group
    raise KeyError(key)


def grouped_visible_actions() -> tuple[tuple[ActionGroup, tuple[GuiAction, ...]], ...]:
    """The visible actions grouped into sections, each action appearing once.

    An action that no group names is appended to the last group, so a newly
    added action can never silently vanish from the sidebar; the order inside a
    group is the group's own ``keys`` order.
    """
    visible = {action.key: action for action in visible_actions()}
    seen: set[str] = set()
    grouped: list[tuple[ActionGroup, tuple[GuiAction, ...]]] = []
    for group in ACTION_GROUPS:
        members = tuple(
            visible[key] for key in group.keys if key in visible and key not in seen
        )
        seen.update(member.key for member in members)
        if members:
            grouped.append((group, members))
    leftover = tuple(action for key, action in visible.items() if key not in seen)
    if leftover:
        if grouped:
            last_group, last_members = grouped[-1]
            grouped[-1] = (last_group, (*last_members, *leftover))
        else:
            grouped.append((ACTION_GROUPS[-1], leftover))
    return tuple(grouped)


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
    GuiAction(
        "natasha-traits",
        "Natasha: her standing character traits (the realism settings)",
        run_natasha,
        ("traits",),
    ),
    GuiAction(
        "natasha-mood",
        "Natasha: her emotional state (named mood for the moment)",
        run_natasha,
        ("mood",),
    ),
    GuiAction(
        "natasha-initiative",
        "Natasha: the initiatives she may take (never repeated)",
        run_natasha,
        ("initiative",),
    ),
    GuiAction(
        "natasha-relationship-memory",
        "Natasha: the structured relationship memory (facts, story, preferences)",
        run_natasha,
        ("relationship-memory",),
    ),
    GuiAction(
        "hakeem",
        "Answer from Hakeem (scholar of religion and history)",
        run_hakeem_chat,
        (),
        hidden=True,
    ),
    GuiAction(
        "models",
        "Model Manager: discover, register, inspect, and test local models",
        run_models,
        ("status",),
    ),
    GuiAction(
        "shared-models",
        "Shared models: one public folder every character may run",
        run_shared_models,
        ("status",),
    ),
    GuiAction(
        "shared-models-download",
        "Shared models: download the small general models (~1.7 GB)",
        run_shared_models,
        ("download-all", "--approve"),
    ),
    GuiAction(
        "shared-models-migrate",
        "Shared models: move a legacy model from Natasha's folder",
        run_shared_models,
        ("migrate",),
    ),
    GuiAction(
        "characters",
        "Characters: manage the seven characters and the project's own model",
        run_characters,
        ("list",),
    ),
    GuiAction(
        "characters-shareable",
        "Characters: which characters may share knowledge with the main model",
        run_characters,
        ("shareable",),
        hidden=True,
    ),
    GuiAction("learn", "Continuous learning and regression gate", run_learn, ()),
    GuiAction(
        "cognition",
        "Autonomous mind: proactive thinking, learning, and gradual training",
        run_cognition,
        ("status",),
    ),
    GuiAction(
        "cognition-characters",
        "Autonomous mind: evaluate the eight characters",
        run_cognition,
        ("characters",),
        hidden=True,
    ),
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
    GuiAction(
        "docs-studio",
        "Doc's Studio: read any document (incl. scanned)",
        run_docs,
        ("status",),
        hidden=True,
    ),
    GuiAction(
        "scene-studio",
        "Scene Studio: build, pose, and render a 3D figure",
        run_scene3d,
        ("status",),
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
    GuiAction(
        "llm-characters",
        "Train the language model on the characters' own learning",
        run_llm,
        ("--llm-characters", "--steps", "4000"),
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
    "ACTION_GROUPS",
    "ActionGroup",
    "Command",
    "GuiAction",
    "SessionItem",
    "action_by_key",
    "action_group_by_key",
    "grouped_visible_actions",
    "saved_session_items",
    "visible_actions",
]
