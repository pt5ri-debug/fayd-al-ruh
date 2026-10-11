"""The two operating modes of the interface: **AI Mode** and **Jarvics Mode**.

The window used to expose the two *answering mouths* as "From sources" and "From
trained model", and left the sandboxed tools reachable only through the Actions
list. This module names the two ways the application actually works, so the
header can show one obvious switch and the sidebar can stay compact:

* **AI Mode** -- the conversational, knowledge-grounded assistant. Answers come
  from retrieval (with citations) or from the trained model; no tool ever runs.
* **Jarvics Mode** -- the same conversation *plus* the agent's sandboxed tools,
  so an explicit request ("list the files in notes/", "calculate 2 * (3 + 4)")
  can run under every existing guarantee.

Nothing here invents a backend capability. A mode only decides *which of the
project's real, existing code paths a message may take*, exactly the paths the
CLI already offers:

- :class:`~core.jarvis.JarvisMode` routes a message to a tool plan or an answer
  (``python -m cli jarvis``);
- :class:`ai.chat.ModelSwitcher` swaps the answering mouth (``python -m cli chat``);
- :class:`~core.command_router.NaturalLanguageRouter` only proposes a tool when
  the sandbox's allow-list permits it and the intent is unambiguous.

The module is deliberately Tk-free and dependency-free, so the mode logic and its
persistence are testable without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.command_router import NaturalLanguageRouter, RouteKind, RoutedCommand

#: The exact labels the task requires, and their one-line descriptions.
AI_MODE = "AI Mode"
JARVICS_MODE = "Jarvics Mode"


@dataclass(frozen=True)
class ModeSpec:
    """One operating mode: its exact label, a short blurb, and its powers."""

    key: str
    label: str
    description: str
    allows_tools: bool


MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        key="ai",
        label=AI_MODE,
        description="Conversational answers from knowledge and the trained model.",
        allows_tools=False,
    ),
    ModeSpec(
        key="jarvics",
        label=JARVICS_MODE,
        description="Answers plus sandboxed tools for file, code, and knowledge tasks.",
        allows_tools=True,
    ),
)

MODES_BY_KEY: dict[str, ModeSpec] = {mode.key: mode for mode in MODES}
#: The label -> key map, so a persisted label or a CLI ``--mode`` value resolves.
MODE_KEYS_BY_LABEL: dict[str, str] = {mode.label.casefold(): mode.key for mode in MODES}

DEFAULT_MODE_KEY = "ai"


def resolve_mode(value: str | None) -> ModeSpec:
    """Resolve a mode by key (``"ai"``) or label (``"AI Mode"``).

    Falls back to :data:`DEFAULT_MODE_KEY` for ``None`` or anything unknown, so a
    stale saved preference can never stop the window opening.
    """
    if value is None:
        return MODES_BY_KEY[DEFAULT_MODE_KEY]
    candidate = value.strip().casefold()
    if candidate in MODES_BY_KEY:
        return MODES_BY_KEY[candidate]
    if candidate in MODE_KEYS_BY_LABEL:
        return MODES_BY_KEY[MODE_KEYS_BY_LABEL[candidate]]
    return MODES_BY_KEY[DEFAULT_MODE_KEY]


class AnswerOnlyRouter(NaturalLanguageRouter):
    """A router that never proposes a tool -- the brain of **AI Mode**.

    AI Mode must not execute anything, even when the user writes a sentence the
    Jarvics router would read as a command ("list the files in notes/"). This
    subclass keeps the real router's sandbox and API but always decides
    :attr:`~core.command_router.RouteKind.ANSWER`, so the message is answered
    conversationally and no tool is ever planned or run.

    The class is intentionally tiny: it reuses everything the real router does
    and overrides only the one decision that gives AI Mode its boundary.
    """

    def route(self, utterance: str) -> RoutedCommand:
        text = utterance.strip()
        if not text:
            raise ValueError("Cannot route an empty utterance.")
        return RoutedCommand(
            RouteKind.ANSWER,
            text,
            "AI Mode answers from knowledge; tools are off. Switch to Jarvics Mode to run one.",
        )


# --------------------------------------------------------------- persistence

#: Where the chosen mode is remembered, matching the GUI's other preferences.
_NAMESPACE = "gui"
_KEY = "operating_mode"


def load_mode_key(root: Any) -> str:
    """The saved operating-mode key, defaulting to AI Mode; never raises."""
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(root).load(_NAMESPACE, _KEY)
    except Exception:  # a broken store must not stop the window opening
        return DEFAULT_MODE_KEY
    name = saved.get("mode") if isinstance(saved, dict) else None
    return resolve_mode(name).key


def save_mode_key(root: Any, key: str) -> None:
    """Persist the chosen operating mode; a store error is silently ignored."""
    from storage.json_store import JsonStore

    if key not in MODES_BY_KEY:
        return
    try:
        JsonStore(root).save(_NAMESPACE, _KEY, {"mode": key})
    except Exception:
        pass


# --------------------------------------------------------- history grouping


@dataclass(frozen=True)
class HistoryItem:
    """One saved conversation, ready to render in the sidebar's history."""

    session_id: str
    label: str
    created_at: datetime | None
    is_current: bool = False
    snippet: str = ""


def group_by_date(
    items: Sequence[HistoryItem],
    *,
    now: datetime | None = None,
) -> tuple[tuple[str, tuple[HistoryItem, ...]], ...]:
    """Group history newest-first into the buckets ChatGPT uses.

    Buckets are *Today*, *Yesterday*, *Previous 7 days*, *Previous 30 days*, and
    *Older*. An item with no timestamp lands in *Older* rather than being dropped,
    so a store that lost a date never hides a conversation.
    """
    reference = now or datetime.now(timezone.utc)
    buckets: dict[str, list[HistoryItem]] = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 days": [],
        "Previous 30 days": [],
        "Older": [],
    }
    order = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"]
    for item in items:
        buckets[_bucket_for(item.created_at, reference)].append(item)
    return tuple((name, tuple(buckets[name])) for name in order if buckets[name])


def _bucket_for(created_at: datetime | None, now: datetime) -> str:
    if created_at is None:
        return "Older"
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (now.date() - created_at.date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days <= 7:
        return "Previous 7 days"
    if days <= 30:
        return "Previous 30 days"
    return "Older"


def search_items(
    items: Sequence[HistoryItem], query: str
) -> tuple[HistoryItem, ...]:
    """Filter history by a case-insensitive match on the label or the snippet."""
    needle = query.strip().casefold()
    if not needle:
        return tuple(items)
    return tuple(
        item
        for item in items
        if needle in item.label.casefold() or needle in item.snippet.casefold()
    )


__all__ = [
    "AI_MODE",
    "AnswerOnlyRouter",
    "DEFAULT_MODE_KEY",
    "HistoryItem",
    "JARVICS_MODE",
    "MODE_KEYS_BY_LABEL",
    "MODES",
    "MODES_BY_KEY",
    "ModeSpec",
    "group_by_date",
    "load_mode_key",
    "resolve_mode",
    "save_mode_key",
    "search_items",
]
