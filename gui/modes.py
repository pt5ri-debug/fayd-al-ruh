"""The interface's one operating mode: **Fayd Al-Ruh Model**.

The window used to expose two operating modes -- **AI Mode** (conversation only)
and **Jarvics Mode** (conversation plus the agent's sandboxed tools) -- switched
from the header. That split is gone: the header now names the project's own
model, and this module is the single mode it stands for.

* **Fayd Al-Ruh Model** -- the conversational, knowledge-grounded assistant. Its
  answers come from the project's own knowledge (with citations), from the
  internet when the user turns that on, and from the project's own trained
  language model. No sandboxed tool runs.

Nothing here invents a backend capability. The mode decides *which of the
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
from datetime import UTC, datetime
from typing import Any

from core.best_effort import best_effort
from core.command_router import NaturalLanguageRouter, RoutedCommand, RouteKind

#: The exact label the header shows, and its one-line description. The mode is
#: the project's own model, so the label names it.
FAYD_MODEL_MODE = "Fayd Al-Ruh Model"


@dataclass(frozen=True)
class ModeSpec:
    """The operating mode: its exact label, a short blurb, and its powers."""

    key: str
    label: str
    description: str
    allows_tools: bool


MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        key="ai",
        label=FAYD_MODEL_MODE,
        description="Knowledge · web · trained model",
        allows_tools=False,
    ),
)

MODES_BY_KEY: dict[str, ModeSpec] = {mode.key: mode for mode in MODES}
#: The label -> key map, so a persisted label or a CLI ``--mode`` value resolves.
MODE_KEYS_BY_LABEL: dict[str, str] = {mode.label.casefold(): mode.key for mode in MODES}

DEFAULT_MODE_KEY = "ai"


def resolve_mode(value: str | None) -> ModeSpec:
    """Resolve the mode by key (``"ai"``) or label (``"Fayd Al-Ruh Model"``).

    Falls back to :data:`DEFAULT_MODE_KEY` for ``None`` or anything unknown, so a
    stale saved preference (including the retired ``"jarvics"`` key) can never
    stop the window opening.
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
    """A router that never proposes a tool -- the brain of the Fayd Al-Ruh Model.

    The Fayd Al-Ruh Model answers conversationally and must not execute anything,
    even when the user writes a sentence the Jarvis router would read as a command
    ("list the files in notes/"). This subclass keeps the real router's sandbox and
    API but always decides :attr:`~core.command_router.RouteKind.ANSWER`, so the
    message is answered and no tool is ever planned or run.

    The class is intentionally tiny: it reuses everything the real router does
    and overrides only the one decision that gives the mode its boundary.
    """

    def route(self, utterance: str) -> RoutedCommand:
        text = utterance.strip()
        if not text:
            raise ValueError("Cannot route an empty utterance.")
        return RoutedCommand(
            RouteKind.ANSWER,
            text,
            "Fayd Al-Ruh Model answers from knowledge; no sandboxed tool runs.",
        )


# --------------------------------------------------------------- persistence

#: Where the chosen mode is remembered, matching the GUI's other preferences.
_NAMESPACE = "gui"
_KEY = "operating_mode"


def load_mode_key(root: Any) -> str:
    """The saved operating-mode key, defaulting to the Fayd Al-Ruh Model; never raises.

    A retired key (the old ``"jarvics"``) resolves to the one mode, so an upgrade
    never leaves the window in a mode it no longer has.
    """
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(root).load(_NAMESPACE, _KEY)
    except Exception:  # a broken store must not stop the window opening
        return DEFAULT_MODE_KEY
    name = saved.get("mode") if isinstance(saved, dict) else None
    return resolve_mode(name).key


def save_mode_key(root: Any, key: str) -> None:
    """Persist the chosen operating mode; a store error is recorded, not fatal."""
    from storage.json_store import JsonStore

    if key not in MODES_BY_KEY:
        return
    with best_effort("save the operating mode"):
        JsonStore(root).save(_NAMESPACE, _KEY, {"mode": key})


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
    reference = now or datetime.now(UTC)
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
        created_at = created_at.replace(tzinfo=UTC)
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
    "DEFAULT_MODE_KEY",
    "FAYD_MODEL_MODE",
    "MODES",
    "MODES_BY_KEY",
    "MODE_KEYS_BY_LABEL",
    "AnswerOnlyRouter",
    "HistoryItem",
    "ModeSpec",
    "group_by_date",
    "load_mode_key",
    "resolve_mode",
    "save_mode_key",
    "search_items",
]
