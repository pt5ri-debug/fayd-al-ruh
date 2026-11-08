"""The Home / Dashboard model (Phase 2).

The main window's first sidebar section is a **Home** dashboard: what is
answering, what the session is, the recent conversations, the knowledge and
learning activity, and the machine's own figures. This module is the Tk-free
*model* behind it, so the layout can be built and tested with no display and the
window only paints what it is handed.

Honest by construction: every value is read from a service the caller already
has (never invented), an absent figure stays an explicit ``"unknown"`` rather
than a fabricated zero, and an empty section is dropped instead of drawn blank.
The module imports no Tk and loads no model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: The quick actions the dashboard offers, by their :mod:`gui.actions` keys --
#: the everyday entries a user reaches for, so the dashboard is a real launch
#: pad rather than a status page. A key that no longer resolves is skipped.
QUICK_ACTION_KEYS: tuple[str, ...] = (
    "insight",
    "cognition",
    "training-center",
    "kb-manager",
    "models",
)

#: How many recent conversations the dashboard lists before it stops.
DEFAULT_RECENT_CHATS = 5

#: How many recent notifications the dashboard lists before it stops.
DEFAULT_RECENT_NOTIFICATIONS = 5


@dataclass(frozen=True)
class DashboardRow:
    """One labelled line of a dashboard section."""

    label: str
    value: str
    detail: str = ""


@dataclass(frozen=True)
class DashboardSection:
    """A titled group of dashboard rows (drawn like every other sidebar section)."""

    key: str
    title: str
    icon: str
    rows: tuple[DashboardRow, ...]


@dataclass(frozen=True)
class QuickAction:
    """A dashboard shortcut that runs an existing sidebar action by its key."""

    key: str
    label: str


@dataclass(frozen=True)
class Dashboard:
    """The whole dashboard: the sections to draw and the shortcuts to offer."""

    sections: tuple[DashboardSection, ...]
    quick_actions: tuple[QuickAction, ...]


def _size(value: int | None) -> str:
    """A readable byte figure, or ``"unknown"`` when the platform cannot say."""
    if value is None:
        return "unknown"
    step = 1024.0
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < step:
            return f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} PB"


def _safe(getter: Callable[[], Any], default: Any) -> Any:
    """Read a value defensively: a missing attribute is the default, never a raise."""
    try:
        return getter()
    except Exception:
        return default


def knowledge_base_summary(root: Any = None) -> str:
    """A one-line figure for the bulk knowledge base, or ``""`` when there is none.

    Read through the project's own store (never a second scan of the folder), so
    the dashboard and ``cli knowledge stats`` can never disagree. A missing store,
    or a platform that cannot read it, is an honest empty string -- no section is
    drawn rather than a fabricated zero.
    """
    try:
        from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir

        store_dir = default_knowledge_base_dir() if root is None else root
        store = KnowledgeBaseStore(store_dir)
        if not store.shard_paths():
            return ""
        info = store.stats()
        chunks = int(info.get("chunks") or 0)
        size = _size(int(info.get("size_bytes") or 0))
        return f"{chunks:,} chunks · {size}"
    except Exception:
        return ""


def _session_section(
    *,
    mouth_label: str,
    model_description: str,
    session: str,
    mode_label: str,
) -> DashboardSection:
    rows = [
        DashboardRow("Model", mouth_label or "unknown", model_description),
        DashboardRow("Session", session or "unsaved"),
        DashboardRow("Mode", mode_label or "unknown"),
    ]
    return DashboardSection("home:session", "Home", "🏠", tuple(rows))


def _when(item: Any) -> str:
    """A short date for a saved-conversation row, or ``""`` when it has none."""
    stamp = _attr(item, "created_at", None)
    if stamp is None:
        return ""
    try:
        return stamp.strftime("%Y-%m-%d")
    except Exception:
        return str(stamp)


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``obj.name`` defensively: a missing attribute is the default."""
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _recent_section(items: Sequence[Any]) -> DashboardSection | None:
    rows: list[DashboardRow] = []
    for item in items:
        label = _attr(item, "label", "")
        if not label:
            label = _attr(item, "session_id", "")
        if not label:
            continue
        when = _when(item)
        rows.append(DashboardRow(str(label), when or "saved"))
    if not rows:
        return None
    return DashboardSection("home:recent", "Recent conversations", "🗂", tuple(rows))


def _knowledge_section(knowledge: str, learning: str) -> DashboardSection | None:
    rows: list[DashboardRow] = []
    if knowledge:
        rows.append(DashboardRow("Knowledge", knowledge))
    if learning:
        rows.append(DashboardRow("Learning", learning))
    if not rows:
        return None
    return DashboardSection("home:knowledge", "Knowledge & learning", "📚", tuple(rows))


def _notifications_section(feed: Any) -> DashboardSection | None:
    """The most recent background results, or ``None`` when there are none.

    Read from the notification feed (a :class:`gui.notifications.NotificationFeed`
    or any object with ``recent``), so the dashboard and the window's own feed can
    never disagree. An empty feed draws no section rather than a blank one.
    """
    if feed is None:
        return None
    try:
        items = tuple(feed.recent(DEFAULT_RECENT_NOTIFICATIONS))
    except Exception:
        return None
    rows: list[DashboardRow] = []
    for item in items:
        level = _attr(item, "level", "info") or "info"
        title = _attr(item, "title", "") or ""
        if not title:
            continue
        rows.append(DashboardRow(f"[{level}] {title}", _attr(item, "detail", "") or ""))
    if not rows:
        return None
    return DashboardSection("home:notifications", "Notifications", "🔔", tuple(rows))


def _resources_section(resources: Any) -> DashboardSection | None:
    if resources is None:
        return None
    system = _attr(resources, "system", "") or "unknown"
    machine = _attr(resources, "machine", "") or "unknown"
    cores = _attr(resources, "cpu_count", None)
    total = _attr(resources, "total_memory_bytes", None)
    available = _attr(resources, "available_memory_bytes", None)
    disk = _attr(resources, "disk_free_bytes", None)
    disk_path = _attr(resources, "disk_path", "") or ""
    rows = [
        DashboardRow("System", f"{system} ({machine})"),
        DashboardRow("CPU cores", str(cores or "unknown")),
        DashboardRow("Memory", _size(total), f"{_size(available)} available"),
        DashboardRow("Disk free", _size(disk), str(disk_path)),
    ]
    return DashboardSection("home:resources", "This machine", "🖥", tuple(rows))


def build_dashboard(
    *,
    mouth_label: str = "",
    model_description: str = "",
    session: str = "",
    mode_label: str = "",
    recent_chats: Iterable[Any] = (),
    knowledge_summary: str = "",
    learning_summary: str = "",
    notifications: Any = None,
    resources: Any = None,
    quick_actions: Sequence[QuickAction] = (),
) -> Dashboard:
    """Assemble the dashboard from values the caller already read.

    Every argument is optional: a caller that has no resources simply gets no
    *This machine* section, so a headless build is a smaller dashboard, never an
    error. The sections keep a fixed order (session, recent, knowledge,
    notifications, resources) so two builds read the same way.
    """
    sections: list[DashboardSection] = [
        _session_section(
            mouth_label=mouth_label,
            model_description=model_description,
            session=session,
            mode_label=mode_label,
        )
    ]
    for section in (
        _recent_section(tuple(recent_chats)),
        _knowledge_section(knowledge_summary, learning_summary),
        _notifications_section(notifications),
        _resources_section(resources),
    ):
        if section is not None:
            sections.append(section)
    return Dashboard(sections=tuple(sections), quick_actions=tuple(quick_actions))


def dashboard_for(
    service: Any,
    *,
    mouth_label: str = "",
    recent_chats: Sequence[Any] = (),
    knowledge_summary: str = "",
    learning_summary: str = "",
    notifications: Any = None,
    resources: Any = None,
    quick_actions: Sequence[QuickAction] = (),
) -> Dashboard:
    """Build the dashboard straight from a chat service, reading defensively.

    The service is duck-typed (the real :class:`gui.service.ChatService` or a
    test stub): a missing member is an honest ``"unknown"``, never a crash. The
    model description is the service's own ``describe_mouth_model`` for the mouth
    currently answering, so the dashboard never names a model the session is not
    actually using.
    """
    mouth = _attr(service, "source", "") or ""
    description = ""
    if mouth:
        describe = getattr(service, "describe_mouth_model", None)
        if callable(describe):
            description = _safe(lambda: describe(mouth), "")
    mode = _attr(service, "operating_mode", None)
    return build_dashboard(
        mouth_label=mouth_label,
        model_description=description,
        session=str(_attr(service, "session", "") or ""),
        mode_label=str(_attr(mode, "label", "") or ""),
        recent_chats=recent_chats,
        knowledge_summary=knowledge_summary,
        learning_summary=learning_summary,
        notifications=notifications,
        resources=resources,
        quick_actions=quick_actions,
    )


__all__ = [
    "DEFAULT_RECENT_CHATS",
    "DEFAULT_RECENT_NOTIFICATIONS",
    "QUICK_ACTION_KEYS",
    "Dashboard",
    "DashboardRow",
    "DashboardSection",
    "QuickAction",
    "build_dashboard",
    "dashboard_for",
    "knowledge_base_summary",
]
