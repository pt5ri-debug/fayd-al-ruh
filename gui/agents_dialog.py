"""The Personalities (agents) window: the seven characters as independent agents.

The whole multi-agent subsystem (:mod:`core.agents`) was reachable only from the
CLI (``python -m cli agents``), so a user could not *see* the seven personalities
as independent agents -- what each may and may not do, which one would answer a
question, or what each has actually done. This window closes that gap.

It drives the Tk-free :class:`gui.agents.AgentsService`, so the window and the CLI
read **one** system -- no logic is reimplemented in either front end. It is
built the way every other window in the project is built: the shared
:class:`gui.components.DialogHeader` title band, the shared
:class:`gui.sidebar.SidebarColumn` on the left, and a worker thread whose result
is handed back through a queue the UI thread drains (Tk is not thread-safe).

Everything here is **read-only plus routing**: it shows the registry, the
capability ceiling, the routing decision, and the measured activity, and it never
runs a task, changes a permission, or widens what a personality may do. Routing
is a *suggestion* the router produced, not an action.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Any

from core.agents.registry import PersonalityProfile

from gui.agents import AgentsService, RouteReport
from gui.dialog_base import ThemedDialog
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import fit_geometry

TITLE = "Personalities (agents)"
SUBTITLE = (
    "The seven characters as independent agents in one shared system: what each "
    "may do, whose records it may read, which one would answer a question, and "
    "what each has actually done."
)
HONESTY_LINE = (
    "Read-only plus routing. Nothing here runs a task, changes a permission, or "
    "widens what a personality may do — a capability outside a character's ceiling "
    "is shown as denied, and a routing decision is the router's own suggestion."
)

DEFAULT_WIDTH = 1180
DEFAULT_HEIGHT = 720
MIN_WIDTH = 940
MIN_HEIGHT = 600
SIDEBAR_WIDTH = 320

#: The transcript column is capped, like the chat window's, so it reads centred.
_COLUMN_MAX = 880


class AgentsDialog(ThemedDialog):
    """The one window for the multi-agent (personalities) layer."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: AgentsService,
        *,
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._list_type = resolve(
            parent, ui_family=font, mono_family=mono, overrides=list_overrides
        )
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._pump_after: str | None = None
        self._cards: list[tk.Text] = []
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._selected_owner = ""
        self._rows: list[Any] = []

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry(fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT))
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    def _after_build(self) -> None:
        """Run the window's post-build work (first build, and after a re-theme)."""
        self._reload()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        body.columnconfigure(1, weight=1)
        self._body = body
        self._build_sidebar(body)
        self._build_main(body)

    def _build_header(self) -> None:
        """The shared title band (the one :class:`gui.components.DialogHeader`)."""
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close,
        )
        self._window_controls = header.controls

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        """A themed flat button (the shared :func:`gui.components.button`)."""
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant="secondary", small=True, anchor="w",
        )

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, width=SIDEBAR_WIDTH,
            scope="agents",
        )
        self._sidebar_column = column
        theme = self.theme

        people = column.add("people", "Personalities", icon="🧑", open=True)
        self._people_list = tk.Listbox(
            people, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._list_type.font("xs"), height=9, activestyle="none",
            exportselection=False,
        )
        self._people_list.pack(fill="x", padx=12, pady=(4, 6))
        self._people_list.bind("<<ListboxSelect>>", self._on_pick_personality)

        routing = column.add("routing", "Route a question", icon="🧭", open=True)
        tk.Label(
            routing, text="Which personality should answer?", bg=theme.surface,
            fg=theme.text_secondary, font=self._type.font("2xs"), anchor="w",
            justify="left", wraplength=280,
        ).pack(fill="x", padx=14, pady=(4, 2))
        self._route_var = tk.StringVar(value="how do I measure porosity?")
        route_entry = tk.Entry(
            routing, textvariable=self._route_var, bg=theme.base, fg=theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=theme.border_subtle,
            highlightcolor=theme.primary, insertbackground=theme.foreground,
            font=self._type.font("xs"),
        )
        route_entry.pack(fill="x", padx=14, pady=(0, 4), ipady=4)
        route_entry.bind("<Return>", lambda _e: self._on_route())
        self._route_button = self._button(routing, "Route it", self._on_route)
        self._route_button.pack(fill="x", padx=12, pady=(0, 8))

        isolation = column.add("isolation", "Isolation policy", icon="🔒", open=False)
        self._isolation_label = tk.Label(
            isolation, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=280,
        )
        self._isolation_label.pack(fill="x", padx=14, pady=(4, 8))

        about = column.add("about", "About this layer", icon="ℹ", open=False)
        tk.Label(
            about, text=HONESTY_LINE, bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=280,
        ).pack(fill="x", padx=14, pady=(4, 8))

    # -- main -------------------------------------------------------------

    def _build_main(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        head = tk.Frame(col, bg=theme.base)
        head.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            head, text="THE SEVEN PERSONALITIES", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._status_label = tk.Label(
            head, text="", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        )
        self._status_label.pack(side="right")

        # The profile of the chosen personality, and its capability ceiling.
        detail_card = tk.Frame(col, bg=theme.surface)
        detail_card.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        inner = tk.Frame(detail_card, bg=theme.surface)
        inner.pack(fill="both", expand=True, padx=16, pady=14)
        self._profile_label = tk.Label(
            inner, text="", bg=theme.surface, fg=theme.foreground,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=720,
        )
        self._profile_label.pack(fill="x", anchor="w")
        tk.Label(
            inner, text="CAPABILITIES (the ceiling — denied by default)",
            bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(anchor="w", pady=(10, 2))
        self._capability_label = tk.Label(
            inner, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=720,
        )
        self._capability_label.pack(fill="x", anchor="w")

        # The transcript: routing decisions, activity, and isolation reports.
        holder = tk.Frame(col, bg=theme.base)
        holder.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._canvas = tk.Canvas(holder, bg=theme.base, highlightthickness=0, borderwidth=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(holder, orient="vertical", command=self._canvas.yview)
        scroll.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=scroll.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._transcript_id = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._transcript_id, width=e.width))
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_wheel)
            widget.bind("<Button-4>", self._on_wheel)
            widget.bind("<Button-5>", self._on_wheel)

    def _on_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self._canvas.yview_scroll(int(-delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            self._canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._canvas.yview_scroll(1, "units")

    # -- data -------------------------------------------------------------

    def _reload(self) -> None:
        """Read the personalities and the isolation policy, then draw the page."""
        try:
            rows = self.service.rows()
        except Exception as error:  # a broken registry must not blank the window
            self._set_status(f"Could not read the personalities: {error}")
            return
        self._rows = list(rows)
        self._fill_people()
        self._set_status(f"{len(rows)} personality(ies)")
        self._isolation_label.configure(text=self.service.isolation().describe())
        if self._selected_owner:
            self._show_profile(self._selected_owner)
        self._add_note(
            "Pick a personality on the left to see its profile and capability "
            "ceiling, or route a question to see which one would answer it.",
            title="Start here",
        )

    def _fill_people(self) -> None:
        self._people_list.delete(0, tk.END)
        for index, row in enumerate(self._rows):
            flag = "🔒" if row.private else "•"
            self._people_list.insert(tk.END, f"{flag} {row.name}")
            if row.owner == self._selected_owner:
                self._people_list.selection_clear(0, tk.END)
                self._people_list.selection_set(index)
        if not self._selected_owner and self._rows:
            self._selected_owner = self._rows[0].owner
            self._people_list.selection_set(0)
            self._show_profile(self._selected_owner)

    def _on_pick_personality(self, _event: tk.Event | None = None) -> None:
        selection = self._people_list.curselection()
        if not selection:
            return
        index = int(selection[-1])
        if 0 <= index < len(self._rows):
            self._selected_owner = self._rows[index].owner
            self._show_profile(self._selected_owner)

    def _show_profile(self, owner: str) -> None:
        try:
            profile: PersonalityProfile = self.service.profile(owner)
        except Exception as error:
            self._set_status(f"Could not read {owner}: {error}")
            return
        self._profile_label.configure(text=profile.describe())
        lines = []
        for line in self.service.capabilities(owner):
            mark = "✓" if line.allowed else "✕"
            lines.append(f"{mark} {line.capability}: {line.reason}")
        self._capability_label.configure(text="\n".join(lines))

    # -- actions ----------------------------------------------------------

    def _on_route(self) -> None:
        if self._busy:
            return
        question = self._route_var.get().strip()
        if not question:
            self._set_status("Type a question to route.")
            return
        self._run_background(
            "Routing the question…",
            lambda: self.service.route(question),
            self._finish_route,
        )

    def _finish_route(self, report: RouteReport, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"Could not route the question: {error}", title="Problem")
            return
        self._add_note(report.describe(), title="Routing decision")

    def _on_activity(self) -> None:
        if self._busy:
            return
        self._run_background("Reading activity…", self.service.activity, self._finish_activity)

    def _finish_activity(self, activities: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"Could not read the activity: {error}", title="Problem")
            return
        lines = [activity.describe() for activity in activities] or ["No task has run yet."]
        self._add_note("\n".join(lines), title="Measured activity")

    # -- transcript -------------------------------------------------------

    def _add_note(self, text: str, *, title: str = "") -> None:
        theme = self.theme
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        pad = 12 if width <= 1 else max(12, (width - _COLUMN_MAX) // 2)
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        if title:
            tk.Label(
                row, text=title, bg=theme.base, fg=theme.text_dim,
                font=self._type.font("2xs", weight="bold"), anchor="w",
            ).pack(fill="x", anchor="w")
        card = tk.Frame(row, bg=theme.surface)
        card.pack(fill="x")
        body = tk.Text(
            card, height=1, bg=theme.surface, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, wrap="word", cursor="arrow",
            font=self._type.font("2xs"),
        )
        body.insert("1.0", text)
        body.configure(state="disabled")
        body.pack(fill="x", padx=12, pady=10)
        self._fit_height(body)
        self._cards.append(body)
        self.window.after(30, lambda: self._canvas.yview_moveto(1.0))

    def _fit_height(self, body: tk.Text) -> None:
        """Size a card to its content, capped so a tall block scrolls itself."""
        try:
            body.update_idletasks()
            lines = int(body.index("end-1c").split(".")[0])
        except tk.TclError:
            return
        body.configure(height=max(1, min(lines, _MAX_CARD_LINES)))

    def _set_status(self, text: str) -> None:
        if hasattr(self, "_status_label"):
            self._status_label.configure(text=text)

    # -- worker thread ----------------------------------------------------

    def _run_background(
        self,
        note: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        self._busy = True
        self._set_status(note)

        def run() -> None:
            try:
                result = job()
                error: Exception | None = None
            except Exception as exc:
                result, error = None, exc
            self._results.put((finish, result, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        try:
            self._pump_after = self.window.after(60, self._pump)
        except tk.TclError:
            self._pump_after = None

    def _pump(self) -> None:
        self._pump_after = None
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self._schedule_pump()
            return
        self._busy = False
        self._set_status("")
        finish(result, error)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pump_after is not None:
            try:
                self.window.after_cancel(self._pump_after)
            except tk.TclError:
                pass
            self._pump_after = None
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        if self._on_close is not None:
            self._on_close()


#: A card taller than this many lines scrolls inside itself.
_MAX_CARD_LINES = 18

__all__ = ["SUBTITLE", "TITLE", "AgentsDialog"]
