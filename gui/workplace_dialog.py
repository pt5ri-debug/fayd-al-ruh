"""The Workplace window: EPRI/TSTC's digital office, in one professional page.

Reached from the chat composer's **Workplace** control (beside **Code Studio**)
and from the sidebar, this is the project's own enterprise workspace for the
Egyptian Petroleum Research Institute and its Technical Support & Technology
Center (TSTC) -- *معهد بحوث البترول المصري – مركز الدعم الفني والتكنولوجي*.

Phase 2 gave the page its shape; Phase 3 completes the *configurable hierarchy*:

* a collapsible **institutional sidebar** (the shared :class:`~gui.sidebar.SidebarColumn`)
  holding the dashboard, the institution tree, the divisions, the division
  **profile**, the professional taxonomy, and the settings;
* a **dashboard** of real figures read from the Workplace database (never
  demonstration data);
* the **institution tree** and its ten *configurable* divisions, with add, rename,
  move, deactivate, and reactivate -- all without touching source code;
* each division's **technical profile** (the brief's §3): technical disciplines,
  personnel and qualifications, applicable standards, inspection and testing
  capabilities, procedures, equipment, project assignments, reports, historical
  records, and knowledge documents -- every item added and removed in the window;
* the **professional knowledge taxonomy** (the brief's §4), stored and editable:
  an administrator adds an area or a discipline without touching source code;
* the **interface language and layout direction** (English/Arabic, LTR/RTL).

Everything is a thin view over :class:`gui.workplace.WorkplaceService`, so the
whole workflow is testable with no display and the CLI (``python -m cli
workplace``) shares one database with the window. Tk is imported lazily with the
rest of the window, so the CLI never touches it.

The page is bilingual and RTL-ready: every label carries an English and an Arabic
form, and the sidebar sits on the side the direction preference asks for.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Any

from core.workplace.records import DOMAIN_RECORDS

from gui import clipboard
from gui.app import RoundedBox
from gui.dialog_base import ThemedDialog
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.workplace import (
    DivisionSummary,
    WorkplaceDomainError,
    WorkplaceError,
    WorkplaceService,
)

TITLE = "Workplace"
SUBTITLE = (
    "Egyptian Petroleum Research Institute — Technical Support & Technology Center "
    "(Petroleum - Industry) · معهد بحوث البترول المصري – مركز الدعم الفني والتكنولوجي"
)

#: The honest note shown from the first moment, so the page never overstates what
#: the ten divisions are.
HONESTY_LINE = (
    "The ten divisions below are **proposed, configurable defaults** — not an "
    "officially approved structure. Add, rename, reorganise, or deactivate any of "
    "them without touching source code; complete them from the centre's own "
    "approved documents."
)

#: The transcript reads as a centred column, capped like the chat's.
_COLUMN_MAX = 880
_MAX_BUBBLE_LINES = 30
SECTION_PAD = 16


class WorkplaceDialog(ThemedDialog):
    """The Workplace page: dashboard, institution tree, divisions, taxonomy."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: WorkplaceService,
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
        self._list_type = resolve(parent, ui_family=font, mono_family=mono, overrides=list_overrides)
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._row_frames: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []
        self._division_ids: list[str] = []
        self._profile_ids: list[str] = []
        self._results: queue.Queue[tuple[Callable[[Any, Exception | None], None], Any, Exception | None]] = queue.Queue()

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1240x780")
        self.window.minsize(980, 640)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    def _after_build(self) -> None:
        """Refresh the page from the service (first build, and after a re-theme)."""
        self._add_message("note", HONESTY_LINE)
        self._refresh_all()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        self._body = body
        body.bind("<Configure>", self._on_body_configure)
        self._build_sidebar(body)
        self._build_centre(body)
        self._layout_columns()

    def _build_header(self) -> None:
        """The shared title band (the one :class:`gui.components.DialogHeader`)."""
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close,
        )
        self._spinner = header.spinner
        self._window_controls = header.controls

    def _button(self, parent: tk.Misc, text: str, command: Any, *, small: bool = True) -> tk.Button:
        """A themed flat button (the shared :func:`gui.components.button`)."""
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant="secondary", small=small, anchor="w",
        )

    def _build_detail_kind_menu(self, parent: tk.Frame) -> tk.StringVar:
        """A compact picker for the kind of profile item to add (the brief's §3)."""
        theme = self.theme
        tk.Label(
            parent, text="New item kind", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", pady=(6, 0))
        variable = tk.StringVar(value="capability")
        labels = dict(self.service.detail_kind_labels())
        menu = tk.OptionMenu(parent, variable, *labels.keys())
        menu.configure(
            bg=theme.surface_raised, fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.foreground, relief="flat", highlightthickness=0,
            borderwidth=0, font=self._type.font("2xs"), anchor="w", padx=8,
        )
        menu["menu"].configure(
            bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.primary, activeforeground=theme.primary_foreground,
            font=self._type.font("2xs"),
        )
        menu.pack(fill="x", pady=(2, 0))
        return variable

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, scope="workplace", width=320,
        )
        self._sidebar_column = column
        self._sidebar_canvas = column.canvas
        self._sidebar_scrollbar = column.scrollbar
        self._sidebar = column.body

        theme = self.theme
        brand = tk.Frame(column.body, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=26, height=26, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 25, 25, fill=theme.primary, outline="")
        dot.create_text(
            13, 14, text="🏢", fill=theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        dot.pack(side="left")
        tk.Label(
            brand, text="Workplace · المكتب", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("body", weight="bold"),
        ).pack(side="left", padx=(8, 0))

        # Dashboard figures (read from the database, never fabricated).
        dash = column.add("dashboard", "Dashboard", icon="📊")
        self._dash_labels: dict[str, tk.Label] = {}
        for key, caption in (
            ("institutions", "Institutions"),
            ("units", "Organisational units"),
            ("divisions", "Divisions"),
            ("active_divisions", "Active divisions"),
            ("taxonomy", "Taxonomy areas · disciplines"),
            ("active_projects", "Active projects"),
            ("completed_projects", "Completed projects"),
            ("upcoming_inspections", "Inspections (planned)"),
            ("outstanding_findings", "Outstanding findings"),
            ("reports_awaiting_approval", "Reports awaiting approval"),
            ("pending_assignments", "Assignments"),
            ("documents", "Documents registered"),
            ("personnel", "Personnel"),
            ("clients", "Clients / contractors"),
            ("standards", "Standards in the library"),
            ("knowledge_records", "Knowledge records"),
        ):
            row = tk.Frame(dash, bg=theme.surface)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(
                row, text=caption, bg=theme.surface, fg=theme.text_secondary,
                font=self._type.font("xs"), anchor="w",
            ).pack(side="left")
            value = tk.Label(
                row, text="—", bg=theme.surface, fg=theme.contrast,
                font=self._type.font("xs", weight="bold"), anchor="e",
            )
            value.pack(side="right")
            self._dash_labels[key] = value
        self._status_line = tk.Label(
            dash, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=270,
        )
        self._status_line.pack(fill="x", padx=14, pady=(4, 10))

        # Institution tree + divisions.
        institution = column.add("institution", "Institution", icon="🏛")
        tk.Label(
            institution, text="The institution and its units", bg=theme.surface,
            fg=theme.text_dim, font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(6, 2))
        self._tree = tk.Text(
            institution, height=6, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            font=self._type.font("xs"), padx=10, pady=8, cursor="xterm", takefocus=1,
        )
        self._tree.pack(fill="x", padx=14, pady=(0, 6))
        self._tree.configure(state="disabled")

        tk.Label(
            institution, text="Divisions (configurable)", bg=theme.surface,
            fg=theme.text_dim, font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(6, 2))
        self._division_list = tk.Listbox(
            institution, height=8, activestyle="none", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            selectbackground=theme.primary, selectforeground=theme.primary_foreground,
            font=self._list_type.font("xs"), exportselection=False,
        )
        self._division_list.pack(fill="x", padx=14, pady=(0, 4))
        self._division_list.bind("<<ListboxSelect>>", lambda _e: self._on_division_selected())

        div_buttons = tk.Frame(institution, bg=theme.surface)
        div_buttons.pack(fill="x", padx=14, pady=(0, 6))
        self._button(div_buttons, "Add division…", self._on_add_division).pack(fill="x", pady=2)
        self._rename_button = self._button(div_buttons, "Rename…", self._on_rename_division)
        self._rename_button.pack(fill="x", pady=2)
        self._move_button = self._button(div_buttons, "Move up", self._on_move_up)
        self._move_button.pack(fill="x", pady=2)
        self._toggle_active_button = self._button(
            div_buttons, "Deactivate", self._on_toggle_active
        )
        self._toggle_active_button.pack(fill="x", pady=2)
        self._show_inactive = tk.BooleanVar(value=False)
        tk.Checkbutton(
            div_buttons, text="Show deactivated", variable=self._show_inactive,
            command=self._refresh_divisions, bg=theme.surface, fg=theme.text_secondary,
            activebackground=theme.surface, activeforeground=theme.contrast,
            selectcolor=theme.surface_raised, highlightthickness=0, borderwidth=0,
            font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", pady=(4, 0))

        # The selected division's technical profile (the brief's §3).
        profile = column.add("profile", "Divisional profile", icon="🧾")
        self._profile_title = tk.Label(
            profile, text="", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("xs", weight="bold"), anchor="w", justify="left",
            wraplength=270,
        )
        self._profile_title.pack(fill="x", padx=14, pady=(6, 2))
        self._profile_text = tk.Text(
            profile, height=8, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            font=self._type.font("xs"), padx=10, pady=8, cursor="xterm", takefocus=1,
        )
        self._profile_text.pack(fill="x", padx=14, pady=(0, 4))
        self._profile_text.configure(state="disabled")
        profile_buttons = tk.Frame(profile, bg=theme.surface)
        profile_buttons.pack(fill="x", padx=14, pady=(0, 6))
        self._button(profile_buttons, "Add profile item…", self._on_add_profile_item).pack(
            fill="x", pady=2
        )
        self._button(profile_buttons, "Remove selected item…", self._on_remove_profile_item).pack(
            fill="x", pady=2
        )
        self._detail_kinds = self._build_detail_kind_menu(profile_buttons)

        # The professional knowledge taxonomy.
        taxonomy = column.add("taxonomy", "Professional taxonomy", icon="🧭", open=False)
        self._taxonomy_text = tk.Text(
            taxonomy, height=10, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            font=self._type.font("xs"), padx=10, pady=8, cursor="xterm", takefocus=1,
        )
        self._taxonomy_text.pack(fill="x", padx=14, pady=(6, 4))
        self._taxonomy_text.configure(state="disabled")
        tax_buttons = tk.Frame(taxonomy, bg=theme.surface)
        tax_buttons.pack(fill="x", padx=14, pady=(0, 12))
        self._button(tax_buttons, "Add area…", self._on_add_taxonomy_area).pack(fill="x", pady=2)
        self._button(tax_buttons, "Add discipline…", self._on_add_taxonomy_discipline).pack(
            fill="x", pady=2
        )

        # The work records (the brief's §12): projects, people, standards, and
        # the audited domain service behind them.
        records = column.add("records", "Projects & records", icon="🗂", open=False)
        self._records_text = tk.Text(
            records, height=10, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            font=self._type.font("xs"), padx=10, pady=8, cursor="xterm", takefocus=1,
        )
        self._records_text.pack(fill="x", padx=14, pady=(6, 4))
        self._records_text.configure(state="disabled")

        tk.Label(
            records, text="Entity", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(4, 2))
        self._record_entity = tk.StringVar(value="projects")
        entity_menu = tk.OptionMenu(
            records, self._record_entity, *sorted(DOMAIN_RECORDS),
            command=lambda _value: self._refresh_records(),
        )
        entity_menu.configure(
            bg=theme.surface_raised, fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.foreground, relief="flat", highlightthickness=0,
            borderwidth=0, font=self._type.font("2xs"), anchor="w", padx=8,
        )
        entity_menu["menu"].configure(
            bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.primary, activeforeground=theme.primary_foreground,
            font=self._type.font("2xs"),
        )
        entity_menu.pack(fill="x", padx=14, pady=(0, 4))

        record_buttons = tk.Frame(records, bg=theme.surface)
        record_buttons.pack(fill="x", padx=14, pady=(0, 12))
        self._button(record_buttons, "Add a record…", self._on_add_record).pack(fill="x", pady=2)
        self._button(record_buttons, "Show the audit trail", self._on_show_audit).pack(
            fill="x", pady=2
        )

        # Settings (database, appearance, backup, defaults).
        settings = column.add("settings", "Settings", icon="⚙", open=False)
        self._db_label = tk.Label(
            settings, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=270,
        )
        self._db_label.pack(fill="x", padx=14, pady=(6, 6))

        tk.Label(
            settings, text="Interface language", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(4, 2))
        self._language = tk.StringVar(value="en")
        for code, label in (("en", "English"), ("ar", "العربية")):
            tk.Radiobutton(
                settings, text=label, value=code, variable=self._language,
                command=self._on_language, bg=theme.surface, fg=theme.foreground,
                activebackground=theme.surface, activeforeground=theme.contrast,
                selectcolor=theme.surface_raised, highlightthickness=0, borderwidth=0,
                font=self._type.font("2xs"), anchor="w",
            ).pack(fill="x", padx=18)
        self._rtl = tk.BooleanVar(value=False)
        tk.Checkbutton(
            settings, text="Right-to-left layout", variable=self._rtl, command=self._on_rtl,
            bg=theme.surface, fg=theme.text_secondary, activebackground=theme.surface,
            activeforeground=theme.contrast, selectcolor=theme.surface_raised,
            highlightthickness=0, borderwidth=0, font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=14, pady=(4, 0))

        self._button(settings, "Back up the database…", self._on_backup).pack(
            fill="x", padx=14, pady=(8, 2)
        )
        self._button(settings, "Reinstall the defaults…", self._on_reseed).pack(
            fill="x", padx=14, pady=(2, 12)
        )

    # -- centre -----------------------------------------------------------

    def _build_centre(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")
        self._centre = col

        container = tk.Frame(col, bg=theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side="right", fill="y")
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

    # -- geometry ---------------------------------------------------------

    def _on_body_configure(self, _event: tk.Event | None = None) -> None:
        self._layout_columns()

    def _layout_columns(self) -> None:
        try:
            width = self.window.winfo_width()
        except tk.TclError:
            return
        if width <= 1:
            return
        self._body.columnconfigure(0, minsize=min(320, max(260, int(width * 0.28))))

    def _column_pad(self) -> int:
        try:
            width = self._transcript.winfo_width() or self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            width = _COLUMN_MAX
        if width <= _COLUMN_MAX:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")

    def _reflow(self) -> None:
        pad = self._column_pad()
        for _role, _text, row, _widget, _card in self._row_frames:
            try:
                row.pack_configure(padx=(pad, pad))
            except tk.TclError:
                continue

    # -- transcript -------------------------------------------------------

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False
    ) -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        card: RoundedBox | None = None
        head = tk.Frame(row, bg=theme.base)
        head.pack(fill="x", pady=(0, 4))
        tk.Label(
            head, text=title or ("Workplace" if role != "note" else "Note"),
            bg=theme.base, fg=theme.text_dim, font=self._type.font("xs", weight="bold"),
            anchor="w",
        ).pack(side="left")
        widget = tk.Text(
            row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            bg=theme.base, fg=theme.foreground,
            font=self._type.mono("code") if mono_body else self._type.font("body"),
            padx=0, pady=0, cursor="xterm", takefocus=1,
        )
        widget.insert("1.0", text)
        widget.pack(anchor="w", fill="x")
        self._row_frames.append((role, text, row, widget, card))
        clipboard.make_read_only(widget)
        self._fit_height(widget)
        self._scroll_to_bottom()
        return widget

    def _fit_height(self, widget: tk.Text) -> None:
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    # -- refresh ----------------------------------------------------------

    def _refresh_all(self) -> None:
        self._refresh_dashboard()
        self._refresh_tree()
        self._refresh_divisions()
        self._refresh_profile()
        self._refresh_taxonomy()
        self._refresh_records()
        self._refresh_settings()

    def _refresh_dashboard(self) -> None:
        status = self.service.status()
        board = self.service.dashboard()
        values = {
            "institutions": str(status.institutions),
            "units": str(status.units),
            "divisions": str(status.divisions),
            "active_divisions": str(status.active_divisions),
            "taxonomy": f"{status.taxonomy_areas} · {status.taxonomy_disciplines}",
            "active_projects": str(board.active_projects),
            "completed_projects": str(board.completed_projects),
            "upcoming_inspections": str(board.upcoming_inspections),
            "outstanding_findings": str(board.outstanding_findings),
            "reports_awaiting_approval": str(board.reports_awaiting_approval),
            "pending_assignments": str(board.pending_assignments),
            "documents": str(board.documents),
            "personnel": str(board.personnel),
            "clients": str(board.clients),
            "standards": str(board.standards),
            "knowledge_records": str(board.knowledge_records),
        }
        for key, label in self._dash_labels.items():
            label.configure(text=values.get(key, "—"))
        self._status_line.configure(
            text=("Ready." if status.ready else status.message),
            fg=self.theme.text_dim if status.ready else self.theme.danger,
        )
        self._db_label.configure(text=f"Database:\n{status.db_path}")

    def _refresh_tree(self) -> None:
        view = self.service.hierarchy()
        lines: list[str] = []
        if view.institution is not None:
            mark = "" if view.institution.official else "  (internal / proposed)"
            lines.append(f"{view.institution.name} / {view.institution.name_ar}{mark}")
        for unit in view.units:
            mark = "" if unit.official else "  (proposed)"
            lines.append(f"  └─ {unit.name} / {unit.name_ar}{mark}")
        if not lines:
            lines.append("No institution yet — reinstall the defaults in Settings.")
        self._set_text(self._tree, "\n".join(lines))

    def _refresh_divisions(self) -> None:
        rows = self.service.divisions(include_inactive=self._show_inactive.get())
        self._division_list.delete(0, "end")
        self._division_ids = []
        for row in rows:
            self._division_list.insert("end", f"{row.name} · {row.name_ar}" if row.name_ar else row.name)
            self._division_ids.append(row.record_id)
        if rows:
            self._division_list.selection_set(0)
        self._on_division_selected()

    def _refresh_taxonomy(self) -> None:
        areas = self.service.taxonomy_tree()
        if areas:
            lines = []
            for area in areas:
                lines.append(f"{area.name} / {area.name_ar}")
                for discipline in area.metadata.get("disciplines", ()):
                    lines.append(f"    • {discipline}")
            self._set_text(self._taxonomy_text, "\n".join(lines))
            return
        # No stored taxonomy yet (a database created before Phase 3): fall back to
        # the shipped default so the tree is never empty, and say so honestly.
        lines = ["(shipped defaults — stored once you edit the taxonomy)"]
        for shipped in self.service.taxonomy():
            lines.append(f"{shipped.name} / {shipped.name_ar}")
            for discipline in shipped.disciplines:
                lines.append(f"    • {discipline}")
        self._set_text(self._taxonomy_text, "\n".join(lines))

    def _refresh_profile(self) -> None:
        summary = self._selected_division()
        if summary is None:
            self._profile_title.configure(text="Select a division to see its profile.")
            self._set_text(self._profile_text, "")
            self._profile_ids = []
            return
        profile = self.service.division_profile(summary.record_id)
        self._profile_title.configure(text=f"{summary.name} · profile")
        self._profile_ids = [d.record_id for d in profile.details]
        grouped = profile.by_kind()
        lines: list[str] = []
        if not grouped:
            lines.append("No profile items yet. Add one below.")
        for _kind, items in grouped.items():
            lines.append(f"{items[0].kind_label}:")
            for item in items:
                text = item.title or item.detail
                if item.title and item.detail:
                    text += f" — {item.detail}"
                if item.reference:
                    text += f"  [{item.reference}]"
                lines.append(f"    • {text}")
        self._set_text(self._profile_text, "\n".join(lines))

    def _refresh_settings(self) -> None:
        settings = self.service.settings()
        self._language.set(settings.language)
        self._rtl.set(settings.rtl)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # -- selection --------------------------------------------------------

    def _selected_division(self) -> DivisionSummary | None:
        selection = self._division_list.curselection()
        if not selection:
            return None
        index = selection[0]
        if index >= len(self._division_ids):
            return None
        return self.service.division(self._division_ids[index])

    def _on_division_selected(self) -> None:
        summary = self._selected_division()
        if summary is None:
            self._toggle_active_button.configure(text="Deactivate", state="disabled")
            self._refresh_profile()
            return
        self._toggle_active_button.configure(
            text="Reactivate" if not summary.active else "Deactivate", state="normal"
        )
        self._refresh_profile()

    # -- division actions -------------------------------------------------

    def _on_add_division(self) -> None:
        values = _prompt_division(self.window, self.theme, self._type)
        if values is None:
            return
        name, name_ar, summary = values
        try:
            created = self.service.add_division(name=name, name_ar=name_ar, summary=summary)
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", f"Added the **{created.name}** division.", title="Workplace")
        self._refresh_all()

    def _on_rename_division(self) -> None:
        summary = self._selected_division()
        if summary is None:
            return
        values = _prompt_division(
            self.window, self.theme, self._type,
            name=summary.name, name_ar=summary.name_ar, summary=summary.summary,
        )
        if values is None:
            return
        name, name_ar, _summary = values
        try:
            updated = self.service.rename_division(summary.record_id, name=name, name_ar=name_ar)
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", f"Renamed the division to **{updated.name}**.", title="Workplace")
        self._refresh_all()

    def _on_toggle_active(self) -> None:
        summary = self._selected_division()
        if summary is None:
            return
        try:
            if summary.active:
                updated = self.service.deactivate_division(summary.record_id)
                note = f"Deactivated the **{updated.name}** division. Its history is preserved."
            else:
                updated = self.service.activate_division(summary.record_id)
                note = f"Reactivated the **{updated.name}** division."
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", note, title="Workplace")
        self._refresh_all()

    def _on_move_up(self) -> None:
        summary = self._selected_division()
        if summary is None or summary.order <= 1:
            return
        try:
            self.service.set_division_order(summary.record_id, order=summary.order - 1)
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._refresh_all()

    # -- profile actions --------------------------------------------------

    def _on_add_profile_item(self) -> None:
        summary = self._selected_division()
        if summary is None:
            return
        kind = self._detail_kinds.get()
        values = _prompt_profile_item(
            self.window, self.theme, self._type,
            kind_label=dict(self.service.detail_kind_labels()).get(kind, kind),
        )
        if values is None:
            return
        title, detail, reference = values
        try:
            item = self.service.add_division_detail(
                division_id=summary.record_id, kind=kind,
                title=title, detail=detail, reference=reference,
            )
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message(
            "workplace",
            f"Added a **{item.kind_label}** item to the **{summary.name}** division.",
            title="Workplace",
        )
        self._refresh_all()

    def _on_remove_profile_item(self) -> None:
        summary = self._selected_division()
        if summary is None:
            return
        profile = self.service.division_profile(summary.record_id)
        if profile.empty:
            self._add_message("note", "This division has no profile items yet.", title="Note")
            return
        options = [f"{d.kind_label}: {d.title or d.detail}" for d in profile.details]
        chosen = _choose(self.window, self.theme, self._type, "Remove a profile item", options)
        if chosen is None:
            return
        target = profile.details[chosen]
        try:
            self.service.remove_division_detail(target.record_id)
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message(
            "workplace", f"Removed a profile item from **{summary.name}**.", title="Workplace"
        )
        self._refresh_all()

    # -- taxonomy actions -------------------------------------------------

    def _on_add_taxonomy_area(self) -> None:
        values = _prompt_division(self.window, self.theme, self._type)
        if values is None:
            return
        name, name_ar, summary = values
        try:
            node = self.service.add_taxonomy_area(name=name, name_ar=name_ar, summary=summary)
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", f"Added the **{node.name}** taxonomy area.", title="Workplace")
        self._refresh_all()

    def _on_add_taxonomy_discipline(self) -> None:
        areas = self.service.taxonomy_tree()
        if not areas:
            self._add_message(
                "note",
                "The taxonomy is still the shipped default. Add an area first, then a "
                "discipline under it.",
                title="Note",
            )
            return
        options = [area.name for area in areas]
        chosen = _choose(self.window, self.theme, self._type, "Add a discipline under", options)
        if chosen is None:
            return
        values = _prompt_division(self.window, self.theme, self._type)
        if values is None:
            return
        name, name_ar, _summary = values
        try:
            node = self.service.add_taxonomy_discipline(
                area_id=areas[chosen].record_id, name=name, name_ar=name_ar
            )
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", f"Added the **{node.name}** discipline.", title="Workplace")
        self._refresh_all()

    # -- records (the brief's §12) ---------------------------------------

    def _refresh_records(self) -> None:
        entity = self._record_entity.get()
        rows = self.service.domain.list_records(entity)
        lines: list[str] = []
        if not rows:
            lines.append(f"No {entity} records yet. Use “Add a record…” below.")
        for row in rows:
            title = next(
                (str(getattr(row, name)) for name in ("title", "name", "subject", "number")
                 if getattr(row, name, "")),
                row.record_id,
            )
            lines.append(f"• {row.record_id}  {title}")
        self._set_text(self._records_text, "\n".join(lines))

    def _on_add_record(self) -> None:
        entity = self._record_entity.get()
        record_type = DOMAIN_RECORDS[entity]
        fields = [
            spec.name
            for spec in dataclasses.fields(record_type)
            if spec.name
            not in {"record_id", "created_at", "updated_at", "active", "metadata"}
            and spec.type in {"str", "bool"}
        ]
        values = _prompt_record(self.window, self.theme, self._type, entity, fields)
        if values is None:
            return
        try:
            record = self.service.domain.create(entity, **values)
        except WorkplaceDomainError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", f"Added the **{entity}** record `{record.record_id}`.",
                          title="Workplace")
        self._refresh_all()

    def _on_show_audit(self) -> None:
        rows = self.service.domain.audit_tail(limit=20)
        if not rows:
            self._add_message("note", "The audit trail is empty.", title="Audit")
            return
        lines = [
            f"{row.created_at} · {row.actor} · {row.action} · {row.entity} {row.entity_id}"
            + (f" — {row.detail}" if row.detail else "")
            for row in rows
        ]
        self._add_message("workplace", "\n".join(lines), title="Audit trail (newest first)")

    # -- settings actions -------------------------------------------------

    def _on_language(self) -> None:
        try:
            self.service.update_settings(language=self._language.get())
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")

    def _on_rtl(self) -> None:
        try:
            self.service.update_settings(rtl=self._rtl.get())
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")

    def _on_backup(self) -> None:
        def job() -> str:
            return self.service.backup()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("note", str(error), title="Problem")
                return
            self._add_message("workplace", f"Backed up the database to:\n{result}", title="Workplace")

        self._run_background("Backing up…", job, finish)

    def _on_reseed(self) -> None:
        from tkinter import messagebox

        if not messagebox.askyesno(
            "Reinstall the defaults",
            "Reinstall the Workplace defaults? Existing records are kept; missing "
            "ones are added back.",
            parent=self.window,
        ):
            return
        try:
            report = self.service.seed()
        except WorkplaceError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("workplace", report.describe(), title="Workplace")
        self._refresh_all()

    # -- background work --------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        self._busy = True
        if self._status_line is not None:
            self._status_line.configure(text=message)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:  # reported to the caller, never swallowed
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self.window.after(60, self._pump_results)
            return
        self._busy = False
        self._spinner.stop()
        self._spinner.pack_forget()
        finish(result, error)
        self._refresh_dashboard()

    # -- close ------------------------------------------------------------

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _prompt_division(
    parent: tk.Misc,
    theme: Any,
    typography: Any,
    *,
    name: str = "",
    name_ar: str = "",
    summary: str = "",
) -> tuple[str, str, str] | None:
    """A small modal asking for a division's name, Arabic name, and summary."""
    dialog = tk.Toplevel(parent)
    dialog.title("Division")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, tuple[str, str, str] | None] = {"value": None}

    frame = tk.Frame(dialog, bg=theme.surface)
    frame.pack(fill="both", expand=True, padx=18, pady=18)

    def field(label: str, value: str) -> tk.Entry:
        tk.Label(
            frame, text=label, bg=theme.surface, fg=theme.text_secondary,
            font=typography.font("xs"), anchor="w",
        ).pack(fill="x", pady=(6, 2))
        entry = tk.Entry(
            frame, bg=theme.surface_raised, fg=theme.foreground, insertbackground=theme.foreground,
            relief="flat", font=typography.font("body"),
        )
        entry.insert(0, value)
        entry.pack(fill="x")
        return entry

    name_entry = field("Name (English)", name)
    name_ar_entry = field("الاسم بالعربية", name_ar)
    summary_entry = field("Summary", summary)

    def submit() -> None:
        result["value"] = (
            name_entry.get().strip(),
            name_ar_entry.get().strip(),
            summary_entry.get().strip(),
        )
        dialog.destroy()

    def cancel() -> None:
        result["value"] = None
        dialog.destroy()

    buttons = tk.Frame(frame, bg=theme.surface)
    buttons.pack(fill="x", pady=(12, 0))
    from gui.components import button

    button(buttons, "Save", submit, theme=theme, typography=typography, variant="primary").pack(
        side="right"
    )
    button(buttons, "Cancel", cancel, theme=theme, typography=typography).pack(
        side="right", padx=(0, 8)
    )
    dialog.bind("<Return>", lambda _e: submit())
    dialog.bind("<Escape>", lambda _e: cancel())
    name_entry.focus_set()
    parent.wait_window(dialog)
    return result["value"]


def _prompt_profile_item(
    parent: tk.Misc,
    theme: Any,
    typography: Any,
    *,
    kind_label: str = "",
) -> tuple[str, str, str] | None:
    """A small modal asking for a profile item's title, detail, and reference."""
    dialog = tk.Toplevel(parent)
    dialog.title("Profile item")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, tuple[str, str, str] | None] = {"value": None}

    frame = tk.Frame(dialog, bg=theme.surface)
    frame.pack(fill="both", expand=True, padx=18, pady=18)
    if kind_label:
        tk.Label(
            frame, text=kind_label, bg=theme.surface, fg=theme.text_dim,
            font=typography.font("xs", weight="bold"), anchor="w",
        ).pack(fill="x", pady=(0, 4))

    def field(label: str, value: str = "") -> tk.Entry:
        tk.Label(
            frame, text=label, bg=theme.surface, fg=theme.text_secondary,
            font=typography.font("xs"), anchor="w",
        ).pack(fill="x", pady=(6, 2))
        entry = tk.Entry(
            frame, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", font=typography.font("body"),
        )
        entry.insert(0, value)
        entry.pack(fill="x")
        return entry

    title_entry = field("Title")
    detail_entry = field("Detail")
    reference_entry = field("Reference (optional)")

    def submit() -> None:
        result["value"] = (
            title_entry.get().strip(),
            detail_entry.get().strip(),
            reference_entry.get().strip(),
        )
        dialog.destroy()

    def cancel() -> None:
        result["value"] = None
        dialog.destroy()

    buttons = tk.Frame(frame, bg=theme.surface)
    buttons.pack(fill="x", pady=(12, 0))
    from gui.components import button

    button(buttons, "Save", submit, theme=theme, typography=typography, variant="primary").pack(
        side="right"
    )
    button(buttons, "Cancel", cancel, theme=theme, typography=typography).pack(
        side="right", padx=(0, 8)
    )
    dialog.bind("<Return>", lambda _e: submit())
    dialog.bind("<Escape>", lambda _e: cancel())
    title_entry.focus_set()
    parent.wait_window(dialog)
    return result["value"]


def _prompt_record(
    parent: tk.Misc,
    theme: Any,
    typography: Any,
    entity: str,
    fields: list[str],
) -> dict[str, str] | None:
    """A modal asking for a new domain record's fields (empty ones are skipped)."""
    dialog = tk.Toplevel(parent)
    dialog.title(f"New {entity}")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, dict[str, str] | None] = {"value": None}

    frame = tk.Frame(dialog, bg=theme.surface)
    frame.pack(fill="both", expand=True, padx=18, pady=18)
    tk.Label(
        frame, text=f"New {entity} record", bg=theme.surface, fg=theme.contrast,
        font=typography.font("body", weight="bold"), anchor="w",
    ).pack(fill="x", pady=(0, 6))

    entries: dict[str, tk.Entry] = {}
    # A canvas scroll is unnecessary for the handful of fields an entity has.
    for name in fields:
        tk.Label(
            frame, text=name, bg=theme.surface, fg=theme.text_secondary,
            font=typography.font("xs"), anchor="w",
        ).pack(fill="x", pady=(6, 2))
        entry = tk.Entry(
            frame, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", font=typography.font("body"),
        )
        entry.pack(fill="x")
        entries[name] = entry

    def submit() -> None:
        result["value"] = {
            name: entry.get().strip() for name, entry in entries.items() if entry.get().strip()
        }
        dialog.destroy()

    def cancel() -> None:
        result["value"] = None
        dialog.destroy()

    buttons = tk.Frame(frame, bg=theme.surface)
    buttons.pack(fill="x", pady=(12, 0))
    from gui.components import button

    button(buttons, "Save", submit, theme=theme, typography=typography, variant="primary").pack(
        side="right"
    )
    button(buttons, "Cancel", cancel, theme=theme, typography=typography).pack(
        side="right", padx=(0, 8)
    )
    dialog.bind("<Return>", lambda _e: submit())
    dialog.bind("<Escape>", lambda _e: cancel())
    if entries:
        next(iter(entries.values())).focus_set()
    parent.wait_window(dialog)
    return result["value"]


def _choose(
    parent: tk.Misc, theme: Any, typography: Any, title: str, options: list[str]
) -> int | None:
    """A small modal asking the user to choose one of ``options`` (its index)."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, int | None] = {"value": None}

    frame = tk.Frame(dialog, bg=theme.surface)
    frame.pack(fill="both", expand=True, padx=18, pady=18)
    tk.Label(
        frame, text=title, bg=theme.surface, fg=theme.contrast,
        font=typography.font("body", weight="bold"), anchor="w",
    ).pack(fill="x", pady=(0, 6))
    listbox = tk.Listbox(
        frame, height=min(10, max(3, len(options))), activestyle="none", relief="flat",
        borderwidth=0, highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
        selectbackground=theme.primary, selectforeground=theme.primary_foreground,
        font=typography.font("xs"), exportselection=False,
    )
    for option in options:
        listbox.insert("end", option)
    listbox.pack(fill="both", expand=True)
    if options:
        listbox.selection_set(0)

    def submit() -> None:
        selection = listbox.curselection()
        result["value"] = selection[0] if selection else None
        dialog.destroy()

    def cancel() -> None:
        result["value"] = None
        dialog.destroy()

    buttons = tk.Frame(frame, bg=theme.surface)
    buttons.pack(fill="x", pady=(12, 0))
    from gui.components import button

    button(buttons, "OK", submit, theme=theme, typography=typography, variant="primary").pack(
        side="right"
    )
    button(buttons, "Cancel", cancel, theme=theme, typography=typography).pack(
        side="right", padx=(0, 8)
    )
    listbox.bind("<Double-Button-1>", lambda _e: submit())
    dialog.bind("<Return>", lambda _e: submit())
    dialog.bind("<Escape>", lambda _e: cancel())
    listbox.focus_set()
    parent.wait_window(dialog)
    return result["value"]


__all__ = [
    "HONESTY_LINE",
    "SUBTITLE",
    "TITLE",
    "WorkplaceDialog",
]
