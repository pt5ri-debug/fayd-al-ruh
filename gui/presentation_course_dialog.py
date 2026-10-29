"""The Course window: the curriculum behind the deck you are building.

This window opens from the Presentation Studio when the deck has actually reached
suitable course content -- that is, when :mod:`core.presentation_courses` finds a
curriculum course the deck's *own* slides are substantially about. It shows the
user what the deck has covered and, just as importantly, what it has not.

It is deliberately a *sibling* window that drives the same
:class:`gui.presentation.PresentationService` as the Studio, the final editor, and
``python -m cli presentation`` -- so there is one set of slides, one curriculum,
and one source of truth, never a second copy of anything.

Three honesty rules, carried over from the rest of the project:

* **nothing is invented.** The course, its concepts, and its relations all come
  from the project's own curriculum seed; this window only *shows* them.
* **the deck's own words decide.** Coverage is matched against the deck's title,
  outline, and slides (via the service), not the discussion, so the marks reflect
  what the presentation actually says.
* **the user stays in control.** The window never edits the deck itself: it offers
  the missing concepts as text to paste into the Studio's own question box, and
  jumps to the slides that already cover a concept. Nothing is added or changed
  without the user asking.

Tk is imported at module import, so this module is only ever imported lazily from
:mod:`gui.app` -- ``cli/`` must never import Tk.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

from core.presentation_courses import CourseDetail, CourseMatch

from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Course content"
SUBTITLE = (
    "The project's own curriculum behind this deck. Concepts the deck already "
    "covers are marked; the rest are the ideas it has not reached yet. Nothing "
    "here changes the deck — copy what you want into the Studio's question box."
)

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860


class PresentationCourseDialog:
    """Show the curriculum course the deck is about, and what it still lacks."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: Any,
        *,
        project_id: str,
        course_key: str,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
        on_use_concepts: Callable[[str], None] | None = None,
        on_open_slide: Callable[[int], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        # Hand the chosen text back to the Studio's own question box; the window
        # never drafts or edits on its own, so the user stays in control.
        self._on_use_concepts = on_use_concepts
        # Jump the Studio to the slides that already cover a concept.
        self._on_open_slide = on_open_slide
        self._closed = False
        self._project_id = project_id
        self._course_key = course_key
        self._detail: CourseDetail | None = None
        self._matches: tuple[CourseMatch, ...] = ()

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1060x640")
        self.window.minsize(860, 560)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.refresh()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=theme.titlebar)
        inner.pack(fill="x", padx=16, pady=(12, 8))
        titles = tk.Frame(inner, bg=theme.titlebar)
        titles.pack(side="left", anchor="w")
        tk.Label(
            titles, text=TITLE, bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=SUBTITLE, bg=theme.titlebar, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=760,
        ).pack(anchor="w", pady=(2, 0))
        self._close_button = tk.Button(
            inner, text="Close", command=self._close, relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4, cursor="hand2",
        )
        self._close_button.pack(side="right")
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        self._build_sidebar(body)
        self._build_main(body)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        # A stack of collapsible sections (the shared :class:`~gui.sidebar.SidebarColumn`):
        # the courses the deck covers and the concepts of the chosen course.
        column = SidebarColumn(
            parent,
            theme=self.theme,
            typography=self._type,
            width=320,
            scope="presentation-course",
        )
        self._sidebar_column = column
        self._sidebar_canvas = column.canvas
        self._sidebar_scrollbar = column.scrollbar
        self._sidebar = column.body

        theme = self.theme
        self._course_label = tk.Label(
            column.body, text="", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("heading4", weight="bold"), anchor="w", justify="left",
            wraplength=290,
        )
        self._course_label.pack(fill="x", padx=14, pady=(14, 0))
        self._branch_label = tk.Label(
            column.body, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=290,
        )
        self._branch_label.pack(fill="x", padx=14, pady=(2, 6))

        matches = column.add("matches", "Courses the deck covers", icon="📚")
        self._match_list = tk.Listbox(
            matches, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), height=6, activestyle="none",
        )
        self._match_list.pack(fill="x", padx=14, pady=(4, 6))
        self._match_list.bind("<<ListboxSelect>>", self._on_match_select)

        concepts = column.add("concepts", "Concepts", icon="🔬")
        self._concept_list = tk.Listbox(
            concepts, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), height=12, activestyle="none",
        )
        self._concept_list.pack(fill="both", expand=True, padx=14, pady=(4, 6))
        self._concept_list.bind("<<ListboxSelect>>", self._on_concept_select)

        actions = column.add("actions", "Use a concept", icon="✔")
        action_row = tk.Frame(actions, bg=theme.surface)
        action_row.pack(fill="x", padx=14, pady=(4, 12))
        self._use_button = self._button(action_row, "Use the missing concepts", self._on_use_missing)
        self._use_button.pack(side="left")
        self._use_button.configure(state="disabled")
        self._goto_button = self._button(action_row, "Go to its slides", self._on_goto_concept)
        self._goto_button.pack(side="left", padx=(6, 0))
        self._goto_button.configure(state="disabled")

    def _build_main(self, parent: tk.Frame) -> None:
        theme = self.theme
        main = tk.Frame(parent, bg=theme.base)
        main.grid(row=0, column=1, sticky="nsew")
        holder = tk.Frame(main, bg=theme.base)
        holder.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(holder, bg=theme.base, highlightthickness=0, borderwidth=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(holder, orient="vertical", command=self._canvas.yview)
        scroll.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=scroll.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._transcript_id = self._canvas.create_window(
            (0, 0), window=self._transcript, anchor="nw"
        )
        self._transcript.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._transcript_id, width=event.width)

    def _on_wheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4, cursor="hand2",
        )

    def _add_note(self, text: str, *, title: str = "") -> None:
        """One full-width note row in the chat window's own style."""
        theme = self.theme
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        pad = 16 if width <= 1 else max(16, (width - _COLUMN_MAX) // 2)
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        if title:
            tk.Label(
                row, text=title, bg=theme.base, fg=theme.text_dim,
                font=self._type.font("2xs", weight="bold"), anchor="w",
            ).pack(fill="x", anchor="w")
        body = tk.Frame(row, bg=theme.base)
        body.pack(fill="x")
        tk.Label(
            body, text=text, bg=theme.base, fg=theme.foreground,
            font=self._type.font("body"), anchor="w", justify="left",
            wraplength=max(360, _COLUMN_MAX - 40),
        ).pack(anchor="w")
        self.window.after(30, lambda: self._canvas.yview_moveto(1.0))

    # -- state ------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the deck's coverage from the service (the one source of truth)."""
        try:
            detail = self.service.course_detail(self._project_id, self._course_key)
            matches = self.service.course_matches(self._project_id)
        except Exception as error:
            self._add_note(f"Could not read the course: {error}", title="Problem")
            return
        self._detail = detail
        self._matches = matches

        self._course_label.configure(text=detail.label)
        self._branch_label.configure(text=detail.branch)

        self._match_list.delete(0, tk.END)
        for index, match in enumerate(matches):
            flag = "●" if match.strong else "○"
            self._match_list.insert(
                tk.END, f"{flag} {match.label} ({match.match_count}/{match.total})"
            )
            if match.key == detail.key:
                self._match_list.selection_clear(0, tk.END)
                self._match_list.selection_set(index)

        self._concept_list.delete(0, tk.END)
        for concept in detail.concepts:
            mark = "✓" if concept.present else "·"
            self._concept_list.insert(tk.END, f"{mark} {concept.name}")
        self._sync_buttons()

        self._render_transcript()

    def _render_transcript(self) -> None:
        for child in self._transcript.winfo_children():
            child.destroy()
        detail = self._detail
        if detail is None:
            return
        covered = ", ".join(concept.name for concept in detail.covered) or "nothing yet"
        missing = ", ".join(concept.name for concept in detail.missing) or "nothing — every concept is covered"
        self._add_note(
            f"**{detail.label}** — {detail.branch}\n\n"
            f"Covered by this deck ({len(detail.covered)}): {covered}\n\n"
            f"Not yet covered ({len(detail.missing)}): {missing}",
            title="Coverage",
        )
        if detail.relations:
            lines = "\n".join(f"• {relation.describe()}" for relation in detail.relations[:12])
            self._add_note(lines, title="The course's own relations")

    def _sync_buttons(self) -> None:
        detail = self._detail
        has_missing = bool(detail and detail.missing)
        self._use_button.configure(state="normal" if has_missing else "disabled")
        concept = self._selected_concept()
        has_slides = bool(concept and self.service.slides_for_concept(self._project_id, concept))
        self._goto_button.configure(state="normal" if has_slides else "disabled")

    def _selected_concept(self) -> str:
        selection = self._concept_list.curselection()
        if not selection or self._detail is None:
            return ""
        # ``curselection`` can hold more than one index (Tk's ``selection_set``
        # adds to the selection), so the *most recent* index is the one the user
        # means -- the same rule the match list uses.
        index = int(selection[-1])
        concepts = self._detail.concepts
        if 0 <= index < len(concepts):
            return concepts[index].name
        return ""

    # -- handlers ---------------------------------------------------------

    def _on_match_select(self, _event: tk.Event | None = None) -> None:
        selection = self._match_list.curselection()
        if not selection:
            return
        index = int(selection[-1])
        if 0 <= index < len(self._matches):
            self._course_key = self._matches[index].key
            self.refresh()

    def _on_concept_select(self, _event: tk.Event | None = None) -> None:
        self._sync_buttons()

    def _on_use_missing(self) -> None:
        """Hand the missing concepts to the Studio's question box."""
        text = self.service.course_concepts_text(self._project_id, self._course_key)
        if not text:
            return
        if callable(self._on_use_concepts):
            self._on_use_concepts(text)

    def _on_goto_concept(self) -> None:
        concept = self._selected_concept()
        if not concept:
            return
        numbers = self.service.slides_for_concept(self._project_id, concept)
        if numbers and callable(self._on_open_slide):
            self._on_open_slide(numbers[0])

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if callable(self._on_close):
            self._on_close()
        self.window.destroy()


__all__ = ["PresentationCourseDialog"]
