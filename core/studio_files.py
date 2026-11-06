"""The Code Studio's file engine: the file work a developer does beside git.

:mod:`core.code_studio` reads, writes, edits, and searches one file at a time.
This module adds the operations that act on the *tree*: a nested file listing
(the sidebar's explorer), create / rename / copy / delete, a **project-wide**
search and replace, a symbol outline of a file, and a scan for the notes a
developer leaves behind (``TODO``/``FIXME``/``XXX``/``HACK``).

It is dependency-free (stdlib only) and, like the rest of the Studio, it is
**confined**: every path is resolved through :class:`core.code_studio.CodeStudio`,
which refuses anything that escapes the workspace, and every write goes through
the same text-suffix and size checks. A binary file is never rewritten, and a
delete is refused for a path the Studio would not edit in the first place.

Nothing here executes a file's contents -- an outline is read with ``ast`` for a
Python file and with a conservative regular expression for anything else.
"""

from __future__ import annotations

import ast
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from core.code_studio import (
    SKIP_DIRS,
    CodeStudio,
    StudioError,
    Workspace,
)

#: A file larger than this is not opened for a scan or a replace: the operations
#: here act on source, and a huge generated file is not source a user edits.
DEFAULT_MAX_SCAN_BYTES = 2_000_000

#: A ceiling on how many matches one replace touches, so a careless pattern
#: cannot rewrite thousands of files in one call.
DEFAULT_MAX_REPLACEMENTS = 5_000

#: The note markers a source tree commonly carries, in the order a report shows
#: them (most urgent first).
TODO_MARKERS: tuple[str, ...] = ("FIXME", "BUG", "XXX", "HACK", "TODO")

_TODO_RE = re.compile(r"\b(FIXME|BUG|XXX|HACK|TODO)\b[:\s]*(.*)", re.IGNORECASE)


@dataclass(frozen=True)
class TreeEntry:
    """One node in the workspace's file tree (a file or a directory)."""

    path: str
    name: str
    directory: bool
    size: int = 0
    children: tuple[TreeEntry, ...] = ()

    @property
    def file_count(self) -> int:
        if not self.directory:
            return 1
        return sum(child.file_count for child in self.children)

    def describe(self, *, indent: int = 0) -> str:
        pad = "  " * indent
        mark = "/" if self.directory else ""
        return f"{pad}{self.name}{mark}"


@dataclass(frozen=True)
class FileChange:
    """One file a batch operation touched (or would touch)."""

    path: str
    action: str
    detail: str = ""

    def describe(self) -> str:
        return f"{self.action} {self.path}" + (f"  {self.detail}" if self.detail else "")


@dataclass
class BatchReport:
    """The outcome of a tree-wide operation (search and replace, or delete)."""

    ok: bool
    message: str
    changes: tuple[FileChange, ...] = ()
    applied: bool = False
    detail: str = ""

    def summary_lines(self) -> list[str]:
        return [change.describe() for change in self.changes]


@dataclass(frozen=True)
class Symbol:
    """One definition (or class member) found by an outline."""

    name: str
    kind: str
    line: int
    detail: str = ""

    def describe(self) -> str:
        suffix = f"  ({self.detail})" if self.detail else ""
        return f"{self.line:>5}  {self.kind:<10} {self.name}{suffix}"


@dataclass(frozen=True)
class Note:
    """One TODO-style note found in the tree."""

    path: str
    line: int
    marker: str
    text: str

    def describe(self) -> str:
        return f"{self.path}:{self.line}: {self.marker}: {self.text}"


@dataclass
class OutlineReport:
    """A file's symbols, or the honest reason none could be read."""

    path: str
    symbols: tuple[Symbol, ...] = ()
    language: str = ""
    error: str = ""

    def describe(self) -> str:
        if self.error:
            return f"{self.path}: {self.error}"
        if not self.symbols:
            return f"{self.path}: no symbols found."
        header = f"{self.path} ({self.language}):"
        return "\n".join([header, *(symbol.describe() for symbol in self.symbols)])


@dataclass
class NoteScan:
    """Every note found in the tree, with its own count summary."""

    notes: tuple[Note, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.notes:
            return "No TODO-style notes found."
        head = ", ".join(f"{marker}: {count}" for marker, count in self.counts.items())
        return "\n".join([head, *(note.describe() for note in self.notes)])


class FileEngine:
    """Tree operations over one :class:`core.code_studio.CodeStudio` workspace."""

    def __init__(self, studio: CodeStudio, *, max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES) -> None:
        self.studio = studio
        self.max_scan_bytes = max_scan_bytes

    # --- listing --------------------------------------------------------
    def editable_paths(self, workspace: Workspace, *, limit: int = 20_000) -> tuple[str, ...]:
        """Every path the Studio may edit, as POSIX-relative strings, sorted."""
        return tuple(entry.path for entry in self.studio.list_files(workspace, limit=limit))

    def tree(self, workspace: Workspace, *, relative: str = "") -> tuple[TreeEntry, ...]:
        """A nested listing of the workspace's editable files.

        Directories with no editable file anywhere beneath them are omitted, so
        the tree shows exactly what the Studio can act on and nothing else.
        """
        base = self.studio.resolve(workspace, relative) if relative.strip() else workspace.path
        if not base.exists():
            raise StudioError(f"Not a path in the workspace: {relative}")
        if base.is_file():
            return (TreeEntry(path=base.name, name=base.name, directory=False, size=base.stat().st_size),)
        return self._tree_children(workspace, base)

    def _tree_children(self, workspace: Workspace, directory: Path) -> tuple[TreeEntry, ...]:
        entries: list[TreeEntry] = []
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
        except OSError:
            return ()
        for child in children:
            if child.name in SKIP_DIRS or child.name == ".git":
                continue
            relative = child.relative_to(workspace.path).as_posix()
            if child.is_dir():
                nested = self._tree_children(workspace, child)
                if nested:
                    entries.append(
                        TreeEntry(path=relative, name=child.name, directory=True, children=nested)
                    )
                continue
            try:
                self.studio._check_suffix(child)
            except StudioError:
                continue
            try:
                size = child.stat().st_size
            except OSError:
                continue
            entries.append(TreeEntry(path=relative, name=child.name, directory=False, size=size))
        return tuple(entries)

    # --- create / rename / copy / delete --------------------------------
    def create_file(self, workspace: Workspace, relative: str, content: str = "") -> int:
        """Create a new text file (refusing to overwrite an existing one)."""
        path = self.studio.resolve(workspace, relative)
        self.studio._check_suffix(path)
        if path.exists():
            raise StudioError(f"`{relative}` already exists.")
        return self.studio.write_file(workspace, relative, content)

    def create_directory(self, workspace: Workspace, relative: str) -> None:
        """Create a directory inside the workspace (no-op if it exists)."""
        path = self.studio.resolve(workspace, relative)
        if path.exists() and not path.is_dir():
            raise StudioError(f"`{relative}` is a file, not a directory.")
        path.mkdir(parents=True, exist_ok=True)

    def rename(self, workspace: Workspace, relative: str, new_relative: str) -> str:
        """Rename (or move) a file, confined to the workspace."""
        source = self.studio.resolve(workspace, relative)
        target = self.studio.resolve(workspace, new_relative)
        self.studio._check_suffix(target)
        if not source.exists():
            raise StudioError(f"Not a path in the workspace: {relative}")
        if target.exists():
            raise StudioError(f"`{new_relative}` already exists.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        self.studio._invalidate_checks(workspace)
        return target.relative_to(workspace.path).as_posix()

    def copy(self, workspace: Workspace, relative: str, new_relative: str) -> str:
        """Copy a file to a new path inside the workspace."""
        source = self.studio.resolve(workspace, relative)
        target = self.studio.resolve(workspace, new_relative)
        self.studio._check_suffix(target)
        if not source.is_file():
            raise StudioError(f"Not a file in the workspace: {relative}")
        if target.exists():
            raise StudioError(f"`{new_relative}` already exists.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        self.studio._invalidate_checks(workspace)
        return target.relative_to(workspace.path).as_posix()

    def delete(self, workspace: Workspace, relative: str) -> None:
        """Delete one editable file from the workspace.

        A delete is refused for anything the Studio would not edit, so a stray
        binary or a run-state file cannot be removed through this path.
        """
        path = self.studio.resolve(workspace, relative)
        if path.is_dir():
            raise StudioError("Use delete_directory for a folder.")
        self.studio._check_suffix(path)
        if not path.is_file():
            raise StudioError(f"Not a file in the workspace: {relative}")
        path.unlink()
        self.studio._invalidate_checks(workspace)

    def delete_directory(self, workspace: Workspace, relative: str) -> int:
        """Delete a directory and everything under it; returns the files removed."""
        path = self.studio.resolve(workspace, relative)
        if path == workspace.path.resolve():
            raise StudioError("Refusing to delete the workspace root.")
        if not path.is_dir():
            raise StudioError(f"Not a directory in the workspace: {relative}")
        removed = sum(1 for item in path.rglob("*") if item.is_file())
        shutil.rmtree(path, ignore_errors=False)
        self.studio._invalidate_checks(workspace)
        return removed

    # --- project-wide search and replace --------------------------------
    def replace_across(
        self,
        workspace: Workspace,
        old: str,
        new: str,
        *,
        relative: str = "",
        apply: bool = False,
        max_replacements: int = DEFAULT_MAX_REPLACEMENTS,
    ) -> BatchReport:
        """Replace a literal string across the tree, previewing by default.

        The match is **literal** (never a regular expression), so a search for
        ``self.x`` finds exactly that. With ``apply=False`` the report lists what
        *would* change and writes nothing -- the preview a user reviews first.
        A file with no match is skipped, so the report names only real changes.
        """
        if not old:
            return BatchReport(False, "The text to replace cannot be empty.")
        targets = self._scan_targets(workspace, relative)
        changes: list[FileChange] = []
        total = 0
        for path in targets:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            count = text.count(old)
            if not count:
                continue
            relative_path = path.relative_to(workspace.path).as_posix()
            total += count
            changes.append(FileChange(relative_path, "replace", f"{count} occurrence(s)"))
            if total > max_replacements:
                return BatchReport(
                    False,
                    f"Refusing to replace more than {max_replacements} occurrences in one call.",
                    changes=tuple(changes),
                )
        if not changes:
            return BatchReport(True, f"No file contains {old!r}.", changes=())
        if apply:
            for change in changes:
                path = workspace.path / change.path
                text = path.read_text(encoding="utf-8")
                path.write_text(text.replace(old, new), encoding="utf-8")
            self.studio._invalidate_checks(workspace)
            verb = "Replaced"
        else:
            verb = "Would replace"
        summary = f"{verb} {old!r} in {len(changes)} file(s) ({total} occurrence(s))."
        return BatchReport(True, summary, changes=tuple(changes), applied=apply)

    def _scan_targets(self, workspace: Workspace, relative: str = "") -> tuple[Path, ...]:
        base = self.studio.resolve(workspace, relative) if relative.strip() else workspace.path
        if base.is_file():
            return (base,)
        found: list[Path] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace.path)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            try:
                self.studio._check_suffix(path)
            except StudioError:
                continue
            try:
                if path.stat().st_size > self.max_scan_bytes:
                    continue
            except OSError:
                continue
            found.append(path)
        return tuple(found)

    # --- outline --------------------------------------------------------
    def outline(self, workspace: Workspace, relative: str) -> OutlineReport:
        """The definitions in one file: classes, functions, methods, or headings.

        A Python file is read with ``ast`` (exact); any other text file falls back
        to a conservative pattern that recognises common definition and heading
        syntax, and a file with no recognised symbol reports that honestly rather
        than inventing one.
        """
        path = self.studio.resolve(workspace, relative)
        if not path.is_file():
            return OutlineReport(relative, error="Not a file in the workspace.")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return OutlineReport(relative, error="The file is not readable as UTF-8 text.")
        if path.suffix.casefold() == ".py":
            try:
                return OutlineReport(relative, symbols=self._python_symbols(text), language="python")
            except SyntaxError as error:
                return OutlineReport(
                    relative, error=f"Python could not parse it: {error}", language="python"
                )
        return OutlineReport(
            relative, symbols=self._generic_symbols(text), language=path.suffix.lstrip(".") or "text"
        )

    @staticmethod
    def _python_symbols(text: str) -> tuple[Symbol, ...]:
        tree = ast.parse(text)
        symbols: list[Symbol] = []

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    args = [a.arg for a in child.args.args]
                    symbols.append(
                        Symbol(f"{prefix}{child.name}", kind, child.lineno, ", ".join(args))
                    )
                    walk(child, f"{prefix}{child.name}.")
                elif isinstance(child, ast.ClassDef):
                    bases = [ast.unparse(b) for b in child.bases]
                    symbols.append(
                        Symbol(f"{prefix}{child.name}", "class", child.lineno, ", ".join(bases))
                    )
                    walk(child, f"{prefix}{child.name}.")
                else:
                    walk(child, prefix)

        walk(tree, "")
        return tuple(symbols)

    @staticmethod
    def _generic_symbols(text: str) -> tuple[Symbol, ...]:
        """A best-effort outline for a non-Python file: definitions and headings."""
        symbols: list[Symbol] = []
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):  # Markdown / shell heading or comment
                heading = stripped.lstrip("#").strip()
                if heading:
                    symbols.append(Symbol(heading[:80], "heading", number))
                continue
            match = re.match(
                r"(?:export\s+)?(?:async\s+)?(function|class|def|interface|struct|enum|const|let|var)\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                symbols.append(Symbol(match.group(2), match.group(1), number))
        return tuple(symbols)

    # --- notes (TODO scan) ----------------------------------------------
    def scan_notes(self, workspace: Workspace, *, relative: str = "") -> NoteScan:
        """Find ``TODO``/``FIXME``/``BUG``/``XXX``/``HACK`` notes in the tree."""
        notes: list[Note] = []
        for path in self._scan_targets(workspace, relative):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = path.relative_to(workspace.path).as_posix()
            for number, line in enumerate(text.splitlines(), start=1):
                match = _TODO_RE.search(line)
                if not match:
                    continue
                marker = match.group(1).upper()
                body = match.group(2).strip().rstrip("*/").strip()
                notes.append(Note(path=rel, line=number, marker=marker, text=body[:200]))
        counts: dict[str, int] = {}
        for note in notes:
            counts[note.marker] = counts.get(note.marker, 0) + 1
        ordered = {marker: counts[marker] for marker in TODO_MARKERS if marker in counts}
        return NoteScan(notes=tuple(notes), counts=ordered)

    # --- a quick file summary -------------------------------------------
    def file_summary(self, workspace: Workspace, relative: str) -> dict[str, int | str]:
        """Size, line count, and a rough non-blank line count for one file."""
        path = self.studio.resolve(workspace, relative)
        if not path.is_file():
            raise StudioError(f"Not a file in the workspace: {relative}")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "lines": len(lines),
            "non_blank": sum(1 for line in lines if line.strip()),
        }


def iter_tree(entries: Iterable[TreeEntry], *, indent: int = 0) -> Iterable[str]:
    """Flatten a tree into indented lines, for a plain-text rendering."""
    for entry in entries:
        yield entry.describe(indent=indent)
        if entry.directory:
            yield from iter_tree(entry.children, indent=indent + 1)


__all__ = [
    "DEFAULT_MAX_REPLACEMENTS",
    "DEFAULT_MAX_SCAN_BYTES",
    "TODO_MARKERS",
    "BatchReport",
    "FileChange",
    "FileEngine",
    "Note",
    "NoteScan",
    "OutlineReport",
    "Symbol",
    "TreeEntry",
    "iter_tree",
]
