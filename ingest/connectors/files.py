"""Read local files: TXT, Markdown, DOCX, PDF, CSV/TSV, JSON, and JSONL.

Documents go through the project's existing :func:`core.document_reader.read_document`
(reused, not reimplemented) for TXT/Markdown/DOCX/PDF. The tabular and
line-delimited formats are read here, streamed, because they are the ones that
can be large: a JSONL file is read a line at a time, a CSV file a row at a time,
and rows are grouped into records of ``row_group`` lines so the store sees
useful passages rather than one tiny chunk per row.

A source may be a single file or a directory. A directory is walked lazily and
sorted, so a run is deterministic and memory stays flat regardless of how many
files it holds. When ``root`` is given the source is confined to it, matching
the path-confinement rule of :class:`core.tools.ToolSandbox`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from core.document_reader import DocumentReadError, read_document
from ingest.errors import ConnectorError
from ingest.hashing import file_fingerprint, source_key
from ingest.records import (
    ConnectorInfo,
    IngestItem,
    SourceRecord,
    SourceSpec,
)

#: Text-ish formats read through ``core.document_reader``.
READER_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".docx", ".docm", ".pdf"})
#: Tabular / line-delimited formats read and streamed here.
STREAM_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".jsonl", ".ndjson"})
ALL_SUFFIXES = READER_SUFFIXES | STREAM_SUFFIXES

_SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest-tmp", "node_modules"})


class FileConnector:
    """A connector over a single file or a directory of files."""

    info = ConnectorInfo(
        name="files",
        description="Local TXT/Markdown/DOCX/PDF/CSV/JSON/JSONL files and directories.",
        suffixes=ALL_SUFFIXES,
        default=True,
    )

    # -- planning ---------------------------------------------------------

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        row_group = int(spec.option("row_group", 0) or 40)
        normalize = _flag(spec.option("normalize", True))
        fold_arabic = _flag(spec.option("fold_arabic", True))
        allow_ocr = _flag(spec.option("allow_ocr", False))
        for path in self._paths(spec):
            suffix = path.suffix.lower()
            if suffix not in ALL_SUFFIXES:
                continue
            relative = self._relative(path, spec)
            yield IngestItem(
                source_key=source_key(relative),
                uri=str(path),
                title=path.stem or path.name,
                fingerprint=file_fingerprint(path),
                metadata={
                    "suffix": suffix,
                    "relative": relative,
                    "path": str(path),
                    "row_group": row_group,
                    "normalize": normalize,
                    "fold_arabic": fold_arabic,
                    "allow_ocr": allow_ocr,
                },
            )

    def _paths(self, spec: SourceSpec) -> list[Path]:
        path = self._resolve(spec)
        if path.is_file():
            return [path]
        if path.is_dir():
            recursive = bool(spec.option("recursive", True))
            pattern = "**/*" if recursive else "*"
            excludes = _exclude_roots(spec)
            candidates = []
            for candidate in path.glob(pattern):
                if not candidate.is_file():
                    continue
                relative = candidate.relative_to(path)
                if _skipped(relative) or _excluded(candidate, excludes):
                    continue
                candidates.append(candidate)
            return sorted(candidates)
        raise ConnectorError(f"No such file or directory: {path}")

    def _resolve(self, spec: SourceSpec) -> Path:
        root = spec.option("root")
        path = Path(spec.uri).expanduser()
        if root is not None:
            base = Path(root).expanduser().resolve()
            candidate = (base / path).resolve() if not path.is_absolute() else path.resolve()
            if base != candidate and base not in candidate.parents:
                raise ConnectorError(f"Path escapes the configured root: {spec.uri!r}")
            return candidate
        return path

    def _relative(self, path: Path, spec: SourceSpec) -> str:
        root = spec.option("root")
        if root is not None:
            try:
                return str(path.resolve().relative_to(Path(root).expanduser().resolve()))
            except ValueError:
                pass
        return str(path)

    # -- reading ----------------------------------------------------------

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        path = Path(item.uri)
        suffix = path.suffix.lower()
        metadata = dict(item.metadata)
        if suffix in STREAM_SUFFIXES:
            yield from self._stream_tabular(item, path, suffix, metadata)
            return
        yield from self._stream_document(item, path, metadata)

    def _stream_document(
        self, item: IngestItem, path: Path, metadata: dict
    ) -> Iterator[SourceRecord]:
        try:
            document = read_document(
                path,
                normalize=bool(metadata.get("normalize", True)),
                fold_arabic=bool(metadata.get("fold_arabic", True)),
                allow_ocr=bool(metadata.get("allow_ocr", False)),
            )
        except DocumentReadError as error:
            raise ConnectorError(str(error)) from error
        yield SourceRecord(
            title=document.title,
            text=document.text,
            source="file",
            source_key=item.source_key,
            metadata={
                **metadata,
                "kind": document.kind,
                "truncated": document.truncated,
            },
        )

    def _stream_tabular(
        self, item: IngestItem, path: Path, suffix: str, metadata: dict
    ) -> Iterator[SourceRecord]:
        row_group = int(item.metadata.get("row_group", 0)) or 40
        title = item.title
        buffer: list[str] = []
        index = 0
        for line in _iter_rows(path, suffix):
            buffer.append(line)
            if len(buffer) >= row_group:
                index += 1
                yield SourceRecord(
                    title=f"{title} — part {index}",
                    text="\n".join(buffer),
                    source="file",
                    source_key=item.source_key,
                    metadata={**metadata, "kind": suffix.lstrip("."), "part": index},
                )
                buffer = []
        if buffer:
            index += 1
            yield SourceRecord(
                title=f"{title} — part {index}" if index > 1 else title,
                text="\n".join(buffer),
                source="file",
                source_key=item.source_key,
                metadata={**metadata, "kind": suffix.lstrip("."), "part": index},
            )

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        for item in self.plan(spec):
            yield from self.stream_item(item)


def _iter_rows(path: Path, suffix: str) -> Iterator[str]:
    """Stream one text line per row from a CSV/TSV/JSON/JSONL file."""
    if suffix in {".csv", ".tsv"}:
        yield from _iter_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
        return
    if suffix in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl(path)
        return
    yield from _iter_json(path)


def _iter_csv(path: Path, *, delimiter: str) -> Iterator[str]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return
        for row in reader:
            if not row:
                continue
            pairs = (
                f"{column.strip()}: {value.strip()}"
                for column, value in zip(header, row)
                if value and value.strip()
            )
            line = "; ".join(pairs)
            if line:
                yield line


def _iter_jsonl(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            yield _render_json(value)


def _iter_json(path: Path) -> Iterator[str]:
    # A plain JSON document is read whole; use JSONL for very large collections.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise ConnectorError(f"Could not read {path.name!r}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConnectorError(f"{path.name!r} is not valid JSON: {error}") from error
    if isinstance(value, list):
        for entry in value:
            yield _render_json(entry)
    else:
        yield _render_json(value)


def _render_json(value) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_render_json(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(_render_json(item) for item in value)
    return str(value)


def _flag(value: object) -> bool:
    """Coerce an option value to a bool, treating "false"/"0"/"no" as False."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    return text not in {"0", "false", "no", "off", ""}


def _skipped(relative: Path) -> bool:
    """Skip hidden/known-noise directories inside a source (relative to its root)."""
    return any(
        part in _SKIP_DIRS or (part.startswith(".") and part not in (".", ".."))
        for part in relative.parts
    )


def _exclude_roots(spec: SourceSpec) -> tuple[Path, ...]:
    """Directories (given by option) whose subtrees are never walked."""
    roots = spec.option("exclude_roots") or ()
    return tuple(Path(root).expanduser().resolve() for root in roots)


def _excluded(path: Path, excludes: tuple[Path, ...]) -> bool:
    if not excludes:
        return False
    resolved = path.resolve()
    return any(root == resolved or root in resolved.parents for root in excludes)


__all__ = ["ALL_SUFFIXES", "FileConnector", "READER_SUFFIXES", "STREAM_SUFFIXES"]
