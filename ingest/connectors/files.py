"""Read local files: TXT, Markdown, CSV/TSV, JSON, and JSONL.

Stage 1 of file ingestion supports four families:

* **TXT** and **Markdown** (``.txt``, ``.md``, ``.markdown``) -- read through the
  project's existing :func:`core.document_reader.read_document` (reused, not
  reimplemented), which decodes them safely (BOM, then UTF-8, then a short list
  of legacy encodings, each strict) and preserves Arabic and English text.
* **CSV/TSV** -- streamed a row at a time; each row becomes a deterministic
  ``column: value`` line, and rows are grouped into records of ``row_group``
  lines so the store sees useful passages rather than one tiny chunk per row.
* **JSON** and **JSONL** -- a JSON array yields one record per element, a JSON
  object yields one record, and JSONL yields one record per line. Values are
  rendered to text by :func:`_render_json` (see below).

**Structured-to-text conversion (deterministic, documented).** A CSV row is
rendered as ``header1: value1; header2: value2`` for the columns that have a
non-empty value, in header order. A JSON value is rendered recursively: an object
as ``key: value`` pairs joined by ``"; "`` in insertion order, an array as its
elements joined by ``", "``, and a scalar via ``str``. Nothing is reordered or
re-keyed, so the same input always yields the same text.

**Resource limits (honest, not "constant memory").** CSV and JSONL are streamed
and never held whole; memory is bounded by one row plus the ``row_group`` buffer.
A plain JSON document must be parsed whole (the standard library offers no
streaming JSON parser), so it is capped by ``max_json_bytes`` (default 64 MiB)
and a larger file is refused with a pointer to JSONL rather than silently
exhausting memory. TXT/Markdown larger than the reader's stream threshold are
read line by line with a hard character cap (see ``core.document_reader``).

A source may be a single file or a directory. A directory is walked lazily --
one directory's entries are sorted and yielded, then its subdirectories are
visited depth-first -- so a run is deterministic *and* memory stays bounded
regardless of how many files the folder holds. When ``root`` is given the source
is confined to it, matching the path-confinement rule of
:class:`core.tools.ToolSandbox`.

Nothing here executes an imported file or follows instructions inside it: every
reader only *parses* bytes into text.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Iterator
from pathlib import Path

from core.document_reader import (
    DocumentReadError,
    ENCODING_SAMPLE_BYTES,
    read_document,
)
from core.text_encoding import EncodingDetectionError, decode_bytes, detect_encoding
from ingest.errors import ConnectorError
from ingest.hashing import file_fingerprint, source_key
from ingest.records import (
    ConnectorInfo,
    IngestItem,
    SourceRecord,
    SourceSpec,
)

#: Text-ish formats read through ``core.document_reader`` (Stage 1: TXT/Markdown).
READER_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
#: Tabular / line-delimited formats read and streamed here.
STREAM_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".jsonl", ".ndjson"})
ALL_SUFFIXES = READER_SUFFIXES | STREAM_SUFFIXES

#: A plain JSON document is parsed whole; above this it is refused (use JSONL).
DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024

_SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest-tmp", "node_modules"})


class FileConnector:
    """A connector over a single file or a directory of files."""

    info = ConnectorInfo(
        name="files",
        description="Local TXT/Markdown/CSV/JSON/JSONL files and directories.",
        suffixes=ALL_SUFFIXES,
        default=True,
    )

    # -- planning ---------------------------------------------------------

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        row_group = int(spec.option("row_group", 0) or 40)
        normalize = _flag(spec.option("normalize", True))
        fold_arabic = _flag(spec.option("fold_arabic", False))
        allow_ocr = _flag(spec.option("allow_ocr", False))
        encoding = spec.option("encoding") or None
        max_json_bytes = int(spec.option("max_json_bytes", 0) or DEFAULT_MAX_JSON_BYTES)
        for path in self._iter_paths(spec):
            suffix = path.suffix.lower()
            relative = self._relative(path, spec)
            yield IngestItem(
                source_key=source_key(relative),
                uri=str(path),
                title=path.stem or path.name,
                fingerprint=file_fingerprint(path),
                metadata={
                    "suffix": suffix,
                    "extension": suffix,
                    "filename": path.name,
                    "relative": relative,
                    "path": str(path),
                    "row_group": row_group,
                    "normalize": normalize,
                    "fold_arabic": fold_arabic,
                    "allow_ocr": allow_ocr,
                    "encoding": encoding,
                    "max_json_bytes": max_json_bytes,
                },
            )

    def _iter_paths(self, spec: SourceSpec) -> Iterator[Path]:
        """Yield source files lazily and deterministically (bounded memory)."""
        path = self._resolve(spec)
        if path.is_file():
            if path.suffix.lower() not in ALL_SUFFIXES:
                raise ConnectorError(
                    f"Unsupported file type {path.suffix or '(none)'!r}. "
                    f"Supported: {', '.join(sorted(ALL_SUFFIXES))}."
                )
            yield path
            return
        if path.is_dir():
            recursive = bool(spec.option("recursive", True))
            excludes = _exclude_roots(spec)
            yield from self._walk(path, recursive=recursive, excludes=excludes)
            return
        raise ConnectorError(f"No such file or directory: {path}")

    def _walk(self, root: Path, *, recursive: bool, excludes: tuple[Path, ...]) -> Iterator[Path]:
        """Depth-first walk that holds only one directory's entries at a time.

        Entries are sorted so the order is deterministic; hidden and known-noise
        directories are skipped; excluded subtrees are never entered.
        """
        try:
            entries = sorted(os.scandir(root), key=lambda entry: entry.name)
        except OSError as error:
            raise ConnectorError(f"Could not read directory {root}: {error}") from error
        for entry in entries:
            if _skipped_name(entry.name):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    if recursive and not _excluded(Path(entry.path), excludes):
                        yield from self._walk(
                            Path(entry.path), recursive=recursive, excludes=excludes
                        )
                elif entry.is_file(follow_symlinks=False):
                    candidate = Path(entry.path)
                    if (
                        candidate.suffix.lower() in ALL_SUFFIXES
                        and not _excluded(candidate, excludes)
                    ):
                        yield candidate
            except OSError:
                # A single unreadable entry must not abort the walk.
                continue

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
                encoding=metadata.get("encoding") or None,
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
                "encoding": document.encoding,
            },
        )

    def _stream_tabular(
        self, item: IngestItem, path: Path, suffix: str, metadata: dict
    ) -> Iterator[SourceRecord]:
        row_group = int(item.metadata.get("row_group", 0)) or 40
        encoding = metadata.get("encoding") or None
        title = item.title
        kind = suffix.lstrip(".")
        buffer: list[str] = []
        index = 0
        for line in _iter_rows(path, suffix, encoding=encoding, metadata=metadata):
            buffer.append(line)
            if len(buffer) >= row_group:
                index += 1
                yield SourceRecord(
                    title=f"{title} — part {index}",
                    text="\n".join(buffer),
                    source="file",
                    source_key=item.source_key,
                    metadata={**metadata, "kind": kind, "part": index},
                )
                buffer = []
        if buffer:
            index += 1
            yield SourceRecord(
                title=f"{title} — part {index}" if index > 1 else title,
                text="\n".join(buffer),
                source="file",
                source_key=item.source_key,
                metadata={**metadata, "kind": kind, "part": index},
            )

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        for item in self.plan(spec):
            yield from self.stream_item(item)


def _iter_rows(
    path: Path, suffix: str, *, encoding: str | None, metadata: dict
) -> Iterator[str]:
    """Stream one text line per row from a CSV/TSV/JSON/JSONL file."""
    if suffix in {".csv", ".tsv"}:
        yield from _iter_csv(path, delimiter="\t" if suffix == ".tsv" else ",", encoding=encoding)
        return
    if suffix in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl(path, encoding=encoding)
        return
    max_bytes = int(metadata.get("max_json_bytes") or DEFAULT_MAX_JSON_BYTES)
    yield from _iter_json(path, encoding=encoding, max_bytes=max_bytes)


def _open_text(path: Path, encoding: str | None, *, newline: str | None = None) -> io.TextIOWrapper:
    """Open a file for streaming text with a safely detected, strict encoding.

    Detection samples the head of the file; the codec is then applied strictly,
    so a mid-file decode error surfaces instead of being replaced with ``\\ufffd``.
    """
    chosen = encoding or _detect_stream_encoding(path)
    try:
        handle = path.open("rb")
    except OSError as error:
        raise ConnectorError(f"Could not open {path.name!r}: {error}") from error
    return io.TextIOWrapper(handle, encoding=chosen, errors="strict", newline=newline)


def _detect_stream_encoding(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            sample = handle.read(ENCODING_SAMPLE_BYTES)
    except OSError as error:
        raise ConnectorError(f"Could not open {path.name!r}: {error}") from error
    try:
        return detect_encoding(sample, allow_truncated=True)
    except EncodingDetectionError as error:
        raise ConnectorError(str(error)) from error


def _iter_csv(path: Path, *, delimiter: str, encoding: str | None) -> Iterator[str]:
    handle = _open_text(path, encoding, newline="")
    try:
        reader = csv.reader(handle, delimiter=delimiter, strict=True)
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
    except csv.Error as error:
        raise ConnectorError(
            f"{path.name!r} is not valid {delimiter!r}-separated data: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise ConnectorError(f"Could not decode {path.name!r}: {error}") from error
    finally:
        handle.close()


def _iter_jsonl(path: Path, encoding: str | None) -> Iterator[str]:
    handle = _open_text(path, encoding)
    try:
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
    except UnicodeDecodeError as error:
        raise ConnectorError(f"Could not decode {path.name!r}: {error}") from error
    finally:
        handle.close()


def _iter_json(path: Path, *, encoding: str | None, max_bytes: int) -> Iterator[str]:
    # A plain JSON document is parsed whole (no streaming parser in the stdlib);
    # cap it so a huge file is refused rather than pulled into memory.
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ConnectorError(f"Could not open {path.name!r}: {error}") from error
    if max_bytes > 0 and size > max_bytes:
        raise ConnectorError(
            f"{path.name!r} is {size / 1024 / 1024:.1f} MB, above the "
            f"{max_bytes / 1024 / 1024:.0f} MB JSON limit. Use JSONL (one JSON value "
            "per line) so it can be streamed, or raise max_json_bytes."
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConnectorError(f"Could not read {path.name!r}: {error}") from error
    try:
        text = decode_bytes(raw, encoding=encoding).text
    except EncodingDetectionError as error:
        raise ConnectorError(str(error)) from error
    try:
        value = json.loads(text)
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


def _skipped_name(name: str) -> bool:
    """Whether an entry name is hidden or a known-noise directory."""
    return name in _SKIP_DIRS or (name.startswith(".") and name not in (".", ".."))


def _exclude_roots(spec: SourceSpec) -> tuple[Path, ...]:
    """Directories (given by option) whose subtrees are never walked."""
    roots = spec.option("exclude_roots") or ()
    return tuple(Path(root).expanduser().resolve() for root in roots)


def _excluded(path: Path, excludes: tuple[Path, ...]) -> bool:
    if not excludes:
        return False
    resolved = path.resolve()
    return any(root == resolved or root in resolved.parents for root in excludes)


__all__ = [
    "ALL_SUFFIXES",
    "DEFAULT_MAX_JSON_BYTES",
    "FileConnector",
    "READER_SUFFIXES",
    "STREAM_SUFFIXES",
]
