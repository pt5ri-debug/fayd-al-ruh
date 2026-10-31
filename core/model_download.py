"""Fetch a model's files, safely, resumably, and behind the project's gates.

The Model Manager could describe models but not obtain them: the fetch itself
stayed with each engine's own ``download_model``, which is right for those engines
but leaves the manager unable to answer "download this" for a model it lists. This
module is that missing half, written once.

What it is careful about, in the order the audit asks:

* **The gates come first.** A download leaves the machine, so it is the
  ``model_download`` external tool: denied by default, and it runs only when the
  permission policy *and* a human approval both allow it. Nothing here reaches the
  network on its own.
* **Nothing is appended blindly.** A resume is attempted only when the server
  supports ranges (``Accept-Ranges``/``206``), the file identity matches (ETag or
  Last-Modified), and the partial's length is consistent with the total. Otherwise
  the transfer restarts, and it says so.
* **A partial file is not a model.** Data lands in a clearly-marked ``.part``
  beside a small ``.meta.json``, and it is only promoted to its final name after
  its size and (when one is known) its digest check out -- promoted with
  ``os.replace``, which is atomic on one filesystem, so a reader sees either the
  old file or the whole new one.
* **A digest is only worth what its provenance says.** The downloader computes a
  local digest always, and compares it to the expected one when the caller
  supplies it -- reporting :data:`core.model_sources.DIGEST_TRUSTED` for a digest
  the user pinned and ``DIGEST_SOURCE`` for one published beside the file, never
  conflating the two.
* **Paths are confined.** Every destination is resolved and checked to stay inside
  the download folder, so a hostile file name cannot escape it.

The transport is injectable (``opener``), so the entire module is tested against a
deterministic fake with no network, no server, and no large file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.model_sources import (
    DIGEST_NONE,
    DIGEST_SOURCE,
    DIGEST_TRUSTED,
    ModelFile,
    validate_download_url,
)

#: The suffix a partial download carries. A ``.part`` file is never a model, and
#: every surface says so rather than offering it for inference.
PART_SUFFIX = ".part"

#: The sidecar that makes a resume safe: what the partial is, and whether the
#: source still offers the same file.
META_SUFFIX = ".meta.json"

#: The tool name the permission policy and the audit see.
DOWNLOAD_TOOL = "model_download"

#: Bounded, so a long transfer cannot grow without limit.
DEFAULT_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 1.5
DEFAULT_TIMEOUT = 60.0

#: How much free space to insist on beyond the file itself, so a download cannot
#: fill the disk and take the machine with it.
DISK_HEADROOM_BYTES = 64 * 1024 * 1024

#: What happened to one download, in one word.
STATUS_COMPLETE = "complete"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"
STATUS_RESUMED = "resumed"


class DownloadError(RuntimeError):
    """A download could not be completed. Always carries the real next step."""


class DownloadDenied(DownloadError):
    """A permission or approval gate refused the download."""


class DownloadCancelled(DownloadError):
    """The caller cancelled the download. The partial file is left resumable."""


class _RangeIgnored(DownloadError):
    """The server accepted the range header but answered with the whole file.

    Not a retryable failure: the partial cannot be trusted, so the transfer is
    restarted cleanly once rather than appended to.
    """


class InsufficientSpace(DownloadError):
    """There is not enough free disk space to attempt the download."""


class Transport(Protocol):
    """What a download needs from a network client.

    A deliberately small surface so the whole module is exercised with a fake:
    one call that returns headers without a body, and one that streams a body.
    """

    def head(self, url: str, *, timeout: float) -> Response: ...

    def open(self, url: str, *, headers: dict[str, str], timeout: float) -> Response: ...


@dataclass(frozen=True)
class Response:
    """A transport response, reduced to what a download actually uses."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    url: str = ""

    def header(self, name: str) -> str:
        """One header, case-insensitively, or ``""``."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return str(value)
        return ""

    @property
    def total_bytes(self) -> int | None:
        """The declared total size, from ``Content-Range`` or ``Content-Length``."""
        content_range = self.header("content-range")
        if content_range and "/" in content_range:
            tail = content_range.rsplit("/", 1)[-1].strip()
            if tail.isdigit():
                return int(tail)
        length = self.header("content-length")
        if length.isdigit():
            return int(length)
        return None

    @property
    def accepts_ranges(self) -> bool:
        return self.header("accept-ranges").strip().lower() == "bytes"

    @property
    def etag(self) -> str:
        return self.header("etag").strip()

    @property
    def last_modified(self) -> str:
        return self.header("last-modified").strip()


class UrllibTransport:
    """The real transport, on the standard library -- no new dependency.

    HTTPS only, certificate verification always on (``urllib``'s default), and a
    timeout on every call. It never disables verification to work around a network
    problem, and it never logs a header.
    """

    def head(self, url: str, *, timeout: float) -> Response:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return Response(
                    status=int(getattr(handle, "status", 200) or 200),
                    headers=dict(handle.headers.items()),
                    url=str(handle.geturl()),
                )
        except urllib.error.HTTPError as error:
            return Response(
                status=int(error.code),
                headers=dict((error.headers or {}).items()),
                url=url,
            )

    def open(self, url: str, *, headers: dict[str, str], timeout: float) -> Response:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, headers=dict(headers))
        try:
            handle = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            return Response(
                status=int(error.code),
                headers=dict((error.headers or {}).items()),
                url=url,
            )
        return Response(
            status=int(getattr(handle, "status", 200) or 200),
            headers=dict(handle.headers.items()),
            body=handle,
            url=str(handle.geturl()),
        )


# ------------------------------------------------------------------- results


@dataclass(frozen=True)
class FileResult:
    """What happened to one file: where it landed, and how far it can be trusted."""

    filename: str
    path: Path | None = None
    ok: bool = False
    bytes_written: int = 0
    resumed: bool = False
    digest: str = ""
    digest_algorithm: str = ""
    #: How strong the digest check was (see ``core.model_sources.DIGEST_*``).
    digest_strength: str = DIGEST_NONE
    #: ``True`` only when the digest matched an expected value.
    verified: bool = False
    detail: str = ""
    #: Left in place, and safe to resume, when a transfer was cancelled.
    partial_path: Path | None = None

    def describe(self) -> str:
        head = "ok" if self.ok else "failed"
        bits = [f"{self.filename}: {head}", f"{self.bytes_written} byte(s)"]
        if self.resumed:
            bits.append("resumed")
        if self.digest:
            bits.append(f"{self.digest_algorithm}:{self.digest[:12]}…")
        bits.append(f"integrity {self.digest_strength}")
        line = " · ".join(bits)
        return f"{line} — {self.detail}" if self.detail else line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "bytes_written": self.bytes_written,
            "detail": self.detail,
            "digest": self.digest,
            "digest_algorithm": self.digest_algorithm,
            "digest_strength": self.digest_strength,
            "filename": self.filename,
            "ok": self.ok,
            "path": str(self.path) if self.path is not None else "",
            "partial_path": str(self.partial_path) if self.partial_path is not None else "",
            "resumed": self.resumed,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class DownloadReport:
    """The whole outcome of one model download, in a shape a surface can post."""

    source_id: str
    destination: Path
    status: str = STATUS_FAILED
    files: tuple[FileResult, ...] = field(default=())
    bytes_total: int = 0
    seconds: float = 0.0
    detail: str = ""
    #: ``True`` only when **every** required file was promoted into place.
    complete: bool = False
    #: Files that a later run may safely resume, when a transfer was interrupted.
    resumable: tuple[Path, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.complete

    def describe(self) -> str:
        head = {
            STATUS_COMPLETE: "downloaded",
            STATUS_RESUMED: "downloaded (resumed)",
            STATUS_CANCELLED: "cancelled",
            STATUS_FAILED: "failed",
        }.get(self.status, self.status)
        lines = [
            f"{head}: {self.source_id}",
            f"  destination: {self.destination}",
            f"  {self.bytes_total} byte(s) in {self.seconds:.2f}s",
        ]
        lines.extend(f"  {item.describe()}" for item in self.files)
        if self.resumable:
            lines.append("  resumable partial file(s):")
            lines.extend(f"    {path}" for path in self.resumable)
        if self.detail:
            lines.append(f"  {self.detail}")
        if not self.complete and self.status != STATUS_COMPLETE:
            lines.append("  Nothing was registered: an incomplete download is not a model.")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "bytes_total": self.bytes_total,
            "complete": self.complete,
            "destination": str(self.destination),
            "detail": self.detail,
            "files": [item.to_primitive() for item in self.files],
            "resumable": [str(path) for path in self.resumable],
            "seconds": self.seconds,
            "source_id": self.source_id,
            "status": self.status,
        }


# ----------------------------------------------------------------- progress


@dataclass
class DownloadProgress:
    """Byte-level progress when the total is known, indeterminate when it is not.

    A caller renders this; the downloader never assumes a total it was not given,
    so a server that omits ``Content-Length`` produces honest indeterminate
    progress instead of a bar that lies.
    """

    filename: str = ""
    bytes_done: int = 0
    bytes_total: int | None = None
    resumed_from: int = 0
    file_index: int = 0
    file_count: int = 1

    @property
    def measurable(self) -> bool:
        return bool(self.bytes_total)

    @property
    def fraction(self) -> float | None:
        if not self.bytes_total:
            return None
        return min(1.0, self.bytes_done / self.bytes_total)

    def describe(self) -> str:
        where = f"[{self.file_index}/{self.file_count}] {self.filename}"
        if self.measurable:
            percent = 100.0 * (self.fraction or 0.0)
            return (
                f"{where}: {percent:.1f}% "
                f"({self.bytes_done}/{self.bytes_total} bytes)"
            )
        return f"{where}: {self.bytes_done} bytes (total unknown)"


# ------------------------------------------------------------- partial state


@dataclass(frozen=True)
class PartialState:
    """What a ``.meta.json`` sidecar records, so a resume can be judged safe."""

    url: str
    filename: str
    size_bytes: int | None = None
    etag: str = ""
    last_modified: str = ""
    bytes_written: int = 0
    started_at: str = ""

    def to_primitive(self) -> dict[str, Any]:
        return {
            "bytes_written": self.bytes_written,
            "etag": self.etag,
            "filename": self.filename,
            "last_modified": self.last_modified,
            "size_bytes": self.size_bytes,
            "started_at": self.started_at,
            "url": self.url,
        }


def partial_path(destination: Path, filename: str) -> Path:
    """Where a file's partial data lives while it is being fetched."""
    return destination / f"{_safe_name(filename)}{PART_SUFFIX}"


def meta_path(destination: Path, filename: str) -> Path:
    """Where a partial's sidecar lives."""
    return destination / f"{_safe_name(filename)}{META_SUFFIX}"


def load_partial_state(destination: Path, filename: str) -> PartialState | None:
    """The stored partial state, or ``None`` when there is none to read.

    A corrupt sidecar reads as ``None`` rather than raising: the worst case is
    that a resume is not attempted, which is safe.
    """
    path = meta_path(destination, filename)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    size = raw.get("size_bytes")
    return PartialState(
        url=str(raw.get("url") or ""),
        filename=str(raw.get("filename") or filename),
        size_bytes=int(size) if isinstance(size, int) else None,
        etag=str(raw.get("etag") or ""),
        last_modified=str(raw.get("last_modified") or ""),
        bytes_written=int(raw.get("bytes_written") or 0),
        started_at=str(raw.get("started_at") or ""),
    )


def save_partial_state(destination: Path, state: PartialState) -> None:
    """Write the sidecar atomically, so a crash cannot leave it half-written."""
    path = meta_path(destination, state.filename)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(state.to_primitive(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp, path)
    except OSError:
        # A sidecar that cannot be written only costs a future resume, so it is
        # not worth failing the download over.
        return


def clear_partial_state(destination: Path, filename: str) -> None:
    """Remove a partial's data and sidecar, ignoring anything already gone."""
    for path in (partial_path(destination, filename), meta_path(destination, filename)):
        try:
            path.unlink()
        except OSError:
            continue


def resumable_files(destination: Path) -> tuple[Path, ...]:
    """Every partial file in ``destination`` that a later run could resume."""
    try:
        return tuple(
            sorted(
                path
                for path in destination.iterdir()
                if path.is_file() and path.name.endswith(PART_SUFFIX)
            )
        )
    except OSError:
        return ()


# ------------------------------------------------------------------- policy


def _gate(
    *,
    approved: bool,
    permissions: Any = None,
    approvals: Any = None,
) -> None:
    """Raise unless the permission policy *and* a human approval both allow it.

    The same shape every governed external action in this project uses: denied by
    default, and ``approved=True`` is the CLI's convenience that records a human
    approval rather than bypassing the gate.
    """
    from core.human_approval import HumanApprovalGate
    from core.permissions import Capability, PermissionPolicy

    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({DOWNLOAD_TOOL}),
        )
        approvals = approvals or _approve(DOWNLOAD_TOOL)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, DOWNLOAD_TOOL)
    if not decision.allowed:
        raise DownloadDenied(f"Denied by policy ({decision.reason}).")
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise DownloadDenied(
            f"Needs human approval ({approval.reason}). Re-run with --approve."
        )


def _approve(tool: str) -> Any:
    from core.human_approval import HumanApprovalGate

    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="model-manager")
    request.approve("user")
    return gate


# -------------------------------------------------------------- path safety


def safe_destination(destination: Path | str, filename: str) -> Path:
    """``destination/filename``, refused when it would leave ``destination``.

    The one guard against a path-traversal file name (``../../etc/passwd``, an
    absolute path, a Windows drive) arriving from untrusted metadata. A refusal is
    a :class:`DownloadError` with the reason, never a silent skip.
    """
    root = Path(destination).expanduser()
    name = str(filename or "").strip().replace("\\", "/")
    if not name:
        raise DownloadError("A download needs a file name.")
    if name.startswith("/") or ":" in name.split("/")[0]:
        raise DownloadError(f"The file name {filename!r} is an absolute path, which is refused.")
    candidate = (root / name).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise DownloadError(
            f"The file name {filename!r} would write outside {resolved_root}."
        ) from None
    return candidate


def _safe_name(filename: str) -> str:
    """A file name safe to use for the partial and its sidecar."""
    name = str(filename or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name or "model"


def check_space(destination: Path, needed: int | None, *, headroom: int = DISK_HEADROOM_BYTES) -> None:
    """Raise :class:`InsufficientSpace` when the volume cannot hold the download.

    ``needed=None`` (an unknown size) is *not* an error -- the download proceeds
    and the honest indeterminate progress is shown. A *known* size that does not
    fit is refused **before** anything is written.
    """
    if needed is None or needed <= 0:
        return
    import shutil

    try:
        destination.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(destination).free
    except OSError:
        return
    if free < needed + headroom:
        raise InsufficientSpace(
            f"The download needs about {needed / 1e9:.2f} GB plus headroom, but only "
            f"{free / 1e9:.2f} GB is free at {destination}."
        )


# ------------------------------------------------------------------ download


@dataclass
class DownloadRequest:
    """Everything one download needs, so the downloader takes no hidden state."""

    destination: Path
    files: tuple[ModelFile, ...]
    source_id: str = ""
    #: Digests the *user* pinned, by file name. These are the only ones that earn
    #: :data:`core.model_sources.DIGEST_TRUSTED`.
    pinned_digests: dict[str, str] = field(default_factory=dict)
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    timeout: float = DEFAULT_TIMEOUT
    #: Restart from scratch even when a safe resume looks possible.
    restart: bool = False


class ModelDownloader:
    """Fetch model files, with the safety rules this module's docstring states.

    ``transport`` is the seam: the default is the real
    :class:`UrllibTransport`, and a test injects a fake. ``approved`` is the
    human-approval convenience; a caller that wants to supply its own gates passes
    ``permissions``/``approvals`` instead.
    """

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        approved: bool = False,
        permissions: Any = None,
        approvals: Any = None,
    ) -> None:
        self._transport = transport or UrllibTransport()
        self._approved = approved
        self._permissions = permissions
        self._approvals = approvals

    # --- public --------------------------------------------------------

    def download(
        self,
        request: DownloadRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[DownloadProgress], None] | None = None,
    ) -> DownloadReport:
        """Fetch every file in ``request``, promoting each only once it verifies.

        Never raises for a network or integrity problem: the outcome is a
        :class:`DownloadReport` whose ``complete`` is ``False`` and whose ``detail``
        says what to do next. The two exceptions that *are* raised are the gates
        (:class:`DownloadDenied`, before any network call) and an invalid request
        (:class:`DownloadError`), because neither is a transfer outcome.
        """
        _gate(
            approved=self._approved,
            permissions=self._permissions,
            approvals=self._approvals,
        )
        started = time.monotonic()
        destination = Path(request.destination).expanduser()
        destination.mkdir(parents=True, exist_ok=True)

        results: list[FileResult] = []
        resumable: list[Path] = []
        total_written = 0
        resumed_any = False
        cancelled = False

        for index, item in enumerate(request.files, start=1):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            try:
                result = self._fetch_one(
                    destination,
                    item,
                    index=index,
                    count=len(request.files),
                    request=request,
                    should_cancel=should_cancel,
                    on_progress=on_progress,
                )
            except DownloadCancelled:
                partial = partial_path(destination, item.filename)
                cancelled = True
                results.append(
                    FileResult(
                        filename=item.filename,
                        ok=False,
                        bytes_written=_size_of(partial),
                        partial_path=partial,
                        detail="Cancelled; the partial file is kept so a later run can resume it.",
                    )
                )
                if partial.exists():
                    resumable.append(partial)
                break
            except InsufficientSpace as error:
                return DownloadReport(
                    source_id=request.source_id,
                    destination=destination,
                    status=STATUS_FAILED,
                    files=tuple(results),
                    bytes_total=total_written,
                    seconds=time.monotonic() - started,
                    detail=str(error),
                )
            except DownloadError as error:
                results.append(
                    FileResult(filename=item.filename, ok=False, detail=str(error))
                )
                continue
            results.append(result)
            total_written += result.bytes_written
            resumed_any = resumed_any or result.resumed

        complete = bool(results) and all(result.ok for result in results) and not cancelled
        if cancelled:
            status = STATUS_CANCELLED
        elif complete:
            status = STATUS_RESUMED if resumed_any else STATUS_COMPLETE
        else:
            status = STATUS_FAILED
        detail = ""
        if not complete and not cancelled:
            failed = [result.filename for result in results if not result.ok]
            missing = len(request.files) - len(results)
            bits = []
            if failed:
                bits.append(f"failed: {', '.join(failed)}")
            if missing:
                bits.append(f"{missing} file(s) were not attempted")
            detail = "; ".join(bits) or "the download did not complete"
            # Surface the first real reason, so a caller does not have to dig.
            first_reason = next(
                (result.detail for result in results if not result.ok and result.detail), ""
            )
            if first_reason:
                detail = f"{detail} — {first_reason}"
        return DownloadReport(
            source_id=request.source_id,
            destination=destination,
            status=status,
            files=tuple(results),
            bytes_total=total_written,
            seconds=time.monotonic() - started,
            detail=detail,
            complete=complete,
            resumable=tuple(resumable),
        )

    def head(self, url: str, *, timeout: float | None = None) -> Response:
        """Ask a URL what it offers, without downloading it.

        A ``HEAD`` is not available everywhere, so a 405/501 falls back to a
        ranged ``GET`` for the first byte, which is the standard way to read a
        file's identity without fetching it.
        """
        problem = validate_download_url(url)
        if problem:
            raise DownloadError(problem)
        wait = timeout or DEFAULT_TIMEOUT
        response = self._transport.head(url, timeout=wait)
        if response.status in (405, 501) or (response.status == 200 and not response.headers):
            response = self._transport.open(
                url, headers={"Range": "bytes=0-0"}, timeout=wait
            )
            _close(response)
        if response.status >= 400:
            raise DownloadError(f"The source answered HTTP {response.status} for {url}.")
        return response

    # --- one file ------------------------------------------------------

    def _fetch_one(
        self,
        destination: Path,
        item: ModelFile,
        *,
        index: int,
        count: int,
        request: DownloadRequest,
        should_cancel: Callable[[], bool] | None,
        on_progress: Callable[[DownloadProgress], None] | None,
    ) -> FileResult:
        target = safe_destination(destination, item.filename)
        url = item.url
        if not url:
            raise DownloadError(f"No URL was given for {item.filename!r}.")
        problem = validate_download_url(url)
        if problem:
            raise DownloadError(problem)

        part = partial_path(destination, item.filename)
        meta = meta_path(destination, item.filename)
        if request.restart:
            clear_partial_state(destination, item.filename)

        # What the source says about the file -- used for identity and the size
        # check, and never trusted as authenticity on its own.
        remote = self.head(url, timeout=request.timeout)
        remote_total = remote.total_bytes
        etag = remote.etag
        last_modified = remote.last_modified
        accepts_ranges = remote.accepts_ranges or remote.status == 206

        expected_size = item.size_bytes if item.size_bytes is not None else remote_total
        check_space(destination, expected_size)

        resume_from = 0
        state = load_partial_state(destination, item.filename)
        if part.is_file() and state is not None and not request.restart:
            resume_from = self._safe_resume_offset(
                part, state, url=url, etag=etag, last_modified=last_modified,
                total=expected_size, accepts_ranges=accepts_ranges,
            )
            if resume_from == 0:
                # A resume was not safe: restart cleanly rather than appending.
                clear_partial_state(destination, item.filename)
        elif part.is_file() and not request.restart:
            # A partial with no sidecar cannot be judged, so it is not trusted.
            clear_partial_state(destination, item.filename)

        resumed = resume_from > 0
        headers: dict[str, str] = {}
        if resumed:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            written = self._stream(
                url,
                part,
                headers=headers,
                start_offset=resume_from,
                expected_total=expected_size,
                filename=item.filename,
                index=index,
                count=count,
                request=request,
                should_cancel=should_cancel,
                on_progress=on_progress,
                etag=etag,
                last_modified=last_modified,
            )
        except _RangeIgnored:
            # The server would not honour a range, so the partial is unusable.
            # Restart the transfer cleanly -- once, not in a loop.
            clear_partial_state(destination, item.filename)
            resumed = False
            written = self._stream(
                url,
                part,
                headers={},
                start_offset=0,
                expected_total=expected_size,
                filename=item.filename,
                index=index,
                count=count,
                request=request,
                should_cancel=should_cancel,
                on_progress=on_progress,
                etag=etag,
                last_modified=last_modified,
            )

        # --- verify before promoting -----------------------------------
        size = _size_of(part)
        if expected_size is not None and size != expected_size:
            raise DownloadError(
                f"{item.filename} is {size} bytes but {expected_size} were expected. "
                "The partial file was left in place; re-run to try again."
            )
        digest, algorithm, strength, verified = self._verify(item, part, request)
        if strength == DIGEST_TRUSTED and not verified:
            # A pinned digest that does not match is the one case where the bytes
            # must not be promoted: the user told us what they expect.
            raise DownloadError(
                f"{item.filename} does not match the digest you pinned. The file was "
                "not installed. Check the revision you asked for."
            )
        if strength == DIGEST_SOURCE and not verified:
            raise DownloadError(
                f"{item.filename} does not match the digest the source published. "
                "The file was not installed."
            )

        _promote(part, target)
        try:
            meta.unlink()
        except OSError:
            pass
        return FileResult(
            filename=item.filename,
            path=target,
            ok=True,
            bytes_written=written,
            resumed=resumed,
            digest=digest,
            digest_algorithm=algorithm,
            digest_strength=strength,
            verified=verified,
            detail="" if verified else "installed; no trusted digest was available to compare",
        )

    def _safe_resume_offset(
        self,
        part: Path,
        state: PartialState,
        *,
        url: str,
        etag: str,
        last_modified: str,
        total: int | None,
        accepts_ranges: bool,
    ) -> int:
        """How many bytes may be kept, or ``0`` when a resume is not safe.

        Every condition the audit lists is checked here: the server must support
        ranges, the URL must be the same, the file identity (ETag, else
        Last-Modified) must be unchanged, the partial must be non-empty, and its
        length must be *consistent* with the total -- never larger, and never
        larger than the recorded progress.
        """
        if not accepts_ranges:
            return 0
        if state.url and state.url != url:
            return 0
        if state.etag and etag and state.etag != etag:
            return 0
        if not state.etag and state.last_modified and last_modified:
            if state.last_modified != last_modified:
                return 0
        local = _size_of(part)
        if local <= 0:
            return 0
        if total is not None and local >= total:
            # Already complete, or longer than the source now says: do not append.
            return 0
        if state.bytes_written and local > state.bytes_written:
            return 0
        return local

    def _stream(
        self,
        url: str,
        part: Path,
        *,
        headers: dict[str, str],
        start_offset: int,
        expected_total: int | None,
        filename: str,
        index: int,
        count: int,
        request: DownloadRequest,
        should_cancel: Callable[[], bool] | None,
        on_progress: Callable[[DownloadProgress], None] | None,
        etag: str,
        last_modified: str,
    ) -> int:
        """Stream the body to ``part`` with bounded retries. Returns bytes written.

        A transient failure is retried with backoff, and each retry re-reads what
        is on disk so it resumes from the right offset. The loop is bounded by
        ``max_attempts`` -- it never retries forever.
        """
        attempt = 0
        last_error = ""
        while attempt < max(1, request.max_attempts):
            attempt += 1
            try:
                return self._attempt(
                    url,
                    part,
                    headers=headers,
                    start_offset=start_offset,
                    expected_total=expected_total,
                    filename=filename,
                    index=index,
                    count=count,
                    request=request,
                    should_cancel=should_cancel,
                    on_progress=on_progress,
                    etag=etag,
                    last_modified=last_modified,
                )
            except DownloadCancelled:
                raise
            except _RangeIgnored:
                # Not retryable: the caller restarts cleanly instead.
                raise
            except DownloadError as error:
                last_error = str(error)
                if attempt >= max(1, request.max_attempts):
                    break
                time.sleep(min(DEFAULT_RETRY_BACKOFF ** attempt, 8.0))
                # Resume from whatever actually landed, so a retry is not a restart.
                start_offset = _size_of(part)
                headers = {"Range": f"bytes={start_offset}-"} if start_offset else {}
        raise DownloadError(
            f"{filename} could not be downloaded after {attempt} attempt(s): {last_error}"
        )

    def _attempt(
        self,
        url: str,
        part: Path,
        *,
        headers: dict[str, str],
        start_offset: int,
        expected_total: int | None,
        filename: str,
        index: int,
        count: int,
        request: DownloadRequest,
        should_cancel: Callable[[], bool] | None,
        on_progress: Callable[[DownloadProgress], None] | None,
        etag: str,
        last_modified: str,
    ) -> int:
        """One streaming attempt. Appends only at the exact expected offset."""
        response = self._transport.open(url, headers=headers, timeout=request.timeout)
        try:
            if start_offset and response.status != 206:
                raise _RangeIgnored(
                    "The source ignored the range request, so the partial file "
                    "cannot be resumed safely. Restarting."
                )
            if response.status >= 400:
                raise DownloadError(f"The source answered HTTP {response.status}.")
            if response.status == 206:
                expected_range = f"bytes {start_offset}-"
                content_range = response.header("content-range")
                if content_range and not content_range.startswith(expected_range):
                    raise DownloadError(
                        f"The source's range response ({content_range}) does not start "
                        f"at {start_offset}, so the partial cannot be trusted."
                    )
            total = expected_total if expected_total is not None else response.total_bytes
            mode = "ab" if start_offset else "wb"
            written = start_offset
            if on_progress is not None:
                on_progress(
                    DownloadProgress(
                        filename=filename,
                        bytes_done=written,
                        bytes_total=total,
                        resumed_from=start_offset,
                        file_index=index,
                        file_count=count,
                    )
                )
            with part.open(mode) as handle:
                cursor = _as_cursor(response.body)
                while True:
                    if should_cancel is not None and should_cancel():
                        raise DownloadCancelled(filename)
                    chunk = _read_chunk(cursor, request.chunk_bytes)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if on_progress is not None:
                        on_progress(
                            DownloadProgress(
                                filename=filename,
                                bytes_done=written,
                                bytes_total=total,
                                resumed_from=start_offset,
                                file_index=index,
                                file_count=count,
                            )
                        )
            if total is not None and written < total:
                # The body ended early. That is a transient failure worth a
                # retry, and the retry resumes from what really landed.
                raise DownloadError(
                    f"the transfer ended at {written} of {total} bytes"
                )
            # Record the identity of what we fetched, so a later resume can tell
            # whether the source changed under us.
            save_partial_state(
                part.parent,
                PartialState(
                    url=url,
                    filename=filename,
                    size_bytes=total,
                    etag=etag,
                    last_modified=last_modified,
                    bytes_written=written,
                    started_at=_now(),
                ),
            )
            return written
        finally:
            _close(response)

    def _verify(
        self,
        item: ModelFile,
        part: Path,
        request: DownloadRequest,
    ) -> tuple[str, str, str, bool]:
        """Digest the downloaded bytes and decide how strong that check was.

        Returns ``(digest, algorithm, strength, verified)``. The strength is the
        *provenance* of the expected digest, never its mere presence: a pinned
        digest is trusted, a digest published beside the file is only the source's
        word, and no expected digest means the local hash identifies the file but
        proves nothing about who made it.
        """
        algorithm = "sha256"
        digest = hash_file(part, algorithm=algorithm)
        pinned = request.pinned_digests.get(item.filename) or request.pinned_digests.get("*")
        if pinned:
            return digest, algorithm, DIGEST_TRUSTED, _digest_matches(digest, pinned)
        if item.expected_digest:
            return (
                digest,
                item.digest_algorithm or algorithm,
                DIGEST_SOURCE,
                _digest_matches(digest, item.expected_digest),
            )
        return digest, algorithm, DIGEST_NONE, False


# ------------------------------------------------------------------ helpers


def _read_chunk(body: Any, size: int) -> bytes:
    """One bounded chunk from a response body, or ``b""`` at the end.

    A ``_Cursor`` (what :func:`_as_cursor` returns) advances on every call, so the
    stream loop always terminates -- whether the body was a real HTTP response or a
    plain ``bytes`` object in a test.
    """
    if body is None:
        return b""
    if hasattr(body, "read_chunk"):
        return bytes(body.read_chunk(size) or b"")
    if hasattr(body, "read"):
        try:
            return bytes(body.read(size) or b"")
        except OSError as error:
            raise DownloadError(f"The transfer failed: {error}") from error
    return b""


class _Cursor:
    """A cursor over a byte body, so a plain ``bytes`` body can be read in chunks."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read_chunk(self, size: int) -> bytes:
        block = self._data[self._offset:self._offset + size]
        self._offset += len(block)
        return block


def _as_cursor(body: Any) -> Any:
    """A body a stream can read in bounded chunks, advancing on each read."""
    if body is None or isinstance(body, (bytes, bytearray)):
        return _Cursor(bytes(body or b""))
    return body


def _close(response: Response) -> None:
    """Close a response body if it can be closed, ignoring anything else."""
    body = response.body
    closer = getattr(body, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _promote(part: Path, target: Path) -> None:
    """Move a verified partial onto its final name, atomically on one filesystem."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, target)


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def hash_file(path: Path | str, *, algorithm: str = "sha256", chunk: int = 1024 * 1024) -> str:
    """The hex digest of a file, read in bounded chunks so a big model fits.

    Used both for verification and for identifying a local file; the caller is the
    one that decides what the digest *means* (see
    :data:`core.model_sources.DIGEST_STRENGTH`).
    """
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _digest_matches(actual: str, expected: str) -> bool:
    """Compare two hex digests, tolerating an ``algo:`` prefix and case."""
    want = str(expected or "").strip().lower()
    if ":" in want:
        want = want.split(":", 1)[1]
    return bool(want) and actual.strip().lower() == want


def normalize_digest(text: str) -> tuple[str, str]:
    """Split ``sha256:abc…`` into ``("sha256", "abc…")``; a bare digest is sha256."""
    raw = str(text or "").strip()
    if ":" in raw:
        algorithm, _, value = raw.partition(":")
        algorithm = algorithm.strip().lower() or "sha256"
        if algorithm in hashlib.algorithms_available:
            return algorithm, value.strip()
        return "sha256", raw
    return "sha256", raw


def parse_pinned_digests(items: object) -> dict[str, str]:
    """Read ``name=digest`` / ``name=algo:digest`` pairs into a mapping.

    A malformed pair raises :class:`DownloadError` naming the offending text,
    rather than being silently dropped -- a mistyped pinned digest that did
    nothing would be worse than a refusal.
    """
    result: dict[str, str] = {}
    if items is None:
        return result
    if isinstance(items, str):
        values: list[object] = [items]
    elif isinstance(items, (list, tuple, set, frozenset)):
        values = list(items)
    else:
        values = [items]
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise DownloadError(
                f"A pinned digest is written NAME=DIGEST (for example "
                f"`model.gguf=sha256:abc…`), not {text!r}."
            )
        name, _, digest = text.partition("=")
        name = name.strip()
        digest = digest.strip()
        if not name or not digest:
            raise DownloadError(f"A pinned digest needs both a name and a digest: {text!r}.")
        result[name] = digest
    return result


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT",
    "DISK_HEADROOM_BYTES",
    "DOWNLOAD_TOOL",
    "META_SUFFIX",
    "PART_SUFFIX",
    "STATUS_CANCELLED",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_RESUMED",
    "DownloadCancelled",
    "DownloadDenied",
    "DownloadError",
    "DownloadProgress",
    "DownloadReport",
    "DownloadRequest",
    "FileResult",
    "InsufficientSpace",
    "ModelDownloader",
    "PartialState",
    "Response",
    "Transport",
    "UrllibTransport",
    "check_space",
    "clear_partial_state",
    "hash_file",
    "load_partial_state",
    "meta_path",
    "normalize_digest",
    "parse_pinned_digests",
    "partial_path",
    "resumable_files",
    "safe_destination",
    "save_partial_state",
]
