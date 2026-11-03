"""Resource limits, cancellation, and temp cleanup for perception (section 13).

Perception runs on a modest machine (the target is a Windows 11 laptop with 16 GB
of RAM and integrated graphics), so every expensive step is bounded and every
long step is cancellable. This module is the one place those knobs live, so no
reader invents its own.

Three things it provides:

* :class:`PerceptionLimits` -- a validated, bounded set of caps (max file size,
  max characters, max pages, max audio seconds, an optional wall-clock timeout)
  that a caller can lower but not raise past a hard ceiling. A nonsensical value
  fails loudly at construction rather than doing nothing at run time.
* :func:`cancelled` / :class:`CancelToken` -- a cooperative cancellation signal,
  checked between pages/records so a long read stops promptly without corrupting
  anything. It never *interrupts* a reader mid-write; it is polled at safe points.
* :class:`TempWorkspace` -- a temporary directory that is always removed, even
  when the body raises, so a rasterised page or a rendered WAV never litters the
  disk. The project's own rule: a temp file is measured against a byte budget
  before it is written (see :func:`core.pdf_ocr.check_raster_limits`), and this
  wrapper makes the cleanup unconditional.

Nothing here imports a third-party package or loads a model.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

#: The hard ceiling a caller cannot raise a limit past, in bytes. A single source
#: larger than this is refused rather than streamed, because a perception result
#: is held in memory as text.
MAX_SOURCE_BYTES = 512 * 1024 * 1024

#: The hard ceiling on characters kept from one source, mirroring the readers'
#: own caps so the layers agree.
MAX_CONTENT_CHARS = 4_000_000

#: The hard ceiling on pages rendered/OCR'd from one document.
MAX_PAGES_CEILING = 5_000

#: The hard ceiling on seconds of audio processed from one recording.
MAX_AUDIO_SECONDS_CEILING = 24 * 60 * 60

#: The default wall-clock timeout for one perception operation (seconds). ``0``
#: means "no timeout"; the default is generous because a CPU OCR run is slow, but
#: it is finite so a pathological source cannot hang a batch forever.
DEFAULT_TIMEOUT_SECONDS = 1800.0


class PerceptionTimeout(RuntimeError):
    """A perception operation exceeded its wall-clock budget."""


@dataclass(frozen=True)
class PerceptionLimits:
    """Validated caps for one perception operation.

    Every field has a conservative default, and each is checked against a hard
    ceiling on construction: a caller may *lower* a limit freely but cannot raise
    one past the ceiling, so a bad configuration cannot make perception unbounded.
    """

    max_source_bytes: int = 256 * 1024 * 1024
    max_content_chars: int = MAX_CONTENT_CHARS
    max_pages: int = 1_000
    max_audio_seconds: float = 2 * 60 * 60
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    keep_temp: bool = False
    """When true, the temp workspace is left on disk for inspection (debugging)."""

    def __post_init__(self) -> None:
        if not 1 <= self.max_source_bytes <= MAX_SOURCE_BYTES:
            raise ValueError(f"max_source_bytes must be in [1, {MAX_SOURCE_BYTES}].")
        if not 1 <= self.max_content_chars <= MAX_CONTENT_CHARS:
            raise ValueError(f"max_content_chars must be in [1, {MAX_CONTENT_CHARS}].")
        if not 1 <= self.max_pages <= MAX_PAGES_CEILING:
            raise ValueError(f"max_pages must be in [1, {MAX_PAGES_CEILING}].")
        if not 0 < self.max_audio_seconds <= MAX_AUDIO_SECONDS_CEILING:
            raise ValueError(f"max_audio_seconds must be in (0, {MAX_AUDIO_SECONDS_CEILING}].")
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative.")

    def deadline(self) -> float | None:
        """An absolute monotonic deadline, or ``None`` when no timeout is set."""
        if self.timeout_seconds <= 0:
            return None
        return time.monotonic() + self.timeout_seconds

    def check_timeout(self, deadline: float | None) -> None:
        """Raise :class:`PerceptionTimeout` when ``deadline`` has passed."""
        if deadline is not None and time.monotonic() > deadline:
            raise PerceptionTimeout(
                f"Perception exceeded its {self.timeout_seconds:.0f}s time budget."
            )


class CancelToken:
    """A cooperative cancellation flag, shared across a run's steps.

    A caller flips it (from a UI button, a signal handler, a parent task); a
    perception loop polls :meth:`cancelled` between pages or records and stops. It
    is deliberately not a thread interrupt -- stopping at a safe point is what
    keeps a partially written store consistent.
    """

    def __init__(self, is_cancelled: Callable[[], bool] | None = None) -> None:
        self._is_cancelled = is_cancelled or (lambda: False)

    def cancel(self) -> None:  # pragma: no cover - a caller-driven flip, no state to test
        """Mark this token cancelled (only meaningful for the built-in flag)."""
        self._is_cancelled = lambda: True

    def cancelled(self) -> bool:
        return bool(self._is_cancelled())


def cancelled(token: CancelToken | None) -> bool:
    """Whether ``token`` is set; ``None`` means "never cancelled"."""
    return token is not None and token.cancelled()


@contextmanager
def temp_workspace(
    *,
    limits: PerceptionLimits | None = None,
    prefix: str = "fayd-perception-",
) -> Iterator[Path]:
    """Yield a temporary directory that is removed afterwards, even on error.

    When ``limits.keep_temp`` is set the directory is left in place and its path
    is printed, so a debugging run can inspect a rasterised page; otherwise it is
    removed unconditionally in the ``finally``.
    """
    settings = limits or PerceptionLimits()
    folder = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield folder
    finally:
        if not settings.keep_temp:
            shutil.rmtree(folder, ignore_errors=True)


def size_within(path: Path, limit: int) -> bool:
    """Whether ``path``'s size is within ``limit`` bytes (a missing file is fine).

    Used before a read so a too-large source is refused up front instead of being
    loaded and then rejected.
    """
    try:
        return path.stat().st_size <= limit
    except OSError:
        return True


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_AUDIO_SECONDS_CEILING",
    "MAX_CONTENT_CHARS",
    "MAX_PAGES_CEILING",
    "MAX_SOURCE_BYTES",
    "CancelToken",
    "PerceptionLimits",
    "PerceptionTimeout",
    "cancelled",
    "size_within",
    "temp_workspace",
]
