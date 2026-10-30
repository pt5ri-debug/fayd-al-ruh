"""What this machine has, for the Model Manager's dashboard -- and nothing more.

A model manager that only lists files leaves the user to guess whether a model
will *fit*. This module answers that honestly with the **standard library only**:
no ``psutil``, no vendor SDK, no mandatory dependency. It reports what the
platform can actually tell it and returns ``None`` (never a made-up number) where
the platform cannot.

Two rules, matching the rest of the project:

* **Nothing here is a measurement of a model.** A model's memory need is an
  *estimate* the manager derives from the file (see
  :func:`core.model_manager._estimate_memory`); this module only reports the
  machine's own totals, so the two are never confused.
* **Nothing here is guessed.** ``available_memory_bytes`` is ``None`` on a
  platform that does not expose it rather than being equated with the total.

The module imports nothing heavy and never runs a subprocess, so importing it is
free and it is safe to call on a machine with no model, no GPU, and no network.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: The one honest sentence every surface shows beside the figures.
HONESTY_LINE = (
    "These are this machine's own totals, reported by the operating system. A "
    "model's memory need is an estimate derived from its file, never a measurement."
)


@dataclass(frozen=True)
class SystemResources:
    """The machine's own figures, each ``None`` when the platform cannot say."""

    system: str = ""
    machine: str = ""
    python: str = ""
    cpu_count: int | None = None
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    disk_free_bytes: int | None = None
    disk_path: str = ""
    notes: tuple[str, ...] = field(default=())

    def describe(self) -> str:
        """A readable block, honest about every unknown."""
        lines = [
            "This machine",
            f"  system:     {self.system or 'unknown'} ({self.machine or 'unknown'})",
            f"  python:     {self.python or 'unknown'}",
            f"  cpu cores:  {self.cpu_count if self.cpu_count else 'unknown'}",
            f"  memory:     {_size(self.total_memory_bytes)}"
            + (
                f" total, {_size(self.available_memory_bytes)} available"
                if self.available_memory_bytes is not None
                else " total"
            ),
        ]
        if self.disk_path:
            lines.append(
                f"  disk free:  {_size(self.disk_free_bytes)} at {self.disk_path}"
            )
        lines.extend(f"  note:       {note}" for note in self.notes)
        lines.append("")
        lines.append(HONESTY_LINE)
        return "\n".join(lines)


def _size(value: int | None) -> str:
    """A readable byte size, or ``unknown`` -- never a zero standing in for one."""
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def _total_memory_bytes() -> int | None:
    """Physical memory, via the platform's own call, or ``None``.

    Three real paths and no guess: ``/proc/meminfo`` on Linux (which also gives
    the *available* figure), ``sysconf`` on Linux/macOS, and ``GlobalMemoryStatusEx``
    on Windows. A platform that answers none of them reports ``None``.
    """
    system = platform.system().lower()
    if system == "linux":
        total, _ = _linux_meminfo()
        if total is not None:
            return total
    if system == "windows":
        total, _ = _windows_memory()
        if total is not None:
            return total
    return _sysconf_total()


def _available_memory_bytes() -> int | None:
    """Memory the OS says is available, or ``None`` where it cannot say.

    Only Linux (``/proc/meminfo``) and Windows report this reliably here; macOS
    and the rest return ``None`` rather than reusing the total.
    """
    system = platform.system().lower()
    if system == "linux":
        _, available = _linux_meminfo()
        return available
    if system == "windows":
        _, available = _windows_memory()
        return available
    return None


def _linux_meminfo() -> tuple[int | None, int | None]:
    """``(total, available)`` from ``/proc/meminfo``, each ``None`` if unreadable."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return (None, None)
    total = available = None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total = _kb_line(line)
        elif line.startswith("MemAvailable:"):
            available = _kb_line(line)
        if total is not None and available is not None:
            break
    return (total, available)


def _kb_line(line: str) -> int | None:
    """The kilobyte figure in a ``/proc/meminfo`` line, as bytes."""
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[1]) * 1024
    except ValueError:
        return None


def _windows_memory() -> tuple[int | None, int | None]:
    """``(total, available)`` from ``GlobalMemoryStatusEx``, each ``None`` if it fails."""
    try:  # pragma: no cover - Windows only
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        windll = ctypes.windll  # type: ignore[attr-defined]  # Windows-only attribute
        if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return (None, None)
        return (int(status.ullTotalPhys), int(status.ullAvailPhys))
    except Exception:
        return (None, None)


def _sysconf_total() -> int | None:
    """Total memory through ``os.sysconf``, or ``None`` where it is unsupported."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
        return pages * page_size
    return None


def disk_free(path: Path | str | None = None) -> int | None:
    """Free bytes on the volume holding ``path`` (the project's data folder by default)."""
    target = Path(path) if path is not None else Path.home()
    try:
        probe = target if target.exists() else target.parent
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return None


def system_resources(*, disk_path: Path | str | None = None) -> SystemResources:
    """This machine's figures, in one read. Never raises; every unknown is ``None``."""
    total = _total_memory_bytes()
    available = _available_memory_bytes()
    notes: list[str] = []
    if available is None:
        notes.append("This platform does not report available memory; only the total is shown.")
    if total is None:
        notes.append("This platform did not report its total memory.")

    resolved_disk = Path(disk_path) if disk_path is not None else None
    if resolved_disk is None:
        try:
            from core.paths import data_dir

            resolved_disk = data_dir()
        except Exception:  # pragma: no cover - a broken data path must not crash the read
            resolved_disk = None

    try:
        cpu_count = os.cpu_count()
    except Exception:  # pragma: no cover - defensive
        cpu_count = None

    return SystemResources(
        system=platform.system() or "",
        machine=platform.machine() or "",
        python=platform.python_version(),
        cpu_count=cpu_count,
        total_memory_bytes=total,
        available_memory_bytes=available,
        disk_free_bytes=disk_free(resolved_disk),
        disk_path=str(resolved_disk) if resolved_disk is not None else "",
        notes=tuple(notes),
    )


__all__ = [
    "HONESTY_LINE",
    "SystemResources",
    "disk_free",
    "system_resources",
]
