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
from typing import Any

#: The one honest sentence every surface shows beside the figures.
HONESTY_LINE = (
    "These are this machine's own totals, reported by the operating system. A "
    "model's memory need is an estimate derived from its file, never a measurement."
)

#: Where a figure came from, so a reading is never presented as more than it is.
READING_MEASURED = "measured"
READING_OS_ESTIMATE = "os-estimate"
READING_RUNTIME = "runtime-reported"
READING_UNKNOWN = "unknown"

READING_LABELS: dict[str, str] = {
    READING_MEASURED: "measured by the operating system",
    READING_OS_ESTIMATE: "an operating-system estimate",
    READING_RUNTIME: "reported by the runtime",
    READING_UNKNOWN: "unknown on this platform",
}


def describe_reading(reading: str) -> str:
    """A readable label for a reading's provenance."""
    return READING_LABELS.get(reading, READING_LABELS[READING_UNKNOWN])


@dataclass(frozen=True)
class GpuInfo:
    """What this machine can say about its GPUs -- and nothing it cannot.

    Deliberately **not** a claim that a GPU can accelerate any given model. A GPU
    that exists is not a GPU the installed ``llama.cpp`` build can offload to, and
    the two facts are kept apart: this records what the *system* reports, and
    :mod:`core.model_runtime` separately reports what the *runtime* supports.
    """

    available: bool = False
    #: What the system called the device, when it named one.
    name: str = ""
    vendor: str = ""
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None
    #: How the information was obtained (``nvidia-smi``, ``sysfs``, ...), so the
    #: provenance of a figure is never lost.
    source: str = ""
    #: Why a device is not usable here, when that is known.
    note: str = ""
    #: Devices found, as ``(name, vendor, memory_bytes)`` triples.
    devices: tuple[tuple[str, str, int | None], ...] = field(default=())

    def describe(self) -> str:
        if not self.available:
            line = "GPU: none detected that this program can report on"
            return f"{line} — {self.note}" if self.note else line
        bits = [f"GPU: {self.name or 'unnamed device'}"]
        if self.vendor:
            bits.append(self.vendor)
        if self.memory_total_bytes is not None:
            total = f"{self.memory_total_bytes / 1e9:.2f} GB"
            if self.memory_free_bytes is not None:
                bits.append(f"{total} total, {self.memory_free_bytes / 1e9:.2f} GB free")
            else:
                bits.append(f"{total} total")
        if self.source:
            bits.append(f"via {self.source}")
        line = " · ".join(bits)
        if self.note:
            line = f"{line} — {self.note}"
        return line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "devices": [
                {"memory_bytes": memory, "name": name, "vendor": vendor}
                for name, vendor, memory in self.devices
            ],
            "memory_free_bytes": self.memory_free_bytes,
            "memory_total_bytes": self.memory_total_bytes,
            "name": self.name,
            "note": self.note,
            "source": self.source,
            "vendor": self.vendor,
        }


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
    #: What the system reports about GPUs, and nothing more.
    gpu: GpuInfo = field(default_factory=GpuInfo)
    #: The process's own resident memory, when the platform can report it.
    process_memory_bytes: int | None = None
    #: How each group of figures was obtained, so a reading is never overstated.
    memory_reading: str = READING_UNKNOWN
    disk_reading: str = READING_UNKNOWN
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
        if self.process_memory_bytes is not None:
            lines.append(f"  this process: {_size(self.process_memory_bytes)} resident")
        lines.append(f"  {self.gpu.describe()}")
        lines.extend(f"  note:       {note}" for note in self.notes)
        lines.append("")
        lines.append(HONESTY_LINE)
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        """A deterministic, JSON-safe mapping -- every unknown stays ``None``."""
        return {
            "available_memory_bytes": self.available_memory_bytes,
            "cpu_count": self.cpu_count,
            "disk_free_bytes": self.disk_free_bytes,
            "disk_path": self.disk_path,
            "disk_reading": self.disk_reading,
            "gpu": self.gpu.to_primitive(),
            "machine": self.machine,
            "memory_reading": self.memory_reading,
            "notes": list(self.notes),
            "process_memory_bytes": self.process_memory_bytes,
            "python": self.python,
            "system": self.system,
            "total_memory_bytes": self.total_memory_bytes,
        }


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


# ------------------------------------------------------------------- GPU


def process_memory_bytes() -> int | None:
    """This process's resident memory, or ``None`` where the platform cannot say.

    Read from ``/proc/self/statm`` on Linux and ``GetProcessMemoryInfo`` on
    Windows; macOS and the rest report ``None`` rather than an approximation. Used
    to separate "this program is using X" from "this machine has Y", which a
    resource warning must not conflate.
    """
    if platform.system().lower() == "linux":
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            return None
    if platform.system().lower() == "windows":  # pragma: no cover - Windows only
        try:
            import ctypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            windll = ctypes.windll  # type: ignore[attr-defined]  # Windows-only attribute
            handle = windll.kernel32.GetCurrentProcess()
            ok = windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None
    return None


def gpu_info(*, runner: Any = None) -> GpuInfo:
    """What this machine can report about GPUs, without importing a vendor SDK.

    Two lightweight paths and no dependency: ``nvidia-smi`` if the program is on
    ``PATH`` (an NVIDIA machine), and the DRM ``sysfs`` entries on Linux (which
    name the device and vendor, and on AMD expose the memory). Neither is
    mandatory, and a machine with no GPU, or with a GPU neither path can describe,
    reports ``available=False`` with the reason -- never a fabricated figure.

    This says nothing about whether the installed inference runtime can *use* a
    device: that is :func:`core.model_runtime.llama_cpp_gpu_offload`'s question,
    and the two answers are deliberately kept separate.

    ``runner`` is injectable (``(command) -> (returncode, stdout)``) so the whole
    function is tested with no GPU and no subprocess.
    """
    run = runner or _run_quietly
    info = _gpu_from_nvidia_smi(run)
    if info.available:
        return info
    linux = _gpu_from_sysfs()
    if linux.available:
        return linux
    note = ""
    if info.note:
        note = info.note
    elif linux.note:
        note = linux.note
    else:
        note = (
            "No GPU was reported by nvidia-smi or the DRM sysfs entries. A GPU the "
            "system does not expose here may still exist."
        )
    return GpuInfo(available=False, note=note)


def _gpu_from_nvidia_smi(run: Any) -> GpuInfo:
    """Parse ``nvidia-smi --query-gpu``, which names devices and their memory."""
    query = (
        "name,vendor,memory.total,memory.free"
    )
    code, output = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if code != 0 or not output.strip():
        return GpuInfo(
            available=False,
            note="nvidia-smi is not installed here, or reported no GPU.",
        )
    devices: list[tuple[str, str, int | None]] = []
    first: tuple[str, str, int | None, int | None] | None = None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        name, vendor = parts[0], parts[1]
        total = _mb_to_bytes(parts[2])
        free = _mb_to_bytes(parts[3])
        devices.append((name, vendor, total))
        if first is None:
            first = (name, vendor, total, free)
    if not devices or first is None:
        return GpuInfo(available=False, note="nvidia-smi reported no usable GPU.")
    name, vendor, total, free = first
    return GpuInfo(
        available=True,
        name=name,
        vendor=vendor,
        memory_total_bytes=total,
        memory_free_bytes=free,
        source="nvidia-smi",
        devices=tuple(devices),
    )


def _gpu_from_sysfs() -> GpuInfo:
    """Read the DRM cards Linux exposes, which need no program and no driver SDK."""
    root = Path("/sys/class/drm")
    if not root.is_dir():
        return GpuInfo(available=False, note="This platform exposes no DRM GPU entries.")
    devices: list[tuple[str, str, int | None]] = []
    first: tuple[str, str, int | None] | None = None
    try:
        cards = sorted(root.glob("card[0-9]*"))
    except OSError:
        return GpuInfo(available=False, note="The DRM entries could not be read.")
    for card in cards:
        if "-" in card.name:
            continue  # a connector (card0-HDMI-A-1), not the device
        device = card / "device"
        vendor = _read_text(device / "vendor")
        name = vendor_name(vendor) or _read_text(device / "uevent", key="DRIVER") or card.name
        memory = _gpu_memory_from_sysfs(device)
        devices.append((name, vendor, memory))
        if first is None:
            first = (name, vendor, memory)
    if not devices or first is None:
        return GpuInfo(available=False, note="No DRM card device was found.")
    name, vendor, memory = first
    return GpuInfo(
        available=True,
        name=name,
        vendor=vendor,
        memory_total_bytes=memory,
        memory_free_bytes=None,  # sysfs exposes the total, not a live free figure
        source="sysfs",
        note="The DRM entries name the device; they do not report a live free memory.",
        devices=tuple(devices),
    )


#: A vendor id as Linux reports it, so a device reads as a name and not a hex code.
GPU_VENDOR_IDS: dict[str, str] = {
    "0x1002": "AMD",
    "0x10de": "NVIDIA",
    "0x8086": "Intel",
}


def vendor_name(vendor_id: str) -> str:
    """The vendor for a DRM vendor id, or ``""`` when it is not one this knows."""
    return GPU_VENDOR_IDS.get(str(vendor_id or "").strip().lower(), "")


def _gpu_memory_from_sysfs(device: Path) -> int | None:
    """An AMD device's VRAM total from ``mem_info_vram_total``, or ``None``.

    NVIDIA does not expose this through sysfs, so the honest answer there is
    ``None``; a machine that reports no total is not given one.
    """
    value = _read_text(device / "mem_info_vram_total")
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _read_text(path: Path, *, key: str = "") -> str:
    """Read a small sysfs file, or ``""``; with ``key``, one ``uevent`` field."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if not key:
        return text.strip()
    for line in text.splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return ""


def _mb_to_bytes(value: str) -> int | None:
    """A megabyte figure from a tool's output as bytes, or ``None``."""
    try:
        return int(float(value.strip())) * 1024 * 1024
    except (ValueError, AttributeError):
        return None


def _run_quietly(command: list[str]) -> tuple[int, str]:
    """Run a small probe, returning ``(code, stdout)``; never raises.

    Used only for a read-only query (``nvidia-smi --query-gpu``), never for
    anything that changes the machine, and always with a short timeout.
    """
    import subprocess

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return (1, "")
    return (int(result.returncode), result.stdout or "")


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

    memory_reading = READING_UNKNOWN
    if total is not None:
        memory_reading = READING_MEASURED if available is not None else READING_OS_ESTIMATE
    disk_value = disk_free(resolved_disk)
    return SystemResources(
        system=platform.system() or "",
        machine=platform.machine() or "",
        python=platform.python_version(),
        cpu_count=cpu_count,
        total_memory_bytes=total,
        available_memory_bytes=available,
        disk_free_bytes=disk_value,
        disk_path=str(resolved_disk) if resolved_disk is not None else "",
        gpu=gpu_info(),
        process_memory_bytes=process_memory_bytes(),
        memory_reading=memory_reading,
        disk_reading=READING_MEASURED if disk_value is not None else READING_UNKNOWN,
        notes=tuple(notes),
    )


__all__ = [
    "GPU_VENDOR_IDS",
    "HONESTY_LINE",
    "READING_LABELS",
    "READING_MEASURED",
    "READING_OS_ESTIMATE",
    "READING_RUNTIME",
    "READING_UNKNOWN",
    "GpuInfo",
    "SystemResources",
    "describe_reading",
    "disk_free",
    "gpu_info",
    "process_memory_bytes",
    "system_resources",
    "vendor_name",
]
