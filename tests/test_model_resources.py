"""This machine's own figures, read honestly and without a dependency.

Every test here runs with no GPU, no model, and no network. The module is
dependency-free by design, so these tests exercise the real code paths -- the
``/proc/meminfo`` parse, the ``sysconf`` fallback, the disk read -- and, where a
platform path cannot run on this machine, the *honest* ``None`` it returns.
"""

from __future__ import annotations

from pathlib import Path

from core.model_resources import (
    HONESTY_LINE,
    SystemResources,
    _kb_line,
    _linux_meminfo,
    _size,
    _sysconf_total,
    disk_free,
    system_resources,
)


def test_size_renders_units_and_never_fakes_a_zero():
    assert _size(None) == "unknown"
    assert _size(0) == "0.0 B"
    assert _size(1024) == "1.0 KB"
    assert _size(2 * 1024**3) == "2.0 GB"


def test_kb_line_reads_the_figure_as_bytes():
    assert _kb_line("MemTotal:       16384000 kB") == 16384000 * 1024
    assert _kb_line("garbage") is None
    assert _kb_line("MemTotal: notanumber kB") is None


def test_linux_meminfo_reports_a_total_on_linux():
    # On Linux this is the real file; on another platform it is the honest
    # (None, None). Either way the call must not raise.
    total, available = _linux_meminfo()
    assert total is None or total > 0
    assert available is None or available > 0


def test_sysconf_total_is_positive_or_none():
    total = _sysconf_total()
    assert total is None or total > 0


def test_disk_free_reads_a_real_volume_or_reports_none(tmp_path):
    value = disk_free(tmp_path)
    assert value is None or value > 0


def test_disk_free_falls_back_to_the_parent_for_a_missing_path(tmp_path):
    missing = tmp_path / "does" / "not" / "exist"
    value = disk_free(missing)
    assert value is None or value > 0


def test_system_resources_reports_real_figures_without_raising():
    resources = system_resources(disk_path=Path.cwd())
    assert isinstance(resources, SystemResources)
    assert resources.system  # platform.system() answers everywhere
    assert resources.python
    assert resources.cpu_count is None or resources.cpu_count > 0
    assert resources.total_memory_bytes is None or resources.total_memory_bytes > 0
    assert resources.disk_free_bytes is None or resources.disk_free_bytes > 0


def test_describe_names_every_unknown_and_keeps_the_honesty_line():
    resources = SystemResources(system="TestOS", machine="test", python="3.13")
    text = resources.describe()
    assert "unknown" in text
    assert HONESTY_LINE in text


def test_describe_reports_available_memory_when_it_is_known():
    resources = SystemResources(
        system="TestOS", machine="test", python="3.13",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=8 * 1024**3,
    )
    text = resources.describe()
    assert "16.0 GB total" in text
    assert "8.0 GB available" in text


def test_available_memory_is_none_rather_than_the_total_when_unknown(monkeypatch):
    import core.model_resources as module

    monkeypatch.setattr(module.platform, "system", lambda: "Plan9")
    assert module._available_memory_bytes() is None
