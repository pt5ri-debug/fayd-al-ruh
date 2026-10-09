"""Live progress reporting for long training runs.

The reporter is pure stdlib, so these tests cover the cadence, the estimated
time remaining, the final line, and the GUI's line stream directly.
"""

from __future__ import annotations

import io
import queue

from core.progress import ProgressCallback, ProgressReporter, format_duration, null_reporter
from gui.app import LineStream


def test_format_duration_units() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3900) == "1h 05m"


def test_reporter_prints_every_step_when_every_is_one() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter("policy", 3, every=1, stream=stream)
    for step in range(1, 4):
        reporter(step)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert "policy" in lines[0]
    assert "100.0%" in lines[-1]
    assert "done" in lines[-1]


def test_reporter_default_cadence_is_about_twenty_lines() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter("brain", 100, stream=stream)
    for step in range(1, 101):
        reporter(step)
    # The default cadence prints roughly twenty updates, and always the last one.
    assert 15 <= len(stream.getvalue().splitlines()) <= 25
    assert "100.0%" in stream.getvalue().splitlines()[-1]


def test_reporter_estimates_time_remaining() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter("policy", 10, every=1, stream=stream)
    reporter.started -= 20  # pretend twenty seconds already elapsed
    reporter(5)
    line = stream.getvalue().splitlines()[-1]
    assert "elapsed 20s" in line
    assert "eta 20s" in line


def test_reporter_is_disabled_by_default_flag() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter("policy", 5, every=1, stream=stream, enabled=False)
    for step in range(1, 6):
        reporter(step)
    assert stream.getvalue() == ""


def test_reporter_finish_forces_a_final_line() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter("policy", 10, every=100, stream=stream)
    reporter(3)
    reporter.finish(detail="saved")
    line = stream.getvalue().splitlines()[-1]
    assert "100.0%" in line
    assert "saved" in line
    # A second finish must not print twice.
    reporter.finish()
    assert len(stream.getvalue().splitlines()) == 1


def test_reporter_satisfies_the_callback_protocol() -> None:
    reporter: ProgressCallback = ProgressReporter("policy", 1, enabled=False)
    reporter(1, 1, detail="ok")
    null_reporter(1, 1, detail="ignored")


def test_line_stream_forwards_complete_lines() -> None:
    lines: queue.Queue[str] = queue.Queue()
    stream = LineStream(lines)
    stream.write("a\nb\npart")
    stream.flush()
    assert [lines.get_nowait(), lines.get_nowait(), lines.get_nowait()] == ["a", "b", "part"]
