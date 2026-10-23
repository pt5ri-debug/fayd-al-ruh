"""A destroyed Tk root must be reaped on the thread that created it.

A window test builds ``tk.Tk()`` and destroys it, but the interpreter object
survives until the garbage collector runs. If that collection happens on a
**worker thread** -- the next window test's background job -- Tk refuses to free
an interpreter off its creating thread (CPython gh-83274). On 3.12 the process
aborts outright (``Fatal Python error: Aborted`` inside ``Garbage-collecting``);
on 3.13 it surfaces as a ``RuntimeWarning`` naming gh-83274.

``tests/conftest.py`` reaps a dead root on the main thread right after the test
that built it, which is what makes the window tests safe. These tests pin both
halves: the hazard is real (collecting on a worker warns) and the remedy works
(collecting on the main thread first leaves nothing for the worker to reap).

The check runs in a subprocess so it can install its own GC/thread schedule
without perturbing the rest of the suite, and it needs a display, so it skips
headless like every other window test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_HAZARD_MARKER = "gh-83274"

_SCRIPT = textwrap.dedent(
    """
    import gc, sys, threading
    import tkinter as tk

    # --- a window test: build a root, destroy it, drop the last reference ---
    root = tk.Tk()
    root.withdraw()
    root.cycle = root
    root.destroy()
    del root

    if sys.argv[1] == "reap":
        gc.collect()  # what tests/conftest.py does, on the main thread

    # --- the next window test's worker: allocates, so it may collect ---
    def worker():
        junk = []
        for i in range(80000):
            junk.append([i] * 4)
            if i % 10000 == 0:
                gc.collect()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    print("DONE")
    """
)


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _run(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, mode],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_worker_thread_reaping_a_dead_root_is_the_hazard() -> None:
    """Without the main-thread reap, the worker thread collects the dead root."""
    result = _run("noreap")
    assert "DONE" in result.stdout
    assert _HAZARD_MARKER in result.stderr, (
        "expected the worker thread to reap the dead Tk root; if this no longer "
        "happens the hazard may be fixed upstream and this guard can be relaxed"
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_reaping_on_the_main_thread_leaves_nothing_for_the_worker() -> None:
    """The conftest's remedy: collect on the main thread first, worker stays clean."""
    result = _run("reap")
    assert "DONE" in result.stdout
    assert _HAZARD_MARKER not in result.stderr
