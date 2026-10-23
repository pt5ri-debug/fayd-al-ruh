"""Shared test configuration.

The one job here is to keep a *dead* Tk root from being collected by the wrong
thread. A window test destroys its ``tk.Tk()`` in a ``finally`` block, but the
interpreter object it owns survives until the garbage collector happens to run.
If that collection lands on a **worker thread** -- the next window test's
background job, say -- Tk refuses to free an interpreter off the thread that
created it and the whole process aborts (CPython gh-83274; the reported symptom
is ``Fatal Python error: Aborted`` inside ``Garbage-collecting``).

``tests/test_scene3d_window.py`` documents this and calls ``gc.collect()`` on the
main thread after each of its windows. Doing it here for every test that *builds
a Tk root* means no window test has to remember, and a root that dies in one test
can never be reaped on another test's worker.

The collection is deliberately gated on a root actually having been created: a
blanket collect after all ~3100 tests costs minutes, while only the window tests
need it.
"""

from __future__ import annotations

import gc
import sys

import pytest

_tk_roots_created = 0


@pytest.fixture(scope="session", autouse=True)
def _count_tk_roots():
    """Count ``tkinter.Tk`` constructions so only those tests collect."""
    try:
        import tkinter
    except Exception:  # pragma: no cover - no Tk on this machine
        yield
        return

    original_init = tkinter.Tk.__init__

    def counting_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        global _tk_roots_created
        _tk_roots_created += 1
        original_init(self, *args, **kwargs)

    tkinter.Tk.__init__ = counting_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        tkinter.Tk.__init__ = original_init  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reap_dead_tk_roots():
    """After a test that built a Tk root, reap the dead one on this thread."""
    global _tk_roots_created
    before = _tk_roots_created
    yield
    if _tk_roots_created > before and "tkinter" in sys.modules:
        gc.collect()
