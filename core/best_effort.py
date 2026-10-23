"""A single, inspectable place for a deliberate best-effort operation.

Some work must not fail its caller: persisting a cosmetic preference into a
possibly read-only data folder, probing an optional engine that may not import
here, asking a diffusion pipeline for a feature older versions lack. Swallowing
the error is *correct* in those places -- but swallowing it invisibly is not,
because a genuine bug then disappears with no trace at all.

``best_effort`` keeps the operation non-fatal while recording what was dropped,
so the failure is discoverable (in a test, a diagnostic, or a bug report)
instead of lost. It is dependency-free and bounded, so a long run cannot grow
without limit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

#: How many dropped errors to remember. A best-effort failure is rare, so a
#: small ring keeps the whole history of an ordinary run and cannot grow.
MAX_RECORDED = 64


@dataclass(frozen=True)
class SwallowedError:
    """One deliberate swallow: what was being attempted, and why it failed."""

    what: str
    error: BaseException

    def describe(self) -> str:
        return f"{self.what}: {type(self.error).__name__}: {self.error}"


_swallowed: deque[SwallowedError] = deque(maxlen=MAX_RECORDED)


@contextmanager
def best_effort(what: str, *, on_error: Callable[[BaseException], None] | None = None) -> Iterator[None]:
    """Run a best-effort operation: never raise, but never hide the failure.

    ``what`` names the operation in the record (for example "save the reading
    voice"), so a later reader can tell what was dropped without guessing. An
    optional ``on_error`` observes the error at the call site when the caller
    wants to react without failing (for example to fall back to a sibling).
    """
    try:
        yield
    except Exception as error:
        _swallowed.append(SwallowedError(what, error))
        if on_error is not None:
            on_error(error)


def swallowed_errors() -> tuple[SwallowedError, ...]:
    """Every recorded best-effort failure, oldest first."""
    return tuple(_swallowed)


def clear_swallowed() -> None:
    """Forget the recorded failures (a test, or a fresh diagnostic window)."""
    _swallowed.clear()


__all__ = ["MAX_RECORDED", "SwallowedError", "best_effort", "clear_swallowed", "swallowed_errors"]
