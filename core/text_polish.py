"""The one linguistic-polish entry point every answering character shares.

The project answers through seven characters (Jarvis, Ultron, Vision, Banner,
Natasha, Hakeem, Cloud) and its own trained model. Each of them produced a reply
that went straight to the reader, so the same small-model scars -- a space before a
full stop, a missing space after a comma, a lowercase sentence opening, doubled
punctuation, an Arabic sentence punctuated with ASCII ``?`` -- appeared in every
mouth but the companion's. The companion (Natasha) already had a polisher
(:mod:`companion.polish`); this module makes it **universal**: the Jarvis pipeline
runs every answer through it (``core.jarvis`` is the wiring), so a reply from any
character reads cleanly.

Two things it adds on top of the companion's rules:

* **Full Arabic vocalization (tashkeel).** An Arabic reply can be fully vocalized
  through :mod:`core.arabic_tashkeel`, so the reader sees the vowels and the case
  endings instead of having to infer them. The user chooses the form:

  - ``"formal"`` -- classical Arabic (فصحى): the lexical and pattern vowels **and**
    the case endings (إعراب).
  - ``"colloquial"`` -- the spoken form (عامية): the lexical and pattern vowels
    only; the case endings are dropped, because speech does not pronounce them.

  Vocalization is **off by default** (a bare, undiacritized answer is the project's
  long-standing output) and turned on by the saved preference or an explicit
  argument, so nothing changes for a user who did not ask for it.

* **A saved preference**, so the choice survives a restart: ``tashkeel`` (on/off)
  and ``tashkeel_mode`` (``"formal"``/``"colloquial"``) live in the project's
  :class:`core.preferences.PreferenceRecord`, flipped by the ``tashkeel on`` /
  ``tashkeel off`` / ``tashkeel colloquial`` conversation controls. A headless run
  can override the saved choice for one process with ``FAYD_TASHKEEL`` and
  ``FAYD_TASHKEEL_MODE``; the precedence is argument → environment → saved →
  default.

Honesty by construction
-----------------------

* **Presentation only.** The polish never adds, removes, or reorders a word; the
  vocalization never touches a Latin identifier, a number, a formula, or a code
  span, and an English reply is returned byte-identical by it.
* **Idempotent.** Polishing an already-polished reply changes nothing, and
  vocalizing an already-vocalized reply is a no-op.
* **Explainable.** :func:`polish_answer` returns the text *plus* the changes and
  the vocalization mode, so a caller can show what was touched.
* **Fails soft.** Reading the saved preference is best-effort: an unreadable or
  read-only preference store yields the conservative default (off), never an error.

This module imports only the standard library at import time and loads the
companion's rules lazily, so importing it never pulls a model or reaches the
network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: The two vocalization modes, mirroring :mod:`core.arabic_tashkeel`.
TASHKEEL_FORMAL = "formal"
TASHKEEL_COLLOQUIAL = "colloquial"
TASHKEEL_MODES = (TASHKEEL_FORMAL, TASHKEEL_COLLOQUIAL)

#: The environment variables a headless run can set instead of the saved
#: preference: ``FAYD_TASHKEEL`` (``1``/``0``/``on``/``off``) and
#: ``FAYD_TASHKEEL_MODE`` (``formal``/``colloquial``). They sit between the
#: explicit argument and the saved preference, so a script can force the setting
#: for one run without writing to the user's preferences.
ENV_TASHKEEL = "FAYD_TASHKEEL"
ENV_TASHKEEL_MODE = "FAYD_TASHKEEL_MODE"

_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_FALSY = {"0", "false", "no", "off", "disabled"}


def _env_tashkeel() -> tuple[bool | None, str | None]:
    """The ``FAYD_TASHKEEL``/``FAYD_TASHKEEL_MODE`` overrides, or ``(None, None)``.

    An unrecognised value is ignored rather than guessed, so a typo cannot silently
    turn vocalization on or off.
    """
    enabled: bool | None = None
    raw = os.environ.get(ENV_TASHKEEL, "").strip().lower()
    if raw in _TRUTHY:
        enabled = True
    elif raw in _FALSY:
        enabled = False
    mode = os.environ.get(ENV_TASHKEEL_MODE, "").strip().lower() or None
    if mode is not None and mode not in TASHKEEL_MODES:
        mode = None
    return enabled, mode


@dataclass(frozen=True)
class PolishReport:
    """What one universal polish did: the text, the language, and the changes."""

    text: str
    language: str
    changes: tuple[str, ...] = field(default_factory=tuple)
    tashkeel_mode: str = ""
    """The vocalization mode applied, or ``""`` when the reply was not vocalized."""

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    @property
    def diacritized(self) -> bool:
        """Whether this pass fully vocalized an Arabic reply."""
        return bool(self.tashkeel_mode)

    def describe(self) -> str:
        if not self.changes:
            return f"polish ({self.language}): no change needed"
        mode = f", {self.tashkeel_mode} tashkeel" if self.tashkeel_mode else ""
        return f"polish ({self.language}{mode}): " + "; ".join(self.changes)


def _saved_tashkeel() -> tuple[bool, str]:
    """The saved tashkeel preference, or the conservative default ``(False, formal)``.

    Reading the preference is best-effort: a missing, unreadable, or read-only
    store yields the default rather than breaking an answer. The failure is
    recorded through :func:`core.best_effort.best_effort`, so a real problem leaves
    a trace instead of being swallowed.
    """
    from core.best_effort import best_effort

    with best_effort("read the saved tashkeel preference"):
        from core.paths import data_dir
        from core.preferences import UserPreferences

        record = UserPreferences(data_dir()).record
        return bool(record.tashkeel), str(record.tashkeel_mode or TASHKEEL_FORMAL)
    return False, TASHKEEL_FORMAL


def resolve_tashkeel(
    *, enabled: bool | None = None, mode: str | None = None
) -> tuple[bool, str]:
    """Resolve the vocalization settings: argument, env, saved, then off.

    The precedence is the project's own convention: an explicit argument wins for
    one call, then ``FAYD_TASHKEEL``/``FAYD_TASHKEEL_MODE`` (a headless run's
    override), then the saved preference, then the conservative default. An
    unknown *mode* is rejected with a ``ValueError`` rather than silently coerced,
    so a typo is reported.
    """
    if enabled is None or mode is None:
        env_enabled, env_mode = _env_tashkeel()
        if enabled is None:
            enabled = env_enabled
        if mode is None:
            mode = env_mode
    if enabled is None or mode is None:
        saved_enabled, saved_mode = _saved_tashkeel()
        if enabled is None:
            enabled = saved_enabled
        if mode is None:
            mode = saved_mode
    if mode not in TASHKEEL_MODES:
        raise ValueError(f"Unknown tashkeel mode: {mode!r}")
    return bool(enabled), mode


def polish_answer(
    text: str,
    *,
    language: str | None = None,
    tashkeel: bool | None = None,
    tashkeel_mode: str | None = None,
) -> PolishReport:
    """Polish an answer with the shared rules, optionally vocalizing Arabic.

    ``language`` may be supplied when the caller already knows it; otherwise it is
    detected from the text, so an English reply gets the English rules and an
    Arabic one the Arabic rules. ``tashkeel`` / ``tashkeel_mode`` override the
    saved preference for one call (``None`` means "follow the preference").
    """
    from companion.polish import polish_report

    enabled, mode = resolve_tashkeel(enabled=tashkeel, mode=tashkeel_mode)
    result = polish_report(text, language=language, tashkeel=enabled, tashkeel_mode=mode)
    return PolishReport(
        text=result.text,
        language=result.language,
        changes=tuple(result.changes),
        tashkeel_mode=result.tashkeel_mode,
    )


def polish_text(
    text: str,
    *,
    language: str | None = None,
    tashkeel: bool | None = None,
    tashkeel_mode: str | None = None,
) -> str:
    """The polished text alone, for callers that do not need the report."""
    return polish_answer(
        text, language=language, tashkeel=tashkeel, tashkeel_mode=tashkeel_mode
    ).text


def tashkeel_text(text: str, *, mode: str = TASHKEEL_FORMAL) -> str:
    """Fully vocalize an Arabic text directly, without the typographic rules.

    A thin, explicit doorway to :mod:`core.arabic_tashkeel` for a caller that
    wants the vowels only (e.g. the ``tashkeel`` CLI command).
    """
    from core.arabic_tashkeel import diacritize

    return diacritize(text, mode=mode)


__all__ = [
    "ENV_TASHKEEL",
    "ENV_TASHKEEL_MODE",
    "TASHKEEL_COLLOQUIAL",
    "TASHKEEL_FORMAL",
    "TASHKEEL_MODES",
    "PolishReport",
    "polish_answer",
    "polish_text",
    "resolve_tashkeel",
    "tashkeel_text",
]
