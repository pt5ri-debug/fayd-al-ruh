"""A character's own model and voice settings, folded onto the mouth it speaks with.

A character's configuration (see :mod:`core.character_config`) carries a whole
**Model** section (which model, and its temperature, top-p, context, and output
cap) and a whole **Voice** section (engine, voice, language, gender, speed,
pitch, volume). The Characters window lets a user edit all of it and press APPLY
-- but until this module existed, nothing read those fields: the window saved
them, and the mouth went on answering with whatever model and voice it would
have picked on its own. A setting that is editable, saved, and reported as
applied while changing nothing is worse than no setting at all, so this module is
the one place that turns the configuration into real effect.

Two rules keep it honest:

* **It reports what it could not do.** A mouth that exposes no engine, an engine
  that does not carry a given knob (a cloud provider has no ``context``), a voice
  this machine does not have -- each is named in the returned
  :class:`EngineSettingsReport` rather than silently dropped.
* **It invents nothing.** A blank field means "leave the mouth's own choice
  alone", never "reset to some default"; and a voice key is resolved through the
  project's own voice catalog, so an unknown key is a reported miss, not a guess.

Nothing here imports Tk or loads a model: it only *sets attributes* on an
already-built assistant, so a test can exercise it against a stub engine with no
runtime and no weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The attribute names a mouth's engine may carry for each model setting, in the
#: order they are tried. The in-process GGUF engines and the cloud engine agree on
#: ``temperature``/``max_tokens``/``system_prompt``; ``context``/``top_p`` are the
#: in-process names (the cloud engine takes no context), and ``num_ctx`` is
#: Ollama's. Trying each name is what lets one function serve every runtime.
_SETTING_ATTRS: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature",),
    "top_p": ("top_p",),
    "context_length": ("context", "num_ctx"),
    "max_tokens": ("max_tokens",),
    "system_prompt": ("system_prompt",),
}


@dataclass(frozen=True)
class EngineSettingsReport:
    """What a character's model settings actually did to a mouth.

    ``applied`` names the settings that reached the engine; ``skipped`` names the
    ones that did not, each with the reason (``"no engine"``, ``"engine has no
    top_p"``, ...). ``describe()`` is the one readable line a surface shows, so a
    user is told the truth rather than shown a saved setting that did nothing.
    """

    mouth: str
    applied: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def changed(self) -> bool:
        """True when at least one setting reached the engine."""
        return bool(self.applied)

    def describe(self) -> str:
        """One readable line naming what was applied and what was not."""
        if not self.applied and not self.skipped:
            return f"{self.mouth}: no model settings to apply."
        parts: list[str] = []
        if self.applied:
            parts.append(f"{self.mouth}: applied {', '.join(self.applied)}")
        else:
            parts.append(f"{self.mouth}: applied nothing")
        if self.skipped:
            reasons = "; ".join(f"{name} ({why})" for name, why in self.skipped)
            parts.append(f"skipped {reasons}")
        return ". ".join(parts) + "."


def _engine_of(assistant: Any) -> Any:
    """The object a mouth keeps its generation settings on, or ``None``.

    The in-process mouths keep an ``engine``; the trained-model mouth keeps a
    ``model``; the companion and the scholar keep an ``agent`` whose ``config``
    holds the settings. Returning the first that exists (never raising) is what
    lets one function serve them all, and a mouth with none is reported honestly.
    """
    for name in ("engine", "model"):
        candidate = getattr(assistant, name, None)
        if candidate is not None:
            return candidate
    agent = getattr(assistant, "agent", None)
    if agent is not None:
        config = getattr(agent, "config", None)
        if config is not None:
            return config
    return None


def _set_setting(engine: Any, name: str, value: Any) -> bool:
    """Set one setting on ``engine`` under the first attribute name it carries."""
    for attr in _SETTING_ATTRS[name]:
        if hasattr(engine, attr):
            try:
                setattr(engine, attr, value)
            except (AttributeError, TypeError):  # a read-only or frozen setting
                continue
            return True
    return False


def _default_model_for(owner: str) -> Any:
    """The shipped default Model section for ``owner``, or ``None``.

    Used to tell a *changed* setting from one still sitting at its default: a
    character that was never edited must keep the mouth's own tuning (Vision's
    0.6, not the generic 0.7), so a field equal to the shipped default is left
    alone rather than written over the engine's choice.
    """
    try:
        from core.character_config import default_config_for

        return getattr(default_config_for(owner), "model", None)
    except (ImportError, ValueError, OSError):  # pragma: no cover - defensive
        return None


def apply_model_settings(
    mouth: str,
    assistant: Any,
    config: Any,
    *,
    base_system_prompt: str | None = None,
    include_system_prompt: bool = True,
) -> EngineSettingsReport:
    """Fold a character's Model section onto an already-built ``assistant``.

    Only a field the user actually **changed** is applied: a value still equal to
    the character's shipped default is left alone, so a character that was never
    edited keeps the mouth's own tuning exactly as before (Vision's temperature is
    0.6, and a generic 0.7 must not be written over it), and an empty
    ``system_prompt`` is skipped too. The ``system_prompt`` is *appended* to the
    mouth's own prompt (``base_system_prompt``) rather than replacing it, so a
    character's extra instruction can never drop the honesty rules the mouth must
    keep.

    ``include_system_prompt=False`` leaves the prompt to the caller -- the
    character wrapper renders the model-specific prompt as part of its standing
    instruction (see :meth:`core.character_config.CharacterConfig.system_prompt`),
    so it must not be written twice.
    """
    if config is None:
        return EngineSettingsReport(mouth=mouth, skipped=(("all", "no configuration"),))
    model = getattr(config, "model", None)
    if model is None:
        return EngineSettingsReport(mouth=mouth, skipped=(("all", "no model section"),))
    engine = _engine_of(assistant)
    if engine is None:
        return EngineSettingsReport(mouth=mouth, skipped=(("all", "the mouth exposes no engine"),))

    baseline = _default_model_for(getattr(config, "owner", "") or mouth)
    applied: list[str] = []
    skipped: list[tuple[str, str]] = []

    numeric = (
        ("temperature", model.temperature),
        ("top_p", model.top_p),
        ("context_length", model.context_length),
        ("max_tokens", model.max_tokens),
    )
    for name, value in numeric:
        if value is None:
            continue
        if baseline is not None and getattr(baseline, name, None) == value:
            continue  # unchanged from the shipped default: leave the mouth's tuning
        if _set_setting(engine, name, value):
            applied.append(name)
        else:
            skipped.append((name, "the engine does not carry this setting"))

    extra = (model.system_prompt or "").strip()
    if extra and include_system_prompt:
        combined = f"{base_system_prompt}\n\n{extra}" if base_system_prompt else extra
        if _set_setting(engine, "system_prompt", combined):
            applied.append("system_prompt")
        else:
            skipped.append(("system_prompt", "the engine does not carry this setting"))

    return EngineSettingsReport(mouth=mouth, applied=tuple(applied), skipped=tuple(skipped))


@dataclass(frozen=True)
class VoiceChoice:
    """A resolved voice: the object to speak with, or the reason there is none."""

    voice: Any = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.voice is not None


def voice_language(config: Any) -> str:
    """The two-letter language a character's Voice section pins, or ``""``.

    ``""`` (or ``match``) means "follow the text's own language", which is the
    reader's default, so only an explicit ``ar``/``en`` is returned.
    """
    voice_config = getattr(config, "voice", None)
    if voice_config is None:
        return ""
    language = (getattr(voice_config, "language", "") or "").strip().lower()
    return language if language in ("ar", "en") else ""


def resolve_character_prosody(config: Any) -> Any:
    """The prosody a character's Voice section asks for, or ``None`` for neutral.

    The Voice section's ``speed`` (a multiplier), ``pitch`` (an offset), and
    ``volume`` (a loudness) are turned into the project's engine-neutral
    :class:`~core.voice_moods.Prosody`, clamped to that type's own bounds -- so a
    character's speed/pitch/volume really shape the reading instead of being
    stored and ignored. A section sitting at its shipped defaults (speed 1.0,
    pitch 0.0, volume 1.0) is *neutral* and yields ``None``, so a character that
    was never edited is unchanged.
    """
    voice_config = getattr(config, "voice", None)
    if voice_config is None or not getattr(voice_config, "active", True):
        return None
    speed = float(getattr(voice_config, "speed", 1.0) or 1.0)
    pitch = float(getattr(voice_config, "pitch", 0.0) or 0.0)
    volume = float(getattr(voice_config, "volume", 1.0) or 1.0)
    if speed == 1.0 and pitch == 0.0 and volume == 1.0:
        return None
    try:
        from core.voice_moods import (
            MAX_ENERGY,
            MAX_PITCH_DELTA,
            MAX_SPEED_SCALE,
            MIN_ENERGY,
            MIN_PITCH_DELTA,
            MIN_SPEED_SCALE,
            Prosody,
        )
    except ImportError:  # pragma: no cover - core.voice_moods is a base module
        return None

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    return Prosody(
        speed_scale=clamp(speed, MIN_SPEED_SCALE, MAX_SPEED_SCALE),
        pitch_delta=int(clamp(pitch, MIN_PITCH_DELTA, MAX_PITCH_DELTA)),
        energy=clamp(volume, MIN_ENERGY, MAX_ENERGY),
    )


def apply_character_voice(reader: Any, config: Any) -> VoiceChoice:
    """Apply a character's whole Voice section to a live ``reader`` (a TextReader).

    Resolves and sets the **voice** (through the project's catalog, so an
    uninstalled choice is reported rather than guessed), pins the **language** when
    the character asks for one, and folds the section's speed/pitch/volume onto the
    reader as an explicit **prosody**. Returns the resolved :class:`VoiceChoice`, so
    the caller can report what took effect. Nothing raises: a reader that refuses a
    voice is reported, and the rest of the section is still applied.
    """
    choice = resolve_character_voice(config)
    if choice.ok:
        try:
            reader.set_voice(choice.voice)
        except (ValueError, AttributeError):
            choice = VoiceChoice(reason="the reader refused the voice")
    language = voice_language(config)
    if language:
        try:
            reader.language = language
        except AttributeError:  # pragma: no cover - a reader without a language
            pass
    prosody = resolve_character_prosody(config)
    setter = getattr(reader, "set_prosody", None)
    if prosody is not None and callable(setter):
        try:
            setter(prosody)
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            pass
    return choice


def resolve_character_voice(config: Any) -> VoiceChoice:
    """Resolve a character's Voice section to a real voice, honestly.

    A saved key is looked up against what this machine currently has -- a catalog
    voice, an installed ``system:`` voice, or a local ``model:`` voice (see
    :mod:`core.voices`) -- so a saved choice that left the machine falls back with
    a reason rather than leaving the reading broken. A blank voice means "leave the
    reader's own choice alone", which is a *skip*, not a failure.
    """
    if config is None:
        return VoiceChoice(reason="no configuration")
    voice_config = getattr(config, "voice", None)
    if voice_config is None:
        return VoiceChoice(reason="no voice section")
    if not getattr(voice_config, "active", True):
        return VoiceChoice(reason="the voice is switched off")
    key = (getattr(voice_config, "voice", "") or "").strip()
    if not key:
        return VoiceChoice(reason="no voice is chosen")
    try:
        from core.voices import (
            MODEL_VOICE_PREFIX,
            SYSTEM_VOICE_PREFIX,
            find_model_voice,
            find_system_voice,
            find_voice,
        )
    except ImportError:  # pragma: no cover - core.voices is a base module
        return VoiceChoice(reason="the voice catalog is unavailable")

    found: Any
    if key.startswith(MODEL_VOICE_PREFIX):
        found = find_model_voice(key)
        return VoiceChoice(voice=found, reason="" if found else "the local voice is not installed")
    if key.startswith(SYSTEM_VOICE_PREFIX):
        found = find_system_voice(key)
        return VoiceChoice(voice=found, reason="" if found else "the installed voice is gone")
    try:
        return VoiceChoice(voice=find_voice(key))
    except ValueError:
        return VoiceChoice(reason=f"unknown voice key {key!r}")


@dataclass
class SettingsLedger:
    """A running record of what a character's settings could not do.

    Collected rather than raised, so a caller can apply several settings and
    report the misses together -- and so a read-only or absent engine never
    breaks a chat. It is bounded (the last :data:`MAX_LEDGER` misses) so a long
    session cannot grow it without limit.
    """

    MAX_LEDGER: int = 40
    misses: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.misses.append(message)
        if len(self.misses) > self.MAX_LEDGER:
            del self.misses[: len(self.misses) - self.MAX_LEDGER]

    def describe(self) -> str:
        return "; ".join(self.misses) if self.misses else "nothing was skipped."


__all__ = [
    "EngineSettingsReport",
    "SettingsLedger",
    "VoiceChoice",
    "apply_character_voice",
    "apply_model_settings",
    "resolve_character_prosody",
    "resolve_character_voice",
    "voice_language",
]
