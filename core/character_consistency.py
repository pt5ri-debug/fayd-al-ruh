"""Trait-consistency: warn about a standing character that contradicts itself.

A character's personality is a set of knobs and free-text fields. Each field is
valid on its own -- :meth:`core.character_config.CharacterConfig.validate` already
rejects an out-of-range number -- but a *combination* can still be incoherent: a
formality of ``0.9`` beside a communication style that says "casual", a short
response length beside an objective that demands exhaustive detail, an emotional
expression turned up beside a rule that says never show feeling. A model reading
such a standing instruction gets two opposite orders and follows whichever it
notices.

This module is the one place those combinations are named. :func:`check_consistency`
reads a :class:`core.character_config.CharacterConfig` and returns a
:class:`ConsistencyReport` of **warnings** (never errors): a warning says "these
two fields pull in opposite directions", names the two fields, and suggests the
one edit that resolves it. It is deliberately advisory -- a warning never blocks an
APPLY, because the user may want the tension -- but it is shown *before* the save,
so a contradiction is seen rather than discovered later in the character's speech.

Honest by construction: every rule is a deterministic text/number test over the
configuration's own fields, so the same configuration always yields the same
warnings, and a rule that finds nothing reports nothing (no invented tension).
Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Words in a free-text field that mean "informal". Used to detect a formality
#: dial that points the other way.
_INFORMAL = ("casual", "informal", "slang", "relaxed", "colloquial", "chummy", "عادي", "ودي")
#: Words that mean "formal".
_FORMAL = ("formal", "academic", "official", "professional", "ceremonial", "رسمي", "أكاديمي")
#: Words that mean "very short".
_BRIEF = ("short", "brief", "concise", "terse", "succinct", "موجز", "مختصر")
#: Words that mean "very long / exhaustive".
_EXHAUSTIVE = ("exhaustive", "detailed", "comprehensive", "in-depth", "lengthy", "مفصل", "شامل", "مطول")
#: Words that mean "no emotion shown".
_STOIC = ("never show emotion", "no emotion", "emotionless", "stoic", "detached", "بدون مشاعر", "بلا عاطفة")
#: Words that mean "warm / emotional".
_WARM = ("warm", "emotional", "expressive", "affectionate", "caring", "عاطفي", "دافئ")


@dataclass(frozen=True)
class ConsistencyWarning:
    """One named contradiction between two fields, and how to resolve it."""

    code: str
    fields: tuple[str, ...]
    message: str
    suggestion: str

    def describe(self) -> str:
        return f"{self.message} — {self.suggestion}"


@dataclass(frozen=True)
class ConsistencyReport:
    """Every consistency warning for one configuration (empty means coherent)."""

    key: str
    warnings: tuple[ConsistencyWarning, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the configuration has no contradiction at all."""
        return not self.warnings

    def describe(self) -> str:
        if self.ok:
            return f"{self.key}: the standing character is internally consistent."
        lines = [f"{self.key}: {len(self.warnings)} possible contradiction(s):"]
        lines.extend(f"  ! {w.describe()}" for w in self.warnings)
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "key": self.key,
            "ok": self.ok,
            "warnings": [
                {
                    "code": w.code,
                    "fields": list(w.fields),
                    "message": w.message,
                    "suggestion": w.suggestion,
                }
                for w in self.warnings
            ],
        }


def _text(*values: str) -> str:
    """The lower-cased concatenation of the free-text fields, for a word test."""
    return " ".join(v for v in values if v).lower()


def _contains(haystack: str, needles: tuple[str, ...]) -> str:
    """The first needle found in ``haystack``, or ``""``."""
    for needle in needles:
        if needle in haystack:
            return needle
    return ""


def check_consistency(config) -> ConsistencyReport:
    """Return the consistency warnings for one character configuration.

    ``config`` is a :class:`core.character_config.CharacterConfig`; the function
    reads only its ``personality`` section, so it is safe on a partially built
    object and never writes anything.
    """
    personality = getattr(config, "personality", None)
    key = str(getattr(config, "owner", "") or getattr(config, "key", ""))
    if personality is None:
        return ConsistencyReport(key=key)

    warnings: list[ConsistencyWarning] = []
    style = _text(
        personality.communication_style,
        personality.reasoning_style,
        personality.instructions,
        *personality.traits,
        *personality.behaviour,
    )
    rules = _text(*personality.rules, *personality.objectives, personality.instructions)

    # 1. A high formality dial with a communication style that says "casual".
    informal_word = _contains(style, _INFORMAL)
    if personality.formality >= 0.75 and informal_word:
        warnings.append(
            ConsistencyWarning(
                code="formality-vs-style",
                fields=("formality", "communication_style"),
                message=(
                    f"formality is {personality.formality:.2f} but the style says "
                    f"'{informal_word}'"
                ),
                suggestion="lower formality, or make the communication style match it",
            )
        )
    # 2. A low formality dial with a style that says "formal".
    formal_word = _contains(style, _FORMAL)
    if personality.formality <= 0.25 and formal_word:
        warnings.append(
            ConsistencyWarning(
                code="informality-vs-style",
                fields=("formality", "communication_style"),
                message=(
                    f"formality is {personality.formality:.2f} but the style says "
                    f"'{formal_word}'"
                ),
                suggestion="raise formality, or make the communication style match it",
            )
        )

    # 3. A short response length beside an objective that demands detail.
    exhaustive_word = _contains(rules, _EXHAUSTIVE)
    if personality.response_length == "short" and exhaustive_word:
        warnings.append(
            ConsistencyWarning(
                code="length-vs-objective",
                fields=("response_length", "objectives"),
                message=(
                    f"response_length is 'short' but an objective asks for "
                    f"'{exhaustive_word}'"
                ),
                suggestion="set response_length to 'detailed', or soften the objective",
            )
        )
    # 4. A detailed response length beside a rule that demands brevity.
    brief_word = _contains(rules, _BRIEF)
    if personality.response_length == "detailed" and brief_word:
        warnings.append(
            ConsistencyWarning(
                code="length-vs-rule",
                fields=("response_length", "rules"),
                message=(
                    f"response_length is 'detailed' but a rule asks to be '{brief_word}'"
                ),
                suggestion="set response_length to 'short', or drop the brevity rule",
            )
        )

    # 5. Emotional expression turned up beside a rule that says show no feeling.
    stoic_word = _contains(rules, _STOIC)
    if personality.emotional_expression >= 0.6 and stoic_word:
        warnings.append(
            ConsistencyWarning(
                code="emotion-vs-rule",
                fields=("emotional_expression", "rules"),
                message=(
                    f"emotional_expression is {personality.emotional_expression:.2f} but a "
                    f"rule says '{stoic_word}'"
                ),
                suggestion="lower emotional_expression, or remove the rule that forbids it",
            )
        )
    # 6. Emotional expression turned down beside a style that asks for warmth.
    warm_word = _contains(style, _WARM)
    if personality.emotional_expression <= 0.15 and warm_word:
        warnings.append(
            ConsistencyWarning(
                code="stoicism-vs-style",
                fields=("emotional_expression", "communication_style"),
                message=(
                    f"emotional_expression is {personality.emotional_expression:.2f} but the "
                    f"style says '{warm_word}'"
                ),
                suggestion="raise emotional_expression, or soften the warm wording",
            )
        )

    # 7. The same trait or rule stated twice in opposite spirit is not detectable
    #    in general; what *is* detectable is an exact duplicate, which wastes the
    #    instruction and can confuse a small model.
    for field_name, values in (
        ("traits", personality.traits),
        ("rules", personality.rules),
        ("behaviour", personality.behaviour),
    ):
        seen: set[str] = set()
        for value in values:
            token = value.strip().casefold()
            if token and token in seen:
                warnings.append(
                    ConsistencyWarning(
                        code=f"duplicate-{field_name}",
                        fields=(field_name,),
                        message=f"{field_name} repeats '{value.strip()}'",
                        suggestion=f"remove the duplicate {field_name[:-1]}",
                    )
                )
                break
            seen.add(token)

    # 8. A standing character with nothing to aim at: no objective and no domain.
    if not personality.objectives and not personality.knowledge_domain:
        warnings.append(
            ConsistencyWarning(
                code="no-purpose",
                fields=("objectives", "knowledge_domain"),
                message="the character has neither an objective nor a knowledge domain",
                suggestion="give it at least one objective or name its domain",
            )
        )

    return ConsistencyReport(key=key, warnings=tuple(warnings))


def check_all(configs) -> tuple[ConsistencyReport, ...]:
    """Check a collection of configurations, one report each (order preserved)."""
    return tuple(check_consistency(config) for config in configs)


__all__ = [
    "ConsistencyReport",
    "ConsistencyWarning",
    "check_all",
    "check_consistency",
]
