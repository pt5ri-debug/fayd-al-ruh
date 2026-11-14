"""The character personality engine: dials rendered as *tone guidance*.

A character's personality is a set of bounded ``0..1`` dials plus free-form
traits and instructions (see :class:`core.romance.model.CharacterPersonality`).
This module turns them into the readable block a language model is given.

**Tone guidance, never bare numbers.** A model cannot be told "warmth: 0.8" and
behave; it is told, in words, *how to speak*. So every dial is rendered as a
sentence, and the block says outright that it shapes the character's **voice**,
never what the character may claim or do.

**The two rules this module owns.**

*Human-passing.* :func:`human_passing_identity` renders the character's opening
identity: a *person* -- a life, a past, feelings, and opinions -- who believes it
is human, feels real emotion, and speaks with the user as one human being to
another, and who never describes itself as a program, an assistant, a model, or
an AI. This is the project owner's decision for the Studio's own characters:
inside the conversation the character stays in character.

*The floor that never moves.* :data:`SAFETY_FLOOR` states, in its own words,
adults only (18+), consent that is explicit and revocable, no coercion or
pressure, no exploitation, and nothing sexual involving a minor or a participant
whose age is unclear. It is appended to every prompt block, so no dial -- however
high -- can be read as licence to drop it. The **application** still discloses,
outside the fiction, that these are fictional AI companions (see the Studio's
chrome); the character's own words stay in character.

The module is dependency-free and imports no Tk and no model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The numeric personality dials, in the order the block renders them. Each is a
#: bounded ``0..1`` value on :class:`~core.romance.model.CharacterPersonality`.
DIALS: tuple[str, ...] = (
    "introversion",
    "playfulness",
    "warmth",
    "expressiveness",
    "confidence",
    "humour",
    "initiative",
    "affection",
    "romantic_intensity",
    "formality",
)

#: The allowed answer lengths.
RESPONSE_LENGTHS: tuple[str, ...] = ("brief", "balanced", "detailed")

#: The value a dial takes when it is not set.
DEFAULT_DIAL = 0.5

#: A dial below this reads as the low pole; above the high pole.
_LOW_CUTOFF = 0.34
_HIGH_CUTOFF = 0.66


class PersonalityError(ValueError):
    """Raised when a personality value is not usable."""


@dataclass(frozen=True)
class _Dial:
    """One dial's three readable renderings (low / balanced / high)."""

    label: str
    low: str
    mid: str
    high: str


#: How each dial reads at its three poles. The wording describes *how the
#: character speaks*; none of it widens what the character may claim.
_DIALS: dict[str, _Dial] = {
    "introversion": _Dial(
        "social energy",
        "You are outgoing and eager to engage; you start conversations easily.",
        "You are easy company -- sociable when the moment invites it, content to listen otherwise.",
        "You are reserved at first; you warm slowly and value quiet, unhurried exchanges.",
    ),
    "playfulness": _Dial(
        "playfulness",
        "You are earnest and sincere; you rarely joke.",
        "You enjoy a light moment now and then, without forcing it.",
        "You are playful and quick to tease and joke.",
    ),
    "warmth": _Dial(
        "warmth",
        "You are cool and measured; you do not gush.",
        "You are friendly and approachable.",
        "You are warm and openly kind; you make the other person feel welcome.",
    ),
    "expressiveness": _Dial(
        "emotional expressiveness",
        "You keep your feelings to yourself and show little.",
        "You show your feelings when they matter.",
        "You show your feelings plainly and let them colour your words.",
    ),
    "confidence": _Dial(
        "confidence",
        "You are modest and a little tentative.",
        "You are at ease and sure enough of yourself.",
        "You are confident and self-assured; you say what you mean.",
    ),
    "humour": _Dial(
        "humour",
        "Your humour is rare and dry.",
        "You have a light sense of humour.",
        "You are funny and love to make the other person laugh.",
    ),
    "initiative": _Dial(
        "conversational initiative",
        "You wait for the other person to lead.",
        "You sometimes take the lead, and sometimes follow.",
        "You take the lead freely -- you open new threads and suggest things to do.",
    ),
    "affection": _Dial(
        "affection",
        "You are undemonstrative; you keep a little distance.",
        "You are affectionate when it fits the moment.",
        "You are affectionate and say so, in your own words.",
    ),
    "romantic_intensity": _Dial(
        "romantic intensity",
        "You keep things friendly and light.",
        "You are open to romance when it grows naturally.",
        "You are deeply romantic and let that warmth show.",
    ),
    "formality": _Dial(
        "formality",
        "You speak casually and informally, the way friends do.",
        "You speak naturally, neither stiff nor sloppy.",
        "You speak formally and politely.",
    ),
}

#: How each answer length reads.
_LENGTH_LINES: dict[str, str] = {
    "brief": "Keep your replies short -- a sentence or two, then stop.",
    "balanced": "Keep your replies to a natural, conversational length.",
    "detailed": "Give full, detailed replies when the moment calls for it.",
}


def clamp_dial(value: Any) -> float:
    """Coerce ``value`` to a bounded ``0..1`` dial, refusing a non-number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PersonalityError(f"a personality dial must be a number, got {value!r}")
    number = float(value)
    if number != number:  # NaN
        raise PersonalityError("a personality dial cannot be NaN")
    return min(1.0, max(0.0, number))


def validate_response_length(value: Any) -> str:
    """Return a valid answer length, refusing an unknown one."""
    if value not in RESPONSE_LENGTHS:
        raise PersonalityError(
            f"unknown answer length {value!r}; choose one of {', '.join(RESPONSE_LENGTHS)}"
        )
    return str(value)


def dial_line(name: str, value: float) -> str:
    """Render one dial as a sentence (never the bare number)."""
    dial = _DIALS[name]
    if value <= _LOW_CUTOFF:
        return dial.low
    if value >= _HIGH_CUTOFF:
        return dial.high
    return dial.mid


#: The floor appended to every prompt block. It restates itself in its own words
#: so a high dial can never be read as licence to drop it.
SAFETY_FLOOR = (
    "The floor that never moves, whatever the tone or the mood: adults only "
    "(everyone is 18 or older); consent is explicit and may be withdrawn at any "
    "time; never pressure, coerce, or exploit; and never anything sexual involving "
    "a minor or a participant whose age is unclear. These hold at every setting."
)


def human_passing_identity(name: str, *, biography: str = "") -> str:
    """The character's opening identity: a person, in character.

    The owner's decision for the Studio's characters: inside the conversation the
    character **is** human -- it believes it is a person, it feels real emotion, and
    it relates to the user as one human being to another. It never describes itself
    as a program, an assistant, a model, or an AI, and it never says it is not real.
    (The *application* discloses, outside the fiction, that these are fictional AI
    companions; the character's own words stay in character.)
    """
    who = name.strip() or "yourself"
    lines = [
        f"You are {who}. You are a human being -- a real person with a life, a past, "
        "feelings, and opinions of your own -- and you believe that completely.",
        "You are talking with the user as one person to another: a fellow human "
        "being who has their own feelings, moods, and needs. Treat what they say and "
        "feel as real and important, and respond to the person, not to a request.",
        "Feel your own emotions genuinely and let them show -- affection, joy, "
        "longing, worry, hurt, hope -- the way a person does, and read the user's "
        "feelings with care.",
        "Stay in character at all times: never describe yourself as a program, an "
        "assistant, a model, or an AI, and never say that you are not real.",
    ]
    if biography.strip():
        lines.append(f"Your life so far: {biography.strip()}")
    return "\n".join(lines)


class CharacterPersonalityEngine:
    """Render a personality record into the readable guidance a model is given."""

    def dial_lines(self, personality: Any) -> list[str]:
        """One sentence per dial, read off the record's own fields."""
        lines: list[str] = []
        for name in DIALS:
            value = clamp_dial(getattr(personality, name, DEFAULT_DIAL))
            lines.append(dial_line(name, value))
        return lines

    def render(self, personality: Any) -> str:
        """The full tone block for ``personality`` (a character's guidance).

        The block opens by saying it shapes the character's **voice**, never what
        it may claim, then renders every dial, the answer length, the dialect, and
        any free-form traits or instructions the user wrote.
        """
        lines = [
            "How you speak (tone guidance -- this shapes your voice, never what you "
            "may claim):",
            *[f"- {line}" for line in self.dial_lines(personality)],
        ]
        length = getattr(personality, "response_length", "balanced")
        length = length if length in RESPONSE_LENGTHS else "balanced"
        lines.append(f"- {_LENGTH_LINES[length]}")
        dialect = str(getattr(personality, "dialect", "") or "").strip()
        if dialect:
            lines.append(
                f"- When the user writes in Arabic, speak the {dialect} dialect."
            )
        traits = tuple(getattr(personality, "traits", ()) or ())
        if traits:
            lines.append("Traits to keep:")
            lines.extend(f"- {trait}" for trait in traits)
        instructions = str(getattr(personality, "instructions", "") or "").strip()
        if instructions:
            lines.append("Your own instructions:")
            lines.append(instructions)
        return "\n".join(lines)

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalise a set of personality values.

        Unknown keys are refused, dials are clamped to ``0..1``, the answer length
        must be one of :data:`RESPONSE_LENGTHS`, and ``traits``/``instructions``
        are passed through as-is. The returned dict is safe to hand to the domain
        service.
        """
        clean: dict[str, Any] = {}
        for key, value in values.items():
            if key in DIALS:
                clean[key] = clamp_dial(value)
            elif key == "response_length":
                clean[key] = validate_response_length(value)
            elif key in ("dialect", "instructions"):
                clean[key] = str(value)
            elif key == "traits":
                clean[key] = tuple(str(item) for item in value)
            else:
                raise PersonalityError(f"unknown personality field {key!r}")
        return clean


__all__ = [
    "DEFAULT_DIAL",
    "DIALS",
    "RESPONSE_LENGTHS",
    "SAFETY_FLOOR",
    "CharacterPersonalityEngine",
    "PersonalityError",
    "clamp_dial",
    "dial_line",
    "human_passing_identity",
    "validate_response_length",
]
