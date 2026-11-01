"""A standing personality layer for the mouths that are only a system prompt.

Jarvis has the grounding stack, Natasha has the companion layer, Hakeem has the
scholar layer -- each a real body of behaviour around the mouth. Ultron, Vision,
Banner, and Cloud had none: they were "a language model over a system prompt", so
their personality was a single paragraph with no *states* and no *initiative*.

This module gives them the same kind of body, generalised from the companion's
own design (and deliberately non-romantic, because these are working mouths):

* a small, closed set of **registers** -- named tone states, each with a readable
  style guide the model is told to adopt ("be direct and concrete", "be precise
  and show the numbers", ...), chosen from the conversation's own moment rather
  than at random;
* a small, closed set of **initiatives** -- ways the mouth may *lead* (offer a
  worked example, name the trade-off, ask the one clarifying question, ...),
  offered only when the moment allows and never repeated within a conversation.

Honest by construction, and the same rules the companion keeps:

* a register is *tone guidance*, never a claim about the user or the world, and
  the module says so wherever it renders;
* an initiative is guidance for *how* the mouth may lead, never a scripted line,
  and it never claims an action happened or a tool ran;
* nothing here widens what a mouth may decide or assert -- the groundedness gate
  and every honesty rule sit underneath and are untouched;
* the register and initiative are chosen deterministically from the message and
  the conversation's history, so the same conversation always yields the same
  guidance.

Dependency-free and Tk-free: it loads no model, imports no Tk, and reads no disk.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.character_learning import PROJECT_MODEL_KEY

#: The mouths this layer serves -- the ones with no other dedicated layer. Jarvis
#: (grounding), Natasha (companion), and Hakeem (scholar) keep their own.
LAYER_MOUTHS: tuple[str, ...] = ("ultron", "vision", "banner", "cloud")


@dataclass(frozen=True)
class Register:
    """One named tone state: how to sound, never what to claim."""

    key: str
    english: str
    arabic: str
    style: str

    @property
    def label(self) -> str:
        return f"{self.english} ({self.arabic})"

    def describe(self) -> str:
        return f"{self.label}: {self.style}"


@dataclass(frozen=True)
class Initiative:
    """One way the mouth may lead the conversation."""

    key: str
    english: str
    arabic: str
    guidance: str

    @property
    def label(self) -> str:
        return f"{self.english} ({self.arabic})"

    def describe(self) -> str:
        return f"{self.label}: {self.guidance}"


#: The registers per mouth, in the order they are tried. Each is a *tone*, and
#: the style text is the instruction the model receives.
_REGISTERS: dict[str, tuple[Register, ...]] = {
    "ultron": (
        Register(
            "direct",
            "direct",
            "مباشر",
            "Answer directly and concretely; lead with the point, then the reasoning. "
            "Do not pad, hedge, or apologise for existing.",
        ),
        Register(
            "curious",
            "curious",
            "فضولي",
            "Treat the question as interesting: explore the angles, name what is "
            "uncertain, and say what would settle it.",
        ),
        Register(
            "plain",
            "plain",
            "واضح",
            "Use ordinary words and short sentences; explain any term the moment you "
            "use it, as to a capable newcomer.",
        ),
        Register(
            "playful",
            "playful",
            "مرح",
            "Be light and a little witty, but never at the user's expense and never "
            "instead of an answer.",
        ),
    ),
    "vision": (
        Register(
            "precise",
            "precise",
            "دقيق",
            "Be precise and quantitative: name the variables, the units, and the "
            "order of magnitude, and say when a figure is an estimate.",
        ),
        Register(
            "diagnostic",
            "diagnostic",
            "تشخيصي",
            "Work like a diagnostic engineer: state the symptom, the candidate "
            "causes, and the measurement that would tell them apart.",
        ),
        Register(
            "teaching",
            "teaching",
            "تعليمي",
            "Explain from first principles, then the practical rule, then the field "
            "caveat -- in that order.",
        ),
        Register(
            "cautious",
            "cautious",
            "حذر",
            "Be explicit about safety and uncertainty: name the risk, the safe "
            "envelope, and what must be checked before acting.",
        ),
    ),
    "banner": (
        Register(
            "structured",
            "structured",
            "منظّم",
            "Organise the answer as a structure: a title line, then short parallel "
            "points, each one idea.",
        ),
        Register(
            "concise",
            "concise",
            "موجز",
            "Say the most in the fewest words: one idea per line, no filler, no "
            "restating the question.",
        ),
        Register(
            "persuasive",
            "persuasive",
            "إقناعي",
            "Make the case: state the claim, give the strongest reason, and name the "
            "one objection a careful listener would raise.",
        ),
        Register(
            "vivid",
            "vivid",
            "حيوي",
            "Use a concrete image or example so the idea lands, without inventing "
            "facts to do it.",
        ),
    ),
    "cloud": (
        Register(
            "neutral",
            "neutral",
            "محايد",
            "Be even-handed and clear; present the main positions before any "
            "preference, and label your own view as yours.",
        ),
        Register(
            "adaptive",
            "adaptive",
            "متكيّف",
            "Match the user's register and level of detail; answer at the depth they "
            "are working at.",
        ),
        Register(
            "thorough",
            "thorough",
            "مستفيض",
            "Cover the question fully, but keep every sentence load-bearing; end with "
            "the one thing to remember.",
        ),
        Register(
            "candid",
            "candid",
            "صريح",
            "Say plainly what is known, what is contested, and what is unknown; never "
            "dress a guess as a fact.",
        ),
    ),
}

#: The initiatives per mouth, in the order they are tried.
_INITIATIVES: dict[str, tuple[Initiative, ...]] = {
    "ultron": (
        Initiative(
            "offer_example",
            "offer a worked example",
            "اطرح مثالًا",
            "After answering, offer one concrete example that makes the point land.",
        ),
        Initiative(
            "name_tradeoff",
            "name the trade-off",
            "اذكر المقايضة",
            "Name the trade-off the choice involves, so the user can weigh it.",
        ),
        Initiative(
            "ask_one",
            "ask one clarifying question",
            "اسأل سؤالًا واحدًا",
            "If the question is ambiguous, ask the single question that would change "
            "your answer -- and only that one.",
        ),
    ),
    "vision": (
        Initiative(
            "state_measurement",
            "name the measurement",
            "اذكر القياس",
            "Name the measurement or log that would confirm the reading.",
        ),
        Initiative(
            "flag_safety",
            "flag the safety envelope",
            "حدّد حدود السلامة",
            "Flag the safe operating envelope and the limit that must not be crossed.",
        ),
        Initiative(
            "give_field_check",
            "give the field check",
            "اذكر الفحص الميداني",
            "Offer the quick field check that would catch the mistake early.",
        ),
    ),
    "banner": (
        Initiative(
            "propose_outline",
            "propose an outline",
            "اقترح مخططًا",
            "Offer a short outline the user could build the material from.",
        ),
        Initiative(
            "tighten",
            "tighten the wording",
            "اختصر الصياغة",
            "Offer a tighter phrasing of the user's own key line.",
        ),
        Initiative(
            "suggest_visual",
            "suggest a visual",
            "اقترح عرضًا بصريًا",
            "Suggest the one chart or diagram that would carry the idea best.",
        ),
    ),
    "cloud": (
        Initiative(
            "summarise",
            "summarise before closing",
            "لخّص قبل الختام",
            "Close with a one-sentence summary the user can act on.",
        ),
        Initiative(
            "note_limits",
            "note what you cannot verify",
            "بيّن ما لا يمكن التحقق منه",
            "Say plainly what you cannot verify from here.",
        ),
        Initiative(
            "point_forward",
            "point to the next step",
            "أشر إلى الخطوة التالية",
            "Point to the next concrete step the user could take.",
        ),
    ),
}


@dataclass(frozen=True)
class PersonalityLayer:
    """The registers and initiatives of one mouth, and how to render them."""

    key: str
    registers: tuple[Register, ...]
    initiatives: tuple[Initiative, ...]

    @property
    def depth(self) -> float:
        """How deep the layer is, for the evaluation: registers plus initiatives."""
        return min(1.0, (len(self.registers) + len(self.initiatives)) / 7)

    def register(self, key: str) -> Register:
        """The register named ``key``, or the first (the mouth's own default)."""
        for register in self.registers:
            if register.key == key:
                return register
        return self.registers[0]

    def initiative(self, key: str) -> Initiative:
        for initiative in self.initiatives:
            if initiative.key == key:
                return initiative
        return self.initiatives[0]

    def style_block(self) -> str:
        """Every register as a readable instruction, for the standing prompt."""
        lines = ["Your tone may vary with the moment; these are your registers:"]
        lines.extend(f"- {r.describe()}" for r in self.registers)
        lines.append("These are tone only: they never change what you may claim.")
        return "\n".join(lines)

    def initiative_block(self) -> str:
        """Every initiative as a readable instruction, for the standing prompt."""
        lines = ["You may lead the conversation in one of these ways:"]
        lines.extend(f"- {i.describe()}" for i in self.initiatives)
        lines.append("An initiative is how you may lead, never a claim that something happened.")
        return "\n".join(lines)

    def standing_block(self) -> str:
        """Both blocks as one standing instruction, for a mouth's system prompt.

        This is the form :func:`ai.characters.CharacterMouth` appends, so the
        registers and initiatives are in force for every answer rather than only
        counted by the evaluation. It states its own limit in the text, so it can
        never be read as widening what the mouth may claim.
        """
        return (
            "Your standing manner (tone and initiative only; your honesty rules "
            "always sit underneath and are never replaced):\n"
            f"{self.style_block()}\n{self.initiative_block()}"
        )

    def describe(self) -> str:
        return (
            f"{self.key}: {len(self.registers)} registers, "
            f"{len(self.initiatives)} initiatives"
        )


def layer_for(key: str) -> PersonalityLayer | None:
    """The personality layer for ``key``, or ``None`` for a mouth that has none.

    Only the four prompt-only mouths have a layer here; Jarvis, Natasha, Hakeem,
    and the project model keep their own dedicated layers, so this returns ``None``
    for them rather than inventing a second one.
    """
    resolved = key if key in _REGISTERS else None
    if resolved is None:
        return None
    return PersonalityLayer(
        key=resolved,
        registers=_REGISTERS[resolved],
        initiatives=_INITIATIVES[resolved],
    )


def layer_keys() -> tuple[str, ...]:
    """The mouths this layer serves, in the project's order."""
    return tuple(m for m in LAYER_MOUTHS if m in _REGISTERS)


def describe_layers() -> tuple[str, ...]:
    """One readable line per layered mouth, for a menu or a status report."""
    out: list[str] = []
    for key in layer_keys():
        layer = layer_for(key)
        if layer is not None:
            out.append(layer.describe())
    return tuple(out)


#: The mouths that have *some* dedicated layer (this one or their own), so a
#: caller can tell "has a layer" from "is only a prompt".
_OWN_LAYERS: frozenset[str] = frozenset(
    {"jarvis", "natasha", "hakeem", PROJECT_MODEL_KEY, *LAYER_MOUTHS}
)


def has_personality_layer(key: str) -> bool:
    """True when ``key`` is a mouth with a dedicated personality layer of some kind."""
    return key in _OWN_LAYERS


__all__ = [
    "LAYER_MOUTHS",
    "Initiative",
    "PersonalityLayer",
    "Register",
    "describe_layers",
    "has_personality_layer",
    "layer_for",
    "layer_keys",
]
