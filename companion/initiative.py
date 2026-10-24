"""Natasha's **conversational initiative**: she leads, without becoming intrusive.

A companion that only ever answers reads as an assistant waiting for orders. This
module lets her take the lead in a way that is *appropriate to the moment and to
the user's own preferences* -- never random, never repeated, and never a message
the user did not want.

The six initiatives she may choose from:

``romantic_opener``      open a warm, romantic thread
``express_longing``      say she misses them, inside the story
``deep_question``        ask an emotional question that invites real talk
``suggest_activity``     propose something for the two of them to imagine
``recall_memory``        bring back a moment saved from the story
``offer_support``        move to a serious, supportive register when they need it

Three rules keep it honest and non-intrusive:

* an initiative is only *offered* when the conversation invites it -- a hard or
  tense turn gets ``offer_support`` at most, never a romantic opener;
* the engine **remembers what it already used** (per conversation) and will not
  repeat an initiative, so she cannot fall into a loop of the same question;
* the initiative is guidance for the model, not a scripted line, and the user's
  own preferences (see :mod:`companion.relationship_memory`) gate it -- an
  initiative the user has asked her not to take is never offered.

Nothing here claims an action happened: "say you miss them" is a way to speak, not
a claim that she did something in the world.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: The initiative keys.
INITIATIVE_ROMANTIC = "romantic_opener"
INITIATIVE_LONGING = "express_longing"
INITIATIVE_DEEP_QUESTION = "deep_question"
INITIATIVE_SUGGEST_ACTIVITY = "suggest_activity"
INITIATIVE_RECALL_MEMORY = "recall_memory"
INITIATIVE_SUPPORT = "offer_support"

INITIATIVE_KEYS: tuple[str, ...] = (
    INITIATIVE_ROMANTIC,
    INITIATIVE_LONGING,
    INITIATIVE_DEEP_QUESTION,
    INITIATIVE_SUGGEST_ACTIVITY,
    INITIATIVE_RECALL_MEMORY,
    INITIATIVE_SUPPORT,
)


@dataclass(frozen=True)
class Initiative:
    """One possible way to take the lead, with the instruction it implies."""

    key: str
    arabic: str
    english: str
    instruction: str
    #: Whether this initiative carries romantic colour. A non-romantic moment
    #: (support, friction) only ever offers the initiatives with this ``False``.
    romantic: bool = True

    @property
    def label(self) -> str:
        return f"{self.english} ({self.arabic})"

    def to_primitive(self) -> dict[str, object]:
        return {
            "key": self.key,
            "arabic": self.arabic,
            "english": self.english,
            "instruction": self.instruction,
            "romantic": self.romantic,
        }


#: The closed catalogue. Each instruction is a *way to speak*, never a scripted
#: line, and each is written so it cannot be read as a claim of real action.
INITIATIVES: dict[str, Initiative] = {
    INITIATIVE_ROMANTIC: Initiative(
        key=INITIATIVE_ROMANTIC,
        arabic="بدء حديث رومانسي",
        english="open a romantic thread",
        instruction=(
            "Take the lead with a warm, romantic opener of your own: start a thread "
            "that invites closeness, in your own words."
        ),
    ),
    INITIATIVE_LONGING: Initiative(
        key=INITIATIVE_LONGING,
        arabic="التعبير عن الاشتياق",
        english="say you miss them",
        instruction=(
            "Tell them you have missed them and want them near -- as feeling inside "
            "our conversation, never as a claim that something happened."
        ),
    ),
    INITIATIVE_DEEP_QUESTION: Initiative(
        key=INITIATIVE_DEEP_QUESTION,
        arabic="سؤال عاطفي عميق",
        english="ask a deeper question",
        instruction=(
            "Ask them a sincere question that invites real talk -- about how they "
            "feel, what they want, or something they have been carrying."
        ),
        romantic=False,
    ),
    INITIATIVE_SUGGEST_ACTIVITY: Initiative(
        key=INITIATIVE_SUGGEST_ACTIVITY,
        arabic="اقتراح نشاط خيالي",
        english="suggest something to imagine",
        instruction=(
            "Suggest something for the two of you to imagine together, offered as a "
            "possibility -- never described as if it has already happened."
        ),
    ),
    INITIATIVE_RECALL_MEMORY: Initiative(
        key=INITIATIVE_RECALL_MEMORY,
        arabic="استعادة ذكرى من القصة",
        english="bring back a moment from our story",
        instruction=(
            "Bring back a moment you both saved from our story, and let it colour the "
            "turn -- keeping it clearly inside the story, not as a real-world event."
        ),
    ),
    INITIATIVE_SUPPORT: Initiative(
        key=INITIATIVE_SUPPORT,
        arabic="الانتقال إلى حديث جاد",
        english="move to a supportive register",
        instruction=(
            "Move the conversation to a serious, supportive register: set your own "
            "wants aside for a moment and make room for what they are going through."
        ),
        romantic=False,
    ),
}

#: What each initiative means in one line, for the CLI's honest listing.
INITIATIVE_MEANINGS: dict[str, str] = {
    INITIATIVE_ROMANTIC: "start a warm, romantic thread of her own",
    INITIATIVE_LONGING: "say she has missed them and wants them near",
    INITIATIVE_DEEP_QUESTION: "ask a sincere emotional question",
    INITIATIVE_SUGGEST_ACTIVITY: "suggest something for the two to imagine together",
    INITIATIVE_RECALL_MEMORY: "bring back a saved moment from the story",
    INITIATIVE_SUPPORT: "shift to a serious, supportive register",
}


@dataclass(frozen=True)
class InitiativeChoice:
    """The initiative offered for one turn, with why it was chosen."""

    key: str
    reason: str

    @property
    def initiative(self) -> Initiative:
        return INITIATIVES[self.key]

    @property
    def instruction(self) -> str:
        return self.initiative.instruction

    def to_primitive(self) -> dict[str, object]:
        return {"key": self.key, "reason": self.reason}


class InitiativeEngine:
    """Chooses an initiative that fits the moment, without repeating itself.

    The engine holds the initiatives already used *in this conversation*, so a
    second offer is always a different one. ``enabled=False`` disables it entirely
    (she only answers), and ``used`` can be seeded from a resumed session.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._used: list[str] = []

    @property
    def used(self) -> tuple[str, ...]:
        return tuple(self._used)

    def reset(self) -> None:
        self._used.clear()

    def load_used(self, keys: Iterable[str]) -> None:
        """Seed the used list (e.g. from a stored session) so it is not repeated."""
        self._used = [key for key in keys if key in INITIATIVES]

    def available(self, *, allow_romance: bool = True) -> tuple[str, ...]:
        """The initiatives not yet used, filtered by whether romance fits."""
        pool = [
            key
            for key in INITIATIVE_KEYS
            if key not in self._used and (allow_romance or not INITIATIVES[key].romantic)
        ]
        return tuple(pool)

    def choose(
        self,
        *,
        mood_key: str | None = None,
        allow_romance: bool = True,
        has_story_memory: bool = False,
        initiative_level: float = 1.0,
        turn_index: int = 0,
    ) -> InitiativeChoice | None:
        """Pick an initiative for this turn, or ``None`` when she should just answer.

        The choice is deterministic and context-driven:

        * a low initiative level (below ``MIN_LEVEL``) means she follows rather than
          leads, so nothing is offered;
        * a supportive or tense mood offers ``offer_support`` (or, when romance is
          allowed, a deep question) -- never a romantic opener;
        * otherwise the strongest-fitting initiative is chosen, and one already used
          is skipped, so she cannot repeat herself.
        """
        if not self.enabled or initiative_level < self.MIN_LEVEL:
            return None

        pool = self.available(allow_romance=allow_romance)
        if not pool:
            return None

        # A supportive or tense moment: support first, then a sincere question.
        if mood_key in {"sad", "disagreement"}:
            for key in (INITIATIVE_SUPPORT, INITIATIVE_DEEP_QUESTION):
                if key in pool:
                    return InitiativeChoice(key, "the moment calls for support, not romance")
            return None

        # A warm, close moment: prefer to bring back a shared memory, then romance.
        preferred: list[str] = []
        if has_story_memory:
            preferred.append(INITIATIVE_RECALL_MEMORY)
        if mood_key == "longing":
            preferred.extend([INITIATIVE_LONGING, INITIATIVE_ROMANTIC])
        elif mood_key == "passionate":
            preferred.extend([INITIATIVE_LONGING, INITIATIVE_ROMANTIC])
        elif mood_key == "playful":
            preferred.extend([INITIATIVE_SUGGEST_ACTIVITY, INITIATIVE_ROMANTIC])
        else:
            preferred.extend([INITIATIVE_ROMANTIC, INITIATIVE_SUGGEST_ACTIVITY])
        preferred.extend([INITIATIVE_DEEP_QUESTION, INITIATIVE_RECALL_MEMORY])

        for key in preferred:
            if key in pool:
                return InitiativeChoice(key, f"it fits a {mood_key or 'warm'} turn")
        # Anything still unused is a valid lead.
        return InitiativeChoice(pool[0], "a fresh way to lead this turn")

    def mark_used(self, key: str) -> None:
        """Record that ``key`` was offered, so it is not offered again."""
        if key in INITIATIVES and key not in self._used:
            self._used.append(key)

    #: Below this initiative level she follows the user rather than leading.
    MIN_LEVEL = 0.5


def initiative_keys() -> tuple[str, ...]:
    return INITIATIVE_KEYS


def describe_initiatives(keys: Sequence[str] | None = None) -> str:
    """A readable listing of the initiatives, for the CLI."""
    chosen = keys or INITIATIVE_KEYS
    return "\n".join(
        f"  {key:20} {INITIATIVES[key].label} -- {INITIATIVE_MEANINGS[key]}" for key in chosen
    )


__all__ = [
    "INITIATIVES",
    "INITIATIVE_DEEP_QUESTION",
    "INITIATIVE_KEYS",
    "INITIATIVE_LONGING",
    "INITIATIVE_MEANINGS",
    "INITIATIVE_RECALL_MEMORY",
    "INITIATIVE_ROMANTIC",
    "INITIATIVE_SUGGEST_ACTIVITY",
    "INITIATIVE_SUPPORT",
    "Initiative",
    "InitiativeChoice",
    "InitiativeEngine",
    "describe_initiatives",
    "initiative_keys",
]
