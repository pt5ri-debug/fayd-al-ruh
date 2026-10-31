"""Compare the characters: tone, length, language, and voice, measurably.

The project ships eight answering identities (the seven characters and its own
model), each with its own configuration: a personality (formality, emotional
expression, preferred answer length, preferred language), a Model section
(temperature, top-p, output cap), and a Voice section (engine, voice, speed,
pitch, volume). Reading all of that across eight characters by hand is tedious,
and "they feel different" is not a measurement.

This module turns those settings into a small, comparable **profile** and a
**contrast** between any two characters, so the differences are visible and
checkable:

* :func:`character_profile` reads one character's effective configuration into a
  :class:`CharacterProfile` -- its tone dials, its answer-length and language
  choices, and its voice settings, each as a readable value plus the raw number;
* :func:`compare_characters` returns a :class:`CharacterContrast` listing, for
  each dimension, both values and which character is higher, lower, or the same;
* :func:`comparison_table` renders every character side by side, one row per
  dimension, so the whole set reads at a glance.

**Honest by construction.** Every value is read from the character's own
effective configuration (the file the Characters window edits), never invented;
a dimension that is *the same* for both characters is reported as the same rather
than forced into a winner; and the "difference score" is the number of dimensions
that genuinely differ, so two characters that were never edited read as close and
two tuned differently read as far apart. Nothing here loads a model, imports Tk,
or reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.character_learning import CHARACTER_KEYS, PROJECT_MODEL_KEY
from core.character_registry import CharacterRegistry, default_registry
from core.learning_isolation import mouth_key, owner_for

#: Every owner this module compares, in the project's order (characters, then the
#: project's own model).
COMPARE_OWNERS: tuple[str, ...] = (*CHARACTER_KEYS, PROJECT_MODEL_KEY)

#: The answer-length choices a character may prefer, as a readable scale. The
#: values are the ones :class:`core.character_config.PersonalityConfig` accepts
#: (it stores the choice as free text), so an unknown value still compares -- it
#: just has no rank, which is reported as ``None`` rather than guessed.
LENGTH_SCALE: dict[str, int] = {
    "very short": 0,
    "short": 1,
    "brief": 1,
    "balanced": 2,
    "moderate": 2,
    "detailed": 3,
    "long": 3,
    "very detailed": 4,
}


@dataclass(frozen=True)
class CharacterProfile:
    """One character's comparable settings: tone, length, language, and voice.

    Every field is a plain value so two profiles can be compared without knowing
    the configuration's shape. ``name`` is the display name, so a table is
    readable; the rest are the dimensions the comparison walks.
    """

    owner: str
    mouth: str
    name: str
    # Tone dials (0..1).
    formality: float
    emotional_expression: float
    # Length and language.
    response_length: str
    preferred_language: str
    # Model.
    temperature: float
    top_p: float
    max_tokens: int
    # Voice.
    voice_engine: str
    voice_key: str
    voice_speed: float
    voice_pitch: float
    voice_volume: float
    # Learning.
    learning_enabled: bool
    private: bool

    @property
    def length_rank(self) -> int | None:
        """The length choice as a rank, or ``None`` for an unrecognised value."""
        return LENGTH_SCALE.get(self.response_length.strip().lower())

    def describe(self) -> str:
        """One readable line naming the character's tone, length, and voice."""
        return (
            f"{self.name} ({self.owner}): formality {self.formality:.2f}, "
            f"expression {self.emotional_expression:.2f}, "
            f"length {self.response_length or 'balanced'}, "
            f"language {self.preferred_language or 'match'}, "
            f"voice {self.voice_key or 'default'}"
            f"{f' ({self.voice_engine})' if self.voice_engine else ''}"
        )


#: The dimensions the comparison walks, each ``(key, readable label)``. The order
#: is the order the table and the contrast show them, so two runs read the same.
DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("formality", "Formality"),
    ("emotional_expression", "Emotional expression"),
    ("response_length", "Preferred answer length"),
    ("preferred_language", "Preferred language"),
    ("temperature", "Temperature"),
    ("top_p", "Top-p"),
    ("max_tokens", "Output cap (tokens)"),
    ("voice_engine", "Voice engine"),
    ("voice_key", "Voice"),
    ("voice_speed", "Voice speed"),
    ("voice_pitch", "Voice pitch"),
    ("voice_volume", "Voice volume"),
    ("learning_enabled", "Learning enabled"),
    ("private", "Private records"),
)


def character_profile(
    key: str,
    *,
    registry: CharacterRegistry | None = None,
) -> CharacterProfile:
    """Read ``key``'s effective configuration into a comparable profile.

    The configuration is loaded through the one registry, so the profile reflects
    what the Characters window last applied rather than a hard-coded default. An
    unknown key still resolves (through :func:`core.learning_isolation.owner_for`)
    and its persona-derived default is used, so a caller never gets an error for a
    stale name.
    """
    owner = owner_for(key)
    registry = registry if registry is not None else default_registry()
    config = registry.load(owner)
    personality = config.personality
    model = config.model
    voice = config.voice
    return CharacterProfile(
        owner=owner,
        mouth=mouth_key(owner),
        name=config.general.name,
        formality=float(personality.formality),
        emotional_expression=float(personality.emotional_expression),
        response_length=str(personality.response_length or ""),
        preferred_language=str(personality.preferred_language or ""),
        temperature=float(model.temperature),
        top_p=float(model.top_p),
        max_tokens=int(model.max_tokens),
        voice_engine=str(voice.engine or ""),
        voice_key=str(voice.voice or ""),
        voice_speed=float(voice.speed),
        voice_pitch=float(voice.pitch),
        voice_volume=float(voice.volume),
        learning_enabled=bool(config.learning.enabled),
        private=owner == "natasha",
    )


def all_profiles(
    keys: tuple[str, ...] | None = None,
    *,
    registry: CharacterRegistry | None = None,
) -> tuple[CharacterProfile, ...]:
    """One profile per character, in the project's order."""
    chosen = COMPARE_OWNERS if keys is None else tuple(keys)
    registry = registry if registry is not None else default_registry()
    return tuple(character_profile(key, registry=registry) for key in chosen)


@dataclass(frozen=True)
class DimensionDiff:
    """How one dimension differs between two characters."""

    key: str
    label: str
    first: object
    second: object
    relation: str  # "same", "first-higher", "second-higher", or "different"

    @property
    def differs(self) -> bool:
        return self.relation != "same"

    def describe(self) -> str:
        """One readable line for the dimension."""
        return f"{self.label}: {self.first!r} vs {self.second!r} ({self.relation})"


@dataclass(frozen=True)
class CharacterContrast:
    """The measurable contrast between two characters, dimension by dimension."""

    first: CharacterProfile
    second: CharacterProfile
    diffs: tuple[DimensionDiff, ...]

    @property
    def differences(self) -> tuple[DimensionDiff, ...]:
        """Only the dimensions that genuinely differ."""
        return tuple(diff for diff in self.diffs if diff.differs)

    @property
    def difference_score(self) -> int:
        """How many dimensions differ -- the plain count, never weighted.

        A high score means the two characters are tuned apart; ``0`` means every
        compared dimension is the same (two characters that were never edited, for
        instance). It is a count, not a judgement of quality.
        """
        return len(self.differences)

    def describe(self) -> str:
        """A short report: the two names and the differing dimensions."""
        lines = [
            f"{self.first.name} vs {self.second.name}: "
            f"{self.difference_score} of {len(self.diffs)} dimensions differ."
        ]
        lines.extend(f"- {diff.describe()}" for diff in self.differences)
        if not self.differences:
            lines.append("- (every compared dimension is the same)")
        return "\n".join(lines)


def _relation(key: str, first: object, second: object) -> str:
    """The relation between two dimension values, ordered when they are numbers."""
    if first == second:
        return "same"
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return "first-higher" if first > second else "second-higher"
    return "different"


def compare_characters(
    key_a: str,
    key_b: str,
    *,
    registry: CharacterRegistry | None = None,
) -> CharacterContrast:
    """Compare two characters dimension by dimension.

    Every dimension in :data:`DIMENSIONS` is reported, with both values and the
    relation between them, so a caller can show exactly *how* two characters
    differ -- not merely that they do.
    """
    registry = registry if registry is not None else default_registry()
    first = character_profile(key_a, registry=registry)
    second = character_profile(key_b, registry=registry)
    diffs = tuple(
        DimensionDiff(
            key=key,
            label=label,
            first=getattr(first, key),
            second=getattr(second, key),
            relation=_relation(key, getattr(first, key), getattr(second, key)),
        )
        for key, label in DIMENSIONS
    )
    return CharacterContrast(first=first, second=second, diffs=diffs)


@dataclass(frozen=True)
class ComparisonTable:
    """Every character side by side, one row per dimension."""

    profiles: tuple[CharacterProfile, ...]
    rows: tuple[tuple[str, tuple[object, ...]], ...]  # (label, one value per profile)

    def describe(self) -> str:
        """A fixed-width table, so the whole set reads at a glance."""
        headers = [profile.name for profile in self.profiles]
        widths: list[int] = []
        for index, header in enumerate(headers):
            widest = max((len(_cell(values[index])) for _label, values in self.rows), default=0)
            widths.append(max(len(header), widest))
        label_width = max((len(label) for label, _values in self.rows), default=0)
        label_width = max(label_width, len("Dimension"))
        header_cells = [f"{header:<{widths[index]}}" for index, header in enumerate(headers)]
        lines = [" | ".join([f"{'Dimension':<{label_width}}", *header_cells])]
        lines.append("-+-".join(["-" * label_width, *("-" * width for width in widths)]))
        for label, values in self.rows:
            cells = [f"{_cell(values[index]):<{widths[index]}}" for index in range(len(headers))]
            lines.append(" | ".join([f"{label:<{label_width}}", *cells]))
        return "\n".join(lines)

    def row(self, label: str) -> tuple[object, ...] | None:
        """One row's values, by label, or ``None`` for an unknown label."""
        for row_label, values in self.rows:
            if row_label == label:
                return values
        return None


def _cell(value: object) -> str:
    """A value as a compact cell (a float keeps two decimals, a bool is yes/no)."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def comparison_table(
    keys: tuple[str, ...] | None = None,
    *,
    registry: CharacterRegistry | None = None,
) -> ComparisonTable:
    """Build the whole-set comparison table."""
    profiles = all_profiles(keys, registry=registry)
    rows: list[tuple[str, tuple[object, ...]]] = []
    for key, label in DIMENSIONS:
        rows.append((label, tuple(getattr(profile, key) for profile in profiles)))
    return ComparisonTable(profiles=profiles, rows=tuple(rows))


def most_different(
    keys: tuple[str, ...] | None = None,
    *,
    registry: CharacterRegistry | None = None,
) -> tuple[tuple[str, str, int], ...]:
    """Every character pair ordered by how many dimensions differ (most first).

    A deterministic ordering (score descending, then by owner names), so the same
    configuration always yields the same ranking. It answers "which two characters
    are the most distinct, and which the closest" as a measurement rather than an
    impression.
    """
    registry = registry if registry is not None else default_registry()
    chosen = COMPARE_OWNERS if keys is None else tuple(keys)
    pairs: list[tuple[str, str, int]] = []
    for index, first in enumerate(chosen):
        for second in chosen[index + 1 :]:
            contrast = compare_characters(first, second, registry=registry)
            pairs.append((owner_for(first), owner_for(second), contrast.difference_score))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return tuple(pairs)


__all__ = [
    "COMPARE_OWNERS",
    "DIMENSIONS",
    "LENGTH_SCALE",
    "CharacterContrast",
    "CharacterProfile",
    "ComparisonTable",
    "DimensionDiff",
    "all_profiles",
    "character_profile",
    "compare_characters",
    "comparison_table",
    "most_different",
]
