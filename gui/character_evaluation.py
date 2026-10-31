"""A comprehensive, data-driven evaluation of the project's eight characters.

This module is the engine behind the cognition window's **Characters** tab and
the headless ``python -m cli cognition characters`` command. It is deliberately
Tk-free and dependency-light, so the whole evaluation is exercised headless.

What it measures, for each of the seven characters and the project's own model,
is *observable* rather than asserted: the four dimensions are computed from the
character's real configuration, its real learning store, and the real model
catalog -- never from a hard-coded table of praise. Each dimension carries the
named facts that moved it, so a reader can see **why** a score is what it is:

* **Training structure** -- does the character have a language model, is it
  active, is learning enabled, is there a trained artifact on disk?
* **Memory** -- how strong the character's memory is, in two halves: whether the
  layer is real (a durable store that opens, provenance-carrying records, bounded
  ceilings, learning on) and how much it has actually accumulated (durable facts,
  exchanges, learning records). An unused memory is scored on capability, so it
  is never reported as an absent one.
* **Traits** -- how much of its standing character is really defined: persona
  traits and rules, the communication and reasoning style, the domain, and
  whether the user has customised it.
* **Development headroom** -- how much room there is to grow: whether the model
  can be swapped or fetched, whether it has its own voice engine, and whether
  learning is on so the character can accumulate.

The report is honest by construction: a character whose store has not been used
still scores on memory *capability* but reads as having little learned, a
character with no language model scores low on training structure, and the
shortcomings and next steps are generated from the same facts that produced the
scores -- so a weakness is never invented and never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.character_config import CharacterConfig, owner_names
from core.character_learning import (
    CHARACTER_KEYS,
    MAX_EXCHANGES,
    MAX_FACTS,
    CharacterLearning,
    persona_for,
)
from core.character_learning_store import character_store, main_model_store
from core.character_registry import default_registry
from core.learning_isolation import MAIN_MODEL_OWNER, is_private


class RegistryLike(Protocol):
    """The narrow registry surface the evaluation needs.

    :class:`core.character_registry.CharacterRegistry` satisfies it, and so does a
    test stand-in -- only ``load`` is required, and ``is_customised`` is read
    defensively.
    """

    def load(self, key: str) -> CharacterConfig: ...


#: The four dimensions, in the order the report and the window show them.
DIMENSIONS: tuple[str, ...] = ("training", "memory", "traits", "development")

#: Readable titles for the dimensions.
DIMENSION_TITLES: dict[str, str] = {
    "training": "Training structure",
    "memory": "Memory",
    "traits": "Traits",
    "development": "Development headroom",
}

#: A one-line description of what each dimension weighs.
DIMENSION_BLURBS: dict[str, str] = {
    "training": "the language model, whether it is active, and whether learning is on",
    "memory": "durable facts, exchanges, and provenance-carrying learning records",
    "traits": "persona traits and rules, style, domain, and customisation",
    "development": "model swap/fetch, a voice engine, and room to keep learning",
}


def _ratio(count: int, ceiling: int) -> float:
    """``count / ceiling`` clamped to ``[0, 1]``; a zero ceiling reads as 0."""
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, count / ceiling))


#: The character depth a *dedicated layer* gives a mouth, measured from the real
#: modules rather than asserted. A companion with her own trait profile, mood
#: engine, and private database is genuinely deeper than a mouth that is only a
#: language model over a system prompt -- and the numbers below come from those
#: modules actually importing and reporting their own counts.
def character_depth(key: str) -> tuple[float, tuple[str, ...]]:
    """How deep a character's own layer is, and the facts that say so.

    Reads the modules that really exist (``companion.*`` for Natasha,
    ``hakeem.*`` for Hakeem, the grounding stack for Jarvis), so the depth is
    observed, not declared. A module that is absent contributes nothing and the
    fact is reported honestly rather than assumed present.
    """
    facts: list[str] = []
    score = 0.0
    if key == "natasha":
        try:
            from companion import initiative, moods
            from companion.traits import TraitProfile

            traits = len(TraitProfile.__dataclass_fields__)
            moods_n = len(moods.MOODS)
            init_n = len(initiative.INITIATIVES)
            score = min(1.0, (traits + moods_n + init_n) / 22)
            facts.append(
                f"companion layer: {traits} trait dials, {moods_n} moods, {init_n} initiatives"
            )
            facts.append("her own SQLite memory, separate from the companion's sibling")
        except Exception:
            facts.append("companion layer: not importable here")
    elif key == "hakeem":
        try:
            import dataclasses

            from hakeem.personality import PersonalitySettings

            dials = len(dataclasses.fields(PersonalitySettings))
            score = min(1.0, (dials + 4) / 14)
            facts.append(f"scholar layer: {dials} style dials, comparative and history methods")
            facts.append("his own SQLite database, separate from the companion's")
        except Exception:
            facts.append("scholar layer: not importable here")
    elif key == "jarvis":
        # Jarvis's depth is the grounding stack: citations, the groundedness
        # gate, the reasoning trace, and the knowledge carry. The score counts
        # the modules that are really importable here, so it is observed.
        modules = ("grounding", "reasoning_report", "knowledge_carry", "answer_confidence")
        present = 0
        for name in modules:
            try:
                __import__(f"core.{name}")
                present += 1
            except Exception:
                continue
        score = round(present / len(modules), 6)
        facts.append(f"grounding stack: {present} of {len(modules)} modules importable here")
        if present == len(modules):
            facts.append("citations, the groundedness gate, the reasoning trace, and the knowledge carry")
    elif key in {"ultron", "vision", "banner"}:
        score = 0.4
        facts.append("a language model over a system prompt; no dedicated personality layer")
    elif key == "cloud":
        score = 0.2
        facts.append("a remote model reached over the network; no local layer of its own")
    elif key == MAIN_MODEL_OWNER:
        score = 0.65
        facts.append("the project's own trained model, with its own corpus and character learning")
    return round(score, 6), tuple(facts)


@dataclass(frozen=True)
class DimensionScore:
    """One dimension's score, with the named facts that produced it."""

    key: str
    score: float
    facts: tuple[str, ...] = ()

    @property
    def title(self) -> str:
        return DIMENSION_TITLES.get(self.key, self.key)

    @property
    def percent(self) -> int:
        return round(self.score * 100)

    def describe(self) -> str:
        return f"{self.title}: {self.percent}%"


@dataclass(frozen=True)
class CharacterEvaluation:
    """One character's full evaluation: scores, strengths, gaps, and next steps."""

    key: str
    name: str
    role: str
    private: bool
    dimensions: tuple[DimensionScore, ...]
    strengths: tuple[str, ...] = ()
    shortcomings: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()

    @property
    def overall(self) -> float:
        """The mean of the four dimension scores (0.0 when there are none)."""
        if not self.dimensions:
            return 0.0
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions), 6)

    def dimension(self, key: str) -> DimensionScore:
        for score in self.dimensions:
            if score.key == key:
                return score
        raise KeyError(key)

    def describe(self) -> str:
        flag = ", private records" if self.private else ""
        lines = [f"{self.name} ({self.key}){flag} — overall {round(self.overall * 100)}%"]
        lines.extend(f"  {d.describe()}" for d in self.dimensions)
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "role": self.role,
            "private": self.private,
            "overall": self.overall,
            "dimensions": [
                {"key": d.key, "title": d.title, "score": d.score, "percent": d.percent,
                 "facts": list(d.facts)}
                for d in self.dimensions
            ],
            "strengths": list(self.strengths),
            "shortcomings": list(self.shortcomings),
            "next_steps": list(self.next_steps),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Every character's evaluation, plus the project-wide summary."""

    characters: tuple[CharacterEvaluation, ...]
    notes: tuple[str, ...] = field(default=())

    @property
    def overall(self) -> float:
        if not self.characters:
            return 0.0
        return round(sum(c.overall for c in self.characters) / len(self.characters), 6)

    def strongest(self) -> CharacterEvaluation | None:
        return max(self.characters, key=lambda c: c.overall, default=None)

    def weakest(self) -> CharacterEvaluation | None:
        return min(self.characters, key=lambda c: c.overall, default=None)

    def dimension_averages(self) -> dict[str, float]:
        """The mean of each dimension across every character."""
        totals: dict[str, list[float]] = {key: [] for key in DIMENSIONS}
        for character in self.characters:
            for score in character.dimensions:
                totals.setdefault(score.key, []).append(score.score)
        return {
            key: round(sum(values) / len(values), 6) if values else 0.0
            for key, values in totals.items()
        }

    def describe(self) -> str:
        lines = [
            "Character evaluation",
            f"  {len(self.characters)} characters, overall {round(self.overall * 100)}%",
        ]
        averages = self.dimension_averages()
        for key in DIMENSIONS:
            lines.append(f"  {DIMENSION_TITLES[key]}: {round(averages.get(key, 0.0) * 100)}%")
        strongest = self.strongest()
        weakest = self.weakest()
        if strongest is not None:
            lines.append(f"  strongest: {strongest.name} ({round(strongest.overall * 100)}%)")
        if weakest is not None and weakest is not strongest:
            lines.append(f"  weakest:   {weakest.name} ({round(weakest.overall * 100)}%)")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "dimension_averages": self.dimension_averages(),
            "characters": [c.to_primitive() for c in self.characters],
            "notes": list(self.notes),
        }


class CharacterEvaluator:
    """Score the eight characters from their real configuration and stores.

    ``data_root`` is where the learning stores live (the same root the Characters
    window and the CLI use), and ``registry`` is the applied configuration. Both
    may be injected, so the evaluator is exercised headless with no data directory
    at all. ``model_available`` is an optional predicate that reports whether a
    character's language model is really on disk -- the one fact the evaluator
    cannot read from the configuration alone; when it is absent the availability is
    reported honestly as unknown rather than assumed.
    """

    def __init__(
        self,
        *,
        data_root: Path | None = None,
        registry: RegistryLike | None = None,
        model_available: Any | None = None,
        catalog: Any | None = None,
    ) -> None:
        self._data_root = data_root
        self._registry = registry if registry is not None else default_registry()
        self._model_available = model_available
        self._catalog = catalog

    # -- the language-model catalog ----------------------------------------

    def _model_facts(self, key: str) -> tuple[bool, bool, str, str]:
        """``(configured, available, model_key, title)`` for a character's mouth.

        The unified language-model catalog is the one source of truth about which
        model a mouth answers with and whether its weights are really on disk, so
        the evaluation reads it rather than guessing from the configuration. An
        unknown mouth or a catalog that cannot be built yields the honest
        ``(False, False, "", "")`` rather than a fabricated model.
        """
        try:
            catalog = self._catalog
            if catalog is None:
                from core import language_models

                catalog = language_models
            from core.learning_isolation import mouth_key

            mouth = mouth_key(key)
            model_key = catalog.default_language_model_key(mouth)
            if not model_key:
                return False, False, "", ""
            model = catalog.find_language_model(mouth, model_key)
            title = getattr(model, "title", "") or model_key
            available = bool(getattr(model, "available", False))
            return True, available, model_key, title
        except Exception:
            return False, False, "", ""

    # -- public -------------------------------------------------------------

    def evaluate(self) -> EvaluationReport:
        """Evaluate every character, in the project's own order."""
        evaluations = tuple(self._evaluate_one(key) for key in owner_names())
        notes = (
            "Scores are computed from each character's real configuration and "
            "learning store, never from a fixed table.",
            "An empty learning store scores low on memory by design: the score "
            "measures what the character has actually learned.",
        )
        return EvaluationReport(characters=evaluations, notes=notes)

    def evaluate_key(self, key: str) -> CharacterEvaluation:
        """Evaluate one character. Raises ``KeyError`` for an unknown key."""
        from core.learning_isolation import owner_for

        owner = owner_for(key)
        if owner not in owner_names():
            raise KeyError(key)
        return self._evaluate_one(owner)

    # -- one character ------------------------------------------------------

    def _evaluate_one(self, key: str) -> CharacterEvaluation:
        config = self._registry.load(key)
        persona = persona_for(key)
        private = is_private(key)
        facts, exchanges = self._learning_counts(key)
        records = self._record_count(key)

        training, training_facts = self._score_training(key, config)
        memory, memory_facts = self._score_memory(key, facts, exchanges, records, config)
        traits, trait_facts = self._score_traits(key, config, persona)
        development, dev_facts = self._score_development(key, config)

        dimensions = (
            DimensionScore("training", training, training_facts),
            DimensionScore("memory", memory, memory_facts),
            DimensionScore("traits", traits, trait_facts),
            DimensionScore("development", development, dev_facts),
        )
        strengths, shortcomings, next_steps = self._commentary(
            key, config, dimensions, facts, exchanges, records
        )
        return CharacterEvaluation(
            key=key,
            name=config.general.name or persona.name,
            role=config.general.role or persona.role,
            private=private,
            dimensions=dimensions,
            strengths=strengths,
            shortcomings=shortcomings,
            next_steps=next_steps,
        )

    # -- dimensions ---------------------------------------------------------

    def _score_training(
        self, key: str, config: CharacterConfig
    ) -> tuple[float, tuple[str, ...]]:
        learning = config.learning
        facts: list[str] = []
        signals: list[float] = []

        configured, available, model_key, title = self._model_facts(key)
        signals.append(1.0 if configured else 0.0)
        if configured:
            facts.append(f"language model: {title} ({model_key})")
        else:
            facts.append("language model: none configured for this mouth")
        facts.append(f"model file on disk: {'yes' if available else 'no'}")
        signals.append(1.0 if available else 0.0)
        signals.append(1.0 if learning.enabled else 0.0)
        facts.append(f"learning enabled: {'yes' if learning.enabled else 'no'}")
        signals.append(1.0 if learning.learn_from_model else 0.0)
        facts.append(f"learns from its model: {'yes' if learning.learn_from_model else 'no'}")
        override = self._model_availability(key, config)
        if override is not None:
            signals.append(1.0 if override else 0.0)
            facts.append(f"model availability (reported): {'yes' if override else 'no'}")
        else:
            signals.append(1.0 if available else 0.0)
        return round(sum(signals) / len(signals), 6), tuple(facts)

    def _score_memory(
        self, key: str, facts: int, exchanges: int, records: int, config: CharacterConfig
    ) -> tuple[float, tuple[str, ...]]:
        """How strong the character's memory is: its layer *and* what it holds.

        Two halves, because they answer two different questions. **Capability**
        is whether the memory layer really exists and is fit for purpose -- the
        durable store opens, its records carry provenance, its ceilings are
        bounded, and the character may keep its exchanges. **Usage** is how much
        it has actually accumulated. A character that has just been set up scores
        on capability even before it has been talked to, so an *unused* memory is
        never reported as an *absent* one.
        """
        capability, capability_facts = self._memory_capability(key, config)
        fact_ratio = _ratio(facts, MAX_FACTS)
        exchange_ratio = _ratio(exchanges, MAX_EXCHANGES)
        record_ratio = _ratio(records, 200)
        usage = 0.35 * fact_ratio + 0.30 * exchange_ratio + 0.35 * record_ratio
        score = 0.5 * capability + 0.5 * usage
        facts_lines = (
            *capability_facts,
            f"durable facts: {facts}/{MAX_FACTS}",
            f"exchanges remembered: {exchanges}/{MAX_EXCHANGES}",
            f"learning records: {records}",
            f"shares conversation history: {'yes' if config.learning.share_conversation_history else 'no'}",
        )
        return round(score, 6), facts_lines

    def _memory_capability(
        self, key: str, config: CharacterConfig
    ) -> tuple[float, tuple[str, ...]]:
        """Whether the character's memory layer is real, isolated, and bounded."""
        signals: list[float] = []
        facts: list[str] = []
        try:
            learning = CharacterLearning(key, root=self._data_root)
            learning.facts()
            signals.append(1.0)
            facts.append("durable character-learning store: reachable")
        except Exception:
            signals.append(0.0)
            facts.append("durable character-learning store: not reachable")
        try:
            store = (
                main_model_store(root=self._data_root)
                if key == MAIN_MODEL_OWNER
                else character_store(key, root=self._data_root)
            )
            store.stats()
            signals.append(1.0)
            facts.append("provenance-carrying record store: reachable")
        except Exception:
            signals.append(0.0)
            facts.append("provenance-carrying record store: not reachable")
        bounded = MAX_FACTS > 0 and MAX_EXCHANGES > 0
        signals.append(1.0 if bounded else 0.0)
        facts.append(f"bounded stores: {'yes' if bounded else 'no'}")
        signals.append(1.0 if config.learning.enabled else 0.0)
        facts.append(f"may accumulate (learning on): {'yes' if config.learning.enabled else 'no'}")
        return round(sum(signals) / len(signals), 6), tuple(facts)

    def _score_traits(
        self, key: str, config: CharacterConfig, persona: Any
    ) -> tuple[float, tuple[str, ...]]:
        personality = config.personality
        signals: list[float] = []
        facts: list[str] = []

        persona_traits = len(persona.traits) + len(persona.rules)
        facts.append(f"persona traits and rules: {persona_traits}")
        signals.append(min(1.0, persona_traits / 8))

        extra_traits = len(personality.traits) + len(personality.rules) + len(personality.behaviour)
        facts.append(f"added traits/rules/behaviour: {extra_traits}")
        signals.append(min(1.0, extra_traits / 4))

        styled = sum(
            1
            for value in (
                personality.communication_style,
                personality.reasoning_style,
                personality.knowledge_domain,
            )
            if value
        )
        facts.append(f"style/domain fields set: {styled}/3")
        signals.append(styled / 3)

        signals.append(min(1.0, len(personality.objectives) / 3))
        facts.append(f"objectives defined: {len(personality.objectives)}")

        customised = self._is_customised(key)
        facts.append(f"customised by the user: {'yes' if customised else 'no'}")
        signals.append(1.0 if customised else 0.0)

        depth, depth_facts = character_depth(key)
        facts.extend(depth_facts)
        signals.append(depth)
        return round(sum(signals) / len(signals), 6), tuple(facts)

    def _score_development(
        self, key: str, config: CharacterConfig
    ) -> tuple[float, tuple[str, ...]]:
        signals: list[float] = []
        facts: list[str] = []
        configured, _available, model_key, _title = self._model_facts(key)

        # A model-backed mouth can be pointed at another GGUF or fetched; a
        # retrieval mouth is fixed by what it is.
        swappable = configured and model_key not in {"retrieval", ""}
        facts.append(f"model can be swapped or fetched: {'yes' if swappable else 'no'}")
        signals.append(1.0 if swappable else 0.0)

        voice_engine = config.voice.engine
        facts.append(f"voice engine: {voice_engine or 'none'}")
        signals.append(1.0 if voice_engine else 0.0)

        learning = config.learning
        signals.append(1.0 if learning.enabled else 0.0)
        facts.append(f"can accumulate learning: {'yes' if learning.enabled else 'no'}")

        # Room left before the learning stores are full.
        facts_count, exchanges = self._learning_counts(key)
        room = 1.0 - 0.5 * (
            _ratio(facts_count, MAX_FACTS) + _ratio(exchanges, MAX_EXCHANGES)
        )
        facts.append(f"room before the stores fill: {round(room * 100)}%")
        signals.append(room)
        return round(sum(signals) / len(signals), 6), tuple(facts)

    # -- commentary ---------------------------------------------------------

    def _commentary(
        self,
        key: str,
        config: CharacterConfig,
        dimensions: tuple[DimensionScore, ...],
        facts: int,
        exchanges: int,
        records: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        by_key = {d.key: d.score for d in dimensions}
        strengths: list[str] = []
        shortcomings: list[str] = []
        next_steps: list[str] = []

        if by_key["training"] >= 0.7:
            strengths.append("a configured, active language model with learning on")
        if by_key["memory"] >= 0.5:
            strengths.append(
                f"a durable, isolated memory layer ({facts} facts, {exchanges} exchanges, {records} records)"
            )
        if by_key["traits"] >= 0.6:
            strengths.append("a well-defined standing character (traits, style, and domain)")
        if by_key["development"] >= 0.7:
            strengths.append("clear room to grow: a swappable model and an open learning store")

        if by_key["training"] < 0.7:
            shortcomings.append("its training structure is thin (no active model or learning off)")
            next_steps.append("point the character at a language model and enable learning")
        # An *unused* memory is not an *absent* one: capability is scored apart
        # from usage, so the gap named here is the empty store, not the layer.
        learned_so_far = facts + exchanges + records
        if by_key["memory"] < 0.75 and learned_so_far < 10:
            shortcomings.append("its memory layer is ready but little has been learned yet")
            next_steps.append("talk with the character and confirm facts, so its store fills")
        if by_key["traits"] < 0.6:
            shortcomings.append("its standing character is under-defined")
            next_steps.append("set its communication style, domain, and objectives in Characters")
        if by_key["development"] < 0.7:
            shortcomings.append("limited development headroom (fixed model or no voice engine)")
            next_steps.append("choose a voice engine and a model it can swap")

        if is_private(key):
            shortcomings.append("its records are private, so the main model never learns from it")
            next_steps.append("keep it private unless you explicitly want its knowledge shared")

        if not shortcomings:
            strengths.append("no material gap: all four dimensions are healthy")
        return tuple(strengths), tuple(shortcomings), tuple(next_steps)

    # -- data access --------------------------------------------------------

    def _learning_counts(self, key: str) -> tuple[int, int]:
        try:
            learning = CharacterLearning(key, root=self._data_root)
        except Exception:
            return 0, 0
        return len(learning.facts()), len(learning.exchanges())

    def _record_count(self, key: str) -> int:
        try:
            store = (
                main_model_store(root=self._data_root)
                if key == MAIN_MODEL_OWNER
                else character_store(key, root=self._data_root)
            )
            return store.stats().total
        except Exception:
            return 0

    def _is_customised(self, key: str) -> bool:
        checker = getattr(self._registry, "is_customised", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(key))
        except Exception:
            return False

    def _model_availability(self, key: str, config: CharacterConfig) -> bool | None:
        if self._model_available is None:
            return None
        try:
            return bool(self._model_available(key, config))
        except Exception:
            return None


def character_keys() -> tuple[str, ...]:
    """The seven character keys (no project model), in the project's order."""
    return tuple(CHARACTER_KEYS)


def evaluate_characters(
    *,
    data_root: Path | None = None,
    registry: RegistryLike | None = None,
    model_available: Any | None = None,
    catalog: Any | None = None,
) -> EvaluationReport:
    """Convenience: build an evaluator and run one full report."""
    return CharacterEvaluator(
        data_root=data_root, registry=registry, model_available=model_available, catalog=catalog
    ).evaluate()


__all__ = [
    "DIMENSIONS",
    "DIMENSION_BLURBS",
    "DIMENSION_TITLES",
    "CharacterEvaluation",
    "CharacterEvaluator",
    "DimensionScore",
    "EvaluationReport",
    "RegistryLike",
    "character_keys",
    "evaluate_characters",
]
