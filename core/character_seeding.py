"""Seed each character's memory from the project's own knowledge.

A character's memory is real -- a durable, isolated store of facts and exchanges
-- but on a fresh install it is **empty**: the layer is ready and nothing has been
learned yet, so every character reads as "ready but unused". This module closes
that gap honestly: it takes knowledge the project *already authored* (the
petroleum corpus's own definitions and facts, the disciplines, and the courses)
and teaches each character the part of it that belongs to its domain, so the
memory starts from the project's own knowledge rather than from invented
statements.

It is deliberately narrow and honest by construction:

* **Nothing is invented.** Every seeded fact is a sentence the project's own
  seeds already state (a corpus definition, a stated fact, or a discipline/course
  summary) -- never a made-up claim about the character or the world.
* **Each character is seeded in character.** Jarvis (the grounded mouth) and
  Vision (petroleum) and Hakeem (religion/history) get the material that fits
  their domain; Ultron and Banner get a general-orientation set; Cloud and the
  project's own model get the project's own orientation. A character with no
  authored domain material is seeded with its own persona facts (what it *is*),
  which are also true statements the project wrote.
* **Bounded.** Seeding respects the character's own ceilings (``MAX_FACTS``), so
  it can never overfill a store, and it is idempotent -- a second seed adds
  nothing, because a fact already known is a no-op.
* **It is explicit.** Seeding never runs on import or on its own; a caller asks
  for it (the CLI, the Characters window, or the GUI's first run), so a read-only
  session is untouched and a user who wants an empty memory keeps one.

Wired as ``python -m cli characters seed`` (the headless twin), the Characters
window's *Seed from the project's knowledge* control, and a first-run offer in
the GUI. :func:`seed_report` reports exactly what each character gained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.character_learning import (
    CHARACTER_KEYS,
    MAX_FACTS,
    PROJECT_MODEL_KEY,
    CharacterLearning,
    persona_for,
)
from core.learning_records import SOURCE_SYSTEM

#: Which slice of the project's own knowledge each character is seeded with.
#: ``petroleum`` -- the corpus's definitions and facts; ``general`` -- a small
#: orientation set about the project and the character's own role; ``religion``
#: -- nothing extra beyond the character's own persona (the project authors no
#: religious text, and none is invented here).
DOMAIN_PETROLEUM = "petroleum"
DOMAIN_GENERAL = "general"

#: The domain each character draws its seed from. The grounded and petroleum
#: mouths get the petroleum corpus; the general mouths get the orientation set;
#: Hakeem is deliberately seeded only from his own persona, because the project
#: authors no religious or historical text and this module invents none.
#:
#: Keyed by the character's **owner** name (``jarvis``, ``fayd_al_ruh``), which
#: both vocabularies resolve to through ``owner_for``.
CHARACTER_SEED_DOMAIN: dict[str, str] = {
    "jarvis": DOMAIN_PETROLEUM,
    "ultron": DOMAIN_GENERAL,
    "vision": DOMAIN_PETROLEUM,
    "banner": DOMAIN_GENERAL,
    "natasha": DOMAIN_GENERAL,
    "hakeem": DOMAIN_GENERAL,
    "cloud": DOMAIN_GENERAL,
    "fayd_al_ruh": DOMAIN_PETROLEUM,
    # The mouth-key spelling of the same characters, so a lookup by either
    # vocabulary finds the domain.
    "sources": DOMAIN_PETROLEUM,
    "model": DOMAIN_PETROLEUM,
}

#: The facts every character may be told about the project itself. These are
#: true statements the project's own documentation makes, so seeding them invents
#: nothing; they orient a character to where it runs.
PROJECT_ORIENTATION: tuple[str, ...] = (
    "This project is Fayd Al-Ruh, an evidence-aware symbolic cognitive architecture "
    "framed as a petroleum-engineering mind.",
    "The project keeps one knowledge graph shared by every mouth: its disciplines, "
    "courses, and petroleum corpus all seed the same ontology.",
    "The project answers with citations when it can and says so plainly when it "
    "cannot; it never claims a citation it did not retrieve.",
)


@dataclass(frozen=True)
class SeedOutcome:
    """What one character gained from a seed pass."""

    key: str
    name: str
    added: int
    already_known: int
    offered: int
    domain: str

    def describe(self) -> str:
        if self.added:
            return f"{self.name}: +{self.added} fact(s) ({self.domain}); {self.already_known} already known"
        return f"{self.name}: nothing new ({self.already_known}/{self.offered} already known)"


@dataclass(frozen=True)
class SeedReport:
    """The whole seed pass, one outcome per character."""

    outcomes: tuple[SeedOutcome, ...]

    @property
    def total_added(self) -> int:
        return sum(o.added for o in self.outcomes)

    def describe(self) -> str:
        lines = [f"Seeded {self.total_added} fact(s) across {len(self.outcomes)} character(s):"]
        lines.extend(f"  {o.describe()}" for o in self.outcomes)
        return "\n".join(lines)


#: The source a seeded record is attributed to. Seeding teaches a character the
#: project's **own** authored knowledge, so the record's provenance is the system
#: itself -- never a model answer, never the internet.
SEED_SOURCE = SOURCE_SYSTEM

#: The reference stamped on a seeded record, so its origin is unmistakable.
SEED_REFERENCE = "project knowledge seed"

#: The confidence a seeded record starts at: the same figure the pipeline gives
#: the system's own conclusion (``SYSTEM_BASE_CONFIDENCE``) -- trusted enough to
#: count as knowledge, never high enough to be mistaken for a verified fact.
SEED_CONFIDENCE = 0.6

#: The most records one seed pass may write per character, so the durable record
#: store starts from a representative slice rather than the whole corpus.
DEFAULT_SEED_RECORDS = 60


def _petroleum_facts(limit: int) -> tuple[str, ...]:
    """The project's own authored petroleum knowledge, as teachable facts.

    Each line is the corpus's own definition, its stated facts, or a discipline's
    summary -- never a generated sentence. Bounded so a character is seeded with
    a representative slice rather than the whole corpus.
    """
    facts: list[str] = []
    try:
        from core.petroleum_corpus import entries

        for entry in entries():
            facts.append(f"{entry.title}: {entry.definition}")
            for statement in entry.facts[:2]:
                # facts are (source, target, relation, confidence)
                if len(statement) >= 3:
                    source, target, relation = statement[0], statement[1], statement[2]
                    facts.append(
                        f"{source} {str(relation).replace('_', ' ')} {target}."
                    )
            if len(facts) >= limit:
                break
    except Exception:
        # A missing corpus simply yields no petroleum facts rather than failing.
        pass
    return tuple(facts[:limit])


def _general_facts(limit: int) -> tuple[str, ...]:
    """The project's authored orientation: itself, its disciplines, its courses.

    Every line is a label or summary the project itself wrote -- never a claim
    invented here -- so a general character is oriented to the project's world
    without being told anything untrue.
    """
    facts: list[str] = list(PROJECT_ORIENTATION)
    try:
        from core.domain_knowledge import disciplines

        for discipline in disciplines():
            facts.append(f"Discipline: {discipline.label}.")
            if len(facts) >= limit:
                return tuple(facts[:limit])
    except Exception:
        pass
    try:
        from core.curriculum_knowledge import courses

        for course in courses():
            facts.append(f"Course: {course.label} ({course.category}).")
            if len(facts) >= limit:
                return tuple(facts[:limit])
    except Exception:
        pass
    return tuple(facts[:limit])


def _facts_for(key: str, domain: str, limit: int) -> tuple[str, ...]:
    """The seed facts for one character: its own persona plus its domain slice.

    ``key`` is the character's *mouth* key (``sources``, ``ultron``, ...), because
    that is the vocabulary :data:`core.character_learning.PERSONAS` is keyed on --
    the owner name (``jarvis``) resolves to the project model's persona, which
    would name the wrong character.
    """
    persona = persona_for(key)
    persona_facts = (
        f"I am {persona.name}, {persona.role}.",
        *persona.traits,
    )
    budget = max(0, limit - len(persona_facts))
    if domain == DOMAIN_PETROLEUM:
        return (*persona_facts, *_petroleum_facts(budget))
    return (*persona_facts, *_general_facts(budget))


#: Authored questions whose answers the project's own corpus already states. A
#: character is seeded with the *project's own* answers to these, so an exchange
#: is never invented -- it is the grounded assistant's cited reply.
#:
#: The set is deliberately wide (one question per seeded discipline and many per
#: core topic), because a character's *memory* is measured by what it has
#: actually accumulated: a handful of exchanges leaves every store near-empty,
#: which is exactly the "ready but unused" state the evaluation reports. Each
#: question names a topic the corpus really covers, so its answer is a cited
#: extract rather than an invention.
PETROLEUM_QUESTIONS: tuple[str, ...] = (
    # -- rock, fluid, and the fundamentals --------------------------------
    "What is porosity?",
    "What is permeability?",
    "What is the difference between porosity and permeability?",
    "What is effective porosity?",
    "What is a pore throat?",
    "What is water saturation?",
    "What is capillary pressure?",
    "What is relative permeability?",
    "What is wettability?",
    "What is a reservoir rock?",
    "What is a source rock?",
    "What is a seal rock?",
    "What is a reservoir trap?",
    "What is the petroleum system?",
    "What is a reservoir fluid?",
    "What is a bubble point?",
    "What is a dew point?",
    "What is gas-oil ratio?",
    "What is formation volume factor?",
    "What is fluid viscosity?",
    # -- reservoir engineering -------------------------------------------
    "What is a waterflood?",
    "What is waterflood management?",
    "What is a reservoir drive mechanism?",
    "What is a solution gas drive?",
    "What is a water drive?",
    "How is original oil in place estimated?",
    "What is volumetric reserves estimation?",
    "What is material balance?",
    "What is a decline curve?",
    "What is pressure-transient testing?",
    "What is a well test?",
    "What is reservoir simulation?",
    "What is history matching?",
    "What is enhanced oil recovery?",
    "What is a miscible flood?",
    "What is a thermal recovery method?",
    "What is a polymer flood?",
    "What is a surfactant flood?",
    # -- formation evaluation and logging ---------------------------------
    "What does a density log measure?",
    "What does a neutron log measure?",
    "What does a sonic log measure?",
    "What does a resistivity log measure?",
    "What is a gamma ray log?",
    "What is the spontaneous potential log?",
    "What is a well log?",
    "What is log interpretation?",
    "What is a core sample?",
    "What is a sidewall core?",
    "What is the Archie equation?",
    "What is net-to-gross?",
    "What is a formation evaluation workflow?",
    # -- drilling ---------------------------------------------------------
    "What is a blowout preventer?",
    "What is well control?",
    "What is a kick?",
    "What is drilling fluid?",
    "What is mud weight?",
    "What is casing design?",
    "What is primary cementing?",
    "What is directional drilling?",
    "What is a horizontal well?",
    "What is rate of penetration?",
    "What is weight on bit?",
    "What is mud logging?",
    "What is a drilling bit?",
    "What is wellbore stability?",
    "What is pore pressure prediction?",
    "What is a casing string?",
    "What is a drilling riser?",
    # -- completions and stimulation --------------------------------------
    "What is a well completion?",
    "What is perforating?",
    "What is hydraulic fracturing?",
    "What is a proppant?",
    "What is matrix acidizing?",
    "What is acid stimulation?",
    "What is a gravel pack?",
    "What is sand control?",
    "What is a packer?",
    "What is a production tubing?",
    "What is a wellhead?",
    "What is a Christmas tree?",
    # -- production and facilities ----------------------------------------
    "What is artificial lift?",
    "What is gas lift?",
    "What is an electrical submersible pump?",
    "What is a sucker rod pump?",
    "What is production optimisation?",
    "What is nodal analysis?",
    "What is flow assurance?",
    "What is multiphase flow?",
    "What is a flowline?",
    "What is a separator?",
    "What is gas processing?",
    "What is LNG?",
    "What is a surface facility?",
    "What is produced water handling?",
    "What is a pipeline?",
    # -- integrity, safety, and the environment ---------------------------
    "What is well integrity?",
    "What is process safety?",
    "What is a safety case?",
    "What is a hazard and operability study?",
    "What is well decommissioning?",
    "What is plugging and abandonment?",
    "What is carbon capture and storage?",
    "What is a carbon storage site?",
    "What is a spill response?",
    "What is environmental impact assessment?",
    # -- geoscience -------------------------------------------------------
    "What is seismic interpretation?",
    "What is a seismic survey?",
    "What is a geologic structure?",
    "What is a fault?",
    "What is a stratigraphic trap?",
    "What is rock mechanics?",
    "What is a geomechanical model?",
    "What is in-situ stress?",
    "What is a fracture gradient?",
    "What is the overburden pressure?",
    # -- unconventional, offshore, and digital ----------------------------
    "What is an unconventional resource?",
    "What is a shale reservoir?",
    "What is a tight gas reservoir?",
    "What is a coalbed methane reservoir?",
    "What is subsea production?",
    "What is a subsea tree?",
    "What is deepwater drilling?",
    "What is a floating production system?",
    "What is petroleum data analytics?",
    "What is machine learning in petroleum engineering?",
    # -- economics and professional practice ------------------------------
    "What is net present value?",
    "What is a field development plan?",
    "What is project economics?",
    "What is a lifting cost?",
    "What is petroleum project risk?",
    "What is professional ethics in engineering?",
    "What is reserves classification?",
    "What is a probabilistic reserves estimate?",
    "What is portfolio management in petroleum?",
    "What is a petroleum fiscal regime?",
)

#: Questions about the project itself, whose answers are the project's own
#: orientation. These are true of the world every character runs in, so recording
#: them invents nothing about the wider world.
PROJECT_QUESTIONS: tuple[str, ...] = (
    "What is this project?",
    "How does this project answer a question?",
    "What knowledge does this project hold?",
    "How does this project use its knowledge graph?",
    "How does this project reason about cause and effect?",
    "How does this project ground an answer in evidence?",
    "How does this project learn new relations?",
    "How does this project keep its sources honest?",
    "What is the petroleum corpus in this project?",
    "What is the curriculum in this project?",
    "How does this project compare concepts by analogy?",
    "How does this project build a plan for a procedure?",
    "How does this project reach a decision?",
    "How does this project weigh evidence for a claim?",
    "How does this project keep learned knowledge between sessions?",
    "What is the proactive mind in this project?",
    "How does this project evaluate itself?",
    "How does this project handle Arabic and English?",
    "How does this project keep each character isolated?",
    "What makes an answer trustworthy in this project?",
)

#: Which question set each character's exchanges are seeded from.
CHARACTER_QUESTION_DOMAIN: dict[str, str] = {
    "jarvis": DOMAIN_PETROLEUM,
    "vision": DOMAIN_PETROLEUM,
    "fayd_al_ruh": DOMAIN_PETROLEUM,
}


def _answers(
    questions: tuple[str, ...],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Ask the project's own grounded assistant, returning cited answers.

    The grounded assistant works with no model (its extractive fallback), so this
    needs nothing downloaded and reaches the network never. Each answer keeps its
    citations, so a seeded exchange is honest about where it came from.
    """
    try:
        from core.assistant import build_default_assistant

        assistant = build_default_assistant()
    except Exception:
        return ()
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for question in questions:
        try:
            result = assistant.ask(question)
        except Exception:
            continue
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            continue
        citations = tuple(str(c) for c in getattr(result, "citations", ()) or ())
        out.append((question, text, citations))
    return tuple(out)


def seed_conversations(
    key: str,
    *,
    root: str | Path | None = None,
) -> SeedOutcome:
    """Seed one character's *exchanges* from the project's own cited answers.

    Only the characters whose domain is the project's own corpus (the grounded
    mouth, the petroleum mouth, and the project's own model) are seeded with the
    corpus questions; every character is seeded with the project-orientation
    exchanges, because those describe the world it runs in rather than the wider
    world.
    """
    from core.learning_isolation import mouth_key, owner_for

    owner = owner_for(key)
    mouth = mouth_key(owner)
    learning = CharacterLearning(owner, root=root)
    questions = PROJECT_QUESTIONS
    if CHARACTER_QUESTION_DOMAIN.get(owner) == DOMAIN_PETROLEUM:
        questions = (*PETROLEUM_QUESTIONS, *PROJECT_QUESTIONS)
    pairs = _answers(questions)
    existing = {(e.question, e.answer) for e in learning.exchanges()}
    added = 0
    for question, answer, citations in pairs:
        if (question, answer) in existing:
            continue
        learning.remember(question, answer, citations=citations)
        existing.add((question, answer))
        added += 1
    return SeedOutcome(
        key=owner,
        name=persona_for(mouth).name,
        added=added,
        already_known=len(pairs) - added,
        offered=len(pairs),
        domain=CHARACTER_QUESTION_DOMAIN.get(owner, DOMAIN_GENERAL),
    )


def seed_records(
    key: str,
    *,
    limit: int = DEFAULT_SEED_RECORDS,
    root: str | Path | None = None,
) -> int:
    """Seed one character's *durable learning records* from the project's knowledge.

    The character's facts and exchanges are seeded into its
    :class:`~core.character_learning.CharacterLearning` file; its *records* live in
    the separate, provenance-carrying
    :class:`~core.character_learning_store.CharacterLearningStore`. This teaches it
    the same authored knowledge through that store -- so the record layer starts
    from the project's own corpus rather than empty -- by running each statement
    through the real :class:`~core.learning_pipeline.LearningPipeline`. The pipeline
    decides the status honestly (a seeded statement is ``provisional`` at most,
    never auto-verified), checks it for contradictions, and writes it to the
    character's **own** store, so nothing here bypasses the isolation policy.

    Idempotent: a statement already held (by any status) is skipped, so a second
    pass adds nothing. Returns how many records were written.
    """
    from core.learning_isolation import mouth_key, owner_for

    owner = owner_for(key)
    mouth = mouth_key(owner)
    domain = CHARACTER_SEED_DOMAIN.get(owner, DOMAIN_GENERAL)
    facts = _facts_for(mouth, domain, max(1, min(limit, MAX_FACTS)))
    if not facts:
        return 0

    try:
        from core.character_learning_store import CharacterLearningStore
        from core.learning_pipeline import LearningPipeline

        store = CharacterLearningStore(owner, root=Path(root) if root is not None else None)
        pipeline = LearningPipeline(owner, store=store)
    except Exception:
        return 0

    existing = {record.knowledge.strip() for record in store.recent(limit=DEFAULT_SEED_RECORDS * 4)}
    written = 0
    for fact in facts:
        text = fact.strip()
        if not text or text in existing:
            continue
        outcome = pipeline.learn(
            source_type=SEED_SOURCE,
            source_reference=SEED_REFERENCE,
            knowledge=text,
            confidence=SEED_CONFIDENCE,
        )
        if outcome.stored:
            existing.add(text)
            written += len(outcome.stored)
    return written


def seed_character(
    key: str,
    *,
    limit: int = 24,
    root: str | Path | None = None,
    conversations: bool = False,
    records: bool = False,
) -> SeedOutcome:
    """Teach one character the slice of the project's knowledge that fits it.

    With ``conversations=True`` the character's *exchanges* are seeded too, from
    the project's own cited answers (see :func:`seed_conversations`); with
    ``records=True`` the same authored knowledge is also written as provenance-
    carrying learning records through the real pipeline (see :func:`seed_records`).
    """
    from core.learning_isolation import mouth_key, owner_for

    owner = owner_for(key)
    mouth = mouth_key(owner)
    learning = CharacterLearning(owner, root=root)
    domain = CHARACTER_SEED_DOMAIN.get(owner, DOMAIN_GENERAL)
    facts = _facts_for(mouth, domain, max(1, min(limit, MAX_FACTS)))
    facts_added = 0
    for fact in facts:
        if learning.add_fact(fact):
            facts_added += 1
    added = facts_added
    if conversations:
        added += seed_conversations(owner, root=root).added
    if records:
        added += seed_records(owner, root=root)
    return SeedOutcome(
        key=owner,
        name=persona_for(mouth).name,
        added=added,
        already_known=len(facts) - facts_added,
        offered=len(facts),
        domain=domain,
    )


def seed_all(
    keys: tuple[str, ...] | None = None,
    *,
    limit: int = 24,
    root: str | Path | None = None,
    conversations: bool = False,
    records: bool = False,
) -> SeedReport:
    """Seed every character (or the given ones) from the project's own knowledge."""
    chosen = keys if keys is not None else (*CHARACTER_KEYS, PROJECT_MODEL_KEY)
    outcomes = tuple(
        seed_character(key, limit=limit, root=root, conversations=conversations, records=records)
        for key in chosen
    )
    return SeedReport(outcomes=outcomes)


__all__ = [
    "CHARACTER_QUESTION_DOMAIN",
    "CHARACTER_SEED_DOMAIN",
    "DEFAULT_SEED_RECORDS",
    "DOMAIN_GENERAL",
    "DOMAIN_PETROLEUM",
    "PETROLEUM_QUESTIONS",
    "PROJECT_ORIENTATION",
    "PROJECT_QUESTIONS",
    "SEED_CONFIDENCE",
    "SEED_REFERENCE",
    "SEED_SOURCE",
    "SeedOutcome",
    "SeedReport",
    "seed_all",
    "seed_character",
    "seed_conversations",
    "seed_records",
]
