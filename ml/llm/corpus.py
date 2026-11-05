"""The in-project training corpus for the language model.

Keeping the corpus in the repository is what makes the model self-contained: no
download, no external dataset. It is small and domain-focused -- English
sentences about the project's own world (water, light, soil, tools,
evidence, beliefs) and about its declared domain, petroleum engineering -- so
the trained model produces coherent text about the things this program reasons
about, rather than pretending to be general.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _sentences() -> list[str]:
    en = [
        "Water makes the soil wet.",
        "Light lights the candle.",
        "Rain waters the garden.",
        "Heat boils the water.",
        "A seed needs soil and water.",
        "Knowledge is built from evidence.",
        "A belief changes when evidence appears.",
        "The model predicts the next state.",
        "An action changes the world.",
        "Planning chooses steps before acting.",
        "Reward tells the agent what worked.",
        "Observation comes before inference.",
        "A good model predicts what happens.",
        "Experience becomes a rule for later.",
        "Curiosity drives the next question.",
        "Learning corrects the error step by step.",
        "Memory keeps what happened.",
        "Thinking connects the cause to the effect.",
        "A tool extends what a hand can do.",
        "Practice makes the weak strong.",
    ]
    return en


def _petroleum_sentences() -> list[str]:
    """Domain sentences: the project declares a petroleum-engineering identity."""
    en = [
        "A reservoir contains hydrocarbons.",
        "Water fills the pores of the rock.",
        "Porosity measures the voids in rock.",
        "Permeability measures how easily fluid flows.",
        "Pressure drives oil toward the well.",
        "A well penetrates the reservoir layer.",
        "Drilling reaches the reservoir.",
        "Cement seals the well from other layers.",
        "A well log measures porosity and permeability.",
        "Production declines when pressure drops.",
        "Hydrocarbons sit in porous rock.",
        "The formation holds oil and gas.",
        "Higher pressure raises the flow rate.",
        "Lower pressure reduces production.",
        "Water appears with oil in some wells.",
        "An impermeable rock blocks fluid flow.",
        "A log reading reveals the rock type.",
        "Fractures increase formation permeability.",
        "A core sample reveals rock porosity.",
        "Heat raises the viscosity of oil.",
    ]
    return en


def curriculum_sentences() -> list[str]:
    """Training prose generated from the whole seeded degree.

    Every discipline and every course contributes a sentence for each of its
    concepts and one for each of its relations, so the language model is trained
    on the same knowledge the ontology carries rather than a hand-written
    sample. Relation sentences use a compact
    "A <relation> B" shape so the type vocabulary is visible to the model.
    """
    from core.curriculum_knowledge import courses
    from core.domain_knowledge import disciplines

    sentences: list[str] = []
    for discipline in disciplines():
        for label, aliases in discipline.concepts:
            sentences.append(f"{label} is a concept from {discipline.label}.")
            if aliases:
                sentences.append(f"{label} is also known as {aliases[0]}.")
        for source, target, relation, _confidence in discipline.relations:
            sentences.append(f"{source} {relation} {target}.")
    for course in courses():
        for label, aliases in course.concepts:
            sentences.append(f"{label} is a concept from {course.label}.")
            if aliases:
                sentences.append(f"{label} is also known as {aliases[0]}.")
        for source, target, relation, _confidence in course.relations:
            sentences.append(f"{source} {relation} {target}.")
    return sentences


def repository_sentences(*, include_runtime: bool = True) -> list[str]:
    """Training prose from the smaller repositories: learned statements and runtime docs.

    The authored seeds are covered by :func:`curriculum_sentences` and
    :func:`corpus_sentences`; this adds the rest of what the project holds at a
    manageable size -- the promoted learned statements and the documents written
    at runtime -- so the model is trained on the same knowledge the retriever
    cites, not only on the seeds.
    """
    from ml.knowledge_corpus import iter_documents, sentences_from_documents

    documents = [
        document
        for document in iter_documents(kb_max_chunks=0, include_runtime=include_runtime)
        if document.source
        not in {"domain_knowledge", "curriculum_knowledge", "petroleum_corpus", "arabic_prose"}
    ]
    return sentences_from_documents(documents)


def knowledge_base_sentences(*, kb_max_chunks: int) -> list[str]:
    """Training prose from a relevance-filtered sample of the bulk knowledge base."""
    if kb_max_chunks <= 0:
        return []
    from ml.knowledge_corpus import knowledge_base_sample, sentences_from_documents

    documents, _relevant = knowledge_base_sample(max_chunks=kb_max_chunks)
    return sentences_from_documents(documents)


def training_corpus_keys() -> tuple[str, ...]:
    """The characters whose **own** corpus may enter the training text.

    A character's Knowledge section has a ``train_on_corpus`` switch, off by
    default. Only a character that switched it on **and** is not private
    contributes: the isolation policy is the authority, so a hand-edited file that
    turned training on for the private character is refused here rather than
    obeyed. The switch is read through the character registry, so an APPLY in the
    Characters window is honoured, and the shipped default is the fallback.
    Returns the owners in the project's order.
    """
    from core.learning_isolation import CHARACTER_OWNERS, owner_for

    try:
        from core.character_registry import default_registry

        registry = default_registry()
        load = registry.load
    except (ImportError, OSError):
        from core.character_config import default_config_for

        load = default_config_for

    chosen: list[str] = []
    for owner in CHARACTER_OWNERS:
        try:
            config = load(owner)
        except (ValueError, OSError):
            continue
        if config.effective_training:
            chosen.append(owner_for(owner))
    return tuple(chosen)


def _character_corpus_documents(
    key: str, *, max_chunks: int, root: str | Path | None = None
) -> list[Any]:
    """The first ``max_chunks`` documents of one character's own corpus.

    Read through the *same* :class:`core.character_knowledge.CharacterKnowledge`
    the answer path uses, so the training text and what the character answers from
    are one corpus. Returns ``()`` for an unbuilt corpus, and never raises.
    """
    from core.character_knowledge import CharacterKnowledge
    from core.knowledge_base import KnowledgeBaseStore

    if max_chunks < 1:
        return []
    try:
        knowledge = CharacterKnowledge(key, root=root)
        if not knowledge.available:
            return []
        store = KnowledgeBaseStore(knowledge.dir)
        documents: list[Any] = []
        for chunk in store.iter_chunks():
            documents.append(chunk.to_document())
            if len(documents) >= max_chunks:
                break
        return documents
    except (OSError, ValueError):
        return []


def _permitted_corpus_keys(keys: Iterable[str] | None) -> tuple[str, ...]:
    """The requested keys minus any private owner, in order.

    The aggregate default is already only the opted-in shareable characters; this
    guard is the second half of the isolation rule, so a caller that names a
    private character's key explicitly is refused just as ``learn_from`` refuses
    her before it resolves a path. Nothing is trained on a private corpus.
    """
    from core.learning_isolation import PRIVATE_OWNERS, owner_for

    chosen = training_corpus_keys() if keys is None else tuple(keys)
    permitted: list[str] = []
    for key in chosen:
        owner = owner_for(key)
        if owner in PRIVATE_OWNERS:
            continue
        if owner not in permitted:
            permitted.append(owner)
    return tuple(permitted)


def character_knowledge_sentences(
    keys: Iterable[str] | None = None,
    *,
    max_chunks: int = 2000,
    root: str | Path | None = None,
) -> list[str]:
    """Training prose from the characters' **own** knowledge corpora.

    A character answers from its own corpus by retrieval (see
    :mod:`core.character_knowledge`); this is the *opt-in* training path, so a
    character whose Knowledge section switched training on also contributes its
    corpus to the project model's training text. The default set is
    :func:`training_corpus_keys` -- only a shareable character that opted in -- and
    a private character is refused even when her key is named explicitly, so a
    private corpus never enters the training text by any route. A character with no
    built corpus contributes nothing.
    """
    from collections import OrderedDict

    spoken: list[str] = []
    for key in _permitted_corpus_keys(keys):
        documents = _character_corpus_documents(key, max_chunks=max_chunks, root=root)
        if not documents:
            continue
        from ml.knowledge_corpus import sentences_from_documents

        spoken.extend(sentences_from_documents(documents))
    return list(OrderedDict.fromkeys(sentence for sentence in spoken if sentence.strip()))


def character_knowledge_units(
    keys: Iterable[str] | None = None,
    *,
    max_chunks: int = 2000,
    root: str | Path | None = None,
) -> list[str]:
    """The characters' own corpora as distinct documents, for an honest split.

    One document per opted-in character, so a document-level train/test split never
    puts one character's corpus on both sides of the line. A private character is
    refused exactly as in :func:`character_knowledge_sentences`, and a character
    with no built corpus contributes nothing.
    """
    from collections import OrderedDict

    units: list[str] = []
    for key in _permitted_corpus_keys(keys):
        documents = _character_corpus_documents(key, max_chunks=max_chunks, root=root)
        if not documents:
            continue
        from ml.knowledge_corpus import sentences_from_documents

        joined = " ".join(sentences_from_documents(documents))
        if joined.strip():
            units.append(joined)
    return list(OrderedDict.fromkeys(unit for unit in units if unit.strip()))


#: The local mouths, mapped to the data sub-path each one persists its
#: conversations under. Jarvis keeps its sessions directly in its own folder;
#: the other three nest a ``sessions`` folder. A caller names a mouth by its
#: short label.
MOUTH_SESSION_SOURCES: dict[str, tuple[str, ...]] = {
    "jarvis": ("jarvis",),
    "ultron": ("ultron", "sessions"),
    "vision": ("vision", "sessions"),
    "banner": ("banner", "sessions"),
}

#: Upper bound on the raw conversation text read, so a long-running install
#: cannot make corpus building unbounded.
DEFAULT_MAX_SESSION_CHARS = 2_000_000


def session_sentences(
    *,
    sources: Iterable[str] | None = None,
    include_user: bool = True,
    include_assistant: bool = True,
    max_chars: int = DEFAULT_MAX_SESSION_CHARS,
) -> list[str]:
    """Training prose from the local mouths' own saved conversations.

    Jarvis, Ultron, Vision, and Banner each persist what was asked and answered
    under the data directory; those transcripts are the only record of the
    voices the user actually talks to, so training on them lets the small
    in-project model learn those voices. A turn is read as one sentence per
    sentence of its text; ``SYSTEM`` and ``CONTEXT`` turns are skipped, because a
    context turn is a whole attached file rather than something a mouth said.

    This is **off by default** (``build_corpus(include_sessions=...)``) on
    purpose: the transcripts are personal, so they are folded into training only
    when a caller asks for them by name. Reading is bounded by ``max_chars`` and
    de-duplicated, and a mouth with no saved session simply contributes nothing.
    """
    from collections import OrderedDict

    from core.composition import clean, sentences
    from core.conversation import ConversationMemory, Role
    from core.paths import data_dir
    from storage.json_store import JsonStore

    chosen = tuple(MOUTH_SESSION_SOURCES) if sources is None else tuple(sources)
    unknown = [name for name in chosen if name not in MOUTH_SESSION_SOURCES]
    if unknown:
        raise ValueError(
            f"unknown session source(s) {unknown}; "
            f"choose from {sorted(MOUTH_SESSION_SOURCES)}."
        )
    allowed = set()
    if include_user:
        allowed.add(Role.USER)
    if include_assistant:
        allowed.add(Role.ASSISTANT)
    if not allowed:
        return []

    root = data_dir()
    spoken: list[str] = []
    budget = max_chars
    exhausted = False
    for name in chosen:
        memory = ConversationMemory(JsonStore(root.joinpath(*MOUTH_SESSION_SOURCES[name])))
        for conversation_id in memory.list_saved():
            conversation = memory.load(conversation_id)
            if conversation is None:
                continue
            for message in conversation.messages:
                if message.role not in allowed:
                    continue
                text = message.text.strip()
                if not text:
                    continue
                # Stop at the budget boundary rather than cutting a message
                # mid-word, so training never sees a torn fragment.
                if len(text) > budget:
                    exhausted = True
                    break
                budget -= len(text)
                spoken.extend(sentences(clean(text)))
            if exhausted:
                break
        if exhausted:
            break
    return list(OrderedDict.fromkeys(sentence for sentence in spoken if sentence.strip()))


def character_learning_sentences() -> list[str]:
    """Training prose from the seven characters' own learning (per-character memory).

    Each character -- Jarvis, Ultron, Vision, Banner, Natasha, Hakeem, and Cloud
    -- keeps a small, file-backed learning record (the facts it was told and the
    exchanges it took part in) under the git-ignored ``data/characters/learning``
    folder (see :mod:`core.character_learning`). This renders that record as
    training prose so the project's own **Fayd Al-Ruh model** trains on what its
    characters learned, which is the bridge the task asks for.

    It is **off by default** in :func:`build_corpus` / :func:`corpus_units`
    (``include_characters=True``) for the same reason the mouths' saved
    conversations are: a character's learning is personal. A character with an
    empty record contributes nothing.
    """
    from core.character_learning import character_learning_sentences as _sentences

    return _sentences()


def build_corpus(
    *,
    repeats: int = 40,
    structured_repeats: int = 60,
    kb_max_chunks: int = 0,
    kb_repeats: int = 1,
    include_runtime: bool = True,
    include_sessions: bool = False,
    session_sources: Iterable[str] | None = None,
    session_repeats: int = 1,
    include_characters: bool = False,
    character_repeats: int = 1,
    include_character_knowledge: bool = False,
    character_knowledge_chunks: int = 2000,
    character_knowledge_repeats: int = 1,
    exclude: set[str] | frozenset[str] | None = None,
    include: set[str] | frozenset[str] | None = None,
) -> str:
    """Return the training text, repeated so the small model can fit it.

    Blocks are concatenated: general prose about the project's world,
    petroleum-engineering prose matching the declared identity, prose generated
    from the whole seeded curriculum (every discipline and course), prose and
    formulas from the seeded petroleum corpus (the knowledge content behind the
    curriculum), numbered procedural text, prose from the other repositories
    (promoted learned statements and runtime documents), the local mouths' own
    saved conversations when ``include_sessions`` is set (Jarvis, Ultron,
    Vision, and Banner -- off by default, because a transcript is personal), the
    seven characters' own learning when ``include_characters`` is set (off by
    default for the same reason), the characters' own **knowledge corpora** when
    ``include_character_knowledge`` is set (off by default, and only for a
    character whose Knowledge section opted in and that is not private), a
    relevance-filtered sample of the bulk knowledge base when ``kb_max_chunks``
    is positive (repeated ``kb_repeats`` times, since it is large and broad),
    and JSON objects in the shape ``MODEL_OUTPUT_SCHEMA`` expects. The JSON
    block teaches the model the field names and English values used by
    schema-guided decoding.

    ``exclude`` drops sentences held out for validation or test, so a held-out
    perplexity is genuinely out-of-sample. ``include`` keeps only the sentences
    of a chosen document set (the sharded schedule's current shard), while still
    processing the validation/test held-out exclusion.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if structured_repeats < 1:
        raise ValueError("structured_repeats must be at least 1.")
    if kb_repeats < 1:
        kb_repeats = 1
    if session_repeats < 1:
        session_repeats = 1
    if character_repeats < 1:
        character_repeats = 1
    if character_knowledge_repeats < 1:
        character_knowledge_repeats = 1
    held = exclude or frozenset()
    chosen = include

    def keep(sentences: list[str]) -> str:
        return " ".join(
            s for s in sentences if s not in held and (chosen is None or s in chosen)
        )

    prose = keep(_sentences())
    petroleum = keep(_petroleum_sentences())
    curriculum = keep(curriculum_sentences())
    corpus = keep(corpus_sentences())
    procedures = keep(procedural_sentences())
    arabic = keep(arabic_sentences())
    repositories = keep(repository_sentences(include_runtime=include_runtime))
    sessions = keep(session_sentences(sources=session_sources)) if include_sessions else ""
    characters = keep(character_learning_sentences()) if include_characters else ""
    character_corpus = (
        keep(character_knowledge_sentences(max_chunks=character_knowledge_chunks))
        if include_character_knowledge
        else ""
    )
    bulk = keep(knowledge_base_sentences(kb_max_chunks=kb_max_chunks))
    structured = " ".join(structured_examples())
    return (
        (prose + " ") * (repeats if prose else 0)
        + (petroleum + " ") * (repeats if petroleum else 0)
        + (curriculum + " ") * (repeats if curriculum else 0)
        + (corpus + " ") * (repeats if corpus else 0)
        + (procedures + " ") * (repeats if procedures else 0)
        + (arabic + " ") * (repeats if arabic else 0)
        + (repositories + " ") * (repeats if repositories else 0)
        + (sessions + " ") * (session_repeats if sessions else 0)
        + (characters + " ") * (character_repeats if characters else 0)
        + (character_corpus + " ") * (character_knowledge_repeats if character_corpus else 0)
        + (bulk + " ") * (kb_repeats if bulk else 0)
        + (structured + " ") * structured_repeats
    )


def corpus_sentences() -> list[str]:
    """Training prose generated from the seeded petroleum corpus."""
    from core.petroleum_corpus import corpus_sentences as _corpus_sentences

    return _corpus_sentences()


def procedural_sentences() -> list[str]:
    """Procedures rendered as explicit, numbered steps.

    The corpus stores each procedure as a sentence, but the model learns the
    *order* better when it is stated: every step is anchored to its task and
    numbered, and each task opens with how many steps it has. This is the text
    that carries the project's recorded know-how into the language model.
    """
    from core.petroleum_corpus import entries

    sentences: list[str] = []
    for entry in entries():
        steps = [step for step in entry.procedures if step.strip()]
        if not steps:
            continue
        sentences.append(f"The procedure for {entry.title} has {len(steps)} steps.")
        for index, step in enumerate(steps, start=1):
            sentences.append(f"Step {index} of {entry.title}: {step}")
    return sentences


def arabic_sentences() -> list[str]:
    """Arabic training prose rendered from the same knowledge (Phase 50).

    The project is Arabic-first, yet before Phase 50 every block above was 100%
    English, so the in-project model could not speak Arabic at all. This renders
    the same disciplines, courses, corpus entries, and relations into Arabic
    sentences (``core.arabic_prose``), so the model is trained on the knowledge in
    the language the project is for. It is *additive*: no English block changes.
    """
    from core.arabic_prose import arabic_sentence_blocks

    return arabic_sentence_blocks()


def corpus_units(
    *,
    include_sessions: bool = False,
    include_characters: bool = False,
    include_character_knowledge: bool = False,
    character_knowledge_chunks: int = 2000,
) -> list[str]:
    """Unique documents for splitting and tokenizer training.

    Concatenating the repeated training text would let the same sentence appear
    in both training and test; these are the distinct units before repetition,
    so a document-level split stays honest. The small structured-JSON block is
    left out on purpose -- it teaches output shape and is always trained on.
    The mouths' saved conversations are included only when ``include_sessions``
    is set, the seven characters' own learning only when ``include_characters``
    is set, and their own knowledge corpora only when
    ``include_character_knowledge`` is set, matching :func:`build_corpus`.
    """
    from collections import OrderedDict

    units = (
        _sentences()
        + _petroleum_sentences()
        + curriculum_sentences()
        + corpus_sentences()
        + procedural_sentences()
        + arabic_sentences()
        + repository_sentences(include_runtime=False)
    )
    if include_sessions:
        units = units + session_sentences()
    if include_characters:
        units = units + character_learning_sentences()
    if include_character_knowledge:
        units = units + character_knowledge_units(max_chunks=character_knowledge_chunks)
    return list(OrderedDict.fromkeys(unit for unit in units if unit.strip()))


def structured_examples() -> list[str]:
    """JSON objects matching ``MODEL_OUTPUT_SCHEMA`` with English content."""
    return [
        '{"concepts": [{"label": "water", "aliases": ["liquid"]}], '
        '"evidence": [{"description": "the soil is wet", "source": "observation", '
        '"supports": true, "strength": 0.8}], '
        '"hypotheses": [{"statement": "water makes the soil wet", "confidence": 0.7}], '
        '"questions": [{"text": "does water dry with heat", "topic": "causality", "priority": 0.6}], '
        '"plans": [{"goal": "water the plant", "priority": 0.9, '
        '"steps": [{"description": "check the soil"}, {"description": "pour the water"}]}]}',
        '{"concepts": [{"label": "light", "aliases": ["illumination"]}], '
        '"evidence": [{"description": "the candle is lit", "source": "vision", '
        '"supports": true, "strength": 0.9}], '
        '"hypotheses": [{"statement": "light illuminates the candle", "confidence": 0.8}], '
        '"questions": [{"text": "what is the source of light", "topic": "causality", "priority": 0.5}], '
        '"plans": [{"goal": "light the candle", "priority": 0.8, '
        '"steps": [{"description": "get a match"}, {"description": "strike the fire"}]}]}',
        '{"concepts": [{"label": "heat", "aliases": ["warmth"]}], '
        '"evidence": [{"description": "the water boils", "source": "observation", '
        '"supports": true, "strength": 0.7}], '
        '"hypotheses": [{"statement": "heat evaporates water", "confidence": 0.6}], '
        '"questions": [{"text": "when does water boil", "topic": "physics", "priority": 0.4}], '
        '"plans": [{"goal": "boil the water", "priority": 0.7, '
        '"steps": [{"description": "put the pot on the fire"}, {"description": "wait for the bubbles"}]}]}',
        '{"concepts": [{"label": "seed", "aliases": ["kernel"]}], '
        '"evidence": [{"description": "the seed is in the soil", "source": "observation", '
        '"supports": true, "strength": 0.6}], '
        '"hypotheses": [{"statement": "the seed grows with water and light", "confidence": 0.7}], '
        '"questions": [{"text": "how much does the seed need to grow", "topic": "agriculture", "priority": 0.5}], '
        '"plans": [{"goal": "plant the seed", "priority": 0.8, '
        '"steps": [{"description": "dig a hole"}, {"description": "place the seed and water it"}]}]}',
        '{"concepts": [{"label": "evidence", "aliases": ["proof"]}], '
        '"evidence": [{"description": "the experiment repeated", "source": "log", '
        '"supports": true, "strength": 0.8}], '
        '"hypotheses": [{"statement": "repetition strengthens belief", "confidence": 0.6}], '
        '"questions": [{"text": "is one piece of evidence enough", "topic": "knowledge", "priority": 0.7}], '
        '"plans": [{"goal": "verify the hypothesis", "priority": 0.9, '
        '"steps": [{"description": "collect the evidence"}, {"description": "compare the results"}]}]}',
        '{"concepts": [{"label": "reservoir", "aliases": ["container"]}], '
        '"evidence": [{"description": "the pressure is high in the well", "source": "pressure survey", '
        '"supports": true, "strength": 0.8}], '
        '"hypotheses": [{"statement": "pressure drives oil toward the well", "confidence": 0.7}], '
        '"questions": [{"text": "what is the pressure decline rate", "topic": "reservoir", "priority": 0.8}], '
        '"plans": [{"goal": "evaluate reservoir productivity", "priority": 0.9, '
        '"steps": [{"description": "read the well log"}, {"description": "compute porosity and permeability"}]}]}',
        '{"concepts": [{"label": "permeability", "aliases": ["flow capacity"]}], '
        '"evidence": [{"description": "the sample is porous rock", "source": "core analysis", '
        '"supports": true, "strength": 0.7}], '
        '"hypotheses": [{"statement": "porosity raises permeability", "confidence": 0.6}], '
        '"questions": [{"text": "how do fractures affect flow", "topic": "formation", "priority": 0.6}], '
        '"plans": [{"goal": "improve production", "priority": 0.8, '
        '"steps": [{"description": "inspect the fractures"}, {"description": "raise the pressure"}]}]}',
    ]


def normalise(text: str) -> str:
    """Unicode-normalise text so the byte model sees consistent forms."""
    return unicodedata.normalize("NFC", text)


__all__ = [
    "DEFAULT_MAX_SESSION_CHARS",
    "MOUTH_SESSION_SOURCES",
    "arabic_sentences",
    "build_corpus",
    "character_knowledge_sentences",
    "character_knowledge_units",
    "character_learning_sentences",
    "corpus_sentences",
    "corpus_units",
    "curriculum_sentences",
    "knowledge_base_sentences",
    "normalise",
    "procedural_sentences",
    "repository_sentences",
    "session_sentences",
    "structured_examples",
    "training_corpus_keys",
]
