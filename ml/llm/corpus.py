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


def build_corpus(*, repeats: int = 40, structured_repeats: int = 60) -> str:
    """Return the training text, repeated so the small model can fit it.

    Four blocks are concatenated: general prose about the project's world,
    petroleum-engineering prose matching the declared identity, prose generated
    from the whole seeded curriculum (every discipline and course), and JSON
    objects in the shape ``MODEL_OUTPUT_SCHEMA`` expects. The JSON block teaches
    the model the field names and English values used by schema-guided decoding.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if structured_repeats < 1:
        raise ValueError("structured_repeats must be at least 1.")
    prose = " ".join(_sentences())
    petroleum = " ".join(_petroleum_sentences())
    curriculum = " ".join(curriculum_sentences())
    structured = " ".join(structured_examples())
    return (
        (prose + " ") * repeats
        + (petroleum + " ") * repeats
        + (curriculum + " ") * repeats
        + (structured + " ") * structured_repeats
    )


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


__all__ = ["build_corpus", "curriculum_sentences", "normalise"]
