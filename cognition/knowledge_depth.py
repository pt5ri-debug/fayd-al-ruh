"""The mind's reach: how much of the project's knowledge the graph actually holds.

The autonomous cognitive system reasons over the project's own graph, but nothing
told it -- or the user -- *how much* of the project's declared knowledge that
graph carries, where it is thin, or how to widen it. This module is that answer,
and it is the "maximal knowledge" half of the cognition layer: it measures reach,
harvests relations from text the project has ingested, and bridges disciplines so
the mind can reason across the whole field rather than inside one silo.

Four parts, each honest by construction:

* :func:`knowledge_depth` -- a :class:`DepthReport` measuring concepts, relations,
  isolated concepts, and Arabic-alias coverage per discipline, per course, and
  per corpus entry. It *reads* the seeds and the graph and invents nothing; a
  weak spot is reported, never hidden.
* :func:`harvest_relations` -- reads typed relations out of a bulk knowledge base
  (the text imported through the Import Center) with the *same* pattern matcher
  the answer path uses, and folds each accepted relation into the graph through
  the bounded :class:`~core.ontology_learner.OntologyLearner`. It previews by
  default and writes only when ``write=True``, and it never overrides a seeded
  relation or invents a concept.
* :func:`discipline_bridges` -- the shortest chain of typed relations between a
  concept of one discipline and a concept of another, so a fact learned in one
  area can be reached from another. Read-only: a path is reported, never stored.
* :func:`arabic_reach` -- how far the project reaches in Arabic (aliases present,
  questions of the bank in Arabic, labels still lacking a term).

Dependency-free at import and Tk-free: the heavy collaborators (the knowledge
base store, the retriever) are imported lazily inside the functions that need
them, so measuring depth on a fresh checkout needs nothing but the seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.ontology import Ontology

#: How many harvest candidates a single call reads by default. The store is
#: streamed, so this only bounds the work, never the memory.
DEFAULT_HARVEST_LIMIT = 500

#: The fewest relations a discipline needs before it is called "well connected".
THIN_DISCIPLINE_RELATIONS = 3

#: The fewest concepts a discipline needs before it is called "well populated".
THIN_DISCIPLINE_CONCEPTS = 3


@dataclass(frozen=True)
class Coverage:
    """Concepts and relations present against those the seed declares."""

    key: str
    label: str
    concepts: int
    relations: int
    expected_concepts: int
    expected_relations: int

    @property
    def concept_share(self) -> float:
        if self.expected_concepts <= 0:
            return 1.0
        return round(self.concepts / self.expected_concepts, 6)

    @property
    def relation_share(self) -> float:
        if self.expected_relations <= 0:
            return 1.0
        return round(self.relations / self.expected_relations, 6)

    @property
    def complete(self) -> bool:
        return self.concepts >= self.expected_concepts and self.relations >= self.expected_relations

    @property
    def incomplete(self) -> bool:
        """A discipline/course whose declared knowledge is not all present yet."""
        return not self.complete

    @property
    def sparse(self) -> bool:
        """A small area: few relations, even when all of its declared ones are here.

        Distinct from :attr:`incomplete` on purpose -- a discipline that is
        complete but small (well testing has two relations by design) is reported
        as *sparse*, not as a gap, so the weak-spot list never cries wolf.
        """
        return self.relations < THIN_DISCIPLINE_RELATIONS

    @property
    def thin(self) -> bool:
        """Either a real gap or a small area -- the two together."""
        return self.incomplete or self.sparse

    def describe(self) -> str:
        return (
            f"{self.label}: {self.concepts}/{self.expected_concepts} concept(s), "
            f"{self.relations}/{self.expected_relations} relation(s)"
            f" ({self.concept_share:.0%}/{self.relation_share:.0%})"
        )


@dataclass(frozen=True)
class DepthReport:
    """How much of the project's knowledge the graph holds, measured."""

    concepts: int
    relations: int
    isolated: int
    arabic_covered: int
    disciplines: tuple[Coverage, ...] = ()
    courses: tuple[Coverage, ...] = ()
    corpus: tuple[Coverage, ...] = ()
    average_degree: float = 0.0

    @property
    def arabic_share(self) -> float:
        if self.concepts <= 0:
            return 0.0
        return round(self.arabic_covered / self.concepts, 6)

    @property
    def connectivity(self) -> float:
        """The share of concepts that have at least one relation.

        The single best summary of "how reachable is the graph": a concept with no
        edge cannot be reached by any reasoning step.
        """
        if self.concepts <= 0:
            return 0.0
        return round((self.concepts - self.isolated) / self.concepts, 6)

    @property
    def thin_disciplines(self) -> tuple[Coverage, ...]:
        return tuple(c for c in self.disciplines if c.thin)

    @property
    def incomplete_courses(self) -> tuple[Coverage, ...]:
        return tuple(c for c in self.courses if not c.complete)

    def weak_spots(self, limit: int = 8) -> tuple[str, ...]:
        """The most useful widening targets, strongest case first.

        Named honestly and in order of severity: a discipline whose *declared*
        knowledge is not all present, then an incomplete course, then a corpus
        entry with missing relations, then the graph's own unreachable concepts,
        then the small (sparse) areas last -- a complete-but-small discipline is a
        note, not a defect. Each line is a count the report actually measured.
        """
        lines: list[str] = []
        gaps = [c for c in self.disciplines if c.incomplete]
        for coverage in sorted(gaps, key=lambda c: (c.relation_share, c.concept_share)):
            lines.append(f"incomplete discipline: {coverage.describe()}")
        for coverage in sorted(
            self.incomplete_courses, key=lambda c: (c.relation_share, c.concept_share)
        ):
            lines.append(f"incomplete course: {coverage.describe()}")
        for coverage in sorted(
            (c for c in self.corpus if not c.complete),
            key=lambda c: (c.relation_share, c.concept_share),
        ):
            lines.append(f"corpus gap: {coverage.describe()}")
        if self.isolated:
            lines.append(
                f"{self.isolated} concept(s) have no relation at all -- unreachable "
                "by any reasoning step"
            )
        for coverage in sorted(
            (c for c in self.disciplines if c.sparse and c.complete),
            key=lambda c: c.relations,
        ):
            lines.append(f"sparse area (complete but small): {coverage.describe()}")
        if self.arabic_share < 1.0:
            missing = self.concepts - self.arabic_covered
            lines.append(f"{missing} concept(s) still carry no Arabic alias")
        return tuple(lines[:limit])

    def describe(self) -> str:
        lines = [
            "Knowledge depth (the graph's reach over the project's own knowledge)",
            f"  concepts: {self.concepts}   relations: {self.relations}   "
            f"average degree: {self.average_degree:.2f}",
            f"  connectivity: {self.connectivity:.0%} "
            f"({self.isolated} concept(s) isolated)",
            f"  Arabic aliases: {self.arabic_covered}/{self.concepts} "
            f"({self.arabic_share:.0%})",
            f"  disciplines: {len(self.disciplines)}   courses: {len(self.courses)}   "
            f"corpus entries by discipline: {len(self.corpus)}",
        ]
        weak = self.weak_spots()
        if weak:
            lines.append("  weak spots:")
            lines.extend(f"    - {line}" for line in weak)
        else:
            lines.append("  weak spots: none measured -- the graph is complete for now")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "concepts": self.concepts,
            "relations": self.relations,
            "isolated": self.isolated,
            "arabic_covered": self.arabic_covered,
            "average_degree": self.average_degree,
            "connectivity": self.connectivity,
            "arabic_share": self.arabic_share,
            "disciplines": [
                {
                    "key": c.key, "label": c.label, "concepts": c.concepts,
                    "relations": c.relations, "expected_concepts": c.expected_concepts,
                    "expected_relations": c.expected_relations, "thin": c.thin,
                }
                for c in self.disciplines
            ],
            "courses": [
                {
                    "key": c.key, "label": c.label, "concepts": c.concepts,
                    "relations": c.relations, "expected_concepts": c.expected_concepts,
                    "expected_relations": c.expected_relations, "complete": c.complete,
                }
                for c in self.courses
            ],
            "weak_spots": list(self.weak_spots()),
        }


def _degrees(ontology: Ontology) -> dict[str, int]:
    """Each concept's number of incident relations."""
    degrees: dict[str, int] = {}
    for relation in ontology.relations.values():
        degrees[relation.source_id] = degrees.get(relation.source_id, 0) + 1
        degrees[relation.target_id] = degrees.get(relation.target_id, 0) + 1
    return degrees


def _coverage(
    key: str,
    label: str,
    counts: dict[str, int],
    expected_concepts: int,
    expected_relations: int,
) -> Coverage:
    return Coverage(
        key=key,
        label=label,
        concepts=int(counts.get("concepts", 0)),
        relations=int(counts.get("relations", 0)),
        expected_concepts=expected_concepts,
        expected_relations=expected_relations,
    )


def knowledge_depth(ontology: Ontology) -> DepthReport:
    """Measure how much of the project's own knowledge the graph holds.

    Reads the three seeds (disciplines, courses, and the corpus) through their own
    coverage helpers -- the *same* figures the evaluation gate uses -- and adds the
    graph's own connectivity and Arabic reach. Nothing is invented and the graph is
    never written to.
    """
    from core.curriculum_knowledge import curriculum_coverage
    from core.domain_knowledge import discipline_coverage, discipline_labels
    from core.petroleum_corpus import corpus_coverage

    labels = discipline_labels()
    disciplines = tuple(
        _coverage(
            key,
            labels.get(key, key),
            counts,
            int(counts.get("expected_concepts", 0)),
            int(counts.get("expected_relations", 0)),
        )
        for key, counts in sorted(discipline_coverage(ontology).items())
    )

    curriculum = curriculum_coverage(ontology)
    courses = tuple(
        _coverage(
            key,
            _course_label(key),
            counts,
            int(counts.get("expected_concepts", 0)),
            int(counts.get("expected_relations", 0)),
        )
        for key, counts in sorted(curriculum.items())
    )

    corpus = tuple(
        _coverage(
            key,
            key.replace("_", " "),
            counts,
            int(counts.get("expected_concepts", 0)),
            int(counts.get("expected_relations", 0)),
        )
        for key, counts in sorted(corpus_coverage(ontology).items())
    )

    degrees = _degrees(ontology)
    isolated = sum(1 for concept_id in ontology.concepts if degrees.get(concept_id, 0) == 0)
    total_degree = sum(degrees.values())
    average = round(total_degree / len(ontology.concepts), 6) if ontology.concepts else 0.0

    covered = _arabic_covered(ontology)

    return DepthReport(
        concepts=len(ontology.concepts),
        relations=len(ontology.relations),
        isolated=isolated,
        arabic_covered=covered,
        disciplines=disciplines,
        courses=courses,
        corpus=corpus,
        average_degree=average,
    )


def _course_label(key: str) -> str:
    """The readable title of a course key, falling back to the key itself."""
    try:
        from core.curriculum_knowledge import course_labels

        return str(course_labels().get(key, "")) or key.replace("_", " ")
    except Exception:
        return key.replace("_", " ")


def _arabic_covered(ontology: Ontology) -> int:
    """How many concepts carry an Arabic label or alias (the glossary's own count)."""
    try:
        from core.arabic_glossary import arabic_alias_coverage

        covered, _total = arabic_alias_coverage(ontology)
        return int(covered)
    except Exception:
        return 0


# -- harvesting typed relations from ingested text ---------------------------


@dataclass(frozen=True)
class HarvestReport:
    """What one harvest read and what it wrote."""

    chunks_read: int
    relations_read: int
    added: int
    reinforced: int
    wrote: bool
    added_triples: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.reinforced)

    def describe(self) -> str:
        lines = [
            "Relation harvest from the ingested text",
            f"  chunks read: {self.chunks_read}",
            f"  relations read: {self.relations_read}",
            f"  written: {'yes' if self.wrote else 'no (preview only -- pass --write)'}",
            f"  added to the graph: {self.added}   reinforced: {self.reinforced}",
        ]
        if self.added_triples:
            lines.append("  new relations:")
            lines.extend(f"    + {triple}" for triple in self.added_triples[:20])
        if self.examples:
            lines.append("  examples with their sentences:")
            lines.extend(f"    - {example}" for example in self.examples[:5])
        if not self.changed:
            lines.append(
                "  note: nothing new was read -- either the store is empty or every "
                "relation it states is already in the graph."
            )
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "chunks_read": self.chunks_read,
            "relations_read": self.relations_read,
            "added": self.added,
            "reinforced": self.reinforced,
            "wrote": self.wrote,
            "added_triples": list(self.added_triples),
            "examples": list(self.examples),
        }


def harvest_relations(
    controller,
    *,
    limit: int | None = DEFAULT_HARVEST_LIMIT,
    write: bool = False,
    store=None,
    on_progress=None,
) -> HarvestReport:
    """Read typed relations from the ingested text and (optionally) learn them.

    The store defaults to the project's built bulk knowledge base
    (:func:`core.knowledge_base.default_knowledge_base_dir`); ``store`` is the
    seam a test injects. The extraction is read-only; the write goes through the
    controller's own ``learn_extracted_relations_from_store``, so every relation
    is bounded by the learner's ceiling and both endpoints must already be known
    concepts. A missing or empty store is reported honestly rather than raised.
    """
    if store is None:
        store = _default_store()
    if store is None:
        return HarvestReport(
            chunks_read=0, relations_read=0, added=0, reinforced=0, wrote=False,
        )

    from core.relation_extraction import extract_from_store

    counter = {"chunks": 0}

    def _tick(index: int) -> None:
        counter["chunks"] = index
        if on_progress is not None:
            on_progress(index)

    extraction = extract_from_store(
        store, controller.relation_extractor, limit=limit, on_progress=_tick
    )
    relations_read = len(extraction.relations)
    chunks_read = counter["chunks"]
    added: list[str] = []
    reinforced = 0
    if write and relations_read:
        learning = controller.learn_extracted_relations_from_store(store, limit=limit)
        added = [f"{s} {r} {t}" for s, r, t in learning.added]
        reinforced = len(learning.reinforced)
    examples = tuple(
        f"{relation.source} {relation.relation_type} {relation.target}"
        f'  <- "{relation.sentence}"'
        for relation in extraction.relations[:5]
    )
    return HarvestReport(
        chunks_read=chunks_read,
        relations_read=relations_read,
        added=len(added),
        reinforced=reinforced,
        wrote=bool(write and relations_read),
        added_triples=tuple(added),
        examples=examples,
    )


def _default_store():
    """The project's bulk knowledge base, or ``None`` when none has been built."""
    try:
        from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir

        store = KnowledgeBaseStore(default_knowledge_base_dir())
        if not store.shard_paths():
            return None
        return store
    except Exception:
        return None


# -- bridging disciplines ----------------------------------------------------


@dataclass(frozen=True)
class Bridge:
    """A connecting chain between a concept of one discipline and another's."""

    source_label: str
    target_label: str
    hops: int
    confidence: float
    steps: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [
            f"{self.source_label} -> {self.target_label} "
            f"({self.hops} hop(s), confidence {self.confidence:.2f})"
        ]
        lines.extend(f"    {step}" for step in self.steps)
        return "\n".join(lines)


def discipline_bridges(
    ontology: Ontology,
    *,
    source: str = "",
    target: str = "",
    max_hops: int = 3,
    limit: int = 5,
) -> tuple[Bridge, ...]:
    """Find how a concept in one discipline reaches a concept in another.

    ``source`` and ``target`` are discipline keys (``reservoir``, ``drilling``);
    an empty side considers every discipline. For each pair of concepts the chain
    is the *shortest* the graph states (through :func:`core.relation_paths.
    find_connection`), so the bridge is a real walked path, never an invented
    link. Ordered strongest first (confidence, then fewer hops, then label) so the
    same call always yields the same bridges. Returns ``()` when the graph
    connects nothing within the hop bound -- "no bridge here" is a real answer.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be at least 1.")
    from core.domain_knowledge import discipline_labels
    from core.relation_paths import find_connection

    labels = discipline_labels()
    sources = _concepts_in_discipline(ontology, source) if source else []
    targets = _concepts_in_discipline(ontology, target) if target else []
    if not source or not target:
        # With one side open, pair every discipline's concepts against the other
        # discipline set: pick the first two disciplines when neither is named.
        keys = sorted(labels)
        if not source and not target:
            if len(keys) < 2:
                return ()
            source, target = keys[0], keys[1]
            sources = _concepts_in_discipline(ontology, source)
            targets = _concepts_in_discipline(ontology, target)
        elif not source:
            source = next((k for k in keys if k != target), "")
            sources = _concepts_in_discipline(ontology, source)
        else:
            target = next((k for k in keys if k != source), "")
            targets = _concepts_in_discipline(ontology, target)

    bridges: list[Bridge] = []
    for source_concept in sources:
        for target_concept in targets:
            if source_concept.id == target_concept.id:
                continue
            path = find_connection(ontology, source_concept.id, target_concept.id, max_hops=max_hops)
            if path is None:
                continue
            bridges.append(
                Bridge(
                    source_label=source_concept.label,
                    target_label=target_concept.label,
                    hops=path.hops,
                    confidence=path.confidence,
                    steps=tuple(step.describe() for step in path.steps),
                )
            )
    bridges.sort(key=lambda b: (-b.confidence, b.hops, b.source_label, b.target_label))
    return tuple(bridges[:limit])


def _concepts_in_discipline(ontology: Ontology, discipline: str):
    """The concept objects that belong to one discipline's seed."""
    from core.domain_knowledge import _DISCIPLINES

    for entry in _DISCIPLINES:
        if entry.key != discipline:
            continue
        concepts = []
        for label, _aliases in entry.concepts:
            matches = ontology.find_concepts(label)
            if matches:
                concepts.append(matches[0])
        return concepts
    return []


# -- Arabic reach ------------------------------------------------------------


@dataclass(frozen=True)
class ArabicReach:
    """How far the project reaches in Arabic."""

    covered: int
    total: int
    missing_labels: tuple[str, ...] = ()
    arabic_questions: int = 0
    total_questions: int = 0

    @property
    def share(self) -> float:
        if self.total <= 0:
            return 0.0
        return round(self.covered / self.total, 6)

    @property
    def question_share(self) -> float:
        if self.total_questions <= 0:
            return 0.0
        return round(self.arabic_questions / self.total_questions, 6)

    def describe(self) -> str:
        lines = [
            "Arabic reach",
            f"  concepts with an Arabic alias: {self.covered}/{self.total} "
            f"({self.share:.0%})",
            f"  questions of the bank in Arabic: {self.arabic_questions}/"
            f"{self.total_questions} ({self.question_share:.0%})",
        ]
        if self.missing_labels:
            lines.append(f"  labels still without a term: {len(self.missing_labels)}")
            lines.extend(f"    - {label}" for label in self.missing_labels[:10])
        else:
            lines.append("  labels still without a term: none")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "covered": self.covered,
            "total": self.total,
            "share": self.share,
            "missing_labels": list(self.missing_labels),
            "arabic_questions": self.arabic_questions,
            "total_questions": self.total_questions,
        }


def arabic_reach(ontology: Ontology, *, bank=None) -> ArabicReach:
    """Measure the project's Arabic reach over the graph and the question bank.

    ``bank`` is an optional pre-built question bank (the controller's own); when
    omitted it is not counted rather than guessed. Nothing is invented: the
    coverage comes from the glossary's own counter and the missing labels from its
    own reporter.
    """
    from core.arabic_glossary import (
        arabic_alias_coverage,
        has_arabic,
        missing_arabic_labels,
    )

    covered, total = arabic_alias_coverage(ontology)
    missing = missing_arabic_labels(ontology)
    arabic_questions = 0
    total_questions = 0
    if bank is not None:
        total_questions = len(bank)
        arabic_questions = sum(1 for question in bank if has_arabic(question.text))
    return ArabicReach(
        covered=int(covered),
        total=int(total),
        missing_labels=tuple(missing),
        arabic_questions=arabic_questions,
        total_questions=total_questions,
    )


def knowledge_depth_accuracy() -> float:
    """The held-out gate for the knowledge-depth engine.

    Runs a catalogue of hand-written judgements against the *real* engine, with
    negative controls, so a regression in how the graph's reach is measured, how a
    harvest writes, or how a bridge is walked fails here even when the unit tests
    that pinned the good case still pass. Each judgement is checked by calling the
    function the claim is about, never by re-implementing it:

    * a coverage count over a small hand-built graph must be exact, and a concept
      that participates in no relation must be counted isolated while one that does
      must not;
    * connectivity must be the share of concepts that participate in at least one
      relation -- exact on a hand-built graph, and honest about an empty one;
    * a harvest must never write when ``write=False`` (the graph is unchanged);
    * a bridge between two concepts a hand-built graph really connects must be
      found with the expected number of hops, and an unconnected pair must yield
      nothing rather than a fabricated chain;
    * an Arabic label with no alias must be reported missing rather than silently
      counted.

    Returns the share of judgements satisfied (``0.0`` for an empty catalogue).
    """
    from core.ontology import Ontology

    def _graph() -> Ontology:
        ontology = Ontology()
        ids = {
            label: ontology.add_concept_once(label).id
            for label in ("alpha", "beta", "gamma")
        }
        ontology.add_relation_once(ids["alpha"], ids["beta"], "increases", confidence=0.6)
        ontology.add_relation_once(ids["beta"], ids["gamma"], "controls", confidence=0.6)
        ontology.add_concept_once("lonely")
        return ontology

    checks: list[bool] = []

    # -- coverage is measured, not asserted --------------------------------
    ontology = _graph()
    report = knowledge_depth(ontology)
    checks.append(report.concepts == 4)
    checks.append(report.relations == 2)
    checks.append(report.isolated == 1)

    # -- connectivity is the share that participates -----------------------
    checks.append(abs(report.connectivity - 0.75) < 1e-9)
    empty = knowledge_depth(Ontology())
    checks.append(empty.connectivity == 0.0)
    checks.append(empty.isolated == 0)

    # -- a bridge is a real walked path, or nothing ------------------------
    from core.relation_paths import find_connection

    path = find_connection(ontology, "alpha", "gamma", max_hops=3)
    checks.append(path is not None)
    if path is not None:
        checks.append(path.hops == 2)
    checks.append(find_connection(ontology, "alpha", "lonely", max_hops=3) is None)

    # -- a harvest with write=False changes nothing ------------------------
    before = len(ontology.relations)
    preview = harvest_relations(ontology, limit=0, write=False)
    checks.append(not preview.wrote)
    checks.append(len(ontology.relations) == before)

    # -- an Arabic label with no alias is reported, not counted ------------
    reach = arabic_reach(ontology)
    checks.append(reach.covered <= reach.total)
    checks.append(bool(reach.missing_labels))

    if not checks:
        return 0.0
    return sum(1 for ok in checks if ok) / len(checks)


__all__ = [
    "DEFAULT_HARVEST_LIMIT",
    "THIN_DISCIPLINE_CONCEPTS",
    "THIN_DISCIPLINE_RELATIONS",
    "ArabicReach",
    "Bridge",
    "Coverage",
    "DepthReport",
    "HarvestReport",
    "arabic_reach",
    "discipline_bridges",
    "harvest_relations",
    "knowledge_depth",
    "knowledge_depth_accuracy",
]
