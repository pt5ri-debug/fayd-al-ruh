from __future__ import annotations

from core.concept import Concept
from core.semantic_relation import SemanticRelation
from core.tokenize import normalize_phrase, token_spans

#: Inflectional endings a single-token concept term may carry on a matching token.
#: Whole-word matching alone would miss the everyday inflections of a corpus
#: ("flow" must reach "flowed", "increase" must reach "increases"), which the old
#: substring test happened to cover -- but it covered them by matching *inside*
#: unrelated words too ("porosity" in "microporosity"). Matching a term only at a
#: token's *start*, followed by one of these short endings, keeps the inflection
#: without the false positive.
_INFLECTIONS = ("s", "es", "ed", "d", "ing")


def _token_matches(token: str, term_token: str) -> bool:
    """Whether a normalised ``token`` names the normalised ``term_token``.

    Exact, or the term with one short inflectional ending (for a term of at least
    four letters, so ``as`` never matches ``asset``). Deliberately not a stemmer:
    it is bounded, deterministic, and reversible to explain.
    """
    if token == term_token:
        return True
    if len(term_token) < 4 or not token.startswith(term_token):
        return False
    return token[len(term_token) :] in _INFLECTIONS


class Ontology:
    """A small concept graph, separate from instance/entity knowledge."""
    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.relations: dict[str, SemanticRelation] = {}

    def add_concept(self, concept: Concept) -> None:
        if concept.id in self.concepts:
            raise ValueError(f"Concept '{concept.id}' already exists.")
        self.concepts[concept.id] = concept

    def add_concept_once(self, label: str, aliases: frozenset[str] = frozenset()) -> Concept:
        """Add a concept, regenerating its id on the rare random-id collision.

        Concept ids carry a truncated random suffix, so two seeded concepts can in
        principle draw the same id and abort seeding. Regenerate until the id is
        free, mirroring :meth:`add_relation_once`, so seeding never fails on a
        clash. The label is not de-duplicated here -- callers check first.
        """
        concept = Concept(label, aliases)
        while concept.id in self.concepts:
            concept = Concept(label, aliases)
        self.add_concept(concept)
        return concept

    def add_relation(self, relation: SemanticRelation) -> None:
        if relation.id in self.relations:
            raise ValueError(f"Semantic relation '{relation.id}' already exists.")
        if relation.source_id not in self.concepts or relation.target_id not in self.concepts:
            raise ValueError("Semantic relations must reference known concepts.")
        self.relations[relation.id] = relation

    def replace_relation(self, relation: SemanticRelation) -> None:
        """Overwrite an existing relation in place, e.g. to adjust its confidence."""
        if relation.id not in self.relations:
            raise KeyError(f"Unknown semantic relation: {relation.id}")
        if relation.source_id not in self.concepts or relation.target_id not in self.concepts:
            raise ValueError("Semantic relations must reference known concepts.")
        self.relations[relation.id] = relation

    def add_relation_once(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 0.5,
    ) -> None:
        """Add a relation unless an equivalent one exists, retrying id collisions."""
        for relation in self.relations.values():
            if (
                relation.source_id == source_id
                and relation.target_id == target_id
                and relation.relation_type == relation_type
            ):
                return
        relation = SemanticRelation(source_id, target_id, relation_type, confidence)
        # Relation ids carry a truncated random suffix; regenerate on the rare
        # collision so seeding never fails on an id clash.
        while relation.id in self.relations:
            relation = SemanticRelation(source_id, target_id, relation_type, confidence)
        self.add_relation(relation)

    def remove_relation(self, relation_id: str) -> None:
        if relation_id not in self.relations:
            raise KeyError(f"Unknown semantic relation: {relation_id}")
        del self.relations[relation_id]

    def find_concepts(self, term: str) -> list[Concept]:
        token = term.casefold().strip()
        exact = [concept for concept in self.concepts.values() if token in concept.terms]
        if exact:
            return exact
        normalized = normalize_phrase(term)
        return [
            concept
            for concept in self.concepts.values()
            if normalized and normalized in concept.normalized_terms
        ]

    def match_concepts(self, text: str) -> list[tuple[int, int, Concept]]:
        """Known concepts in ``text`` as ``(start, end, concept)`` spans.

        The one whole-word matcher the symbolic layer uses, and it is
        Arabic-aware: a term is matched as a run of *normalised* tokens (so
        ``المسامية`` matches a stored ``مسامية``, and ``السجل`` a ``سجل بئر``
        alias) rather than through a regex ``\\b``, which does not treat Arabic
        word boundaries reliably. Where a compound term and a bare one overlap
        (``water saturation`` vs ``saturation``) the longer, leftmost span wins,
        and only one span per concept is kept, so an ambiguous sentence resolves
        to the most specific reading.
        """
        spans = token_spans(text)
        tokens = [normalized for _start, _end, normalized in spans]
        found: list[tuple[int, int, Concept]] = []
        for concept in self.concepts.values():
            for term in concept.normalized_terms:
                term_tokens = term.split()
                if not term_tokens:
                    continue
                width = len(term_tokens)
                for index in range(len(tokens) - width + 1):
                    if not all(
                        _token_matches(tokens[index + offset], term_tokens[offset]) for offset in range(width)
                    ):
                        continue
                    found.append((spans[index][0], spans[index + width - 1][1], concept))
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        kept: list[tuple[int, int, Concept]] = []
        for start, end, concept in found:
            if any(start < other_end and end > other_start for other_start, other_end, _ in kept):
                continue
            if any(kept_concept.id == concept.id for _s, _e, kept_concept in kept):
                continue
            kept.append((start, end, concept))
        kept.sort(key=lambda item: item[0])
        return kept

    def concepts_in(self, text: str) -> set[str]:
        """The ids of every known concept whose term occurs in ``text``.

        Whole-word and Arabic-normalised, but *not* de-overlapped: a term nested
        inside a longer one ("log" inside "log interpretation") still counts, so
        the learner and the ontology-guided retriever keep every concept a text
        actually names. :meth:`match_concepts` is the de-overlapped sibling the
        relation extractor needs, where an endpoint must be a single span.
        """
        tokens = [normalized for _start, _end, normalized in token_spans(text)]
        ids: set[str] = set()
        for concept in self.concepts.values():
            for term in concept.normalized_terms:
                term_tokens = term.split()
                width = len(term_tokens)
                if not width or width > len(tokens):
                    continue
                if any(
                    all(_token_matches(tokens[index + offset], term_tokens[offset]) for offset in range(width))
                    for index in range(len(tokens) - width + 1)
                ):
                    ids.add(concept.id)
                    break
        return ids

    def add_concept_alias(self, concept_id: str, alias: str) -> Concept:
        """Add one alias to a concept in place, keeping its id (and relations).

        A concept is frozen, so the alias is added by replacing it with a new
        concept that carries the same id, which leaves every relation that
        references it intact. Adding an alias is idempotent, and an empty one is
        ignored, so a seed can call it freely.
        """
        concept = self.concepts.get(concept_id)
        if concept is None:
            raise KeyError(f"Unknown concept: {concept_id}")
        value = alias.strip()
        if not value or value in concept.aliases:
            return concept
        updated = Concept(concept.label, concept.aliases | {value}, concept.id)
        self.concepts[concept_id] = updated
        return updated

    def outgoing(self, concept_id: str, relation_type: str | None = None) -> list[SemanticRelation]:
        return [relation for relation in self.relations.values() if relation.source_id == concept_id and (relation_type is None or relation.relation_type == relation_type)]
