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

#: How many characters of a term's first token the reverse index keys on.
#: :func:`_token_matches` lets a token carry one short inflectional ending after the
#: term, and the longest ending is three letters ("ing"), so a token's first four
#: characters always equal its term's first four. Keying on that prefix therefore
#: cannot lose a match, while a term shorter than four letters only matches exactly.
_INDEX_PREFIX = 4

#: How many distinct texts :meth:`Ontology.concepts_in` remembers. A retrieval asks
#: about the same document text once per query, so remembering the answer turns a
#: repeated whole-graph scan into one lookup; the bound keeps a long session from
#: growing without limit.
_TEXT_CACHE_SIZE = 512


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
        # Reverse indexes over the concepts' normalised terms. They are what make a
        # lookup proportional to the *text* being matched rather than to the size of
        # the graph: before them every concept's every term was re-normalised and
        # re-scanned on each call, which cost ~30 s per retrieval query on the real
        # 495-concept graph. Built lazily and dropped whenever a concept changes.
        self._exact_index: dict[str, list[Concept]] | None = None
        self._prefix_index: dict[str, list[tuple[str, list[Concept]]]] | None = None
        self._text_cache: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------- indexes

    def _invalidate(self) -> None:
        """Drop the derived indexes after a bulk change the incremental path cannot serve."""
        self._exact_index = None
        self._prefix_index = None
        self._text_cache.clear()

    def _ensure_index(self) -> None:
        if self._exact_index is not None:
            return
        exact: dict[str, list[Concept]] = {}
        prefix: dict[str, list[tuple[str, list[Concept]]]] = {}
        # The term's own words are grouped first, so a term that several concepts
        # share is stored once and each concept is appended to it.
        by_term: dict[str, list[Concept]] = {}
        for concept in self.concepts.values():
            for term in concept.normalized_terms:
                if term:
                    by_term.setdefault(term, []).append(concept)
        for term, concepts in by_term.items():
            exact.setdefault(term, []).extend(concepts)
            head = term.split(" ", 1)[0]
            if head:
                prefix.setdefault(head[:_INDEX_PREFIX], []).append((term, concepts))
        self._exact_index = exact
        self._prefix_index = prefix

    def _index_concept(self, concept: Concept) -> None:
        """Add one concept to the reverse indexes without rebuilding them.

        Seeding adds concepts one at a time and asks about each one, so a full
        rebuild per addition was quadratic -- it accounted for most of the seed
        time. An entry for the same concept id is *replaced* rather than appended,
        which is what makes :meth:`add_concept_alias` (a concept re-created with a
        wider term set) refresh in place instead of going stale.
        """
        if self._exact_index is None or self._prefix_index is None:
            return  # nothing built yet; the next _ensure_index will include it
        exact = self._exact_index
        prefix = self._prefix_index
        for term in concept.normalized_terms:
            if not term:
                continue
            bucket = exact.setdefault(term, [])
            for index, existing in enumerate(bucket):
                if existing.id == concept.id:
                    bucket[index] = concept
                    break
            else:
                bucket.append(concept)
            head = term.split(" ", 1)[0]
            if not head:
                continue
            entries = prefix.setdefault(head[:_INDEX_PREFIX], [])
            for stored_term, stored in entries:
                if stored_term != term:
                    continue
                for index, existing in enumerate(stored):
                    if existing.id == concept.id:
                        stored[index] = concept
                        break
                else:
                    stored.append(concept)
                break
            else:
                entries.append((term, [concept]))

    def add_concept(self, concept: Concept) -> None:
        if concept.id in self.concepts:
            raise ValueError(f"Concept '{concept.id}' already exists.")
        self.concepts[concept.id] = concept
        self._index_concept(concept)
        self._text_cache.clear()

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
        """Every concept whose label or alias names ``term``.

        An exact term match is tried first; a miss falls back to the normalised
        space (``core.tokenize``), so ``المسامية`` (with the definite article)
        reaches a stored ``مسامية`` alias. Both lookups are index hits, so this no
        longer scans the whole graph.
        """
        token = term.casefold().strip()
        self._ensure_index()
        exact = self._exact_index.get(token) if self._exact_index else None
        if exact:
            return list(exact)
        normalized = normalize_phrase(term)
        if not normalized:
            return []
        found = self._exact_index.get(normalized) if self._exact_index else None
        return list(found) if found else []

    def _match_spans(self, tokens: list[str]) -> list[tuple[int, int, list[Concept]]]:
        """Every ``(start, end, concepts)`` a text's tokens match, in one pass.

        The reverse index is walked **once per token position** (``O(tokens x
        candidates-per-token)``) rather than once per candidate term over every
        position, which is what the callers used to do: on the real graph a query
        scanned ~830 candidate terms across every position of every document, ~3 s
        of a 3.3 s answer. Asking each position what *could* start there is the
        same match set for a fraction of the work, because a term can only match a
        token that starts with its head.

        Spans are half-open token indices; a multi-word term is matched only when
        every one of its words matches at the same offset. Order is positional
        (by start, then longest first), and the callers re-sort for their own needs.
        """
        self._ensure_index()
        assert self._prefix_index is not None
        prefix_index = self._prefix_index
        found: list[tuple[int, int, list[Concept]]] = []
        total = len(tokens)
        for index in range(total):
            token = tokens[index]
            if not token:
                continue
            for term, concepts in prefix_index.get(token[:_INDEX_PREFIX], ()):
                width = term.count(" ") + 1
                if width == 1:
                    if _token_matches(token, term):
                        found.append((index, index + 1, concepts))
                    continue
                end = index + width
                if end > total:
                    continue
                term_tokens = term.split()
                if all(
                    _token_matches(tokens[index + offset], term_tokens[offset])
                    for offset in range(width)
                ):
                    found.append((index, end, concepts))
        return found

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

        Only the terms a token could actually match are considered (the reverse
        index), so the cost is proportional to the text, not to the graph.
        """
        spans = token_spans(text)
        tokens = [normalized for _start, _end, normalized in spans]
        # One pass over the token positions (see _match_spans), then the character
        # span of each match is read off the token spans.
        found: list[tuple[int, int, Concept]] = []
        for start_index, end_index, concepts in self._match_spans(tokens):
            start = spans[start_index][0]
            end = spans[end_index - 1][1]
            for concept in concepts:
                found.append((start, end, concept))
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

        The answer for a given text is remembered (bounded by
        :data:`_TEXT_CACHE_SIZE`): a retrieval asks about the same document text
        once per query, and without the cache each query re-scanned the graph for
        every document.
        """
        cached = self._text_cache.get(text)
        if cached is not None:
            return set(cached)
        tokens = [normalized for _start, _end, normalized in token_spans(text)]
        ids: set[str] = set()
        for _start, _end, concepts in self._match_spans(tokens):
            ids.update(concept.id for concept in concepts)
        if len(self._text_cache) >= _TEXT_CACHE_SIZE:
            self._text_cache.clear()
        self._text_cache[text] = frozenset(ids)
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
        self._index_concept(updated)
        self._text_cache.clear()
        return updated

    def outgoing(self, concept_id: str, relation_type: str | None = None) -> list[SemanticRelation]:
        return [relation for relation in self.relations.values() if relation.source_id == concept_id and (relation_type is None or relation.relation_type == relation_type)]
