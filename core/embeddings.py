"""A real, learned semantic space, trained inside the project.

The rest of the project's "semantic" signal is a *hash*: :func:`core.semantic_memory.hashed_embedding`
scatters tokens into buckets, so two words share a dimension only by accident.
That is why the retriever could match a *word* but never a *meaning* -- a synonym
that never appeared in the seeds matched nothing at all.

This module replaces the hash with a **distributional** space, learned from the
project's own text the way the field learned meaning before neural embeddings:
count which words appear near which, weight each pair by positive pointwise
mutual information (PPMI), and factor the matrix down to a few hundred dense
dimensions with a truncated SVD. Two words end up close when they are used in
similar company -- which is exactly what lets `permeability` be found by a query
about `flow through rock` even when the words do not overlap.

Everything is deterministic (the SVD's sign ambiguity is fixed, sampling is
seeded) and dependency-light: only ``numpy``, which is already a base
requirement, and it is imported lazily so importing this module costs nothing.

Honest by construction: a space is *learned from text*, so it is only as good as
the corpus it was trained on, and :func:`evaluate_semantic_space` measures that
with a real held-out test rather than asserting it.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.tokenize import normalize_token, token_sequence

#: Dimensions of the trained space. Large enough to separate the petroleum
#: vocabulary, small enough that the whole matrix is a few hundred kilobytes.
DEFAULT_DIMENSIONS = 128
#: Co-occurrence window, in tokens. Four is the usual small-corpus compromise:
#: wide enough to catch a relation, narrow enough not to blur the field.
DEFAULT_WINDOW = 4
#: A term seen fewer times than this is too rare to place reliably.
DEFAULT_MIN_COUNT = 2
#: Negative sampling for the PPMI shift; keeps very frequent pairs from
#: dominating the factorization.
_PPMI_SHIFT = 1.0
#: How many nearest terms a search reports by default.
DEFAULT_NEIGHBOURS = 10


@dataclass(frozen=True)
class SemanticSpace:
    """A trained distributional space: a vocabulary and one vector per term."""

    terms: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    window: int
    min_count: int
    documents: int
    tokens: int
    _index: dict[str, int] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.terms) != len(self.vectors):
            raise ValueError("Every term needs exactly one vector.")
        if self.dimensions < 1:
            raise ValueError("Semantic space needs at least one dimension.")
        # Built once and kept: retrieval embeds a document per candidate per
        # query, and rebuilding a vocabulary-sized dict each time would make a
        # linear scan quadratic for no reason.
        object.__setattr__(self, "_index", {term: index for index, term in enumerate(self.terms)})

    @property
    def vocabulary_size(self) -> int:
        return len(self.terms)

    def index_of(self, term: str) -> int | None:
        """The vocabulary position of a term, or ``None`` when it is unknown."""
        return self._index.get(normalize_token(term))

    def vector(self, term: str) -> tuple[float, ...] | None:
        """The vector of one term, or ``None`` when it is not in the vocabulary."""
        index = self.index_of(term)
        return None if index is None else self.vectors[index]

    def embed(self, text: str) -> tuple[float, ...]:
        """The vector of a whole text: the mean of the vectors of its known terms.

        Unknown terms are skipped rather than counted as zero, so a text with one
        rare word is still placed by the words the space does know. An empty
        result (no known term at all) is a zero vector, which matches nothing --
        an honest "I have never seen these words".
        """
        lookup = self._index
        total = [0.0] * self.dimensions
        found = 0
        for token in token_sequence(text):
            index = lookup.get(token)
            if index is None:
                continue
            vector = self.vectors[index]
            for position, value in enumerate(vector):
                total[position] += value
            found += 1
        if found == 0:
            return tuple(total)
        return _normalize_vector(total)

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity of two texts, in ``[0, 1]`` (0 when nothing is known)."""
        return _cosine(self.embed(left), self.embed(right))

    def similar_terms(self, text: str, *, limit: int = DEFAULT_NEIGHBOURS) -> list[tuple[str, float]]:
        """The terms nearest a text, most similar first."""
        if limit < 1:
            raise ValueError("Similarity limit must be at least 1.")
        query = self.embed(text)
        if not any(query):
            return []
        scored = [
            (term, _cosine(query, vector))
            for term, vector in zip(self.terms, self.vectors, strict=False)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [(term, float(score)) for term, score in scored[:limit]]

    def to_dict(self) -> dict:
        return {
            "terms": list(self.terms),
            "vectors": [list(vector) for vector in self.vectors],
            "dimensions": self.dimensions,
            "window": self.window,
            "min_count": self.min_count,
            "documents": self.documents,
            "tokens": self.tokens,
        }

    def describe(self) -> str:
        return (
            f"semantic space: {self.vocabulary_size} terms, {self.dimensions} dimensions, "
            f"trained on {self.documents} documents ({self.tokens} tokens), "
            f"window {self.window}, min count {self.min_count}"
        )


def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    # Coerce to a built-in float: numpy scalars leak through ``np.linalg.svd``,
    # and a vector full of them would serialise differently and compare unequal
    # to an identical vector loaded back from JSON.
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(float(value) / norm for value in vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    # One pass over the pair: the dot product and both norms are accumulated
    # together, so a 128-dimension pair costs one loop instead of three generator
    # passes over the same numbers (a retrieval scores every document, so this was
    # the largest pure-Python cost left on the answer path).
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for a, b in zip(left, right, strict=False):
        a = float(a)
        b = float(b)
        dot += a * b
        left_square += a * a
        right_square += b * b
    if left_square == 0.0 or right_square == 0.0:
        return 0.0
    raw = dot / math.sqrt(left_square * right_square)
    # Clamp: floating point can push a self-similarity a hair past 1.0, and the
    # callers treat the score as a probability-like weight.
    return max(0.0, min(1.0, float(raw)))


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """The dot product of two vectors the caller knows are already unit length.

    :meth:`DistributionalSpace.embed` returns a *normalised* vector, so a retriever
    that scores its own embedded query against its own embedded documents can skip
    the two norm computations the general :func:`cosine_similarity` performs. The
    result is clamped to the same ``[0, 1]`` band, so the fused score is unchanged.
    """
    total = 0.0
    for a, b in zip(left, right, strict=False):
        total += float(a) * float(b)
    return max(0.0, min(1.0, total))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """The same cosine as :func:`_cosine`, exposed for a caller that already has vectors.

    A retriever that embeds a query once and every document once can then score the
    pairs itself, instead of re-embedding the query for every document -- which is
    what made one retrieval cost 160 identical embeddings.
    """
    return _cosine(left, right)


def cooccurrence_counts(
    documents: Iterable[str], *, window: int = DEFAULT_WINDOW
) -> tuple[Counter[str], Counter[tuple[str, str]], int, int]:
    """Count how often each term, and each ordered pair, appears within ``window``.

    Returns the term counts, the ordered-pair counts, the number of documents,
    and the total token count. The window slides over the *normalised* tokens, so
    Arabic letter variants do not split one word into two.
    """
    if window < 1:
        raise ValueError("Co-occurrence window must be at least 1.")
    term_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    document_count = 0
    token_total = 0
    for document in documents:
        tokens = token_sequence(document)
        if not tokens:
            continue
        document_count += 1
        token_total += len(tokens)
        term_counts.update(tokens)
        for position, token in enumerate(tokens):
            stop = min(len(tokens), position + window + 1)
            for offset in range(position + 1, stop):
                other = tokens[offset]
                if other == token:
                    continue
                pair_counts[(token, other)] += 1
                pair_counts[(other, token)] += 1
    return term_counts, pair_counts, document_count, token_total


def _ppmi_matrix(
    term_counts: Counter[str],
    pair_counts: Counter[tuple[str, str]],
    *,
    min_count: int,
    token_total: int,
):
    """Build the dense PPMI matrix over the kept vocabulary.

    PPMI is the standard weighting for a count-based space: it lifts pairs that
    occur together *more than chance* and floors everything else at zero, so
    "the" and "of" contribute nothing even when they are everywhere.
    """
    import numpy as np

    vocabulary = sorted(
        term for term, count in term_counts.items() if count >= max(1, min_count)
    )
    if not vocabulary:
        return np.zeros((0, 0), dtype=np.float64), []
    index = {term: position for position, term in enumerate(vocabulary)}
    matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.float64)
    total = float(max(1, token_total))
    for (left, right), count in pair_counts.items():
        row = index.get(left)
        column = index.get(right)
        if row is None or column is None or row == column:
            continue
        expected = (term_counts[left] * term_counts[right]) / total
        if expected <= 0.0:
            continue
        pmi = math.log(count / expected) - math.log(_PPMI_SHIFT)
        if pmi > 0.0:
            matrix[row, column] = pmi
    return matrix, vocabulary


def _factorize(matrix, dimensions: int):
    """Truncated SVD, with the sign ambiguity fixed so the result is deterministic."""
    import numpy as np

    if matrix.shape[0] == 0:
        return np.zeros((0, dimensions), dtype=np.float64)
    limit = min(dimensions, min(matrix.shape))
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    reduced = left[:, :limit] * singular[:limit]
    # SVD is unique only up to a sign flip per dimension. Make the largest-magnitude
    # entry of each dimension positive, so two builds of the same corpus are equal.
    for column in range(reduced.shape[1]):
        pivot = int(np.argmax(np.abs(reduced[:, column])))
        if reduced[pivot, column] < 0.0:
            reduced[:, column] = -reduced[:, column]
    if reduced.shape[1] < dimensions:
        padding = np.zeros((reduced.shape[0], dimensions - reduced.shape[1]), dtype=np.float64)
        reduced = np.hstack([reduced, padding])
    return reduced


def build_semantic_space(
    documents: Iterable[str],
    *,
    dimensions: int = DEFAULT_DIMENSIONS,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
) -> SemanticSpace:
    """Train a distributional semantic space from ``documents``.

    Deterministic: the same documents and settings always produce the same
    space, so it can be committed or cached and compared between runs.
    """
    if dimensions < 1:
        raise ValueError("Semantic space needs at least one dimension.")
    materialized = [document for document in documents if document and document.strip()]
    term_counts, pair_counts, document_count, token_total = cooccurrence_counts(
        materialized, window=window
    )
    matrix, vocabulary = _ppmi_matrix(
        term_counts, pair_counts, min_count=min_count, token_total=token_total
    )
    reduced = _factorize(matrix, dimensions)
    vectors = tuple(_normalize_vector(row) for row in reduced)
    return SemanticSpace(
        terms=tuple(vocabulary),
        vectors=vectors,
        dimensions=dimensions,
        window=window,
        min_count=min_count,
        documents=document_count,
        tokens=token_total,
    )


@dataclass(frozen=True)
class SemanticReport:
    """What a held-out check found: does the space recover real relations?"""

    pairs_measured: int
    related_cosine: float
    random_cosine: float
    nearest_accuracy: float
    neighbours: int

    @property
    def separation(self) -> float:
        """How far related pairs sit above unrelated ones; the honest headline."""
        return round(self.related_cosine - self.random_cosine, 6)

    @property
    def learned(self) -> bool:
        """A space is only useful when related pairs really are closer."""
        return self.pairs_measured > 0 and self.separation > 0.0 and self.nearest_accuracy > 0.0

    def summary(self) -> str:
        return (
            f"related {self.related_cosine:.3f} vs random {self.random_cosine:.3f} "
            f"(separation {self.separation:+.3f}); partner in top-{self.neighbours} "
            f"for {self.nearest_accuracy:.1%} of {self.pairs_measured} held-out pairs"
        )


def evaluate_semantic_space(
    space: SemanticSpace,
    pairs: Sequence[tuple[str, str]],
    *,
    neighbours: int = DEFAULT_NEIGHBOURS,
    seed: int = 0,
) -> SemanticReport:
    """Measure the space against held-out related pairs.

    ``pairs`` are real relations the space was **not** given as such -- only the
    text they appear in. Two honest signals are reported: how much closer related
    terms are than randomly paired terms, and how often a term's true partner is
    among its nearest neighbours. A space that memorised nothing but still places
    related words together passes both; a space that learned nothing fails both.
    """
    if neighbours < 1:
        raise ValueError("Neighbour count must be at least 1.")
    lookup = space._index
    usable = [
        (normalize_token(left), normalize_token(right))
        for left, right in pairs
        if normalize_token(left) in lookup and normalize_token(right) in lookup
    ]
    if not usable:
        return SemanticReport(0, 0.0, 0.0, 0.0, neighbours)

    related = [
        _cosine(space.vectors[lookup[left]], space.vectors[lookup[right]])
        for left, right in usable
    ]

    # A random baseline drawn from the *same* vocabulary, so a space whose vectors
    # all sit close together cannot look good just by being dense.
    vocabulary = list(space.terms)
    rng = random.Random(seed)
    random_scores: list[float] = []
    for _ in range(max(len(usable), 32)):
        left = rng.choice(vocabulary)
        right = rng.choice(vocabulary)
        if left == right:
            continue
        random_scores.append(_cosine(space.vectors[lookup[left]], space.vectors[lookup[right]]))

    # Nearest-neighbour recall: is the partner among the term's closest terms?
    nearest_hits = 0
    for left, right in usable:
        scored = sorted(
            (
                (_cosine(space.vectors[lookup[left]], space.vectors[position]), term)
                for position, term in enumerate(space.terms)
                if term != left
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if right in {term for _score, term in scored[:neighbours]}:
            nearest_hits += 1

    related_mean = sum(related) / len(related)
    random_mean = sum(random_scores) / len(random_scores) if random_scores else 0.0
    return SemanticReport(
        pairs_measured=len(usable),
        related_cosine=round(related_mean, 6),
        random_cosine=round(random_mean, 6),
        nearest_accuracy=round(nearest_hits / len(usable), 6),
        neighbours=neighbours,
    )


def seed_semantic_documents() -> list[str]:
    """The project's own text, as the documents a space is trained on.

    One document per corpus entry and one per curriculum course, so the
    co-occurrence window never reaches across a boundary that does not exist in
    the material. Imported lazily: this is the only heavyweight dependency.
    """
    from ml.llm.corpus import corpus_units

    from core.petroleum_corpus import corpus_documents

    documents = [f"{document.title}. {document.text}" for document in corpus_documents()]
    documents.extend(corpus_units())
    return documents


def seed_relation_pairs() -> list[tuple[str, str]]:
    """The seeded relations, as held-out pairs for an honest evaluation."""
    from core.petroleum_corpus import _ENTRIES

    pairs: list[tuple[str, str]] = []
    for entry in _ENTRIES:
        for source, target, _relation, _confidence in entry.facts:
            pairs.append((source, target))
    return pairs


#: How many corpus entries the honest evaluation holds out of training.
DEFAULT_HELD_OUT = 8


def held_out_split(
    *, hold_out: int = DEFAULT_HELD_OUT
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split the corpus so an evaluation cannot cheat.

    A space trained on the *same* text its test pairs come from has memorised the
    answer, so its score would be meaningless. This returns the training
    documents and a set of held-out **relation pairs** that were removed from
    training -- the pairs are then genuinely predictive: to get them right the
    space must have learned what those words mean from the *rest* of the corpus.

    Removal is by **relation**, not only by entry. The seeds state the same
    relation in more than one entry (an overview entry and its advanced
    counterpart share facts), so dropping whole entries is not enough: the
    relation is filtered out of *every* entry, and out of the curriculum and
    repository prose too. What remains is a corpus in which no test relation is
    ever stated.
    """
    from core.petroleum_corpus import _ENTRIES

    if hold_out < 1:
        raise ValueError("Held-out count must be at least 1.")
    ordered = sorted(_ENTRIES, key=lambda entry: entry.title)
    held = ordered[: min(hold_out, max(0, len(ordered) - 1))]
    held_titles = {entry.title for entry in held}
    held_triples = {
        (source, relation, target)
        for entry in held
        for source, target, relation, _confidence in entry.facts
    }
    held_statements = {f"{source} {relation} {target}." for source, relation, target in held_triples}

    training: list[str] = []
    for entry in ordered:
        lines: list[str] = []
        if entry.title not in held_titles:
            lines.append(entry.definition)
            for label, aliases in entry.concepts:
                aliases_text = f" (also known as {', '.join(aliases)})" if aliases else ""
                lines.append(f"Concept {label}{aliases_text}.")
            for name, expression in entry.formulas:
                lines.append(f"{name} is computed as {expression}.")
            for step in entry.procedures:
                lines.append(step)
        for source, target, relation, _confidence in entry.facts:
            if (source, relation, target) in held_triples:
                continue
            lines.append(f"{source} {relation} {target}.")
        if lines:
            training.append(" ".join(lines))

    from ml.llm.corpus import curriculum_sentences, repository_sentences

    # The curriculum and the other repositories are separate material from the
    # corpus entries, so they stay in training and give the space the background
    # it needs. ``corpus_sentences()`` and ``procedural_sentences()`` are
    # deliberately *not* added: they restate the corpus entries' own facts, which
    # would put the test relations straight back into the training text.
    auxiliary = curriculum_sentences() + repository_sentences(include_runtime=False)
    for sentence in auxiliary:
        if any(statement in sentence for statement in held_statements):
            continue
        training.append(sentence)

    pairs: list[tuple[str, str]] = []
    for entry in held:
        for source, target, _relation, _confidence in entry.facts:
            pairs.append((source, target))
    return training, pairs


def evaluate_held_out(
    *,
    dimensions: int = DEFAULT_DIMENSIONS,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    hold_out: int = DEFAULT_HELD_OUT,
    neighbours: int = DEFAULT_NEIGHBOURS,
    seed: int = 0,
) -> tuple[SemanticSpace, SemanticReport]:
    """Train on the corpus minus a few entries, then score those entries' relations.

    This is the honest version of "does the space know what these words mean?":
    the evaluated relations were never in the training text, so passing it
    requires real distributional generalisation, not memorisation.
    """
    documents, pairs = held_out_split(hold_out=hold_out)
    space = build_semantic_space(
        documents, dimensions=dimensions, window=window, min_count=min_count
    )
    return space, evaluate_semantic_space(space, pairs, neighbours=neighbours, seed=seed)


def save_semantic_space(space: SemanticSpace, path: Path) -> Path:
    """Write a space to ``path`` as JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(space.to_dict(), ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def load_semantic_space(path: Path) -> SemanticSpace:
    """Read a space written by :func:`save_semantic_space`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SemanticSpace(
        terms=tuple(payload["terms"]),
        vectors=tuple(tuple(float(value) for value in vector) for vector in payload["vectors"]),
        dimensions=int(payload["dimensions"]),
        window=int(payload["window"]),
        min_count=int(payload["min_count"]),
        documents=int(payload["documents"]),
        tokens=int(payload["tokens"]),
    )


def train_and_save_semantic_space(
    path: Path | None = None,
    *,
    dimensions: int = DEFAULT_DIMENSIONS,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    progress=None,
) -> SemanticSpace:
    """Train the space on the project's corpus and cache it, returning it.

    The one entry point a CLI or a release run needs: it builds the space,
    writes it, and reports honestly what it learned (the caller can run
    :func:`evaluate_semantic_space` for the held-out numbers).
    """
    from core.paths import semantic_space_path

    if progress is not None:
        progress("Reading the project's corpus…")
    documents = seed_semantic_documents()
    if progress is not None:
        progress(f"Counting co-occurrences over {len(documents)} documents…")
    space = build_semantic_space(
        documents, dimensions=dimensions, window=window, min_count=min_count
    )
    target = Path(path) if path is not None else semantic_space_path()
    save_semantic_space(space, target)
    if progress is not None:
        progress(f"Saved {space.describe()} to {target}")
    return space


_CACHED: dict[Path, SemanticSpace] = {}


def semantic_space(path: Path | None = None, *, build: bool = True) -> SemanticSpace | None:
    """The project's semantic space, cached in-process.

    Returns ``None`` rather than raising when no space has been trained yet and
    ``build`` is false, so a caller can fall back to the hash-based signal
    instead of failing. With ``build`` true (the default) a missing space is
    trained on first use, so the learned signal is present without a manual step.
    """
    from core.paths import semantic_space_path

    target = Path(path) if path is not None else semantic_space_path()
    cached = _CACHED.get(target)
    if cached is not None:
        return cached
    if target.exists():
        try:
            space = load_semantic_space(target)
        except (OSError, ValueError, KeyError):
            space = None
        if space is not None:
            _CACHED[target] = space
            return space
    if not build:
        return None
    space = train_and_save_semantic_space(target)
    _CACHED[target] = space
    return space


def reset_semantic_cache() -> None:
    """Forget the in-process cache (a test or a retrain calls this)."""
    _CACHED.clear()


__all__ = [
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_NEIGHBOURS",
    "DEFAULT_WINDOW",
    "SemanticReport",
    "SemanticSpace",
    "build_semantic_space",
    "cooccurrence_counts",
    "cosine_similarity",
    "dot_product",
    "evaluate_semantic_space",
    "load_semantic_space",
    "reset_semantic_cache",
    "save_semantic_space",
    "seed_relation_pairs",
    "seed_semantic_documents",
    "semantic_space",
    "train_and_save_semantic_space",
]
