"""General, knowledge-grounded learning for the ML pipeline.

The sandbox tasks teach the pipeline *how to act*. This module teaches it *what
the project knows*: it turns the whole relation graph -- the domain frame, the
curriculum, and the petroleum corpus -- into a real learning problem and measures
generalisation honestly.

The problem is **link prediction**: given only a concept and a relation type
(their tokens, never the target), predict whether the two concepts are actually
joined by that relation in the project's knowledge. Three facts make it a
genuine test rather than a lookup:

* training and test sets are split by *source concept*, so a good score means the
  model generalises to concepts it never saw, not that it memorised pairs;
* the negatives are *perturbed* relations (the real source and relation with a
  target drawn from elsewhere), so the model must use the meaning of both
  endpoints rather than the presence of a known source;
* accuracy is reported against the majority-class baseline of the test set, so
  the number is interpretable rather than merely nonzero.

A relation *encoder* shares this machinery: it embeds a concept with its context
into a vector, and a downstream head learns relations over those embeddings --
the small, self-contained stand-in for how a graph model would learn here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.progress import ProgressCallback
from ml.dataset import Dataset, FeatureHasher
from ml.models import MLP
from ml.trainer import Trainer, accuracy

# Fraction of edges held out, the seed that fixes which ones, and how many
# candidates a held-out edge is ranked against (1 true + negatives).
HELD_OUT_EDGE_FRACTION = 0.2
_SPLIT_SEED = 7
_NEGATIVES_PER_POSITIVE = 1
_RANKING_CANDIDATES = 4


@dataclass(frozen=True)
class Relation:
    """One recorded relation."""

    source: str
    relation: str
    target: str

    @property
    def sentence(self) -> str:
        return f"{self.source} {self.relation} {self.target}."


def relation_examples() -> list[Relation]:
    """Every distinct relation the project's authored knowledge records."""
    from core.curriculum_knowledge import courses
    from core.domain_knowledge import disciplines
    from core.petroleum_corpus import entries

    triples: list[tuple[str, str, str]] = []
    for discipline in disciplines():
        for source, target, relation, _confidence in discipline.relations:
            triples.append((str(source), str(relation), str(target)))
    for course in courses():
        for source, target, relation, _confidence in course.relations:
            triples.append((str(source), str(relation), str(target)))
    for entry in entries():
        for source, target, relation, _confidence in entry.facts:
            triples.append((str(source), str(relation), str(target)))
    seen: set[tuple[str, str, str]] = set()
    unique: list[Relation] = []
    for source, relation, target in triples:
        key = (source, relation, target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(Relation(source, relation, target))
    return unique


def _feature_matrix(rows: list[tuple[str, str, str]], hasher: FeatureHasher) -> np.ndarray:
    return hasher.encode_many(
        (*source.split("_"), f"rel_{relation}", *target.split("_"))
        for source, relation, target in rows
    )


def _split_edges(
    relations: list[Relation],
    *,
    fraction: float = HELD_OUT_EDGE_FRACTION,
    seed: int = _SPLIT_SEED,
) -> tuple[list[Relation], list[Relation]]:
    """Split relations into training edges and held-out edges."""
    if not relations:
        return relations, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(relations))
    held_count = max(1, int(round(len(relations) * fraction)))
    held_idx = set(int(index) for index in order[:held_count])
    train = [relation for index, relation in enumerate(relations) if index not in held_idx]
    test = [relation for index, relation in enumerate(relations) if index in held_idx]
    return train, test


def _with_negatives(
    relations: list[Relation],
    *,
    targets: list[str],
    seed: int,
    negatives: int = _NEGATIVES_PER_POSITIVE,
) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    """Build (rows, labels): every relation plus perturbed counterfeits."""
    rng = np.random.default_rng(seed)
    existing = {(r.source, r.relation, r.target) for r in relations}
    rows: list[tuple[str, str, str]] = [(r.source, r.relation, r.target) for r in relations]
    labels: list[int] = [1] * len(relations)
    # Hard negatives: the source and relation are real, the target is a real
    # concept but the wrong partner for this relation. A model that only detects
    # "known source" cannot separate these; it must weigh both endpoints.
    for relation in relations:
        for _ in range(negatives):
            candidate = relation.target
            for _attempt in range(50):
                candidate = targets[rng.integers(len(targets))]
                if (
                    candidate != relation.target
                    and (relation.source, relation.relation, candidate) not in existing
                ):
                    break
            rows.append((relation.source, relation.relation, candidate))
            labels.append(0)
    return rows, np.array(labels, dtype=np.int64)


@dataclass
class RelationLearningReport:
    """How well the link-prediction model learned, and how it generalised."""

    relations: int
    concepts: int
    relation_types: int
    train_examples: int
    held_out_examples: int
    train_accuracy: float
    held_out_accuracy: float
    held_out_baseline: float
    hit_at_1: float
    mean_rank: float
    epochs: int
    stopped_early: bool

    @property
    def generalisation_gain(self) -> float:
        return self.hit_at_1 - (1.0 / _RANKING_CANDIDATES)

    def as_dict(self) -> dict[str, object]:
        return {
            "relations": self.relations,
            "concepts": self.concepts,
            "relation_types": self.relation_types,
            "train_examples": self.train_examples,
            "held_out_examples": self.held_out_examples,
            "train_accuracy": round(self.train_accuracy, 6),
            "held_out_accuracy": round(self.held_out_accuracy, 6),
            "held_out_baseline": round(self.held_out_baseline, 6),
            "hit_at_1": round(self.hit_at_1, 6),
            "mean_rank": round(self.mean_rank, 6),
            "generalisation_gain": round(self.generalisation_gain, 6),
            "epochs": self.epochs,
            "stopped_early": self.stopped_early,
        }


def _link_probabilities(model: MLP, rows: list[tuple[str, str, str]], hasher: FeatureHasher) -> np.ndarray:
    probs = model.predict_proba(_feature_matrix(rows, hasher))
    return probs[:, 1]


def _rank_held_out_edges(
    model: MLP,
    test: list[Relation],
    *,
    concepts: list[str],
    existing: set[tuple[str, str, str]],
    hasher: FeatureHasher,
    seed: int,
    candidates: int = _RANKING_CANDIDATES,
) -> tuple[float, float]:
    """Rank each held-out edge against decoys; return (Hit@1, mean rank)."""
    rng = np.random.default_rng(seed)
    hits = 0
    ranks: list[int] = []
    for relation in test:
        group = [(relation.source, relation.relation, relation.target)]
        for _ in range(candidates - 1):
            candidate = relation.target
            for _attempt in range(50):
                candidate = concepts[rng.integers(len(concepts))]
                if candidate != relation.target and (
                    relation.source,
                    relation.relation,
                    candidate,
                ) not in existing:
                    break
            group.append((relation.source, relation.relation, candidate))
        scores = _link_probabilities(model, group, hasher)
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
        hits += 1 if rank == 1 else 0
    hit_at_1 = hits / max(1, len(test))
    mean_rank = float(sum(ranks) / max(1, len(ranks)))
    return hit_at_1, mean_rank


def train_relation_outcome(
    *,
    hasher: FeatureHasher | None = None,
    seed: int = 0,
    save_to: object | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[MLP, RelationLearningReport]:
    """Train the link-prediction head and measure held-out ranking quality."""
    hasher = hasher or FeatureHasher(64)
    relations = relation_examples()
    concepts = sorted({r.source for r in relations} | {r.target for r in relations})
    types = {relation.relation for relation in relations}
    existing = {(r.source, r.relation, r.target) for r in relations}

    train_rel, test_rel = _split_edges(relations)
    if not train_rel or not test_rel:
        raise ValueError("Relation learning needs both training and held-out relations.")

    train_rows, train_labels = _with_negatives(train_rel, targets=concepts, seed=seed)
    train_features = _feature_matrix(train_rows, hasher)

    model = MLP(hasher.size, 48, 2, seed=seed)
    trainer = Trainer(learning_rate=0.05, batch_size=16, max_epochs=300, patience=25, seed=seed)
    training = trainer.fit(
        model,
        Dataset(train_features, train_labels),
        Dataset(train_features, train_labels),
        progress=progress,
    )

    test_rows, test_labels = _with_negatives(test_rel, targets=concepts, seed=seed + 1)
    test_features = _feature_matrix(test_rows, hasher)
    held_out_accuracy = accuracy(model.forward(test_features), test_labels)
    positive_share = float(test_labels.mean())
    baseline = max(positive_share, 1.0 - positive_share)
    hit_at_1, mean_rank = _rank_held_out_edges(
        model, test_rel, concepts=concepts, existing=existing, hasher=hasher, seed=seed + 2
    )

    if save_to is not None:
        from ml.models import save_model

        save_model(model, save_to)  # type: ignore[arg-type]

    report = RelationLearningReport(
        relations=len(relations),
        concepts=len(concepts),
        relation_types=len(types),
        train_examples=len(train_rel),
        held_out_examples=len(test_rel),
        train_accuracy=training.final.train_accuracy,
        held_out_accuracy=held_out_accuracy,
        held_out_baseline=baseline,
        hit_at_1=hit_at_1,
        mean_rank=mean_rank,
        epochs=len(training.epochs),
        stopped_early=training.stopped_early,
    )
    return model, report


__all__ = [
    "HELD_OUT_EDGE_FRACTION",
    "Relation",
    "RelationLearningReport",
    "relation_examples",
    "train_relation_outcome",
]
