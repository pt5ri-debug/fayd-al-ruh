"""Structure mapping: align two relational structures and transfer what aligns.

This is Gentner's structure-mapping account of analogy, made deterministic and
dependency-free. An analogy is not a similarity of *names*; it is a one-to-one
correspondence of the **relations** between two structures, so that a system of
relations in a base domain can be carried over to a target domain that shares its
*shape* but not its vocabulary.

The rules that make it structure mapping rather than graph matching:

* **One-to-one.** A base node maps to at most one target node and vice versa.
* **Parallel connectivity.** A relation between two mapped nodes may only be
  matched by a target relation *of the same type* between their images. A mapping
  that leaves a node's relations unaccounted for is not allowed to claim them.
* **Systematicity.** Among consistent mappings the engine prefers the one that
  matches the larger, more deeply connected system of relations, because a
  connected system constrains its parts more than an isolated fact does.
* **No name matching.** Node labels are never compared. Two structures map when
  their relation graphs are isomorphic in shape, whatever the concepts are
  called -- that is what lets the result be a genuine analogy rather than a
  lookup.

Everything the mapping *cannot* account for is reported, not hidden:
``unmapped_base_nodes``, ``unmapped_target_nodes``, and the base relations whose
endpoints all mapped but whose target counterpart is absent. Those last ones are
the ``candidate_inferences`` -- the new statements the analogy *suggests* for the
target. They are always marked as candidates: an analogy proposes, it never
asserts.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A mapping search visits at most this many partial assignments, so a large
#: structure can never make the engine hang. The search is ordered, so the budget
#: is spent on the most promising branches first and the result stays stable.
DEFAULT_MAX_STATES = 200_000

#: How many target nodes a single base node may be tried against. Candidate
#: targets are ordered by how well their relation signature matches, so the cap
#: drops only the least plausible pairings.
DEFAULT_MAX_CANDIDATES = 8

#: The most a single matched relation can be worth (its type weight of 1 plus a
#: confidence of 1). Used for the optimistic prune bound, so the bound is never
#: tighter than reality and a branch is cut only when it truly cannot win.
_MAX_EDGE_WEIGHT = 2.0

#: Relational abstraction: the small set of *roles* a relation type plays.
#:
#: Exact type matching is not enough for analogy. The project's own ontology uses
#: several hundred near-unique relation types ("drives", "causes", "triggers"),
#: and a person does not treat those as unrelated -- "pressure drives flow" and
#: "heat causes expansion" share a shape even though the words differ. Mapping by
#: role is what makes structure mapping work on real knowledge, and it is the same
#: move Gentner's account makes: the analogy is over relational *structure*, not
#: over the vocabulary that names it.
#:
#: A relation type absent from every role is matched by its own name, so an
#: unclassified relation still maps onto the identical relation and nothing else.
RELATION_ROLES: dict[str, tuple[str, ...]] = {
    "causes": (
        "drives", "causes", "caused_by", "induces", "triggers", "generates",
        "creates", "produces", "results_from", "initiates", "releases",
        "flow_together_as", "derived_from", "built_from",
    ),
    "increases": (
        "increases", "improves", "maximizes", "adds_to", "boosts", "recovers",
        "stimulates",
    ),
    "decreases": (
        "reduces", "mitigates", "prevents", "limits", "opposed_by", "discounts",
        "overbalances", "resists", "guards", "protects", "stabilizes",
    ),
    "measures": (
        "measures", "estimates", "estimated_by", "quantifies", "evaluates",
        "evaluated_by", "assesses", "determines", "computes", "forecasts",
        "predicts", "detects", "indicates", "identifies", "characterizes",
        "images", "qualifies", "reveals", "compares_to", "sorted_by",
    ),
    "contains": (
        "contains", "includes", "has", "hosts", "holds", "held_by",
        "composed_of", "part_of", "contained_in", "carries", "carried_by",
        "stores", "accumulates_in",
    ),
    "is_a": ("is_a", "relates_to", "relates", "stands_in", "complements", "differs_from"),
    "requires": (
        "requires", "depends_on", "governed_by", "governs", "checked_by",
        "calibrated_by", "calibrates", "constrained_by", "constrains",
        "bounded_by", "bounds", "applies_to", "used_in", "uses",
    ),
    "enables": (
        "enables", "supports", "provides", "feeds", "supplies", "allows",
        "opens_under", "helps",
    ),
    "precedes": ("precedes", "followed_by", "dated_by", "transition_to", "undergoes", "undergo"),
    "transforms": (
        "transforms", "transformed_by", "processes", "treats", "treated_by",
        "purifies", "strips", "separates", "mixes_with", "migrates_within",
        "condenses", "equalizes", "balances", "balanced_with", "optimizes",
    ),
    "affects": (
        "affects", "influences", "influence", "impacts", "regulates", "governed",
        "controls", "control", "controlled_by", "manages", "prioritizes",
        "allocates", "sets", "overrides", "designs", "plans", "seeks",
    ),
    "verifies": (
        "verifies", "ensures", "validates", "confirms", "calibrates", "records",
        "documents", "describes", "describe", "described_by", "shows",
        "concludes", "informs", "maps", "links_to", "connects", "connects_to",
        "locates",
    ),
    "encloses": ("seals", "fills", "isolates", "blocks", "penetrates", "completes", "drills"),
    "governs_risk": ("threatens", "threaten", "pollutes", "guards_against"),
}

#: The inverse index, built once: relation type -> role.
_ROLE_OF: dict[str, str] = {
    relation_type: role
    for role, relation_types in RELATION_ROLES.items()
    for relation_type in relation_types
}


def role_for(relation_type: str) -> str:
    """The abstract role a relation type plays, or the type itself if unclassified.

    Returning the type unchanged for an unknown relation is deliberate: it still
    matches an identical relation, so abstraction adds power without ever making
    two genuinely different relations look alike.
    """
    return _ROLE_OF.get(relation_type, relation_type)


@dataclass(frozen=True)
class StructureEdge:
    """One typed relation between two nodes of a structure."""

    source: str
    target: str
    relation_type: str
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip() or not self.relation_type.strip():
            raise ValueError("Structure edge fields cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Structure edge confidence must be between 0 and 1.")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation_type)


@dataclass(frozen=True)
class Structure:
    """A named set of nodes joined by typed relations.

    Nodes are plain labels; the structure deliberately carries no node *meaning*,
    because structure mapping is forbidden from using one.
    """

    name: str
    nodes: tuple[str, ...]
    edges: tuple[StructureEdge, ...] = ()

    def __post_init__(self) -> None:
        known = set(self.nodes)
        if len(known) != len(self.nodes):
            raise ValueError("Structure node labels must be unique.")
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"Edge {edge.key} references an unknown node.")

    @property
    def node_set(self) -> frozenset[str]:
        return frozenset(self.nodes)

    def signature(self, node: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """A node's relation profile: outgoing types and incoming types, sorted.

        This is a *shape* fingerprint -- relation types only, never the neighbour's
        name -- so it may guide the search without smuggling in name matching.
        """
        outgoing = tuple(sorted(e.relation_type for e in self.edges if e.source == node))
        incoming = tuple(sorted(e.relation_type for e in self.edges if e.target == node))
        return (outgoing, incoming)


@dataclass(frozen=True)
class MatchedEdge:
    """A base relation and the target relation that realises the same role."""

    base: StructureEdge
    target: StructureEdge

    @property
    def exact(self) -> bool:
        """True when the two relations carry the identical type, not just a role."""
        return self.base.relation_type == self.target.relation_type


@dataclass(frozen=True)
class StructureMapping:
    """The result of aligning a base structure with a target structure."""

    base: str
    target: str
    node_map: tuple[tuple[str, str], ...]
    matched_edges: tuple[MatchedEdge, ...]
    candidate_inferences: tuple[StructureEdge, ...]
    unmapped_base_nodes: tuple[str, ...]
    unmapped_target_nodes: tuple[str, ...]
    relation_coverage: float
    node_coverage: float
    systematicity: float
    score: float
    states_visited: int = 0

    @property
    def exact_matches(self) -> tuple[MatchedEdge, ...]:
        """Matches that carry the identical relation type on both sides."""
        return tuple(match for match in self.matched_edges if match.exact)

    @property
    def abstract_matches(self) -> tuple[MatchedEdge, ...]:
        """Matches that agree on the relational *role* but not on the exact type."""
        return tuple(match for match in self.matched_edges if not match.exact)

    @property
    def exact_share(self) -> float:
        """The share of matches that were literal, so abstraction is never hidden."""
        if not self.matched_edges:
            return 0.0
        return round(len(self.exact_matches) / len(self.matched_edges), 6)

    @property
    def mapped_nodes(self) -> dict[str, str]:
        return dict(self.node_map)

    @property
    def is_empty(self) -> bool:
        return not self.matched_edges

    def transferred(self, minimum_confidence: float = 0.0) -> tuple[tuple[str, str, str, float], ...]:
        """Candidate inferences as ``(source, relation, target, confidence)`` rows.

        The confidence discounts the base relation by how much of the structure
        the analogy actually aligned (``relation_coverage``) and by how systematic
        the mapping was, so a transfer resting on one loose match cannot be as
        confident as one resting on a whole connected system.
        """
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("Minimum confidence must be between 0 and 1.")
        rows: list[tuple[str, str, str, float]] = []
        for edge in self.candidate_inferences:
            confidence = edge.confidence * self.relation_coverage * self._systematicity_factor()
            if confidence < minimum_confidence:
                continue
            rows.append((edge.source, edge.relation_type, edge.target, round(confidence, 6)))
        return tuple(rows)

    def _systematicity_factor(self) -> float:
        """A 0..1 factor that rises with how connected the matched system is."""
        if not self.matched_edges:
            return 0.0
        # A fully connected system scores 1; isolated matched facts score less.
        return round(min(1.0, 0.5 + 0.5 * self.systematicity), 6)

    def describe(self) -> str:
        """A readable, honest summary of what mapped and what did not."""
        lines = [
            f"Analogy: {self.base} ~ {self.target}",
            f"  matched relations: {len(self.matched_edges)}",
            f"  relation coverage: {self.relation_coverage:.2f}",
            f"  node coverage: {self.node_coverage:.2f}",
        ]
        for match in self.matched_edges:
            base, counterpart = match.base, match.target
            marker = "" if match.exact else "  (by role)"
            lines.append(
                f"  {base.source} --{base.relation_type}--> {base.target}"
                f"  ==  {counterpart.source} --{counterpart.relation_type}-->"
                f" {counterpart.target}{marker}"
            )
        if self.candidate_inferences:
            lines.append("  candidate inferences (not asserted):")
            for source, relation, target, confidence in self.transferred():
                lines.append(f"    {source} --{relation}--> {target}  (confidence {confidence:.2f})")
        if self.unmapped_base_nodes:
            lines.append(f"  unmapped in base: {', '.join(self.unmapped_base_nodes)}")
        if self.unmapped_target_nodes:
            lines.append(f"  unmapped in target: {', '.join(self.unmapped_target_nodes)}")
        return "\n".join(lines)


@dataclass
class _SearchState:
    assignment: dict[str, str]
    matched: list[MatchedEdge]
    score: float = 0.0


class StructureMappingEngine:
    """Align two structures by their relations, never by their node names."""

    def __init__(
        self,
        max_states: int = DEFAULT_MAX_STATES,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        if max_states < 1:
            raise ValueError("Maximum states must be at least 1.")
        if max_candidates < 1:
            raise ValueError("Maximum candidates must be at least 1.")
        self.max_states = max_states
        self.max_candidates = max_candidates
        self._states = 0
        self._target_by_role: dict[tuple[str, str, str], list[StructureEdge]] = {}

    def map(self, base: Structure, target: Structure) -> StructureMapping:
        """Return the best consistent mapping from ``base`` onto ``target``."""
        base_degree = self._degree(base)

        # Larger base nodes first: the most constrained choice is made while the
        # fewest alternatives are still open, which prunes the search hardest.
        order = sorted(base.nodes, key=lambda n: (-base_degree[n], n))
        candidates = {
            node: self._candidates(base, node, target) for node in base.nodes
        }

        best: _SearchState = _SearchState({}, [])
        self._states = 0
        self._target_by_role = self._index_target(target)
        self._search(base, target, order, candidates, 0, _SearchState({}, []), best)

        return self._finish(base, target, best)

    @staticmethod
    def _index_target(target: Structure) -> dict[tuple[str, str, str], list[StructureEdge]]:
        """Target edges keyed by ``(source, target, role)``, exact types first.

        Indexing by *role* rather than by the exact type is what lets a genuine
        analogy be found across different vocabularies; ordering exact types first
        means that when the target does carry the identical relation, that is the
        one chosen, so abstraction never overrides an exact match.
        """
        index: dict[tuple[str, str, str], list[StructureEdge]] = {}
        for edge in target.edges:
            index.setdefault((edge.source, edge.target, role_for(edge.relation_type)), []).append(edge)
        for edges in index.values():
            edges.sort(key=lambda e: (e.relation_type, -e.confidence))
        return index

    # ------------------------------------------------------------------ search

    def _search(
        self,
        base: Structure,
        target: Structure,
        order: list[str],
        candidates: dict[str, list[str]],
        index: int,
        current: _SearchState,
        best: _SearchState,
    ) -> None:
        if self._states >= self.max_states:
            return
        self._states += 1

        # A structure mapping need not be total: a base node with no plausible
        # image is simply left unmapped. So every partial assignment is a real
        # candidate result, not only the complete ones -- record it here, or a
        # base whose last node has no free image would report nothing at all.
        # On a tie, the mapping that also aligns more nodes is preferred, so the
        # correspondence is as complete as the relations allow.
        if (current.score, len(current.assignment)) > (best.score, len(best.assignment)):
            best.assignment = dict(current.assignment)
            best.matched = list(current.matched)
            best.score = current.score

        if index == len(order):
            return

        matched_keys = {match.base.key for match in current.matched}
        # Prune: even matching every still-possible base edge at its maximum
        # weight (a relation is worth at most 1 + its confidence) cannot beat the
        # best found, so this branch is abandoned. The bound is optimistic, so it
        # can only cut branches that genuinely cannot win.
        possible = sum(
            1
            for edge in base.edges
            if edge.key not in matched_keys
            and not self._is_dead(base, current.assignment, edge)
        )
        if current.score + _MAX_EDGE_WEIGHT * possible < best.score:
            return

        node = order[index]
        used = set(current.assignment.values())
        for candidate in candidates[node]:
            if candidate in used:
                continue
            newly_matched = self._parallel_edges(base, target, current.assignment, node, candidate)
            gain = sum(self._edge_weight(match.base) for match in newly_matched)
            current.assignment[node] = candidate
            current.matched.extend(newly_matched)
            current.score += gain
            self._search(base, target, order, candidates, index + 1, current, best)
            current.score -= gain
            del current.matched[len(current.matched) - len(newly_matched):]
            del current.assignment[node]

    def _parallel_edges(
        self,
        base: Structure,
        target: Structure,
        assignment: dict[str, str],
        node: str,
        candidate: str,
    ) -> list[MatchedEdge]:
        """Relations that become matched when ``node`` maps to ``candidate``.

        A relation counts only when *both* its endpoints are mapped and the target
        relates their images by the same relational *role* -- parallel
        connectivity, at the level of roles so the analogy survives a different
        vocabulary. An exact type match is preferred when the target offers both.
        """
        extended = dict(assignment)
        extended[node] = candidate
        found: list[MatchedEdge] = []
        for edge in base.edges:
            if edge.source not in extended or edge.target not in extended:
                continue
            if node not in (edge.source, edge.target):
                continue
            options = self._target_by_role.get(
                (extended[edge.source], extended[edge.target], role_for(edge.relation_type)),
                (),
            )
            if options:
                # The index sorts exact types first, so options[0] is the exact
                # match when one exists and the best abstract match otherwise.
                found.append(MatchedEdge(edge, options[0]))
        return found

    def _is_dead(self, base: Structure, assignment: dict[str, str], edge: StructureEdge) -> bool:
        """True when both endpoints are mapped but their images cannot match.

        Such a relation can never be matched by extending the assignment, so it
        must not be counted as still-possible during pruning.
        """
        if edge.source not in assignment or edge.target not in assignment:
            return False
        key = (assignment[edge.source], assignment[edge.target], role_for(edge.relation_type))
        return key not in self._target_by_role

    # --------------------------------------------------------------- utilities

    def _candidates(self, base: Structure, node: str, target: Structure) -> list[str]:
        """Target nodes to try for ``node``, best relation-shape match first.

        The ordering uses relation *roles* only, so it cannot introduce name
        matching; the cap then drops the least plausible pairings.
        """
        base_out, base_in = base.signature(node)
        base_profile = sorted(role_for(t) for t in base_out + base_in)
        scored: list[tuple[int, str]] = []
        for other in target.nodes:
            out, incoming = target.signature(other)
            profile = sorted(role_for(t) for t in out + incoming)
            overlap = self._overlap(base_profile, profile)
            scored.append((-overlap, other))
        scored.sort()
        return [name for _, name in scored[: self.max_candidates]]

    @staticmethod
    def _overlap(left: list[str], right: list[str]) -> int:
        """How many relation types the two profiles share (multiset-aware)."""
        remaining = list(right)
        shared = 0
        for item in left:
            if item in remaining:
                remaining.remove(item)
                shared += 1
        return shared

    def _edge_weight(self, edge: StructureEdge) -> float:
        """A matched relation's value: one, plus its own confidence."""
        return 1.0 + edge.confidence

    @staticmethod
    def _degree(structure: Structure) -> dict[str, int]:
        degree = dict.fromkeys(structure.nodes, 0)
        for edge in structure.edges:
            degree[edge.source] += 1
            degree[edge.target] += 1
        return degree

    def _finish(self, base: Structure, target: Structure, best: _SearchState) -> StructureMapping:
        assignment = best.assignment
        matched_base_keys = {match.base.key for match in best.matched}

        # A base relation is a candidate inference when its endpoints all mapped
        # but the target has no counterpart relation of the same type.
        target_keys = {edge.key for edge in target.edges}
        target_roles = {(edge.source, edge.target, role_for(edge.relation_type)) for edge in target.edges}
        inferences: list[StructureEdge] = []
        for edge in base.edges:
            if edge.key in matched_base_keys:
                continue
            if edge.source not in assignment or edge.target not in assignment:
                continue
            image = (assignment[edge.source], assignment[edge.target], edge.relation_type)
            role_image = (assignment[edge.source], assignment[edge.target], role_for(edge.relation_type))
            if image not in target_keys and role_image not in target_roles:
                inferences.append(edge)

        relation_coverage = len(best.matched) / len(base.edges) if base.edges else 0.0
        node_coverage = len(assignment) / len(base.nodes) if base.nodes else 0.0
        systematicity = self._systematicity(best.matched)

        return StructureMapping(
            base=base.name,
            target=target.name,
            node_map=tuple(sorted(assignment.items())),
            matched_edges=tuple(best.matched),
            candidate_inferences=tuple(inferences),
            unmapped_base_nodes=tuple(sorted(set(base.nodes) - set(assignment))),
            unmapped_target_nodes=tuple(sorted(set(target.nodes) - set(assignment.values()))),
            relation_coverage=round(relation_coverage, 6),
            node_coverage=round(node_coverage, 6),
            systematicity=round(systematicity, 6),
            score=round(best.score, 6),
            states_visited=self._states,
        )

    @staticmethod
    def _systematicity(matched: list[MatchedEdge]) -> float:
        """How connected the matched system is.

        The share of matched relations that share at least one node with another
        matched relation. An isolated matched fact constrains nothing; a match that
        touches the rest of the system is what licenses an inference, so this is
        the property that separates a real analogy from a coincidence.
        """
        if not matched:
            return 0.0
        endpoint_counts: dict[str, int] = {}
        for match in matched:
            endpoint_counts[match.base.source] = endpoint_counts.get(match.base.source, 0) + 1
            endpoint_counts[match.base.target] = endpoint_counts.get(match.base.target, 0) + 1
        connected = sum(
            1
            for match in matched
            if endpoint_counts[match.base.source] > 1 or endpoint_counts[match.base.target] > 1
        )
        return connected / len(matched)


def mapping_score(mapping: StructureMapping) -> float:
    """A single comparable score for ranking analogies.

    Systematicity dominates, because a connected relational system is the
    evidence an analogy rests on; raw edge count is the tie-breaker.
    """
    return round(mapping.systematicity * len(mapping.matched_edges) + mapping.score, 6)


__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_STATES",
    "RELATION_ROLES",
    "MatchedEdge",
    "Structure",
    "StructureEdge",
    "StructureMapping",
    "StructureMappingEngine",
    "mapping_score",
    "role_for",
]
