"""CLI for reasoning by analogy over the project's own concept graph.

    python -m cli analogy disciplines            # list the seeded disciplines
    python -m cli analogy between porosity permeability
    python -m cli analogy like porosity          # what is porosity structurally like?
    python -m cli analogy discipline drilling reservoir

The engine aligns two structures by their **relations**, never by their names, so
a match is a claim about shape: "porosity is embedded the way permeability is".
What it maps, what it cannot map, and the candidate inferences it suggests are all
printed; nothing is ever asserted into the graph.

Exit codes: ``0`` success, ``2`` invalid arguments, ``4`` an unknown concept or
discipline, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.analogy import AnalogyEngine, discipline_names
from core.cognitive_controller import CognitiveController

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNKNOWN = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analogy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("disciplines", help="List the seeded disciplines.")

    between = sub.add_parser("between", help="Align two concepts by their relational neighbourhoods.")
    between.add_argument("base", help="The base concept (the one transferred from).")
    between.add_argument("target", help="The target concept (the one inferred about).")
    between.add_argument("--hops", type=int, default=1, help="Neighbourhood radius (default 1).")
    between.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    like = sub.add_parser("like", help="Rank the concepts most analogous to one concept.")
    like.add_argument("concept", help="The concept to find analogues for.")
    like.add_argument("--limit", type=int, default=5, help="How many to show (default 5).")
    like.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    discipline = sub.add_parser("discipline", help="Align two whole seeded disciplines.")
    discipline.add_argument("base", help="The base discipline key.")
    discipline.add_argument("target", help="The target discipline key.")
    discipline.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _controller() -> CognitiveController:
    return CognitiveController()


def _mapping_payload(analogy) -> dict:
    mapping = analogy.mapping
    return {
        "base": mapping.base,
        "target": mapping.target,
        "node_map": [list(pair) for pair in mapping.node_map],
        "matched_relations": [
            {
                "base": [match.base.source, match.base.relation_type, match.base.target],
                "target": [match.target.source, match.target.relation_type, match.target.target],
                "exact": match.exact,
            }
            for match in mapping.matched_edges
        ],
        "candidate_inferences": [
            {"source": source, "relation": relation, "target": target, "confidence": confidence}
            for source, relation, target, confidence in mapping.transferred()
        ],
        "relation_coverage": mapping.relation_coverage,
        "node_coverage": mapping.node_coverage,
        "systematicity": mapping.systematicity,
        "exact_share": mapping.exact_share,
        "unmapped_base_nodes": list(mapping.unmapped_base_nodes),
        "unmapped_target_nodes": list(mapping.unmapped_target_nodes),
        "score": analogy.score,
        "meaningful": analogy.is_meaningful,
    }


def run_disciplines(_args: argparse.Namespace) -> int:
    for key in discipline_names():
        print(f"  {key}")
    return EXIT_OK


def run_between(args: argparse.Namespace) -> int:
    engine = AnalogyEngine(_controller().ontology)
    try:
        analogy = engine.between_concepts(args.base, args.target, hops=args.hops)
    except (KeyError, ValueError) as error:
        print(f"Could not align the concepts: {error}")
        return EXIT_UNKNOWN
    if args.json:
        print(json.dumps(_mapping_payload(analogy), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(analogy.describe())
    if not analogy.is_meaningful:
        print("\nThis is not a meaningful analogy: the two share no relational system.")
    return EXIT_OK


def run_like(args: argparse.Namespace) -> int:
    engine = AnalogyEngine(_controller().ontology)
    try:
        ranked = engine.neighbours_of(args.concept, limit=args.limit)
    except (KeyError, ValueError) as error:
        print(f"Could not rank analogues: {error}")
        return EXIT_UNKNOWN
    if args.json:
        print(json.dumps([_mapping_payload(a) for a in ranked], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not ranked:
        print(f"No concept is structurally analogous to {args.concept!r} in the current graph.")
        return EXIT_OK
    print(f"Concepts structurally analogous to {args.concept!r} (by relations, not names):")
    for analogy in ranked:
        mapping = analogy.mapping
        print(
            f"  {mapping.base:<28} score={analogy.score:5.2f}"
            f"  matched={len(mapping.matched_edges):2}"
            f"  systematicity={mapping.systematicity:.2f}"
            f"  exact={mapping.exact_share:.2f}"
        )
    return EXIT_OK


def run_discipline(args: argparse.Namespace) -> int:
    engine = AnalogyEngine(_controller().ontology)
    try:
        analogy = engine.between_disciplines(args.base, args.target)
    except KeyError as error:
        print(f"Unknown discipline: {error}")
        return EXIT_UNKNOWN
    if args.json:
        print(json.dumps(_mapping_payload(analogy), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(analogy.describe())
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "disciplines": run_disciplines,
        "between": run_between,
        "like": run_like,
        "discipline": run_discipline,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    return handler(args)
