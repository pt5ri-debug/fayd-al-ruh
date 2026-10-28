"""CLI for reading typed relations out of text, and folding them into the graph.

    python -m cli relations read "Porosity increases permeability."
    python -m cli relations read --file notes.txt
    python -m cli relations learn "The density log measures porosity." --write
    python -m cli relations types                  # what patterns it knows
    python -m cli relations connect porosity permeability  # how two concepts link
    python -m cli relations conflicts                # contradictory claims on file
    python -m cli relations infer "what does a well penetrate?"  # what the graph entails
    python -m cli relations connections "how do casing and perforating link?"  # walked chains
    python -m cli relations why "how does waterflood affect production?"  # the full reasoning trace

The extractor matches a sentence's own verb phrase onto the ontology's relation
types, so it reports *which* concepts are related and *how* -- ``porosity
increases permeability``, not merely that the two co-occurred. It is a pattern
matcher, not a parser: every candidate carries the pattern and the sentence that
produced it, and it claims nothing it did not match.

``read`` never changes the graph. ``learn`` is the writing path and refuses to
run without ``--write``, so a preview is the default and the graph only changes
when the user asks. Every learned relation is bounded by the learner's ceiling
and can never override a seeded one.

Exit codes: ``0`` success, ``2`` invalid arguments, ``4`` nothing could be read
or the input was missing, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.cognitive_controller import CognitiveController
from core.relation_extraction import EXTRACTABLE_RELATIONS

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOTHING = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relations", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    read = sub.add_parser("read", help="Read candidate relations from text (no graph change).")
    read.add_argument("text", nargs="?", help="The text to read.")
    read.add_argument("--file", help="Read the text from a file instead.")
    read.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    learn = sub.add_parser("learn", help="Read relations and fold them into the ontology.")
    learn.add_argument("text", nargs="?", help="The text to read.")
    learn.add_argument("--file", help="Read the text from a file instead.")
    learn.add_argument("--write", action="store_true", help="Actually change the graph (required).")
    learn.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    sub.add_parser("types", help="List the relation types the extractor can produce.")

    connect = sub.add_parser("connect", help="Explain how two concepts are connected, step by step.")
    connect.add_argument("source", help="The first concept (a label, English or Arabic).")
    connect.add_argument("target", help="The second concept (a label, English or Arabic).")
    connect.add_argument("--max-hops", type=int, default=None, help="The longest chain to consider.")
    connect.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    conflicts = sub.add_parser("conflicts", help="Report contradictory relations in the graph (read-only).")
    conflicts.add_argument("--concept", help="Only report conflicts on this concept's own pairs.")
    conflicts.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    infer = sub.add_parser("infer", help="Show what the graph *entails* about a question (read-only).")
    infer.add_argument("question", nargs="?", help="The question or concept to reason about.")
    infer.add_argument("--limit", type=int, default=None, help="State at most this many derivations.")
    infer.add_argument("--min-confidence", type=float, default=None, help="Drop weaker derivations.")
    infer.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    connections = sub.add_parser(
        "connections",
        help="Show how the concepts a question names are connected (read-only).",
    )
    connections.add_argument("question", nargs="?", help="The question naming two or more concepts.")
    connections.add_argument("--limit", type=int, default=None, help="State at most this many chains.")
    connections.add_argument("--max-hops", type=int, default=None, help="The longest chain to consider.")
    connections.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    causal = sub.add_parser(
        "causal",
        help="Show the directed causal paths the graph states between a question's concepts.",
    )
    causal.add_argument("question", nargs="?", help="The question naming two or more concepts.")
    causal.add_argument("--limit", type=int, default=None, help="State at most this many paths.")
    causal.add_argument("--max-depth", type=int, default=None, help="The longest causal chain to walk.")
    causal.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    why = sub.add_parser(
        "why",
        help="Ask a question and show the full reasoning trace behind the answer.",
    )
    why.add_argument("question", nargs="?", help="The question to answer.")
    why.add_argument("--limit", type=int, default=4, help="Evidence passages to read.")
    why.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    scan = sub.add_parser("scan", help="Read relations from the ingested knowledge store.")
    scan.add_argument("--store-dir", help="The knowledge store to read (default: the data folder).")
    scan.add_argument("--limit", type=int, default=None, help="Read at most this many chunks.")
    scan.add_argument("--write", action="store_true", help="Fold the relations into the graph.")
    scan.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _text_from(args: argparse.Namespace) -> str | None:
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8")
        except OSError as error:
            print(f"Could not read {args.file!r}: {error}")
            return None
    if args.text:
        return args.text
    print("Nothing to read: pass some text or --file <path>.")
    return None


def _payload(extraction) -> dict:
    return {
        "relations": [
            {
                "source": relation.source,
                "relation_type": relation.relation_type,
                "target": relation.target,
                "confidence": relation.confidence,
                "pattern": relation.pattern,
                "sentence": relation.sentence,
            }
            for relation in extraction.relations
        ],
        "sentences": extraction.sentences,
        "extracted": extraction.extracted,
        "unmatched": extraction.unmatched,
        "single_concept": extraction.single_concept,
    }


def run_read(args: argparse.Namespace) -> int:
    text = _text_from(args)
    if text is None:
        return EXIT_NOTHING
    extraction = CognitiveController().extract_relations(text)
    if args.json:
        print(json.dumps(_payload(extraction), ensure_ascii=False, indent=2))
        return EXIT_OK
    if not extraction.relations:
        print(f"No relations matched. {extraction.summary()}")
        return EXIT_OK
    for relation in extraction.relations:
        print(f"  {relation.source} --{relation.relation_type}--> {relation.target}"
              f"  (confidence {relation.confidence:.2f})")
        print(f"      pattern: {relation.pattern}")
    print(f"\n{extraction.summary()}")
    return EXIT_OK


def run_learn(args: argparse.Namespace) -> int:
    text = _text_from(args)
    if text is None:
        return EXIT_NOTHING
    controller = CognitiveController()
    extraction = controller.extract_relations(text)
    if not args.write:
        # The default is a preview: show what *would* be learned, change nothing.
        print("Preview only (pass --write to fold these into the graph):")
        for relation in extraction.relations:
            print(f"  {relation.source} --{relation.relation_type}--> {relation.target}")
        print(f"\n{extraction.summary()}")
        return EXIT_OK
    learning = controller.learn_extracted_relations(text)
    if args.json:
        print(json.dumps(
            {"added": [list(item) for item in learning.added],
             "reinforced": [list(item) for item in learning.reinforced],
             "skipped": learning.skipped},
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    if not learning.changed:
        print("Nothing new was learned (no match, an unknown concept, or a seeded relation).")
        return EXIT_OK
    for source, relation_type, target in learning.added:
        print(f"  added: {source} --{relation_type}--> {target}")
    for source, target, confidence in learning.reinforced:
        print(f"  reinforced: {source} -> {target} (confidence {confidence:.2f})")
    return EXIT_OK


def run_types(_args: argparse.Namespace) -> int:
    for relation in EXTRACTABLE_RELATIONS:
        print(f"  {relation}")
    return EXIT_OK


def run_connect(args: argparse.Namespace) -> int:
    """Explain how two concepts are connected, one typed link at a time."""
    from core.relation_paths import DEFAULT_MAX_HOPS, describe_connection, find_connection

    max_hops = args.max_hops if args.max_hops is not None else DEFAULT_MAX_HOPS
    if max_hops < 1:
        print("--max-hops must be at least 1.")
        return EXIT_USAGE
    ontology = CognitiveController().ontology
    path = find_connection(ontology, args.source, args.target, max_hops=max_hops)
    if args.json:
        payload = {
            "source": args.source,
            "target": args.target,
            "connected": path is not None,
            "hops": path.hops if path is not None else None,
            "confidence": path.confidence if path is not None else None,
            "steps": (
                [
                    {
                        "source": step.source,
                        "relation_type": step.relation_type,
                        "target": step.target,
                        "confidence": step.confidence,
                        "forward": step.forward,
                    }
                    for step in path.steps
                ]
                if path is not None
                else []
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(describe_connection(path, args.source, args.target))
    return EXIT_OK


def run_conflicts(args: argparse.Namespace) -> int:
    """Report contradictory relations in the graph, without resolving them."""
    from core.relation_consistency import relation_conflicts

    ontology = CognitiveController().ontology
    conflicts = relation_conflicts(ontology, args.concept)
    if args.json:
        print(json.dumps(
            {
                "concept": args.concept,
                "conflicts": [
                    {
                        "source": conflict.source,
                        "relation_type": conflict.relation_type,
                        "target": conflict.target,
                        "opposing_type": conflict.opposing_type,
                        "severity": conflict.severity,
                    }
                    for conflict in conflicts
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if not conflicts:
        scope = f" on {args.concept!r}" if args.concept else ""
        print(f"No contradictory relations found{scope}.")
        return EXIT_OK
    for conflict in conflicts:
        print(f"  {conflict.describe()}")
    return EXIT_OK


def run_infer(args: argparse.Namespace) -> int:
    """Show what the graph *entails* about a question, without storing anything."""
    from core.relation_inference import (
        DEFAULT_INFERENCE_LIMIT,
        DEFAULT_MIN_CONFIDENCE,
        inferred_relations,
    )

    question = args.question or ""
    if not question.strip():
        print("Nothing to reason about: pass a question or a concept.")
        return EXIT_NOTHING
    limit = args.limit if args.limit is not None else DEFAULT_INFERENCE_LIMIT
    min_confidence = args.min_confidence if args.min_confidence is not None else DEFAULT_MIN_CONFIDENCE
    if limit < 1:
        print("--limit must be at least 1.")
        return EXIT_USAGE
    if not 0.0 <= min_confidence <= 1.0:
        print("--min-confidence must be between 0 and 1.")
        return EXIT_USAGE
    ontology = CognitiveController().ontology
    inferences = inferred_relations(
        ontology, question, limit=limit, min_confidence=min_confidence
    )
    if args.json:
        print(json.dumps(
            {
                "question": question,
                "inferred": [
                    {
                        "source": item.source,
                        "relation_type": item.relation_type,
                        "target": item.target,
                        "confidence": item.confidence,
                        "steps": list(item.steps),
                    }
                    for item in inferences
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if not inferences:
        print(f"The graph entails nothing on {question!r}.")
        return EXIT_OK
    print("Inferred from the knowledge graph (derived, not stored):")
    for item in inferences:
        print(f"  - {item.explain()}  (confidence {item.confidence:.2f})")
    return EXIT_OK


def run_connections(args: argparse.Namespace) -> int:
    """Show how the concepts a question names are connected, chain by chain."""
    from core.relation_connections import (
        DEFAULT_CONNECTION_LIMIT,
        concept_connections,
        describe_concept_connections,
    )
    from core.relation_paths import DEFAULT_MAX_HOPS

    question = args.question or ""
    if not question.strip():
        print("Nothing to reason about: pass a question naming two or more concepts.")
        return EXIT_NOTHING
    limit = args.limit if args.limit is not None else DEFAULT_CONNECTION_LIMIT
    max_hops = args.max_hops if args.max_hops is not None else DEFAULT_MAX_HOPS
    if limit < 1:
        print("--limit must be at least 1.")
        return EXIT_USAGE
    if max_hops < 1:
        print("--max-hops must be at least 1.")
        return EXIT_USAGE
    ontology = CognitiveController().ontology
    connections = concept_connections(ontology, question, limit=limit, max_hops=max_hops)
    if args.json:
        print(json.dumps(
            {
                "question": question,
                "connections": [
                    {
                        "source": item.source,
                        "target": item.target,
                        "hops": item.hops,
                        "confidence": item.confidence,
                        "steps": list(item.steps),
                    }
                    for item in connections
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if not connections:
        print(f"The graph connects no two concepts in {question!r} within the hop bound.")
        return EXIT_OK
    print(describe_concept_connections(connections))
    return EXIT_OK


def run_causal(args: argparse.Namespace) -> int:
    """Show the directed causal paths the graph states between a question's concepts."""
    from core.causal_paths import (
        DEFAULT_CAUSAL_LIMIT,
        DEFAULT_MAX_DEPTH,
        causal_explanations,
        describe_causal_explanations,
    )

    question = args.question or ""
    if not question.strip():
        print("Nothing to reason about: pass a question naming two or more concepts.")
        return EXIT_NOTHING
    limit = args.limit if args.limit is not None else DEFAULT_CAUSAL_LIMIT
    max_depth = args.max_depth if args.max_depth is not None else DEFAULT_MAX_DEPTH
    if limit < 1:
        print("--limit must be at least 1.")
        return EXIT_USAGE
    if max_depth < 1:
        print("--max-depth must be at least 1.")
        return EXIT_USAGE
    ontology = CognitiveController().ontology
    explanations = causal_explanations(
        ontology, question, limit=limit, max_depth=max_depth
    )
    if args.json:
        print(json.dumps(
            {
                "question": question,
                "causal_paths": [
                    {
                        "cause": item.cause,
                        "effect": item.effect,
                        "hops": item.hops,
                        "confidence": item.confidence,
                        "nodes": list(item.nodes),
                        "mechanisms": list(item.mechanisms),
                        "sentence": item.sentence,
                    }
                    for item in explanations
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if not explanations:
        print(f"The graph states no causal path between the concepts in {question!r}.")
        return EXIT_OK
    print(describe_causal_explanations(explanations))
    return EXIT_OK


def run_why(args: argparse.Namespace) -> int:
    """Answer a question and print the full reasoning trace behind the answer."""
    from ai.jarvis import build_session

    question = args.question or ""
    if not question.strip():
        print("Nothing to explain: pass a question.")
        return EXIT_NOTHING
    if args.limit < 1:
        print("--limit must be at least 1.")
        return EXIT_USAGE
    session = build_session(text_only=True)
    answer = session.assistant.ask(question, limit=args.limit)
    report = answer.reasoning
    if args.json:
        payload = {
            "question": question,
            "answer": answer.text,
            "confidence": answer.confidence,
            "used_model": answer.used_model,
            "citations": list(answer.citations),
            "plan": (
                {"topic": answer.plan.topic, "source": answer.plan.source, "steps": list(answer.plan.steps)}
                if answer.plan is not None
                else None
            ),
            "decision": (
                {
                    "goal": answer.decision.goal,
                    "direction": answer.decision.direction,
                    "selected": answer.decision.selected,
                    "options": [
                        {"option": option, "relation": relation, "confidence": confidence}
                        for option, relation, confidence in answer.decision.options
                    ],
                }
                if answer.decision is not None
                else None
            ),
            "belief": (
                {
                    "statement": answer.belief.statement,
                    "confidence": answer.belief.confidence,
                    "status": answer.belief.status,
                    "supporting": list(answer.belief.supporting),
                    "contradicting": list(answer.belief.contradicting),
                }
                if answer.belief is not None
                else None
            ),
            "temporal": (
                {
                    "relation": answer.temporal.relation,
                    "change": answer.temporal.change,
                    "entries": [
                        {"role": entry.role, "text": entry.text, "at": entry.at.isoformat()}
                        for entry in answer.temporal.entries
                    ],
                    "pattern": (
                        {
                            "description": answer.temporal.pattern.description,
                            "occurrences": answer.temporal.pattern.occurrences,
                            "direction": answer.temporal.pattern.direction,
                        }
                        if answer.temporal.pattern is not None
                        else None
                    ),
                }
                if answer.temporal is not None
                else None
            ),
            "analogy": (
                {
                    "base": answer.analogy.base,
                    "target": answer.analogy.target,
                    "meaningful": answer.analogy.meaningful,
                    "score": answer.analogy.score,
                    "matched": list(answer.analogy.matched),
                    "transferred": [
                        {"source": source, "relation": relation, "target": target, "confidence": confidence}
                        for source, relation, target, confidence in answer.analogy.transferred
                    ],
                }
                if answer.analogy is not None
                else None
            ),
            "reasoning": (
                {
                    "steps": [
                        {"label": step.label, "lines": list(step.lines), "derived": step.derived}
                        for step in report.steps
                    ]
                }
                if report is not None
                else None
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(answer.text)
    if report is not None:
        print()
        print(report.describe())
    return EXIT_OK


def run_scan(args: argparse.Namespace) -> int:
    from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir
    from core.relation_extraction import extract_from_store

    store_dir = Path(args.store_dir) if args.store_dir else default_knowledge_base_dir()
    store = KnowledgeBaseStore(store_dir)
    if not store.shard_paths():
        print(f"No knowledge store at {store_dir}. Ingest something first "
              "(python -m cli ingest add <path>).")
        return EXIT_NOTHING
    controller = CognitiveController()
    extraction = extract_from_store(store, controller.relation_extractor, limit=args.limit)
    if args.json:
        print(json.dumps(_payload(extraction), ensure_ascii=False, indent=2))
        return EXIT_OK
    if not args.write:
        print(f"Preview only (pass --write to fold these into the graph): "
              f"{extraction.extracted} relation(s).")
        for relation in extraction.relations:
            print(f"  {relation.source} --{relation.relation_type}--> {relation.target}")
        print(f"\n{extraction.summary()}")
        return EXIT_OK
    learning = controller.learn_extracted_relations_from_store(store, limit=args.limit)
    for source, relation_type, target in learning.added:
        print(f"  added: {source} --{relation_type}--> {target}")
    for source, target, confidence in learning.reinforced:
        print(f"  reinforced: {source} -> {target} (confidence {confidence:.2f})")
    if not learning.changed:
        print("Nothing new was learned (no match, an unknown concept, or a seeded relation).")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "read": run_read,
        "learn": run_learn,
        "types": run_types,
        "connect": run_connect,
        "conflicts": run_conflicts,
        "infer": run_infer,
        "connections": run_connections,
        "causal": run_causal,
        "why": run_why,
        "scan": run_scan,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except Exception as error:  # pragma: no cover - a defensive last resort
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
