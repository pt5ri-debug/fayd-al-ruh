"""CLI for Phase 3 -- cognitive and experience-based learning.

    python -m cli learning status            # what has been learned, by kind
    python -m cli learning knowledge "porosity increases permeability"
    python -m cli learning experience "goal" "it worked" --success
    python -m cli learning feedback correct "answer-1"
    python -m cli learning patterns          # extract versioned procedures
    python -m cli learning history           # the audit trail, newest first
    python -m cli learning metrics           # the measured learning metrics
    python -m cli learning evaluate "<text>" # measure the learning a text produces
    python -m cli learning rollback <version>  # restore an earlier version

**Learning is measured, not asserted.** Every command drives the *same*
:class:`core.learning_coordinator.LearningCoordinator` the controller and the GUI
use, so the CLI and the window share one learning store per personality and one
isolation policy. An unsupported claim is refused, a single success never promotes
a procedure, and a rollback keeps the audit trail.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the learning system is
unavailable, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.cognitive_controller import CognitiveController
from core.learning_isolation import LearningAccessError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_GATE = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learning", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="What has been learned, by kind.")
    status.add_argument("--owner", default="fayd_al_ruh", help="Which personality (default: the model).")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    knowledge = sub.add_parser("knowledge", help="Learn one knowledge item from text.")
    knowledge.add_argument("text", help="The statement to learn.")
    knowledge.add_argument("--source", default="user", help="Where it came from (default: user).")
    knowledge.add_argument("--reference", default="", help="A source reference.")
    knowledge.add_argument("--confidence", type=float, default=0.6, help="The source's reliability.")

    experience = sub.add_parser("experience", help="Record one experience and its outcome.")
    experience.add_argument("goal", help="What was being attempted.")
    experience.add_argument("outcome", help="What happened.")
    experience.add_argument("--success", action="store_true", help="Mark the experience successful.")
    experience.add_argument("--approach", default="", help="The approach used.")
    experience.add_argument("--importance", type=float, default=0.5, help="How important it is.")

    feedback = sub.add_parser("feedback", help="Process one feedback record.")
    feedback.add_argument(
        "kind",
        choices=(
            "correct",
            "incorrect",
            "missing",
            "preferred",
            "rejected",
            "correction",
            "task_success",
            "task_failure",
        ),
    )
    feedback.add_argument("subject", help="What the feedback is about.")
    feedback.add_argument("--correction", default="", help="The corrected text (for 'correction').")
    feedback.add_argument("--confidence", type=float, default=0.7, help="How sure the feedback is (0..1).")

    patterns = sub.add_parser("patterns", help="Extract versioned procedures from experiences.")
    patterns.add_argument("--min-support", type=int, default=2, help="Corroborating experiences needed.")

    history = sub.add_parser("history", help="The learning audit trail, newest first.")
    history.add_argument("--limit", type=int, default=20, help="How many events to show.")
    history.add_argument("--owner", default="fayd_al_ruh", help="Which personality.")
    history.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    metrics = sub.add_parser("metrics", help="The measured learning metrics.")
    metrics.add_argument("--owner", default="fayd_al_ruh", help="Which personality.")
    metrics.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    evaluate = sub.add_parser("evaluate", help="Measure the learning a text produces.")
    evaluate.add_argument("text", help="The statement to learn and measure.")
    evaluate.add_argument("--source", default="user", help="Where it came from.")

    rollback = sub.add_parser("rollback", help="Restore an earlier version (audit kept).")
    rollback.add_argument("version", type=int, help="The version to roll back to.")
    rollback.add_argument("--approve", action="store_true", help="confirm the rollback")

    for name in ("knowledge", "experience", "feedback", "patterns", "evaluate", "rollback"):
        sub.choices[name].add_argument("--owner", default="fayd_al_ruh", help="Which personality.")

    return parser


def _coordinator_for(owner: str):
    """The learning coordinator for one owner, on the **real** knowledge.

    The main model's coordinator is built over the controller's own graph,
    ontology, and bounded learner, so a text it learns from is read against the
    project's actual knowledge rather than an empty graph. A character's
    coordinator keeps its own isolated store (and no shared graph), exactly as the
    isolation policy requires.
    """
    from core.learning_coordinator import coordinator_for
    from core.learning_isolation import is_main_model, owner_for

    if is_main_model(owner_for(owner)):
        return CognitiveController().learning()
    return coordinator_for(owner)


def run_status(args: argparse.Namespace) -> int:
    coordinator = _coordinator_for(args.owner)
    status = coordinator.status()
    if args.json:
        print(json.dumps(status.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(status.describe())
    return EXIT_OK


def run_knowledge(args: argparse.Namespace) -> int:
    coordinator = _coordinator_for(args.owner)
    result = coordinator.learn_knowledge(
        args.text, source=args.source, source_reference=args.reference, confidence=args.confidence
    )
    print(result.describe())
    return EXIT_OK


def run_experience(args: argparse.Namespace) -> int:
    from core.experience_learning import experience_from_result

    coordinator = _coordinator_for(args.owner)
    experience = experience_from_result(
        args.goal,
        args.outcome,
        success=args.success,
        approach=args.approach,
        importance=args.importance,
    )
    result = coordinator.learn_experience(experience)
    print(result.describe())
    return EXIT_OK


def run_feedback(args: argparse.Namespace) -> int:
    coordinator = _coordinator_for(args.owner)
    outcome = coordinator.record_feedback(
        args.kind,
        args.subject,
        correction=args.correction,
        confidence=args.confidence,
    )
    print(outcome.describe())
    return EXIT_OK


def run_patterns(args: argparse.Namespace) -> int:
    coordinator = _coordinator_for(args.owner)
    coordinator.patterns.min_support = args.min_support
    report = coordinator.extract_patterns()
    print(report.describe())
    return EXIT_OK


def run_history(args: argparse.Namespace) -> int:
    coordinator = _coordinator_for(args.owner)
    events = coordinator.history(limit=args.limit)
    if args.json:
        print(json.dumps([event.to_primitive() for event in events], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not events:
        print("no learning events recorded yet.")
        return EXIT_OK
    for event in events:
        print(event.describe())
    return EXIT_OK


def run_metrics(args: argparse.Namespace) -> int:
    from core.learning_evaluation import learning_metrics

    coordinator = _coordinator_for(args.owner)
    metrics = learning_metrics(coordinator)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return EXIT_OK
    for name, value in sorted(metrics.items()):
        print(f"{name:24} {value:g}")
    return EXIT_OK


def run_evaluate(args: argparse.Namespace) -> int:
    from core.learning_evaluation import evaluate_learning

    coordinator = _coordinator_for(args.owner)
    report = evaluate_learning(
        coordinator,
        lambda: coordinator.learn_knowledge(args.text, source=args.source),
        name="learn knowledge",
    )
    print(report.describe())
    return EXIT_OK


def run_rollback(args: argparse.Namespace) -> int:
    """Roll the learning back to an earlier version.

    A real rollback changes what the system believes, so it is governed and needs
    ``--approve``; the reversal is append-only, so the audit trail survives.
    """
    if not args.approve:
        print("Refused: rolling the learning back is a governed action -- pass --approve.")
        return EXIT_GATE
    coordinator = _coordinator_for(args.owner)
    reversals = coordinator.rollback(args.version)
    if not reversals:
        print(f"nothing to roll back to version {args.version}.")
        return EXIT_OK
    print(f"rolled back {len(reversals)} event(s); the audit trail is kept.")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    if not args.command:
        parser.print_help()
        return EXIT_OK
    handlers = {
        "status": run_status,
        "knowledge": run_knowledge,
        "experience": run_experience,
        "feedback": run_feedback,
        "patterns": run_patterns,
        "history": run_history,
        "metrics": run_metrics,
        "evaluate": run_evaluate,
        "rollback": run_rollback,
    }
    try:
        return handlers[args.command](args)
    except LearningAccessError as error:
        print(f"refused: {error}")
        return EXIT_GATE
    except ValueError as error:
        print(f"invalid: {error}")
        return EXIT_USAGE
    except FileNotFoundError as error:
        print(f"not available: {error}")
        return EXIT_UNAVAILABLE
    except Exception as error:  # pragma: no cover - defensive
        print(f"unexpected failure: {type(error).__name__}: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
