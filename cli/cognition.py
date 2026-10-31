"""The Autonomous Cognitive System CLI -- the headless twin of the cognition panel.

Every command drives the *same* objects the GUI panel and the controller use, so
the CLI, the window, and the answer path share one proactive mind, one learning
system, one version log, and one evaluator -- no logic is reimplemented here.

Commands::

    python -m cli cognition status                 the mind's own snapshot
    python -m cli cognition think [--ideas N]      run one thinking cycle
    python -m cli cognition ideas [--limit N]      the ideas awaiting review
    python -m cli cognition drives                 what the mind would pursue next
    python -m cli cognition learn "<statement>"    learn a statement (with its source)
    python -m cli cognition recall "<query>"       retrieve from every memory
    python -m cli cognition experience --expect ... --outcome ... [--failed]
    python -m cli cognition train [--stage S] [--approve]  staged training
    python -m cli cognition train --gradual [--up-to S] [--approve]  gradual training
    python -m cli cognition versions               the version log
    python -m cli cognition rollback               return to the last stable version
    python -m cli cognition progression            how far gradual training has come
    python -m cli cognition evaluate [--compare]   the ten cognitive indicators
    python -m cli cognition teach "<question>"     one student-first exchange
    python -m cli cognition mode [light|balanced|intensive]  the resource mode
    python -m cli cognition tick                   one background tick

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` a model is not
available, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from cognition.evaluation import INDICATORS, LABELS, CognitiveEvaluator
from cognition.learning_system import CognitiveLearningSystem
from cognition.proactive_mind import ProactiveMind
from cognition.records import Source, SourceKind
from cognition.runner import CognitiveRunner, ResourceMode
from cognition.teacher_student import TeacherStudentSystem
from cognition.training import STAGE_SETTINGS, IncrementalTrainer, TrainingStage
from cognition.versions import ModelRegistry
from core.cognitive_controller import CognitiveController

#: Exit codes, the same convention every other CLI twin uses.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_GATE = 4
EXIT_FAILURE = 5

#: The default controller, built once so every command shares one graph.
_CONTROLLER: CognitiveController | None = None


def _controller() -> CognitiveController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = CognitiveController()
    return _CONTROLLER


def _mind() -> ProactiveMind:
    return _controller().proactive_mind()


def _learning() -> CognitiveLearningSystem:
    return _controller().cognitive_learning()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognition",
        description="The autonomous cognitive system: proactive mind, learning, training.",
    )
    # ``--json`` belongs to each subcommand, the convention every CLI twin here
    # uses (``python -m cli cognition evaluate --json``), not a global flag.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="the mind's own snapshot", parents=[common])

    think = sub.add_parser("think", help="run one thinking cycle", parents=[common])
    think.add_argument("--ideas", type=int, default=None, help="ideas to produce this cycle")

    prove = sub.add_parser("prove", help="prove the whole loop end to end", parents=[common])
    prove.add_argument("--ideas", type=int, default=None)
    prove.add_argument("--train", action="store_true", help="also run a smoke training round")

    ideas = sub.add_parser("ideas", help="the ideas awaiting review", parents=[common])
    ideas.add_argument("--limit", type=int, default=20)

    sub.add_parser("drives", help="what the mind would pursue next", parents=[common])

    learn = sub.add_parser("learn", help="learn a statement with its source", parents=[common])
    learn.add_argument("statement")
    learn.add_argument("--source", default="user", choices=[k.value for k in SourceKind])
    learn.add_argument("--confidence", type=float, default=0.6)
    learn.add_argument("--verified", action="store_true", help="treat it as verified")

    recall = sub.add_parser("recall", help="retrieve from every memory", parents=[common])
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=5)

    experience = sub.add_parser(
        "experience", help="record an expectation and its outcome", parents=[common]
    )
    experience.add_argument("--expect", required=True)
    experience.add_argument("--outcome", required=True)
    experience.add_argument("--failed", action="store_true", help="the outcome was a failure")

    train = sub.add_parser("train", help="run a staged training round", parents=[common])
    train.add_argument("--stage", choices=[s.value for s in TrainingStage], default="smoke")
    train.add_argument("--gradual", action="store_true", help="advance stage by stage")
    train.add_argument("--up-to", choices=[s.value for s in TrainingStage], default=None)
    train.add_argument("--approve", action="store_true", help="confirm the training run")

    sub.add_parser("versions", help="the model version log", parents=[common])
    sub.add_parser("rollback", help="return to the last stable version", parents=[common])
    sub.add_parser("progression", help="how far gradual training has come", parents=[common])

    evaluate = sub.add_parser(
        "evaluate", help="the ten cognitive indicators", parents=[common]
    )
    evaluate.add_argument("--compare", action="store_true", help="compare two runs")

    teach = sub.add_parser("teach", help="one student-first exchange", parents=[common])
    teach.add_argument("question")

    mode = sub.add_parser("mode", help="show or set the resource mode", parents=[common])
    mode.add_argument("value", nargs="?", choices=[m.value for m in ResourceMode])

    sub.add_parser("tick", help="one background tick (runs a cycle if due)", parents=[common])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:  # argparse already printed the reason
        return int(exit_.code or EXIT_USAGE)
    if not args.command:
        parser.print_help()
        return EXIT_OK
    if args.command not in _HANDLERS:
        parser.print_help()
        return EXIT_USAGE
    try:
        return _dispatch(args)
    except (FileNotFoundError, ModuleNotFoundError) as error:
        print(f"Not available: {error}")
        return EXIT_MODEL
    except ValueError as error:
        print(f"Refused: {error}")
        return EXIT_GATE
    except Exception as error:  # never traceback at the user
        print(f"Unexpected failure: {type(error).__name__}: {error}")
        return EXIT_FAILURE


#: The command table, filled at import time after the command functions are
#: defined, so ``main`` can reject an unknown command before running it.
_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {}


def _dispatch(args: argparse.Namespace) -> int:
    return _HANDLERS[args.command](args)


# -- commands ---------------------------------------------------------------
def _cmd_status(args: argparse.Namespace) -> int:
    status = _mind().status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    print("Autonomous cognitive system")
    print(f"  enabled: {status['enabled']}   paused: {status['paused']}")
    print(f"  cycles: {status['cycles']}   interval: {status['interval_seconds']}s")
    print(f"  ideas: {status['ideas']}")
    last = status.get("last_cycle")
    if isinstance(last, dict):
        print(f"  last cycle: {last.get('id')} ({last.get('duration_ms')} ms)")
    return 0


def _cmd_think(args: argparse.Namespace) -> int:
    cycle = _mind().cycle(max_ideas=args.ideas)
    if args.json:
        print(json.dumps(cycle.to_primitive(), ensure_ascii=False, indent=2))
        return 0
    print(cycle.describe())
    for note in cycle.notes:
        print(f"  drive: {note}")
    return 0


def _cmd_prove(args: argparse.Namespace) -> int:
    """Prove the whole loop on this machine: think -> evaluate -> learn -> train.

    This is the honest demonstration the task asks for: it shows the mind
    thinking, the ten indicators measured, the learning system acquiring a fact,
    and (with ``--train``) a staged training run that either improves the model or
    rolls back. Nothing is asserted that the run did not do.
    """
    controller = _controller()
    print("1. Thinking")
    cycle = controller.think(max_ideas=args.ideas)
    print(f"   {cycle.describe()}")

    print("2. Measuring the ten indicators")
    evaluator = CognitiveEvaluator(controller.ontology)
    report = evaluator.evaluate()
    print(report.describe())

    print("3. Learning a fact and recalling it")
    learning = controller.cognitive_learning()
    learning.learn_fact(
        "prove-1",
        "acid stimulation increases permeability",
        source=Source(SourceKind.CORPUS, "petroleum_corpus"),
        confidence=0.9,
        verified=True,
    )
    recalled = learning.recall("permeability", limit=3)
    print(f"   recalled: {recalled['semantic']}")

    if args.train:
        print("4. A staged training run (smoke)")
        outcome = IncrementalTrainer().run(stage=TrainingStage.SMOKE)
        print(f"   {outcome.describe()}")
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
    return 0


def _cmd_ideas(args: argparse.Namespace) -> int:
    ideas = _mind().pending_ideas(limit=args.limit)
    if args.json:
        print(json.dumps([i.to_primitive() for i in ideas], ensure_ascii=False, indent=2))
        return 0
    if not ideas:
        print("No ideas awaiting review. Run `cognition think` first.")
        return 0
    for idea in ideas:
        print(f"[{idea.kind.value}] {idea.text}")
        print(f"    engine {idea.engine}  confidence {idea.confidence:.2f}  novelty {idea.novelty:.2f}")
        for note in idea.critique:
            print(f"    critique: {note}")
    return 0


def _cmd_drives(args: argparse.Namespace) -> int:
    drives = _mind().top_drives()
    if args.json:
        print(json.dumps([d.describe() for d in drives], ensure_ascii=False, indent=2))
        return 0
    if not drives:
        print("Nothing to pursue: the graph is complete for the current gaps.")
        return 0
    for drive in drives:
        print(f"  {drive.describe()}")
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    source = Source(SourceKind(args.source), "cli")
    trusted = _learning().learn_fact(
        f"cli-{abs(hash(args.statement)) % 10**8}",
        args.statement,
        source=source,
        confidence=args.confidence,
        verified=args.verified,
    )
    verdict = "trusted fact" if trusted else "stored, not a fact (unverified)"
    if args.json:
        print(json.dumps({"trusted": trusted, "verdict": verdict}, ensure_ascii=False))
    else:
        print(f"Learned: {args.statement}")
        print(f"  {verdict}")
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    found = _learning().recall(args.query, limit=args.limit)
    if args.json:
        print(json.dumps(found, ensure_ascii=False, indent=2))
        return 0
    print(f"Recall for: {args.query}")
    for kind, items in found.items():
        print(f"  {kind}:")
        for item in items:
            print(f"    - {item}")
        if not items:
            print("    (nothing)")
    return 0


def _cmd_experience(args: argparse.Namespace) -> int:
    experience = _learning().learn_from_error(
        expectation=args.expect,
        expected_confidence=0.8,
        outcome=args.outcome,
        success=not args.failed,
    )
    if args.json:
        print(json.dumps(experience.to_primitive(), ensure_ascii=False, indent=2))
        return 0
    print(experience.describe())
    print(f"  lesson: {experience.lesson}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    if not args.approve:
        print("Training is a governed action: pass --approve to run it.")
        return EXIT_GATE
    trainer = IncrementalTrainer()
    if args.gradual:
        report = trainer.run_gradual(
            up_to=TrainingStage(args.up_to) if args.up_to else None,
            progress=lambda m: print(f"  | {m}"),
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "stopped_at": None if report.stopped_at is None else report.stopped_at.value,
                        "reason": report.reason,
                        "advanced": [s.value for s in report.advanced()],
                    },
                    ensure_ascii=False,
                )
            )
            return EXIT_OK
        print(report.describe())
        print(trainer.registry.report())
        return EXIT_OK
    stage = TrainingStage(args.stage)
    outcome = trainer.run(stage=stage, progress=lambda m: print(f"  | {m}"))
    if args.json:
        print(json.dumps({"accepted": outcome.accepted, "reason": outcome.reason}, ensure_ascii=False))
        return EXIT_OK
    print(outcome.describe())
    print(trainer.registry.report())
    return EXIT_OK


def _cmd_versions(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    if args.json:
        print(json.dumps([v.to_primitive() for v in registry.versions()], ensure_ascii=False, indent=2))
        return 0
    print(registry.report())
    return 0


def _cmd_progression(args: argparse.Namespace) -> int:
    trainer = IncrementalTrainer()
    progress = trainer.progression()
    next_stage = trainer.next_stage()
    if args.json:
        print(
            json.dumps(
                {
                    "stages": [
                        {
                            "stage": p.stage.value,
                            "accepted": p.accepted,
                            "attempts": p.attempts,
                            "done": p.done,
                        }
                        for p in progress
                    ],
                    "next": None if next_stage is None else next_stage.value,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK
    for p in progress:
        print(p.describe())
    print(f"next stage: {next_stage.value if next_stage else 'none (all done)'}")
    return EXIT_OK


def _cmd_rollback(args: argparse.Namespace) -> int:
    stable = ModelRegistry().rollback()
    if stable is None:
        print("No accepted version to roll back to; the base model is in use.")
        return 0
    print(f"Rolled back to v{stable.number}.")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    evaluator = CognitiveEvaluator(_controller().ontology)
    report = evaluator.evaluate()
    if args.compare:
        second = evaluator.evaluate()
        comparison = evaluator.compare(report, second)
        if args.json:
            print(json.dumps(
                {"before": report.to_primitive(), "after": second.to_primitive()},
                ensure_ascii=False, indent=2,
            ))
            return 0
        print(comparison.describe())
        return 0
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
        return 0
    print(report.describe())
    return 0


def _cmd_teach(args: argparse.Namespace) -> int:
    from cognition.backends import build_backends

    student, teacher = build_backends()
    if not student.available():
        print("The student model is not available; train one with `python -m ml.llm`.")
        return 3
    system = TeacherStudentSystem(student, teacher, ontology=_controller().ontology)
    outcome = system.teach(args.question)
    if args.json:
        print(json.dumps(
            {
                "question": outcome.question,
                "teacher_used": outcome.teacher_used,
                "student_quality": outcome.student_quality.score,
                "example_qualified": bool(outcome.example and outcome.example.qualified),
                "notes": list(outcome.notes),
            },
            ensure_ascii=False, indent=2,
        ))
        return 0
    print(outcome.describe())
    for note in outcome.notes:
        print(f"  {note}")
    if outcome.example is not None:
        print(f"  example: {outcome.example.describe()}")
    return 0


def _cmd_mode(args: argparse.Namespace) -> int:
    runner = CognitiveRunner(_mind())
    if args.value is None:
        status = runner.status()
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(f"Resource mode: {status['mode']}")
            print(f"  load: {status['load']}")
            print(f"  next due: {status['next_due']}")
        return 0
    runner.set_mode(ResourceMode(args.value))
    print(f"Resource mode set to {args.value}.")
    return 0


def _cmd_tick(args: argparse.Namespace) -> int:
    runner = CognitiveRunner(_mind())
    cycle = runner.tick()
    if cycle is None:
        decision = runner.decision()
        reason = decision.reason if decision is not None else "not due"
        if args.json:
            print(json.dumps({"ran": False, "reason": reason}, ensure_ascii=False))
        else:
            print(f"No cycle this tick: {reason}")
        return 0
    if args.json:
        print(json.dumps(cycle.to_primitive(), ensure_ascii=False, indent=2))
    else:
        print(cycle.describe())
    return 0


_HANDLERS.update(
    {
        "status": _cmd_status,
        "think": _cmd_think,
        "prove": _cmd_prove,
        "ideas": _cmd_ideas,
        "drives": _cmd_drives,
        "learn": _cmd_learn,
        "recall": _cmd_recall,
        "experience": _cmd_experience,
        "train": _cmd_train,
        "versions": _cmd_versions,
        "rollback": _cmd_rollback,
        "progression": _cmd_progression,
        "evaluate": _cmd_evaluate,
        "teach": _cmd_teach,
        "mode": _cmd_mode,
        "tick": _cmd_tick,
    }
)

__all__ = ["INDICATORS", "LABELS", "STAGE_SETTINGS", "main"]
