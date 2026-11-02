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
    python -m cli cognition characters [--key K]   evaluate the eight characters
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
from cognition.study import StudyTurn
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

    study = sub.add_parser(
        "study", help="ask the mind's own questions and report the benefit", parents=[common]
    )
    study.add_argument("--count", type=int, default=200, help="how many questions to ask")
    study.add_argument("--no-indicators", action="store_true", help="skip the ten indicators")
    study.add_argument("--approve", action="store_true", help="confirm the session (it learns)")

    questions = sub.add_parser(
        "questions", help="the mind's own question bank", parents=[common]
    )
    questions.add_argument("--next", type=int, default=0, help="preview the next N questions")
    questions.add_argument("--record", action="store_true", help="mark them asked")

    sub.add_parser("last-study", help="the last self-study benefit report", parents=[common])

    sub.add_parser("metacognition", help="the mind's model of its own thinking", parents=[common])
    sub.add_parser("plan", help="how the mind intends to search next", parents=[common])

    consolidate = sub.add_parser(
        "consolidate", help="turn raw experience into durable, reviewed knowledge", parents=[common]
    )
    consolidate.add_argument("--review", action="store_true", help="also settle due reviews")
    sub.add_parser("patterns", help="the regularities the mind has extracted", parents=[common])
    sub.add_parser("skills", help="the procedures the mind has practised", parents=[common])
    review = sub.add_parser("review", help="settle the spaced-review items that are due", parents=[common])
    review.add_argument("--forgot", action="store_true", help="mark them as forgotten instead")

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
    train.add_argument("--autonomous", action="store_true", help="run the adaptive curriculum")
    train.add_argument("--up-to", choices=[s.value for s in TrainingStage], default=None)
    train.add_argument("--max-stages", type=int, default=None, help="effort budget for --autonomous")
    train.add_argument("--approve", action="store_true", help="confirm the training run")

    sub.add_parser("versions", help="the model version log", parents=[common])
    rollback = sub.add_parser("rollback", help="return to the last stable version", parents=[common])
    rollback.add_argument("--approve", action="store_true", help="confirm the rollback")
    sub.add_parser("progression", help="how far gradual training has come", parents=[common])
    sub.add_parser("curriculum", help="the adaptive curriculum's training memory", parents=[common])

    evaluate = sub.add_parser(
        "evaluate", help="the ten cognitive indicators", parents=[common]
    )
    evaluate.add_argument("--compare", action="store_true", help="compare two runs")

    teach = sub.add_parser("teach", help="one student-first exchange", parents=[common])
    teach.add_argument("question")

    characters = sub.add_parser(
        "characters", help="evaluate the eight characters", parents=[common]
    )
    characters.add_argument("--key", default=None, help="evaluate only this character")

    mode = sub.add_parser("mode", help="show or set the resource mode", parents=[common])
    mode.add_argument("value", nargs="?", choices=[m.value for m in ResourceMode])

    sub.add_parser("tick", help="one background tick (runs a cycle if due)", parents=[common])

    # -- widening the mind's reach (knowledge depth) ----------------------
    sub.add_parser(
        "depth", help="how much of the project's knowledge the graph holds", parents=[common]
    )
    sub.add_parser(
        "coverage", help="one line per discipline and per course", parents=[common]
    )
    harvest = sub.add_parser(
        "harvest", help="read typed relations from the ingested text", parents=[common]
    )
    harvest.add_argument("--limit", type=int, default=500, help="how many chunks to read")
    harvest.add_argument("--write", action="store_true", help="fold them into the graph")
    harvest.add_argument("--approve", action="store_true", help="confirm the write")

    bridges = sub.add_parser(
        "bridges", help="chains that connect one discipline to another", parents=[common]
    )
    bridges.add_argument("--from", dest="source", default="", help="source discipline key")
    bridges.add_argument("--to", dest="target", default="", help="target discipline key")
    bridges.add_argument("--max-hops", type=int, default=3, help="how far to search")
    bridges.add_argument("--limit", type=int, default=5, help="how many bridges to show")

    sub.add_parser("arabic", help="how far the project reaches in Arabic", parents=[common])
    sub.add_parser(
        "catalog", help="what every cognition control does, and its command", parents=[common]
    )

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


def _cmd_study(args: argparse.Namespace) -> int:
    from cognition.study import StudySession

    def on_turn(turn: StudyTurn) -> None:
        if not args.json:
            print(turn.describe())  # stream the questions and answers as they happen

    session = StudySession(_controller(), measure_indicators=not args.no_indicators)
    result = session.run(count=args.count, on_turn=on_turn)
    if args.json:
        print(
            json.dumps(
                {
                    "report": result.report.to_primitive(),
                    "turns": [t.to_primitive() for t in result.turns],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print()
        print(result.report.describe())
    return 0


def _cmd_questions(args: argparse.Namespace) -> int:
    controller = _controller()
    if args.next > 0:
        questions = controller.next_questions(count=args.next, record=args.record)
        if args.json:
            print(
                json.dumps(
                    [{"text": q.text, "topic": q.topic, "kind": q.kind} for q in questions],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        for question in questions:
            print(f"  [{question.kind}] {question.text}")
        return 0
    bank = controller.question_bank()
    if args.json:
        print(json.dumps({"size": len(bank)}, ensure_ascii=False, indent=2))
    else:
        print(f"The mind's question bank holds {len(bank)} distinct questions.")
    return 0


def _cmd_last_study(args: argparse.Namespace) -> int:
    from cognition.study import last_study_report

    report = last_study_report()
    if report is None:
        print("No self-study session has been run yet.")
        return 0
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
    else:
        print(report.describe())
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
    if getattr(args, "autonomous", False):
        from cognition.curriculum import AutonomousCurriculum

        curriculum = AutonomousCurriculum(trainer=trainer)
        report = curriculum.run(
            approved=True,
            max_stages=args.max_stages,
            progress=lambda m: print(f"  | {m}"),
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "runs": len(report.outcomes),
                        "decisions": [
                            {"action": d.action, "stage": None if d.stage is None else d.stage.value, "reason": d.reason}
                            for d in report.decisions
                        ],
                        "stopped_because": report.stopped_because,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return EXIT_OK
        print(report.describe())
        print(trainer.registry.report())
        return EXIT_OK
    if args.gradual:
        gradual = trainer.run_gradual(
            up_to=TrainingStage(args.up_to) if args.up_to else None,
            progress=lambda m: print(f"  | {m}"),
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "stopped_at": None if gradual.stopped_at is None else gradual.stopped_at.value,
                        "reason": gradual.reason,
                        "advanced": [s.value for s in gradual.advanced()],
                    },
                    ensure_ascii=False,
                )
            )
            return EXIT_OK
        print(gradual.describe())
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
    """Return to the last accepted version.

    A real rollback changes which model is active, so it is governed and needs
    ``--approve``; when there is no accepted version to return to, the command is a
    no-op that reports the base model honestly and needs no confirmation.
    """
    registry = ModelRegistry()
    if not registry.accepted_versions():
        print("No accepted version to roll back to; the base model is in use.")
        return EXIT_OK
    if not args.approve:
        print("Refused: returning to a previous version is a governed action -- pass --approve.")
        return EXIT_GATE
    stable = registry.rollback()
    if stable is None:
        print("No accepted version to roll back to; the base model is in use.")
        return EXIT_OK
    print(f"Rolled back to v{stable.number}.")
    return EXIT_OK


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


def _cmd_characters(args: argparse.Namespace) -> int:
    """Evaluate the eight characters from their real configuration and stores."""
    from gui.character_evaluation import CharacterEvaluator

    evaluator = CharacterEvaluator()
    if args.key:
        try:
            character = evaluator.evaluate_key(args.key)
        except KeyError:
            print(f"Refused: {args.key!r} is not a known character.")
            return EXIT_GATE
        if args.json:
            print(json.dumps(character.to_primitive(), ensure_ascii=False, indent=2))
            return EXIT_OK
        print(character.describe())
        print("strengths:")
        for line in character.strengths or ("(none observed yet)",):
            print(f"  + {line}")
        print("shortcomings:")
        for line in character.shortcomings or ("(none)",):
            print(f"  - {line}")
        print("next steps:")
        for line in character.next_steps or ("(none)",):
            print(f"  → {line}")
        return EXIT_OK
    report = evaluator.evaluate()
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(report.describe())
    for character in report.characters:
        print()
        print(character.describe())
    return EXIT_OK


def _cmd_metacognition(args: argparse.Namespace) -> int:
    """The mind's model of its own thinking: which strategies pay off."""
    mind = _mind()
    if args.json:
        print(json.dumps(mind.metacognition.status(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(mind.metacognition.describe())
    return EXIT_OK


def _cmd_plan(args: argparse.Namespace) -> int:
    """How the mind intends to search next, and why."""
    plan = _mind().plan_next()
    if args.json:
        print(
            json.dumps(
                {
                    "concept_budget": plan.concept_budget,
                    "idea_budget": plan.idea_budget,
                    "novelty_floor": plan.novelty_floor,
                    "strategy_focus": plan.strategy_focus,
                    "focus_topics": list(plan.focus_topics),
                    "escalation": None if plan.escalation is None else plan.escalation.describe(),
                    "stagnation": None if plan.stagnation is None else plan.stagnation.describe(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK
    print(plan.describe())
    return EXIT_OK


def _cmd_consolidate(args: argparse.Namespace) -> int:
    """Turn the learning system's raw experience into durable, reviewed knowledge."""
    learning = _learning()
    report = learning.consolidate()
    if getattr(args, "review", False):
        learning.consolidator().review_due()
    if args.json:
        print(
            json.dumps(
                {
                    "patterns": [p.to_primitive() for p in report.patterns],
                    "upgraded": list(report.upgraded),
                    "scheduled": report.scheduled,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK
    print(report.describe())
    return EXIT_OK


def _cmd_patterns(args: argparse.Namespace) -> int:
    """The regularities the mind has extracted from its own experience."""
    patterns = _learning().consolidator().patterns()
    if args.json:
        print(json.dumps([p.to_primitive() for p in patterns], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not patterns:
        print("No patterns consolidated yet; record experiences, then run `consolidate`.")
        return EXIT_OK
    for pattern in patterns:
        print(pattern.describe())
    return EXIT_OK


def _cmd_skills(args: argparse.Namespace) -> int:
    """The procedures the mind has practised, with their measured proficiency."""
    skills = _learning().consolidator().skills.skills()
    if args.json:
        print(json.dumps([s.to_primitive() for s in skills], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not skills:
        print("No skills practised yet.")
        return EXIT_OK
    for skill in skills:
        print(skill.describe())
    return EXIT_OK


def _cmd_review(args: argparse.Namespace) -> int:
    """Settle the spaced-review items that are due."""
    consolidator = _learning().consolidator()
    due = consolidator.review_due(remembered=not getattr(args, "forgot", False))
    if args.json:
        print(json.dumps([i.to_primitive() for i in due], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not due:
        print("No knowledge is due for review yet.")
        return EXIT_OK
    for item in due:
        print(item.describe())
    return EXIT_OK


def _cmd_curriculum(args: argparse.Namespace) -> int:
    """The adaptive curriculum's training memory (what it tried and what happened)."""
    from cognition.curriculum import TrainingMemory

    memory = TrainingMemory.load()
    if args.json:
        print(json.dumps([a.to_primitive() for a in memory.attempts()], ensure_ascii=False, indent=2))
        return EXIT_OK
    print(memory.describe())
    return EXIT_OK


def _cmd_depth(args: argparse.Namespace) -> int:
    """How much of the project's own knowledge the graph actually holds."""
    report = _controller().knowledge_depth()
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(report.describe())
    return EXIT_OK


def _cmd_coverage(args: argparse.Namespace) -> int:
    """One line per discipline and per course, with the graph's connectivity."""
    report = _controller().knowledge_depth()
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Disciplines:")
    for coverage in report.disciplines:
        print(f"  {coverage.describe()}")
    print("Courses:")
    for coverage in report.courses:
        print(f"  {coverage.describe()}")
    print(
        f"connectivity: {report.connectivity:.0%}   isolated: {report.isolated}   "
        f"average degree: {report.average_degree:.2f}"
    )
    return EXIT_OK


def _cmd_harvest(args: argparse.Namespace) -> int:
    """Read typed relations from the ingested text, previewing or writing.

    Writing changes the graph, so it is a governed action: it needs ``--approve``
    (the same convention ``train`` uses). Without ``--write`` the run is a preview
    that touches nothing.
    """
    if args.write and not args.approve:
        print("Refused: writing to the graph is a governed action -- pass --approve.")
        return EXIT_GATE

    def on_progress(index: int) -> None:
        if not args.json and index % 100 == 0:
            print(f"  ... read {index} chunk(s)", flush=True)

    report = _controller().harvest_relations(
        limit=args.limit, write=args.write, on_progress=on_progress
    )
    if args.json:
        print(json.dumps(report.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(report.describe())
    return EXIT_OK


def _cmd_bridges(args: argparse.Namespace) -> int:
    """The shortest typed chains between one discipline's concepts and another's."""
    bridges = _controller().discipline_bridges(
        source=args.source or "", target=args.target or "", max_hops=args.max_hops, limit=args.limit
    )
    if args.json:
        payload = [
            {
                "source": b.source_label,
                "target": b.target_label,
                "hops": b.hops,
                "confidence": b.confidence,
                "steps": list(b.steps),
            }
            for b in bridges
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    if not bridges:
        print("No bridge found within the hop bound.")
        return EXIT_OK
    for bridge in bridges:
        print(bridge.describe())
    return EXIT_OK


def _cmd_arabic(args: argparse.Namespace) -> int:
    """How far the project reaches in Arabic, over the graph and the bank."""
    reach = _controller().arabic_reach()
    if args.json:
        print(json.dumps(reach.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(reach.describe())
    return EXIT_OK


def _cmd_catalog(args: argparse.Namespace) -> int:
    """What every cognition control does, with its headless command."""
    from core.cognition_catalog import CAPABILITIES, capability_groups

    if args.json:
        payload = [
            {
                "key": cap.key,
                "category": cap.category,
                "title": cap.title,
                "summary": cap.summary,
                "outcome": cap.outcome,
                "governed": cap.governed,
                "cli": cap.cli,
                "arabic_title": cap.arabic_title,
                "arabic_summary": cap.arabic_summary,
            }
            for cap in CAPABILITIES
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    from core.cognition_catalog import CATEGORY_TITLES

    for category, caps in capability_groups():
        if not caps:
            continue
        print(CATEGORY_TITLES.get(category, category))
        for cap in caps:
            kind = "governed" if cap.governed else "read-only"
            print(f"  {cap.title}  [{kind}]")
            print(f"    what: {cap.summary}")
            print(f"    returns: {cap.outcome}")
            print(f"    command: {cap.cli}")
        print()
    return EXIT_OK


_HANDLERS.update(
    {
        "status": _cmd_status,
        "think": _cmd_think,
        "prove": _cmd_prove,
        "ideas": _cmd_ideas,
        "drives": _cmd_drives,
        "study": _cmd_study,
        "questions": _cmd_questions,
        "last-study": _cmd_last_study,
        "metacognition": _cmd_metacognition,
        "plan": _cmd_plan,
        "consolidate": _cmd_consolidate,
        "patterns": _cmd_patterns,
        "skills": _cmd_skills,
        "review": _cmd_review,
        "learn": _cmd_learn,
        "recall": _cmd_recall,
        "experience": _cmd_experience,
        "train": _cmd_train,
        "versions": _cmd_versions,
        "rollback": _cmd_rollback,
        "progression": _cmd_progression,
        "curriculum": _cmd_curriculum,
        "evaluate": _cmd_evaluate,
        "teach": _cmd_teach,
        "mode": _cmd_mode,
        "tick": _cmd_tick,
        "characters": _cmd_characters,
        "depth": _cmd_depth,
        "coverage": _cmd_coverage,
        "harvest": _cmd_harvest,
        "bridges": _cmd_bridges,
        "arabic": _cmd_arabic,
        "catalog": _cmd_catalog,
    }
)

__all__ = ["INDICATORS", "LABELS", "STAGE_SETTINGS", "main"]
