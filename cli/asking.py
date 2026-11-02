"""CLI for the learner-asks-the-teacher loop (Fayd Al-Ruh learning from a teacher).

    python -m cli ask-teacher                 # 20 questions, project model asks
    python -m cli ask-teacher --count 5       # a short run
    python -m cli ask-teacher --teacher ultron  # a chosen mouth answers
    python -m cli ask-teacher status          # who is the learner and the teacher
    python -m cli ask-teacher --json          # machine-readable turns

**Fayd Al-Ruh the learner** invents a question, the **linked model** -- Fayd
Al-Ruh the teacher -- answers it, the answer is verified against the project's own
graph and folded into the learner's knowledge, and the loop asks a *different*
question next. The answers are always the teacher's, never the learner's, and a
question never repeats.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` no model can answer,
``4`` a gate refused the run, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.cognitive_controller import CognitiveController

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_GATE = 4
EXIT_FAILURE = 5

#: The mouths a teacher may be chosen from, by the CLI flag value.
_TEACHER_MOUTHS: dict[str, str] = {
    "ultron": "ultron",
    "vision": "vision",
    "banner": "banner",
    "cloud": "cloud",
    "natasha": "natasha",
    "hakeem": "hakeem",
    "sources": "sources",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask-teacher", description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "status"),
        default="run",
        help="'run' (default) asks the teacher; 'status' reports who is who.",
    )
    parser.add_argument("--count", type=int, default=20, help="How many questions to ask (default 20).")
    parser.add_argument(
        "--teacher",
        default="",
        choices=("", *_TEACHER_MOUTHS),
        help="Which mouth answers (default: the best local model).",
    )
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Qualify at or above this.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _controller() -> CognitiveController:
    return CognitiveController()


def _teacher_backend(teacher: str):
    """A backend for the chosen teacher mouth, or the default local teacher.

    With no explicit ``--teacher`` the project's own default local teacher is used
    (a llama.cpp GGUF or a reachable Ollama model); an explicit mouth is built
    through :func:`ai.mouths.build_mouth` so the *same* mouth the chat uses answers
    here, and its availability error is surfaced as the real next step.
    """
    from cognition.backends import AssistantBackend

    if not teacher:
        from cognition.backends import LocalTeacherBackend

        backend = LocalTeacherBackend()
        if not backend.available():
            raise FileNotFoundError(
                "No local teacher model is available. Install one (see `python -m cli models`), "
                "or pass --teacher <ultron|vision|banner|cloud|natasha|hakeem>."
            )
        return backend
    from ai import mouths as mouths_module

    mouth = _TEACHER_MOUTHS[teacher]
    assistant = mouths_module.build_mouth(mouth)
    return AssistantBackend(assistant, name=mouth, can_propose=False)


def _student_backend():
    from cognition.backends import FaydStudentBackend

    return FaydStudentBackend()


def run_status(args: argparse.Namespace) -> int:
    from cognition.backends import build_backends

    student, teacher = build_backends()
    payload = {
        "learner": _describe(student, "the project's own model"),
        "learner_available": bool(student.available()),
        "teacher": _describe(teacher, "no local teacher model") if teacher else "no local teacher model",
        "teacher_available": teacher is not None,
        "next_step": (
            "ready: the learner can ask the teacher"
            if teacher is not None
            else "install a local teacher model, or pass --teacher <mouth>"
        ),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Learner-asks-teacher loop")
    print(f"  learner: {payload['learner']}")
    print(f"  teacher: {payload['teacher']}")
    print(f"  {payload['next_step']}")
    return EXIT_OK


def _describe(backend, fallback: str) -> str:
    """A backend's own description when it has one, else an honest fallback."""
    describe = getattr(backend, "describe", None)
    if callable(describe):
        return str(describe())
    return fallback


def run_loop(args: argparse.Namespace) -> int:
    if args.count < 1:
        print("--count must be at least 1.")
        return EXIT_USAGE
    try:
        student = _student_backend()
        teacher = _teacher_backend(args.teacher)
    except (FileNotFoundError, ModuleNotFoundError) as error:
        print(f"Not available: {error}")
        return EXIT_MODEL
    except Exception as error:
        print(f"Could not build a model: {type(error).__name__}: {error}")
        return EXIT_MODEL

    controller = _controller()
    loop = controller.asking_loop(student=student, teacher=teacher)
    turns: list = []

    def on_turn(turn) -> None:
        turns.append(turn)
        if not args.json:
            print(f"\nQ{turn.index}: {turn.question}")
            if turn.error:
                print(f"  (no answer: {turn.error})")
            else:
                print(f"  A ({turn.teacher_name}): {turn.answer}")
                verdict = "qualified" if turn.qualified else "kept (unverified)"
                print(f"  [{verdict}; {turn.reason}]")

    try:
        report = loop.run(count=args.count, on_turn=on_turn)
    except KeyboardInterrupt:
        loop.stop()
        report = None

    if args.json:
        payload = {
            "turns": [turn.to_primitive() for turn in turns],
            "report": report.to_primitive() if report is not None else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    if report is not None:
        print()
        print(report.describe())
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "status":
        return run_status(args)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
