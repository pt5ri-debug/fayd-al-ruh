"""Menu-driven front end used by the portable build and by developers alike.

Each action delegates to the module that already owns the behaviour
(``main`` demo, ``ml`` pipeline, ``ml.llm`` trainer, ``evaluation`` suite), so
the menu adds a friendly entry point without duplicating logic.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from core.paths import artifacts_dir, data_dir, default_checkpoint

Action = Callable[[Sequence[str]], int]


def _run_module(module_name: str, argv: Sequence[str]) -> int:
    """Import ``module_name`` and run its ``main(argv)``."""
    module = importlib.import_module(module_name)
    main = getattr(module, "main")
    return int(main(list(argv)))


def missing_dependency_message(error: ModuleNotFoundError) -> str:
    """An actionable message for a missing third-party import."""
    name = (error.name or "").split(".")[0]
    lines = [f"Missing dependency: {name!r}."]
    if name in {"numpy", "requests"}:
        lines.append("Install the runtime requirements first:")
        lines.append("    python -m pip install -r requirements.txt")
    return "\n".join(lines)


def run_demo(argv: Sequence[str] = ()) -> int:
    """Run the small local cognitive-loop demonstration."""
    if argv:
        print("The demo takes no arguments.")
        return 1
    module = importlib.import_module("main")
    module.main()
    return 0


def run_ml(argv: Sequence[str] = ()) -> int:
    """Run the machine-learning pipeline, saving artifacts to the data folder."""
    args = list(argv)
    if "--save-dir" not in args:
        args += ["--save-dir", str(artifacts_dir())]
    return _run_module("ml.__main__", args)


def run_llm(argv: Sequence[str] = ()) -> int:
    """Train the in-project language model into the data folder."""
    return _run_module("ml.llm.__main__", argv)


def run_evaluation(argv: Sequence[str] = ()) -> int:
    """Run the scientific evaluation suite."""
    return _run_module("evaluation.__main__", argv)


def run_selfcheck(argv: Sequence[str] = ()) -> int:
    """Verify the install can load the bundled language model."""
    if argv:
        print("The self-check takes no arguments.")
        return 1
    from ai.local_model import LocalLanguageModel

    model = LocalLanguageModel(temperature=0.0)
    response = model.generator.generate("الماء", max_new_tokens=24)
    print(f"Checkpoint: {default_checkpoint()}")
    print(f"Sample    : {response.strip()[:120]}")
    return 0


MENU: tuple[tuple[str, Action, Sequence[str]], ...] = (
    ("Run the local cognitive demo", run_demo, ()),
    ("Train the ML pipeline", run_ml, ("--episodes", "60")),
    ("Train the language model", run_llm, ("--steps", "200")),
    ("Run the evaluation suite", run_evaluation, ()),
    ("Self-check the install", run_selfcheck, ()),
)


def describe_paths() -> str:
    """A short report of where the build keeps its assets and data."""
    checkpoint = default_checkpoint()
    return "\n".join(
        (
            f"Data folder     : {data_dir()}",
            f"Artifacts folder: {artifacts_dir()}",
            f"Checkpoint      : {checkpoint}",
        )
    )


def run_choice(choice: str) -> int:
    """Run the action numbered ``choice`` (1-based). Returns 0 for 'quit'."""
    text = choice.strip().lower()
    if text in {"q", "quit", "exit", ""}:
        return 0
    if not text.isdigit():
        print(f"Unrecognised choice: {choice!r}")
        return 1
    index = int(text) - 1
    if not 0 <= index < len(MENU):
        print(f"Choice out of range: {choice!r}")
        return 1
    _title, action, args = MENU[index]
    try:
        return action(args)
    except ModuleNotFoundError as error:
        print(missing_dependency_message(error))
        return 1


def _print_menu() -> None:
    print()
    print("Fayd Al-Ruh — portable menu")
    for number, (title, _action, _args) in enumerate(MENU, start=1):
        print(f"  {number}) {title}")
    print("  q) Quit")
    print(describe_paths())


def menu_loop(reader: Callable[[str], str] = input) -> int:
    """Interactive loop; returns the last non-zero exit code seen, else 0."""
    while True:
        _print_menu()
        try:
            choice = reader("Choose: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice.strip().lower() in {"q", "quit", "exit", ""}:
            return 0
        code = run_choice(choice)
        if code != 0:
            print(f"(action exited with code {code})")


__all__ = [
    "MENU",
    "run_demo",
    "run_ml",
    "run_llm",
    "run_evaluation",
    "run_choice",
    "menu_loop",
    "describe_paths",
    "missing_dependency_message",
]
