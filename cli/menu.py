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


def run_petroleum(argv: Sequence[str] = ()) -> int:
    """Train the policy and brain on every petroleum-engineering discipline."""
    args = list(argv)
    if "--save-dir" not in args:
        args += ["--save-dir", str(artifacts_dir() / "petroleum")]
    return _run_module("ml.petroleum", args)


def run_release(argv: Sequence[str] = ()) -> int:
    """Train every learning layer and store all artifacts for a release."""
    args = list(argv)
    if "--save-dir" not in args:
        args += ["--save-dir", str(artifacts_dir())]
    return _run_module("ml.release", args)


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
    response = model.generator.generate("water", max_new_tokens=24)
    print(f"Checkpoint: {default_checkpoint()}")
    print(f"Sample    : {response.strip()[:120]}")
    return 0


def run_voice(argv: Sequence[str] = ()) -> int:
    """Run one voice turn through the assistant (transcript in, WAV out)."""
    return _run_module("ai.voice", argv)


def run_agent(argv: Sequence[str] = ()) -> int:
    """Run a tool plan inside the safety sandbox."""
    return _run_module("ai.agent_demo", argv)


def run_jarvis(argv: Sequence[str] = ()) -> int:
    """Run Jarvis mode: listen, understand, act, and speak."""
    return _run_module("ai.jarvis", argv)


def run_chat(argv: Sequence[str] = ()) -> int:
    """Talk with the AI in text, like Jarvis without the voice."""
    return _run_module("ai.chat", argv)


def run_learn(argv: Sequence[str] = ()) -> int:
    """Demonstrate continuous learning and the regression gate."""
    return _run_module("ai.learn_demo", argv)


def run_insight(argv: Sequence[str] = ()) -> int:
    """Ask the AI what it can do and what it has learned (report or question)."""
    return _run_module("ai.insight", argv)


def run_internet_learner(argv: Sequence[str] = ()) -> int:
    """Learn petroleum knowledge from trusted internet sites into the data store."""
    return _run_module("ai.internet_learner", argv)


def run_web(argv: Sequence[str] = ()) -> int:
    """Answer from the web, rate trust, and learn trusted answers."""
    return _run_module("ai.web_verify", argv)


def run_knowledge(argv: Sequence[str] = ()) -> int:
    """Build, inspect, or search the bulk knowledge base."""
    return _run_module("cli.knowledge", argv)


def run_curriculum(argv: Sequence[str] = ()) -> int:
    """Show the seeded petroleum-engineering curriculum and its coverage."""
    if argv:
        print("The curriculum view takes no arguments.")
        return 1
    from collections import defaultdict

    from core.curriculum_knowledge import (
        course_categories,
        course_labels,
        curriculum_coverage,
        seed_curriculum,
    )
    from core.domain_knowledge import seed_ontology
    from core.ontology import Ontology
    from core.petroleum_corpus import corpus_coverage, seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    seed_petroleum_corpus(ontology)
    coverage = curriculum_coverage(ontology)
    labels = course_labels()
    categories = course_categories()

    grouped: dict[str, list[str]] = defaultdict(list)
    for key, counts in coverage.items():
        grouped[categories[key]].append(
            f"  {labels[key]:32s} "
            f"concepts={counts['concepts']}/{counts['expected_concepts']} "
            f"relations={counts['relations']}/{counts['expected_relations']}"
        )

    print("Seeded petroleum-engineering curriculum")
    for category in ("Foundations", "Earth Sciences", "Engineering Science", "Petroleum Disciplines"):
        if category in grouped:
            print(f"\n[{category}]")
            print("\n".join(grouped[category]))

    corpus = corpus_coverage(ontology)
    corpus_concepts = sum(counts["concepts"] for counts in corpus.values())
    corpus_concepts_expected = sum(counts["expected_concepts"] for counts in corpus.values())
    corpus_relations = sum(counts["relations"] for counts in corpus.values())
    corpus_relations_expected = sum(counts["expected_relations"] for counts in corpus.values())
    print("\n[Petroleum corpus]")
    for key, counts in corpus.items():
        print(
            f"  {key:22s} "
            f"concepts={counts['concepts']}/{counts['expected_concepts']} "
            f"relations={counts['relations']}/{counts['expected_relations']}"
        )
    print(
        f"\nCourses: {len(coverage)}  "
        f"Concepts: {len(ontology.concepts)}  "
        f"Relations: {len(ontology.relations)}"
    )
    print(
        f"Corpus entries: {len(corpus)} disciplines  "
        f"concepts={corpus_concepts}/{corpus_concepts_expected}  "
        f"relations={corpus_relations}/{corpus_relations_expected}"
    )
    return 0


MENU: tuple[tuple[str, Action, Sequence[str]], ...] = (
    ("One voice turn", run_voice, ("--write-wav", "voice-turn.wav", "water")),
    ("Tool agent demo (sandbox)", run_agent, ()),
    ("Jarvis mode (listen, act, speak)", run_jarvis, ()),
    ("Chat with the AI (text, like Jarvis without voice)", run_chat, ()),
    ("Continuous learning and regression gate", run_learn, ()),
    ("Ask the AI: what can it do and what has it learned", run_insight, ()),
    (
        "Continuous Internet Learner",
        run_internet_learner,
        ("--interactive", "--approve", "--promote"),
    ),
    ("Ask the web and verify the answer", run_web, ("--approve",)),
    ("Build the bulk knowledge base (100 MB)", run_knowledge, ("build", "--target-mb", "100")),
    ("Show the bulk knowledge base status", run_knowledge, ("stats",)),
    ("Run the local cognitive demo", run_demo, ()),
    ("Train the ML pipeline", run_ml, ("--episodes", "60")),
    (
        "Train the language model from all repositories",
        run_llm,
        ("--steps", "200", "--kb-max-chunks", "2000"),
    ),
    (
        "Train petroleum-engineering tasks (with procedures)",
        run_petroleum,
        ("--episodes", "120", "--procedural"),
    ),
    ("Show the petroleum-engineering curriculum", run_curriculum, ()),
    ("Train everything for a release", run_release, ("--episodes", "120", "--llm-steps", "4000")),
    ("Run the evaluation suite", run_evaluation, ()),
    ("Self-check the install", run_selfcheck, ()),
)


def describe_paths() -> str:
    """A short report of where the build keeps its assets and data."""
    checkpoint = default_checkpoint()
    lines = [
        f"Data folder     : {data_dir()}",
        f"Artifacts folder: {artifacts_dir()}",
        f"Checkpoint      : {checkpoint}",
    ]
    from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir

    kb_dir = default_knowledge_base_dir()
    stats = KnowledgeBaseStore(kb_dir).stats()
    if stats["shards"]:
        lines.append(f"Knowledge base  : {kb_dir} ({stats['size_mb']:.1f} MB, {stats['chunks']} chunks)")
    else:
        lines.append("Knowledge base  : absent (run `python -m cli knowledge build`)")
    return "\n".join(lines)


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
    "run_voice",
    "run_agent",
    "run_jarvis",
    "run_chat",
    "run_learn",
    "run_insight",
    "run_internet_learner",
    "run_web",
    "run_demo",
    "run_ml",
    "run_llm",
    "run_petroleum",
    "run_curriculum",
    "run_release",
    "run_evaluation",
    "run_selfcheck",
    "run_choice",
    "menu_loop",
    "describe_paths",
    "missing_dependency_message",
]
