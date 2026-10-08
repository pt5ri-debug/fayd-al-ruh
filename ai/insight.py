"""Ask the project's own AI how well it learns, and what it has learned.

    python -m cli insight                            # learning report (no model needed)
    python -m cli insight "what is porosity?"        # ask the AI, with citations
    python -m cli insight --interactive              # ask questions until 'quit'
    python -m cli insight --json                     # machine-readable report

The report answers two questions honestly.

* **How well does it learn?** Every learning layer reports the metric it was
  trained and measured on -- the sandbox policies and predictive-coding brains,
  the supervised and general-knowledge heads, the link-prediction score over the
  whole relation graph, the petroleum tasks and their procedural coverage, and
  the in-project language model. The numbers are read from the release manifest
  (``python -m ml.release``), so the report repeats a *recorded* result rather
  than a fresh, unverified claim; when no manifest exists it says so.
* **What has it learned?** The knowledge repositories it was trained on, the
  learned statements promoted into the project, and the knowledge gained at
  runtime (internet learning, Jarvis feedback).

It also answers questions. With the bundled language model it writes the prose;
without one it answers extractively from the retrieved passages and still cites
its sources, so a fresh clone is useful either way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.assistant import AssistantAnswer, GroundedAssistant, build_default_assistant
from core.composition import sources_block
from core.documents import build_document_store
from core.paths import artifacts_dir

#: The manifest filename written by :mod:`ml.release` (kept as a literal so the
#: report does not import the whole training stack just to read a path).
_MANIFEST_NAME = "release_manifest.json"

#: The metric the link-prediction head is compared against: 1 in
#: ``ml.general_learning._RANKING_CANDIDATES``, i.e. guessing among four.
_LINK_BASELINE = 0.25


def manifest_path() -> Path:
    """Where the release manifest is expected to live."""
    return artifacts_dir() / _MANIFEST_NAME


def load_manifest(path: Path | None = None) -> dict[str, Any] | None:
    """Read the release manifest, or ``None`` when it is absent or unreadable."""
    target = path or manifest_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# -- how well it learns -------------------------------------------------------


def learning_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per recorded learning result, read straight from the manifest."""
    rows: list[dict[str, Any]] = []
    base = manifest.get("base")
    if isinstance(base, dict):
        policies = base.get("policies") or {}
        if policies:
            rows.append(_mean_row("Sandbox policies", "held-out success", policies, "held_out_success_rate"))
            rows.append(_mean_row("Sandbox policies", "train success", policies, "train_success_rate"))
        brains = base.get("brains") or {}
        if brains:
            rows.append(_mean_row("Predictive-coding brains", "error improvement", brains, "error_improvement"))
            rows.append(_mean_row("Predictive-coding brains", "final error", brains, "final_error"))
        supervised = base.get("supervised") or {}
        if supervised:
            rows.append(_row("Supervised outcome head", "validation accuracy", supervised.get("val_accuracy"), "ratio"))
            rows.append(_row("Supervised outcome head", "validation Brier", supervised.get("val_brier"), "lower-better"))
        general = base.get("general_knowledge") or {}
        if general:
            rows.append(
                _row(
                    "Knowledge-graph link prediction",
                    "held-out Hit@1",
                    general.get("hit_at_1"),
                    "ratio",
                    detail=f"baseline {_ratio(_LINK_BASELINE)}",
                )
            )
            rows.append(_row("Knowledge-graph link prediction", "held-out accuracy", general.get("held_out_accuracy"), "ratio"))
            rows.append(_row("Knowledge-graph link prediction", "generalisation gain", general.get("generalisation_gain"), "delta"))
        interaction = base.get("interaction_reward_model") or {}
        if interaction:
            rows.append(_row("Learned reward model", "Brier score", interaction.get("brier_score"), "lower-better"))
            rows.append(
                _row("Learned reward model", "calibration error", interaction.get("expected_calibration_error"), "lower-better")
            )
        generalisation = base.get("generalisation") or {}
        if generalisation:
            rows.append(_mean_row("Held-out generalisation", "policy success", generalisation, "policy_success_rate"))

    petroleum = manifest.get("petroleum")
    if isinstance(petroleum, dict):
        rows.append(_row("Petroleum tasks", "overall success", petroleum.get("overall_success"), "ratio"))
        if "procedural_tasks" in petroleum:
            procedural = petroleum["procedural_tasks"]
            rows.append(
                _row(
                    "Procedural tasks",
                    "coverage",
                    petroleum.get("procedural_coverage"),
                    "ratio",
                    detail=f"{procedural.get('covered')}/{procedural.get('total')} corpus procedures",
                )
            )
        tasks = petroleum.get("tasks") or []
        if tasks:
            rows.append(_mean_row("Petroleum tasks", "held-out success", {t["name"]: t for t in tasks}, "held_out_success"))

    language_model = manifest.get("language_model")
    if isinstance(language_model, dict):
        rows.append(_row("Language model", "training steps", language_model.get("steps"), "count"))
        rows.append(_row("Language model", "parameters", language_model.get("parameters"), "count"))
    return rows


def _row(layer: str, metric: str, value: Any, unit: str, *, detail: str = "") -> dict[str, Any]:
    return {"layer": layer, "metric": metric, "value": value, "unit": unit, "detail": detail}


def _mean_row(layer: str, metric: str, group: dict[str, Any], key: str) -> dict[str, Any]:
    values = [entry[key] for entry in group.values() if isinstance(entry, dict) and isinstance(entry.get(key), (int, float))]
    mean = sum(values) / len(values) if values else None
    return _row(layer, metric, mean, "mean", detail=f"{len(values)} tasks")


def _ratio(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _format_value(row: dict[str, Any]) -> str:
    value = row["value"]
    unit = row["unit"]
    if not isinstance(value, (int, float)):
        return "—"
    if unit in {"ratio", "mean"}:
        return f"{value:.3f}"
    if unit == "delta":
        return f"{value:+.3f}"
    if unit == "lower-better":
        return f"{value:.4f}"
    if unit == "count":
        return f"{int(value):,}"
    return str(value)


def format_learning_report(manifest: dict[str, Any] | None) -> str:
    """A readable table of every recorded learning result."""
    lines = ["How well the project learns"]
    if manifest is None:
        lines.append(f"  No release manifest at {manifest_path()}.")
        lines.append("  Train one with `python -m ml.release` to record real results here.")
        return "\n".join(lines)

    rows = learning_rows(manifest)
    lines.append(
        f"  Release {manifest.get('version', '?')} "
        f"trained {manifest.get('trained_at', '?')} "
        f"at revision {manifest.get('git_revision') or '—'}"
    )
    if not rows:
        lines.append("  The manifest carries no result sections (was it written with --manifest-only?).")
        return "\n".join(lines)

    width = max(len(row["layer"]) for row in rows)
    for row in rows:
        detail = f"  ({row['detail']})" if row["detail"] else ""
        lines.append(f"  {row['layer']:<{width}}  {row['metric']:<22} {_format_value(row):>14}{detail}")
    return "\n".join(lines)


# -- what it has learned ------------------------------------------------------


def knowledge_report() -> dict[str, Any]:
    """The repositories the project holds, and how much each contributes."""
    from ml.knowledge_corpus import knowledge_sample

    sample = knowledge_sample()
    from core.learned_knowledge import learned_documents, learned_path

    promoted = learned_documents()
    knowledge_base = _knowledge_base_stats()
    return {
        "repositories": dict(sorted(sample.counts.items())),
        "available": dict(sorted(sample.available.items())),
        "documents_seen": len(sample),
        "promoted": {"count": len(promoted), "path": str(learned_path())},
        "knowledge_base": knowledge_base,
    }


def _knowledge_base_stats() -> dict[str, Any]:
    from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir

    try:
        return KnowledgeBaseStore(default_knowledge_base_dir()).stats()
    except (OSError, ValueError):
        return {}


def format_knowledge_report(manifest: dict[str, Any] | None, report: dict[str, Any] | None = None) -> str:
    """A readable account of what the project knows, from seeds to runtime."""
    report = report or knowledge_report()
    lines = ["What the project has learned"]

    knowledge = (manifest or {}).get("knowledge")
    if isinstance(knowledge, dict):
        lines.append(
            f"  Authored knowledge: {knowledge.get('disciplines')} disciplines, "
            f"{knowledge.get('courses')} courses, "
            f"{knowledge.get('ontology_concepts')} concepts, "
            f"{knowledge.get('ontology_relations')} relations"
        )
        lines.append(
            f"  Coverage: curriculum {_ratio(knowledge.get('curriculum_coverage'))}, "
            f"corpus {_ratio(knowledge.get('corpus_coverage'))}"
        )

    lines.append("  Repositories trained on:")
    for name, count in report["repositories"].items():
        lines.append(f"    {name:<22} {count:>6} documents")

    promoted = report["promoted"]
    lines.append(f"  Promoted learned statements: {promoted['count']} ({promoted['path']})")

    runtime = report["repositories"].get("runtime", 0)
    lines.append(f"  Learned at runtime (internet / Jarvis feedback): {runtime} documents")

    kb = report["knowledge_base"]
    if kb.get("shards"):
        lines.append(
            f"  Bulk knowledge base: {kb.get('size_mb')} MB, {kb.get('chunks')} chunks, "
            f"{kb.get('articles')} articles"
        )
    else:
        lines.append("  Bulk knowledge base: absent (build with `python -m cli knowledge build`)")
    return "\n".join(lines)


# -- asking the AI ------------------------------------------------------------


def _load_model() -> Any | None:
    """The bundled language model, or ``None`` when it is not installed."""
    try:
        from ai.local_model import LocalLanguageModel

        return LocalLanguageModel(temperature=0.0, seed=0)
    except (FileNotFoundError, OSError, ValueError):
        return None


def build_insight_assistant(*, model: Any | None = None, use_model: bool = True) -> GroundedAssistant:
    """Assemble the assistant the insight command asks its questions through."""
    if use_model and model is None:
        model = _load_model()
    return build_default_assistant(documents=build_document_store(), model=model)


def ask(question: str, *, assistant: GroundedAssistant | None = None, use_model: bool = True, limit: int = 4) -> AssistantAnswer:
    """Answer one question with citations, using the model when available."""
    assistant = assistant or build_insight_assistant(use_model=use_model)
    return assistant.ask(question, limit=limit, temperature=0.0, seed=0)


def format_answer(answer: AssistantAnswer) -> str:
    """The answer text, followed by its sources, as the assistant renders them."""
    parts = [answer.text.strip()]
    block = sources_block(answer.citations)
    if block:
        parts.append(block)
    return "\n\n".join(parts)


# -- command line -------------------------------------------------------------


def _report_payload(manifest: dict[str, Any] | None, knowledge: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest": {
            "present": manifest is not None,
            "path": str(manifest_path()),
            "version": (manifest or {}).get("version"),
            "trained_at": (manifest or {}).get("trained_at"),
            "git_revision": (manifest or {}).get("git_revision"),
        },
        "learning": learning_rows(manifest or {}),
        "knowledge": knowledge,
    }


def _run_interactive(assistant: GroundedAssistant) -> int:
    print("Ask the project's AI. Type 'quit' to exit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.casefold() in {"quit", "exit", "q", ""}:
            return 0
        print()
        print(format_answer(assistant.ask(question, limit=4, temperature=0.0, seed=0)))
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="insight",
        description="Ask the project's AI what it can do and what it has learned.",
    )
    parser.add_argument("question", nargs="*", help="a question for the AI; omit for the learning report")
    parser.add_argument("--report", action="store_true", help="print the learning report (default with no question)")
    parser.add_argument("--interactive", action="store_true", help="ask questions until 'quit'")
    parser.add_argument("--no-model", action="store_true", help="answer extractively; do not use the language model")
    parser.add_argument("--limit", type=int, default=4, help="maximum passages to retrieve per answer")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    if args.interactive:
        assistant = build_insight_assistant(use_model=not args.no_model)
        return _run_interactive(assistant)

    if args.question and not args.report:
        answer = ask(" ".join(args.question), use_model=not args.no_model, limit=args.limit)
        if args.json:
            print(
                json.dumps(
                    {
                        "question": " ".join(args.question),
                        "answer": answer.text,
                        "citations": list(answer.citations),
                        "confidence": answer.confidence,
                        "used_model": answer.used_model,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(format_answer(answer))
        return 0

    manifest = load_manifest()
    knowledge = knowledge_report()
    if args.json:
        print(json.dumps(_report_payload(manifest, knowledge), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(format_learning_report(manifest))
    print()
    print(format_knowledge_report(manifest, knowledge))
    return 0


__all__ = [
    "ask",
    "build_insight_assistant",
    "format_answer",
    "format_knowledge_report",
    "format_learning_report",
    "knowledge_report",
    "learning_rows",
    "load_manifest",
    "main",
    "manifest_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
