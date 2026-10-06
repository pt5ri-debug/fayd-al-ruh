"""Ask the grounded assistant a question from the command line.

    python -m cli ask "What is the relationship between porosity and permeability?"
    python -m ai.ask --no-model "What is a well log?"
    python -m ai.ask            # interactive, multi-turn

The answer always carries the sources it was built from. When a trained
checkpoint is present the in-project language model writes the prose; otherwise
the assistant answers extractively from the retrieved passages, so the command
works on a fresh clone with no external service.
"""

from __future__ import annotations

import argparse
import json

from core.assistant import AssistantAnswer, build_default_assistant


def _load_model(checkpoint: str | None, *, enabled: bool):
    if not enabled:
        return None
    try:
        from ai.local_model import LocalLanguageModel

        return LocalLanguageModel(checkpoint, temperature=0.0)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _print_answer(answer: AssistantAnswer, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "text": answer.text,
                    "citations": list(answer.citations),
                    "confidence": answer.confidence,
                    "used_model": answer.used_model,
                    "conversation_id": answer.conversation_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(f"  - {citation}")
    print(f"\nConfidence: {answer.confidence:.2f}  "
          f"({'language model' if answer.used_model else 'extracted from sources'})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the grounded petroleum assistant.")
    parser.add_argument("question", nargs="*", help="the question (omit for interactive mode)")
    parser.add_argument("--no-model", action="store_true", help="answer extractively only")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    assistant = build_default_assistant(model=_load_model(args.checkpoint, enabled=not args.no_model))

    if args.question:
        question = " ".join(args.question)
        _print_answer(assistant.ask(question, limit=args.limit), as_json=args.json)
        return 0

    print("Fayd Al-Ruh assistant - type your question, or 'exit' to finish.")
    conversation = assistant.memory.start(title="Terminal session")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question in {"exit", "quit", ""}:
            return 0
        _print_answer(assistant.ask(question, conversation=conversation, limit=args.limit), as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
