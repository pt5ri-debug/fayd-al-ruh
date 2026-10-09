"""Talk with the project's AI, like Jarvis but text-first.

    python -m cli chat                          # interactive conversation
    python -m cli chat "what is porosity?"      # one question, then exit
    python -m cli chat --no-web                 # answer only from what it knows
    python -m cli chat --mind                   # answer from the trained model, not the sources
    python -m cli chat --session demo           # resume/autosave a named session
    python -m cli chat --list-sessions          # show saved conversations

This is the plain conversation entry point: it runs the same grounded loop as
Jarvis mode -- one session, follow-up questions keep context, every answer cites
its sources -- but stays in text. No microphone, no speaker, no wake word: it
uses the transcript recognizer and the silent synthesizer, so it behaves the same
on a machine with no audio at all.

The same spoken controls still work as typed commands: "new conversation" starts
over, "save" persists the session, "sessions" lists saved ones, "web on"/"web
off" toggle internet lookups, "that's right" and "that's wrong, it's ..." teach it
from explicit feedback, and "help" lists them. Type "quit" to leave.

With ``--mind`` the answer comes from the trained language model itself -- no
retrieval, no citations -- so the quality of what it has learned is visible. That
mouth is swapped in **here**, in this module, without touching the Jarvis loop or
the session builder, so Jarvis mode is unaffected.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ai import mind as mind_module
from ai.jarvis import build_session
from core.composition import sources_block
from core.jarvis import JarvisMode, JarvisTurn

INTRO = "Chat with the project's AI. Ask a question, or type 'help' for controls; 'quit' to exit."
MIND_HINT = (
    "AI: (the sources could not answer; type 'from the model' to answer from the "
    "trained model instead, or 'from the sources' to go back)"
)

# The composer's wording when retrieval found nothing to stand on.
_NO_SOURCE_MARKER = "not enough evidence"

# Strips punctuation so "From the model." and "model on" match the same phrase.
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize(text: str) -> str:
    cleaned = _NON_WORD.sub(" ", text.strip().casefold())
    return " ".join(cleaned.split())


def _match_switch(text: str) -> str | None:
    """Return "model" or "sources" when the line is a mouth switch, else ``None``.

    Recognised only for the whole line, so an ordinary question is never captured.
    The phrases live in :mod:`ai.mind`, not in the shared Jarvis controls, so the
    Jarvis command's wording never changes.
    """
    normalized = _normalize(text)
    if not normalized:
        return None
    if normalized in _normalize_set(mind_module.SWITCH_TO_MODEL):
        return "model"
    if normalized in _normalize_set(mind_module.SWITCH_TO_SOURCES):
        return "sources"
    return None


def _normalize_set(phrases: tuple[str, ...]) -> set[str]:
    return {_normalize(phrase) for phrase in phrases}


class ModelSwitcher:
    """Swaps the session's answering mouth between the model and the sources.

    The grounded assistant and its help text are kept so switching back is exact.
    Only ``mode.assistant``/``mode.voice.assistant``/``mode.help_text`` change;
    nothing in the Jarvis loop, the controls, or the session builder is touched.

    The model is loaded lazily on the first switch to it, so an interactive chat
    that never asks for the model never pays for loading one.
    """

    def __init__(self, mode: JarvisMode, *, loader, max_new_tokens: int = 96) -> None:
        self.mode = mode
        self.grounded = mode.assistant
        self.grounded_help = mode.help_text
        self._loader = loader
        self._max_new_tokens = max_new_tokens
        self._mind: mind_module.MindAssistant | None = None
        self.active = False

    @property
    def mind(self) -> mind_module.MindAssistant:
        if self._mind is None:
            self._mind = mind_module.MindAssistant(self._loader(), max_new_tokens=self._max_new_tokens)
        return self._mind

    def use_model(self) -> None:
        """Activate the model mouth. Raises when the checkpoint cannot be loaded."""
        assistant = self.mind  # load now, so a failure is reported before switching
        self.mode.assistant = assistant
        self.mode.voice.assistant = assistant
        self.mode.help_text = f"{self.grounded_help} {mind_module.MIND_HELP}"
        self.active = True

    def use_sources(self) -> None:
        self.mode.assistant = self.grounded
        self.mode.voice.assistant = self.grounded
        self.mode.help_text = self.grounded_help
        self.active = False


def format_turn(turn: JarvisTurn) -> str:
    """Render one turn as text: the answer, then its sources."""
    if turn.routed == "ignored":
        return "(no wake word; ignored)"
    if turn.routed == "silence":
        return "(no speech detected)"
    if turn.routed == "control":
        return f"[{turn.intent}] {turn.response}"
    if turn.agent_run is not None:
        lines = []
        for step in turn.agent_run.steps:
            status = "ok " if step.result.ok else "ERR"
            detail = step.result.output if step.result.ok else step.result.error
            lines.append(f"  [{status}] {step.call.tool}: {detail}")
        return "\n".join(lines) if lines else turn.response

    parts = [turn.response]
    block = sources_block(turn.citations)
    if block:
        parts.append(block)
    return "\n\n".join(parts)


def turn_payload(turn: JarvisTurn) -> dict:
    """A machine-readable record of one turn."""
    payload = {
        "heard": turn.heard,
        "routed": turn.routed,
        "intent": turn.intent,
        "response": turn.response,
        "citations": list(turn.citations),
        "used_web": turn.used_web,
        "learned": turn.learned,
        "conversation_id": turn.conversation_id,
    }
    if turn.agent_run is not None:
        payload["steps"] = [
            {
                "tool": step.call.tool,
                "ok": step.result.ok,
                "output": step.result.output,
                "error": step.result.error,
            }
            for step in turn.agent_run.steps
        ]
    return payload


def print_turn(turn: JarvisTurn, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(turn_payload(turn), ensure_ascii=False))
        return
    print(format_turn(turn))


def _print_switch(kind: str, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"routed": "switch", "mode": kind}, ensure_ascii=False))
        return
    if kind == "model":
        print("AI: [model] Now answering from the trained model only (no sources).")
    else:
        print("AI: [sources] Now answering from retrieved passages with citations.")
    print()


def run_loop(mode, *, as_json: bool, autosave: bool = False, switcher=None) -> int:
    """Read questions from the terminal and answer until 'quit' or EOF.

    With a ``switcher`` the typed controls "from the model" / "from the sources"
    change which mouth answers; without one the loop is exactly as before.
    """
    if not as_json:
        print(INTRO)
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().casefold() in {"quit", "exit", "q", ""}:
            return 0
        if switcher is not None:
            switch = _match_switch(line)
            if switch in {"model", "sources"}:
                try:
                    switcher.use_model() if switch == "model" else switcher.use_sources()
                except (FileNotFoundError, OSError, ValueError) as error:
                    print(f"Could not load the trained model: {error}")
                    print("Train it first with `python -m cli llm` (or point --model-checkpoint at one).")
                    print()
                    continue
                _print_switch(switch, as_json=as_json)
                continue
        turn = mode.handle(line, require_wake=False, speak=False)
        if not as_json:
            print("AI:", end=" ")
        print_turn(turn, as_json=as_json)
        if not as_json:
            print()
            if switcher is not None and not switcher.active and _NO_SOURCE_MARKER in turn.response:
                print(MIND_HINT)
                print()
        if autosave:
            mode.save_session()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chat",
        description="Talk with the project's AI in text (like Jarvis, without voice).",
    )
    parser.add_argument("message", nargs="*", help="one message; omit to enter the interactive loop")
    parser.add_argument("--root", default=None, help="sandbox root directory for tools")
    parser.add_argument("--read-write", action="store_true", help="allow writing tools in the sandbox")
    parser.add_argument("--no-web", action="store_true", help="answer only from internal knowledge; do not search the internet")
    parser.add_argument(
        "--web-threshold",
        type=float,
        default=None,
        help="top internal score below which it searches the web (0-1; higher searches more)",
    )
    parser.add_argument("--no-feedback", action="store_true", help="disable learning from confirm/correct controls")
    parser.add_argument("--session", default=None, help="resume a saved session by id, or start a new one")
    parser.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--mind",
        action="store_true",
        help="answer from the trained model itself (no retrieval, no citations), not from the sources",
    )
    parser.add_argument("--model-checkpoint", default=None, help="path to a language-model checkpoint to answer with")
    parser.add_argument("--mind-temperature", type=float, default=0.0, help="sampling temperature for model answers (0 = deterministic)")
    parser.add_argument("--mind-tokens", type=int, default=96, help="maximum tokens to generate per model answer")
    args = parser.parse_args(argv)

    if args.list_sessions:
        mode = build_session(no_web=args.no_web, no_feedback=args.no_feedback, text_only=True)
        saved = mode.list_sessions()
        if saved:
            print("Saved sessions:")
            for session_id in saved:
                print(f"  - {session_id}")
        else:
            print("No saved sessions yet.")
        return 0

    mode = build_session(
        root=Path(args.root) if args.root else None,
        read_write=args.read_write,
        no_web=args.no_web,
        web_threshold=args.web_threshold,
        no_feedback=args.no_feedback,
        text_only=True,
    )

    switcher = ModelSwitcher(
        mode,
        loader=lambda: mind_module.load_mind_model(
            args.model_checkpoint,
            temperature=args.mind_temperature,
            max_new_tokens=args.mind_tokens,
        ),
        max_new_tokens=args.mind_tokens,
    )
    if args.mind:
        try:
            switcher.use_model()
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Could not load the trained model: {error}")
            print("Train it first with `python -m cli llm` (or point --model-checkpoint at one).")
            return 1
        if not args.json:
            print(mind_module.status_line(mind_module.mind_status(args.model_checkpoint)))

    if args.session:
        if not mode.resume_session(args.session):
            if not args.json:
                print(f"No saved session '{args.session}'. Starting a new one.")
        elif not args.json:
            print(f"Resumed session {mode.conversation.id}.")

    if args.message:
        turn = mode.handle(" ".join(args.message), require_wake=False, speak=False)
        print_turn(turn, as_json=args.json)
        if args.session:
            mode.save_session()
        return 0

    return run_loop(mode, as_json=args.json, autosave=args.session is not None, switcher=switcher)


__all__ = [
    "INTRO",
    "ModelSwitcher",
    "format_turn",
    "main",
    "print_turn",
    "run_loop",
    "turn_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
