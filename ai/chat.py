"""Talk with the project's AI, like Jarvis but text-first.

    python -m cli chat                          # interactive conversation
    python -m cli chat "what is porosity?"      # one question, then exit
    python -m cli chat --no-web                 # answer only from what it knows
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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.jarvis import build_session
from core.composition import sources_block
from core.jarvis import JarvisTurn

INTRO = "Chat with the project's AI. Ask a question, or type 'help' for controls; 'quit' to exit."


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


def run_loop(mode, *, as_json: bool, autosave: bool = False) -> int:
    """Read questions from the terminal and answer until 'quit' or EOF."""
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
        turn = mode.handle(line, require_wake=False, speak=False)
        if not as_json:
            print("AI:", end=" ")
        print_turn(turn, as_json=as_json)
        if not as_json:
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

    return run_loop(mode, as_json=args.json, autosave=args.session is not None)


__all__ = [
    "INTRO",
    "format_turn",
    "main",
    "print_turn",
    "run_loop",
    "turn_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
