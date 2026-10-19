"""Natasha from the command line: talk with the companion in a lasting dialogue.

    python -m cli natasha                          # interactive chat (type 'quit' to exit)
    python -m cli natasha "مرحبا يا ناتاشا"          # one message
    python -m cli natasha --session first-chat "..."  # keep one named session across runs
    python -m cli natasha --list-sessions          # the sessions saved so far
    python -m cli natasha --new                    # start a fresh session
    python -m cli natasha status                   # is the local model reachable
    python -m cli natasha remember "أحب القهوة" --kind preference
    python -m cli natasha memories                 # what she remembers
    python -m cli natasha memories --correct MEM-... "نص مصحّح"
    python -m cli natasha memories --forget MEM-...
    python -m cli natasha settings --warmth 0.8 --romance 0.3
    python -m cli natasha emotion                  # the current dialogue tone
    python -m cli natasha reset-emotion            # back to a balanced tone
    python -m cli natasha relationship             # the story so far (no love score)
    python -m cli natasha export                   # everything, as JSON (for a backup)
    python -m cli natasha backup                   # a bounded, timestamped DB copy

Natasha is a *companion*, not the cognitive core: she keeps her own database and
never writes to the project's entities, relations, evidence, beliefs, or knowledge
graph. Everything stays on this machine -- the only call is to the local Ollama
server. With no Ollama running the companion still runs; every command reports the
real next step instead of pretending.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from companion.companion_agent import CompanionAgent, CompanionError
from companion.config import CompanionConfig
from companion.personality import NAME
from companion.memory_manager import KINDS

INTRO = (
    f"{NAME}: a warm, curious companion speaking through a local Qwen3 model. "
    "Type 'quit' to exit."
)

#: Recognised subcommands. Anything else is read as a chat message.
_COMMANDS = (
    "chat", "status", "sessions", "remember", "memories", "settings",
    "emotion", "reset-emotion", "relationship", "export", "backup",
)


def _agent(args: argparse.Namespace) -> CompanionAgent:
    overrides: dict[str, object] = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "base_url", None):
        overrides["base_url"] = args.base_url
    if getattr(args, "no_memory", False):
        overrides["persistent"] = False
    config = CompanionConfig.from_env(**overrides)
    return CompanionAgent(config)


def status_line(agent: CompanionAgent) -> str:
    """An honest one-line status, reused by the CLI, the menu, and the window."""
    return agent.model_status().summary()


# -- commands -------------------------------------------------------------
def run_status(args: argparse.Namespace) -> int:
    agent = _agent(args)
    status = agent.model_status()
    if getattr(args, "as_json", False):
        print(
            json.dumps(
                {
                    "model": status.model,
                    "available": status.available,
                    "server_reachable": status.server_reachable,
                    "installed": status.installed,
                    "detail": status.detail,
                    "status": status.summary(),
                    "config": agent.config.describe(),
                },
                ensure_ascii=False,
            )
        )
        return 0 if status.available else 1
    print(status.summary())
    print(f"  model : {agent.config.model}")
    print(f"  server: {agent.config.base_url}")
    print(f"  storage: {agent.config.describe().split('; storage: ')[-1]}")
    if not status.available and status.server_reachable:
        print(f"  next step: ollama pull {agent.config.model}")
    elif not status.server_reachable:
        print("  next step: start Ollama with `ollama serve`")
    return 0 if status.available else 1


def run_sessions(args: argparse.Namespace) -> int:
    agent = _agent(args)
    sessions = agent.list_sessions()
    if getattr(args, "as_json", False):
        print(json.dumps([s.__dict__ for s in sessions], ensure_ascii=False, indent=2))
        return 0
    if not sessions:
        print("No sessions yet. Start one with: python -m cli natasha")
        return 0
    print(f"{len(sessions)} session(s), most recent first:")
    for session in sessions:
        print(f"  {session.id}  {session.message_count:4d} msg  {session.last_active[:19]}  {session.title}")
    print("\nResume one with: python -m cli natasha --session <id> \"your message\"")
    return 0


def run_remember(args: argparse.Namespace) -> int:
    agent = _agent(args)
    text = " ".join(args.text).strip()
    if not text:
        print("Nothing to remember: give the text to store.", file=sys.stderr)
        return 2
    try:
        memory = agent.remember(
            text,
            kind=args.kind,
            source=args.source,
            fictional=args.fictional,
            allow_sensitive=args.allow_sensitive,
        )
    except ValueError as error:
        print(f"Not saved: {error}", file=sys.stderr)
        return 2
    print(f"Remembered [{memory.kind}] {memory.id}: {memory.content}")
    return 0


def run_memories(args: argparse.Namespace) -> int:
    agent = _agent(args)
    if args.forget:
        removed = agent.forget(args.forget)
        print(f"Forgot {args.forget}." if removed else f"No memory with id {args.forget}.")
        return 0 if removed else 4
    if args.correct:
        memory_id, _, new_text = args.correct.partition(" ")
        if not new_text.strip():
            print("A correction needs the memory id and the new text.", file=sys.stderr)
            return 2
        try:
            memory = agent.memories.correct(memory_id, new_text.strip(), allow_sensitive=args.allow_sensitive)
        except (KeyError, ValueError) as error:
            print(f"Not corrected: {error}", file=sys.stderr)
            return 2
        print(f"Corrected {memory_id} -> {memory.id}: {memory.content}")
        return 0
    memories = agent.list_memories(include_inactive=args.all)
    if getattr(args, "as_json", False):
        print(json.dumps([m.to_primitive() for m in memories], ensure_ascii=False, indent=2))
        return 0
    if not memories:
        print("Natasha remembers nothing yet.")
        return 0
    print(f"{len(memories)} memory(ies):")
    for memory in memories:
        flag = " [fiction]" if memory.fictional else ""
        state = "" if memory.active else " (superseded)"
        print(f"  {memory.id}  [{memory.kind}/{memory.confidence:.2f}]{flag}{state}  {memory.content}")
    print("\nCorrect one: python -m cli natasha memories --correct <id> \"new text\"")
    print("Delete one : python -m cli natasha memories --forget <id>")
    return 0


def run_settings(args: argparse.Namespace) -> int:
    agent = _agent(args)
    changes = {
        "language": args.language,
        "dialect": args.dialect,
        "humor": args.humor,
        "warmth": args.warmth,
        "romance": args.romance,
        "calm": args.calm,
        "depth": args.depth,
    }
    changes = {k: v for k, v in changes.items() if v is not None}
    if changes:
        try:
            settings = agent.update_settings(**changes)
        except ValueError as error:
            print(f"Not changed: {error}", file=sys.stderr)
            return 2
        print(f"Updated: {settings.describe()}")
    else:
        print(agent.settings().describe())
    return 0


def run_emotion(args: argparse.Namespace) -> int:
    agent = _agent(args)
    emotion = agent.emotion
    if getattr(args, "as_json", False):
        print(json.dumps(emotion.to_primitive(), ensure_ascii=False))
        return 0
    print(f"Dialogue tone: {emotion.describe()}")
    print("(A narrative tone for how she speaks, not a measurement of your feelings.)")
    return 0


def run_reset_emotion(args: argparse.Namespace) -> int:
    agent = _agent(args)
    emotion = agent.reset_emotion()
    print(f"Tone reset: {emotion.describe()}")
    return 0


def run_relationship(args: argparse.Namespace) -> int:
    agent = _agent(args)
    summary = agent.relationship_summary()
    events = agent.relationship_events(include_fiction=not args.no_fiction)
    if getattr(args, "as_json", False):
        print(
            json.dumps(
                {"summary": summary.describe(), "events": [e.to_primitive() for e in events]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(summary.describe())
    if events:
        print("\nSaved moments:")
        for event in events:
            print(f"  {event.id}  [{event.kind}]  {event.occurred_at[:10]}  {event.title}")
    print("\n(There is no love or loyalty score: the story is described in plain words.)")
    return 0


def run_export(args: argparse.Namespace) -> int:
    agent = _agent(args)
    data = agent.export_data()
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Exported Natasha's data to {args.output}")
    else:
        print(text)
    return 0


def run_backup(args: argparse.Namespace) -> int:
    agent = _agent(args)
    path = agent.backup()
    if path is None:
        print("Nothing to back up (the companion is running without a database file).")
        return 0
    print(f"Backed up to {path}")
    print(f"Keeping the newest {agent.config.backup_keep} backup(s).")
    return 0


def _print_exchange(reply, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(reply.to_primitive(), ensure_ascii=False))
        return
    print(f"{NAME}: {reply.text}")


def _one_shot(agent: CompanionAgent, session_id: str, message: str, *, as_json: bool) -> int:
    try:
        reply = agent.ask(session_id, message)
    except CompanionError as error:
        print(f"{NAME} could not answer: {error}", file=sys.stderr)
        return 3
    except (KeyError, ValueError) as error:
        print(f"Bad request: {error}", file=sys.stderr)
        return 2
    _print_exchange(reply, as_json=as_json)
    return 0


def _interactive(agent: CompanionAgent, session_id: str, *, as_json: bool) -> int:
    if not as_json:
        print(INTRO)
        print(status_line(agent))
        print()
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        command = line.strip().casefold()
        if command in {"quit", "exit", "q"}:
            return 0
        if not line.strip():
            continue
        try:
            reply = agent.ask(session_id, line)
        except CompanionError as error:
            print(f"{NAME} could not answer: {error}", file=sys.stderr)
            continue
        except (KeyError, ValueError) as error:
            print(f"Bad request: {error}", file=sys.stderr)
            continue
        _print_exchange(reply, as_json=as_json)


def _resolve_session(agent: CompanionAgent, args: argparse.Namespace) -> str:
    """Pick the session for a chat turn: an explicit one, a resumed one, or new."""
    if args.session:
        existing = agent.conversations.get_session(args.session)
        if existing is not None:
            agent.resume_session(args.session)
            return args.session
        title = args.message[0] if args.message else args.session
        agent.new_session(title[:60], session_id=args.session)
        return args.session
    title = " ".join(args.message)[:60] if args.message else "Conversation"
    return agent.new_session(title).id


def run_chat(args: argparse.Namespace) -> int:
    agent = _agent(args)
    if getattr(args, "list_sessions", False):
        return run_sessions(args)
    if getattr(args, "new_session", False):
        title = " ".join(args.message)[:60] if args.message else "Conversation"
        session_id = agent.new_session(title).id
        if args.message:
            return _one_shot(agent, session_id, " ".join(args.message), as_json=args.as_json)
        return _interactive(agent, session_id, as_json=args.as_json)
    session_id = _resolve_session(agent, args)
    if args.message:
        return _one_shot(agent, session_id, " ".join(args.message), as_json=args.as_json)
    return _interactive(agent, session_id, as_json=args.as_json)


# -- argument parsing -----------------------------------------------------
def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="the local model to use (default qwen3:4b).")
    parser.add_argument("--base-url", default=None, help="the Ollama server URL.")
    parser.add_argument(
        "--no-memory", dest="no_memory", action="store_true",
        help="run without a lasting database (nothing is kept after the run).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="natasha", description=f"{NAME}: talk with the companion.")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Talk with Natasha (the default).")
    chat.add_argument("message", nargs="*", help="one message; omit for the interactive loop")
    _add_connection_arguments(chat)
    chat.add_argument("--session", default=None, help="resume a session by id, or start a new one")
    chat.add_argument("--new", dest="new_session", action="store_true", help="start a fresh session")
    chat.add_argument("--list-sessions", action="store_true", help="list sessions and exit")
    chat.add_argument("--json", dest="as_json", action="store_true", help="emit machine-readable JSON")

    status = sub.add_parser("status", help="Show whether the local model can answer.")
    _add_connection_arguments(status)
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    sessions = sub.add_parser("sessions", help="List saved sessions.")
    _add_connection_arguments(sessions)
    sessions.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    remember = sub.add_parser("remember", help="Store a memory explicitly.")
    remember.add_argument("text", nargs="*", help="the text to remember")
    _add_connection_arguments(remember)
    remember.add_argument("--kind", choices=KINDS, default="fact", help="the memory kind")
    remember.add_argument("--source", default="user", help="where this came from")
    remember.add_argument("--fictional", action="store_true", help="mark it as a story detail")
    remember.add_argument(
        "--allow-sensitive", dest="allow_sensitive", action="store_true",
        help="allow storing something that looks like a secret or an ID (off by default).",
    )

    memories = sub.add_parser("memories", help="Show, correct, or delete memories.")
    _add_connection_arguments(memories)
    memories.add_argument("--correct", default=None, help='correct a memory: "<id> <new text>"')
    memories.add_argument("--forget", default=None, help="delete a memory by id")
    memories.add_argument("--all", action="store_true", help="include superseded memories")
    memories.add_argument(
        "--allow-sensitive", dest="allow_sensitive", action="store_true",
        help="allow a correction that looks like a secret (off by default).",
    )
    memories.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    settings = sub.add_parser("settings", help="Show or change the dialogue style.")
    _add_connection_arguments(settings)
    settings.add_argument("--language", choices=("ar", "en"), default=None)
    settings.add_argument("--dialect", choices=("msa", "egyptian"), default=None)
    settings.add_argument("--humor", type=float, default=None)
    settings.add_argument("--warmth", type=float, default=None)
    settings.add_argument("--romance", type=float, default=None)
    settings.add_argument("--calm", type=float, default=None)
    settings.add_argument("--depth", type=float, default=None)

    emotion = sub.add_parser("emotion", help="Show the current dialogue tone.")
    _add_connection_arguments(emotion)
    emotion.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    reset = sub.add_parser("reset-emotion", help="Reset the dialogue tone to balanced.")
    _add_connection_arguments(reset)

    relationship = sub.add_parser("relationship", help="Show the relationship story so far.")
    _add_connection_arguments(relationship)
    relationship.add_argument("--no-fiction", dest="no_fiction", action="store_true")
    relationship.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    export = sub.add_parser("export", help="Export all of Natasha's data as JSON.")
    _add_connection_arguments(export)
    export.add_argument("--output", default=None, help="write to this file instead of stdout")

    backup = sub.add_parser("backup", help="Write a bounded, timestamped database copy.")
    _add_connection_arguments(backup)
    return parser


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Read a bare message as ``chat``; keep a real subcommand as-is."""
    args = list(argv)
    if args and args[0] in {"-h", "--help", "help"}:
        return args
    if args and args[0] in _COMMANDS:
        return args
    return ["chat", *args]


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(raw))

    if getattr(args, "list_sessions", False):
        return run_sessions(args)

    handlers = {
        "status": run_status,
        "sessions": run_sessions,
        "remember": run_remember,
        "memories": run_memories,
        "settings": run_settings,
        "emotion": run_emotion,
        "reset-emotion": run_reset_emotion,
        "relationship": run_relationship,
        "export": run_export,
        "backup": run_backup,
    }
    handler = handlers.get(args.command)
    if handler is not None:
        return handler(args)
    return run_chat(args)


__all__ = ["INTRO", "main", "status_line"]


if __name__ == "__main__":
    raise SystemExit(main())
