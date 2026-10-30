"""Hakeem from the command line: talk with the scholar in a lasting dialogue.

    python -m cli hakeem                             # interactive chat (type 'quit' to exit)
    python -m cli hakeem "قارن بين الأديان الثلاثة"     # one message
    python -m cli hakeem --session first-chat "..."   # keep one named session across runs
    python -m cli hakeem --list-sessions             # the sessions saved so far
    python -m cli hakeem --new                       # start a fresh session
    python -m cli hakeem status                      # which runtime answers, and the next step
    python -m cli hakeem models                      # list the models he can fetch
    python -m cli hakeem download --approve           # one-time: fetch his own GGUF (~2.5 GB)
    python -m cli hakeem download --approve --backend ollama   # or pull an Ollama tag
    python -m cli hakeem remember "أفضل كتب التاريخ" --kind preference
    python -m cli hakeem memories                    # what he remembers
    python -m cli hakeem memories --correct MEM-... "نص مصحّح"
    python -m cli hakeem memories --forget MEM-...
    python -m cli hakeem settings --depth 0.9 --citation 0.9 --comparison 0.8
    python -m cli hakeem emotion                     # the current dialogue tone
    python -m cli hakeem reset-emotion               # back to a balanced tone
    python -m cli hakeem export                      # everything, as JSON (for a backup)
    python -m cli hakeem backup                      # a bounded, timestamped DB copy

Hakeem (Arabic ``حكيم``) is the project's **scholar of religion and history**: he
discusses the Abrahamic faiths -- Islam, Christianity, and Judaism -- compares
them fairly, and is at home in the ancient civilisations. He has his **own** local
model (``qwen3.5-4b``) in his **own** folder (``data/hakeem``), separate from the
companion Natasha's; a model in the project's shared folder is used too, so a
shared download works for him as well.

He is a scholar, not a preacher: he respects every faith, never proselytises,
never issues a religious ruling, keeps scripture and history and scholarly opinion
clearly apart, and never invents a verse, a hadith, or a historical fact. Every
one of those rules is in :data:`hakeem.personality.BOUNDARIES` and is stated to the
model on every request. He is honest that he is a virtual character, not a real
scholar.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from hakeem.config import HakeemConfig
from hakeem.hakeem_agent import HakeemAgent, HakeemError
from hakeem.memory_manager import KINDS
from hakeem.model import (
    BACKEND_NAMES,
    DEFAULT_SPEC_KEY,
    HakeemModelUnavailable,
    download_model,
    pull_model,
    spec_summaries,
)
from hakeem.personality import NAME

INTRO = (
    f"{NAME} (حكيم): a scholar of the Abrahamic religions and of history, speaking "
    "through his own local model. Type 'quit' to exit."
)

#: Recognised subcommands. Anything else is read as a chat message.
_COMMANDS = (
    "chat", "status", "models", "download", "sessions", "remember", "memories",
    "settings", "emotion", "reset-emotion", "export", "backup",
)


def _agent(args: argparse.Namespace) -> HakeemAgent:
    overrides: dict[str, object] = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "backend", None):
        overrides["backend"] = args.backend
    if getattr(args, "model_file", None):
        overrides["model_file"] = args.model_file
    if getattr(args, "base_url", None):
        overrides["base_url"] = args.base_url
    if getattr(args, "no_memory", False):
        overrides["persistent"] = False
    # The speed knobs, shared with the companion: ``think`` is tri-state.
    if getattr(args, "think", None) is not None:
        overrides["think"] = args.think
    if getattr(args, "max_tokens", None) is not None:
        overrides["max_tokens"] = args.max_tokens
    if getattr(args, "context", None) is not None:
        overrides["context"] = args.context
    if getattr(args, "threads", None) is not None:
        overrides["threads"] = args.threads
    if getattr(args, "context_turns", None) is not None:
        overrides["history_window"] = args.context_turns
    config = HakeemConfig.from_env(**overrides)
    return HakeemAgent(config)


def status_line(agent: HakeemAgent) -> str:
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
                    "backend": status.backend,
                    "available": status.available,
                    "server_reachable": status.server_reachable,
                    "installed": status.installed,
                    "detail": status.detail,
                    "hint": status.hint,
                    "status": status.summary(),
                    "config": agent.config.describe(),
                },
                ensure_ascii=False,
            )
        )
        return 0 if status.available else 1
    print(status.summary())
    print(f"  model  : {agent.config.model}")
    print(f"  backend: {agent.config.backend}")
    if status.backend == "ollama":
        print(f"  server : {agent.config.base_url}")
    print(f"  storage: {agent.config.describe().split('; storage: ')[-1]}")
    print(
        f"  speed  : hidden reasoning {'on (slower)' if agent.config.think else 'off (fast)'}"
        f"; max {agent.config.max_tokens} tokens, {agent.config.context}-token window,"
        f" {agent.config.history_window} turns of context"
    )
    if not status.available:
        print(f"  next step: {status.hint}")
    return 0 if status.available else 1


def run_models(args: argparse.Namespace) -> int:
    """List the models Hakeem can fetch (his own catalog, in his own folder)."""
    if getattr(args, "as_json", False):
        print(
            json.dumps(
                [
                    {"key": key, "title": title, "size_gb": size, "summary": summary}
                    for key, title, size, summary in spec_summaries()
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Models Hakeem can fetch (default: {DEFAULT_SPEC_KEY}):")
    print("(Fetched into his own folder, data/hakeem; a shared model is used too.)")
    for key, title, size, summary in spec_summaries():
        print(f"  {key:14s} {title:34s} ~{size:.1f} GB")
        print(f"  {'':14s} {summary}")
    print("\nFetch one with: python -m cli hakeem download --approve --model <key>")
    print("(Or, into a local Ollama server: ... download --approve --model <key> --backend ollama)")
    return 0


def run_download(args: argparse.Namespace) -> int:
    """Fetch Hakeem's model through the permission + human-approval gates.

    The default is the in-process GGUF weights (``llama-cpp-python``); pass
    ``--backend ollama`` to pull the same model into a local Ollama server instead.
    """
    if not args.approve:
        print(
            "error: fetching a model is an external action and needs --approve "
            "(it fetches several GB from the internet)."
        )
        return 2
    backend = (getattr(args, "backend", None) or "gguf").strip().lower()
    try:
        if backend == "ollama":
            tag = pull_model(
                args.model or DEFAULT_SPEC_KEY,
                approved=True,
                progress=lambda message: print(message, file=sys.stderr),
            )
            result: str = tag
            message = f"Hakeem's model ready in Ollama: {tag}."
        else:
            path = download_model(
                args.model or DEFAULT_SPEC_KEY,
                approved=True,
                progress=lambda message: print(message, file=sys.stderr),
            )
            result = str(path)
            message = f"Hakeem's model ready: {path}."
    except (HakeemModelUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    if getattr(args, "as_json", False):
        print(json.dumps({"model": result, "backend": backend}, ensure_ascii=False))
        return 0
    print(message)
    if backend != "ollama":
        print("He will use it automatically (backend: auto). No server needed.")
    return 0


def run_sessions(args: argparse.Namespace) -> int:
    agent = _agent(args)
    sessions = agent.list_sessions()
    if getattr(args, "as_json", False):
        print(json.dumps([s.__dict__ for s in sessions], ensure_ascii=False, indent=2))
        return 0
    if not sessions:
        print("No sessions yet. Start one with: python -m cli hakeem")
        return 0
    print(f"{len(sessions)} session(s), most recent first:")
    for session in sessions:
        print(f"  {session.id}  {session.message_count:4d} msg  {session.last_active[:19]}  {session.title}")
    print("\nResume one with: python -m cli hakeem --session <id> \"your message\"")
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
        print("Hakeem remembers nothing yet.")
        return 0
    print(f"{len(memories)} memory(ies):")
    for memory in memories:
        flag = " [fiction]" if memory.fictional else ""
        state = "" if memory.active else " (superseded)"
        print(f"  {memory.id}  [{memory.kind}/{memory.confidence:.2f}]{flag}{state}  {memory.content}")
    print("\nCorrect one: python -m cli hakeem memories --correct <id> \"new text\"")
    print("Delete one : python -m cli hakeem memories --forget <id>")
    return 0


def run_settings(args: argparse.Namespace) -> int:
    agent = _agent(args)
    changes = {
        "language": args.language,
        "reply_language": args.reply_language,
        "dialect": args.dialect,
        "formality": args.formality,
        "depth": args.depth,
        "citation": args.citation,
        "comparison": args.comparison,
        "history": args.history,
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
    print("(A narrative tone for how he speaks, not a measurement of your feelings.)")
    return 0


def run_reset_emotion(args: argparse.Namespace) -> int:
    agent = _agent(args)
    emotion = agent.reset_emotion()
    print(f"Tone reset: {emotion.describe()}")
    return 0


def run_export(args: argparse.Namespace) -> int:
    agent = _agent(args)
    data = agent.export_data()
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Exported Hakeem's data to {args.output}")
    else:
        print(text)
    return 0


def run_backup(args: argparse.Namespace) -> int:
    agent = _agent(args)
    path = agent.backup()
    if path is None:
        print("Nothing to back up (Hakeem is running without a database file).")
        return 0
    print(f"Backed up to {path}")
    print(f"Keeping the newest {agent.config.backup_keep} backup(s).")
    return 0


def _print_exchange(reply, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(reply.to_primitive(), ensure_ascii=False))
        return
    print(f"{NAME}: {reply.text}")


def _one_shot(agent: HakeemAgent, session_id: str, message: str, *, as_json: bool) -> int:
    try:
        reply = agent.ask(session_id, message)
    except HakeemError as error:
        print(f"{NAME} could not answer: {error}", file=sys.stderr)
        return 3
    except (KeyError, ValueError) as error:
        print(f"Bad request: {error}", file=sys.stderr)
        return 2
    _print_exchange(reply, as_json=as_json)
    return 0


def _interactive(agent: HakeemAgent, session_id: str, *, as_json: bool) -> int:
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
        except HakeemError as error:
            print(f"{NAME} could not answer: {error}", file=sys.stderr)
            continue
        except (KeyError, ValueError) as error:
            print(f"Bad request: {error}", file=sys.stderr)
            continue
        _print_exchange(reply, as_json=as_json)


def _resolve_session(agent: HakeemAgent, args: argparse.Namespace) -> str:
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
    parser.add_argument(
        "--model", default=None, help="the local model to use (default qwen3.5:4b)."
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_NAMES,
        default=None,
        help="which runtime answers: auto (default), gguf (in-process), or ollama.",
    )
    parser.add_argument(
        "--model-file", dest="model_file", default=None,
        help="a specific .gguf file to load (the gguf runtime).",
    )
    parser.add_argument("--base-url", default=None, help="the Ollama server URL.")
    parser.add_argument(
        "--think", dest="think", action="store_true", default=None,
        help="allow the model's hidden reasoning trace (slower, deeper answers).",
    )
    parser.add_argument(
        "--no-think", dest="think", action="store_false", default=None,
        help="answer without the hidden reasoning trace (the default; faster).",
    )
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None,
        help="cap the reply length in tokens (default 512).",
    )
    parser.add_argument(
        "--context", type=int, default=None,
        help="context window in tokens (default 4096).",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="CPU threads for the in-process model (default: let llama.cpp choose).",
    )
    parser.add_argument(
        # Its own dest on purpose: the ``settings`` subcommand already uses
        # ``--history`` for Hakeem's *style* dial, and the two must not collide.
        "--context-turns", dest="context_turns", type=int, default=None,
        help="how many recent turns to send as context (default 8).",
    )
    parser.add_argument(
        "--no-memory", dest="no_memory", action="store_true",
        help="run without a lasting database (nothing is kept after the run).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hakeem", description=f"{NAME}: talk with the scholar.")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Talk with Hakeem (the default).")
    chat.add_argument("message", nargs="*", help="one message; omit for the interactive loop")
    _add_connection_arguments(chat)
    chat.add_argument("--session", default=None, help="resume a session by id, or start a new one")
    chat.add_argument("--new", dest="new_session", action="store_true", help="start a fresh session")
    chat.add_argument("--list-sessions", action="store_true", help="list sessions and exit")
    chat.add_argument("--json", dest="as_json", action="store_true", help="emit machine-readable JSON")

    status = sub.add_parser("status", help="Which runtime answers, and the next step.")
    _add_connection_arguments(status)
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    models = sub.add_parser("models", help="List the models Hakeem can fetch.")
    models.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    download = sub.add_parser(
        "download", help="Fetch a model: GGUF weights by default, or an Ollama tag."
    )
    download.add_argument("--approve", action="store_true", help="grant the external fetch.")
    download.add_argument(
        "--model", default=None, help=f"which model to fetch (default {DEFAULT_SPEC_KEY})."
    )
    download.add_argument(
        "--backend", choices=("gguf", "ollama"), default=None,
        help="gguf fetches the in-process weights (default); ollama pulls the tag.",
    )
    download.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

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

    settings = sub.add_parser("settings", help="Show or change the scholarly style.")
    _add_connection_arguments(settings)
    settings.add_argument("--language", choices=("ar", "en"), default=None,
                          help="the fallback language when a message carries no letters.")
    settings.add_argument(
        "--reply-language", dest="reply_language", choices=("match", "ar", "en"),
        default=None,
        help="match (default) answers in the user's own language; ar/en pin it.",
    )
    settings.add_argument("--dialect", choices=("msa", "egyptian"), default=None)
    settings.add_argument(
        "--formality", type=float, default=None,
        help="how formal the register is (0 relaxed .. 1 formal scholarly prose).",
    )
    settings.add_argument(
        "--depth", type=float, default=None,
        help="how thorough the answer is (0 brief .. 1 a full, carefully argued treatment).",
    )
    settings.add_argument(
        "--citation", type=float, default=None,
        help="how rigorously he names his sources (0 light .. 1 always names the text).",
    )
    settings.add_argument(
        "--comparison", type=float, default=None,
        help="how far he compares traditions (0 within one .. 1 actively compares).",
    )
    settings.add_argument(
        "--history", type=float, default=None,
        help="how much historical context he gives (0 minimal .. 1 full context).",
    )

    emotion = sub.add_parser("emotion", help="Show the current dialogue tone.")
    _add_connection_arguments(emotion)
    emotion.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    reset = sub.add_parser("reset-emotion", help="Reset the dialogue tone to balanced.")
    _add_connection_arguments(reset)

    export = sub.add_parser("export", help="Export all of Hakeem's data as JSON.")
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
        "models": run_models,
        "download": run_download,
        "sessions": run_sessions,
        "remember": run_remember,
        "memories": run_memories,
        "settings": run_settings,
        "emotion": run_emotion,
        "reset-emotion": run_reset_emotion,
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
