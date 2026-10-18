"""Cloud from the command line: talk to a remote, OpenAI-compatible model.

    python -m cli cloud                          # interactive chat (type 'quit' to exit)
    python -m cli cloud "explain how a jet engine works"   # one question
    python -m cli cloud --provider groq "..."   # choose the provider
    python -m cli cloud --session demo "and who invented it?"  # keep context across runs
    python -m cli cloud providers                # list the providers it can reach
    python -m cli cloud status                   # is a key set and the endpoint reachable
    python -m cli cloud status --probe           # also ask the endpoint (a real request)

Cloud is the project's *remote* mouth. Unlike the grounded assistant (which
answers only from the petroleum corpus, with citations) and unlike the local
models (Ultron, Vision, Banner, and ``--mind``), Cloud talks to a **remote**
OpenAI-compatible endpoint -- OpenAI's own API (the ChatGPT models), Anthropic,
Gemini, Groq, OpenRouter, Mistral, DeepSeek, or an OpenAI-compatible server on
this machine (Ollama, LM Studio, vLLM).

Because a cloud call sends the user's words off the machine, it is a governed
external action: it is denied by default, and running this command (or selecting
the Cloud mouth in the chat) is the user's explicit approval. The **API key is
read from an environment variable** -- for example ``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ``GROQ_API_KEY`` -- and is never
written to the repository, a config file, or a log. Nothing needs installing:
the client is plain ``urllib`` from the standard library, so a base install can
reach a cloud model the moment a key is set.

Sessions persist under the data directory, so a conversation keeps its context
across runs. Every answer is labelled as coming from the remote model and is
*not* grounded in cited sources -- that honesty is deliberate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.cloud_model import (
    BLURB,
    DEFAULT_PROVIDER_KEY,
    HEADER,
    NAME,
    CloudEngine,
    CloudUnavailable,
    cloud_status,
    provider_summaries,
)
from core.conversation import Conversation, ConversationMemory, Role
from core.paths import data_dir
from storage.json_store import JsonStore

INTRO = (
    f"{NAME}: a remote model over the network. Ask about anything; type 'quit' to exit."
)


def _sessions_dir() -> Path:
    return data_dir() / "cloud" / "sessions"


def _memory() -> ConversationMemory:
    """A persistent session store, or in-memory if the data folder is not writable."""
    try:
        return ConversationMemory(JsonStore(_sessions_dir()))
    except (OSError, ValueError):
        return ConversationMemory()


def _engine(args: argparse.Namespace) -> CloudEngine:
    return CloudEngine(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=getattr(args, "temperature", None),
        max_tokens=getattr(args, "max_tokens", None),
        # Running this command *is* the user's approval to send their words to
        # the provider; the permission/approval/audit gates are still applied.
        approved=True,
    )


def status_line(engine: CloudEngine | None = None, *, probe: bool = False) -> str:
    """A one-line, honest status: is a key set, and (optionally) is the endpoint up."""
    engine = engine or CloudEngine(approved=True)
    status = cloud_status(engine.client, probe=probe)
    if status.available:
        return f"{NAME} ready. {engine.describe()}"
    return status.summary() + (" " + status.hint if status.hint else "")


def _history(conversation: Conversation) -> list[tuple[str, str]]:
    """The prior turns as ``(role, text)`` pairs, newest last, for the model."""
    pairs: list[tuple[str, str]] = []
    for message in conversation.messages:
        if message.role is Role.USER:
            pairs.append(("user", message.text))
        elif message.role is Role.ASSISTANT:
            pairs.append(("assistant", message.text))
    return pairs


def _emit(text: str, *, as_json: bool, prompt: str = "") -> None:
    if as_json:
        print(json.dumps({"prompt": prompt, "response": text, "model": NAME}, ensure_ascii=False))
        return
    print(f"{NAME}: {text}")


def _one_shot(engine: CloudEngine, prompt: str, conversation: Conversation, *, as_json: bool) -> int:
    try:
        answer = engine.reply(prompt, history=_history(conversation))
    except CloudUnavailable as error:
        print(f"{NAME} is not available: {error}")
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        print(f"{NAME} could not answer: {error}")
        return 1
    conversation.add_user(prompt)
    conversation.add_assistant(answer)
    if not as_json:
        print(HEADER)
    _emit(answer, as_json=as_json, prompt=prompt)
    return 0


def _interactive(
    engine: CloudEngine,
    conversation: Conversation,
    memory: ConversationMemory,
    *,
    stream: bool,
    as_json: bool,
    autosave: bool,
) -> int:
    if not as_json:
        print(INTRO)
        print(status_line(engine))
        print(HEADER)
        print()
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().casefold() in {"quit", "exit", "q", ""}:
            return 0
        history = _history(conversation)
        try:
            if stream and not as_json:
                print(f"{NAME}: ", end="", flush=True)
                pieces: list[str] = []
                for piece in engine.stream_reply(line, history=history):
                    pieces.append(piece)
                    print(piece, end="", flush=True)
                answer = "".join(pieces).strip()
                print("\n")
            else:
                answer = engine.reply(line, history=history)
                _emit(answer, as_json=as_json, prompt=line)
        except CloudUnavailable as error:
            print(f"{NAME} is not available: {error}")
            return 1
        except (ValueError, RuntimeError, OSError) as error:
            print(f"{NAME} could not answer: {error}")
            continue
        conversation.add_user(line)
        conversation.add_assistant(answer)
        if autosave:
            memory.save(conversation)


def run_providers(args: argparse.Namespace) -> int:
    if args.as_json:
        print(
            json.dumps(
                [
                    {
                        "key": key,
                        "title": title,
                        "default_model": model,
                        "summary": summary,
                        "local": local,
                    }
                    for key, title, model, summary, local in provider_summaries()
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Cloud providers Cloud can reach (default: {DEFAULT_PROVIDER_KEY}):")
    for key, title, model, summary, local in provider_summaries():
        where = "this machine" if local else "remote"
        print(f"  {key:12s} {title:34s} [{where}] default: {model}")
        print(f"  {'':12s} {summary}")
    print("\nChoose one with: python -m cli cloud --provider <key> \"your question\"")
    print("The API key is read from the provider's environment variable (e.g. OPENAI_API_KEY).")
    return 0


def run_status(args: argparse.Namespace) -> int:
    engine = _engine(args)
    line = status_line(engine, probe=args.probe)
    if args.as_json:
        status = cloud_status(engine.client, probe=args.probe)
        print(
            json.dumps(
                {
                    "provider": status.provider,
                    "model": status.model,
                    "available": status.available,
                    "key_required": status.key_required,
                    "key_present": status.key_present,
                    "local": status.local,
                    "reachable": status.reachable,
                    "hint": status.hint,
                    "status": line,
                },
                ensure_ascii=False,
            )
        )
        return 0 if status.available else 1
    print(line)
    return 0 if cloud_status(engine.client, probe=args.probe).available else 1


#: The subcommands Cloud understands. Anything else is read as a chat message.
_COMMANDS = ("chat", "providers", "status")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloud",
        description=f"{NAME}: talk to a remote, OpenAI-compatible model (ChatGPT, Claude, Gemini, ...).",
    )
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Talk with the cloud model (the default).")
    _add_chat_arguments(chat)

    providers = sub.add_parser("providers", help="List the providers Cloud can reach.")
    providers.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    status = sub.add_parser("status", help="Show whether a key is set and the endpoint answers.")
    _add_connection_arguments(status)
    status.add_argument(
        "--probe",
        action="store_true",
        help="also ask the endpoint for its model list (a real network request).",
    )
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")
    return parser


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default=None,
        help=f"which provider to use (default {DEFAULT_PROVIDER_KEY}).",
    )
    parser.add_argument("--model", default=None, help="the model to ask for (overrides the default).")
    parser.add_argument("--base-url", default=None, help="an OpenAI-compatible base URL to use instead.")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="the environment variable holding the API key (overrides the provider's own).",
    )


def _add_chat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("message", nargs="*", help="one question; omit for the interactive loop")
    _add_connection_arguments(parser)
    parser.add_argument(
        "--temperature", type=float, default=None, help="sampling temperature (provider default otherwise)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="max tokens per reply"
    )
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false", help="print each reply whole"
    )
    parser.add_argument(
        "--session", default=None, help="resume a saved session by id, or start a new one"
    )
    parser.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit machine-readable JSON")
    parser.set_defaults(stream=True)


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Read a bare question as ``chat``; keep a real subcommand as-is.

    ``python -m cli cloud "hi"`` and ``python -m cli cloud --provider groq "hi"``
    must both chat, while ``cloud status`` must report. Only the *first* token is
    inspected, and only a known subcommand is honoured, so a question that happens
    to start with a flag still chats.
    """
    args = list(argv)
    if args and args[0] in {"-h", "--help", "help"}:
        return args
    if args and args[0] in _COMMANDS:
        return args
    return ["chat", *args]


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(_normalize_argv(raw))

    if args.command == "providers":
        return run_providers(args)
    if args.command == "status":
        return run_status(args)

    memory = _memory()
    if args.list_sessions:
        saved = memory.list_saved()
        if saved:
            print("Saved sessions:")
            for session_id in saved:
                print(f"  - {session_id}")
        else:
            print("No saved sessions yet.")
        return 0

    engine = _engine(args)
    if not engine.is_available():
        print(status_line(engine), file=sys.stderr if args.as_json else sys.stdout)
        return 1

    if args.session:
        loaded = memory.load(args.session)
        conversation = loaded or memory.start(title=args.session)
        conversation.id = args.session
        if loaded is not None and not args.as_json:
            print(f"Resumed session {conversation.id}.")
    else:
        title = args.message[0] if args.message else NAME
        conversation = memory.start(title=title[:40])

    if args.message:
        code = _one_shot(engine, " ".join(args.message), conversation, as_json=args.as_json)
        # A one-shot turn in a named session must persist too, or the next run
        # would start from nothing and the session would be useless.
        if args.session:
            memory.save(conversation)
        return code
    return _interactive(
        engine,
        conversation,
        memory,
        stream=args.stream,
        as_json=args.as_json,
        autosave=args.session is not None,
    )


__all__ = ["BLURB", "INTRO", "main", "status_line"]


if __name__ == "__main__":
    raise SystemExit(main())
