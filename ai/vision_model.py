"""Vision from the command line: a petroleum-engineering local conversation.

    python -m cli vision                          # interactive chat (type 'quit' to exit)
    python -m cli vision "what is the material balance equation?"   # one question
    python -m cli vision --session demo "and for a gas cap?"  # keep context across runs
    python -m cli vision download --approve        # one-time: fetch a model (~5 GB)
    python -m cli vision models                    # list the models it can fetch
    python -m cli vision status                    # what is installed and present

Vision is the project's *petroleum* mouth. Unlike the grounded assistant (which
answers only from the petroleum corpus, with citations) and unlike ``--mind``
(which speaks the small in-project checkpoint), Vision runs a real, advanced
model **locally** and specialises in petroleum engineering -- reservoir,
drilling, production, petrophysics, PVT, facilities -- built on the OGAI Reasoner
weights. It is wired directly below Ultron in the menu.

The model is a GGUF file under the writable data folder (``data/vision`` by
default). Drop any ``.gguf`` there, or download a known-good one with
``download --approve``. Nothing is sent to a remote server: the model runs on
this machine through ``llama-cpp-python``.

Sessions persist under the data directory, so a conversation keeps its context
across runs. Every answer is labelled as coming from the local petroleum model
and is *not* grounded in cited sources -- that honesty is deliberate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.conversation import Conversation, ConversationMemory, Role
from core.paths import data_dir
from core.vision_model import (
    BLURB,
    DEFAULT_CONTEXT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SPEC_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    HEADER,
    NAME,
    VisionEngine,
    VisionUnavailable,
    download_model,
    find_model_file,
    spec_summaries,
)
from storage.json_store import JsonStore

INTRO = (
    f"{NAME}: a petroleum-engineering local model. Ask about reservoir, drilling, "
    "production, or PVT; type 'quit' to exit."
)


def _sessions_dir() -> Path:
    return data_dir() / "vision" / "sessions"


def _memory() -> ConversationMemory:
    """A persistent session store, or in-memory if the data folder is not writable."""
    try:
        return ConversationMemory(JsonStore(_sessions_dir()))
    except (OSError, ValueError):
        return ConversationMemory()


def _engine(args: argparse.Namespace) -> VisionEngine:
    return VisionEngine(
        model_path=args.model_path,
        context=args.context,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_layers=args.gpu_layers,
    )


def status_line(engine: VisionEngine | None = None) -> str:
    """A one-line, honest status: is the runtime and a model present, and which."""
    engine = engine or VisionEngine()
    if engine.is_available():
        return f"{NAME} ready. {engine.describe()}"
    missing = find_model_file()
    if missing is None:
        return (
            f"{NAME} has no local model yet. Run `python -m cli vision download --approve` "
            f"or drop a `.gguf` file into {data_dir() / 'vision'}."
        )
    return (
        f"{NAME} found {missing.name}, but the runtime `llama-cpp-python` is not installed. "
        "Install it with `python -m pip install -r requirements-vision.txt`."
    )


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


def _one_shot(engine: VisionEngine, prompt: str, conversation: Conversation, *, as_json: bool) -> int:
    try:
        answer = engine.reply(prompt, history=_history(conversation))
    except VisionUnavailable as error:
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
    engine: VisionEngine,
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
        except VisionUnavailable as error:
            print(f"{NAME} is not available: {error}")
            return 1
        except (ValueError, RuntimeError, OSError) as error:
            print(f"{NAME} could not answer: {error}")
            continue
        conversation.add_user(line)
        conversation.add_assistant(answer)
        if autosave:
            memory.save(conversation)


def run_download(args: argparse.Namespace) -> int:
    if not args.approve:
        print(
            "error: downloading the Vision model weights is an external action and "
            "needs --approve (it fetches several GB from the internet)."
        )
        return 1
    try:
        path = download_model(
            args.model or DEFAULT_SPEC_KEY,
            approved=True,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except (VisionUnavailable, ValueError) as error:
        print(f"error: {error}")
        return 1
    if args.as_json:
        print(json.dumps({"model_file": str(path)}, ensure_ascii=False))
        return 0
    print(f"Vision model ready at {path}.")
    return 0


def run_models(args: argparse.Namespace) -> int:
    if args.as_json:
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
    print(f"Models Vision can download (default: {DEFAULT_SPEC_KEY}):")
    for key, title, size, summary in spec_summaries():
        print(f"  {key:14s} {title:34s} ~{size:.1f} GB")
        print(f"  {'':14s} {summary}")
    print("\nFetch one with: python -m cli vision download --approve --model <key>")
    return 0


def run_status(args: argparse.Namespace) -> int:
    line = status_line()
    if args.as_json:
        print(json.dumps({"status": line}, ensure_ascii=False))
        return 0
    print(line)
    return 0


#: The subcommands Vision understands. Anything else is read as a chat message.
_COMMANDS = ("chat", "download", "models", "status")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision",
        description=f"{NAME}: a petroleum-engineering local language model.",
    )
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Talk with Vision (the default).")
    _add_chat_arguments(chat)

    download = sub.add_parser("download", help="Download a model's weights (one-time, ~5 GB).")
    download.add_argument("--approve", action="store_true", help="grant the external download.")
    download.add_argument(
        "--model", default=None, help=f"which model to fetch (default {DEFAULT_SPEC_KEY})."
    )
    download.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    models = sub.add_parser("models", help="List the models Vision can download.")
    models.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")

    status = sub.add_parser("status", help="Show whether the runtime and a model are present.")
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON.")
    return parser


def _add_chat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("message", nargs="*", help="one question; omit for the interactive loop")
    parser.add_argument("--model-path", default=None, help="a specific .gguf file to load")
    parser.add_argument(
        "--context", type=int, default=DEFAULT_CONTEXT, help="context window in tokens"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max tokens per reply"
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="nucleus sampling probability"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=0, help="layers to offload to a GPU (0 = CPU only)"
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

    ``python -m cli vision "hi"`` and ``python -m cli vision --session s "hi"`` must
    both chat, while ``vision download`` must download. Only the *first* token is
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

    if args.command == "download":
        return run_download(args)
    if args.command == "models":
        return run_models(args)
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
        print(status_line(engine))
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
