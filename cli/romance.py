"""CLI for the Romance & Companion Studio, headless.

    python -m cli romance status                 # what the Studio holds right now
    python -m cli romance characters             # the character library
    python -m cli romance characters --archived  # include archived characters
    python -m cli romance new "Layla" --age 27 --interest coffee --interest poetry
    python -m cli romance show CHR-…             # one character's full profile
    python -m cli romance favorite CHR-…         # star a character
    python -m cli romance archive CHR-…          # move to the archive
    python -m cli romance restore CHR-…          # bring it back
    python -m cli romance delete CHR-… --choice keep   # delete, keeping memories
    python -m cli romance export CHR-… out.json
    python -m cli romance import out.json
    python -m cli romance modes                  # the conversation modes
    python -m cli romance chat CHR-… "hello"     # one turn (opens a conversation)
    python -m cli romance chat CHR-… "hello" --conversation CON-…
    python -m cli romance send CON-… "hello"     # a turn in an existing conversation
    python -m cli romance history CON-…          # a conversation's messages
    python -m cli romance conversations CHR-…    # a character's conversations
    python -m cli romance relationship CHR-…     # the relationship state
    python -m cli romance milestones CHR-…       # the relationship history
    python -m cli romance reset-relationship CHR-…
    python -m cli romance memories CHR-…         # the character's memories
    python -m cli romance recall CHR-… "coffee"  # rank memories for a query
    python -m cli romance remember CHR-… "likes tea" --kind semantic_personal
    python -m cli romance forget MEM-…           # forget one memory
    python -m cli romance export-memories CHR-…  # the memories as JSON
    python -m cli romance scenarios              # the scenario library
    python -m cli romance scenario "Coffee" --category coffee_shop --character CHR-…
    python -m cli romance voices                 # the voices this machine really has
    python -m cli romance set-voice CHR-… --voice system:… --speed 1.1
    python -m cli romance safety                 # the safety/local-only status
    python -m cli romance adult --on --confirm   # enable adult mode (needs --confirm)
    python -m cli romance boundary CHR-… "no politics"

This is the headless twin of the Studio window: it drives the *same*
:class:`gui.romance.RomanceStudioService`, so the CLI and the window share one
database, one character library, and one conversation history, and no workflow
logic is reimplemented here.

The shared flags (``--data-root``, ``--json``) are accepted **before or after**
the subcommand, exactly like the Workplace and Code Studio CLIs.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the Studio or a model
is not available, ``4`` an operation was refused, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from core.romance.characters import CharacterError
from core.romance.memory import MemoryError
from core.romance.orchestrator import OrchestratorError
from core.romance.relationship import RelationshipError
from core.romance.roleplay import RoleplayError
from core.romance.voice import VoiceAdapterError
from gui.romance import RomanceStudioService

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_REFUSED = 4
EXIT_FAILURE = 5

_DOMAIN_ERRORS = (
    CharacterError,
    MemoryError,
    OrchestratorError,
    RelationshipError,
    RoleplayError,
    VoiceAdapterError,
)


def _common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--json", action="store_true", dest="as_json", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def _sub(parser: argparse._SubParsersAction, name: str, help: str) -> argparse.ArgumentParser:
    child = parser.add_parser(name, help=help)
    _common_flags(child)
    return child


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="romance", description=__doc__)
    parser.add_argument("--data-root", default=None, help="Override the data folder.")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Machine-readable output."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _sub(sub, "status", help="What the Studio holds right now.")
    _sub(sub, "modes", help="The conversation modes.")

    characters = _sub(sub, "characters", help="The character library.")
    characters.add_argument("--archived", action="store_true", help="Include archived.")
    characters.add_argument("--search", default="", help="Filter by name or tag.")
    characters.add_argument("--favorites", action="store_true", help="Only favorites.")

    new = _sub(sub, "new", help="Create a character.")
    new.add_argument("name")
    new.add_argument("--age", type=int, default=None, help="Fictional age (adults only).")
    new.add_argument("--nickname", default="")
    new.add_argument("--gender", default="")
    new.add_argument("--biography", default="")
    new.add_argument("--nationality", default="")
    new.add_argument("--language", action="append", default=[], dest="languages")
    new.add_argument("--appearance", default="")
    new.add_argument("--relationship-style", default="")
    new.add_argument("--mode", default="general", help="Default conversation mode.")
    new.add_argument("--model", default="", help="Assigned language-model key.")
    new.add_argument("--interest", action="append", default=[], dest="interests")
    new.add_argument("--boundary", action="append", default=[], dest="boundaries")

    show = _sub(sub, "show", help="One character's full profile.")
    show.add_argument("character_id")

    _sub(sub, "discover", help="The clearly-labelled example companions this build ships.")
    add_example = _sub(sub, "add-example", help="Add one shipped example companion by name.")
    add_example.add_argument("name")

    for name, help_text in (
        ("favorite", "Star a character."),
        ("unfavorite", "Un-star a character."),
        ("archive", "Move a character to the archive."),
        ("restore", "Restore a character from the archive."),
    ):
        command = _sub(sub, name, help=help_text)
        command.add_argument("character_id")

    delete = _sub(sub, "delete", help="Delete a character (with a retention choice).")

    delete.add_argument("character_id")
    delete.add_argument(
        "--choice", default="keep", choices=("keep", "purge", "anonymize"),
        help="keep the conversations and memories, purge them, or anonymize them.",
    )

    export = _sub(sub, "export", help="Export a character to a file.")
    export.add_argument("character_id")
    export.add_argument("destination")

    importc = _sub(sub, "import", help="Import a character from a file.")
    importc.add_argument("source")
    importc.add_argument("--name", default="")

    chat = _sub(sub, "chat", help="One turn with a character (opens a conversation).")
    chat.add_argument("character_id")
    chat.add_argument("message")
    chat.add_argument("--conversation", default="", help="Use an existing conversation.")
    chat.add_argument("--mode", default="", help="Override the conversation mode.")
    chat.add_argument("--title", default="Conversation")

    send = _sub(sub, "send", help="A turn in an existing conversation.")
    send.add_argument("conversation_id")
    send.add_argument("message")
    send.add_argument("--mode", default="")

    history = _sub(sub, "history", help="A conversation's messages.")
    history.add_argument("conversation_id")
    history.add_argument("--limit", type=int, default=0)

    conversations = _sub(sub, "conversations", help="A character's conversations.")
    conversations.add_argument("character_id")

    recent = _sub(sub, "recent", help="The most recent conversations.")
    recent.add_argument("--limit", type=int, default=20)

    relationship = _sub(sub, "relationship", help="The relationship state.")
    relationship.add_argument("character_id")

    milestones = _sub(sub, "milestones", help="The relationship history.")
    milestones.add_argument("character_id")

    reset = _sub(sub, "reset-relationship", help="Reset a relationship.")
    reset.add_argument("character_id")

    memories = _sub(sub, "memories", help="A character's memories.")
    memories.add_argument("character_id")

    recall = _sub(sub, "recall", help="Rank a character's memories for a query.")
    recall.add_argument("character_id")
    recall.add_argument("query", nargs="?", default="")
    recall.add_argument("--limit", type=int, default=6)

    remember = _sub(sub, "remember", help="Store a memory for a character.")
    remember.add_argument("character_id")
    remember.add_argument("content")
    remember.add_argument("--kind", default="semantic_personal")
    remember.add_argument("--importance", type=float, default=0.5)

    forget = _sub(sub, "forget", help="Forget one memory.")
    forget.add_argument("memory_id")

    export_memories = _sub(sub, "export-memories", help="A character's memories as JSON.")
    export_memories.add_argument("character_id")

    _sub(sub, "scenarios", help="The scenario library.")
    scenario = _sub(sub, "scenario", help="Create a scenario.")
    scenario.add_argument("title")
    scenario.add_argument("--category", default="first_meeting")
    scenario.add_argument("--description", default="")
    scenario.add_argument("--location", default="")
    scenario.add_argument("--character", default="", help="Add a participating character.")

    voices = _sub(sub, "voices", help="The voices this machine really has.")
    voices.add_argument("--engine", default="", help="Narrow to one engine.")
    voices.add_argument("--language", default="", help="Narrow to one language.")

    set_voice = _sub(sub, "set-voice", help="Set a character's voice.")
    set_voice.add_argument("character_id")
    set_voice.add_argument("--engine", default="")
    set_voice.add_argument("--voice", default="", dest="voice_key")
    set_voice.add_argument("--language", default="")
    set_voice.add_argument("--speed", type=float, default=None)
    set_voice.add_argument("--pitch", type=float, default=None)
    set_voice.add_argument("--volume", type=float, default=None)
    set_voice.add_argument("--mood", default="")
    set_voice.add_argument("--enable", action="store_true")
    set_voice.add_argument("--disable", action="store_true")

    _sub(sub, "safety", help="The safety and local-only status.")
    adult = _sub(sub, "adult", help="Turn adult mode on or off.")
    adult.add_argument("--on", action="store_true", dest="adult_on")
    adult.add_argument("--off", action="store_true", dest="adult_off")
    adult.add_argument("--confirm", action="store_true", help="Confirm adult eligibility.")

    boundary = _sub(sub, "boundary", help="Add a conversation boundary.")
    boundary.add_argument("character_id")
    boundary.add_argument("value")

    return parser


# -- helpers --------------------------------------------------------------


def _service(args: argparse.Namespace) -> RomanceStudioService:
    root = getattr(args, "data_root", None)
    return RomanceStudioService(root=root)


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if getattr(args, "as_json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(text)


def _character_row(row: Any) -> dict[str, Any]:
    return {
        "character_id": row.character_id,
        "name": row.name,
        "nickname": row.nickname,
        "favorite": row.favorite,
        "archived": row.archived,
        "conversation_count": row.conversation_count,
        "memory_count": row.memory_count,
        "last_active": row.last_active,
        "stage": row.stage_title,
    }


# -- handlers -------------------------------------------------------------


def run_status(args: argparse.Namespace) -> int:
    status = _service(args).status()
    _emit(args, status.__dict__, status.describe())
    return EXIT_OK


def run_modes(args: argparse.Namespace) -> int:
    rows = _service(args).modes()
    text = "\n".join(f"{key}: {title} — {summary}" for key, title, summary in rows)
    _emit(args, [{"key": k, "title": t, "summary": s} for k, t, s in rows], text)
    return EXIT_OK


def run_characters(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.character_rows(
        include_archived=args.archived,
        search=args.search,
        favorite=True if args.favorites else None,
    )
    if not rows:
        _emit(args, [], "No characters yet. Create one with `romance new`.")
        return EXIT_OK
    lines = []
    for row in rows:
        lines.append(
            f"{row.character_id}  {row.label()}"
            + (f"  [{row.stage_title}]" if row.stage_title else "")
            + f"  ({row.conversation_count} chats, {row.memory_count} memories)"
        )
    _emit(args, [_character_row(r) for r in rows], "\n".join(lines))
    return EXIT_OK


def run_new(args: argparse.Namespace) -> int:
    service = _service(args)
    bundle = service.create_character(
        args.name,
        fictional_age=args.age,
        nickname=args.nickname,
        gender=args.gender,
        biography=args.biography,
        nationality=args.nationality,
        languages=tuple(args.languages),
        appearance=args.appearance,
        relationship_style=args.relationship_style,
        default_mode=args.mode,
        model_key=args.model,
        interests=tuple(args.interests),
        boundaries=tuple(args.boundaries),
    )
    _emit(
        args,
        {"character_id": bundle.id, "name": bundle.name},
        f"Created {bundle.name} ({bundle.id}).",
    )
    return EXIT_OK


def run_show(args: argparse.Namespace) -> int:
    service = _service(args)
    bundle = service.character(args.character_id)
    character = bundle.character
    payload = {
        "character_id": bundle.id,
        "name": character.name,
        "nickname": character.nickname,
        "fictional_age": character.fictional_age,
        "adult_eligible": character.adult_eligible,
        "gender": character.gender,
        "biography": character.biography,
        "nationality": character.nationality,
        "languages": list(character.languages),
        "interests": list(bundle.interests()),
        "default_mode": character.default_mode,
        "model_key": character.model_key,
        "favorite": character.favorite,
        "archived": character.archived,
    }
    lines = [f"{character.name} ({bundle.id})"]
    for key, value in payload.items():
        if key in {"character_id", "name"}:
            continue
        if value not in ("", None, [], ()):
            lines.append(f"  {key}: {value}")
    _emit(args, payload, "\n".join(lines))
    return EXIT_OK


def run_discover(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.discover_examples()
    payload = [
        {"name": draft.name, "nickname": draft.nickname, "gender": draft.gender,
         "fictional_age": draft.fictional_age, "present": present}
        for draft, present in rows
    ]
    if not payload:
        _emit(args, [], "This build ships no example companions.")
        return EXIT_OK
    lines = [
        f"{'✓' if row['present'] else '＋'} {row['name']}"
        + (f" ({row['nickname']})" if row["nickname"] else "")
        + f" — {row['fictional_age']}, {row['gender']}"
        for row in payload
    ]
    _emit(args, payload, "\n".join(lines))
    return EXIT_OK


def run_add_example(args: argparse.Namespace) -> int:
    service = _service(args)
    bundle = service.add_example(args.name)
    _emit(
        args,
        {"character_id": bundle.id, "name": bundle.name},
        f"Added {bundle.name} ({bundle.id}).",
    )
    return EXIT_OK


def run_favorite(args: argparse.Namespace) -> int:
    bundle = _service(args).set_favorite(args.character_id, True)
    _emit(args, {"character_id": bundle.id}, f"Starred {bundle.name}.")
    return EXIT_OK


def run_unfavorite(args: argparse.Namespace) -> int:
    bundle = _service(args).set_favorite(args.character_id, False)
    _emit(args, {"character_id": bundle.id}, f"Un-starred {bundle.name}.")
    return EXIT_OK


def run_archive(args: argparse.Namespace) -> int:
    bundle = _service(args).archive_character(args.character_id)
    _emit(args, {"character_id": bundle.id}, f"Archived {bundle.name}.")
    return EXIT_OK


def run_restore(args: argparse.Namespace) -> int:
    bundle = _service(args).restore_character(args.character_id)
    _emit(args, {"character_id": bundle.id}, f"Restored {bundle.name}.")
    return EXIT_OK


def run_delete(args: argparse.Namespace) -> int:
    report = _service(args).delete_character(args.character_id, choice=args.choice)
    payload = report.to_primitive() if hasattr(report, "to_primitive") else {"character_id": args.character_id, "choice": args.choice}
    _emit(args, payload, f"Deleted {args.character_id} ({args.choice}).")
    return EXIT_OK


def run_export(args: argparse.Namespace) -> int:
    path = _service(args).export_character(args.character_id, args.destination)
    _emit(args, {"path": str(path)}, f"Exported to {path}.")
    return EXIT_OK


def run_import(args: argparse.Namespace) -> int:
    bundle = _service(args).import_character(args.source, name=args.name)
    _emit(args, {"character_id": bundle.id, "name": bundle.name}, f"Imported {bundle.name} ({bundle.id}).")
    return EXIT_OK


def run_chat(args: argparse.Namespace) -> int:
    service = _service(args)
    conversation_id = args.conversation
    if not conversation_id:
        conversation = service.start_conversation(
            args.character_id, title=args.title, mode=args.mode or "general"
        )
        conversation_id = conversation.record_id
    result = service.send(conversation_id, args.message, mode=args.mode)
    if result.refused:
        _emit(args, {"refused": True, "reason": result.reason}, f"Refused: {result.reason}")
        return EXIT_REFUSED
    if not result.ok:
        _emit(args, {"ok": False, "error": result.error}, f"No answer: {result.error}")
        return EXIT_UNAVAILABLE
    payload = {
        "conversation_id": conversation_id,
        "reply": result.reply,
        "speaker": result.speakers[0] if result.speakers else "",
    }
    _emit(args, payload, f"{payload['speaker']}: {result.reply}")
    return EXIT_OK


def run_send(args: argparse.Namespace) -> int:
    result = _service(args).send(args.conversation_id, args.message, mode=args.mode)
    if result.refused:
        _emit(args, {"refused": True, "reason": result.reason}, f"Refused: {result.reason}")
        return EXIT_REFUSED
    if not result.ok:
        _emit(args, {"ok": False, "error": result.error}, f"No answer: {result.error}")
        return EXIT_UNAVAILABLE
    _emit(args, {"reply": result.reply}, f"{result.speakers[0] if result.speakers else ''}: {result.reply}")
    return EXIT_OK


def run_history(args: argparse.Namespace) -> int:
    turns = _service(args).messages(args.conversation_id, limit=args.limit or None)
    payload = [{"role": t.role, "speaker": getattr(t, "speaker", ""), "content": t.content} for t in turns]
    text = "\n".join(
        f"[{t['role']}] {t['speaker'] + ': ' if t['speaker'] else ''}{t['content']}" for t in payload
    )
    _emit(args, payload, text or "(no messages)")
    return EXIT_OK


def run_conversations(args: argparse.Namespace) -> int:
    rows = _service(args).conversations_for(args.character_id)
    payload = [
        {
            "conversation_id": r.conversation_id,
            "title": r.title,
            "mode": r.mode,
            "message_count": r.message_count,
            "last_active": r.last_active,
        }
        for r in rows
    ]
    text = "\n".join(f"{r['conversation_id']}  {r['title']}  [{r['mode']}]  {r['message_count']} msgs" for r in payload)
    _emit(args, payload, text or "(no conversations)")
    return EXIT_OK


def run_recent(args: argparse.Namespace) -> int:
    rows = _service(args).recent_conversations(limit=args.limit)
    payload = [
        {"conversation_id": r.conversation_id, "title": r.title, "character_id": r.character_id, "message_count": r.message_count}
        for r in rows
    ]
    text = "\n".join(f"{r['conversation_id']}  {r['title']}  ({r['message_count']} msgs)" for r in payload)
    _emit(args, payload, text or "(no conversations)")
    return EXIT_OK


def run_relationship(args: argparse.Namespace) -> int:
    snapshot = _service(args).relationship(args.character_id)
    payload = {
        "character_id": args.character_id,
        "stage": snapshot.stage,
        "stage_title": snapshot.stage_title,
        "overall": snapshot.overall,
        "dimensions": snapshot.dimensions,
        "event_count": snapshot.event_count,
    }
    _emit(args, payload, snapshot.describe())
    return EXIT_OK


def run_milestones(args: argparse.Namespace) -> int:
    events = _service(args).relationship_history(args.character_id)
    payload = [
        {"kind": e.kind, "title": getattr(e, "title", ""), "occurred_at": getattr(e, "occurred_at", ""), "message_id": getattr(e, "message_id", "")}
        for e in events
    ]
    text = "\n".join(f"{e['occurred_at']}  {e['title'] or e['kind']}" for e in payload)
    _emit(args, payload, text or "(no milestones yet)")
    return EXIT_OK


def run_reset_relationship(args: argparse.Namespace) -> int:
    snapshot = _service(args).reset_relationship(args.character_id)
    _emit(args, {"stage": snapshot.stage}, f"Relationship reset to {snapshot.stage_title}.")
    return EXIT_OK


def run_memories(args: argparse.Namespace) -> int:
    memories = _service(args).memories(args.character_id)
    payload = [
        {"memory_id": m.record_id, "kind": m.kind, "content": m.content, "importance": m.importance, "created_at": m.created_at}
        for m in memories
    ]
    text = "\n".join(f"{m['memory_id']}  [{m['kind']}]  {m['content']}" for m in payload)
    _emit(args, payload, text or "(no memories)")
    return EXIT_OK


def run_recall(args: argparse.Namespace) -> int:
    recalled = _service(args).recall(args.character_id, args.query, limit=args.limit)
    payload = [
        {"memory_id": m.memory_id, "kind": m.kind, "content": m.content, "score": m.score}
        for m in recalled
    ]
    text = "\n".join(f"{m['score']:.3f}  [{m['kind']}]  {m['content']}" for m in payload)
    _emit(args, payload, text or "(nothing relevant)")
    return EXIT_OK


def run_remember(args: argparse.Namespace) -> int:
    memory = _service(args).add_memory(
        args.character_id, args.kind, args.content, importance=args.importance
    )
    _emit(args, {"memory_id": memory.record_id}, f"Remembered ({memory.record_id}).")
    return EXIT_OK


def run_forget(args: argparse.Namespace) -> int:
    _service(args).forget_memory(args.memory_id)
    _emit(args, {"memory_id": args.memory_id}, "Forgotten.")
    return EXIT_OK


def run_export_memories(args: argparse.Namespace) -> int:
    export = _service(args).export_memories(args.character_id)
    payload = export.to_primitive() if hasattr(export, "to_primitive") else {"memories": []}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return EXIT_OK


def run_scenarios(args: argparse.Namespace) -> int:
    rows = _service(args).scenarios()
    payload = [
        {"scenario_id": r.scenario_id, "title": r.title, "category": r.category, "category_label": r.category_label, "content_level": r.content_level, "participants": r.participant_count}
        for r in rows
    ]
    text = "\n".join(f"{r['scenario_id']}  {r['title']}  [{r['category_label']}]" for r in payload)
    _emit(args, payload, text or "(no scenarios)")
    return EXIT_OK


def run_scenario(args: argparse.Namespace) -> int:
    service = _service(args)
    scenario = service.create_scenario(
        args.title,
        category=args.category,
        description=args.description,
        location=args.location,
    )
    scenario_id = scenario.record_id
    if args.character:
        service.roleplay.add_participant(scenario_id, args.character)
    _emit(args, {"scenario_id": scenario_id}, f"Created scenario {scenario_id}.")
    return EXIT_OK


def run_voices(args: argparse.Namespace) -> int:
    options = _service(args).voices(
        engine=args.engine or None, language=args.language or None
    )
    payload = [
        {"key": o.key, "label": o.label, "engine": o.engine, "language": o.language, "source": o.source}
        for o in options
    ]
    text = "\n".join(f"{o['key']}  {o['label']}  [{o['engine']}/{o['language']}]" for o in payload)
    _emit(args, payload, text or "(no voices installed — install espeak-ng, Piper, or Qwen3-TTS)")
    return EXIT_OK


def run_set_voice(args: argparse.Namespace) -> int:
    changes: dict[str, Any] = {}
    if args.engine:
        changes["engine"] = args.engine
    if args.voice_key:
        changes["voice_key"] = args.voice_key
    if args.language:
        changes["language"] = args.language
    if args.speed is not None:
        changes["speed"] = args.speed
    if args.pitch is not None:
        changes["pitch"] = args.pitch
    if args.volume is not None:
        changes["volume"] = args.volume
    if args.mood:
        changes["mood"] = args.mood
    if args.enable:
        changes["enabled"] = True
    if args.disable:
        changes["enabled"] = False
    _service(args).set_voice(args.character_id, **changes)
    _emit(args, {"character_id": args.character_id, "changes": changes}, "Voice updated.")
    return EXIT_OK


def run_safety(args: argparse.Namespace) -> int:
    status = _service(args).safety_status()
    text = "\n".join(f"{k}: {v}" for k, v in status.items())
    _emit(args, status, text)
    return EXIT_OK


def run_adult(args: argparse.Namespace) -> int:
    service = _service(args)
    if args.adult_off:
        enabled = service.set_adult_mode(False)
        _emit(args, {"adult_mode": enabled}, "Adult mode is off.")
        return EXIT_OK
    if not args.adult_on:
        _emit(args, {"adult_mode": service.safety_status().get("adult_mode")}, "Use --on or --off.")
        return EXIT_USAGE
    try:
        enabled = service.set_adult_mode(True, confirmed=args.confirm)
    except Exception as error:
        _emit(args, {"adult_mode": False, "reason": str(error)}, f"Refused: {error}")
        return EXIT_REFUSED
    _emit(args, {"adult_mode": enabled}, "Adult mode is on." if enabled else "Adult mode stayed off.")
    return EXIT_OK


def run_boundary(args: argparse.Namespace) -> int:
    _service(args).add_boundary(args.character_id, args.value)
    _emit(args, {"character_id": args.character_id, "value": args.value}, "Boundary added.")
    return EXIT_OK


_HANDLERS = {
    "status": run_status,
    "modes": run_modes,
    "characters": run_characters,
    "new": run_new,
    "show": run_show,
    "discover": run_discover,
    "add-example": run_add_example,
    "favorite": run_favorite,
    "unfavorite": run_unfavorite,
    "archive": run_archive,
    "restore": run_restore,
    "delete": run_delete,
    "export": run_export,
    "import": run_import,
    "chat": run_chat,
    "send": run_send,
    "history": run_history,
    "conversations": run_conversations,
    "recent": run_recent,
    "relationship": run_relationship,
    "milestones": run_milestones,
    "reset-relationship": run_reset_relationship,
    "memories": run_memories,
    "recall": run_recall,
    "remember": run_remember,
    "forget": run_forget,
    "export-memories": run_export_memories,
    "scenarios": run_scenarios,
    "scenario": run_scenario,
    "voices": run_voices,
    "set-voice": run_set_voice,
    "safety": run_safety,
    "adult": run_adult,
    "boundary": run_boundary,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handler = _HANDLERS.get(args.command)
    if handler is None:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return EXIT_USAGE
    try:
        return handler(args)
    except _DOMAIN_ERRORS as error:
        if getattr(args, "as_json", False):
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
        else:
            print(f"error: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as error:  # pragma: no cover - unexpected
        print(f"unexpected failure: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
