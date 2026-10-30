"""The Characters CLI: the headless twin of the Characters window.

It drives the *same* :class:`gui.character_manager.CharacterManagerService` the
window drives, so the CLI and the window share one set of characters, one applied
configuration, one learning store per character, and one knowledge-sharing gate --
no logic is reimplemented here.

Commands:

* ``list`` -- the seven characters and the project's own model, with their state;
* ``show <character>`` -- the full configuration, section by section;
* ``set <character> --section ...`` -- apply an edited section (validated, with
  rollback, exactly like the window's APPLY);
* ``reset <character>`` -- back to the shipped default;
* ``stats <character>`` / ``records <character>`` / ``search <character> <q>`` --
  the character's own learning store;
* ``verify`` / ``reject`` -- supervised verification of a record;
* ``export <character>`` -- back the records up to a JSON file;
* ``reset-learning <character> --confirm`` -- forget a character's records;
* ``share <character>|--all`` -- the main model learning from the permitted
  characters (Natasha is refused in the backend);
* ``main`` -- what the main model has learned into its own store;
* ``audit`` -- the learning audit, including refused accesses;
* ``shareable`` -- which characters may be shared and which may not, and why.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the character or record
is unknown, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace

from core.character_config import (
    CharacterConfig,
    ConfigError,
    GeneralConfig,
    LearningSettings,
    ModelConfig,
    PersonalityConfig,
    VoiceConfig,
)
from core.learning_isolation import LearningAccessError
from gui.character_manager import CharacterManagerService


def _service() -> CharacterManagerService:
    return CharacterManagerService()


def _fail(message: str, code: int = 2) -> int:
    print(message)
    return code


# ------------------------------------------------------------------ commands


def _cmd_list(args: argparse.Namespace) -> int:
    service = _service()
    if args.json:
        rows = [
            {
                "key": row.key,
                "name": row.name,
                "role": row.role,
                "active": row.active,
                "private": row.private,
                "records": row.records,
            }
            for row in service.summaries()
        ]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(service.status())
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    service = _service()
    try:
        text = service.describe_config(args.character)
    except LearningAccessError as error:
        return _fail(str(error), 3)
    if args.json:
        print(
            json.dumps(
                service.config(args.character).to_primitive(), ensure_ascii=False, indent=2
            )
        )
        return 0
    print(text)
    return 0


def _cmd_set(args: argparse.Namespace) -> int:
    """Apply an edited section (the CLI's APPLY)."""
    service = _service()
    try:
        config = service.config(args.character)
    except LearningAccessError as error:
        return _fail(str(error), 3)
    try:
        updated = _edit_section(config, args)
    except ConfigError as error:
        return _fail(f"Not applied: {error}", 4)
    result = service.apply(updated)
    print(result.describe())
    if result.error is not None:
        return 4
    return 0 if result.ok else 4


def _cmd_reset(args: argparse.Namespace) -> int:
    service = _service()
    try:
        config = service.reset(args.character)
    except LearningAccessError as error:
        return _fail(str(error), 3)
    print(f"{args.character} is back to the shipped default ({config.general.name}).")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    service = _service()
    stats = service.stats(args.character)
    if args.json:
        print(json.dumps(stats.to_primitive(), ensure_ascii=False, indent=2))
        return 0
    print(stats.summary())
    return 0


def _cmd_records(args: argparse.Namespace) -> int:
    service = _service()
    rows = service.records(args.character, limit=args.limit)
    if args.json:
        print(json.dumps([record.to_primitive() for record in rows], ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print(f"{args.character} has no learning records yet.")
        return 0
    for record in rows:
        print(service.describe_record(record))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    service = _service()
    rows = service.search(args.character, args.query, limit=args.limit)
    if not rows:
        print(f"No records in {args.character} match {args.query!r}.")
        return 0
    for record in rows:
        print(service.describe_record(record))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    service = _service()
    record = service.verify(args.character, args.record_id, note=args.note)
    if record is None:
        return _fail(f"No record {args.record_id!r} in {args.character}.", 3)
    print(f"Verified: {service.describe_record(record)}")
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    service = _service()
    record = service.reject(args.character, args.record_id, note=args.note)
    if record is None:
        return _fail(f"No record {args.record_id!r} in {args.character}.", 3)
    print(f"Rejected: {service.describe_record(record)}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    service = _service()
    path = service.export(args.character, args.output)
    print(f"Exported {args.character}'s learning records to {path}.")
    return 0


def _cmd_reset_learning(args: argparse.Namespace) -> int:
    service = _service()
    if not args.confirm:
        return _fail(
            "Refusing to forget the records: pass --confirm to confirm the reset.", 4
        )
    service.reset_learning(args.character, confirm=True)
    print(f"{args.character}'s learning records were reset.")
    return 0


def _cmd_share(args: argparse.Namespace) -> int:
    service = _service()
    if args.all:
        report = service.learn_from_all_characters(limit=args.limit)
        print(report.describe())
        # ``--all`` never targets a private character, so a refusal here would be
        # a real fault; it is still reported through the exit code.
        return 4 if any(result.skipped_private for result in report.results) else 0
    result = service.learn_from_character(args.character, limit=args.limit)
    print(result.describe())
    # A refused cross-character access is a gate refusal, so it is machine-
    # detectable (exit 4) as well as logged -- not reported as a plain success.
    return 4 if result.skipped_private else 0


def _cmd_main(args: argparse.Namespace) -> int:
    service = _service()
    rows = service.main_model_records(limit=args.limit)
    if args.json:
        print(json.dumps([record.to_primitive() for record in rows], ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("The main model has not learned from any character yet.")
        return 0
    for record in rows:
        print(service.describe_record(record))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    service = _service()
    if args.denials:
        lines = service.denials(limit=args.limit)
        header = "Refused cross-character access attempts:"
    else:
        lines = service.audit_tail(limit=args.limit)
        header = "Recent learning operations:"
    print(header)
    if not lines:
        print("  (none)")
    for line in lines:
        print(f"  {line}")
    return 0


def _cmd_shareable(args: argparse.Namespace) -> int:
    service = _service()
    if args.json:
        print(
            json.dumps(
                {
                    "shareable": list(service.shareable_characters()),
                    "private": list(service.private_characters()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print("The main model may learn from:")
    for key in service.shareable_characters():
        print(f"  {key}")
    print("Never shared:")
    for key in service.private_characters():
        print(f"  {key} — {service.share_reason(key)}")
    return 0


# ------------------------------------------------------------------- editing


def _edit_section(config: CharacterConfig, args: argparse.Namespace) -> CharacterConfig:
    """Build the edited configuration from the flags the caller passed.

    Only the fields actually given are changed, so ``set jarvis --temperature 0.2``
    does not disturb anything else. A value that is out of range is caught by
    :meth:`CharacterConfig.validate` (through the service's APPLY), so a bad edit
    is refused with a rollback rather than silently clamped here.
    """
    general: GeneralConfig = config.general
    personality: PersonalityConfig = config.personality
    model: ModelConfig = config.model
    voice: VoiceConfig = config.voice
    learning: LearningSettings = config.learning
    changed = False

    if args.name is not None:
        general = replace(general, name=args.name)
        changed = True
    if args.description is not None:
        general = replace(general, description=args.description)
        changed = True
    if args.role is not None:
        general = replace(general, role=args.role)
        changed = True
    if args.active is not None:
        general = replace(general, active=args.active)
        changed = True

    if args.instructions is not None:
        personality = replace(personality, instructions=args.instructions)
        changed = True
    if args.formality is not None:
        personality = replace(personality, formality=args.formality)
        changed = True
    if args.language is not None:
        personality = replace(personality, preferred_language=args.language)
        changed = True

    if args.model is not None:
        model = replace(model, key=args.model)
        changed = True
    if args.temperature is not None:
        model = replace(model, temperature=args.temperature)
        changed = True
    if args.context is not None:
        model = replace(model, context_length=args.context)
        changed = True
    if args.max_tokens is not None:
        model = replace(model, max_tokens=args.max_tokens)
        changed = True
    if args.model_active is not None:
        model = replace(model, active=args.model_active)
        changed = True

    if args.voice is not None:
        voice = replace(voice, voice=args.voice)
        changed = True
    if args.voice_engine is not None:
        voice = replace(voice, engine=args.voice_engine)
        changed = True
    if args.speed is not None:
        voice = replace(voice, speed=args.speed)
        changed = True
    if args.pitch is not None:
        voice = replace(voice, pitch=args.pitch)
        changed = True
    if args.volume is not None:
        voice = replace(voice, volume=args.volume)
        changed = True

    if args.learning is not None:
        learning = replace(learning, enabled=args.learning)
        changed = True
    if args.learn_from_model is not None:
        learning = replace(learning, learn_from_model=args.learn_from_model)
        changed = True
    if args.learn_from_internet is not None:
        learning = replace(learning, learn_from_internet=args.learn_from_internet)
        changed = True
    if args.mode is not None:
        learning = replace(learning, mode=args.mode)
        changed = True
    if args.verification_threshold is not None:
        learning = replace(learning, verification_threshold=args.verification_threshold)
        changed = True
    if args.confidence_threshold is not None:
        learning = replace(learning, confidence_threshold=args.confidence_threshold)
        changed = True
    if args.share is not None:
        learning = replace(learning, share_with_main_model=args.share)
        changed = True

    if not changed:
        raise ConfigError("general", "", "nothing was given to change.")
    return replace(
        config,
        general=general,
        personality=personality,
        model=model,
        voice=voice,
        learning=learning,
    )


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli characters",
        description="Manage the seven characters and the project's own model.",
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="list every character and its state")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="show one character's configuration")
    p_show.add_argument("character")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=_cmd_show)

    p_set = sub.add_parser("set", help="edit and APPLY one character's settings")
    p_set.add_argument("character")
    p_set.add_argument("--name")
    p_set.add_argument("--description")
    p_set.add_argument("--role")
    p_set.add_argument("--active", type=_boolean, default=None)
    p_set.add_argument("--instructions")
    p_set.add_argument("--formality", type=float)
    p_set.add_argument("--language")
    p_set.add_argument("--model")
    p_set.add_argument("--temperature", type=float)
    p_set.add_argument("--context", type=int)
    p_set.add_argument("--max-tokens", type=int, dest="max_tokens")
    p_set.add_argument("--model-active", type=_boolean, default=None, dest="model_active")
    p_set.add_argument("--voice")
    p_set.add_argument("--voice-engine", dest="voice_engine")
    p_set.add_argument("--speed", type=float)
    p_set.add_argument("--pitch", type=float)
    p_set.add_argument("--volume", type=float)
    p_set.add_argument("--learning", type=_boolean, default=None)
    p_set.add_argument("--learn-from-model", type=_boolean, default=None, dest="learn_from_model")
    p_set.add_argument(
        "--learn-from-internet", type=_boolean, default=None, dest="learn_from_internet"
    )
    p_set.add_argument("--mode")
    p_set.add_argument(
        "--verification-threshold", type=float, dest="verification_threshold"
    )
    p_set.add_argument("--confidence-threshold", type=float, dest="confidence_threshold")
    p_set.add_argument("--share", type=_boolean, default=None)
    p_set.set_defaults(func=_cmd_set)

    p_reset = sub.add_parser("reset", help="back to the shipped default")
    p_reset.add_argument("character")
    p_reset.set_defaults(func=_cmd_reset)

    p_stats = sub.add_parser("stats", help="a character's learning statistics")
    p_stats.add_argument("character")
    p_stats.add_argument("--json", action="store_true")
    p_stats.set_defaults(func=_cmd_stats)

    p_records = sub.add_parser("records", help="a character's learning records")
    p_records.add_argument("character")
    p_records.add_argument("--limit", type=int, default=50)
    p_records.add_argument("--json", action="store_true")
    p_records.set_defaults(func=_cmd_records)

    p_search = sub.add_parser("search", help="search a character's own records")
    p_search.add_argument("character")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=50)
    p_search.set_defaults(func=_cmd_search)

    p_verify = sub.add_parser("verify", help="verify one record (supervised)")
    p_verify.add_argument("character")
    p_verify.add_argument("record_id")
    p_verify.add_argument("--note", default="")
    p_verify.set_defaults(func=_cmd_verify)

    p_reject = sub.add_parser("reject", help="reject one record, keeping it")
    p_reject.add_argument("character")
    p_reject.add_argument("record_id")
    p_reject.add_argument("--note", default="")
    p_reject.set_defaults(func=_cmd_reject)

    p_export = sub.add_parser("export", help="back a character's records up to JSON")
    p_export.add_argument("character")
    p_export.add_argument("--output")
    p_export.set_defaults(func=_cmd_export)

    p_reset_learning = sub.add_parser("reset-learning", help="forget a character's records")
    p_reset_learning.add_argument("character")
    p_reset_learning.add_argument("--confirm", action="store_true")
    p_reset_learning.set_defaults(func=_cmd_reset_learning)

    p_share = sub.add_parser("share", help="the main model learns from a character")
    p_share.add_argument("character", nargs="?", default="")
    p_share.add_argument("--all", action="store_true")
    p_share.add_argument("--limit", type=int, default=100)
    p_share.set_defaults(func=_cmd_share)

    p_main = sub.add_parser("main", help="what the main model has learned")
    p_main.add_argument("--limit", type=int, default=50)
    p_main.add_argument("--json", action="store_true")
    p_main.set_defaults(func=_cmd_main)

    p_audit = sub.add_parser("audit", help="the learning audit")
    p_audit.add_argument("--denials", action="store_true")
    p_audit.add_argument("--limit", type=int, default=40)
    p_audit.set_defaults(func=_cmd_audit)

    p_shareable = sub.add_parser("shareable", help="who may be shared, and who may not")
    p_shareable.add_argument("--json", action="store_true")
    p_shareable.set_defaults(func=_cmd_shareable)

    return parser


def _boolean(value: str) -> bool:
    """Parse an on/off flag value."""
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        return _cmd_list(argparse.Namespace(json=False))
    try:
        return int(args.func(args))
    except LearningAccessError as error:
        return _fail(str(error), 3)
    except (ValueError, OSError, KeyError) as error:
        return _fail(f"{type(error).__name__}: {error}", 5)


__all__ = ["build_parser", "main"]
