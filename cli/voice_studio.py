"""CLI for the speech-engine studios: read a document aloud, per engine.

    python -m cli voice-studio list                     # the four engine studios
    python -m cli voice-studio status piper             # is that engine ready?
    python -m cli voice-studio voices piper             # only *its* voices
    python -m cli voice-studio moods                    # every emotional state
    python -m cli voice-studio voice piper man-1        # choose (and remember) a voice
    python -m cli voice-studio mood piper happy         # choose (and remember) a mood
    python -m cli voice-studio say piper report.txt     # read a document aloud
    python -m cli voice-studio say piper --text "…" --write-wav out.wav
    python -m cli voice-studio logo piper photo.png     # replace the studio's logo

This is the headless twin of the window (``＋ row → the speech-engine buttons``):
it drives the *same* :class:`gui.voice_studio.VoiceStudioService`, so the CLI and
the window share one voice choice, one mood, and one document reader, and no
workflow logic is reimplemented here. The four studios are the four engines the
project can speak through: the voices the OS installed, VoiceTut-TTS over
audio.cpp, Piper, and Qwen3-TTS.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the engine is not
available, ``4`` the document or the choice was refused, ``5`` an unexpected
failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.document_reader import DocumentReadError
from core.voice_moods import mood_summaries
from core.voice_studio_profile import source_names, studio_title
from gui.voice_studio import VoiceStudioService

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENGINE = 3
EXIT_REFUSED = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-studio", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List the four speech-engine studios.")

    moods = sub.add_parser("moods", help="List every emotional state a voice can take.")
    moods.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    status = sub.add_parser("status", help="Report whether one engine can read aloud.")
    status.add_argument("engine", choices=source_names(), help="The engine studio.")

    voices = sub.add_parser("voices", help="List the voices one engine really offers.")
    voices.add_argument("engine", choices=source_names(), help="The engine studio.")
    voices.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    voice = sub.add_parser("voice", help="Choose (and remember) a studio's reading voice.")
    voice.add_argument("engine", choices=source_names(), help="The engine studio.")
    voice.add_argument("key", help="The voice key to use (see `voices`).")

    mood = sub.add_parser("mood", help="Choose (and remember) a studio's mood.")
    mood.add_argument("engine", choices=source_names(), help="The engine studio.")
    mood.add_argument("key", help="The mood key to use (see `moods`).")

    logo = sub.add_parser("logo", help="Replace a studio's logo with an image.")
    logo.add_argument("engine", choices=source_names(), help="The engine studio.")
    logo.add_argument("image", help="Path to the image to use.")

    name = sub.add_parser("name", help="Rename a studio.")
    name.add_argument("engine", choices=source_names(), help="The engine studio.")
    name.add_argument("value", help="The new display name.")

    say = sub.add_parser("say", help="Read a document (or text) aloud.")
    say.add_argument("engine", choices=source_names(), help="The engine studio.")
    say.add_argument("document", nargs="?", default="", help="Path to a document to read.")
    say.add_argument("--text", default="", help="Read this text instead of a document.")
    say.add_argument("--write-wav", default="", help="Also save the speech to this WAV file.")
    say.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _service(engine: str) -> VoiceStudioService:
    return VoiceStudioService(engine)


def run_list(_args: argparse.Namespace) -> int:
    print("Speech-engine studios (each reads a document aloud with its own voice and mood):")
    for source in source_names():
        service = VoiceStudioService(source)
        marker = "ready" if service.reading_available() else "no engine installed"
        print(f"  {source:<12} {studio_title(source):<38} {marker}")
    return EXIT_OK


def run_moods(args: argparse.Namespace) -> int:
    rows = mood_summaries()
    if args.json:
        print(json.dumps(
            [{"key": key, "label_en": en, "label_ar": ar, "description": desc}
             for key, en, ar, desc in rows],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print("Moods (choose with `python -m cli voice-studio mood <engine> <key>`):")
    for key, label_en, label_ar, description in rows:
        print(f"  {key:<14} {label_en:<22} {label_ar:<16} {description}")
    return EXIT_OK


def run_status(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    print(f"{service.title}: {service.blurb()}")
    print(service.reading_status())
    print(service.describe())
    print(service.mood_support_note())
    return EXIT_OK


def run_voices(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    current = service.voice_key()
    rows = []
    for voice_group in service.voice_groups():
        for voice in voice_group.voices:
            rows.append({
                "key": voice.key,
                "label": voice.label() if hasattr(voice, "label") else voice.label_en,
                "group": voice_group.label,
                "gender": getattr(voice, "gender", ""),
                "chosen": voice.key == current,
            })
    for _key, label_en, catalog_group in service.catalog_voices():
        for voice in catalog_group:
            rows.append({
                "key": voice.key,
                "label": voice.label_en,
                "group": label_en,
                "gender": getattr(voice, "gender", ""),
                "chosen": voice.key == current,
            })
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print(f"{service.title} reports no voice here. {service.reading_status()}")
        return EXIT_OK
    print(f"Voices {service.title} really offers (chosen: {current}):")
    last_group = ""
    for row in rows:
        if row["group"] != last_group:
            print(f"  {row['group']}:")
            last_group = row["group"]
        tick = "●" if row["chosen"] else "○"
        gender = f"  ({row['gender']})" if row["gender"] else ""
        print(f"    {tick} {row['key']:<28} {row['label']}{gender}")
    return EXIT_OK


def run_voice(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    try:
        chosen = service.set_voice(args.key)
    except ValueError as error:
        print(str(error))
        return EXIT_REFUSED
    print(f"{service.title} will read with {service.voice_label(chosen)} (remembered).")
    return EXIT_OK


def run_mood(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    try:
        chosen = service.set_mood(args.key)
    except ValueError as error:
        print(str(error))
        return EXIT_REFUSED
    print(
        f"{service.title} mood set to {service.mood_label(chosen)} (remembered, applied at once)."
    )
    return EXIT_OK


def run_logo(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    try:
        path = service.set_logo(args.image)
    except ValueError as error:
        print(str(error))
        return EXIT_REFUSED
    print(f"{service.title} logo set to {path}.")
    return EXIT_OK


def run_name(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    name = service.set_name(args.value)
    print(f"{studio_title(args.engine)} is now called {name}.")
    return EXIT_OK


def run_say(args: argparse.Namespace) -> int:
    service = _service(args.engine)
    if not service.reading_available():
        print(f"{service.title} cannot read aloud here. {service.reading_status()}")
        return EXIT_ENGINE
    text = args.text
    if not text and args.document:
        try:
            document = service.load_document(args.document)
        except DocumentReadError as error:
            print(f"Could not read the document: {error}")
            return EXIT_REFUSED
        print(f"Loaded {document.summary()}", file=sys.stderr)
    elif not text:
        print("Give a document path or --text to read.")
        return EXIT_USAGE
    if args.write_wav:

        target = service.save_speech(args.write_wav, text or None)
        print(f"Saved the speech to {target}", file=sys.stderr)
    report = service.read_aloud(text or None)
    if args.json:
        print(json.dumps(
            {"ok": report.ok, "chunks": report.chunks, "characters": report.characters,
             "message": report.message},
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK if report.ok else EXIT_ENGINE
    print(report.summary())
    return EXIT_OK if report.ok else EXIT_ENGINE


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "list": run_list,
        "moods": run_moods,
        "status": run_status,
        "voices": run_voices,
        "voice": run_voice,
        "mood": run_mood,
        "logo": run_logo,
        "name": run_name,
        "say": run_say,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except DocumentReadError as error:
        print(f"Could not read the document: {error}")
        return EXIT_REFUSED
    except Exception as error:
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
