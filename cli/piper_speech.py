"""Manage the local Piper neural voices from the command line.

    python -m cli piper status
    python -m cli piper models
    python -m cli piper download --approve                 # fetch the default voice (~63 MB)
    python -m cli piper download en_US-amy-medium --approve
    python -m cli piper say "Hello there" --write-wav out.wav
    python -m cli piper say "مرحبا" --voice woman-1

The headless twin of the window's Piper controls. It drives the *same*
:mod:`core.piper_speech` engine the GUI and every other mouth use, so the CLI and
the window can never disagree about what is installed. Piper runs **locally**:
nothing is sent off the machine, and the voices download only through the
permission + human-approval gates (``--approve``).

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the runtime or a voice
is missing, ``4`` a download was refused, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.piper_speech import (
    PIPER_ENGINE_NAME,
    PiperUnavailable,
    download_model,
    piper_status,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from core.speech import load_synthesizer
from core.voices import detect_language, find_voice


def _status() -> int:
    status = piper_status()
    print("Piper (local neural speech)")
    print(f"  runtime installed: {'yes' if status.extra_installed else 'no'}")
    print(f"  voice cached:      {'yes' if status.model_cached else 'no'}")
    if status.model_file is not None:
        print(f"  voice file:        {status.model_file}")
    print(f"  next step:         {status.hint}")
    return 0 if status.available else 3


def _models() -> int:
    print("Fetchable Piper voices (use the key with `download`):")
    for key, title, language, size_mb, summary in spec_summaries():
        print(f"  {key:<28} {title}  [{language}, ~{size_mb} MB]")
        print(f"      {summary}")
    return 0


def _download(spec_key: str | None, approve: bool) -> int:
    if not approve:
        spec = resolve_spec(spec_key)
        print(
            f"Refused: downloading Piper voice {spec.key!r} needs approval. "
            "Re-run with --approve."
        )
        return 4
    try:
        directory = download_model(spec_key, approved=True)
    except (PiperUnavailable, ValueError) as error:
        print(str(error))
        return 3
    print(f"Downloaded Piper voice into: {directory}")
    return 0


def _say(
    text: str,
    spec_key: str | None,
    voice_key: str | None,
    write_wav: Path | None,
) -> int:
    voice = find_voice(voice_key) if voice_key else None
    try:
        synthesizer = load_synthesizer(PIPER_ENGINE_NAME, voice=voice, spec_key=spec_key)
    except ValueError as error:
        print(str(error))
        return 3
    try:
        clip = synthesizer.synthesize(text)
    except (ValueError, PiperUnavailable) as error:
        print(str(error))
        return 3
    code = detect_language(text)
    print(f"Language: {'Arabic' if code == 'ar' else 'English'}")
    print(f"Audio duration: {clip.duration_seconds:.2f} s")
    if write_wav is not None:
        clip.write_wav(write_wav)
        print(f"Wrote audio file: {write_wav}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piper",
        description="Manage the local Piper neural text-to-speech voices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="report what is installed")
    sub.add_parser("models", help="list the fetchable voices")

    download = sub.add_parser("download", help="fetch a voice's weights (needs --approve)")
    download.add_argument("voice", nargs="?", help=f"a voice key, one of: {', '.join(spec_names())}")
    download.add_argument("--approve", action="store_true", help="grant human approval")

    say = sub.add_parser("say", help="synthesize speech from text")
    say.add_argument("text", help="the text to read aloud")
    say.add_argument("--voice", help="a project catalog voice key (man-1, woman-2, ...)")
    say.add_argument("--spec", dest="spec_key", help="a Piper voice key (overrides --voice)")
    say.add_argument("--write-wav", type=Path, help="also write the audio to this WAV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return _status()
    if args.command == "models":
        return _models()
    if args.command == "download":
        return _download(args.voice, args.approve)
    if args.command == "say":
        return _say(args.text, args.spec_key, args.voice, args.write_wav)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
