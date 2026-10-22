"""Manage the local VoiceTut-TTS Arabic voice (over the audio.cpp engine).

    python -m cli audio status
    python -m cli audio models
    python -m cli audio install-engine --approve     # fetch a prebuilt engine (~50 MB)
    python -m cli audio download --approve           # fetch the model (~1.3 GB)
    python -m cli audio download --speakers --approve
    python -m cli audio voices                       # the model's own voices
    python -m cli audio say "إزاي حالك النهارده؟" --voice model:audio.cpp:Sarah
    python -m cli audio say "hello" --write-wav out.wav

The headless twin of the window's voice menu. It drives the *same*
:mod:`core.audiocpp_speech` engine the GUI and every other mouth use, so the CLI
and the window can never disagree about what is installed. VoiceTut-TTS runs
**locally**: nothing is sent off the machine, and both the engine and the model
download only through the permission + human-approval gates (``--approve``).

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the engine or a model is
missing, ``4`` a download was refused, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.audiocpp_speech import (
    AUDIOCPP_ENGINE_NAME,
    AudioCppUnavailable,
    audiocpp_status,
    download_model,
    download_speakers,
    install_engine,
    spec_names,
    spec_summaries,
    voice_list,
)
from core.speech import load_synthesizer
from core.voices import detect_language, find_model_voice


def _status() -> int:
    status = audiocpp_status()
    print("audio.cpp (local VoiceTut-TTS speech engine)")
    print(f"  engine installed:  {'yes' if status.engine_installed else 'no'}")
    if status.cli_path is not None:
        print(f"  engine binary:     {status.cli_path}")
    print(f"  model cached:      {'yes' if status.model_cached else 'no'}")
    if status.model_file is not None:
        print(f"  model file:        {status.model_file}")
    print(f"  built-in voices:   {status.voices}")
    print(f"  next step:         {status.hint}")
    return 0 if status.available else 3


def _models() -> int:
    print("Fetchable VoiceTut-TTS packages (use the key with `download`):")
    for key, title, precision, size_mb, summary in spec_summaries():
        print(f"  {key:<16} {title}  [{precision}, ~{size_mb} MB]")
        print(f"      {summary}")
    return 0


def _voices() -> int:
    voices = voice_list()
    if not voices:
        print("No VoiceTut-TTS voices are available yet.")
        print(f"  {audiocpp_status().hint}")
        return 3
    print(f"VoiceTut-TTS voices ({len(voices)}), spoken by name:")
    for voice in voices:
        gender = f", {voice.gender}" if voice.gender else ""
        print(f"  {voice.key:<32} {voice.name} [{voice.language}{gender}]")
    return 0


def _install(approve: bool) -> int:
    if not approve:
        print(
            "Refused: installing the audio.cpp engine downloads a native binary and "
            "needs approval. Re-run with --approve."
        )
        return 4
    try:
        directory = install_engine(approved=True)
    except AudioCppUnavailable as error:
        print(str(error))
        return 3
    print(f"Installed the audio.cpp engine into: {directory}")
    return 0


def _download(spec_key: str | None, approve: bool, speakers: bool) -> int:
    if not approve:
        what = "the built-in reference voices" if speakers else "the VoiceTut-TTS model"
        print(f"Refused: downloading {what} needs approval. Re-run with --approve.")
        return 4
    try:
        if speakers:
            directory = download_speakers(approved=True)
        else:
            directory = download_model(spec_key, approved=True)
    except (AudioCppUnavailable, ValueError) as error:
        print(str(error))
        return 3
    print(f"Downloaded into: {directory}")
    return 0


def _say(
    text: str,
    voice_key: str | None,
    write_wav: Path | None,
    steps: int,
) -> int:
    voice = None
    if voice_key:
        voice = find_model_voice(voice_key)
        if voice is None:
            print(
                f"Unknown or unavailable voice {voice_key!r}. "
                "Run `python -m cli audio voices` to list the installed ones."
            )
            return 3
    try:
        synthesizer = load_synthesizer(AUDIOCPP_ENGINE_NAME, voice=voice, steps=steps)
    except ValueError as error:
        print(str(error))
        return 3
    try:
        clip = synthesizer.synthesize(text)
    except (ValueError, AudioCppUnavailable) as error:
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
        prog="audio",
        description="Manage the local VoiceTut-TTS Arabic voice (over audio.cpp).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="report what is installed")
    sub.add_parser("models", help="list the fetchable packages")
    sub.add_parser("voices", help="list the model's own voices")

    install = sub.add_parser("install-engine", help="fetch a prebuilt engine (needs --approve)")
    install.add_argument("--approve", action="store_true", help="grant human approval")

    download = sub.add_parser("download", help="fetch the model or its voices (needs --approve)")
    download.add_argument("spec", nargs="?", help=f"a package key, one of: {', '.join(spec_names())}")
    download.add_argument("--speakers", action="store_true", help="fetch the built-in voices instead")
    download.add_argument("--approve", action="store_true", help="grant human approval")

    say = sub.add_parser("say", help="synthesize speech from text")
    say.add_argument("text", help="the text to read aloud")
    say.add_argument("--voice", help="a model voice key (model:audio.cpp:Sarah, ...)")
    say.add_argument("--steps", type=int, default=8, help="diffusion steps (default 8)")
    say.add_argument("--write-wav", type=Path, help="also write the audio to this WAV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return _status()
    if args.command == "models":
        return _models()
    if args.command == "voices":
        return _voices()
    if args.command == "install-engine":
        return _install(args.approve)
    if args.command == "download":
        if args.speakers and args.spec:
            parser.error("--speakers and a package key cannot be combined")
        return _download(args.spec, args.approve, args.speakers)
    if args.command == "say":
        return _say(args.text, args.voice, args.write_wav, args.steps)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
