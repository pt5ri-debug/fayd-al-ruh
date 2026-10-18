"""Manage the Qwen3 speech models (ASR and TTS) from the command line.

    python -m cli qwen-asr status
    python -m cli qwen-asr models
    python -m cli qwen-asr download --approve          # fetch the weights (~3.4 GB)
    python -m cli qwen-asr transcribe audio.wav
    python -m cli qwen-tts status
    python -m cli qwen-tts models
    python -m cli qwen-tts download --approve
    python -m cli qwen-tts say "Hello there" --write-wav out.wav

The headless twin of the window's Qwen3 speech controls. It drives the *same*
:mod:`core.qwen_speech` engines the GUI and every other mouth use, so the CLI and
the window can never disagree about what is installed. The models run **locally**:
nothing is sent off the machine, and the weights download only through the
permission + human-approval gates (``--approve``).

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the runtime or model is
missing, ``4`` a download was refused, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.qwen_speech import (
    ASR_ENGINE_NAME,
    QWEN_TTS_LANGUAGES,
    TTS_ENGINE_NAME,
    QwenSpeechUnavailable,
    asr_spec_summaries,
    download_model,
    qwen_asr_status,
    qwen_tts_status,
    resolve_asr_spec,
    resolve_tts_spec,
    tts_spec_summaries,
)
from core.speech import AudioClip, load_recognizer, load_synthesizer
from core.voices import detect_language


def _status(kind: str) -> int:
    status = qwen_asr_status() if kind == "asr" else qwen_tts_status()
    label = "Qwen3-ASR" if kind == "asr" else "Qwen3-TTS"
    print(f"{label}")
    print(f"  runtime installed: {'yes' if status.extra_installed else 'no'}")
    print(f"  model cached:      {'yes' if status.model_cached else 'no'}")
    if status.model_dir is not None:
        print(f"  model folder:      {status.model_dir}")
    if kind == "tts":
        print(f"  speaks:            {', '.join(QWEN_TTS_LANGUAGES)}")
        print("  (Qwen3-TTS does not speak Arabic; the system engine reads Arabic aloud.)")
    print(f"  next step:         {status.hint}")
    return 0 if status.available else 3


def _models(kind: str) -> int:
    rows = asr_spec_summaries() if kind == "asr" else tts_spec_summaries()
    for key, title, size, summary in rows:
        print(f"{key:<18} {title:<44} ~{size:.1f} GB")
        print(f"  {summary}")
    return 0


def _download(kind: str, spec_key: str | None, approved: bool) -> int:
    spec = resolve_asr_spec(spec_key) if kind == "asr" else resolve_tts_spec(spec_key)
    print(f"Downloading {spec.repo} (~{spec.approx_size_gb:.1f} GB)…")
    try:
        directory = download_model(kind, spec.key, approved=approved)
    except QwenSpeechUnavailable as error:
        print(str(error))
        return 4 if not approved else 3
    print(f"Downloaded to {directory}")
    return 0


def _transcribe(path: Path, model: str | None, language: str | None) -> int:
    try:
        recognizer = load_recognizer(
            ASR_ENGINE_NAME,
            model_dir=None,
            language=language,
        )
    except ValueError as error:
        print(str(error))
        return 3
    clip = AudioClip.read_wav(path)
    try:
        text = recognizer.transcribe(clip)
    except ValueError as error:
        print(str(error))
        return 5
    print(text)
    return 0


def _say(text: str, model: str | None, speaker: str | None, write_wav: Path | None) -> int:
    try:
        synthesizer = load_synthesizer(TTS_ENGINE_NAME, speaker=speaker)
    except ValueError as error:
        print(str(error))
        return 3
    code = detect_language(text)
    try:
        clip = synthesizer.synthesize(text)
    except ValueError as error:
        print(str(error))
        return 5
    if code == "ar":
        print("(Arabic is read by the system engine; Qwen3-TTS has no Arabic.)")
    print(f"Audio duration: {clip.duration_seconds:.2f} s")
    if write_wav is not None:
        clip.write_wav(write_wav)
        print(f"Wrote audio file: {write_wav}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-speech",
        description="Manage the local Qwen3 speech models (ASR and TTS).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for kind in ("qwen-asr", "qwen-tts"):
        group = sub.add_parser(kind, help=f"{kind} engine management")
        actions = group.add_subparsers(dest="action", required=True)
        actions.add_parser("status", help="report what is installed")
        actions.add_parser("models", help="list the fetchable models")
        download = actions.add_parser("download", help="fetch the model weights")
        download.add_argument("--model", default=None, help="which catalog model to fetch")
        download.add_argument(
            "--approve", action="store_true", help="record a human approval for the download"
        )
        if kind == "qwen-asr":
            run = actions.add_parser("transcribe", help="transcribe a WAV file")
            run.add_argument("file", type=Path)
            run.add_argument("--model", default=None)
            run.add_argument("--language", default=None, help="pin the language (e.g. Arabic)")
        else:
            run = actions.add_parser("say", help="synthesize speech from text")
            run.add_argument("text", nargs="+")
            run.add_argument("--model", default=None)
            run.add_argument("--speaker", default=None, help="a built-in speaker name")
            run.add_argument("--write-wav", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    kind = "asr" if args.command == "qwen-asr" else "tts"
    if args.action == "status":
        return _status(kind)
    if args.action == "models":
        return _models(kind)
    if args.action == "download":
        return _download(kind, args.model, args.approve)
    if args.action == "transcribe":
        return _transcribe(args.file, args.model, args.language)
    if args.action == "say":
        return _say(" ".join(args.text), args.model, args.speaker, args.write_wav)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
