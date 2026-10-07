"""Jarvis mode from the command line.

    python -m cli jarvis "what is permeability?"       # one turn, answer with citations
    python -m cli jarvis "list the files in notes/"    # one turn, run a sandboxed tool
    python -m cli jarvis                               # interactive text loop (type 'quit' to exit)

Voice is opt-in through the system: pass ``--play`` to speak answers through the
speakers, ``--tts auto`` to pick an installed engine (``espeak``/``say``/...),
``--asr auto`` to transcribe from ``whisper`` when it is present, and ``--mic``
to record from the microphone (``arecord``/``sox``). With nothing installed,
Jarvis still runs as a complete *text* loop -- the same fallback philosophy as
the rest of the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.assistant import build_default_assistant
from core.command_router import NaturalLanguageRouter
from core.jarvis import JarvisMode, JarvisTurn
from core.paths import data_dir
from core.preferences import UserPreferences
from core.speech import (
    MicrophoneRecorder,
    Speaker,
    VoiceAssistant,
    load_recognizer,
    load_synthesizer,
)
from core.tool_agent import ToolAgent
from core.tools import build_sandbox
from core.wake_word import WakeWordDetector


def _default_sandbox_root() -> Path:
    return data_dir() / "jarvis-sandbox"


def _build_mode(args: argparse.Namespace) -> JarvisMode:
    root = Path(args.root).expanduser() if args.root else _default_sandbox_root()
    sandbox = build_sandbox(root, read_only=not args.read_write)
    assistant = build_default_assistant()
    router = NaturalLanguageRouter(sandbox)
    preferences = UserPreferences(data_dir())
    wake = WakeWordDetector(phrases=(preferences.record.wake_word, "hey " + preferences.record.wake_word, "يا جارفيس"))
    recognizer = load_recognizer(args.asr)
    synthesizer = load_synthesizer(args.tts)
    voice = VoiceAssistant(assistant, recognizer=recognizer, synthesizer=synthesizer)
    return JarvisMode(
        assistant,
        router=router,
        agent=ToolAgent(sandbox),
        voice=voice,
        wake=wake,
        speaker=Speaker() if args.play else None,
    )


def _print_turn(turn: JarvisTurn, *, as_json: bool) -> None:
    if as_json:
        import json

        payload = {
            "heard": turn.heard,
            "woke": turn.woke,
            "routed": turn.routed,
            "response": turn.response,
            "citations": list(turn.citations),
        }
        if turn.agent_run is not None:
            payload["steps"] = [
                {"tool": step.call.tool, "ok": step.result.ok, "output": step.result.output, "error": step.result.error}
                for step in turn.agent_run.steps
            ]
        print(json.dumps(payload, ensure_ascii=False))
        return

    if turn.routed == "ignored":
        print("(no wake word; ignored)")
        return
    if turn.routed == "silence":
        print("(no speech detected)")
        return
    print(f"Heard  : {turn.heard}")
    print(f"Routed : {turn.routed}")
    if turn.agent_run is not None:
        for step in turn.agent_run.steps:
            status = "ok " if step.result.ok else "ERR"
            detail = step.result.output if step.result.ok else step.result.error
            print(f"  [{status}] {step.call.tool}: {detail}")
    else:
        print(f"Answer : {turn.response}")
        if turn.citations:
            print("Sources:")
            for citation in turn.citations:
                print(f"  - {citation}")


def run_loop(mode: JarvisMode, *, as_json: bool, require_wake: bool, speak: bool) -> int:
    print("Jarvis mode. Type a request, or 'quit' to exit.")
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().casefold() in {"quit", "exit", "q", ""}:
            return 0
        turn = mode.handle(line, require_wake=require_wake, speak=speak)
        _print_turn(turn, as_json=as_json)
        playback = mode.play(turn)
        if playback:
            print(f"(audio: {playback})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis mode: listen, understand, act, speak.")
    parser.add_argument("utterance", nargs="*", help="one request; omit to enter the interactive loop")
    parser.add_argument("--asr", default="auto", help="ASR engine: auto, transcript, whisper")
    parser.add_argument("--tts", default="auto", help="TTS engine: auto, silent, espeak, say, flite")
    parser.add_argument("--root", default=None, help="sandbox root directory for tools")
    parser.add_argument("--read-write", action="store_true", help="allow writing tools in the sandbox")
    parser.add_argument("--wake", action="store_true", help="require the wake word before responding")
    parser.add_argument("--play", action="store_true", help="play spoken answers through the speakers")
    parser.add_argument("--mic", action="store_true", help="record one clip from the microphone")
    parser.add_argument("--write-wav", type=Path, default=None, help="write the spoken answer to a WAV file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        mode = _build_mode(args)
    except ValueError as error:
        print(f"Could not start Jarvis mode: {error}")
        return 1
    require_wake = args.wake
    speak = args.play or args.write_wav is not None

    if args.mic:
        recorder = MicrophoneRecorder()
        if not recorder.available():
            print("No microphone recorder found (install arecord or sox).")
            return 1
        if mode.voice.recognizer.name == "transcript":
            print("No speech recognizer found (install whisper, or use a text request).")
            return 1
        try:
            turn = mode.handle_audio(recorder.record(), require_wake=require_wake, speak=speak)
        except ValueError as error:
            print(f"Could not process the recording: {error}")
            return 1
        _print_turn(turn, as_json=args.json)
        playback = mode.play(turn)
        if playback:
            print(f"(audio: {playback})")
        return 0

    if args.utterance:
        utterance = " ".join(args.utterance)
        turn = mode.handle(utterance, require_wake=require_wake, speak=speak)
        _print_turn(turn, as_json=args.json)
        if args.write_wav is not None and turn.spoken is not None:
            turn.spoken.write_wav(args.write_wav)
            print(f"Wrote audio file: {args.write_wav}")
        playback = mode.play(turn)
        if playback:
            print(f"(audio: {playback})")
        return 0

    if args.write_wav is not None:
        print("--write-wav needs a single request, not the interactive loop.")
        return 1
    return run_loop(mode, as_json=args.json, require_wake=require_wake, speak=speak)


if __name__ == "__main__":
    raise SystemExit(main())
