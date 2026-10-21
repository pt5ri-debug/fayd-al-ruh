"""Jarvis mode from the command line.

    python -m cli jarvis "what is permeability?"       # one turn, answer with citations
    python -m cli jarvis "list the files in notes/"    # one turn, run a sandboxed tool
    python -m cli jarvis                               # interactive text loop (type 'quit' to exit)

Jarvis keeps one conversation session per run, so follow-up questions stay in
context. A few spoken controls work alongside questions and tool commands:
"stop" mutes the voice, "speak" turns it back on, "repeat" replays the last
answer, "new conversation" starts over, "save" persists the session, "sessions"
lists saved ones, and "help" lists them all.

Sessions persist under the data directory. In the interactive loop, resume a
saved session with ``--session <id>`` (it autosaves after every turn), or list
what is stored with ``--list-sessions``.

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

from core.assistant import _WEAK_EVIDENCE, build_default_assistant, bulk_knowledge_available
from core.command_router import NaturalLanguageRouter
from core.composition import sources_block
from core.continuous_learning import ContinuousLearner
from core.conversation import ConversationMemory
from core.documents import build_document_store
from core.jarvis import HELP_TEXT, TEXT_HELP_TEXT, JarvisMode, JarvisTurn
from core.jarvis_controls import JarvisControls
from core.knowledge_update import KnowledgeUpdater
from core.paths import data_dir
from core.preferences import UserPreferences
from core.speech import (
    MicrophoneRecorder,
    Speaker,
    VoiceAssistant,
    load_recognizer,
    load_synthesizer,
    microphone_install_hint,
)
from core.tool_agent import ToolAgent
from core.tools import build_sandbox
from core.wake_word import WakeWordDetector
from storage.json_store import JsonStore


def _default_sandbox_root() -> Path:
    return data_dir() / "jarvis-sandbox"


def _memory() -> ConversationMemory:
    """A persistent session store under the data directory, or in-memory if it fails."""
    try:
        return ConversationMemory(JsonStore(data_dir() / "jarvis"))
    except (OSError, ValueError):
        return ConversationMemory()


def _documents():
    """A persistent knowledge store under the data directory, or in-memory.

    Learned statements are written here (through the assistant's on-learn hook)
    and reloaded at startup, so knowledge Jarvis gains today is still there
    tomorrow. On a read-only data directory it silently falls back to memory.
    """
    try:
        documents = build_document_store(store=JsonStore(data_dir() / "documents"))
        documents.load_all()
        return documents
    except (OSError, ValueError):
        return build_document_store()


def build_session(
    *,
    root: Path | None = None,
    read_write: bool = False,
    no_web: bool = False,
    web_threshold: float | None = None,
    no_feedback: bool = False,
    asr: str = "auto",
    tts: str = "auto",
    play: bool = False,
    text_only: bool = False,
) -> JarvisMode:
    """Assemble a fully-wired Jarvis session from explicit options.

    Shared by the ``jarvis`` and ``chat`` commands so both run the same loop over
    the same assistant, learner, and sandbox. Keeping the wiring in one place
    means a fix to any gate applies to both entry points. With ``text_only`` the
    session uses the transcript/silent speech engines and the text-worded help,
    so a voice-less conversation never talks about muting or replaying audio.
    """
    sandbox_root = root.expanduser() if root else _default_sandbox_root()
    sandbox = build_sandbox(sandbox_root, read_only=not read_write)
    documents = _documents()
    assistant = build_default_assistant(
        documents=documents,
        web_enabled=not no_web,
        web_threshold=web_threshold if web_threshold is not None else _WEAK_EVIDENCE,
        on_learn=documents.save,
        bulk=bulk_knowledge_available(),
    )
    learner = ContinuousLearner(KnowledgeUpdater(documents))
    router = NaturalLanguageRouter(sandbox)
    preferences = UserPreferences(data_dir())
    wake = WakeWordDetector(
        phrases=(preferences.record.wake_word, "hey " + preferences.record.wake_word, "يا جارفيس")
    )
    recognizer = load_recognizer("transcript" if text_only else asr)
    synthesizer = load_synthesizer("silent" if text_only else tts)
    voice = VoiceAssistant(assistant, recognizer=recognizer, synthesizer=synthesizer)
    return JarvisMode(
        assistant,
        router=router,
        agent=ToolAgent(sandbox),
        voice=voice,
        wake=wake,
        speaker=None if text_only else (Speaker() if play else None),
        memory=_memory(),
        controls=JarvisControls(),
        learner=None if no_feedback else learner,
        mute=text_only or not preferences.record.speak,
        help_text=TEXT_HELP_TEXT if text_only else HELP_TEXT,
    )


def _build_mode(args: argparse.Namespace) -> JarvisMode:
    return build_session(
        root=Path(args.root) if args.root else None,
        read_write=args.read_write,
        no_web=args.no_web,
        web_threshold=args.web_threshold,
        no_feedback=args.no_feedback,
        asr=args.asr,
        tts=args.tts,
        play=args.play,
    )


def _print_turn(turn: JarvisTurn, *, as_json: bool) -> None:
    if as_json:
        import json

        payload = {
            "heard": turn.heard,
            "woke": turn.woke,
            "routed": turn.routed,
            "intent": turn.intent,
            "response": turn.response,
            "citations": list(turn.citations),
            "used_web": turn.used_web,
            "learned": turn.learned,
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
    if turn.routed == "control":
        print(f"Control: {turn.intent}")
        print(f"Jarvis : {turn.response}")
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
        block = sources_block(turn.citations)
        if block:
            print(block)


def run_loop(mode: JarvisMode, *, as_json: bool, require_wake: bool, speak: bool, autosave: bool = False) -> int:
    if not as_json:
        print(f"Jarvis mode (session {mode.conversation.id}). Type a request, or 'quit' to exit.")
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
        if autosave:
            mode.save_session()
        playback = mode.play(turn)
        if playback:
            print(f"(audio: {playback})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis mode: listen, understand, act, speak.")
    parser.add_argument("utterance", nargs="*", help="one request; omit to enter the interactive loop")
    parser.add_argument(
        "--asr",
        default="auto",
        help="ASR engine: auto, transcript, whisper, qwen3-asr (local Qwen3-ASR model)",
    )
    parser.add_argument(
        "--tts",
        default="auto",
        help="TTS engine: auto, silent, espeak, say, flite, qwen3-tts (local Qwen3-TTS model)",
    )
    parser.add_argument("--root", default=None, help="sandbox root directory for tools")
    parser.add_argument("--read-write", action="store_true", help="allow writing tools in the sandbox")
    parser.add_argument("--wake", action="store_true", help="require the wake word before responding")
    parser.add_argument("--play", action="store_true", help="play spoken answers through the speakers")
    parser.add_argument("--mic", action="store_true", help="record one clip from the microphone")
    parser.add_argument("--write-wav", type=Path, default=None, help="write the spoken answer to a WAV file")
    parser.add_argument("--session", default=None, help="resume a saved session by id, or start a new one")
    parser.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--no-feedback", action="store_true", help="disable learning from confirm/correct controls")
    parser.add_argument("--no-web", action="store_true", help="answer only from internal knowledge; do not search the internet")
    parser.add_argument(
        "--web-threshold",
        type=float,
        default=None,
        help="top internal score below which Jarvis searches the web (0-1; higher searches more)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.list_sessions:
        saved = _memory().list_saved()
        if saved:
            print("Saved sessions:")
            for session_id in saved:
                print(f"  - {session_id}")
        else:
            print("No saved sessions yet.")
        return 0

    try:
        mode = _build_mode(args)
    except ValueError as error:
        print(f"Could not start Jarvis mode: {error}")
        return 1

    if args.session:
        if not mode.resume_session(args.session):
            print(f"No saved session '{args.session}'. Starting a new one.")
        elif not args.json:
            print(f"Resumed session {mode.conversation.id}.")
    require_wake = args.wake
    speak = args.play or args.write_wav is not None

    if args.mic:
        recorder = MicrophoneRecorder()
        if not recorder.available():
            print("No microphone recorder found. " + microphone_install_hint())
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
    return run_loop(mode, as_json=args.json, require_wake=require_wake, speak=speak, autosave=args.session is not None)


if __name__ == "__main__":
    raise SystemExit(main())
