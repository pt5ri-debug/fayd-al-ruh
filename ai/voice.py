"""Run one voice turn through the assistant.

    python -m ai.voice "What is the relationship between porosity and permeability?"
    python -m ai.voice --write-wav out.wav "permeability"

This is a text-driven voice loop: the transcript stands in for a live microphone
and the synthesized clip is a real 16-bit WAV. It exercises the exact pipeline a
microphone and a speech engine would drive, so a real ASR/TTS backend can be
dropped in without changing the assistant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.assistant import build_default_assistant, bulk_knowledge_available
from core.speech import (
    SAMPLE_RATE,
    AudioClip,
    VoiceAssistant,
    load_recognizer,
    load_synthesizer,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One voice turn with the assistant.")
    parser.add_argument("text", nargs="+", help="the spoken question (transcript)")
    parser.add_argument("--asr", default="transcript")
    parser.add_argument("--tts", default="silent")
    parser.add_argument("--write-wav", type=Path, default=None)
    args = parser.parse_args(argv)

    transcript = " ".join(args.text)
    recognizer = load_recognizer(args.asr, transcript=transcript)
    synthesizer = load_synthesizer(args.tts)
    voice = VoiceAssistant(
        build_default_assistant(bulk=bulk_knowledge_available()),
        recognizer=recognizer,
        synthesizer=synthesizer,
    )

    turn = voice.respond_to_speech(AudioClip.silence(0.5))
    print(f"Heard: {turn.heard}")
    print(f"Answer: {turn.answer_text}")
    if turn.citations:
        print("Sources:")
        for citation in turn.citations:
            print(f"  - {citation}")
    print(f"Audio duration: {turn.spoken.duration_seconds:.2f} s at {SAMPLE_RATE} Hz")
    if args.write_wav is not None:
        turn.spoken.write_wav(args.write_wav)
        print(f"Wrote audio file: {args.write_wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
