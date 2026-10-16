"""Read text aloud from the command line, with a chosen voice.

    python -m ai.speak list
    python -m ai.speak "السلام عليكم" --voice man-1
    python -m ai.speak "Hello there" --voice girl-2 --write-wav out.wav

This is the headless twin of the window's "Read aloud" control: the same
:class:`~core.reading.TextReader`, the same :mod:`core.voices` catalog, and the
same system speech engine. ``list`` prints the catalog grouped by category so a
user can see every man, woman, boy, and girl voice and its key.
"""

from __future__ import annotations

import argparse

from core.reading import TextReader
from core.voices import voice_categories, voices_for


def _list_voices(reader: TextReader) -> int:
    print(f"Speech engine: {reader.engine_name}  ({reader.describe()})")
    print("\nReading voices (use the key with --voice):\n")
    for category, label_en, label_ar in voice_categories():
        print(f"{label_en} / {label_ar}:")
        for voice in voices_for(category):
            print(
                f"  {voice.key:<9} {voice.label_en:<8} {voice.label_ar:<8} "
                f"variant={voice.variant} pitch={voice.pitch} speed={voice.speed}"
            )
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read text aloud with a chosen voice.")
    parser.add_argument("text", nargs="*", help="the text to read (or 'list' to show voices)")
    parser.add_argument("--voice", default=None, help="a voice key from the catalog (see 'list')")
    parser.add_argument("--engine", default="auto", help="TTS engine: auto, espeak-ng, espeak, ...")
    parser.add_argument("--language", default=None, help="pin the spoken language (ar/en)")
    parser.add_argument("--write-wav", default=None, help="also save the audio to this file")
    parser.add_argument("--no-play", action="store_true", help="render the audio but do not play it")
    args = parser.parse_args(argv)

    reader = TextReader(engine=args.engine, voice=args.voice, language=args.language)

    if not args.text or args.text == ["list"]:
        return _list_voices(reader)

    text = " ".join(args.text)
    if not reader.available():
        print("No speech engine is installed; install espeak-ng (or espeak) to hear a voice.")
        return 2
    if args.no_play:
        clip = reader.render(text)
    else:
        clip = reader.speak(text)
    print(reader.describe())
    print(f"Voice: {reader.voice.key} ({reader.voice.label_en} / {reader.voice.label_ar})")
    print(f"Audio: {clip.duration_seconds:.2f} s{' (silent)' if clip.is_silent else ''}")
    if args.write_wav:
        target = reader.save(text, args.write_wav)
        print(f"Wrote audio file: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
