"""Read text aloud from the command line, with a chosen voice.

    python -m ai.speak list
    python -m ai.speak "السلام عليكم" --voice man-1
    python -m ai.speak "Hello there" --voice girl-2 --write-wav out.wav
    python -m ai.speak --install-engine        # install a real speech engine (system package)

This is the headless twin of the window's "Read aloud" control: the same
:class:`~core.reading.TextReader`, the same :mod:`core.voices` catalog, and the
same system speech engine. ``list`` prints the **real** voices the machine reports
first (the Windows OneCore/SAPI or macOS ``say`` voices, by their own names, e.g.
``Microsoft Hoda``) -- those are what the engine will actually speak with -- then
the synthetic ``espeak-ng`` catalog grouped by category so a user can see every
man, woman, boy, and girl voice and its key. The engine itself
(``espeak-ng``) is an ordinary system program, not a Python package, so it is
installed as an OS package -- ``--install-engine`` runs the platform's package
manager, and says plainly what to do when it cannot.
"""

from __future__ import annotations

import argparse

from core.reading import TextReader
from core.speech import (
    install_speech_engine,
    speech_available,
    speech_install_plan,
    system_voices,
)
from core.voices import voice_categories, voices_for


def _list_voices(reader: TextReader) -> int:
    print(f"Speech engine: {reader.engine_name}  ({reader.describe()})")
    installed = system_voices()
    if installed:
        # The real voices the machine reports, grouped by language. This is the
        # honest list: these are the voices the engine will actually speak with.
        print("\nInstalled voices (the real voices on this machine; use the key with --voice):\n")
        groups: dict[str, list] = {}
        for voice in installed:
            groups.setdefault(voice.language_code, []).append(voice)
        for code in sorted(groups, key=lambda c: (c != "ar", c)):
            voices = groups[code]
            label = voices[0].language_label()
            print(f"{label} ({code}):")
            for voice in voices:
                gender = f" {voice.gender}" if voice.gender else ""
                print(f"  {voice.key:<40} {voice.name}{gender}  [{voice.language}]")
            print()
        print("These installed voices are what the system really speaks with.")
        print("A synthetic catalog voice is also available (--voice man-1 ...).\n")
    print("Reading voices (use the key with --voice):\n")
    for category, label_en, label_ar in voice_categories():
        print(f"{label_en} / {label_ar}:")
        for catalog_voice in voices_for(category):
            print(
                f"  {catalog_voice.key:<9} {catalog_voice.label_en:<8} {catalog_voice.label_ar:<8} "
                f"variant={catalog_voice.variant} pitch={catalog_voice.pitch} "
                f"speed={catalog_voice.speed}"
            )
        print()
    return 0


def _install_engine() -> int:
    """Install a real speech engine as a system package, reporting honestly."""
    if speech_available():
        print("A speech engine is already installed.")
        return 0
    plan = speech_install_plan()
    if plan.can_install:
        print(f"Installing a speech engine: {' '.join(plan.command)}")
    result = install_speech_engine()
    print(result.message)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read text aloud with a chosen voice.")
    parser.add_argument("text", nargs="*", help="the text to read (or 'list' to show voices)")
    parser.add_argument("--voice", default=None, help="a voice key from the catalog (see 'list')")
    parser.add_argument("--engine", default="auto", help="TTS engine: auto, espeak-ng, espeak, ...")
    parser.add_argument("--language", default=None, help="pin the spoken language (ar/en)")
    parser.add_argument("--write-wav", default=None, help="also save the audio to this file")
    parser.add_argument("--no-play", action="store_true", help="render the audio but do not play it")
    parser.add_argument(
        "--install-engine",
        action="store_true",
        help="install a real speech engine (espeak-ng) as a system package",
    )
    args = parser.parse_args(argv)

    if args.install_engine:
        return _install_engine()

    reader = TextReader(engine=args.engine, voice=args.voice, language=args.language)

    if not args.text or args.text == ["list"]:
        return _list_voices(reader)

    text = " ".join(args.text)
    if not reader.available():
        plan = speech_install_plan()
        print("No speech engine is installed, so there is no real voice to hear.")
        print(plan.hint)
        print("Or run `python -m cli speak --install-engine`.")
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
