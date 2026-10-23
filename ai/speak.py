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
man, woman, boy, and girl voice and its key. When the OS reports no named voice at
all it says so plainly (with the Windows reason, if any) rather than printing only
the catalog and looking as though the invented voices are all this machine has.
The engine itself
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
    windows_voice_report,
)
from core.voices import model_voices, voice_sections


def _list_voices(reader: TextReader) -> int:
    print(f"Speech engine: {reader.engine_name}  ({reader.describe()})")
    installed = system_voices()
    if installed:
        print("\nInstalled voices (the real voices on this machine; use the key with --voice).")
        print("These installed voices are what the system really speaks with.")
    else:
        # No named voice was reported. Say so plainly (with the reason on Windows)
        # rather than listing only the synthetic catalog and looking like the
        # invented voices are all this machine has.
        print("\nInstalled voices: none were reported by this machine's speech engine.")
        note = windows_voice_report()
        if note:
            print(f"  {note}")
        print("The synthetic catalog below still works, but it is engine settings,")
        print("not the voices installed in the OS.")
    print("\nReading voices (grouped by the source each came from; use the key with --voice):")
    for section in voice_sections(system=installed, model=model_voices()):
        print(f"\n{section.title}:")
        for header, group in section.groups:
            print(f"  {header}:")
            for voice in group:
                gender = f" {voice.gender}" if getattr(voice, "gender", "") else ""
                label = voice.label() if hasattr(voice, "label") else voice.label_en
                detail = voice.language if hasattr(voice, "language") else voice.category
                print(f"    {voice.key:<34} {label}{gender}  [{detail}]")
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
