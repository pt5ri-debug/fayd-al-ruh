"""CLI for the per-model chat windows: identity, voice, and model, per mouth.

    python -m cli model-chat mouths                 # the mouths that have a window
    python -m cli model-chat models ultron          # every model Ultron's window offers
    python -m cli model-chat profile natasha        # her saved name, logo, voice, model
    python -m cli model-chat set-model ultron qwen2.5-3b
    python -m cli model-chat set-voice natasha woman-2
    python -m cli model-chat set-name hakeem "الحكيم"
    python -m cli model-chat set-logo vision logo.png
    python -m cli model-chat clear-logo vision
    python -m cli model-chat ask ultron "who wrote Hamlet?"

This is the headless twin of the per-model chat window (``gui.model_chat_dialog``):
it drives the *same* :class:`gui.model_chat.ModelChatService`, so the CLI and the
window share one saved profile per mouth -- one name, one logo, one reading voice,
and one language model -- and no logic is reimplemented here.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the mouth's model is not
available, ``4`` the profile or logo could not be read or written, ``5`` an
unexpected failure.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from core import language_models as catalog
from core.mouth_profile import LOGO_SUFFIXES
from gui.model_chat import MOUTH_BLURBS, ModelChatService

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_PROFILE = 4
EXIT_FAILURE = 5

#: The mouths that have a dedicated window: every answering mouth.
WINDOW_MOUTHS: tuple[str, ...] = catalog.MOUTHS


def _mouth(value: str) -> str:
    """Validate a mouth name, raising ``ValueError`` for an unknown one."""
    wanted = (value or "").strip().lower()
    if wanted not in WINDOW_MOUTHS:
        raise ValueError(
            f"Unknown model {value!r}. Choose one of: {', '.join(WINDOW_MOUTHS)}."
        )
    return wanted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-chat", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("mouths", help="List the models that have their own chat window.")

    models = sub.add_parser("models", help="List the language models a window offers.")
    models.add_argument("mouth", help="The model's name (e.g. ultron, natasha).")
    models.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    profile = sub.add_parser("profile", help="Show a model's saved profile.")
    profile.add_argument("mouth", help="The model's name.")
    profile.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    set_model = sub.add_parser("set-model", help="Choose the language model a window uses.")
    set_model.add_argument("mouth", help="The model's name.")
    set_model.add_argument("model", help="The catalog key to use (empty clears the choice).")

    set_voice = sub.add_parser("set-voice", help="Choose the reading voice for a window.")
    set_voice.add_argument("mouth", help="The model's name.")
    set_voice.add_argument("voice", help="A voice key (a catalog, system:, or model: key).")

    set_name = sub.add_parser("set-name", help="Rename a model's character.")
    set_name.add_argument("mouth", help="The model's name.")
    set_name.add_argument("name", help="The new display name.")

    set_logo = sub.add_parser("set-logo", help="Replace a model's logo.")
    set_logo.add_argument("mouth", help="The model's name.")
    set_logo.add_argument("image", help=f"A picture ({', '.join(LOGO_SUFFIXES)}).")

    clear_logo = sub.add_parser("clear-logo", help="Reset a model's logo to the default badge.")
    clear_logo.add_argument("mouth", help="The model's name.")

    ask = sub.add_parser("ask", help="Send one question to a model's own session.")
    ask.add_argument("mouth", help="The model's name.")
    ask.add_argument("question", help="The question to ask.")
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _service(mouth: str) -> ModelChatService:
    return ModelChatService(_mouth(mouth))


def run_mouths(_args: argparse.Namespace) -> int:
    for mouth in WINDOW_MOUTHS:
        print(f"{mouth}: {catalog.ENGINE_LABELS.get(mouth, mouth)}")
        blurb = MOUTH_BLURBS.get(mouth, "")
        if blurb:
            print(f"    {blurb}")
    return EXIT_OK


def run_models(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    models = service.model_choices()
    if args.json:
        import json

        print(json.dumps(
            {
                "mouth": service.mouth,
                "selected": service.model_key(),
                "models": [
                    {
                        "key": model.key,
                        "title": model.title,
                        "engine": model.engine,
                        "available": model.available,
                        "detail": model.detail,
                        "path": str(model.path) if model.path else "",
                    }
                    for model in models
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if not models:
        print(f"No language models are offered for {service.mouth}.")
        return EXIT_OK
    selected = service.model_key()
    for model in models:
        marker = "*" if model.key == selected else " "
        print(f"{marker} {model.label()}")
        print(f"    {model.describe()}")
    return EXIT_OK


def run_profile(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    logo = service.logo_path()
    if args.json:
        import json

        print(json.dumps(
            {
                "mouth": service.mouth,
                "name": service.name,
                "logo": str(logo) if logo else "",
                "voice": service.voice_key(),
                "voice_label": service.voice_label(service.voice_key()),
                "model": service.model_key(),
                "model_status": service.model_status(),
                "available": service.model_ready(),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    print(f"Model:        {service.mouth} ({catalog.ENGINE_LABELS.get(service.mouth, '')})")
    print(f"Name:         {service.name}")
    print(f"Logo:         {logo if logo else '(default badge)'}")
    print(f"Reading voice:{service.voice_label(service.voice_key())}")
    print(f"Model:        {service.model_key() or '(the mouth own default)'}")
    print(f"Status:       {service.model_status()}")
    return EXIT_OK


def run_set_model(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    key = args.model.strip()
    if key and catalog.resolve_choice(service.mouth, key) is None:
        print(
            f"{key!r} is not one of the models this window offers. "
            f"Run `python -m cli model-chat models {service.mouth}` to see them."
        )
        return EXIT_USAGE
    service.set_model(key)
    print(f"{service.name} will answer with {key or 'its own default model'}.")
    return EXIT_OK


def run_set_voice(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    try:
        chosen = service.set_voice(args.voice)
    except ValueError as error:
        print(str(error))
        return EXIT_USAGE
    print(f"{service.name} will read aloud with {service.voice_label(chosen)}.")
    return EXIT_OK


def run_set_name(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    name = service.set_name(args.name)
    print(f"{service.mouth} is now called {name}.")
    return EXIT_OK


def run_set_logo(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    try:
        path = service.set_logo(args.image)
    except (ValueError, OSError) as error:
        print(f"Could not set the logo: {error}")
        return EXIT_PROFILE
    print(f"{service.name}'s logo is now {path}.")
    return EXIT_OK


def run_clear_logo(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    service.clear_logo()
    print(f"{service.name} is back to the default badge.")
    return EXIT_OK


def run_ask(args: argparse.Namespace) -> int:
    service = _service(args.mouth)
    reply = service.ask(args.question)
    if args.json:
        import json

        print(json.dumps(
            {
                "mouth": service.mouth,
                "question": args.question,
                "answer": reply.text,
                "citations": list(reply.citations),
                "hint": reply.hint,
                "model": service.model_key(),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if reply.hint:
        print(reply.hint)
        return EXIT_MODEL
    print(reply.text)
    if reply.citations:
        print("\nSources:")
        for citation in reply.citations:
            print(f"  - {citation}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "mouths": run_mouths,
        "models": run_models,
        "profile": run_profile,
        "set-model": run_set_model,
        "set-voice": run_set_voice,
        "set-name": run_set_name,
        "set-logo": run_set_logo,
        "clear-logo": run_clear_logo,
        "ask": run_ask,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except ValueError as error:
        print(str(error))
        return EXIT_USAGE
    except OSError as error:
        print(f"Could not read or write the profile: {error}")
        return EXIT_PROFILE
    except Exception as error:
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
