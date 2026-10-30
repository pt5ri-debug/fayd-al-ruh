"""Build one answering mouth, optionally against a chosen language model.

The chat's :class:`~ai.chat.ModelSwitcher` builds each mouth from the model the
mouth would pick on its own. A dedicated per-mouth window
(:mod:`gui.model_chat`) lets the user choose a *different* model from the
project's full catalog (see :mod:`core.language_models`), so this module is the
one place that turns "(mouth, model key) -> a ready assistant".

It exists so the choice is honoured in exactly one way: the loader is handed the
resolved GGUF path (or, for a cloud mouth, the provider and model name) rather
than re-guessing it from the model folder. A key this build does not know falls
back to the mouth's own default, so a stale saved choice can never break a mouth.

Nothing here imports Tk, and every model loads lazily inside the mouth itself, so
importing this module is free.
"""

from __future__ import annotations

from typing import Any

from core import language_models as catalog

from ai import banner_mouth, cloud_mouth, hakeem_mouth, natasha_mouth, ultron_mouth, vision_mouth
from ai import mind as mind_module

#: The mouths this module can build a *model-backed* assistant for. ``sources``
#: is deliberately absent: the grounded assistant retrieves from the session's
#: own store and has no model choice.
MODEL_MOUTHS: tuple[str, ...] = (
    catalog.MOUTH_MODEL,
    catalog.MOUTH_ULTRON,
    catalog.MOUTH_VISION,
    catalog.MOUTH_BANNER,
    catalog.MOUTH_CLOUD,
    catalog.MOUTH_NATASHA,
    catalog.MOUTH_HAKEEM,
)


def model_choice(mouth: str, model_key: str | None) -> catalog.LanguageModel | None:
    """The catalog entry ``model_key`` names for ``mouth``, or ``None``.

    ``None`` (or an unknown key) means "the mouth's own default", which is what
    every loader already does when handed no explicit model. The lookup is against
    what the mouth's dedicated window offers (:func:`core.language_models.models_for_choice`),
    so a GGUF model the user picked from another engine's catalog still resolves to
    its file.
    """
    return catalog.resolve_choice(mouth, model_key)


def build_mouth(
    mouth: str,
    *,
    model_key: str | None = None,
    max_new_tokens: int = 96,
    **kwargs: Any,
):
    """Build the assistant for ``mouth``, honouring ``model_key`` when offered.

    Raises the mouth's own availability error (``UltronUnavailable``,
    ``CompanionError``, ...) when it cannot answer, exactly as the switcher's
    loaders do, so a caller reports the real next step instead of switching to a
    mouth that could not speak. ``kwargs`` are passed to the loader unchanged, so
    a test can inject its own agent/client.

    This is a pure factory: it returns the mouth's own assistant. Wrapping a mouth
    as its character -- its standing personality and its own learning record -- is
    the switcher's job (see :mod:`ai.characters` and ``ai.chat.ModelSwitcher``), so
    the wrapping happens once, in one place, and a caller here gets the real mouth.
    """
    choice = model_choice(mouth, model_key)
    path = choice.path if choice is not None else None

    if mouth == catalog.MOUTH_MODEL:
        return mind_module.load_mind_model(path, max_new_tokens=max_new_tokens)
    if mouth == catalog.MOUTH_ULTRON:
        return ultron_mouth.load_ultron_assistant(path, **kwargs)
    if mouth == catalog.MOUTH_VISION:
        return vision_mouth.load_vision_assistant(path, **kwargs)
    if mouth == catalog.MOUTH_BANNER:
        return banner_mouth.load_banner_assistant(path, **kwargs)
    if mouth == catalog.MOUTH_CLOUD:
        provider = choice.provider if choice is not None else None
        # The catalog key is ``<provider>:<model>``; the loader wants the two apart.
        model_name = choice.key.split(":", 1)[1] if choice is not None else None
        return cloud_mouth.load_cloud_assistant(model_name, provider=provider, **kwargs)
    if mouth in (catalog.MOUTH_NATASHA, catalog.MOUTH_HAKEEM):
        return _build_character(mouth, choice, **kwargs)
    raise ValueError(f"{mouth!r} has no model-backed assistant to build.")


def _build_character(mouth: str, choice: catalog.LanguageModel | None, **kwargs: Any):
    """Build Natasha's or Hakeem's mouth against a chosen (shared) model.

    They speak through the same weights, so the choice is resolved the same way
    for both: the spec's Ollama ``tag`` names the model (the fallback runtime),
    and the resolved GGUF file is handed over explicitly so the in-process runtime
    loads *that* file rather than the largest one in the folder.
    """
    loader = natasha_mouth.load_natasha_assistant if mouth == catalog.MOUTH_NATASHA else hakeem_mouth.load_hakeem_assistant
    if choice is None:
        return loader(**kwargs)
    tag = _character_tag(choice.catalog_key)
    return loader(model=tag, model_file=choice.path, **kwargs)


def _character_tag(key: str) -> str:
    """The Ollama tag for a companion/hakeem catalog key.

    A key that is in the companion's own catalog resolves to its spec's tag. A key
    that came from *another* engine's catalog (the window lets a user pick any GGUF
    model) is used verbatim: it is only the fallback runtime's name for the model,
    and a wrong guess would be worse than the honest key.
    """
    from companion.model import CompanionModelUnavailable, resolve_spec

    try:
        return resolve_spec(key).tag
    except (CompanionModelUnavailable, ValueError):
        return key


def describe_mouth_model(mouth: str, model_key: str | None) -> str:
    """An honest one-line description of the model a mouth will answer with."""
    choice = model_choice(mouth, model_key)
    if choice is None:
        default = catalog.default_language_model_key(mouth)
        choice = catalog.find_language_model(mouth, default)
    if choice is None:
        return f"{catalog.ENGINE_LABELS.get(mouth, mouth)}: no selectable model."
    return choice.describe()

__all__ = [
    "MODEL_MOUTHS",
    "build_mouth",
    "describe_mouth_model",
    "model_choice",
]
