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
    """Build Natasha's or Hakeem's mouth against a chosen model.

    Each character has its **own** model catalog and its own folder, so the choice
    is resolved through *that* character's catalog: the spec's Ollama ``tag`` names
    the model (the fallback runtime), and the resolved GGUF file is handed over
    explicitly so the in-process runtime loads *that* file rather than the largest
    one in the folder.
    """
    loader = natasha_mouth.load_natasha_assistant if mouth == catalog.MOUTH_NATASHA else hakeem_mouth.load_hakeem_assistant
    if choice is None:
        return loader(**kwargs)
    tag = _character_tag(mouth, choice.catalog_key)
    return loader(model=tag, model_file=choice.path, **kwargs)


def _character_tag(mouth: str, key: str) -> str:
    """The Ollama tag for a catalog key, from whichever catalog names it.

    Each character's *own* catalog is tried first (Natasha's for the companion,
    Hakeem's for the scholar). A character's window also offers every GGUF model
    the project has -- they share one in-process runtime -- so a key from another
    catalog (the companion's, the public shared folder's) is resolved too. A key no
    catalog knows is used verbatim: it is only the fallback runtime's name for the
    model, and a wrong guess would be worse than the honest key.
    """
    modules: list[Any] = []
    if mouth == catalog.MOUTH_HAKEEM:
        from hakeem import model as hakeem_model

        modules.append(hakeem_model)
    from companion import model as companion_model

    modules.append(companion_model)
    try:
        from core import shared_models

        modules.append(shared_models)
    except Exception:
        pass
    for module in modules:
        try:
            return module.resolve_spec(key).tag
        except Exception:
            continue
    return key


def configured_model_key(mouth: str) -> str:
    """The model key a character's **own** configuration names for ``mouth``.

    The Characters window's Language-model field is the canonical place a user
    picks a character's model, while a per-mouth window's picker is a per-window
    override. This reads the character's saved choice so a loader can honour it
    when no override is set -- without it, a model chosen in the Characters window
    would be saved and never used. An empty string means "the mouth's own default",
    and a non-character mouth (or an unreadable configuration) returns "".
    """
    if not is_character_mouth(mouth):
        return ""
    try:
        from core.character_registry import default_registry

        config = default_registry().load(mouth)
    except (ValueError, OSError):
        return ""
    model = getattr(config, "model", None)
    if model is None or not getattr(model, "active", True):
        return ""
    return (getattr(model, "key", "") or "").strip()


def is_character_mouth(mouth: str) -> bool:
    """True when ``mouth`` has a character configuration of its own."""
    from ai.characters import is_character

    return is_character(mouth)


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
    "configured_model_key",
    "describe_mouth_model",
    "is_character_mouth",
    "model_choice",
]
