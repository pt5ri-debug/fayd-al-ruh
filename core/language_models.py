"""Every language model the project can answer with, in one honest catalog.

Each answering mouth already owns its own model catalog (``core.ultron``,
``core.vision_model``, ``core.presentation_model``, ``companion.model``,
``hakeem.model``, ``core.cloud_model``) and its own folder of weights, and the
project keeps one **public** folder of general models every character may run
(``core.shared_models``). That is right for the *mouth* -- it is what lets Ultron
and Vision never disagree about their own models -- but it leaves the user with no
single answer to "which language models does this program actually have?". This
module is that answer.

It does not re-implement any catalog: it *reads* the existing ones and reports
each model once, with the facts a chooser needs:

* every mouth that can run it (``engines``) -- the GGUF mouths share one
  ``llama-cpp-python`` runtime, so a model fetched for one of them (or into the
  public shared folder) can answer in any of them;
* whether its file is really present here (``available``), and the real next step
  when it is not (``detail``);
* the GGUF file it resolves to, when one is on disk (``path``), so a loader can be
  handed the exact file rather than re-guessing it.

Two engines often name the **same** model (``qwen2.5-7b`` is in both Ultron's and
the companion's catalogs, ``qwen3-8b`` in Banner's and the companion's). They are
the same weights, so they are reported **once**, under one key, with every mouth
that can run it listed -- which is what makes "show all the project's language
models" honest rather than a list of near-duplicates.

Nothing here downloads, loads, or executes anything: it is a read-only view, so
it is dependency-free and testable with no model and no network. The dedicated
per-mouth chat windows (:mod:`gui.model_chat`) and the CLI both build their model
picker from it, so the two can never offer a different list.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.paths import (
    data_dir,
    default_checkpoint,
    hakeem_models_dir,
    natasha_models_dir,
    presentation_models_dir,
    shared_models_dir,
    ultron_models_dir,
    vision_models_dir,
)

#: The mouth keys, as plain strings rather than an import of :mod:`ai.chat`, so
#: this module stays a dependency-free core module (``ai.chat`` imports Tk-free
#: GUI code, and the core must never depend on the front ends).
MOUTH_SOURCES = "sources"
MOUTH_MODEL = "model"
MOUTH_ULTRON = "ultron"
MOUTH_VISION = "vision"
MOUTH_BANNER = "banner"
MOUTH_CLOUD = "cloud"
MOUTH_NATASHA = "natasha"
MOUTH_HAKEEM = "hakeem"
#: Not a *mouth* the user picks, but the project's **public** shared-model folder
#: (:mod:`core.shared_models`). It is a valid ``engine`` for a catalog model -- a
#: model downloaded once into ``data/shared_models`` is offered to every GGUF
#: mouth -- so it lives here beside the mouths it serves.
MOUTH_SHARED = "shared"

#: A short label per mouth, for a section header in a model picker.
ENGINE_LABELS: dict[str, str] = {
    MOUTH_SOURCES: "Retrieval (no model)",
    MOUTH_MODEL: "Trained in-project model",
    MOUTH_ULTRON: "Ultron (general local model)",
    MOUTH_VISION: "Vision (petroleum local model)",
    MOUTH_BANNER: "Banner (presentation local model)",
    MOUTH_CLOUD: "Cloud (remote provider)",
    MOUTH_NATASHA: "Natasha (companion)",
    MOUTH_HAKEEM: "Hakeem (scholar)",
    MOUTH_SHARED: "Shared models (every character)",
}

#: Every mouth this catalog knows about, in the order the composer's model row
#: offers them.
MOUTHS: tuple[str, ...] = (
    MOUTH_SOURCES,
    MOUTH_ULTRON,
    MOUTH_VISION,
    MOUTH_BANNER,
    MOUTH_NATASHA,
    MOUTH_HAKEEM,
    MOUTH_CLOUD,
)

#: The mouths that run a **GGUF file in-process** through ``llama-cpp-python``.
#: They share one runtime and one chat-template mechanism, so any downloaded GGUF
#: can back any of them -- which is why a dedicated window for one of these can
#: offer *every* GGUF model the project has, not only its own engine's catalog.
GGUF_MOUTHS: tuple[str, ...] = (
    MOUTH_ULTRON,
    MOUTH_VISION,
    MOUTH_BANNER,
    MOUTH_NATASHA,
    MOUTH_HAKEEM,
)

#: A short name per mouth, for a one-line model description that has to name
#: every mouth a shared model runs in.
SHORT_LABELS: dict[str, str] = {
    MOUTH_SOURCES: "Sources",
    MOUTH_MODEL: "trained model",
    MOUTH_ULTRON: "Ultron",
    MOUTH_VISION: "Vision",
    MOUTH_BANNER: "Banner",
    MOUTH_CLOUD: "Cloud",
    MOUTH_NATASHA: "Natasha",
    MOUTH_HAKEEM: "Hakeem",
    MOUTH_SHARED: "Shared",
}

#: The two non-GGUF keys, so a caller can name them without a magic string.
TRAINED_KEY = "fayd-llm"
RETRIEVAL_KEY = "retrieval"


@dataclass(frozen=True)
class LanguageModel:
    """One selectable language model, with the facts a chooser needs.

    ``available`` is the honest answer to "would choosing this produce an answer
    now?", and ``detail`` names the real next step when it is ``False``. ``path``
    is the GGUF file the choice resolves to, when one is present, so the loader
    can be handed the exact file rather than re-guessing it.

    ``engines`` names every mouth that can run this model: the GGUF mouths share
    one runtime, and the companion's weights are Hakeem's too, so one entry can
    legitimately belong to several. ``catalog_key`` is the name the owning engine's
    own catalog uses, which is what a loader wants.
    """

    key: str
    title: str
    engine: str
    summary: str
    available: bool
    detail: str = ""
    path: Path | None = None
    #: For the cloud engine, the provider this model belongs to.
    provider: str = ""
    #: Every mouth that can run this model.
    engines: tuple[str, ...] = ()
    #: The key in the owning engine's own catalog.
    catalog_key: str = ""
    #: The folder(s) the file may live in, in the order they are checked.
    folders: tuple[Path, ...] = field(default=())
    #: The GGUF filename, empty for a non-file model (cloud, retrieval).
    filename: str = ""
    #: The identity of the weights (repo + file), which is how two engines naming
    #: the same model are recognised as one.
    identity: str = ""
    #: The governed command that fetches this model's weights, empty when there is
    #: nothing to fetch (a file the user added, the trained checkpoint, a cloud
    #: model). The window and the CLI show it, and the download action runs the
    #: same engine's own gated fetch rather than a re-implementation.
    download_command: str = ""
    #: True when the user added this model by hand (see :mod:`core.custom_models`),
    #: so a window can offer to forget it again rather than only list it.
    custom: bool = False

    def label(self) -> str:
        """The title plus an honest marker when the model is not present here."""
        return self.title if self.available else f"{self.title} — not installed"

    def describe(self) -> str:
        """A one-line, honest description, including the next step when absent."""
        line = f"{self.title} ({self.engines_label()})"
        if self.available:
            return f"{line} — ready."
        return f"{line} — {self.detail or 'not installed here.'}"

    def engines_label(self) -> str:
        """The mouths this model runs in, as readable names."""
        names = [SHORT_LABELS.get(mouth, mouth) for mouth in self.engines]
        return " / ".join(names) if names else ENGINE_LABELS.get(self.engine, self.engine)

    def runs_in(self, mouth: str) -> bool:
        """True when ``mouth`` can run this model."""
        return mouth in self.engines


@dataclass(frozen=True)
class _Spec:
    """One engine's catalog entry, before it is merged with another's."""

    key: str
    title: str
    summary: str
    engine: str
    folder: Path
    filename: str
    #: The identity of the weights (repo + file), so the same model named by two
    #: engines is merged rather than listed twice.
    identity: str
    download: str


def _clean_title(title: str) -> str:
    """A merged entry's title, without an engine-specific "(default)" marker.

    Each engine marks *its own* default in its title, but a merged model is the
    default for at most one of the mouths that run it, so carrying the marker
    would be misleading. The per-mouth ordering (:func:`_mouth_default_key`)
    already puts a mouth's own default first, which is what tells the picker.
    """
    return title.replace(" (default)", "").strip()


def _register(
    registry: dict[str, LanguageModel],
    order: list[str],
    spec: _Spec,
) -> None:
    """Merge one engine's spec into the project-wide registry.

    The same weights named by two engines (the identity matches) become one entry
    that both can run, so a model is never listed twice under two names. The
    first engine to name it keeps its title and download hint; a later one only
    adds itself to ``engines`` and its folder to ``folders``.
    """
    existing = registry.get(spec.identity)
    if existing is None:
        registry[spec.identity] = LanguageModel(
            key=spec.key,
            title=_clean_title(spec.title),
            engine=spec.engine,
            engines=(spec.engine,),
            summary=spec.summary,
            available=False,  # resolved in _finalise once every folder is known
            detail=(
                f"download it with `{spec.download}`, "
                f"or drop its .gguf file into {spec.folder}."
            ),
            download_command=spec.download,
            catalog_key=spec.key,
            folders=(spec.folder,),
            filename=spec.filename,
            identity=spec.identity,
        )
        order.append(spec.identity)
        return
    if spec.engine in existing.engines:
        return
    registry[spec.identity] = replace(
        existing,
        engines=(*existing.engines, spec.engine),
        folders=(*existing.folders, spec.folder),
    )


def _finalise(model: LanguageModel) -> LanguageModel:
    """Resolve a merged model's file against every folder it may live in."""
    for folder in model.folders:
        candidate = folder / model.filename
        if candidate.is_file():
            return replace(model, available=True, path=candidate)
    return replace(model, available=False, path=None)


def _gguf_specs() -> list[_Spec]:
    """Every GGUF model any in-process mouth names, as raw specs."""
    specs: list[_Spec] = []

    from core.ultron import resolve_spec as ultron_resolve
    from core.ultron import spec_summaries as ultron_specs

    for key, title, size, summary in ultron_specs():
        spec: Any = ultron_resolve(key)
        specs.append(
            _Spec(
                key=key, title=f"{title} (~{size:.1f} GB)", summary=summary,
                engine=MOUTH_ULTRON, folder=ultron_models_dir(), filename=spec.filename,
                identity=f"{spec.repo}/{spec.filename}",
                download=f"python -m cli ultron download {key} --approve",
            )
        )

    from core.vision_model import resolve_spec as vision_resolve
    from core.vision_model import spec_summaries as vision_specs

    for key, title, size, summary in vision_specs():
        spec = vision_resolve(key)
        specs.append(
            _Spec(
                key=key, title=f"{title} (~{size:.1f} GB)", summary=summary,
                engine=MOUTH_VISION, folder=vision_models_dir(), filename=spec.filename,
                identity=f"{spec.repo}/{spec.filename}",
                download=f"python -m cli vision download {key} --approve",
            )
        )

    from core.presentation_model import (
        resolve_spec as banner_resolve,
    )
    from core.presentation_model import (
        spec_summaries as banner_specs,
    )

    for key, title, size, summary in banner_specs():
        spec = banner_resolve(key)
        specs.append(
            _Spec(
                key=key, title=f"{title} (~{size:.1f} GB)", summary=summary,
                engine=MOUTH_BANNER, folder=presentation_models_dir(), filename=spec.filename,
                identity=f"{spec.repo}/{spec.filename}",
                download=f"python -m cli presentation model-download {key} --approve",
            )
        )

    from companion.model import resolve_spec as character_resolve
    from companion.model import spec_summaries as character_specs

    for key, title, size, summary in character_specs():
        spec = character_resolve(key)
        specs.append(
            _Spec(
                key=key, title=f"{title} (~{size:.1f} GB)", summary=summary,
                engine=MOUTH_NATASHA, folder=natasha_models_dir(), filename=spec.filename,
                identity=f"{spec.repo}/{spec.filename}",
                download=f"python -m cli natasha download {key} --approve",
            )
        )

    from hakeem.model import resolve_spec as hakeem_resolve
    from hakeem.model import spec_summaries as hakeem_specs

    for key, title, size, summary in hakeem_specs():
        spec = hakeem_resolve(key)
        specs.append(
            _Spec(
                key=key, title=f"{title} (~{size:.1f} GB)", summary=summary,
                engine=MOUTH_HAKEEM, folder=hakeem_models_dir(), filename=spec.filename,
                identity=f"{spec.repo}/{spec.filename}",
                download=f"python -m cli hakeem download {key} --approve",
            )
        )

    from core.shared_models import resolve_spec as shared_resolve
    from core.shared_models import spec_summaries as shared_specs

    for key, title, size, summary in shared_specs():
        spec = shared_resolve(key)
        specs.append(
            _Spec(
                key=key, title=f"{title} (~{size:.1f} GB)", summary=summary,
                engine=MOUTH_SHARED, folder=shared_models_dir(), filename=spec.filename,
                identity=f"{spec.repo}/{spec.filename}",
                download=f"python -m cli shared-models download {key} --approve",
            )
        )
    return specs


def _hakeem_identities() -> set[str]:
    """The identities of Hakeem's own models, which he runs in his own folder.

    Read from Hakeem's own catalog (not from an entry's ``engine``) so a shared
    model that also appears in the public folder still marks Hakeem as an engine.
    """
    from hakeem.model import resolve_spec, spec_summaries

    return {
        f"{resolve_spec(key).repo}/{resolve_spec(key).filename}"
        for key, *_rest in spec_summaries()
    }


def _trained_model() -> LanguageModel:
    checkpoint = default_checkpoint()
    present = checkpoint.is_file()
    return LanguageModel(
        key=TRAINED_KEY,
        title="Fayd Al-Ruh trained model",
        engine=MOUTH_MODEL,
        engines=(MOUTH_MODEL,),
        summary="The small in-project language model trained on the project's own corpus.",
        available=present,
        detail=(
            "train one with `python -m cli release` (or `python -m ml.llm`) — "
            f"no checkpoint was found at {checkpoint}."
        ),
        path=checkpoint if present else None,
        catalog_key=TRAINED_KEY,
    )


def _retrieval_model() -> LanguageModel:
    return LanguageModel(
        key=RETRIEVAL_KEY,
        title="Retrieved passages (cited)",
        engine=MOUTH_SOURCES,
        engines=(MOUTH_SOURCES,),
        summary="No language model: answers are assembled from cited corpus passages.",
        available=True,
        catalog_key=RETRIEVAL_KEY,
    )


def _cloud_models() -> list[LanguageModel]:
    """One entry per provider model, keyed ``<provider>:<model>``.

    A cloud answer needs a *provider* and a *model*, so both are carried in the
    key and the report states plainly whether the provider's key is set here --
    the honest answer to "would this answer now?".
    """
    from core.cloud_model import CATALOG, load_model_key, load_provider_key, provider_models

    models: list[LanguageModel] = []
    for key, provider in CATALOG.items():
        names = list(provider_models(key))
        saved = load_model_key(data_dir(), key)
        if saved and saved not in names:
            names.insert(0, saved)
        if not names:
            names = [provider.default_model]
        for name in names:
            models.append(
                LanguageModel(
                    key=f"{key}:{name}",
                    title=f"{provider.title} — {name}",
                    engine=MOUTH_CLOUD,
                    engines=(MOUTH_CLOUD,),
                    summary=provider.summary,
                    available=_cloud_ready(provider),
                    detail=(
                        f"set {provider.api_key_env} for {provider.title}"
                        + (f" (free tier: {provider.signup_url})" if provider.free else "")
                        if provider.needs_key
                        else "start the local server this provider points at."
                    ),
                    provider=key,
                    catalog_key=f"{key}:{name}",
                )
            )
    # The provider the user last chose first, so the picker opens on it.
    current = load_provider_key(data_dir())
    return sorted(models, key=lambda model: model.provider != current)


def _cloud_ready(provider) -> bool:
    """True when this provider could answer now (a key is set, or none is needed)."""
    if not provider.needs_key:
        return True
    import os

    return bool(os.environ.get(provider.api_key_env, "").strip())


def _build_gguf() -> tuple[LanguageModel, ...]:
    """The merged GGUF catalog: each model once, with every mouth that can run it.

    Hakeem's own models are marked as his (they live in his own folder), and the
    public shared folder's models are marked for **every** GGUF mouth, because any
    character can run them. That is decided by the weights' identity, not by an
    entry's ``engine``: a model named by two engines merges into the first entry,
    so an engine check would miss the second.
    """
    registry: dict[str, LanguageModel] = {}
    order: list[str] = []
    for spec in _gguf_specs():
        _register(registry, order, spec)
    hakeem_own = _hakeem_identities()
    shared_engine = {
        identity
        for identity in order
        if registry[identity].engine == MOUTH_SHARED
        or MOUTH_SHARED in registry[identity].engines
    }
    models: list[LanguageModel] = []
    for identity in order:
        model = _finalise(registry[identity])
        engines = model.engines
        if identity in hakeem_own and MOUTH_HAKEEM not in engines:
            engines = (*engines, MOUTH_HAKEEM)
        if identity in shared_engine:
            # A shared model answers in every GGUF mouth (and the shared folder).
            extra = tuple(mouth for mouth in (*GGUF_MOUTHS, MOUTH_SHARED) if mouth not in engines)
            engines = (*engines, *extra)
        if engines != model.engines:
            model = replace(model, engines=engines)
        models.append(model)
    return tuple(models)


def _custom_models() -> tuple[LanguageModel, ...]:
    """The GGUF files the user added by hand (see :mod:`core.custom_models`).

    Each is a real file the user pointed a window at, so it is offered to every
    GGUF mouth -- they share the one in-process runtime -- and its availability is
    the file's own presence, checked on every read. Nothing is copied: the entry
    references the file where the user keeps it.
    """
    from core import custom_models

    models: list[LanguageModel] = []
    try:
        entries = custom_models.list_models()
    except Exception:
        return ()
    for entry in entries:
        present = entry.available()
        models.append(
            LanguageModel(
                key=entry.key,
                title=entry.title,
                engine=MOUTH_ULTRON,
                engines=GGUF_MOUTHS,
                summary="A GGUF model you added from this machine.",
                available=present,
                detail=(
                    "ready."
                    if present
                    else f"the file {entry.path} is not there any more; "
                    "add it again from the window."
                ),
                path=entry.path if present else None,
                catalog_key=entry.key,
                folders=(entry.path.parent,),
                filename=entry.path.name,
                identity=f"custom:{entry.path}",
                custom=True,
            )
        )
    return tuple(models)


def _all() -> tuple[LanguageModel, ...]:
    """Every language model the project knows, freshly resolved.

    Rebuilt on each read rather than cached: the honest fields (``available`` and
    the resolved ``path``) depend on which files are on disk *right now*, so a
    cached catalog would keep reporting a model as missing after the user
    downloaded it -- or as present after they removed it. The work is a handful of
    file checks, cheap enough for a picker to call per redraw.
    """
    models: list[LanguageModel] = []
    try:
        models.extend(_build_gguf())
    except Exception:
        pass  # a broken optional catalog must not make the whole picker unusable
    try:
        models.extend(_custom_models())
    except Exception:
        pass  # a corrupt custom-model registry must not break the picker either
    models.append(_trained_model())
    models.append(_retrieval_model())
    try:
        models.extend(_cloud_models())
    except Exception:
        pass
    return tuple(models)


def reset_catalog_cache() -> None:
    """Present for callers that expected a cache; the catalog is always live."""


def _mouth_default_key(mouth: str) -> str | None:
    """The catalog key each mouth would choose on its own, from its own catalog.

    Read from the engine's own ``DEFAULT_SPEC_KEY`` rather than inferred from the
    merged order, because the merge is global: Ultron's first model would
    otherwise decide the companion's default too.
    """
    try:
        if mouth == MOUTH_ULTRON:
            from core.ultron import DEFAULT_SPEC_KEY

            return DEFAULT_SPEC_KEY
        if mouth == MOUTH_VISION:
            from core.vision_model import DEFAULT_SPEC_KEY

            return DEFAULT_SPEC_KEY
        if mouth == MOUTH_BANNER:
            from core.presentation_model import DEFAULT_SPEC_KEY

            return DEFAULT_SPEC_KEY
        if mouth == MOUTH_NATASHA:
            from companion.model import DEFAULT_SPEC_KEY

            return DEFAULT_SPEC_KEY
        if mouth == MOUTH_HAKEEM:
            from hakeem.model import DEFAULT_SPEC_KEY

            return DEFAULT_SPEC_KEY
        if mouth == MOUTH_SHARED:
            from core.shared_models import DEFAULT_SPEC_KEY

            return DEFAULT_SPEC_KEY
    except Exception:
        return None
    if mouth == MOUTH_MODEL:
        return TRAINED_KEY
    if mouth == MOUTH_SOURCES:
        return RETRIEVAL_KEY
    return None


def language_models(mouth: str) -> tuple[LanguageModel, ...]:
    """Every language model ``mouth`` itself can run, its own default first.

    An unknown mouth yields nothing rather than raising: a caller that asks about
    a mouth this build does not know simply gets no models to offer.
    """
    models = [model for model in _all() if model.runs_in(mouth)]
    default = _mouth_default_key(mouth)
    if default is not None:
        models.sort(key=lambda model: model.catalog_key != default)
    return tuple(models)


def _gguf_models_all() -> tuple[LanguageModel, ...]:
    """Every GGUF model the project has, plus the trained checkpoint."""
    return tuple(
        model
        for model in _all()
        if model.engine in (*GGUF_MOUTHS, MOUTH_SHARED) or model.engine == MOUTH_MODEL
    )


def models_for_choice(mouth: str) -> tuple[LanguageModel, ...]:
    """The models a dedicated ``mouth`` window should offer in its picker.

    For a mouth that loads a GGUF file in-process (Ultron, Vision, Banner,
    Natasha, Hakeem) this is **every** GGUF model the project has, because they
    all share the one ``llama-cpp-python`` runtime -- so a model downloaded for
    any mouth can answer in any of them. That is the honest meaning of "show all
    the project's language models": the same weights, whichever catalog first
    named them. The trained checkpoint is added too, since the same runtime can
    load it. The mouth's own default is listed first, so the picker opens on it.

    Every other mouth (the cloud provider, retrieval) offers only its own models,
    because those are genuinely different kinds of model rather than files.
    """
    if mouth in GGUF_MOUTHS:
        models = list(_gguf_models_all())
        default = _mouth_default_key(mouth)
        if default is not None:
            models.sort(key=lambda model: model.catalog_key != default)
        return tuple(models)
    return language_models(mouth)


def find_language_model(mouth: str, key: str | None) -> LanguageModel | None:
    """The model named by ``key`` that ``mouth`` can run, or ``None``.

    Searches what the mouth's dedicated window offers (:func:`models_for_choice`),
    so a model downloaded for one GGUF engine still resolves in another.
    """
    wanted = (key or "").strip()
    if not wanted:
        return None
    for model in models_for_choice(mouth):
        if model.key == wanted:
            return model
    return None


def resolve_choice(mouth: str, key: str | None) -> LanguageModel | None:
    """The model ``key`` names for a dedicated ``mouth`` window, or ``None``."""
    return find_language_model(mouth, key)


def all_language_models() -> tuple[tuple[str, LanguageModel], ...]:
    """Every model of every mouth, as ``(mouth, model)`` pairs, in mouth order.

    A model two mouths share (the companion's, which Hakeem runs too) appears once
    per mouth, so a caller that walks the pairs sees each mouth's own list.
    """
    pairs: list[tuple[str, LanguageModel]] = []
    for mouth in MOUTHS:
        pairs.extend((mouth, model) for model in language_models(mouth))
    return tuple(pairs)


def find_language_model_anywhere(key: str | None) -> tuple[str, LanguageModel] | None:
    """The ``(mouth, model)`` a key names anywhere in the project, or ``None``.

    A key is unique across the project, so this is unambiguous; it returns the
    first mouth that can run it, in :data:`MOUTHS` order.
    """
    wanted = (key or "").strip()
    if not wanted:
        return None
    for mouth, model in all_language_models():
        if model.key == wanted:
            return mouth, model
    return None


def default_language_model_key(mouth: str) -> str | None:
    """The key of the model the mouth would choose on its own, or ``None``.

    Each mouth's own catalog orders its default first, and the merged list keeps
    the registration order, so the first model a mouth can run is the one it would
    have used anyway -- which is what a picker's initial selection should be.
    """
    models = language_models(mouth)
    return models[0].key if models else None


def ready_language_model_key(mouth: str) -> str | None:
    """The mouth's **default** model when its weights are on disk, else any ready one.

    The default is preferred so a user who installed the mouth's own model keeps
    it. When the default is not on disk but *another* model the mouth can run is
    (for instance a general model in the shared folder), that model is returned:
    the mouth can really answer with it, so a status line or an evaluation that
    reported "no model" would be wrong. ``None`` when nothing the mouth can run is
    on disk -- the honest "this mouth has no weights here".
    """
    models = language_models(mouth)
    if not models:
        return None
    default = models[0]
    if default.available:
        return default.key
    for model in models[1:]:
        if model.available:
            return model.key
    return None


def ready_language_model(mouth: str) -> LanguageModel | None:
    """The model :func:`ready_language_model_key` names, or ``None``."""
    key = ready_language_model_key(mouth)
    return find_language_model(mouth, key) if key else None


def resolve_model_path(mouth: str, key: str | None) -> Path | None:
    """The GGUF file a chosen model resolves to, or ``None`` when there is none.

    Only a mouth that loads a GGUF file in-process has a path; a cloud model, the
    trained checkpoint's absence, or a retrieval-only mouth yields ``None``. The
    caller passes this to the mouth's loader as an explicit model file, so the
    choice is honoured rather than re-guessed from the folder.
    """
    model = find_language_model(mouth, key)
    return model.path if model is not None else None


def describe_language_models(mouth: str) -> str:
    """A multi-line, honest report of a mouth's models, for a status line."""
    models = language_models(mouth)
    if not models:
        return f"No language models are offered for {ENGINE_LABELS.get(mouth, mouth)}."
    lines = [ENGINE_LABELS.get(mouth, mouth) + ":"]
    lines.extend(f"  - {model.describe()}" for model in models)
    return "\n".join(lines)


def can_download(model: LanguageModel | None) -> bool:
    """True when ``model`` has weights the project knows how to fetch here.

    A catalog model has a governed download; a model the user added, the trained
    checkpoint, a cloud model, and the retrieval mouth have nothing to fetch, so a
    window offers no download button for them.
    """
    if model is None or model.custom or not model.download_command:
        return False
    return model.engine in {
        MOUTH_ULTRON,
        MOUTH_VISION,
        MOUTH_BANNER,
        MOUTH_NATASHA,
        MOUTH_HAKEEM,
        MOUTH_SHARED,
    }


def download_model(
    mouth: str,
    key: str,
    *,
    approved: bool = False,
    progress: Any = None,
) -> Path:
    """Fetch a chosen model's weights through the owning engine's own gate.

    This is the one place a window's *Download* button and the CLI's
    ``download-model`` command turn a catalog key into a real, human-approved
    fetch: it dispatches to the engine that names the model (Ultron's, Vision's,
    Banner's, or the companion's shared folder) so the download obeys that
    engine's permission and approval gates, and no second download path is
    written. Raises the engine's own availability error on any failure, and
    :class:`ValueError` when the model has nothing to fetch.
    """
    model = find_language_model(mouth, key)
    if model is None:
        found = find_language_model_anywhere(key)
        model = found[1] if found is not None else None
    if model is None:
        raise ValueError(f"{key!r} is not a model this build knows.")
    if not can_download(model):
        raise ValueError(f"{model.title!r} has nothing to download.")
    catalog_key = model.catalog_key or key
    if model.engine == MOUTH_ULTRON:
        from core.ultron import download_model as fetch

        return fetch(catalog_key, approved=approved, progress=progress)
    if model.engine == MOUTH_VISION:
        from core.vision_model import download_model as fetch

        return fetch(catalog_key, approved=approved, progress=progress)
    if model.engine == MOUTH_BANNER:
        from core.presentation_model import download_model as fetch

        return fetch(catalog_key, approved=approved, progress=progress)
    if model.engine == MOUTH_NATASHA:
        from companion.model import download_model as fetch

        return fetch(catalog_key, approved=approved, progress=progress)
    if model.engine == MOUTH_HAKEEM:
        from hakeem.model import download_model as fetch

        return fetch(catalog_key, approved=approved, progress=progress)
    if model.engine == MOUTH_SHARED:
        from core.shared_models import download_model as fetch

        return fetch(catalog_key, approved=approved, progress=progress)
    raise ValueError(f"{model.title!r} has nothing to download.")


__all__ = [
    "ENGINE_LABELS",
    "GGUF_MOUTHS",
    "MOUTHS",
    "MOUTH_BANNER",
    "MOUTH_CLOUD",
    "MOUTH_HAKEEM",
    "MOUTH_MODEL",
    "MOUTH_NATASHA",
    "MOUTH_SHARED",
    "MOUTH_SOURCES",
    "MOUTH_ULTRON",
    "MOUTH_VISION",
    "RETRIEVAL_KEY",
    "SHORT_LABELS",
    "TRAINED_KEY",
    "LanguageModel",
    "all_language_models",
    "can_download",
    "default_language_model_key",
    "describe_language_models",
    "download_model",
    "find_language_model",
    "find_language_model_anywhere",
    "language_models",
    "models_for_choice",
    "ready_language_model",
    "ready_language_model_key",
    "reset_catalog_cache",
    "resolve_choice",
    "resolve_model_path",
]
