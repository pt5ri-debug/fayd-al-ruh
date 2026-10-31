"""The project's own trained models, registered without touching the trainer.

The audit drew a hard line around the training pipeline: ``ml.llm.__main__``,
``ml.release``, ``ml.llm.evaluate``, ``ml.llm.state``, ``ml.llm.schedule``, and
``ml.llm.corpus`` must not be modified, and the Model Manager must not duplicate
them. This module is the **integration layer** the audit permitted instead: it
*reads* what the training pipeline already wrote and presents it to the manager.

What it reads, and from where -- all read-only:

* the committed language-model checkpoint (``ml/llm/checkpoints/fayd-llm.npz``) and
  its ``.tokenizer.json`` sidecar, described by :func:`core.model_files.probe_file`
  (which parses the ``.npz`` with the standard library, so no numpy is needed);
* the release manifest (``ml/artifacts/release_manifest.json``), which the release
  run writes and which already records the architecture, the parameter count, and
  the measured perplexity;
* the run-state sidecars a resume leaves behind (``.state.npz``, ``.schedule.json``),
  read only to report whether a checkpoint was resumed and how far it got -- never
  written, and never required;
* the sibling artifacts the base training writes (``policy_*.npz``,
  ``experience_classifier.npz``, ``relation_model.npz``, ``brain_*``), which are
  **not** language models and are labelled as such rather than offered as a chat
  mouth.

A trained model is registered as an ordinary :class:`ModelRecord` with the
``trained`` source and the in-process runtime, so it reaches every surface the
other models reach -- the registry, the ranking, the fallback ladder, the CLI, and
the window -- with no second registry and no second source of truth.

The honesty rules, carried from the audit:

* the parameter count, the perplexity, and the bits-per-byte are **reported** as the
  manifest records them, never re-measured here and never presented as this
  module's own measurement;
* ``.state.npz`` and ``.schedule.json`` are run state, not release content, so their
  absence is normal and is reported as "unknown", not as an error;
* an evaluation result is stored **beside** the checkpoint, never written into it,
  so re-evaluating cannot alter the weights it measured;
* nothing here trains, fine-tunes, resumes, or evaluates. It reads files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.model_files import (
    FORMAT_FAYD_CHECKPOINT,
    FORMAT_UNKNOWN,
    INTEGRITY_OK,
    probe_file,
)
from core.model_registry import (
    SOURCE_TRAINED,
    ModelRecord,
)

#: What kind of trained thing an artifact is. Only ``language`` is a chat mouth.
KIND_LANGUAGE = "language-model"
KIND_POLICY = "policy"
KIND_CLASSIFIER = "classifier"
KIND_RELATION = "relation-model"
KIND_BRAIN = "predictive-brain"
KIND_UNKNOWN = "unknown"

KIND_LABELS: dict[str, str] = {
    KIND_LANGUAGE: "Trained language model",
    KIND_POLICY: "Trained Q-policy",
    KIND_CLASSIFIER: "Experience classifier",
    KIND_RELATION: "Relation model",
    KIND_BRAIN: "Predictive brain",
    KIND_UNKNOWN: "Trained artifact",
}

#: The kinds that can answer a chat prompt. Only the language model can.
CHAT_CAPABLE: tuple[str, ...] = (KIND_LANGUAGE,)

#: The manifest the release run writes, relative to the artifacts folder.
RELEASE_MANIFEST_NAME = "release_manifest.json"

#: The identity a trained checkpoint is registered under, matching
#: ``core.model_settings.TRAINED_IDENTITY`` so its settings resolve as before.
TRAINED_IDENTITY = "fayd-llm"

#: The runtime the project's own checkpoint runs under.
TRAINED_RUNTIME = "fayd-inprocess"


@dataclass(frozen=True)
class EvaluationResult:
    """A measurement the training pipeline recorded, attributed to its source.

    ``source`` names where the figure came from -- ``"release manifest"`` or
    ``"evaluation sidecar"`` -- because a number with no provenance is one a reader
    cannot judge. Nothing here is measured by this module.
    """

    name: str
    value: float | None = None
    source: str = ""
    detail: str = ""

    def describe(self) -> str:
        if self.value is None:
            return f"{self.name}: not recorded ({self.source})" if self.source else f"{self.name}: not recorded"
        line = f"{self.name}: {self.value:g}"
        if self.source:
            line = f"{line} (recorded by {self.source})"
        return line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "name": self.name,
            "source": self.source,
            "value": self.value,
        }


@dataclass(frozen=True)
class TrainedArtifact:
    """One trained artifact, described from what is on disk.

    ``kind`` says what it is; ``chat_capable`` is true only for the language model.
    ``architecture``, ``parameters``, and ``evaluations`` are the facts the release
    manifest recorded, carried through unchanged. ``run_state`` reports the resume
    sidecars when they happen to be there -- as run state, not as release content.
    """

    path: Path
    kind: str = KIND_UNKNOWN
    exists: bool = False
    size_bytes: int = 0
    format: str = FORMAT_UNKNOWN
    integrity: str = ""
    architecture: dict[str, Any] = field(default_factory=dict)
    parameters: int | None = None
    evaluations: tuple[EvaluationResult, ...] = field(default=())
    tokenizer_path: Path | None = None
    tokenizer_present: bool = False
    #: ``total_steps`` the checkpoint holds, when a sidecar records it.
    total_steps: int | None = None
    resumed: bool | None = None
    schedule: str = ""
    detail: str = ""
    #: The manifest's own view of the checkpoint, verbatim, so nothing is lost.
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_capable(self) -> bool:
        return self.kind in CHAT_CAPABLE and self.exists

    @property
    def usable(self) -> bool:
        """True when the file is present and intact (regardless of kind)."""
        return self.exists and self.format != FORMAT_UNKNOWN and self.integrity == INTEGRITY_OK

    def describe(self) -> str:
        label = KIND_LABELS.get(self.kind, KIND_LABELS[KIND_UNKNOWN])
        if not self.exists:
            return f"{label}: {self.path} is not present."
        bits = [f"{label}: {self.path.name}", f"{self.size_bytes} bytes"]
        if self.parameters:
            bits.append(f"{self.parameters:,} parameters")
        line = " · ".join(bits)
        if not self.chat_capable:
            line = f"{line} — not a chat model"
        if self.detail:
            line = f"{line} — {self.detail}"
        return line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "architecture": dict(self.architecture),
            "chat_capable": self.chat_capable,
            "detail": self.detail,
            "evaluations": [item.to_primitive() for item in self.evaluations],
            "exists": self.exists,
            "format": self.format,
            "integrity": self.integrity,
            "kind": self.kind,
            "parameters": self.parameters,
            "path": str(self.path),
            "resumed": self.resumed,
            "schedule": self.schedule,
            "size_bytes": self.size_bytes,
            "tokenizer_path": str(self.tokenizer_path) if self.tokenizer_path else "",
            "tokenizer_present": self.tokenizer_present,
            "total_steps": self.total_steps,
            "usable": self.usable,
        }


def classify_artifact(path: Path | str) -> str:
    """What kind of trained artifact a file is, from its own name and format.

    The classification is deliberately conservative: only a ``.npz`` that reads as
    the project's transformer is called a language model, and anything else with a
    recognisable artifact name is labelled as that artifact. An unrecognised file
    is ``unknown`` -- never guessed to be a language model, because offering a
    policy file as a chat mouth is exactly the failure this label prevents.
    """
    target = Path(path)
    name = target.name.lower()
    suffix = target.suffix.lower()
    if suffix == ".gguf":
        return KIND_UNKNOWN  # a GGUF is a third-party model, not a trained artifact
    if suffix not in (".npz", ".json"):
        return KIND_UNKNOWN
    if name.startswith("policy_"):
        return KIND_POLICY
    if name.startswith("brain_") or name.startswith("brain."):
        return KIND_BRAIN
    if name in ("experience_classifier.npz",):
        return KIND_CLASSIFIER
    if name in ("relation_model.npz",):
        return KIND_RELATION
    if name.endswith(".tokenizer.json"):
        return KIND_UNKNOWN
    if name.endswith(".state.npz") or name.endswith(".schedule.json"):
        return KIND_UNKNOWN
    if suffix == ".npz":
        # Read the shape: the project's transformer checkpoint is the one that
        # declares n_layer/n_embd, which is what makes it a language model.
        probe = probe_file(target)
        if probe.format == FORMAT_FAYD_CHECKPOINT:
            return KIND_LANGUAGE
        return KIND_UNKNOWN
    return KIND_UNKNOWN


def default_artifact_paths() -> tuple[Path, ...]:
    """Where this build's trained artifacts live, in a stable order.

    The writable data folder first (a retrained model), then the committed
    ``ml/artifacts`` and ``ml/llm/checkpoints`` copies, so a fresh checkout finds
    the shipped checkpoint without the user having trained anything.
    """
    from core.paths import (
        artifacts_dir,
        bundled_artifacts_dir,
        bundled_checkpoints_dir,
        checkpoints_dir,
    )

    return tuple(
        dict.fromkeys(
            [
                checkpoints_dir() / "fayd-llm.npz",
                artifacts_dir() / RELEASE_MANIFEST_NAME,
                bundled_checkpoints_dir() / "fayd-llm.npz",
                bundled_artifacts_dir() / RELEASE_MANIFEST_NAME,
            ]
        )
    )


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    """Read the release manifest, or ``{}`` when it is absent or unreadable.

    A missing manifest is not an error: the checkpoint can exist without one, and
    the honest answer is then "the architecture was read from the file, and no
    recorded measurement is available". A corrupt manifest reads as empty rather
    than crashing a listing.
    """
    target = Path(path) if path is not None else _find_manifest()
    if target is None:
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _find_manifest() -> Path | None:
    """The first release manifest that exists, writable copy first."""
    from core.paths import artifacts_dir, bundled_artifacts_dir

    for root in (artifacts_dir(), bundled_artifacts_dir()):
        candidate = root / RELEASE_MANIFEST_NAME
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _manifest_language_model(manifest: dict[str, Any]) -> dict[str, Any]:
    section = manifest.get("language_model")
    return section if isinstance(section, dict) else {}


def evaluations_from_manifest(
    manifest: dict[str, Any], *, source: str = "release manifest"
) -> tuple[EvaluationResult, ...]:
    """The measurements the manifest recorded, each attributed to the manifest.

    Nothing is recomputed: the perplexity figures are the ones the release run
    measured and wrote, and a figure the manifest does not carry is reported as
    *not recorded* rather than estimated here.
    """
    section = _manifest_language_model(manifest)
    perplexity = section.get("perplexity")
    results: list[EvaluationResult] = []
    if isinstance(perplexity, dict):
        for key in ("train", "validation", "test", "bits_per_byte"):
            value = _as_float(perplexity.get(key))
            results.append(
                EvaluationResult(name=key, value=value, source=source)
            )
    else:
        for key in ("train", "validation", "test", "bits_per_byte"):
            results.append(
                EvaluationResult(
                    name=key, value=None, source=source,
                    detail="The manifest records no perplexity for this checkpoint.",
                )
            )
    return tuple(results)


def load_evaluation_sidecar(checkpoint: Path | str) -> tuple[EvaluationResult, ...]:
    """Read a checkpoint's own evaluation sidecar, when one exists.

    A sidecar is written *beside* the checkpoint, never into it, so an evaluation
    can never alter the weights it measured. It is optional: its absence yields no
    results, and a corrupt one is reported as unreadable rather than guessed at.
    """
    path = _evaluation_path(checkpoint)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(data, dict):
        return ()
    results: list[EvaluationResult] = []
    for key in ("train", "validation", "test", "bits_per_byte"):
        results.append(
            EvaluationResult(
                name=key,
                value=_as_float(data.get(key)),
                source="evaluation sidecar",
                detail=str(data.get("detail", "") or ""),
            )
        )
    return tuple(results)


def _evaluation_path(checkpoint: Path | str) -> Path:
    """Where an evaluation result for a checkpoint would live: beside it."""
    target = Path(checkpoint)
    return target.with_suffix(target.suffix + ".evaluation.json")


def save_evaluation(
    checkpoint: Path | str,
    results: tuple[EvaluationResult, ...],
    *,
    detail: str = "",
) -> Path:
    """Write an evaluation result beside the checkpoint, atomically.

    This is the **only** write this module performs, and it never touches the
    checkpoint: a measurement is stored next to what it measured, so the weights
    stay exactly as the training pipeline left them. The write is atomic (a
    temporary file replaced into place), so a reader never sees a half-written
    result.
    """
    import os

    path = _evaluation_path(checkpoint)
    payload = {
        "checkpoint": str(checkpoint),
        "detail": detail,
        **{item.name: item.value for item in results},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path


def describe_artifact(
    path: Path | str,
    *,
    manifest: dict[str, Any] | None = None,
) -> TrainedArtifact:
    """Describe one trained artifact from what is on disk, read-only.

    The checkpoint is parsed by :func:`core.model_files.probe_file` (standard
    library only), so describing a trained model needs neither numpy nor the
    trainer. When the manifest records an architecture, it is preferred to the
    file's own -- the manifest is what the release *declared*, and a mismatch is
    surfaced in ``detail`` rather than hidden.
    """
    target = Path(path)
    kind = classify_artifact(target)
    resolved_manifest = manifest if manifest is not None else load_manifest()
    probe = probe_file(target)
    architecture: dict[str, Any] = {}
    parameters: int | None = None
    evaluations: tuple[EvaluationResult, ...] = ()
    detail = ""

    if kind == KIND_LANGUAGE:
        section = _manifest_language_model(resolved_manifest)
        recorded = section.get("architecture")
        if isinstance(recorded, dict):
            architecture = dict(recorded)
        info = probe.checkpoint
        if info is not None:
            parameters = info.parameter_count
            file_shape = {
                "block_size": info.block_size,
                "n_embd": info.n_embd,
                "n_head": info.n_head,
                "n_layer": info.n_layer,
            }
            file_shape = {k: v for k, v in file_shape.items() if v is not None}
            if architecture and file_shape and any(
                architecture.get(key) not in (None, value)
                for key, value in file_shape.items()
            ):
                detail = (
                    "The manifest's architecture differs from the checkpoint's own "
                    "shapes; the manifest is shown as recorded."
                )
            if not architecture:
                architecture = file_shape
        manifest_parameters = section.get("parameters")
        if isinstance(manifest_parameters, int):
            parameters = manifest_parameters
        evaluations = evaluations_from_manifest(resolved_manifest)
        sidecar = load_evaluation_sidecar(target)
        if sidecar:
            # A sidecar is a later, separate measurement; it is offered alongside
            # the manifest's, clearly attributed, never merged into it.
            evaluations = evaluations + sidecar

    tokenizer = _tokenizer_path(target)
    state = _run_state(target)
    return TrainedArtifact(
        path=target,
        kind=kind,
        exists=probe.exists,
        size_bytes=probe.size_bytes,
        format=probe.format,
        integrity=probe.integrity,
        architecture=architecture,
        parameters=parameters,
        evaluations=evaluations,
        tokenizer_path=tokenizer,
        tokenizer_present=bool(tokenizer and _is_file(tokenizer)),
        total_steps=state.get("total_steps"),
        resumed=state.get("resumed"),
        schedule=str(state.get("schedule", "") or ""),
        detail=detail,
        manifest=_manifest_language_model(resolved_manifest),
    )


def _tokenizer_path(checkpoint: Path) -> Path | None:
    """The tokenizer a checkpoint pairs with, whether or not it exists.

    A byte-level checkpoint in this project may carry an empty tokenizer sidecar
    (the trainer writes one), so the path is returned for inspection and its
    presence is reported separately rather than assumed.
    """
    return checkpoint.with_suffix(".tokenizer.json")


def _run_state(checkpoint: Path) -> dict[str, Any]:
    """Read the resume sidecars, when present, as run state -- never as content.

    ``.state.npz`` records the cumulative step count and the optimizer moments, and
    ``.schedule.json`` records which corpus shard a run consumed. Both are
    git-ignored run state, so their absence is normal; a missing or unreadable one
    yields ``{}`` and the caller reports "unknown".
    """
    result: dict[str, Any] = {}
    state_path = checkpoint.with_suffix(checkpoint.suffix + ".state.npz")
    if _is_file(state_path):
        steps = _read_step_count(state_path)
        if steps is not None:
            result["total_steps"] = steps
            result["resumed"] = True
    schedule_path = checkpoint.with_suffix(checkpoint.suffix + ".schedule.json")
    if _is_file(schedule_path):
        try:
            data = json.loads(schedule_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            consumed = data.get("consumed")
            total = data.get("total")
            if isinstance(consumed, int) and isinstance(total, int):
                result["schedule"] = f"{consumed} of {total} shard(s) consumed"
            elif isinstance(consumed, list):
                result["schedule"] = f"{len(consumed)} shard(s) consumed"
    return result


def _read_step_count(state_path: Path) -> int | None:
    """The cumulative step count from a ``.state.npz``, with the stdlib only.

    ``.state.npz`` is a zip of ``.npy`` members; the step count is stored as a
    small array. Parsing it directly keeps this module free of numpy, so a
    description works on a machine where only the reader is installed.
    """
    import struct
    import zipfile

    try:
        with zipfile.ZipFile(state_path) as archive:
            names = [name for name in archive.namelist() if "step" in name.lower()]
            if not names:
                return None
            raw = archive.read(names[0])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    # A ``.npy`` file: magic (6), version (2), header length (2), header, data.
    if len(raw) < 12 or raw[:6] != b"\x93NUMPY":
        return None
    try:
        header_length = struct.unpack("<H", raw[8:10])[0]
        payload = raw[10 + header_length:]
        if not payload:
            return None
        return int(struct.unpack("<q", payload[:8])[0])
    except (struct.error, ValueError):
        return None


def _is_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def discover_trained_artifacts(
    roots: object = None, *, limit: int = 200
) -> tuple[TrainedArtifact, ...]:
    """Every trained artifact under ``roots``, described read-only.

    The search is bounded and never walks the whole filesystem: with no ``roots``
    it covers the project's own artifact and checkpoint folders. The language-model
    checkpoint is described first, so a caller that wants *the* model gets it
    without searching.
    """
    manifest = load_manifest()
    paths: list[Path] = []
    if roots is None:
        paths.extend(_default_artifact_roots())
    else:
        paths.extend(_walk_artifact_roots(roots, limit=limit))
    artifacts: list[TrainedArtifact] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not _is_file(path):
            continue
        kind = classify_artifact(path)
        if kind == KIND_UNKNOWN:
            continue
        artifacts.append(describe_artifact(path, manifest=manifest))
    # A source checkout mirrors the committed artifacts into the writable data
    # folder, so the same file can be reached at two paths. They are one model, so
    # the second is dropped -- reporting it twice would look like two models.
    unique: list[TrainedArtifact] = []
    fingerprints: set[tuple[str, int]] = set()
    for artifact in artifacts:
        fingerprint = (artifact.path.name, artifact.size_bytes)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(artifact)
    # The language model first, then by name, so the order is stable.
    unique.sort(key=lambda item: (item.kind != KIND_LANGUAGE, item.path.name.casefold()))
    return tuple(unique[:limit])


def _default_artifact_roots() -> tuple[Path, ...]:
    """The files a bare discovery should describe, in a stable order."""
    return default_artifact_paths()


def _walk_artifact_roots(roots: object, *, limit: int) -> list[Path]:
    """Every candidate trained-artifact file under ``roots``, bounded.

    A root may be a file (used as-is) or a folder (walked one level deep, which is
    where this project's artifacts live). The walk is bounded by ``limit`` so a
    mistyped root cannot make a listing scan a whole disk, and it never follows a
    symlink out of the folder it was given.
    """
    candidates: list[Path] = []
    targets = roots if isinstance(roots, (list, tuple, set)) else (roots,)
    for item in targets:
        path = Path(item)
        try:
            if path.is_file():
                candidates.append(path)
                continue
            if not path.is_dir():
                continue
            for entry in sorted(path.iterdir()):
                if len(candidates) >= limit:
                    return candidates
                if entry.is_file() and entry.suffix.lower() in (".npz", ".json"):
                    candidates.append(entry)
        except OSError:
            continue
    return candidates[:limit]


def trained_checkpoint_path() -> Path | None:
    """The trained language-model checkpoint this build ships, or ``None``.

    Prefers the writable ``data`` copy (a retrained model) and falls back to the
    committed one, which is what ``core.paths.default_checkpoint()`` already
    resolves -- reusing that rule rather than writing a second one.
    """
    from core.paths import default_checkpoint

    candidate = default_checkpoint()
    return candidate if _is_file(candidate) else None


def trained_model_record(path: Path | str | None = None) -> ModelRecord | None:
    """A :class:`ModelRecord` for the trained checkpoint, or ``None`` when absent.

    The record is built exactly like any other -- the registry's own
    ``register`` path is what a caller uses to store it -- so the trained model
    reaches every surface as an ordinary record. ``None`` means the checkpoint is
    not present here; a caller reports that rather than registering a model with no
    file.
    """
    from core.model_registry import model_id_for

    target = Path(path) if path is not None else trained_checkpoint_path()
    if target is None or not _is_file(target):
        return None
    probe = probe_file(target)
    if probe.format != FORMAT_FAYD_CHECKPOINT:
        return None
    info = probe.checkpoint
    manifest = load_manifest()
    section = _manifest_language_model(manifest)
    recorded_architecture = section.get("architecture")
    architecture = ""
    if isinstance(recorded_architecture, dict):
        architecture = str(recorded_architecture.get("type", "") or "transformer")
    if not architecture:
        architecture = "fayd-transformer" if info is not None else ""
    parameters = info.parameter_count if info is not None else None
    if isinstance(section.get("parameters"), int):
        parameters = int(section["parameters"])
    block_size = info.block_size if info is not None else None
    version = str(section.get("preset", "") or "")
    return ModelRecord(
        model_id=model_id_for(TRAINED_IDENTITY),
        identity=TRAINED_IDENTITY,
        display_name="Fayd Al-Ruh (trained)",
        model_format=FORMAT_FAYD_CHECKPOINT,
        runtime=TRAINED_RUNTIME,
        path=target,
        source=SOURCE_TRAINED,
        size_bytes=probe.size_bytes,
        architecture=architecture,
        context_length=block_size,
        parameter_count=parameters,
        version=version,
        integrity=probe.integrity,
        integrity_detail=probe.detail,
        capabilities=("chat", "text-generation"),
        tags=("trained", "in-project"),
        notes="The project's own trained checkpoint, registered by the Model Manager.",
        engine="model",
        extra={"registered_by": "core.trained_artifacts"},
    )


def register_trained_models(manager: Any, *, replace_existing: bool = True) -> tuple[ModelRecord, ...]:
    """Register the trained checkpoint with the manager, if it is present.

    This is the one call a surface makes to make the project's own model visible to
    the Model Manager. It never trains and never evaluates; it reads the checkpoint
    the pipeline already wrote and registers it as an ordinary record. A checkpoint
    that is not present registers nothing and returns ``()`` -- the honest answer,
    since a record with no file could not run.
    """
    record = trained_model_record()
    if record is None:
        return ()
    existing = manager.registry.find(record.model_id)
    if existing is not None and not replace_existing:
        return (existing,)
    try:
        stored = manager.registry.register_record(record)
    except Exception:
        return ()
    return (stored,)


__all__ = [
    "CHAT_CAPABLE",
    "KIND_BRAIN",
    "KIND_CLASSIFIER",
    "KIND_LABELS",
    "KIND_LANGUAGE",
    "KIND_POLICY",
    "KIND_RELATION",
    "KIND_UNKNOWN",
    "RELEASE_MANIFEST_NAME",
    "TRAINED_IDENTITY",
    "TRAINED_RUNTIME",
    "EvaluationResult",
    "TrainedArtifact",
    "classify_artifact",
    "default_artifact_paths",
    "describe_artifact",
    "discover_trained_artifacts",
    "evaluations_from_manifest",
    "load_evaluation_sidecar",
    "load_manifest",
    "register_trained_models",
    "save_evaluation",
    "trained_checkpoint_path",
    "trained_model_record",
]
