"""The **shared** language-model folder: models every character may use.

Each character already owns a folder of its own weights (``data/ultron``,
``data/vision``, ``data/presentation``, ``data/natasha``, ``data/hakeem``, ...).
That is right for a *specialist* mouth, but a small **general** model is useful to
every one of them -- the companion, the scholar, and the answering mouths all run
through the project's one in-process runtime (``llama-cpp-python``). Keeping a
separate copy of the same weights per character would waste gigabytes for nothing.

This module is that public folder: ``data/shared_models``
(:func:`core.paths.shared_models_dir`, git-ignored). A model downloaded here is
offered to **every** GGUF mouth (:mod:`core.language_models`) and is also the
*fallback* a character's own model layer looks in when its own folder is empty, so
a shared download works everywhere with no per-character step.

Three models ship in the catalog (all general, small, and Apache-2.0):

* ``qwen3.5-0.8b`` -- the lightest, replies in a second or two on a CPU;
* ``qwen3.5-2b`` -- the default: a good balance of quality and speed;
* ``qwen3-1.7b`` -- **moved here** from Natasha's folder, because it is general
  rather than specific to the companion. :func:`migrate_from_natasha` performs
  that move on an upgrading machine.

Honest by construction, like every other model layer:

* :func:`shared_model_status` reports *exactly* what is installed, and the real
  next step when nothing is;
* :func:`download_model` leaves the machine, so it runs only through the project's
  permission and human-approval gates, denied by default;
* :func:`migrate_from_natasha` never copies or deletes anything it cannot verify,
  and is a no-op when the file is not there.

Nothing here loads or executes a model: it is file bookkeeping and a governed
fetch, so it is dependency-free and testable with no model and no network.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_SHARED, pip_install_command
from core.paths import natasha_models_dir, shared_models_dir
from core.permissions import Capability, PermissionPolicy

#: The name used in labels and status lines.
NAME = "Shared language models"

#: The external-tool name the permission/approval gates see for a download.
DOWNLOAD_TOOL = "shared_model_download"

#: The registry file, written after a download so the newest choice is remembered.
REGISTRY_FILENAME = "model.json"

#: The folder a legacy model may be moved *out of* (Natasha's own).
LEGACY_SOURCE = natasha_models_dir


class SharedModelUnavailable(RuntimeError):
    """Raised when a shared model cannot be fetched (denied, or no hub installed)."""


@dataclass(frozen=True)
class SharedModelSpec:
    """A known-good general model the shared folder can fetch.

    A spec bundles the Hugging Face repo, the exact GGUF filename, and a short
    description, so "a small general model" is a named choice rather than a URL the
    user has to assemble.
    """

    key: str
    title: str
    repo: str
    filename: str
    params: str
    approx_size_gb: float
    summary: str


#: The download catalog, keyed by name. ``qwen3.5-2b`` is the default: it holds a
#: short conversation well and still replies in seconds on a CPU. ``qwen3.5-0.8b``
#: is the fastest; ``qwen3-1.7b`` is the model moved here from Natasha's folder.
CATALOG: dict[str, SharedModelSpec] = {
    "qwen3.5-0.8b": SharedModelSpec(
        key="qwen3.5-0.8b",
        title="Qwen3.5 0.8B (lightest)",
        repo="unsloth/Qwen3.5-0.8B-GGUF",
        filename="Qwen3.5-0.8B-Q4_K_M.gguf",
        params="0.8B",
        approx_size_gb=0.5,
        summary="The lightest general model: a reply in a second or two on a CPU.",
    ),
    "qwen3.5-2b": SharedModelSpec(
        key="qwen3.5-2b",
        title="Qwen3.5 2B (default)",
        repo="unsloth/Qwen3.5-2B-GGUF",
        filename="Qwen3.5-2B-Q4_K_M.gguf",
        params="2B",
        approx_size_gb=1.2,
        summary="A good balance of quality and speed; the shared default.",
    ),
    "qwen3-1.7b": SharedModelSpec(
        key="qwen3-1.7b",
        title="Qwen3 1.7B (moved from Natasha)",
        repo="unsloth/Qwen3-1.7B-GGUF",
        filename="Qwen3-1.7B-Q4_K_M.gguf",
        params="1.7B",
        approx_size_gb=1.1,
        summary="A general Qwen3 model, shared by every character (was Natasha's).",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "qwen3.5-2b"


def resolve_spec(name: str | None) -> SharedModelSpec:
    """The spec for ``name``, or the default when it is empty. Raises on a bad name."""
    key = (name or "").strip().lower()
    if not key:
        return CATALOG[DEFAULT_SPEC_KEY]
    if key not in CATALOG:
        raise SharedModelUnavailable(
            f"Unknown shared model {name!r}. Available: {', '.join(sorted(CATALOG))}."
        )
    return CATALOG[key]


def spec_names() -> tuple[str, ...]:
    """Every catalog key, sorted, for a menu."""
    return tuple(sorted(CATALOG))


def spec_summaries() -> tuple[tuple[str, str, float, str], ...]:
    """``(key, title, size in GB, summary)`` per catalog entry, for a menu."""
    return tuple(
        (spec.key, spec.title, spec.approx_size_gb, spec.summary)
        for spec in (CATALOG[key] for key in spec_names())
    )


# ----------------------------------------------------------------- model files


def _shared_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else shared_models_dir()


def _registry_file(root: Path | str | None = None) -> Path:
    return _shared_dir(root) / REGISTRY_FILENAME


def _find_gguf(directory: Path) -> Path | None:
    """The registry's file in ``directory``, else its largest ``.gguf``."""
    registry = directory / REGISTRY_FILENAME
    if registry.is_file():
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        name = data.get("file") if isinstance(data, dict) else None
        if name:
            candidate = directory / str(name)
            if candidate.is_file():
                return candidate
    candidates = sorted(
        (path for path in directory.glob("*.gguf") if path.is_file()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_model_file(root: Path | str | None = None) -> Path | None:
    """The shared model a character should fall back to, or ``None``.

    Prefers the file named in the registry, then the largest ``.gguf`` in the
    shared folder -- the biggest file is almost always the highest-quality model
    the user meant to use.
    """
    return _find_gguf(_shared_dir(root))


def model_is_cached(root: Path | str | None = None) -> bool:
    """True when at least one shared model file is present on disk."""
    return find_model_file(root) is not None


@dataclass(frozen=True)
class SharedModel:
    """One ``.gguf`` file in the shared folder, with its real availability."""

    path: Path
    size_bytes: int

    @property
    def title(self) -> str:
        return self.path.name

    def available(self) -> bool:
        return self.path.is_file()

    def describe(self) -> str:
        return f"{self.path.name} ({self.size_bytes / 1e9:.1f} GB)"


def list_models(root: Path | str | None = None) -> tuple[SharedModel, ...]:
    """Every ``.gguf`` in the shared folder, largest first.

    An empty or missing folder yields an empty tuple rather than raising, so a
    status line can report honestly.
    """
    directory = _shared_dir(root)
    if not directory.is_dir():
        return ()
    found = [
        SharedModel(path=path, size_bytes=path.stat().st_size)
        for path in directory.glob("*.gguf")
        if path.is_file()
    ]
    found.sort(key=lambda model: model.size_bytes, reverse=True)
    return tuple(found)


def installed_spec_keys(root: Path | str | None = None) -> tuple[str, ...]:
    """The catalog keys whose file is present in the shared folder."""
    directory = _shared_dir(root)
    return tuple(
        key for key in spec_names() if (directory / CATALOG[key].filename).is_file()
    )


# ---------------------------------------------------------------------- status


@dataclass
class SharedModelStatus:
    """An honest report of what is installed in the shared folder."""

    available: bool
    folder: Path
    models: tuple[str, ...] = ()
    detail: str = ""
    hint: str = "python -m cli shared-models download --approve"

    def summary(self) -> str:
        """A one-line, plain-language status for the CLI and the window."""
        if self.available:
            return (
                f"{NAME}: {len(self.models)} model(s) ready in {self.folder} "
                f"({', '.join(self.models)})."
            )
        return (
            f"{NAME}: no model is downloaded yet in {self.folder}. "
            f"Fetch one with `{self.hint}`."
        )


def shared_model_status(root: Path | str | None = None) -> SharedModelStatus:
    """Report what the shared folder holds right now, without pretending."""
    directory = _shared_dir(root)
    models = list_models(root)
    return SharedModelStatus(
        available=bool(models),
        folder=directory,
        models=tuple(model.title for model in models),
        detail="" if models else "No shared model file is present.",
    )


# -------------------------------------------------------------------- download


#: Runs one command and returns ``(returncode, combined_output)``.
CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def _gate(
    tool: str,
    *,
    approved: bool,
    permissions: PermissionPolicy | None,
    approvals: HumanApprovalGate | None,
) -> None:
    """Raise unless the permission policy *and* a human approval both allow ``tool``."""
    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({tool}),
        )
        approvals = approvals or _approve(tool)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, tool)
    if not decision.allowed:
        raise SharedModelUnavailable(f"Denied by policy ({decision.reason}).")
    approval = gate.decide(tool)
    if not approval.approved:
        raise SharedModelUnavailable(f"Needs human approval ({approval.reason}).")


def download_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download a shared model's **GGUF weights** into the shared folder.

    The download leaves the machine, so it is treated as an external tool: denied
    by default, and it runs only when an approving :class:`PermissionPolicy` *and*
    a live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the
    CLI's convenience that records a human approval). Returns the model file path.
    """
    spec = resolve_spec(spec_key)
    _gate(DOWNLOAD_TOOL, approved=approved, permissions=permissions, approvals=approvals)

    directory = _shared_dir(root)
    target = directory / spec.filename
    if target.is_file():
        return target

    try:
        import huggingface_hub
    except ImportError as error:
        raise SharedModelUnavailable(
            "Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_SHARED)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…")
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
    except Exception as error:
        raise SharedModelUnavailable(f"Could not download {spec.key!r}: {error}") from error
    _write_registry(directory, Path(path).name, spec.key)
    if progress is not None:
        progress(f"Saved {spec.title} to {path}.")
    return Path(path)


def download_all(
    *,
    root: Path | str | None = None,
    approved: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Fetch every catalog model into the shared folder (governed, one approval).

    Convenience for "download the shared models": each fetch is the same governed
    :func:`download_model`, so one approval covers the whole set. A model already on
    disk is skipped.
    """
    return tuple(
        download_model(key, root=root, approved=approved, progress=progress)
        for key in spec_names()
    )


def _write_registry(directory: Path, filename: str, spec_key: str) -> None:
    """Record the newest download, so the largest-file guess is not needed."""
    try:
        (directory / REGISTRY_FILENAME).write_text(
            json.dumps({"file": filename, "spec": spec_key}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _approve(tool: str) -> HumanApprovalGate:
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


# -------------------------------------------------------------------- migration


@dataclass
class MigrationReport:
    """What :func:`migrate_from_natasha` did (or why it did nothing)."""

    moved: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    detail: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.moved)

    def summary(self) -> str:
        if self.moved:
            return f"Moved {len(self.moved)} model(s) from Natasha's folder to the shared one."
        return self.detail or "Nothing to move."


def _find_legacy_file(src_dir: Path, spec: SharedModelSpec) -> Path | None:
    """The legacy copy of ``spec`` in ``src_dir``, matched loosely by name.

    A user's own copy is often named the way their previous download named it
    (``qwen3-1.7b-q4_k_m.gguf``) rather than the Hugging Face spelling
    (``Qwen3-1.7B-Q4_K_M.gguf``), so an exact miss falls back to a case- and
    separator-insensitive comparison of the file's stem. Only a name that still
    identifies *this* model matches, so another model in the folder is left alone.
    """
    exact = src_dir / spec.filename
    if exact.is_file():
        return exact
    if not src_dir.is_dir():
        return None
    wanted = _loose_stem(spec.filename)
    for candidate in sorted(src_dir.glob("*.gguf")):
        if candidate.is_file() and _loose_stem(candidate.name) == wanted:
            return candidate
    return None


def _loose_stem(filename: str) -> str:
    """A filename reduced to letters and digits, lower-cased, for a loose match."""
    return "".join(ch for ch in Path(filename).stem.lower() if ch.isalnum())


def migrate_from_natasha(
    *,
    source: Path | str | None = None,
    root: Path | str | None = None,
    copy: bool = False,
) -> MigrationReport:
    """Move a legacy general model from Natasha's folder into the shared one.

    ``qwen3-1.7b`` used to live in ``data/natasha``; it is general, so it now
    belongs in the shared folder. This moves it (a real ``shutil.move``, so no disk
    space is doubled) when it is present and the shared copy is not. The file is
    found even when it was named differently from the catalog's spelling (see
    :func:`_find_legacy_file`). It never deletes a file it could not move, and it
    is a no-op when there is nothing to do. ``copy=True`` copies instead of moving,
    for a caller that wants the original kept. Returns a :class:`MigrationReport`.
    """
    src_dir = Path(source) if source is not None else natasha_models_dir()
    dst_dir = _shared_dir(root)
    moved: list[str] = []
    skipped: list[str] = []
    for spec in CATALOG.values():
        origin = _find_legacy_file(src_dir, spec)
        if origin is None:
            continue
        destination = dst_dir / spec.filename
        if destination.is_file():
            skipped.append(origin.name)
            continue
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copy2(origin, destination)
            else:
                shutil.move(str(origin), str(destination))
        except OSError as error:
            skipped.append(origin.name)
            return MigrationReport(
                moved=tuple(moved),
                skipped=tuple(skipped),
                detail=f"Could not move {origin.name}: {error}",
            )
        moved.append(origin.name)
    detail = "" if moved else "No legacy model was found in Natasha's folder."
    return MigrationReport(moved=tuple(moved), skipped=tuple(skipped), detail=detail)


def migrate_legacy_models() -> MigrationReport:
    """One-time upgrade step: gather a legacy general model into the shared folder.

    Best-effort and silent for the program's own startup (a read-only data folder
    must not crash a run); the CLI calls :func:`migrate_from_natasha` directly when
    it wants the report.
    """
    try:
        return migrate_from_natasha()
    except Exception:
        return MigrationReport(detail="Migration was skipped.")


__all__ = [
    "CATALOG",
    "DEFAULT_SPEC_KEY",
    "DOWNLOAD_TOOL",
    "LEGACY_SOURCE",
    "NAME",
    "REGISTRY_FILENAME",
    "CommandRunner",
    "MigrationReport",
    "SharedModel",
    "SharedModelSpec",
    "SharedModelStatus",
    "SharedModelUnavailable",
    "download_all",
    "download_model",
    "find_model_file",
    "installed_spec_keys",
    "list_models",
    "migrate_from_natasha",
    "migrate_legacy_models",
    "model_is_cached",
    "resolve_spec",
    "shared_model_status",
    "spec_names",
    "spec_summaries",
]
