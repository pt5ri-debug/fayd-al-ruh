"""Bring ready-made ``.npz`` weights in, so the project's own trainers continue them.

The project already trains and *continues* its learned layers -- the language-model
checkpoint resumes from ``fayd-llm.npz`` plus its optimiser-state sidecar, the
petroleum policies resume from ``policy_<task>.npz``, and the base pipeline
resumes from the artifacts it wrote -- but every one of those trainers reads only
from the project's **own** folders. Weights trained elsewhere (a longer run on a
stronger machine, a checkpoint a colleague shared, a file fetched from a URL)
could not be brought in at all: there was no way to hand them to the trainer that
already knows how to continue them.

This module is that missing half. It does three things, and only three:

* **Read** a ``.npz`` file's own structure and say honestly what it is -- the
  transformer language-model checkpoint, its optimiser-state sidecar, a
  Q-learning policy, or one of the small learned models -- reading the array names
  out of the numpy archive with the standard library only (no numpy needed to
  *inspect*), so a file is never guessed to be something it is not.
* **Install** it into the exact folder the project's trainer reads from, keeping
  a backup of whatever it replaces, and writing atomically so a crash cannot leave
  a half-written checkpoint behind.
* **Name the command** that continues it, so the round trip is one step: import
  the weights, then run the trainer that already knows how to resume.

It is deliberately **read-only about the project's training logic**: it never
trains, never touches a trainer, and never rewrites a weight. It moves a file to
where a trainer looks and reports the command. Fetching a file from a URL goes
through the project's existing download engine behind its own permission and
human-approval gates, so this module opens no second path to the network.

Honest by construction:

* An unreadable, corrupt, or unrecognised file is **refused** with the reason, and
  never installed.
* An existing checkpoint is never replaced silently: it is backed up first, and the
  backup's path is reported.
* The architecture and shape that reach the report are read from the file itself,
  never assumed from its name.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.model_files import CheckpointInfo, read_checkpoint_info

#: What a ``.npz`` file is, read from the array names inside it.
KIND_LANGUAGE = "language-model"
KIND_STATE = "optimiser-state"
KIND_POLICY = "policy"
KIND_MODEL = "model"
KIND_UNKNOWN = "unknown"

#: Every kind a caller may name, in the order the listing shows them.
KINDS: tuple[str, ...] = (KIND_LANGUAGE, KIND_STATE, KIND_POLICY, KIND_MODEL)

#: A short, honest sentence per kind, so no surface overstates what a file is.
KIND_BLURBS: dict[str, str] = {
    KIND_LANGUAGE: "The project's transformer language-model checkpoint; continue it with `python -m cli llm --resume`.",
    KIND_STATE: "An optimiser-state sidecar (Adam's moments and the step count); install it beside the checkpoint it belongs to.",
    KIND_POLICY: "A Q-learning policy for one task; continue it with `python -m cli petroleum --resume`.",
    KIND_MODEL: "A small learned model (an MLP classifier or regressor); retrain it with `python -m cli ml`.",
    KIND_UNKNOWN: "Not a weight file this build recognises.",
}

#: The suffix a backup carries. A backup is always a *sibling* of the file it
#: replaced, so restoring it is a plain move.
BACKUP_SUFFIX = ".imported-backup"


class CheckpointImportError(RuntimeError):
    """An import could not be completed. Always carries the real next step."""


# ------------------------------------------------------------------- reading


@dataclass(frozen=True)
class WeightInfo:
    """What a ``.npz`` weight file *is*, read from the file itself.

    ``members`` is the list of arrays the archive holds, which is what makes the
    classification honest: a file is a language-model checkpoint only when it
    really carries ``config`` and ``p0``, never because its name says so.
    """

    path: Path
    kind: str = KIND_UNKNOWN
    valid: bool = False
    detail: str = ""
    members: tuple[str, ...] = ()
    size_bytes: int = 0
    checkpoint: CheckpointInfo | None = None
    #: The task a policy belongs to, when its file name names one.
    task: str = ""
    #: A tokenizer sidecar sitting beside the file, when one does.
    tokenizer_path: Path | None = None
    #: An optimiser-state sidecar sitting beside the file, when one does.
    state_path: Path | None = None

    @property
    def resumable(self) -> bool:
        """Whether a project trainer can *continue* these weights."""
        return self.valid and self.kind in (KIND_LANGUAGE, KIND_POLICY)

    @property
    def parameter_arrays(self) -> int:
        """How many parameter arrays the file carries (a shape, not a truth)."""
        if self.checkpoint is not None:
            return self.checkpoint.parameter_arrays
        return sum(1 for name in self.members if _looks_like_parameter(name))

    def describe(self) -> str:
        bits = [f"{self.path.name}: {self.kind}"]
        if self.size_bytes:
            bits.append(f"{_human(self.size_bytes)}")
        if self.task:
            bits.append(f"task {self.task}")
        if self.checkpoint is not None and self.checkpoint.n_layer is not None:
            bits.append(
                f"{self.checkpoint.n_layer}L/{self.checkpoint.n_embd}w/"
                f"ctx {self.checkpoint.block_size}/vocab {self.checkpoint.vocab_size}"
            )
        head = " · ".join(bits)
        note = self.detail or KIND_BLURBS.get(self.kind, "")
        return f"{head}\n  {note}" if note else head

    def to_primitive(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "valid": self.valid,
            "detail": self.detail,
            "members": list(self.members),
            "size_bytes": self.size_bytes,
            "task": self.task,
            "resumable": self.resumable,
            "parameter_arrays": self.parameter_arrays,
            "checkpoint": self.checkpoint.to_primitive() if self.checkpoint is not None else None,
            "tokenizer_path": str(self.tokenizer_path) if self.tokenizer_path is not None else "",
            "state_path": str(self.state_path) if self.state_path is not None else "",
        }


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _looks_like_parameter(name: str) -> bool:
    """Whether an array name is a model parameter (``p0``, ``W``, ``W1``, ...)."""
    if name.startswith("p") and name[1:].isdigit():
        return True
    return name in {"W", "W1", "W2", "attn_w", "proj_w"}


def npy_members(path: Path | str) -> tuple[str, ...]:
    """The array names inside a numpy ``.npz`` archive, or ``()`` when unreadable.

    Deliberately dependency-free: a ``.npz`` is a zip of ``<name>.npy`` members, so
    the names come from the zip directory and numpy is not needed to *inspect* a
    file -- which matters when the file is untrusted.
    """
    candidate = Path(path)
    try:
        with zipfile.ZipFile(candidate) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return ()
    members: list[str] = []
    for name in names:
        base = name.rsplit("/", 1)[-1]
        if base.endswith(".npy"):
            members.append(base[: -len(".npy")])
    return tuple(members)


def classify_members(members: tuple[str, ...]) -> str:
    """Which kind of weight file a set of array names describes, or ``unknown``.

    Ordered most-specific first, so a language-model checkpoint (which also
    carries ``W``-shaped arrays) is never mistaken for a policy.
    """
    names = set(members)
    if "config" in names and "p0" in names:
        return KIND_LANGUAGE
    if "steps" in names and "t" in names and "m0" in names:
        return KIND_STATE
    if "W" in names and "epsilon" in names and "alpha" in names:
        return KIND_POLICY
    if "kind" in names and ({"W", "W1"} & names):
        return KIND_MODEL
    return KIND_UNKNOWN


def _task_from_name(path: Path) -> str:
    """The task a ``policy_<task>.npz`` file names, or ``""``."""
    name = path.name
    if not name.startswith("policy_"):
        return ""
    return name[len("policy_") :].rsplit(".npz", 1)[0]


def read_weight_info(path: Path | str) -> WeightInfo:
    """Read a ``.npz`` weight file's own structure. Never raises.

    A missing file, a file that is not a numpy archive, and an archive whose
    arrays this build does not recognise each produce a :class:`WeightInfo`
    carrying the reason -- so a caller reports honestly rather than guessing.
    """
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return WeightInfo(
            path=candidate, detail=f"There is no file at {candidate}."
        )
    if candidate.is_dir():
        return WeightInfo(path=candidate, detail=f"{candidate} is a folder, not a file.")
    try:
        size = candidate.stat().st_size
    except OSError as error:
        return WeightInfo(path=candidate, detail=f"Could not read {candidate}: {error}")

    members = npy_members(candidate)
    if not members:
        return WeightInfo(
            path=candidate,
            size_bytes=size,
            detail=(
                "This is not a numpy `.npz` archive. Supported: a Fayd Al-Ruh "
                "checkpoint, its optimiser-state sidecar, a policy, or a small model."
            ),
        )

    kind = classify_members(members)
    if kind == KIND_UNKNOWN:
        return WeightInfo(
            path=candidate,
            kind=KIND_UNKNOWN,
            size_bytes=size,
            members=members,
            detail=(
                "A numpy archive, but none of the weight shapes this build knows "
                f"(found: {', '.join(members[:6])})."
            ),
        )

    checkpoint: CheckpointInfo | None = None
    if kind == KIND_LANGUAGE:
        checkpoint = read_checkpoint_info(candidate)
        if checkpoint is None:
            return WeightInfo(
                path=candidate,
                kind=kind,
                size_bytes=size,
                members=members,
                detail=(
                    "This looks like a checkpoint but its `config` array could not be "
                    "read, so it is not usable. Re-export it or re-download it."
                ),
            )

    tokenizer = candidate.with_suffix(".tokenizer.json")
    state = candidate.with_suffix(".state.npz")
    return WeightInfo(
        path=candidate,
        kind=kind,
        valid=True,
        size_bytes=size,
        members=members,
        checkpoint=checkpoint,
        task=_task_from_name(candidate),
        tokenizer_path=tokenizer if tokenizer.exists() else None,
        state_path=state if state.exists() else None,
        detail=KIND_BLURBS.get(kind, ""),
    )


# ------------------------------------------------------------- destinations


def checkpoint_import_dir() -> Path:
    """Where fetched weight files are staged before they are installed."""
    from core.paths import checkpoint_import_dir as _staging

    return _staging()


def destination_for(
    kind: str,
    *,
    task: str = "",
    name: str = "",
) -> Path:
    """The exact path the project's trainer reads for ``kind``.

    These are the *same* folders the trainers already use -- the checkpoint folder
    the language model resumes from, and the artifacts folder the petroleum
    trainer writes policies into -- so an imported file is picked up with no
    further wiring.
    """
    from core.paths import artifacts_dir, checkpoints_dir

    if kind == KIND_LANGUAGE:
        return checkpoints_dir() / "fayd-llm.npz"
    if kind == KIND_STATE:
        return checkpoints_dir() / "fayd-llm.state.npz"
    if kind == KIND_POLICY:
        if not task:
            raise CheckpointImportError(
                "A policy is imported into a task's own file, so the task must be "
                "named (for example `--task drill-well`)."
            )
        return artifacts_dir() / "petroleum" / f"policy_{task}.npz"
    if kind == KIND_MODEL:
        return artifacts_dir() / (name or "imported_model.npz")
    raise CheckpointImportError(
        f"{kind!r} is not a weight kind this build imports. Known: {', '.join(KINDS)}."
    )


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class ResumePlan:
    """The command that *continues* the imported weights, and what it will do."""

    kind: str
    path: Path
    command: str = ""
    tool: str = ""
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [f"Continue these weights with: {self.command}"] if self.command else []
        lines.extend(self.notes)
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "command": self.command,
            "tool": self.tool,
            "notes": list(self.notes),
        }


def resume_plan(kind: str, *, path: Path | str | None = None, task: str = "") -> ResumePlan:
    """How the project's own tools continue weights of ``kind``.

    Deliberately a *string*, not an action: continuing training is the existing
    trainer's job, and this module must not run it. The plan reports the exact
    command, so the user runs the trainer that already knows how to resume.
    """
    if kind == KIND_LANGUAGE:
        target = Path(path) if path is not None else destination_for(KIND_LANGUAGE)
        notes = [
            "It resumes from the checkpoint (and its optimiser-state sidecar, when "
            "present), so training continues instead of restarting.",
        ]
        return ResumePlan(
            kind=kind,
            path=target,
            command="python -m cli llm --resume",
            tool="ml.llm",
            notes=tuple(notes),
        )
    if kind == KIND_POLICY:
        target = Path(path) if path is not None else Path("")
        notes = [
            "Point it at the folder that holds the policy with `--save-dir` when it "
            "is not the default.",
        ]
        return ResumePlan(
            kind=kind,
            path=target,
            command="python -m cli petroleum --resume",
            tool="ml.petroleum",
            notes=tuple(notes),
        )
    if kind == KIND_STATE:
        target = Path(path) if path is not None else destination_for(KIND_STATE)
        return ResumePlan(
            kind=kind,
            path=target,
            command="python -m cli llm --resume",
            tool="ml.llm",
            notes=(
                "The state sidecar belongs to the checkpoint beside it; run the "
                "language-model resume and it is read automatically.",
            ),
        )
    if kind == KIND_MODEL:
        return ResumePlan(
            kind=kind,
            path=Path(path) if path is not None else Path(""),
            command="python -m cli ml",
            tool="ml",
            notes=(
                "This is a small model, not a resumable checkpoint; the base "
                "pipeline rebuilds it from the seeds.",
            ),
        )
    return ResumePlan(kind=kind, path=Path(path) if path is not None else Path(""))


# ---------------------------------------------------------------- installing


@dataclass(frozen=True)
class ImportReport:
    """The whole outcome of one import, in a shape a surface can post."""

    source: Path
    info: WeightInfo
    destination: Path | None = None
    installed: bool = False
    backup: Path | None = None
    companions: tuple[Path, ...] = ()
    detail: str = ""
    plan: ResumePlan | None = None

    def describe(self) -> str:
        head = "imported" if self.installed else "not imported"
        lines = [f"{self.source.name}: {head} ({self.info.kind})"]
        if self.destination is not None:
            lines.append(f"  installed at: {self.destination}")
        if self.backup is not None:
            lines.append(f"  backup kept:  {self.backup}")
        if self.companions:
            lines.append("  also installed: " + ", ".join(path.name for path in self.companions))
        if self.detail:
            lines.append(f"  {self.detail}")
        if self.installed and self.plan is not None and self.plan.command:
            lines.append(f"  continue with: {self.plan.command}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "info": self.info.to_primitive(),
            "destination": str(self.destination) if self.destination is not None else "",
            "installed": self.installed,
            "backup": str(self.backup) if self.backup is not None else "",
            "companions": [str(path) for path in self.companions],
            "detail": self.detail,
            "plan": self.plan.to_primitive() if self.plan is not None else None,
        }


def _backup_path(target: Path) -> Path:
    """A free sibling name for the file ``target`` is about to replace."""
    candidate = target.with_name(target.name + BACKUP_SUFFIX)
    if not candidate.exists():
        return candidate
    stamp = time.strftime("%Y%m%dT%H%M%S")
    candidate = target.with_name(f"{target.name}{BACKUP_SUFFIX}-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = target.with_name(f"{target.name}{BACKUP_SUFFIX}-{stamp}-{counter}")
        counter += 1
    return candidate


def _atomic_copy(source: Path, target: Path) -> None:
    """Copy ``source`` onto ``target`` atomically, so a crash leaves no half-file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".import-", suffix=".tmp")
    os.close(handle)
    try:
        shutil.copyfile(source, temp_name)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _companions_for(source: Path, kind: str, target: Path) -> tuple[tuple[Path, Path], ...]:
    """The sidecar files that travel with a checkpoint, as (source, target) pairs.

    A language-model checkpoint is only *fully* resumable with the tokenizer it was
    trained with (otherwise a resume would encode with the wrong token index
    space) and, ideally, its optimiser-state sidecar. When those sit beside the
    source they are installed beside the checkpoint, so the imported weights
    continue exactly as they were trained.
    """
    if kind not in (KIND_LANGUAGE, KIND_STATE):
        return ()
    pairs: list[tuple[Path, Path]] = []
    if kind == KIND_LANGUAGE:
        tokenizer = source.with_suffix(".tokenizer.json")
        if tokenizer.exists():
            pairs.append((tokenizer, target.with_suffix(".tokenizer.json")))
        state = source.with_suffix(".state.npz")
        if state.exists():
            pairs.append((state, target.with_suffix(".state.npz")))
    return tuple(pairs)


def install_weights(
    source: Path | str,
    *,
    kind: str = "",
    destination: Path | str | None = None,
    task: str = "",
    name: str = "",
    force: bool = False,
    backup: bool = True,
) -> ImportReport:
    """Install a weight file into the folder the project's trainer reads.

    Validation comes **first**: an unreadable, corrupt, or unrecognised file is
    refused with the reason and nothing is written. An existing target is never
    replaced silently -- it is backed up first (unless ``backup=False``) and the
    backup's path is reported -- and a target that exists is refused outright
    unless ``force=True``, so importing is always a deliberate act.
    """
    info = read_weight_info(source)
    if not info.valid:
        return ImportReport(source=info.path, info=info, detail=info.detail)

    resolved_kind = kind or info.kind
    if resolved_kind != info.kind:
        return ImportReport(
            source=info.path,
            info=info,
            detail=(
                f"The file is a {info.kind!r}, not a {resolved_kind!r}. Import it as "
                f"`--kind {info.kind}` or let the kind be detected."
            ),
        )
    resolved_task = task or info.task
    try:
        target = (
            Path(destination).expanduser()
            if destination is not None
            else destination_for(resolved_kind, task=resolved_task, name=name)
        )
    except CheckpointImportError as error:
        return ImportReport(source=info.path, info=info, detail=str(error))

    if target.resolve() == info.path.resolve():
        return ImportReport(
            source=info.path,
            info=info,
            destination=target,
            detail="The file is already in place, so nothing was copied.",
            plan=resume_plan(resolved_kind, path=target, task=resolved_task),
        )

    saved_backup: Path | None = None
    if target.exists():
        if not force:
            return ImportReport(
                source=info.path,
                info=info,
                destination=target,
                detail=(
                    f"{target} already exists. Re-run with --force to replace it"
                    + (" (a backup will be kept)." if backup else " (no backup).")
                ),
            )
        if backup:
            saved_backup = _backup_path(target)
            try:
                _atomic_copy(target, saved_backup)
            except OSError as error:
                return ImportReport(
                    source=info.path,
                    info=info,
                    destination=target,
                    detail=f"Could not back up {target}: {error}",
                )

    try:
        _atomic_copy(info.path, target)
    except OSError as error:
        return ImportReport(
            source=info.path,
            info=info,
            destination=target,
            backup=saved_backup,
            detail=f"Could not install {info.path} to {target}: {error}",
        )

    companions: list[Path] = []
    for extra_source, extra_target in _companions_for(info.path, resolved_kind, target):
        try:
            _atomic_copy(extra_source, extra_target)
        except OSError as error:
            return ImportReport(
                source=info.path,
                info=info,
                destination=target,
                backup=saved_backup,
                installed=True,
                detail=(
                    f"Installed, but its companion {extra_source.name} could not be "
                    f"copied ({error}); copy it to {extra_target} yourself."
                ),
                plan=resume_plan(resolved_kind, path=target, task=resolved_task),
            )
        companions.append(extra_target)

    return ImportReport(
        source=info.path,
        info=info,
        destination=target,
        installed=True,
        backup=saved_backup,
        companions=tuple(companions),
        plan=resume_plan(resolved_kind, path=target, task=resolved_task),
    )


# ------------------------------------------------------------------ listing


@dataclass(frozen=True)
class InstalledWeights:
    """A weight file the project's trainers will read, and where it lives."""

    kind: str
    path: Path
    info: WeightInfo

    def describe(self) -> str:
        note = KIND_BLURBS.get(self.kind, "")
        return f"{self.path.name} ({self.kind}) at {self.path}" + (f" — {note}" if note else "")

    def to_primitive(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": str(self.path), "info": self.info.to_primitive()}


def list_installed() -> tuple[InstalledWeights, ...]:
    """Every weight file the project's trainers can continue, read from disk.

    The language-model checkpoint (and its state sidecar) and each task's policy
    are listed; a file that does not exist is simply absent, never a placeholder.
    """
    from core.paths import artifacts_dir, checkpoints_dir

    found: list[InstalledWeights] = []
    checkpoint = checkpoints_dir() / "fayd-llm.npz"
    if checkpoint.exists():
        found.append(
            InstalledWeights(KIND_LANGUAGE, checkpoint, read_weight_info(checkpoint))
        )
    state = checkpoints_dir() / "fayd-llm.state.npz"
    if state.exists():
        found.append(InstalledWeights(KIND_STATE, state, read_weight_info(state)))
    petroleum = artifacts_dir() / "petroleum"
    if petroleum.is_dir():
        for policy in sorted(petroleum.glob("policy_*.npz")):
            found.append(InstalledWeights(KIND_POLICY, policy, read_weight_info(policy)))
    return tuple(found)


# ------------------------------------------------------------------- fetching

def fetch_weights(
    url: str,
    *,
    filename: str = "",
    destination: Path | str | None = None,
    task: str = "",
    name: str = "",
    force: bool = False,
    approved: bool = False,
    transport: Any = None,
) -> ImportReport:
    """Fetch a weight file over HTTPS and install it, behind the project's gates.

    A fetch leaves the machine, so it passes the project's permission policy *and*
    a human approval before a byte is transferred -- ``approved=True`` is the
    caller's explicit approval, and the engine's own
    :mod:`core.model_download` gates still apply underneath. Nothing here opens a
    second path to the network.
    """
    from core.model_download import (
        DownloadError,
        DownloadRequest,
        ModelDownloader,
    )
    from core.model_sources import ModelFile, validate_download_url

    problem = validate_download_url(url)
    if problem:
        info = WeightInfo(path=Path(url), detail=problem)
        return ImportReport(source=info.path, info=info, detail=problem)

    resolved_name = filename or url.rsplit("/", 1)[-1].split("?", 1)[0] or "weights.npz"
    if not resolved_name.lower().endswith(".npz"):
        info = WeightInfo(
            path=Path(resolved_name),
            detail="A weight fetch expects a `.npz` file.",
        )
        return ImportReport(source=info.path, info=info, detail=info.detail)

    staging = Path(destination).expanduser() if destination is not None else checkpoint_import_dir()
    request = DownloadRequest(
        destination=staging,
        files=(ModelFile(filename=resolved_name, url=url),),
        source_id=url,
    )
    downloader = ModelDownloader(transport=transport, approved=approved)
    try:
        report = downloader.download(request)
    except DownloadError as error:
        info = WeightInfo(path=Path(resolved_name), detail=str(error))
        return ImportReport(source=info.path, info=info, detail=str(error))

    if not report.complete:
        info = WeightInfo(path=Path(resolved_name), detail=report.detail)
        return ImportReport(
            source=info.path,
            info=info,
            detail=f"The download did not complete: {report.detail}",
        )
    promoted = next(
        (
            item
            for item in report.files
            if getattr(item, "ok", False) and getattr(item, "path", None) is not None
        ),
        None,
    )
    if promoted is None or promoted.path is None:
        info = WeightInfo(path=Path(resolved_name), detail="The download produced no file.")
        return ImportReport(source=info.path, info=info, detail=info.detail)

    return install_weights(
        promoted.path,
        destination=None,
        task=task,
        name=name,
        force=force,
    )


__all__ = [
    "BACKUP_SUFFIX",
    "KINDS",
    "KIND_BLURBS",
    "KIND_LANGUAGE",
    "KIND_MODEL",
    "KIND_POLICY",
    "KIND_STATE",
    "KIND_UNKNOWN",
    "CheckpointImportError",
    "ImportReport",
    "InstalledWeights",
    "ResumePlan",
    "WeightInfo",
    "checkpoint_import_dir",
    "classify_members",
    "destination_for",
    "fetch_weights",
    "install_weights",
    "list_installed",
    "npy_members",
    "read_weight_info",
    "resume_plan",
]
