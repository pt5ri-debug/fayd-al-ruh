"""The committed pack of ready-made weights the project ships and can continue.

:mod:`core.checkpoint_import` already knows how to *read*, *install*, and *name
the resume command* for a single ready-made ``.npz`` file. What it did not have
is a **pack**: a curated, versioned set of weights that ships with a clone --
committed under the tracked ``weights/`` folder rather than the git-ignored
``data/`` one -- with the provenance that makes each entry trustworthy and a
verifier that proves the file is what it claims before anything is installed.

This module is that pack, and it does three things, and only three:

* **Describe** each entry from a small JSON manifest (its key, its kind, the file
  it lives in, its sidecars, the digest it must have, the architecture it must
  carry, how it was produced, and the exact command that continues it). The
  manifest is the *one* source of truth, so a surface never invents a weight.
* **Verify** an entry against the file on disk -- the file exists, its bytes hash
  to the recorded digest, its own array names classify to the recorded kind, and
  its declared architecture matches what the file actually carries -- so a
  truncated download, a renamed file, or a corrupted checkpoint is **refused**
  with the reason rather than installed.
* **Install** a verified entry into the exact folder the project's trainer reads
  from, through the *same* :func:`core.checkpoint_import.install_weights`, so the
  pack adds no second copy path and an existing file is backed up rather than
  clobbered.

It is deliberately **read-only about the project's training logic**: it never
trains, never touches a trainer, and never rewrites a weight. Installing a pack
entry moves a file to where a trainer already looks, and the pack reports the
command that continues it.

Honest by construction:

* A missing pack, an unreadable manifest, and an unknown entry key are each
  **reported** with the real next step, never raised at a caller that only wanted
  to list what is available.
* A digest mismatch, an unknown kind, and an architecture that does not match the
  file are **refusals**, and nothing is written.
* The digest and the architecture that reach the report are read from the file
  itself (and the manifest), never guessed from a file name.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.checkpoint_import import (
    KIND_LANGUAGE,
    KIND_POLICY,
    KINDS,
    install_weights,
    read_weight_info,
)

#: The manifest file inside the pack folder.
MANIFEST_NAME = "manifest.json"
#: The manifest schema version this build understands.
MANIFEST_VERSION = 1
#: The digest algorithm used for the pack. ``sha256`` is in the standard library
#: and is the one a reader can verify with any tool.
DIGEST_ALGORITHM = "sha256"

#: The verdicts a verification can reach.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_DIGEST_MISMATCH = "digest-mismatch"
STATUS_WRONG_KIND = "wrong-kind"
STATUS_INCOMPATIBLE = "incompatible"
STATUS_UNREADABLE = "unreadable"


class WeightsPackError(RuntimeError):
    """The pack could not be read or an entry could not be acted on."""


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def file_digest(path: Path | str, *, algorithm: str = DIGEST_ALGORITHM) -> str:
    """The hex digest of a file's bytes, streamed so a large file is not held whole."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------- the entry


@dataclass(frozen=True)
class PackEntry:
    """One ready-made weight file the pack ships, described by the manifest."""

    key: str
    title: str
    kind: str
    path: Path
    #: Sidecar files that travel with the entry (a tokenizer, an optimiser state).
    companions: tuple[Path, ...] = ()
    #: The digest the file's bytes must hash to, or ``""`` when the pack records
    #: none (an honest "not pinned", never a silent pass).
    digest: str = ""
    #: The task a policy belongs to.
    task: str = ""
    #: The architecture the file must carry (a language checkpoint's ``config``).
    architecture: dict[str, Any] = field(default_factory=dict)
    #: How the weights were produced, in the producer's own words.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: The command that continues the weights once installed.
    continue_command: str = ""
    notes: str = ""

    @property
    def resumable(self) -> bool:
        """Whether a project trainer can *continue* this entry's weights."""
        return self.kind in (KIND_LANGUAGE, KIND_POLICY)

    def describe(self) -> str:
        bits = [f"{self.key}: {self.kind}"]
        if self.path.name:
            bits.append(self.path.name)
        if self.task:
            bits.append(f"task {self.task}")
        if self.architecture.get("n_layer") is not None:
            bits.append(
                f"{self.architecture.get('n_layer')}L/"
                f"{self.architecture.get('n_embd')}w/"
                f"ctx {self.architecture.get('block_size')}/"
                f"vocab {self.architecture.get('vocab_size')}"
            )
        head = " · ".join(bits)
        return f"{head}\n  {self.notes}" if self.notes else head

    def to_primitive(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind,
            "path": str(self.path),
            "companions": [str(item) for item in self.companions],
            "digest": self.digest,
            "task": self.task,
            "architecture": dict(self.architecture),
            "provenance": dict(self.provenance),
            "continue_command": self.continue_command,
            "notes": self.notes,
            "resumable": self.resumable,
        }


def _entry_from_dict(base: Path, data: dict[str, Any]) -> PackEntry:
    """Build an entry from one manifest row, resolving its paths against ``base``.

    ``base`` is the folder the manifest's relative paths are read from -- the
    repository/bundle root by default, so the pack can *index* the weights the
    project already ships (one copy) rather than duplicating a multi-megabyte
    file. A pack may also hold its own files (an optimiser-state sidecar, say)
    under ``weights/``; a path is relative to the root either way.
    """
    relative = str(data.get("path", "")).strip()
    if not relative:
        raise WeightsPackError("A pack entry must name the file it lives in.")
    companions = tuple(
        base / str(item) for item in data.get("companions", ()) if str(item).strip()
    )
    kind = str(data.get("kind", "")).strip()
    if kind not in KINDS:
        raise WeightsPackError(
            f"Pack entry {data.get('key', '?')!r} names kind {kind!r}; expected one "
            f"of {', '.join(KINDS)}."
        )
    return PackEntry(
        key=str(data.get("key", "")).strip(),
        title=str(data.get("title", "")).strip(),
        kind=kind,
        path=base / relative,
        companions=companions,
        digest=str(data.get("digest", "")).strip(),
        task=str(data.get("task", "")).strip(),
        architecture=dict(data.get("architecture", {}) or {}),
        provenance=dict(data.get("provenance", {}) or {}),
        continue_command=str(data.get("continue_command", "")).strip(),
        notes=str(data.get("notes", "")).strip(),
    )


# ----------------------------------------------------------------- the pack


@dataclass(frozen=True)
class WeightsPack:
    """A loaded pack: its folder, its version, and its entries."""

    root: Path
    version: int
    project_version: str = ""
    created_at: str = ""
    entries: tuple[PackEntry, ...] = ()
    detail: str = ""

    @property
    def available(self) -> bool:
        """Whether a real manifest was read (a missing pack is not available)."""
        return self.version > 0 and bool(self.entries)

    def get(self, key: str) -> PackEntry | None:
        """The entry named ``key``, or ``None`` -- never a made-up entry."""
        wanted = key.strip().lower()
        for entry in self.entries:
            if entry.key.lower() == wanted:
                return entry
        return None

    def keys(self) -> tuple[str, ...]:
        return tuple(entry.key for entry in self.entries)

    def describe(self) -> str:
        if not self.available:
            return self.detail or "No weights pack was found."
        lines = [
            f"Weights pack v{self.version}"
            + (f" (project {self.project_version})" if self.project_version else ""),
            f"  folder: {self.root}",
            f"  entries: {len(self.entries)}",
        ]
        for entry in self.entries:
            lines.append(f"  - {entry.describe()}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "version": self.version,
            "project_version": self.project_version,
            "created_at": self.created_at,
            "available": self.available,
            "detail": self.detail,
            "entries": [entry.to_primitive() for entry in self.entries],
        }


def load_pack(
    root: Path | str | None = None,
    *,
    path_base: Path | str | None = None,
) -> WeightsPack:
    """Read the pack's manifest. Never raises -- a missing or broken pack is reported.

    ``root`` is the folder the manifest lives in (``weights/`` by default).
    ``path_base`` is the folder an entry's relative paths are read from -- the
    pack folder's **parent** by default (the repository/bundle root for the
    shipped ``weights/``), so an entry can *index* a weight the project already
    ships (``ml/llm/checkpoints/fayd-llm.npz``) with no second copy, while a file
    only the pack carries (an optimiser-state sidecar, say) lives under
    ``weights/`` and is referenced the same way. The manifest is the one source of
    truth for what the pack holds, so a caller lists exactly what ships rather
    than scanning a folder and guessing.
    """
    from core.paths import weights_pack_dir

    base = Path(root).expanduser() if root is not None else weights_pack_dir()
    resolve_from = (
        Path(path_base).expanduser() if path_base is not None else base.parent
    )
    manifest = base / MANIFEST_NAME
    if not manifest.exists():
        return WeightsPack(
            root=base,
            version=0,
            detail=(
                f"No weights pack at {manifest}. The project ships one under "
                "`weights/`; restore it from the repository or regenerate it."
            ),
        )
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return WeightsPack(
            root=base,
            version=0,
            detail=f"Could not read {manifest}: {error}",
        )
    if not isinstance(data, dict):
        return WeightsPack(
            root=base, version=0, detail=f"{manifest} is not a pack manifest object."
        )
    version = int(data.get("version", 0) or 0)
    if version != MANIFEST_VERSION:
        return WeightsPack(
            root=base,
            version=0,
            detail=(
                f"The pack manifest is version {version}, but this build understands "
                f"version {MANIFEST_VERSION}. Update the program or the pack."
            ),
        )
    entries: list[PackEntry] = []
    for row in data.get("entries", ()):
        if not isinstance(row, dict):
            continue
        try:
            entries.append(_entry_from_dict(resolve_from, row))
        except WeightsPackError as error:
            return WeightsPack(root=base, version=0, detail=str(error))
    return WeightsPack(
        root=base,
        version=version,
        project_version=str(data.get("project_version", "")),
        created_at=str(data.get("created_at", "")),
        entries=tuple(entries),
    )


# ------------------------------------------------------------ verification


@dataclass(frozen=True)
class Verification:
    """The result of checking one entry against the file on disk."""

    entry: PackEntry
    status: str = STATUS_OK
    detail: str = ""
    digest: str = ""
    size_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def describe(self) -> str:
        mark = "OK" if self.ok else "REFUSED"
        head = f"[{mark}] {self.entry.key} ({self.entry.kind})"
        bits = []
        if self.size_bytes:
            bits.append(_human(self.size_bytes))
        if self.digest:
            bits.append(f"{DIGEST_ALGORITHM} {self.digest[:12]}")
        line = head + (" · " + " · ".join(bits) if bits else "")
        return f"{line}\n  {self.detail}" if self.detail else line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "key": self.entry.key,
            "kind": self.entry.kind,
            "status": self.status,
            "ok": self.ok,
            "detail": self.detail,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "path": str(self.entry.path),
        }


def _check_architecture(entry: PackEntry, info: Any) -> str:
    """Compare the declared architecture to what the file carries; ``""`` when it fits.

    An entry that pins no architecture is not checked, which is the honest
    "not pinned" -- the digest and the kind are still verified.
    """
    declared = entry.architecture
    if not declared:
        return ""
    if entry.kind != KIND_LANGUAGE or info.checkpoint is None:
        return ""
    checkpoint = info.checkpoint
    for field_name in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size"):
        expected = declared.get(field_name)
        if expected is None:
            continue
        actual = getattr(checkpoint, field_name, None)
        if actual is None:
            continue
        if int(expected) != int(actual):
            return (
                f"The file's {field_name} is {actual}, but the pack records "
                f"{expected}; this is a different architecture."
            )
    return ""


def verify_entry(entry: PackEntry) -> Verification:
    """Check one entry against its file: exists, hashes, kind, architecture.

    A verification is a *refusal* the moment anything does not line up, so a
    caller never installs a file the pack cannot vouch for.
    """
    path = entry.path
    if not path.exists():
        return Verification(
            entry=entry,
            status=STATUS_MISSING,
            detail=f"There is no file at {path}.",
        )
    try:
        size = path.stat().st_size
    except OSError as error:
        return Verification(
            entry=entry, status=STATUS_UNREADABLE, detail=f"Could not read {path}: {error}"
        )

    if entry.digest:
        try:
            actual = file_digest(path)
        except OSError as error:
            return Verification(
                entry=entry,
                status=STATUS_UNREADABLE,
                size_bytes=size,
                detail=f"Could not hash {path}: {error}",
            )
        if actual.lower() != entry.digest.lower():
            return Verification(
                entry=entry,
                status=STATUS_DIGEST_MISMATCH,
                size_bytes=size,
                digest=actual,
                detail=(
                    f"The file hashes to {actual[:16]}, but the pack records "
                    f"{entry.digest[:16]}. The file is not the one the pack ships; "
                    "restore it or regenerate the pack."
                ),
            )

    info = read_weight_info(path)
    if not info.valid or info.kind != entry.kind:
        return Verification(
            entry=entry,
            status=STATUS_WRONG_KIND,
            size_bytes=size,
            digest=entry.digest,
            detail=(
                f"The file reads as {info.kind!r}, but the pack records {entry.kind!r}. "
                f"{info.detail}"
            ).strip(),
        )

    problem = _check_architecture(entry, info)
    if problem:
        return Verification(
            entry=entry,
            status=STATUS_INCOMPATIBLE,
            size_bytes=size,
            digest=entry.digest,
            detail=problem,
        )

    # A language checkpoint is only *fully* resumable with the tokenizer it was
    # trained with; a missing sidecar means a resume would encode with the wrong
    # token index space, so it is reported as incompatible rather than installed.
    if entry.kind == KIND_LANGUAGE and entry.companions:
        for companion in entry.companions:
            if not companion.exists():
                return Verification(
                    entry=entry,
                    status=STATUS_INCOMPATIBLE,
                    size_bytes=size,
                    digest=entry.digest,
                    detail=(
                        f"The sidecar {companion.name} is missing, so a resume would "
                        "encode with the wrong token index space. Restore it."
                    ),
                )

    return Verification(
        entry=entry,
        status=STATUS_OK,
        size_bytes=size,
        digest=entry.digest,
        detail="The file matches the pack's digest, kind, and architecture.",
    )


def verify_pack(pack: WeightsPack) -> tuple[Verification, ...]:
    """Verify every entry in a pack, in manifest order."""
    return tuple(verify_entry(entry) for entry in pack.entries)


# -------------------------------------------------------------- installing


@dataclass(frozen=True)
class InstallOutcome:
    """What installing one pack entry did."""

    entry: PackEntry
    verification: Verification
    installed: bool = False
    destination: Path | None = None
    backup: Path | None = None
    companions: tuple[Path, ...] = ()
    detail: str = ""
    continue_command: str = ""

    def describe(self) -> str:
        head = "installed" if self.installed else "not installed"
        lines = [f"{self.entry.key}: {head}"]
        if self.destination is not None:
            lines.append(f"  installed at: {self.destination}")
        if self.backup is not None:
            lines.append(f"  backup kept:  {self.backup}")
        if self.companions:
            lines.append("  also installed: " + ", ".join(item.name for item in self.companions))
        if self.detail:
            lines.append(f"  {self.detail}")
        if self.installed and self.continue_command:
            lines.append(f"  continue with: {self.continue_command}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "key": self.entry.key,
            "kind": self.entry.kind,
            "verification": self.verification.to_primitive(),
            "installed": self.installed,
            "destination": str(self.destination) if self.destination is not None else "",
            "backup": str(self.backup) if self.backup is not None else "",
            "companions": [str(item) for item in self.companions],
            "detail": self.detail,
            "continue_command": self.continue_command,
        }


@dataclass(frozen=True)
class PackInstallReport:
    """The whole outcome of one install run over the pack."""

    outcomes: tuple[InstallOutcome, ...] = ()
    detail: str = ""

    @property
    def installed(self) -> int:
        return sum(1 for item in self.outcomes if item.installed)

    @property
    def refused(self) -> int:
        return sum(1 for item in self.outcomes if not item.installed)

    @property
    def ok(self) -> bool:
        """A run is ok when it installed at least one entry and refused none."""
        return bool(self.outcomes) and self.refused == 0 and self.installed > 0

    def describe(self) -> str:
        if self.detail:
            return self.detail
        lines = [f"Installed {self.installed} of {len(self.outcomes)} pack entries."]
        for outcome in self.outcomes:
            lines.append(outcome.describe())
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "refused": self.refused,
            "ok": self.ok,
            "detail": self.detail,
            "outcomes": [item.to_primitive() for item in self.outcomes],
        }


def install_entry(
    entry: PackEntry,
    *,
    force: bool = False,
    backup: bool = True,
) -> InstallOutcome:
    """Verify one entry and, only if it verifies, install it into the trainer's folder.

    Verification comes **first**: a digest mismatch, a wrong kind, or an
    incompatible architecture refuses the install and writes nothing.
    """
    verification = verify_entry(entry)
    if not verification.ok:
        return InstallOutcome(
            entry=entry,
            verification=verification,
            detail=verification.detail,
            continue_command=entry.continue_command,
        )

    report = install_weights(
        entry.path,
        kind=entry.kind,
        task=entry.task,
        force=force,
        backup=backup,
    )
    command = entry.continue_command
    if not command and report.plan is not None:
        command = report.plan.command
    return InstallOutcome(
        entry=entry,
        verification=verification,
        installed=report.installed,
        destination=report.destination,
        backup=report.backup,
        companions=report.companions,
        detail="" if report.installed else report.detail,
        continue_command=command,
    )


def install_pack(
    pack: WeightsPack,
    *,
    keys: tuple[str, ...] = (),
    force: bool = False,
    backup: bool = True,
) -> PackInstallReport:
    """Install the named entries (or every entry) after verifying each.

    An unknown key is **refused** with the reason rather than skipped silently, so
    a caller learns a name it mistyped instead of believing the pack omitted it.
    """
    if not pack.available:
        return PackInstallReport(detail=pack.detail or "No weights pack was found.")

    if keys:
        selected: list[PackEntry] = []
        for key in keys:
            entry = pack.get(key)
            if entry is None:
                return PackInstallReport(
                    detail=(
                        f"No pack entry named {key!r}. Available: "
                        f"{', '.join(pack.keys()) or '(none)'}."
                    )
                )
            selected.append(entry)
    else:
        selected = list(pack.entries)

    outcomes = tuple(
        install_entry(entry, force=force, backup=backup) for entry in selected
    )
    return PackInstallReport(outcomes=outcomes)


# ------------------------------------------------------------------- status


@dataclass(frozen=True)
class PackStatus:
    """What the pack offers and what is already installed for each entry."""

    pack: WeightsPack
    verifications: tuple[Verification, ...] = ()
    installed: tuple[str, ...] = ()
    detail: str = ""

    @property
    def ready(self) -> int:
        """How many entries verify cleanly and can be installed."""
        return sum(1 for item in self.verifications if item.ok)

    @property
    def present(self) -> int:
        """How many entries are already installed for the trainers."""
        return len(self.installed)

    def describe(self) -> str:
        if not self.pack.available:
            return self.detail or self.pack.detail or "No weights pack was found."
        lines = [
            f"Weights pack v{self.pack.version}"
            + (f" (project {self.pack.project_version})" if self.pack.project_version else ""),
            f"  {self.ready}/{len(self.verifications)} entries verify cleanly, "
            f"{self.present} installed",
        ]
        for verification in self.verifications:
            lines.append("  " + verification.describe().replace("\n", "\n  "))
        if self.installed:
            lines.append("  installed for the trainers: " + ", ".join(self.installed))
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "pack": self.pack.to_primitive(),
            "ready": self.ready,
            "present": self.present,
            "installed": list(self.installed),
            "verifications": [item.to_primitive() for item in self.verifications],
            "detail": self.detail,
        }


def installed_destination(entry: PackEntry) -> Path | None:
    """The file the entry installs to for the trainers, or ``None`` when it cannot.

    A policy with no task, or an unknown kind, has no destination -- reported as
    ``None`` rather than a made-up path.
    """
    from core.checkpoint_import import destination_for

    try:
        return destination_for(entry.kind, task=entry.task)
    except Exception:
        return None


def _is_installed(entry: PackEntry) -> bool:
    """Whether the entry's *destination* already holds these weights.

    Checked by digest when the pack records one (so a stale, different file is not
    mistaken for the pack's own), and by presence otherwise.
    """
    target = installed_destination(entry)
    if target is None or not target.exists():
        return False
    if not entry.digest:
        return True
    try:
        return file_digest(target).lower() == entry.digest.lower()
    except OSError:
        return False


def pack_status(root: Path | str | None = None) -> PackStatus:
    """Verify the pack and report which entries are already installed for the trainers."""
    pack = load_pack(root)
    if not pack.available:
        return PackStatus(pack=pack, detail=pack.detail)

    verifications = verify_pack(pack)
    names = tuple(entry.key for entry in pack.entries if _is_installed(entry))
    return PackStatus(pack=pack, verifications=verifications, installed=names)


# ------------------------------------------------------------- generation


def _relative_to_base(path: Path, base: Path) -> str:
    """``path`` as a string relative to ``base`` when it is inside it, else absolute."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _language_entry(base: Path, checkpoint: Path, *, key: str, title: str) -> PackEntry:
    """Build a language-model entry from a checkpoint on disk (digest read from it)."""
    info = read_weight_info(checkpoint)
    architecture: dict[str, Any] = {}
    if info.checkpoint is not None:
        architecture = {
            "vocab_size": info.checkpoint.vocab_size,
            "n_layer": info.checkpoint.n_layer,
            "n_head": info.checkpoint.n_head,
            "n_embd": info.checkpoint.n_embd,
            "block_size": info.checkpoint.block_size,
            "dropout": info.checkpoint.dropout,
            "tie_weights": info.checkpoint.tie_weights,
        }
    companions: list[str] = []
    tokenizer = checkpoint.with_suffix(".tokenizer.json")
    if tokenizer.exists():
        companions.append(_relative_to_base(tokenizer, base))
    state = checkpoint.with_suffix(".state.npz")
    if state.exists():
        companions.append(_relative_to_base(state, base))
    return PackEntry(
        key=key,
        title=title,
        kind=KIND_LANGUAGE,
        path=checkpoint,
        companions=tuple(base / item for item in companions),
        digest=file_digest(checkpoint),
        architecture=architecture,
        provenance={
            "source": "the project's own trainer",
            "checkpoint": _relative_to_base(checkpoint, base),
        },
        continue_command="python -m cli llm --resume",
        notes=(
            "A language-model checkpoint; continue it with `python -m cli llm "
            "--resume`. Install it so the trainer resumes from these weights."
        ),
    )


def _policy_entry(base: Path, policy: Path) -> PackEntry:
    """Build a policy entry from a ``policy_<task>.npz`` file on disk."""
    info = read_weight_info(policy)
    task = info.task or policy.stem.removeprefix("policy_")
    return PackEntry(
        key=f"policy-{task}",
        title=f"Policy for {task}",
        kind=KIND_POLICY,
        path=policy,
        digest=file_digest(policy),
        task=task,
        provenance={"source": "the project's own trainer", "task": task},
        continue_command="python -m cli petroleum --resume",
        notes=(
            "A Q-learning policy for one task; continue it with `python -m cli "
            "petroleum --resume`."
        ),
    )


def generate_manifest(
    *,
    root: Path | str | None = None,
    checkpoint: Path | str | None = None,
    policy_dir: Path | str | None = None,
    include_policies: bool = True,
    write: bool = True,
) -> WeightsPack:
    """(Re)build the pack manifest from the weights already on disk.

    This is the honest generator: it reads each file, computes its real digest,
    reads a checkpoint's own architecture, and writes ``weights/manifest.json``.
    It never invents an entry -- a weight that is not on disk simply does not
    appear -- and it resolves entry paths against the repository/bundle root so
    the pack *indexes* the project's weights without a second copy.

    The default sources are the **committed** weights (``ml/llm/checkpoints`` and
    ``ml/artifacts/petroleum`` inside the checkout), because those are what a
    fresh clone ships; pass ``checkpoint``/``policy_dir`` to pack weights the user
    trained into the writable ``data/`` folder instead.
    """
    from core.paths import bundle_root, weights_pack_dir
    from core.version import __version__

    base = bundle_root()
    pack_root = Path(root).expanduser() if root is not None else weights_pack_dir()
    source_checkpoint = (
        Path(checkpoint).expanduser()
        if checkpoint is not None
        else base / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
    )
    source_policy_dir = (
        Path(policy_dir).expanduser()
        if policy_dir is not None
        else base / "ml" / "artifacts" / "petroleum"
    )

    entries: list[PackEntry] = []
    if source_checkpoint.exists():
        entries.append(
            _language_entry(base, source_checkpoint, key="fayd-llm", title="Fayd Al-Ruh language model")
        )
    if include_policies and source_policy_dir.is_dir():
        for policy in sorted(source_policy_dir.glob("policy_*.npz")):
            entries.append(_policy_entry(base, policy))

    pack = WeightsPack(
        root=pack_root,
        version=MANIFEST_VERSION,
        project_version=__version__,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        entries=tuple(entries),
        detail="" if entries else "No weights were found to pack.",
    )
    if write and entries:
        _write_manifest(pack)
    return pack


def _write_manifest(pack: WeightsPack) -> Path:
    """Write the manifest atomically, so a crash never leaves a half-written pack."""
    pack.root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": pack.version,
        "project_version": pack.project_version,
        "created_at": pack.created_at,
        "digest_algorithm": DIGEST_ALGORITHM,
        "entries": [
            {
                "key": entry.key,
                "title": entry.title,
                "kind": entry.kind,
                "path": _relative_to_base(entry.path, pack.root.parent),
                "companions": [
                    _relative_to_base(item, pack.root.parent) for item in entry.companions
                ],
                "digest": entry.digest,
                "task": entry.task,
                "architecture": dict(entry.architecture),
                "provenance": dict(entry.provenance),
                "continue_command": entry.continue_command,
                "notes": entry.notes,
            }
            for entry in pack.entries
        ],
    }
    target = pack.root / MANIFEST_NAME
    handle, temp_name = tempfile.mkstemp(dir=str(pack.root), prefix=".manifest-", suffix=".tmp")
    os.close(handle)
    try:
        Path(temp_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "DIGEST_ALGORITHM",
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "STATUS_DIGEST_MISMATCH",
    "STATUS_INCOMPATIBLE",
    "STATUS_MISSING",
    "STATUS_OK",
    "STATUS_UNREADABLE",
    "STATUS_WRONG_KIND",
    "InstallOutcome",
    "PackEntry",
    "PackInstallReport",
    "PackStatus",
    "Verification",
    "WeightsPack",
    "WeightsPackError",
    "file_digest",
    "generate_manifest",
    "install_entry",
    "install_pack",
    "installed_destination",
    "load_pack",
    "pack_status",
    "verify_entry",
    "verify_pack",
]
