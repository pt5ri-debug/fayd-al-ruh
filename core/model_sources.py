"""Where a model may come from, and what a source honestly claims about it.

A model manager that can only *describe* files it already has leaves the user to
fetch weights by hand. This module is the missing half: a small, explicit set of
**source adapters**, each of which can search for a model and describe what it
offers -- without downloading anything and without executing anything.

Three rules, matching the rest of the project:

* **A source is an adapter, not a branch.** Every source implements
  :class:`ModelSource`, so the manager asks the same questions of a Hugging Face
  repository, a direct URL, or a local folder, and a new source can be added
  without touching the manager.
* **A claim is labelled.** Metadata a publisher states is ``claimed``; metadata
  this project read from the bytes is ``verified``; metadata it inferred is
  ``estimated``. The three are never merged into one confident fact (see
  :data:`CLAIM_*`), because "the publisher says 7B" is not the same statement as
  "this file's header says 7B".
* **Nothing here trusts a repository because it is popular.** A digest published
  *beside* the file is ``CLAIM_SOURCE`` -- useful, but not proof of authenticity
  on its own. Only a digest the user pinned independently is ``CLAIM_TRUSTED``.

Nothing here opens a socket. The network is the *downloader's*
(:mod:`core.model_download`) job, behind the project's permission and human
approval gates; a source only describes what a download would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# ------------------------------------------------------------------ provenance

#: How much a piece of metadata is worth. Ordered weakest to strongest so a
#: caller can compare them, and never collapsed into one "fact".
CLAIM_UNKNOWN = "unknown"
CLAIM_ESTIMATED = "estimated"
CLAIM_CLAIMED = "claimed"
CLAIM_VERIFIED = "verified"
CLAIM_TRUSTED = "trusted"

CLAIM_ORDER: tuple[str, ...] = (
    CLAIM_UNKNOWN,
    CLAIM_ESTIMATED,
    CLAIM_CLAIMED,
    CLAIM_VERIFIED,
    CLAIM_TRUSTED,
)

#: Readable labels for a user-facing surface.
CLAIM_LABELS: dict[str, str] = {
    CLAIM_UNKNOWN: "unknown",
    CLAIM_ESTIMATED: "estimated locally",
    CLAIM_CLAIMED: "stated by the source",
    CLAIM_VERIFIED: "read from the file",
    CLAIM_TRUSTED: "matches a digest you pinned",
}

#: How strong a file digest is. A digest published beside the file is *not* proof
#: of authenticity: it proves the bytes match what the source said, and nothing
#: more, unless the user pinned the expected value themselves.
DIGEST_NONE = "none"
DIGEST_SOURCE = "source-provided"
DIGEST_TRUSTED = "pinned-by-user"

DIGEST_STRENGTH: dict[str, str] = {
    DIGEST_NONE: "no expected digest was available",
    DIGEST_SOURCE: "matches the digest the source published beside the file",
    DIGEST_TRUSTED: "matches the digest you pinned independently",
}


def claim_rank(claim: str) -> int:
    """The position of a claim in :data:`CLAIM_ORDER`, for comparison."""
    try:
        return CLAIM_ORDER.index(claim)
    except ValueError:
        return 0


def describe_claim(claim: str) -> str:
    """A readable rendering of a claim level, never blank."""
    return CLAIM_LABELS.get(claim, CLAIM_LABELS[CLAIM_UNKNOWN])


# ---------------------------------------------------------------------- errors


class SourceError(RuntimeError):
    """A source could not answer -- the request itself is the problem."""


class SourceUnavailable(SourceError):
    """The source cannot be reached or is not usable here (no client, no network)."""


class UnknownSource(SourceError):
    """A source key this build does not know, so nothing is invented for it."""


# ----------------------------------------------------------------------- offer


@dataclass(frozen=True)
class ModelFile:
    """One file a download would fetch.

    ``expected_digest`` is what the *source* published; :attr:`digest_strength`
    says how much that is worth (see :data:`DIGEST_STRENGTH`). A file with no
    digest is honestly ``DIGEST_NONE`` rather than being given a made-up one.
    """

    filename: str
    size_bytes: int | None = None
    expected_digest: str = ""
    digest_algorithm: str = ""
    digest_strength: str = DIGEST_NONE
    #: The direct URL for this file, when the source can give one. Empty for a
    #: local source, where the file is already on disk.
    url: str = ""

    def describe(self) -> str:
        size = "unknown size" if self.size_bytes is None else _human(self.size_bytes)
        bits = [self.filename, size]
        if self.expected_digest:
            bits.append(f"{self.digest_algorithm or 'digest'}: {self.digest_strength}")
        else:
            bits.append("no published digest")
        return " · ".join(bits)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "digest_algorithm": self.digest_algorithm,
            "digest_strength": self.digest_strength,
            "expected_digest": self.expected_digest,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "url": self.url,
        }


@dataclass(frozen=True)
class ModelOffer:
    """A model a source offers, described **before** anything is downloaded.

    Every field that is not known is explicitly empty or ``None`` -- a source that
    cannot say a model's license says ``unknown``, never a plausible guess. The
    ``*_claim`` fields say how much each group of metadata is worth.
    """

    source_key: str
    #: Stable within a source -- the repo id, or the URL, or the folder.
    source_id: str
    display_name: str = ""
    publisher: str = ""
    #: The exact revision this offer pins, when the source supports revisions.
    revision: str = ""
    architecture: str = ""
    model_format: str = ""
    quantization: str = ""
    files: tuple[ModelFile, ...] = field(default=())
    #: A license *identifier* (``apache-2.0``), or ``""`` when the source does not
    #: state one. Never assumed open source.
    license: str = ""
    license_url: str = ""
    #: The page a person can read to check the claim.
    source_url: str = ""
    card_url: str = ""
    runtimes: tuple[str, ...] = field(default=())
    context_length: int | None = None
    parameter_count: int | None = None
    languages: tuple[str, ...] = field(default=())
    tasks: tuple[str, ...] = field(default=())
    #: When this description was read from the source.
    retrieved_at: str = ""
    claim: str = CLAIM_UNKNOWN
    #: What the source itself says about access (``gated``, ``public``, ...).
    access: str = ""
    notes: tuple[str, ...] = field(default=())

    @property
    def total_bytes(self) -> int | None:
        """The total download size, or ``None`` when any file's size is unknown."""
        if not self.files:
            return None
        total = 0
        for item in self.files:
            if item.size_bytes is None:
                return None
            total += item.size_bytes
        return total

    @property
    def is_gated(self) -> bool:
        """True when the source says the model needs an accepted licence or login."""
        return self.access.strip().lower() in {"gated", "private", "restricted"}

    def describe(self) -> str:
        """A readable block, honest about every unknown and every claim."""
        lines = [f"{self.display_name or self.source_id}  [{self.source_key}]"]
        lines.append(f"  id:        {self.source_id}")
        if self.publisher:
            lines.append(f"  publisher: {self.publisher}")
        if self.revision:
            lines.append(f"  revision:  {self.revision}")
        lines.append(f"  format:    {self.model_format or 'unknown'}"
                     + (f" / {self.quantization}" if self.quantization else ""))
        lines.append(f"  license:   {self.license or 'unknown'}"
                     + (f" ({self.license_url})" if self.license_url else ""))
        if self.parameter_count:
            lines.append(f"  parameters: {self.parameter_count / 1e9:.2f}B (claimed)")
        if self.context_length:
            lines.append(f"  context:   {self.context_length} tokens")
        if self.runtimes:
            lines.append(f"  runtimes:  {', '.join(self.runtimes)}")
        if self.languages:
            lines.append(f"  languages: {', '.join(self.languages)}")
        if self.tasks:
            lines.append(f"  tasks:     {', '.join(self.tasks)} (claimed, not a guarantee)")
        total = self.total_bytes
        lines.append(
            "  download:  "
            + (_human(total) if total is not None else "size unknown until the source answers")
        )
        for item in self.files:
            lines.append(f"    - {item.describe()}")
        lines.append(f"  metadata:  {describe_claim(self.claim)}")
        if self.is_gated:
            lines.append("  access:    gated — you must accept the licence before downloading")
        if self.source_url:
            lines.append(f"  source:    {self.source_url}")
        lines.extend(f"  note:      {note}" for note in self.notes)
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "access": self.access,
            "architecture": self.architecture,
            "card_url": self.card_url,
            "claim": self.claim,
            "context_length": self.context_length,
            "display_name": self.display_name,
            "files": [item.to_primitive() for item in self.files],
            "languages": list(self.languages),
            "license": self.license,
            "license_url": self.license_url,
            "model_format": self.model_format,
            "notes": list(self.notes),
            "parameter_count": self.parameter_count,
            "publisher": self.publisher,
            "quantization": self.quantization,
            "retrieved_at": self.retrieved_at,
            "revision": self.revision,
            "runtimes": list(self.runtimes),
            "source_id": self.source_id,
            "source_key": self.source_key,
            "source_url": self.source_url,
            "tasks": list(self.tasks),
            "total_bytes": self.total_bytes,
        }


# --------------------------------------------------------------------- source


class ModelSource(Protocol):
    """What the manager needs from a place models come from.

    Deliberately narrow and synchronous: a caller that needs it off the UI thread
    runs it on a worker (see :meth:`core.model_manager.ModelManager.download`),
    so a source never has to know about threading.
    """

    key: str
    title: str

    def available(self) -> tuple[bool, str]:
        """Whether this source can be used here, and why not when it cannot."""
        ...

    def search(self, query: str, *, limit: int = 10) -> tuple[ModelOffer, ...]:
        """Models matching ``query``, best first. Never downloads."""
        ...

    def resolve(self, source_id: str, *, revision: str = "") -> ModelOffer:
        """One model's full description, or :class:`SourceError`."""
        ...

    def describe(self) -> str:
        """A readable one-block description of the source itself."""
        ...


# ------------------------------------------------------------------- adapters


class LocalFolderSource:
    """Models that are already on this machine, in a folder the user names.

    The honest baseline: it needs no network, no account, and no dependency. A
    model is *offered* here only when its file is really present and really a
    supported format, because a local source that listed absent files would be
    inventing them.
    """

    key = "local"
    title = "Local folder"

    def __init__(self, roots: object = None) -> None:
        self._roots = _as_paths(roots)

    def available(self) -> tuple[bool, str]:
        if not self._roots:
            return False, "No local folder was given to search."
        present = [root for root in self._roots if root.is_dir()]
        if not present:
            return False, f"None of these folders exists: {', '.join(str(r) for r in self._roots)}"
        return True, ""

    def search(self, query: str, *, limit: int = 10) -> tuple[ModelOffer, ...]:
        from core.model_files import discover_models

        report = discover_models(self._roots, limit=max(limit, 1) * 20)
        needle = str(query or "").strip().casefold()
        offers: list[ModelOffer] = []
        for probe in report.probes:
            if needle and needle not in probe.path.name.casefold():
                continue
            offers.append(_offer_from_probe(probe))
            if len(offers) >= limit:
                break
        return tuple(offers)

    def resolve(self, source_id: str, *, revision: str = "") -> ModelOffer:
        from core.model_files import probe_file

        path = Path(source_id).expanduser()
        if not path.is_file():
            raise SourceError(f"No file is at {path}.")
        return _offer_from_probe(probe_file(path))

    def describe(self) -> str:
        ok, why = self.available()
        roots = ", ".join(str(root) for root in self._roots) or "(none given)"
        line = f"{self.title} — {roots}"
        return line if ok else f"{line}\n  {why}"


class DirectUrlSource:
    """A model at a direct HTTPS URL the user names themselves.

    This is the *user-provided* source the audit asks for, and it is deliberately
    strict: HTTPS only, no embedded credentials, and no assumed digest. It
    describes a single file; the downloader fetches it behind the approval gate.

    Nothing here trusts the URL's hostname. A file downloaded from an arbitrary
    URL is registered as ``CLAIM_CLAIMED`` at best and its digest as
    ``DIGEST_SOURCE`` (or ``DIGEST_NONE``), which is exactly what it is worth.
    """

    key = "url"
    title = "Direct download URL"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def search(self, query: str, *, limit: int = 10) -> tuple[ModelOffer, ...]:
        text = str(query or "").strip()
        if not text:
            return ()
        try:
            return (self.resolve(text),)
        except SourceError:
            return ()

    def resolve(self, source_id: str, *, revision: str = "") -> ModelOffer:
        url = str(source_id or "").strip()
        problem = validate_download_url(url)
        if problem:
            raise SourceError(problem)
        filename = url.rsplit("/", 1)[-1].split("?", 1)[0] or "model.gguf"
        return ModelOffer(
            source_key=self.key,
            source_id=url,
            display_name=filename,
            model_format="gguf" if filename.lower().endswith(".gguf") else "",
            files=(ModelFile(filename=filename, url=url),),
            source_url=url,
            access="public",
            claim=CLAIM_CLAIMED,
            notes=(
                "This is a URL you supplied. Nothing here verifies the publisher, "
                "and no digest was published with it.",
            ),
        )

    def describe(self) -> str:
        return (
            f"{self.title}\n"
            "  Give an https:// URL to a model file. The host is not trusted on "
            "your behalf, and the file is hashed locally after download."
        )


class HuggingFaceSource:
    """A Hugging Face repository, through the project's existing hub client.

    Optional by design: ``huggingface_hub`` is imported lazily and its absence is
    reported as an actionable next step, never as a crash. The client is
    injectable, so the whole adapter is exercised in tests with no network and no
    package.

    Metadata is reported as ``CLAIM_CLAIMED`` -- the publisher states it -- and a
    digest the hub publishes is ``DIGEST_SOURCE``. The project does not treat
    either as proof of authenticity.
    """

    key = "huggingface"
    title = "Hugging Face"

    def __init__(self, *, client: Any = None, root: Path | str | None = None) -> None:
        self._client = client
        self._root = root

    def _hub(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import huggingface_hub  # type: ignore[import-not-found]
        except ImportError as error:
            raise SourceUnavailable(
                "Searching Hugging Face needs `huggingface_hub`. Install it with: "
                "python -m pip install -r requirements-shared.txt"
            ) from error
        return huggingface_hub

    def available(self) -> tuple[bool, str]:
        try:
            self._hub()
        except SourceUnavailable as error:
            return False, str(error)
        return True, ""

    def search(self, query: str, *, limit: int = 10) -> tuple[ModelOffer, ...]:
        text = str(query or "").strip()
        if not text:
            return ()
        hub = self._hub()
        try:
            results = hub.list_models(search=text, limit=limit)
        except Exception as error:
            raise SourceUnavailable(f"Hugging Face could not be searched: {error}") from error
        offers: list[ModelOffer] = []
        for item in results:
            offers.append(_offer_from_hub_model(item))
            if len(offers) >= limit:
                break
        return tuple(offers)

    def resolve(self, source_id: str, *, revision: str = "") -> ModelOffer:
        repo = str(source_id or "").strip()
        if not repo or "/" not in repo:
            raise SourceError(
                "A Hugging Face model is named `owner/repo` (for example "
                "`unsloth/Qwen3.5-4B-GGUF`)."
            )
        hub = self._hub()
        try:
            info = hub.model_info(repo, revision=revision or None)
        except Exception as error:
            raise SourceError(f"Could not read {repo!r}: {error}") from error
        return _offer_from_hub_info(info, repo=repo, revision=revision)

    def describe(self) -> str:
        ok, why = self.available()
        line = f"{self.title} — repos are named `owner/repo`"
        return line if ok else f"{line}\n  {why}"


# ------------------------------------------------------------------ registry


def default_sources(*, local_roots: object = None, hub_client: Any = None) -> tuple[ModelSource, ...]:
    """The sources this build supports, in a fixed order.

    Only sources that can be implemented and tested reliably: a local folder (no
    network), a user-supplied HTTPS URL (no account), and Hugging Face through the
    project's existing hub client. A new source is added here and nowhere else.
    """
    return (
        LocalFolderSource(local_roots),
        DirectUrlSource(),
        HuggingFaceSource(client=hub_client),
    )


def source_for(key: str, **kwargs: Any) -> ModelSource:
    """One source by key, or :class:`UnknownSource` -- never a made-up default."""
    wanted = str(key or "").strip().lower()
    for source in default_sources(**kwargs):
        if source.key == wanted:
            return source
    known = ", ".join(source.key for source in default_sources(**kwargs))
    raise UnknownSource(f"No model source is named {key!r}. Known sources: {known}.")


def source_keys(*, local_roots: object = None, hub_client: Any = None) -> tuple[str, ...]:
    """Every source key this build knows, in the fixed order."""
    return tuple(
        source.key for source in default_sources(local_roots=local_roots, hub_client=hub_client)
    )


def describe_sources(*, local_roots: object = None, hub_client: Any = None) -> str:
    """Every source, its availability here, and what it is for."""
    lines = ["Model sources"]
    for source in default_sources(local_roots=local_roots, hub_client=hub_client):
        ok, why = source.available()
        mark = "available" if ok else "unavailable"
        lines.append(f"  {source.key:<12} {mark:<12} {source.title}")
        if why:
            lines.append(f"    {why}")
    return "\n".join(lines)


# -------------------------------------------------------------------- helpers


def validate_download_url(url: str) -> str:
    """Why ``url`` may not be downloaded from, or ``""`` when it may.

    Deliberately strict, and a **pure function** so the rule is testable without a
    network. It refuses anything but HTTPS (so a redirect to plain HTTP or a
    ``file://`` path cannot be used to read the local disk), refuses embedded
    credentials (which would put a token in a log), and refuses a URL that is not
    absolute.
    """
    text = str(url or "").strip()
    if not text:
        return "A download needs a URL."
    lowered = text.lower()
    if lowered.startswith("file:") or lowered.startswith("ftp:"):
        return "Only https:// downloads are supported; a local file is registered directly."
    if not lowered.startswith("https://"):
        return "Only https:// downloads are supported, so the transfer is encrypted."
    authority = text[len("https://"):].split("/", 1)[0]
    if "@" in authority:
        return "The URL carries credentials. Put a token in the source's own setting instead."
    host = authority.split(":", 1)[0]
    if not host or "." not in host:
        return f"{host or 'That host'!r} is not a hostname this project will download from."
    return ""


def _as_paths(roots: object) -> tuple[Path, ...]:
    if roots is None:
        return ()
    if isinstance(roots, (str, Path)):
        return (Path(roots).expanduser(),)
    if isinstance(roots, (list, tuple, set, frozenset)):
        return tuple(Path(item).expanduser() for item in roots)
    return ()


def _human(size: int | None) -> str:
    if size is None:
        return "unknown"
    amount = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def _offer_from_probe(probe: Any) -> ModelOffer:
    """A local file as an offer, with the facts **verified** from its bytes."""
    path = Path(probe.path)
    return ModelOffer(
        source_key="local",
        source_id=str(path),
        display_name=path.name,
        architecture=str(getattr(probe, "architecture", "") or ""),
        model_format=str(getattr(probe, "format", "") or ""),
        quantization=str(getattr(probe, "quantization", "") or ""),
        files=(
            ModelFile(
                filename=path.name,
                size_bytes=int(getattr(probe, "size_bytes", 0) or 0) or None,
                expected_digest=str(getattr(probe, "digest", "") or ""),
                digest_algorithm=str(getattr(probe, "digest_algorithm", "") or ""),
                digest_strength=DIGEST_NONE,
            ),
        ),
        context_length=getattr(probe, "context_length", None),
        parameter_count=getattr(probe, "parameter_count", None),
        source_url="",
        access="local",
        claim=CLAIM_VERIFIED,
        notes=("This file is already on this machine; nothing is downloaded.",),
    )


def _offer_from_hub_model(item: Any) -> ModelOffer:
    """A hub search hit as an offer. Everything here is the publisher's claim."""
    repo = str(getattr(item, "id", "") or getattr(item, "modelId", "") or "")
    return ModelOffer(
        source_key="huggingface",
        source_id=repo,
        display_name=repo,
        publisher=repo.split("/", 1)[0] if "/" in repo else "",
        source_url=f"https://huggingface.co/{repo}" if repo else "",
        access=str(getattr(item, "gated", "") or "public"),
        claim=CLAIM_CLAIMED,
        notes=("Search results carry the publisher's own description only.",),
    )


def _offer_from_hub_info(info: Any, *, repo: str, revision: str) -> ModelOffer:
    """A hub ``model_info`` as an offer, reading only fields the hub really has."""
    siblings = getattr(info, "siblings", None) or ()
    files: list[ModelFile] = []
    for sibling in siblings:
        name = str(getattr(sibling, "rfilename", "") or "")
        if not name:
            continue
        size = getattr(sibling, "size", None)
        files.append(
            ModelFile(
                filename=name,
                size_bytes=int(size) if isinstance(size, int) else None,
                digest_strength=DIGEST_NONE,
                url=f"https://huggingface.co/{repo}/resolve/{revision or 'main'}/{name}",
            )
        )
    card = getattr(info, "cardData", None)
    license_id = ""
    languages: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    if isinstance(card, dict):
        raw_license = card.get("license")
        if isinstance(raw_license, str):
            license_id = raw_license
        elif isinstance(raw_license, (list, tuple)) and raw_license:
            license_id = str(raw_license[0])
        raw_langs = card.get("language")
        if isinstance(raw_langs, str):
            languages = (raw_langs,)
        elif isinstance(raw_langs, (list, tuple)):
            languages = tuple(str(item) for item in raw_langs)
        raw_tasks = card.get("pipeline_tag") or card.get("task")
        if isinstance(raw_tasks, str):
            tasks = (raw_tasks,)
    resolved = str(getattr(info, "sha", "") or revision or "")
    return ModelOffer(
        source_key="huggingface",
        source_id=repo,
        display_name=repo,
        publisher=repo.split("/", 1)[0] if "/" in repo else "",
        revision=resolved,
        files=tuple(files),
        license=license_id,
        license_url=f"https://huggingface.co/{repo}/blob/{resolved or 'main'}/README.md"
        if license_id
        else "",
        source_url=f"https://huggingface.co/{repo}",
        card_url=f"https://huggingface.co/{repo}",
        languages=languages,
        tasks=tasks,
        access=str(getattr(info, "gated", "") or "public"),
        claim=CLAIM_CLAIMED,
        notes=(
            "Everything here is what the repository states about itself. The "
            "project does not verify a publisher's claims.",
        ),
    )


__all__ = [
    "CLAIM_CLAIMED",
    "CLAIM_ESTIMATED",
    "CLAIM_LABELS",
    "CLAIM_ORDER",
    "CLAIM_TRUSTED",
    "CLAIM_UNKNOWN",
    "CLAIM_VERIFIED",
    "DIGEST_NONE",
    "DIGEST_SOURCE",
    "DIGEST_STRENGTH",
    "DIGEST_TRUSTED",
    "DirectUrlSource",
    "HuggingFaceSource",
    "LocalFolderSource",
    "ModelFile",
    "ModelOffer",
    "ModelSource",
    "SourceError",
    "SourceUnavailable",
    "UnknownSource",
    "claim_rank",
    "default_sources",
    "describe_claim",
    "describe_sources",
    "source_for",
    "source_keys",
    "validate_download_url",
]
