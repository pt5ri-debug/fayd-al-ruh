"""The model registry: one durable record per registered model.

What already existed, and why this is not a duplicate of it:

* :mod:`core.language_models` is a **read-only view** -- it answers "which models
  does this build know about, and is each present?" by reading the per-engine
  catalogs on every call. It stores nothing and can describe nothing the catalogs
  do not already carry.
* :mod:`core.custom_models` is a **shortlist for the per-mouth picker**: a key, a
  title, and a path for a GGUF a user pointed a window at. It has no metadata, no
  runtime, no integrity result, no timestamps, and no tags.
* :mod:`core.mouth_profile` stores *per-mouth* choices (name, logo, voice, model).

Neither is a registry of models with metadata, so this module adds one -- and it
coexists with both rather than replacing either. ``custom_models`` keeps serving
the mouth windows unchanged, and :func:`import_custom_models` can pull its
entries in as ordinary records without ever writing back to it.

**Ownership.** This registry owns exactly one fact: *a model someone registered,
and everything knowable about it.* It does not own which model a mouth uses (that
is ``mouth_profile``), which models the build ships (that is
``language_models``), or what a model's settings are (that is ``model_settings``).

**Storage.** One JSON document per model, under a :class:`JsonStore` namespace, so:

* a write touches one model's file, never the whole registry (safe updates);
* one corrupt record costs one model, not the registry (recoverable);
* :class:`JsonStore` already writes atomically (temporary file + ``replace``), so
  a crash cannot leave a half-written record;
* serialization is deterministic -- ``JsonStore`` sorts keys.

**Concurrency.** A read-modify-write (register, update, remove) is guarded by a
process-wide re-entrant lock, so two threads cannot interleave a read with another
thread's write. This is what makes the registry safe to drive from a GUI worker
thread.

**Safety.** A removal drops the record and **never** touches the file. A
destructive call must say so explicitly. A path is validated before it is stored.
No secret-looking value is ever persisted, and every error message is redacted.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.model_files import (
    FORMAT_FAYD_CHECKPOINT,
    FORMAT_GGUF,
    FORMAT_UNKNOWN,
    INTEGRITY_OK,
    FileProbe,
    ModelFileError,
    validate_model_path,
)
from core.model_runtime import (
    RUNTIME_UNKNOWN,
    runtime_for_engine,
    runtime_for_format,
)
from core.paths import model_registry_dir

#: The JsonStore namespace every record lives under.
NAMESPACE = "model_registry"

#: The model formats a record may declare. ``unknown`` is allowed on purpose: a
#: record may exist for a file this build cannot run yet, and saying so is more
#: useful than refusing to register it.
KNOWN_FORMATS: tuple[str, ...] = (FORMAT_GGUF, FORMAT_FAYD_CHECKPOINT, FORMAT_UNKNOWN)

#: Where a model came from. A closed set, so a report can group by it.
SOURCE_CATALOG = "catalog"
SOURCE_LOCAL = "local"
SOURCE_TRAINED = "trained"
SOURCE_REMOTE = "remote"
#: A model this program fetched itself, through the Model Manager's downloader.
#: Distinct from ``remote`` (a model with no local file) and from ``local`` (a file
#: the user already had): a download carries provenance -- a source, a revision,
#: and an integrity verdict -- which neither of those has.
SOURCE_DOWNLOAD = "downloaded"
SOURCE_UNKNOWN = "unknown"
KNOWN_SOURCES: tuple[str, ...] = (
    SOURCE_CATALOG, SOURCE_LOCAL, SOURCE_TRAINED, SOURCE_REMOTE, SOURCE_DOWNLOAD,
    SOURCE_UNKNOWN,
)

#: A tag is a short, lower-case token; anything else is refused rather than
#: stored, so a report can group by tag reliably.
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")

#: Field names that would suggest a credential. A record carrying one is refused,
#: because model metadata is inspectable, exportable, and shown in a window.
_SECRET_HINTS = (
    "api_key", "apikey", "api-key", "token", "secret", "password", "passwd",
    "credential", "authorization", "auth_key", "private_key", "access_key",
)

#: A value that looks like a bearer token or an OpenAI-style key. Refused so a
#: pasted value cannot land in a record even under an innocent field name.
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    r"Bearer\s+[A-Za-z0-9._\-]{16,})"
)

#: The reserved key prefix, so a record key can never collide with a state entry.
#: (``core.code_studio`` learned this the hard way with its serial counter.)
_STATE_PREFIX = "__"

#: A bounded length for every free-text field, so a hand-edited file cannot make a
#: listing unbounded.
MAX_TEXT = 512
#: A bounded number of tags per model.
MAX_TAGS = 32


class RegistryError(RuntimeError):
    """A registry operation failed, with the reason stated."""


class RegistryValidationError(RegistryError):
    """A record would be invalid, so nothing was written."""


class ModelNotFoundError(RegistryError):
    """No registered model has the requested id."""


class ConfirmationRequired(RegistryError):
    """A destructive operation was attempted without an explicit confirmation."""


def _now() -> str:
    """A UTC timestamp, ISO-8601, seconds precision -- sortable and readable."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _clean_text(value: object, limit: int = MAX_TEXT) -> str:
    """A bounded, whitespace-trimmed string. Anything else becomes ``""``."""
    if value is None:
        return ""
    text = str(value).strip()
    return text[:limit]


def _looks_like_a_secret(text: str) -> bool:
    """True when ``text`` carries something that must never be persisted."""
    if not text:
        return False
    if _SECRET_VALUE_RE.search(text):
        return True
    lowered = text.casefold()
    return any(hint in lowered for hint in _SECRET_HINTS)


def _reject_secrets(mapping: dict[str, Any], *, where: str) -> None:
    """Refuse a mapping that names or carries a credential."""
    for key, value in mapping.items():
        if any(hint in str(key).casefold() for hint in _SECRET_HINTS):
            raise RegistryValidationError(
                f"{where} must not carry a credential: the field {key!r} looks like "
                "one. Model metadata is shown and exported, so keys are never stored here."
            )
        if isinstance(value, str) and _looks_like_a_secret(value):
            raise RegistryValidationError(
                f"{where} must not carry a credential: the value of {key!r} looks "
                "like one. Keep API keys in the environment, never in model metadata."
            )


def _slug(text: str) -> str:
    """A readable, filesystem-safe slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(text).strip().casefold()).strip("-")
    return cleaned or "model"


def model_id_for(identity: str) -> str:
    """The stable id for a model identity.

    Deterministic and machine-independent for a catalog model (whose identity is
    ``repo/filename``): the same model yields the same id on every run and every
    machine, which is what makes an id safe to persist and compare. For a file the
    user added, the identity is its path, so the id is derived from the path --
    stable for that file where it sits, and deliberately not hard-coded anywhere.
    """
    import hashlib

    key = str(identity or "").strip()
    if not key:
        raise RegistryValidationError("A model identity is required.")
    suffix = hashlib.blake2b(key.encode("utf-8"), digest_size=4).hexdigest()
    return f"{_slug(key)[:64]}-{suffix}"


@dataclass(frozen=True)
class ModelRecord:
    """Everything knowable about one registered model.

    Every field that is not known is **explicitly unknown** -- ``None`` for a
    number, ``""`` for a string, ``()`` for a list -- rather than guessed from a
    filename or a folder. A record for a GGUF and a record for the project's own
    checkpoint therefore differ in which fields carry a value, and both are valid.
    """

    # --- identity -------------------------------------------------------
    #: The primary key. Deterministic from ``identity`` (see :func:`model_id_for`).
    model_id: str
    #: What the weights *are* -- ``repo/filename`` for a catalog model, the
    #: resolved path for a local file, ``fayd-llm`` for the trained checkpoint.
    identity: str
    display_name: str
    #: ``gguf``, ``fayd-checkpoint``, or ``unknown``.
    model_format: str = FORMAT_UNKNOWN
    #: The runtime that runs it (``llama.cpp``, ``fayd-inprocess``, ``ollama``, ...).
    runtime: str = RUNTIME_UNKNOWN
    #: The file on disk, or ``None`` for a model with no local file (a remote one).
    path: Path | None = None
    #: ``catalog``, ``local``, ``trained``, ``remote``, or ``unknown``.
    source: str = SOURCE_UNKNOWN

    # --- facts read from the file --------------------------------------
    size_bytes: int = 0
    quantization: str = ""
    architecture: str = ""
    context_length: int | None = None
    parameter_count: int | None = None
    version: str = ""
    license: str = ""
    #: The integrity verdict from :func:`core.model_files.probe_file`.
    integrity: str = ""
    integrity_detail: str = ""
    digest: str = ""
    digest_algorithm: str = ""

    # --- the user's own fields -----------------------------------------
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    notes: str = ""
    #: Anything a caller wants to keep that the record does not model. Free-form,
    #: validated for credentials, and bounded.
    extra: dict[str, str] = field(default_factory=dict)

    # --- bookkeeping ----------------------------------------------------
    registered_at: str = ""
    updated_at: str = ""
    #: Which mouth's engine this model belongs to, when it came from a catalog.
    engine: str = ""

    # --- derived --------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when the file is present and usable right now.

        Checked live, never cached: a record can outlive the file it describes,
        and reporting a model as ready when its weights were deleted would be the
        one lie this registry must not tell.
        """
        if self.path is None:
            # A model with no local file (a remote one) is available if it was
            # registered without a path at all.
            return self.source == SOURCE_REMOTE
        try:
            return self.path.is_file()
        except OSError:
            return False

    @property
    def verified(self) -> bool:
        """True when the file passed its integrity check when it was registered."""
        return self.integrity == INTEGRITY_OK

    def describe(self) -> str:
        """A one-line, honest description for a status surface."""
        where = str(self.path) if self.path is not None else "no local file"
        bits = [f"{self.display_name} [{self.model_format}/{self.runtime}]"]
        if self.quantization:
            bits.append(self.quantization)
        if self.size_bytes:
            bits.append(f"{self.size_bytes / 1e9:.2f} GB")
        state = "ready" if self.available else "missing"
        if self.available and self.integrity and not self.verified:
            state = f"present but {self.integrity}"
        bits.append(state)
        return f"{' · '.join(bits)} — {where}"

    def to_primitive(self) -> dict[str, Any]:
        """A deterministic, JSON-safe mapping. Keys are sorted on write."""
        return {
            "architecture": self.architecture,
            "capabilities": list(self.capabilities),
            "context_length": self.context_length,
            "digest": self.digest,
            "digest_algorithm": self.digest_algorithm,
            "display_name": self.display_name,
            "engine": self.engine,
            "extra": dict(sorted(self.extra.items())),
            "format": self.model_format,
            "identity": self.identity,
            "integrity": self.integrity,
            "integrity_detail": self.integrity_detail,
            "license": self.license,
            "model_id": self.model_id,
            "notes": self.notes,
            "parameter_count": self.parameter_count,
            "path": str(self.path) if self.path is not None else "",
            "quantization": self.quantization,
            "registered_at": self.registered_at,
            "runtime": self.runtime,
            "size_bytes": self.size_bytes,
            "source": self.source,
            "tags": list(self.tags),
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_primitive(cls, data: object) -> ModelRecord:
        """Read a stored record. Raises :class:`RegistryValidationError` if unusable."""
        if not isinstance(data, dict):
            raise RegistryValidationError("A model record must be a mapping.")
        model_id = _clean_text(data.get("model_id"))
        identity = _clean_text(data.get("identity"))
        if not model_id or not identity:
            raise RegistryValidationError(
                "A model record needs both a model_id and an identity."
            )
        raw_path = _clean_text(data.get("path"))
        extra_raw = data.get("extra")
        extra: dict[str, str] = {}
        if isinstance(extra_raw, dict):
            extra = {
                _clean_text(key, 64): _clean_text(value)
                for key, value in extra_raw.items()
                if _clean_text(key, 64)
            }
        return cls(
            model_id=model_id,
            identity=identity,
            display_name=_clean_text(data.get("display_name")) or identity,
            model_format=_clean_text(data.get("format")) or FORMAT_UNKNOWN,
            runtime=_clean_text(data.get("runtime")) or RUNTIME_UNKNOWN,
            path=Path(raw_path) if raw_path else None,
            source=_clean_text(data.get("source")) or SOURCE_UNKNOWN,
            size_bytes=_as_int(data.get("size_bytes")),
            quantization=_clean_text(data.get("quantization")),
            architecture=_clean_text(data.get("architecture")),
            context_length=_as_optional_int(data.get("context_length")),
            parameter_count=_as_optional_int(data.get("parameter_count")),
            version=_clean_text(data.get("version")),
            license=_clean_text(data.get("license")),
            integrity=_clean_text(data.get("integrity")),
            integrity_detail=_clean_text(data.get("integrity_detail")),
            digest=_clean_text(data.get("digest")),
            digest_algorithm=_clean_text(data.get("digest_algorithm")),
            tags=_as_tags(data.get("tags")),
            capabilities=_as_tags(data.get("capabilities")),
            notes=_clean_text(data.get("notes")),
            extra=extra,
            registered_at=_clean_text(data.get("registered_at")),
            updated_at=_clean_text(data.get("updated_at")),
            engine=_clean_text(data.get("engine")),
        )


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, int(value))


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _as_tags(value: object) -> tuple[str, ...]:
    """Read a stored tag list, keeping only the tags that are usable.

    Reading is deliberately *lenient* about a hand-edited file: a malformed tag is
    dropped rather than making the whole record unreadable. Writing is strict
    (:func:`normalise_tags`), so a bad tag never gets in through the API.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    cleaned = [
        str(item).strip().casefold()
        for item in value
        if str(item).strip() and _TAG_RE.match(str(item).strip().casefold())
    ]
    # Sorted and de-duplicated, so two records built the same way serialize the
    # same way.
    return tuple(sorted(dict.fromkeys(cleaned)))[:MAX_TAGS]


def normalise_tags(tags: object) -> tuple[str, ...]:
    """Validate and normalise a tag collection, raising on a malformed tag.

    Tags are a closed vocabulary of short lower-case tokens on purpose: they are
    what a report groups by, so "GPU" and "gpu" must not become two groups.
    """
    if tags is None:
        return ()
    if isinstance(tags, str):
        candidates: Sequence[object] = (tags,)
    elif isinstance(tags, Iterable):
        candidates = list(tags)
    else:
        raise RegistryValidationError(f"Tags must be a sequence of names, not {type(tags).__name__}.")
    cleaned: list[str] = []
    for item in candidates:
        tag = str(item).strip().casefold()
        if not tag:
            continue
        if not _TAG_RE.match(tag):
            raise RegistryValidationError(
                f"{tag!r} is not a usable tag. Use short lower-case words: letters, "
                "digits, dot, dash, or underscore (at most 32 characters)."
            )
        cleaned.append(tag)
    if len(cleaned) > MAX_TAGS:
        raise RegistryValidationError(f"A model may carry at most {MAX_TAGS} tags.")
    return tuple(sorted(dict.fromkeys(cleaned)))


# --------------------------------------------------------------- the registry


#: Guards a read-modify-write, so two threads cannot interleave one. Re-entrant,
#: because a caller may register a batch from inside another registry call.
_LOCK = threading.RLock()


class ModelRegistry:
    """One durable record per registered model, in the writable data folder.

    ``root`` defaults to :func:`core.paths.model_registry_dir`, which is
    ``<data>/model_registry`` -- git-ignored and ``FAYD_DATA_DIR``-aware, so no
    machine-specific path is ever baked in. A test passes its own ``root``.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else model_registry_dir()

    # --- store ---------------------------------------------------------
    def _store(self) -> Any:
        from storage.json_store import JsonStore

        return JsonStore(self.root)

    def _key(self, model_id: str) -> str:
        return f"model-{_slug(model_id)}"

    # --- reading -------------------------------------------------------
    def list_records(self) -> tuple[ModelRecord, ...]:
        """Every readable record, ordered by display name then id.

        A corrupt record is skipped and counted rather than raising, so one bad
        file cannot make the whole registry unusable. :meth:`corrupt_keys` reports
        which ones were dropped.
        """
        records: list[ModelRecord] = []
        for key in self._keys():
            record = self._read_key(key)
            if record is not None:
                records.append(record)
        records.sort(key=lambda record: (record.display_name.casefold(), record.model_id))
        return tuple(records)

    def _keys(self) -> list[str]:
        try:
            keys = self._store().list_keys(NAMESPACE)
        except Exception:
            return []
        # The reserved ``__`` prefix holds bookkeeping, never a record. A state
        # entry is skipped here, so it is never counted as a corrupt *record*.
        return [
            key for key in keys
            if not key.startswith(_STATE_PREFIX) and not key[len("model-"):].startswith(_STATE_PREFIX)
        ]

    def _read_key(self, key: str) -> ModelRecord | None:
        try:
            raw = self._store().load(NAMESPACE, key)
        except Exception:
            return None
        try:
            return ModelRecord.from_primitive(raw)
        except RegistryValidationError:
            return None

    def corrupt_keys(self) -> tuple[str, ...]:
        """The store keys that could not be read as a record.

        Reported rather than hidden: a user whose registry "lost" a model needs to
        know the file is there and unreadable, not to be told nothing is wrong.
        """
        broken: list[str] = []
        for key in self._keys():
            if self._read_key(key) is None:
                broken.append(key)
        return tuple(sorted(broken))

    def get(self, model_id: str) -> ModelRecord:
        """The record for ``model_id``. Raises :class:`ModelNotFoundError`."""
        wanted = _clean_text(model_id)
        if not wanted:
            raise RegistryValidationError("A model id is required.")
        with _LOCK:
            record = self._read_key(self._key(wanted))
        if record is None:
            raise ModelNotFoundError(
                f"No model is registered as {wanted!r}. "
                "Run `list` to see what is registered."
            )
        return record

    def find(self, model_id: str) -> ModelRecord | None:
        """The record for ``model_id``, or ``None`` -- never raises for a miss."""
        try:
            return self.get(model_id)
        except RegistryError:
            return None

    def find_by_identity(self, identity: str) -> ModelRecord | None:
        """The record whose ``identity`` matches, or ``None``."""
        wanted = _clean_text(identity)
        if not wanted:
            return None
        for record in self.list_records():
            if record.identity == wanted:
                return record
        return None

    def find_by_path(self, path: Path | str) -> ModelRecord | None:
        """The record whose file is ``path``, or ``None``.

        Compared on the resolved path, so the same file reached by a relative and
        an absolute path is recognised as one model.
        """
        try:
            wanted = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            return None
        for record in self.list_records():
            if record.path is None:
                continue
            try:
                if record.path.expanduser().resolve() == wanted:
                    return record
            except (OSError, RuntimeError):
                continue
        return None

    def count(self) -> int:
        """How many readable records the registry holds."""
        return len(self.list_records())

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[ModelRecord]:
        return iter(self.list_records())

    # --- writing -------------------------------------------------------
    def register(
        self,
        path: Path | str,
        *,
        display_name: str = "",
        model_format: str = "",
        runtime: str = "",
        source: str = SOURCE_LOCAL,
        engine: str = "",
        tags: object = None,
        capabilities: object = None,
        notes: str = "",
        version: str = "",
        license: str = "",
        extra: dict[str, str] | None = None,
        identity: str = "",
        digest: bool = False,
        replace_existing: bool = False,
    ) -> ModelRecord:
        """Register a local model file and return its record.

        The path is validated **before** anything is stored: a missing file, a
        folder, or a file that is not a recognised model format raises
        :class:`ModelValidationError`/``ModelFileError`` with the reason, so a bad
        choice is never persisted and discovered later. The format, runtime, size,
        quantisation, architecture, and context length are read from the file
        itself -- never guessed from its name.

        Re-registering the same file updates its record (that is the safe-update
        path) unless ``replace_existing`` is left ``False`` and a *different* file
        already holds the id, which is refused rather than silently overwritten.
        """
        probe = validate_model_path(path, digest=digest)
        resolved = probe.path.resolve()
        resolved_format = model_format or probe.format
        if resolved_format not in KNOWN_FORMATS:
            raise RegistryValidationError(
                f"{resolved_format!r} is not a known format. Known: {', '.join(KNOWN_FORMATS)}."
            )
        resolved_runtime = runtime or runtime_for_format(resolved_format) or RUNTIME_UNKNOWN
        if source not in KNOWN_SOURCES:
            raise RegistryValidationError(
                f"{source!r} is not a known source. Known: {', '.join(KNOWN_SOURCES)}."
            )
        resolved_identity = _clean_text(identity) or str(resolved)
        resolved_engine = _clean_text(engine)
        # When the caller names the engine it came from but not the runtime, the
        # runtime is the one that engine really uses -- a record of what the
        # existing code does, not a new decision.
        if resolved_engine and not runtime:
            resolved_runtime = runtime_for_engine(resolved_engine)

        record = _record_from_probe(
            probe,
            identity=resolved_identity,
            display_name=_clean_text(display_name) or _default_name(probe),
            model_format=resolved_format,
            runtime=resolved_runtime,
            source=source,
            engine=resolved_engine,
            tags=normalise_tags(tags),
            capabilities=normalise_tags(capabilities),
            notes=notes,
            version=version,
            license=license or (probe.gguf.license if probe.gguf is not None else ""),
            extra=extra,
        )

        with _LOCK:
            existing = self.find_by_identity(resolved_identity)
            if existing is not None:
                # Re-registering the *same file* is an update, and keeps when it
                # was first added. Pointing one identity at a *different* file is
                # not: it would silently move a model the user may have chosen by
                # name, so it is refused unless the caller says so.
                if existing.path is not None and not _same_path(existing.path, resolved):
                    if not replace_existing:
                        raise RegistryValidationError(
                            f"The identity {resolved_identity!r} is already registered "
                            f"for {existing.path}, not {resolved}. Register the new file "
                            "under its own identity, or pass `replace_existing=True` if "
                            "this record really should point at the new file."
                        )
                else:
                    record = replace(
                        record, model_id=existing.model_id,
                        registered_at=existing.registered_at or _now(),
                    )
            else:
                clash = self._id_clash(record.model_id, resolved)
                if clash is not None and not replace_existing:
                    raise RegistryValidationError(
                        f"Another model ({clash.display_name!r}) already uses the id "
                        f"{record.model_id!r}. Register with `replace_existing=True` "
                        "only if that record should be replaced."
                    )
            self._write(record)
        return record

    def _id_clash(self, model_id: str, resolved: Path) -> ModelRecord | None:
        """An existing record holding ``model_id`` for a *different* file."""
        other = self._read_key(self._key(model_id))
        if other is None or other.path is None:
            return None
        try:
            return None if other.path.resolve() == resolved else other
        except (OSError, RuntimeError):
            return other

    def register_record(self, record: ModelRecord) -> ModelRecord:
        """Store an already-built record (used by the catalog and trained imports).

        Validated the same way a file registration is, so no path bypasses the
        checks -- but it does not require the file to exist, because a *catalog*
        model may legitimately be registered before it is downloaded.
        """
        _validate_record(record)
        with _LOCK:
            self._write(replace(record, updated_at=_now()))
        return record

    def update(
        self,
        model_id: str,
        *,
        display_name: str | None = None,
        tags: object = None,
        capabilities: object = None,
        notes: str | None = None,
        version: str | None = None,
        license: str | None = None,
        source: str | None = None,
        runtime: str | None = None,
        extra: dict[str, str] | None = None,
        revalidate: bool = False,
    ) -> ModelRecord:
        """Update a record's editable fields, leaving everything else as it was.

        The read-modify-write happens under the registry lock and is written once,
        so a concurrent registration cannot be lost. Validation happens *before*
        the write, so a rejected edit changes nothing.

        ``revalidate=True`` re-reads the file and refreshes the facts that came
        from it (size, quantisation, architecture, integrity) -- the honest way to
        react to a file that was replaced in place.
        """
        with _LOCK:
            current = self.get(model_id)
            changes: dict[str, Any] = {}
            if display_name is not None:
                cleaned = _clean_text(display_name)
                if not cleaned:
                    raise RegistryValidationError("A display name cannot be empty.")
                changes["display_name"] = cleaned
            if tags is not None:
                changes["tags"] = normalise_tags(tags)
            if capabilities is not None:
                changes["capabilities"] = normalise_tags(capabilities)
            if notes is not None:
                changes["notes"] = _clean_text(notes)
            if version is not None:
                changes["version"] = _clean_text(version)
            if license is not None:
                changes["license"] = _clean_text(license)
            if source is not None:
                if source not in KNOWN_SOURCES:
                    raise RegistryValidationError(
                        f"{source!r} is not a known source. Known: {', '.join(KNOWN_SOURCES)}."
                    )
                changes["source"] = source
            if runtime is not None:
                changes["runtime"] = _clean_text(runtime) or RUNTIME_UNKNOWN
            if extra is not None:
                _reject_secrets(dict(extra), where="A model's extra fields")
                changes["extra"] = {
                    _clean_text(key, 64): _clean_text(value)
                    for key, value in extra.items()
                    if _clean_text(key, 64)
                }
            if revalidate and current.path is not None:
                changes.update(_refresh_from_file(current.path))
            candidate = replace(current, **changes)
            _validate_record(candidate)
            stored = replace(candidate, updated_at=_now())
            self._write(stored)
            return stored

    def remove(self, model_id: str, *, confirm: bool = False) -> ModelRecord:
        """Drop a record. **Never** touches the model file.

        A destructive call has to say so: ``confirm=False`` raises
        :class:`ConfirmationRequired` rather than doing it. The returned record is
        the one that was removed, so a caller can report exactly what it forgot.
        """
        if not confirm:
            raise ConfirmationRequired(
                "Removing a model from the registry is destructive and must be "
                "confirmed explicitly (`confirm=True`). The model file is never "
                "deleted, so it can be registered again."
            )
        with _LOCK:
            record = self.get(model_id)
            try:
                self._store().delete(NAMESPACE, self._key(record.model_id))
            except Exception as error:
                raise RegistryError(f"Could not remove {model_id!r}: {error}") from error
        return record

    def clear(self, *, confirm: bool = False) -> int:
        """Forget every record. **Never** touches any model file."""
        if not confirm:
            raise ConfirmationRequired(
                "Clearing the registry is destructive and must be confirmed "
                "explicitly (`confirm=True`). No model file is ever deleted."
            )
        removed = 0
        with _LOCK:
            for record in self.list_records():
                try:
                    self._store().delete(NAMESPACE, self._key(record.model_id))
                    removed += 1
                except Exception:
                    continue
        return removed

    def _write(self, record: ModelRecord) -> None:
        _validate_record(record)
        try:
            self._store().save(NAMESPACE, self._key(record.model_id), record.to_primitive())
        except Exception as error:
            raise RegistryError(
                f"Could not save the record for {record.display_name!r}: {error}"
            ) from error

    # --- import from what already exists -------------------------------
    def import_custom_models(self, *, replace_existing: bool = False) -> tuple[ModelRecord, ...]:
        """Bring :mod:`core.custom_models` entries in as ordinary records.

        One-way on purpose: the mouth windows keep their own shortlist exactly as
        it is, and this registry gains a record per file the user already pointed a
        window at. A file that is gone is skipped with a reason rather than
        creating a record for something that cannot be run; nothing is written back
        to ``custom_models``.
        """
        try:
            from core import custom_models
        except Exception:
            return ()
        imported: list[ModelRecord] = []
        try:
            entries = list(custom_models.list_models())
        except Exception:
            # An unreadable shortlist must not break the registry: there is simply
            # nothing to import from it.
            return ()
        for entry in entries:
            if self.find_by_path(entry.path) is not None and not replace_existing:
                continue
            try:
                imported.append(
                    self.register(
                        entry.path,
                        display_name=entry.title,
                        source=SOURCE_LOCAL,
                        identity=f"custom:{entry.path}",
                        replace_existing=replace_existing,
                    )
                )
            except (ModelFileError, RegistryError):
                continue
        return tuple(imported)

    def export(self) -> dict[str, Any]:
        """The whole registry as one deterministic mapping, for a backup."""
        return {
            "namespace": NAMESPACE,
            "records": [record.to_primitive() for record in self.list_records()],
            "version": 1,
        }


def _same_path(first: Path, second: Path) -> bool:
    """True when two paths name the same file, resolving each defensively."""
    try:
        return first.expanduser().resolve() == second.expanduser().resolve()
    except (OSError, RuntimeError):
        return first == second


def _default_name(probe: FileProbe) -> str:
    """A display name from the file's own metadata, else its filename.

    The GGUF's ``general.name`` is what the model calls itself, so it is preferred;
    the filename is the honest fallback, not a guess about the model.
    """
    if probe.gguf is not None and probe.gguf.name:
        return probe.gguf.name
    return probe.path.name


def _record_from_probe(
    probe: FileProbe,
    *,
    identity: str,
    display_name: str,
    model_format: str,
    runtime: str,
    source: str,
    engine: str,
    tags: tuple[str, ...],
    capabilities: tuple[str, ...],
    notes: str,
    version: str,
    license: str,
    extra: dict[str, str] | None,
) -> ModelRecord:
    """Build a record from what the file actually said."""
    context_length: int | None = None
    architecture = ""
    quantization = ""
    parameter_count: int | None = None
    if probe.gguf is not None:
        context_length = probe.gguf.context_length
        architecture = probe.gguf.architecture
        quantization = probe.gguf.quantization
        parameter_count = probe.gguf.parameter_count
    elif probe.checkpoint is not None:
        context_length = probe.checkpoint.block_size
        architecture = "fayd-transformer"
        parameter_count = probe.checkpoint.parameter_count
        quantization = "float64"
    clean_extra = dict(extra or {})
    _reject_secrets(clean_extra, where="A model's extra fields")
    now = _now()
    return ModelRecord(
        model_id=model_id_for(identity),
        identity=identity,
        display_name=display_name,
        model_format=model_format,
        runtime=runtime,
        path=probe.path,
        source=source,
        size_bytes=probe.size_bytes,
        quantization=quantization,
        architecture=architecture,
        context_length=context_length,
        parameter_count=parameter_count,
        version=_clean_text(version),
        license=_clean_text(license),
        integrity=probe.integrity,
        integrity_detail=probe.detail,
        digest=probe.digest,
        digest_algorithm=probe.digest_algorithm,
        tags=tags,
        capabilities=capabilities,
        notes=_clean_text(notes),
        extra={key: _clean_text(value) for key, value in clean_extra.items()},
        registered_at=now,
        updated_at=now,
        engine=_clean_text(engine),
    )


def _refresh_from_file(path: Path) -> dict[str, Any]:
    """The fields a re-validation refreshes, from the file as it is now."""
    from core.model_files import probe_file

    probe = probe_file(path)
    changes: dict[str, Any] = {
        "size_bytes": probe.size_bytes,
        "integrity": probe.integrity,
        "integrity_detail": probe.detail,
    }
    if probe.gguf is not None:
        changes["quantization"] = probe.gguf.quantization
        changes["architecture"] = probe.gguf.architecture
        changes["context_length"] = probe.gguf.context_length
        changes["parameter_count"] = probe.gguf.parameter_count
        if probe.gguf.license:
            changes["license"] = probe.gguf.license
    elif probe.checkpoint is not None:
        changes["architecture"] = "fayd-transformer"
        changes["context_length"] = probe.checkpoint.block_size
        changes["parameter_count"] = probe.checkpoint.parameter_count
    return changes


def _validate_record(record: ModelRecord) -> None:
    """Refuse a record that would be unusable, before it is written.

    The checks are the ones that keep a *listing* safe: an id, an identity, a
    known format and source, a bounded display name, and no credential anywhere.
    """
    if not _clean_text(record.model_id):
        raise RegistryValidationError("A model record needs a model_id.")
    if not _clean_text(record.identity):
        raise RegistryValidationError("A model record needs an identity.")
    if not _clean_text(record.display_name):
        raise RegistryValidationError("A model record needs a display name.")
    if len(record.display_name) > MAX_TEXT:
        raise RegistryValidationError(
            f"A display name must be at most {MAX_TEXT} characters."
        )
    if record.model_format not in KNOWN_FORMATS:
        raise RegistryValidationError(
            f"{record.model_format!r} is not a known format. Known: {', '.join(KNOWN_FORMATS)}."
        )
    if record.source not in KNOWN_SOURCES:
        raise RegistryValidationError(
            f"{record.source!r} is not a known source. Known: {', '.join(KNOWN_SOURCES)}."
        )
    _reject_secrets(dict(record.extra), where="A model's extra fields")
    if _looks_like_a_secret(record.notes):
        raise RegistryValidationError(
            "The notes field must not carry a credential. Model metadata is shown "
            "and exported, so keys are never stored here."
        )


# ------------------------------------------------------------- module helpers


def registry(root: Path | str | None = None) -> ModelRegistry:
    """The registry at ``root`` (the default data folder when omitted)."""
    return ModelRegistry(root)


def registered_models(root: Path | str | None = None) -> tuple[ModelRecord, ...]:
    """Every registered model, for a caller that only wants to read."""
    return ModelRegistry(root).list_records()


def registry_path(root: Path | str | None = None) -> Path:
    """Where the registry lives, for a diagnostic or a backup."""
    return Path(root) if root is not None else model_registry_dir()


def time_ago(stamp: str, *, now: float | None = None) -> str:
    """A readable age for a timestamp, or the stamp itself when unreadable.

    Presentation only: it never changes a stored value, and an unparseable stamp
    is returned as-is rather than shown as "0 seconds ago".
    """
    text = _clean_text(stamp)
    if not text:
        return "never"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    reference = time.time() if now is None else now
    delta = reference - parsed.timestamp()
    if delta < 0:
        return "just now"
    for limit, divisor, unit in (
        (60, 1, "second"),
        (3600, 60, "minute"),
        (86400, 3600, "hour"),
        (604800, 86400, "day"),
    ):
        if delta < limit:
            count = max(1, int(delta / divisor))
            return f"{count} {unit}{'' if count == 1 else 's'} ago"
    count = int(delta / 604800)
    return f"{count} week{'' if count == 1 else 's'} ago"


__all__ = [
    "KNOWN_FORMATS",
    "KNOWN_SOURCES",
    "MAX_TAGS",
    "MAX_TEXT",
    "NAMESPACE",
    "SOURCE_CATALOG",
    "SOURCE_DOWNLOAD",
    "SOURCE_LOCAL",
    "SOURCE_REMOTE",
    "SOURCE_TRAINED",
    "SOURCE_UNKNOWN",
    "ConfirmationRequired",
    "ModelNotFoundError",
    "ModelRecord",
    "ModelRegistry",
    "RegistryError",
    "RegistryValidationError",
    "model_id_for",
    "normalise_tags",
    "registered_models",
    "registry",
    "registry_path",
    "time_ago",
]
