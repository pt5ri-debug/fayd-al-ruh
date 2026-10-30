"""The Characters management service: the engine behind the window and the CLI.

One Tk-free service drives the whole Characters surface, so the window and
``python -m cli characters`` share one set of characters, one configuration, and
one APPLY -- no logic is reimplemented in either front end.

What it exposes, in the order the task lists the sections:

* the seven characters and the project's own model, as readable summaries;
* a character's full configuration, section by section (General, Personality,
  Language Model, Voice, Learning);
* the choices each section offers (the project's own model catalog, the voices the
  machine really reports here, the learning modes), so a menu can never invent one;
* **validation** of a draft, and **APPLY** -- which saves, activates what it can
  now, and rolls back on failure;
* the character's own **learning management**: statistics, a search over its
  records, an export/backup, and a reset that must be explicitly confirmed;
* the main model's **knowledge sharing** from the six permitted characters, with
  Natasha refused in the backend.

Everything is exercised headless: no display, no model, and no microphone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.character_config import (
    LANGUAGES,
    LEARNING_MODES,
    SECTION_GENERAL,
    SECTION_LEARNING,
    SECTION_MODEL,
    SECTION_PERSONALITY,
    SECTION_VOICE,
    SECTIONS,
    CharacterConfig,
    ConfigError,
)
from core.character_learning_store import CharacterLearningStore, LearningStats
from core.character_registry import ApplyReport, CharacterRegistry
from core.knowledge_sharing import KnowledgeSharing, ShareReport
from core.learning_audit import LearningAudit
from core.learning_isolation import (
    MAIN_MODEL_OWNER,
    PRIVATE_REASONS,
    LearningAccessError,
    is_owner,
    is_private,
    owner_for,
)
from core.learning_records import STATUS_LABELS, LearningRecord
from core.paths import data_dir

#: What each owner is, in one honest line, for the Characters list.
OWNER_BLURBS: dict[str, str] = {
    "jarvis": "The grounded, cited assistant (your own sources, and the web when you allow it).",
    "ultron": "The general local model: speaks on any subject, runs on this machine.",
    "vision": "The petroleum local model: reservoir, drilling, production, petrophysics.",
    "banner": "The presentation model: turns a discussion into an outline and slides.",
    "natasha": "The companion: memory, tone, and relationship. Her records are private.",
    "hakeem": "The scholar of religion and history: Islam, Christianity, Judaism, and history.",
    "cloud": "The remote model: reaches an OpenAI-compatible provider over the network.",
    MAIN_MODEL_OWNER: "The project's own model: learns from its own work and from six characters.",
}


@dataclass(frozen=True)
class CharacterSummary:
    """One row in the Characters list."""

    key: str
    name: str
    role: str
    active: bool
    private: bool
    records: int
    blurb: str

    def describe(self) -> str:
        flag = "active" if self.active else "inactive"
        private = ", private records" if self.private else ""
        return f"{self.name} ({self.key}) — {flag}{private}, {self.records} learning record(s)"


@dataclass
class SaveResult:
    """The outcome of one APPLY, with the report and any validation error."""

    report: ApplyReport | None = None
    error: ConfigError | None = None

    @property
    def ok(self) -> bool:
        return self.report is not None and self.report.ok

    def describe(self) -> str:
        if self.error is not None:
            return f"Not applied: {self.error}"
        if self.report is None:
            return "Nothing was applied."
        return self.report.describe()


class CharacterManagerService:
    """The one engine behind the Characters window and CLI.

    ``data_root`` keeps everything (the applied configuration, the learning stores,
    the audit) under one folder, which a test uses and a portable build may too. The
    layout beneath it is always the isolation policy's own, so no character can be
    pointed at another's store.
    """

    def __init__(
        self,
        *,
        data_root: Path | None = None,
        registry: CharacterRegistry | None = None,
        sharing: KnowledgeSharing | None = None,
        audit: LearningAudit | None = None,
    ) -> None:
        self._data_root = data_root
        self._registry = registry if registry is not None else CharacterRegistry(data_root=data_root)
        self._audit = audit if audit is not None else self._build_audit()
        self._sharing = (
            sharing
            if sharing is not None
            else KnowledgeSharing(data_root=data_root, audit=self._audit)
        )

    def _build_audit(self) -> LearningAudit:
        if self._data_root is None:
            return LearningAudit()
        return LearningAudit(Path(self._data_root) / "learning" / MAIN_MODEL_OWNER / "audit")

    # ------------------------------------------------------------ characters

    def keys(self) -> tuple[str, ...]:
        """Every owner, characters first, then the project's own model."""
        return self._registry.keys()

    def character_keys(self) -> tuple[str, ...]:
        """Only the seven characters (no main model)."""
        return tuple(key for key in self.keys() if key != MAIN_MODEL_OWNER)

    def summaries(self) -> tuple[CharacterSummary, ...]:
        """One :class:`CharacterSummary` per owner, in the project's order."""
        rows: list[CharacterSummary] = []
        for key in self.keys():
            config = self._registry.load(key)
            rows.append(
                CharacterSummary(
                    key=key,
                    name=config.general.name,
                    role=config.general.role,
                    active=config.general.active,
                    private=is_private(key),
                    records=self.store(key).stats().total,
                    blurb=OWNER_BLURBS.get(key, ""),
                )
            )
        return tuple(rows)

    def summary(self, key: str) -> CharacterSummary:
        """One owner's summary."""
        for row in self.summaries():
            if row.key == owner_for(key):
                return row
        raise KeyError(key)

    # ---------------------------------------------------------- configuration

    def config(self, key: str) -> CharacterConfig:
        """The effective configuration for ``key``.

        An unknown key is **refused** rather than answered with a neutral default:
        the manager exists for the seven characters and the project's own model, so
        a typo should be reported, not silently given a made-up character. (The
        registry itself stays tolerant, so a future character added to the seeds
        still loads -- the strictness belongs to the user-facing surface.)
        """
        if not is_owner(key):
            raise LearningAccessError(f"{key!r} is not a known character or the main model.")
        return self._registry.load(key)

    def sections(self) -> tuple[tuple[str, str], ...]:
        """The configuration sections, in the order the window shows them."""
        return SECTIONS

    def validate(self, config: CharacterConfig) -> ConfigError | None:
        """Validate a draft; returns the first problem, or ``None`` when it is valid."""
        try:
            config.validate()
        except ConfigError as error:
            return error
        return None

    def apply(self, config: CharacterConfig) -> SaveResult:
        """APPLY a draft: validate, save, activate, roll back on failure.

        The activation hook applies what can be applied *now* -- the character's
        personality instruction is re-rendered on the next answer, and the learning
        settings are already read fresh by the pipeline, so those are named as
        activated. A model or voice change needs the mouth rebuilt, which this
        service cannot do for a *live* window, so those are reported as
        restart-required rather than pretended.
        """
        error = self.validate(config)
        if error is not None:
            return SaveResult(error=error)

        def activate(applied: CharacterConfig) -> tuple[str, ...]:
            # The personality and learning settings take effect immediately: the
            # mouth re-reads its configuration on every answer and the pipeline
            # re-reads the settings per record. The model and voice are reloaded by
            # the caller (the window rebuilds the mouth), so they are named here
            # only when the caller has done so -- see ``activate_live``.
            return (SECTION_GENERAL, SECTION_PERSONALITY, SECTION_LEARNING)

        report = self._registry.apply(config, activate=activate)
        return SaveResult(report=report)

    def reset(self, key: str) -> CharacterConfig:
        """Drop the applied configuration, back to the shipped default."""
        return self._registry.reset(key)

    def is_customised(self, key: str) -> bool:
        """True when the user has applied a configuration for ``key``."""
        return self._registry.is_customised(key)

    # ---------------------------------------------------------------- choices

    def model_choices(self, key: str) -> tuple[tuple[str, str, bool], ...]:
        """The language models ``key`` may run: ``(catalog key, title, available)``.

        Uses the project's own catalog through the mouth the character answers
        with, so the choices are exactly what the mouth can really load.
        """
        from core import language_models as catalog
        from core.learning_isolation import mouth_key

        mouth = mouth_key(key)
        rows: list[tuple[str, str, bool]] = []
        try:
            models = catalog.models_for_choice(mouth)
        except (ValueError, KeyError):
            return ()
        for model in models:
            rows.append((model.key, model.title, bool(model.available)))
        return tuple(rows)

    def voice_choices(self) -> tuple[tuple[str, str, str], ...]:
        """Every voice the machine really reports: ``(key, label, source)``.

        Drawn from :mod:`core.voices`, so the menu can only ever offer a voice that
        exists -- a voice whose engine or weights are absent is not listed.
        """
        from core.voices import voice_sections

        rows: list[tuple[str, str, str]] = []
        try:
            sections = voice_sections()
        except (ValueError, OSError):
            return ()
        for section in sections:
            for voice in section.voices:
                rows.append((voice.key, voice.label, section.source))
        return tuple(rows)

    def voice_sources(self) -> tuple[tuple[str, str], ...]:
        """The voice engines the project knows: ``(source key, title)``."""
        from core.voices import SOURCE_TITLES

        return tuple(SOURCE_TITLES)

    def learning_modes(self) -> tuple[str, ...]:
        """The learning modes the settings may choose."""
        return LEARNING_MODES

    def languages(self) -> tuple[str, ...]:
        """The languages a character may be pinned to (or ``match``)."""
        return LANGUAGES

    def voice_support(self, engine: str) -> str:
        """What a voice engine can really do with speed/pitch/etc. (honest note)."""
        from core.voice_moods import support_note

        return support_note(engine)

    # ---------------------------------------------------------------- storage

    def store(self, key: str) -> CharacterLearningStore:
        """The owner's own learning store (never another's)."""
        return CharacterLearningStore(key, root=self._data_root, audit=self._audit)

    def stats(self, key: str) -> LearningStats:
        """The owner's learning statistics."""
        return self.store(key).stats()

    def records(self, key: str, *, limit: int = 100) -> tuple[LearningRecord, ...]:
        """The owner's most recent learning records."""
        return self.store(key).recent(limit=limit)

    def search(self, key: str, query: str, *, limit: int = 50) -> tuple[LearningRecord, ...]:
        """Search the owner's own records (never another character's)."""
        return self.store(key).search(query, limit=limit)

    def history(self, key: str, *, limit: int = 50) -> tuple[str, ...]:
        """The owner's learning history as readable lines, newest first."""
        rows = self.store(key).recent(limit=limit)
        return tuple(record.describe() for record in reversed(rows))

    def verify(self, key: str, record_id: str, *, note: str = "") -> LearningRecord | None:
        """Verify one of the owner's records (supervised verification)."""
        return self.store(key).revise(
            record_id, "verified", note=note or "verified by the user."
        )

    def reject(self, key: str, record_id: str, *, note: str = "") -> LearningRecord | None:
        """Reject one of the owner's records, keeping it and the reason."""
        return self.store(key).revise(record_id, "rejected", note=note or "rejected by the user.")

    def export(self, key: str, path: str | Path | None = None) -> Path:
        """Export the owner's learning records to a JSON file (a backup).

        With no path, a dated name under the owner's own learning directory is
        used, so an export never lands in the working tree.
        """
        store = self.store(key)
        target = Path(path) if path is not None else store.root / f"export-{_stamp()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = store.export()
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    def reset_learning(self, key: str, *, confirm: bool = False) -> bool:
        """Forget the owner's learning records -- only with explicit confirmation.

        The task requires the reset to be explicitly confirmed, so ``confirm`` must
        be ``True``; without it the call refuses and changes nothing.
        """
        if not confirm:
            return False
        self.store(key).clear()
        return True

    # --------------------------------------------------------------- sharing

    def shareable_characters(self) -> tuple[str, ...]:
        """The characters the main model may learn from."""
        return self._sharing.shareable_characters()

    def private_characters(self) -> tuple[str, ...]:
        """The characters whose records are never shared (Natasha)."""
        return self._sharing.private_characters()

    def can_share(self, character: str) -> bool:
        """Whether the main model may read ``character``'s records."""
        return self._sharing.can_read(character)

    def share_reason(self, character: str) -> str:
        """Why a character is (or is not) shareable, in one line."""
        if is_private(character):
            return PRIVATE_REASONS.get(owner_for(character), "private by design.")
        return "shareable: the main model may learn from this character."

    def learn_from_character(self, character: str, *, limit: int = 100) -> Any:
        """Learn from one character into the main model's own store."""
        return self._sharing.learn_from(character, limit=limit)

    def learn_from_all_characters(self, *, limit: int = 100) -> ShareReport:
        """Learn from every shareable character (Natasha is never included)."""
        return self._sharing.learn_from_all(limit=limit)

    def main_model_records(self, *, limit: int = 100) -> tuple[LearningRecord, ...]:
        """What the main model has learned into its own store."""
        return self.store(MAIN_MODEL_OWNER).recent(limit=limit)

    # ----------------------------------------------------------------- audit

    def audit_tail(self, *, limit: int = 40) -> tuple[str, ...]:
        """The recent learning-audit lines, including refused accesses."""
        return tuple(event.describe() for event in self._audit.events(limit=limit))

    def denials(self, *, limit: int = 40) -> tuple[str, ...]:
        """The recent refused cross-character access attempts."""
        return tuple(event.describe() for event in self._audit.denials(limit=limit))

    # -------------------------------------------------------------- reporting

    def status(self) -> str:
        """A readable report of every character and what it holds."""
        lines = ["Characters and their learning:"]
        for row in self.summaries():
            lines.append(f"  {row.describe()}")
            if row.blurb:
                lines.append(f"      {row.blurb}")
        lines.append("")
        lines.append(
            f"Knowledge sharing: the main model learns from "
            f"{', '.join(self.shareable_characters())}; "
            f"{', '.join(self.private_characters())} is private."
        )
        return "\n".join(lines)

    def describe_config(self, key: str) -> str:
        """The character's configuration, section by section, for a status line."""
        config = self.config(key)
        lines = [config.describe(), ""]
        for section, title in SECTIONS:
            lines.append(f"[{title}]")
            lines.append(_section_text(config, section))
        return "\n".join(lines)

    def describe_record(self, record: LearningRecord) -> str:
        """One record, with its status spelled out."""
        status = STATUS_LABELS.get(record.learning_status, record.learning_status)
        origin = record.source_character or record.source_type
        return f"[{status}] {record.knowledge or record.answer} (from {origin}, {record.confidence:.2f})"


#: The markers a report uses for an unset field, kept as constants because an
#: apostrophe inside an f-string expression is a syntax error.
DEFAULT_MARK = "(the mouth's default)"
EMPTY_MARK = "(none)"


def _section_text(config: CharacterConfig, section: str) -> str:
    """A short readable rendering of one configuration section."""
    if section == SECTION_GENERAL:
        general = config.general
        return (
            f"  name: {general.name}\n  role: {general.role or EMPTY_MARK}\n"
            f"  active: {general.active}\n  description: {general.description or EMPTY_MARK}"
        )
    if section == SECTION_PERSONALITY:
        personality = config.personality
        return (
            f"  traits: {len(personality.traits)}\n  rules: {len(personality.rules)}\n"
            f"  formality: {personality.formality}\n"
            f"  emotional expression: {personality.emotional_expression}\n"
            f"  language: {personality.preferred_language}\n"
            f"  instructions: {personality.instructions or '(none)'}"
        )
    if section == SECTION_MODEL:
        model = config.model
        return (
            f"  key: {model.key or DEFAULT_MARK}\n"
            f"  provider: {model.provider or EMPTY_MARK}\n  model: {model.model or EMPTY_MARK}\n"
            f"  temperature: {model.temperature}\n  context: {model.context_length}\n"
            f"  max tokens: {model.max_tokens}\n  active: {model.active}"
        )
    if section == SECTION_VOICE:
        voice = config.voice
        return (
            f"  engine: {voice.engine or DEFAULT_MARK}\n"
            f"  voice: {voice.voice or DEFAULT_MARK}\n"
            f"  language: {voice.language or '(auto)'}\n  speed: {voice.speed}\n"
            f"  pitch: {voice.pitch}\n  volume: {voice.volume}\n  active: {voice.active}"
        )
    learning = config.learning
    return (
        f"  enabled: {learning.enabled}\n  from the model: {learning.learn_from_model}\n"
        f"  from the internet: {learning.learn_from_internet}\n"
        f"  mode: {learning.mode}\n  verification threshold: {learning.verification_threshold}\n"
        f"  confidence threshold: {learning.confidence_threshold}\n"
        f"  sharing: {config.effective_sharing}"
    )


def _stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def default_service() -> CharacterManagerService:
    """The service over the real data directory."""
    return CharacterManagerService(data_root=None)


def data_root_for_test() -> Path:
    """The real data directory (a convenience for a caller that wants it)."""
    return data_dir()


__all__ = [
    "OWNER_BLURBS",
    "CharacterManagerService",
    "CharacterSummary",
    "SaveResult",
    "data_root_for_test",
    "default_service",
]
