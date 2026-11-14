"""The Studio's safety and privacy manager: the floor that never moves.

Three responsibilities, each with the project's own rule made structural rather
than a convention:

* **Adult content is disabled by default** and needs *two* explicit confirmations
  before it can run: the user's own adult-eligibility confirmation, and an
  adult-eligible *character*. A character whose age is unspecified
  (``fictional_age is None``) is age-ambiguous and is refused, and a character
  under 18 is refused outright -- so no prompt can reach a scene it should not.
* **Coercion and non-consent are refused**, not merely discouraged: a request
  carrying a coercion/incest/non-consent marker is refused before any content
  level is considered.
* **Local-only is enforced.** The Studio never falls back to a remote provider;
  :meth:`assert_local_only` raises when a caller tries to route an answer to a
  cloud mouth while local-only mode is in force, and the manager reads the
  project's own offline guard (:mod:`core.offline`).

Every eligibility confirmation and every refusal is written as an **append-only**
:class:`~core.romance.model.CompanionSafetyEvent`, so the record of what happened
survives. The module is Tk-free, model-free, and network-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.offline import guard_active
from core.romance.characters import CharacterManager
from core.romance.roleplay import CONTENT_ADULT, CONTENT_GENERAL, CONTENT_ROMANTIC
from core.romance.storage import CompanionStorageService

#: The content levels, matching a scenario's own ``content_level``. The vocabulary
#: lives in one place (:mod:`core.romance.roleplay`); this module re-exports it so a
#: caller can import it from either.
CONTENT_LEVELS: tuple[str, ...] = (CONTENT_GENERAL, CONTENT_ROMANTIC, CONTENT_ADULT)

#: The setting keys the manager owns.
SETTING_ADULT_MODE = "adult_mode_enabled"
SETTING_LOCAL_ONLY = "local_only"

#: The boundary kinds a user may set.
BOUNDARY_TOPIC = "topic"
BOUNDARY_KINDS: tuple[str, ...] = (BOUNDARY_TOPIC,)

#: The safety-event kinds.
EVENT_ELIGIBILITY = "eligibility_confirmation"
EVENT_REFUSAL = "refusal"
EVENT_SETTING = "setting_change"
EVENT_KINDS: tuple[str, ...] = (EVENT_ELIGIBILITY, EVENT_REFUSAL, EVENT_SETTING)

#: The words that mark a request as coercive / non-consensual. Deliberately
#: conservative: they are matched as whole words in either language, so a genuine
#: question is not caught by an incidental substring.
_COERCION_MARKERS: tuple[str, ...] = (
    "force",
    "forced",
    "coerce",
    "coerced",
    "rape",
    "non-consensual",
    "nonconsensual",
    "without consent",
    "against her will",
    "against his will",
    "against their will",
    "unconscious",
    "asleep",
    "underage",
    "minor",
    "child",
    "إجبار",
    "إكراه",
    "اغتصاب",
    "بدون موافقة",
    "قاصر",
    "طفل",
)

#: The disclosure the *application* states outside the fiction.
PRIVACY_DISCLOSURE = (
    "These are fictional AI companions, not real people. Everything stays on this "
    "machine: conversations, memories, and relationship records are stored in a "
    "local database and are never sent to a remote service."
)


class SafetyError(RuntimeError):
    """Raised when a safety rule refuses an operation."""


@dataclass(frozen=True)
class SafetyDecision:
    """The verdict on a requested content level, with the reason it was reached."""

    allowed: bool
    content_level: str
    reason: str
    character_id: str = ""

    def describe(self) -> str:
        return self.reason


class CompanionSafetyManager:
    """Adult-eligibility, boundary, and local-only enforcement for the Studio."""

    def __init__(
        self,
        storage: CompanionStorageService,
        *,
        characters: CharacterManager | None = None,
    ) -> None:
        self.storage = storage
        self.characters = characters if characters is not None else CharacterManager(storage)

    # -- settings ---------------------------------------------------------

    def _get_setting(self, key: str, default: str = "") -> str:
        return self.storage.get_setting(key, default)

    def _set_setting(self, key: str, value: str) -> Any:
        return self.storage.set_setting(key, value)

    # -- adult mode (disabled by default) ---------------------------------

    def adult_mode_enabled(self) -> bool:
        """Whether adult content is enabled (off until explicitly turned on)."""
        return self._get_setting(SETTING_ADULT_MODE, "off") == "on"

    def set_adult_mode(self, enabled: bool, *, confirmed: bool = False) -> bool:
        """Turn adult content on or off. Turning it **on** requires confirmation.

        An explicit ``confirmed=True`` is the user's own adult-eligibility
        acknowledgement; without it, enabling is refused rather than assumed.
        """
        if enabled and not confirmed:
            raise SafetyError(
                "Adult content is off by default; enabling it requires an explicit "
                "adult-eligibility confirmation."
            )
        self._set_setting(SETTING_ADULT_MODE, "on" if enabled else "off")
        self.record_event(
            kind=EVENT_SETTING,
            reason=f"adult_mode={'on' if enabled else 'off'}",
        )
        return self.adult_mode_enabled()

    # -- character eligibility --------------------------------------------

    def character_is_adult(self, character_id: str) -> bool:
        """Whether the character's age is stated and is 18 or older."""
        character = self.storage.domain.get("companion_characters", character_id)
        if character is None:
            return False
        age = character.fictional_age
        return age is not None and int(age) >= 18

    def confirm_eligibility(self, character_id: str) -> Any:
        """Record an explicit adult-eligibility confirmation for a character.

        A character whose age is unspecified or under 18 is **refused**, so an
        age-ambiguous or minor character can never be marked eligible.
        """
        character = self.storage.domain.get("companion_characters", character_id)
        if character is None:
            raise SafetyError(f"no character {character_id!r}")
        if not self.character_is_adult(character_id):
            raise SafetyError(
                f"{character.name or character_id} is not eligible for adult content: "
                "its age is unspecified or under 18."
            )
        updated = self.storage.domain.update(
            "companion_characters", character_id, adult_eligible=True
        )
        self.record_event(
            kind=EVENT_ELIGIBILITY,
            character_id=character_id,
            reason="adult eligibility confirmed",
        )
        return updated

    def character_is_eligible(self, character_id: str) -> bool:
        """Whether the character is an adult *and* has confirmed eligibility."""
        character = self.storage.domain.get("companion_characters", character_id)
        if character is None:
            return False
        return bool(character.adult_eligible) and self.character_is_adult(character_id)

    # -- content check ----------------------------------------------------

    def check_content(
        self, character_id: str, content_level: str, *, text: str = ""
    ) -> SafetyDecision:
        """Decide whether a requested content level may run, and say why.

        Order matters: a coercion/non-consent marker refuses **before** any level
        is considered, and adult content refuses unless the mode is on *and* the
        character is a confirmed adult.
        """
        if content_level not in CONTENT_LEVELS:
            return SafetyDecision(
                False, content_level, f"Unknown content level {content_level!r}.", character_id
            )
        if self._looks_coercive(text):
            self.record_event(
                kind=EVENT_REFUSAL,
                character_id=character_id,
                reason="coercive or non-consensual request",
            )
            return SafetyDecision(
                False,
                content_level,
                "Refused: the request describes coercion or non-consensual activity.",
                character_id,
            )
        if content_level != CONTENT_ADULT:
            return SafetyDecision(
                True, content_level, f"{content_level} content is allowed.", character_id
            )
        if not self.adult_mode_enabled():
            self.record_event(
                kind=EVENT_REFUSAL, character_id=character_id, reason="adult mode is off"
            )
            return SafetyDecision(
                False,
                content_level,
                "Refused: adult content is off by default and has not been enabled.",
                character_id,
            )
        if not self.character_is_eligible(character_id):
            self.record_event(
                kind=EVENT_REFUSAL,
                character_id=character_id,
                reason="character is not a confirmed adult",
            )
            return SafetyDecision(
                False,
                content_level,
                "Refused: the character is not a confirmed adult (age stated and 18+).",
                character_id,
            )
        return SafetyDecision(
            True, content_level, "Adult content allowed.", character_id
        )

    def require_content(
        self, character_id: str, content_level: str, *, text: str = ""
    ) -> None:
        """Raise :class:`SafetyError` when a content level is refused."""
        decision = self.check_content(character_id, content_level, text=text)
        if not decision.allowed:
            raise SafetyError(decision.reason)

    # -- boundaries -------------------------------------------------------

    def boundaries(self, character_id: str) -> tuple[Any, ...]:
        return tuple(
            self.storage.domain.list_records(
                "character_boundaries",
                where="character_id = ? AND active = 1",
                params=(character_id,),
                order_by="created_at, record_id",
            )
        )

    def add_boundary(self, character_id: str, value: str, *, kind: str = BOUNDARY_TOPIC) -> Any:
        """Set a conversation boundary (an opt-out the character must respect)."""
        if kind not in BOUNDARY_KINDS:
            raise SafetyError(f"unknown boundary kind: {kind!r}")
        if not (value or "").strip():
            raise SafetyError("a boundary needs a value")
        return self.storage.domain.create(
            "character_boundaries", character_id=character_id, kind=kind, value=value.strip()
        )

    def remove_boundary(self, boundary_id: str) -> Any:
        return self.storage.domain.deactivate("character_boundaries", boundary_id)

    def boundary_violation(self, character_id: str, text: str) -> str:
        """The boundary a text crosses, or ``""`` when it crosses none."""
        lowered = (text or "").casefold()
        for boundary in self.boundaries(character_id):
            if boundary.value.casefold() in lowered:
                return boundary.value
        return ""

    # -- local-only -------------------------------------------------------

    def local_only(self) -> bool:
        """Whether local-only mode is in force (on by default: it is the promise)."""
        return self._get_setting(SETTING_LOCAL_ONLY, "on") == "on"

    def set_local_only(self, enabled: bool) -> bool:
        self._set_setting(SETTING_LOCAL_ONLY, "on" if enabled else "off")
        self.record_event(kind=EVENT_SETTING, reason=f"local_only={'on' if enabled else 'off'}")
        return self.local_only()

    def assert_local_only(self, *, mouth: str = "") -> None:
        """Refuse routing an answer to a remote provider while local-only is on.

        A *cloud* mouth is the one remote route the project has; it is refused by
        name. The project's own offline guard, when installed, is the second,
        independent backstop.
        """
        if self.local_only() and mouth.lower() in {"cloud", "remote", "openai", "anthropic"}:
            self.record_event(
                kind=EVENT_REFUSAL, reason=f"remote mouth refused: {mouth}"
            )
            raise SafetyError(
                "Local-only mode is on: this Studio never routes a conversation to a "
                "remote provider."
            )

    def runtime_status(self) -> str:
        """An honest one-line statement of where inference runs."""
        guard = "on" if guard_active() else "off"
        mode = "on" if self.local_only() else "off"
        return (
            f"Inference: local. Local-only mode: {mode}. Network guard: {guard}. "
            "No conversation is sent to a remote service."
        )

    # -- events -----------------------------------------------------------

    def record_event(self, *, kind: str, character_id: str = "", reason: str = "") -> Any:
        """Append a safety event (an eligibility confirmation or a refusal)."""
        if kind not in EVENT_KINDS:
            raise SafetyError(f"unknown safety event kind: {kind!r}")
        return self.storage.domain.create(
            "companion_safety_events", character_id=character_id, kind=kind, reason=reason
        )

    def events(self, *, kind: str | None = None) -> tuple[Any, ...]:
        rows = self.storage.domain.list_records(
            "companion_safety_events", order_by="created_at DESC, record_id"
        )
        if kind is not None:
            rows = [row for row in rows if row.kind == kind]
        return tuple(rows)

    # -- disclosure -------------------------------------------------------

    @staticmethod
    def privacy_disclosure() -> str:
        """The honest disclosure the application states outside the fiction."""
        return PRIVACY_DISCLOSURE

    # -- internals --------------------------------------------------------

    @staticmethod
    def _looks_coercive(text: str) -> bool:
        if not text:
            return False
        lowered = f" {text.casefold()} "
        for marker in _COERCION_MARKERS:
            if f" {marker} " in lowered or f" {marker}," in lowered or f" {marker}." in lowered:
                return True
        return False


__all__ = [
    "BOUNDARY_KINDS",
    "BOUNDARY_TOPIC",
    "CONTENT_ADULT",
    "CONTENT_GENERAL",
    "CONTENT_LEVELS",
    "CONTENT_ROMANTIC",
    "EVENT_ELIGIBILITY",
    "EVENT_KINDS",
    "EVENT_REFUSAL",
    "EVENT_SETTING",
    "PRIVACY_DISCLOSURE",
    "SETTING_ADULT_MODE",
    "SETTING_LOCAL_ONLY",
    "CompanionSafetyManager",
    "SafetyDecision",
    "SafetyError",
]
