"""The roleplay engine: fictional dating and multi-character scenarios.

A scenario is a small, declarative bundle -- a title, a category, a setting, the
participating characters and their roles, the starting situation, the objectives,
the narrative style, the allowed content level, and the boundaries -- and a
*session* is one running play of it, which the user can pause, resume, branch,
save, and restart.

Two rules run through the module:

* **Everything is fictional unless the user says otherwise.** A scenario's
  location is fictional by default (``is_real_place`` must be set explicitly), and
  a session that has been branched from another records its parent, so a story's
  history is never rewritten -- a branch is a new session that points back at the
  one it forked from.
* **The cast is explicit and honest.** Each participant has its own role and a
  stable order, and the engine refuses to add a character twice to the same
  scenario, so the model can never be told two characters are one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.romance.characters import CharacterManager
from core.romance.conversation import ConversationManager
from core.romance.storage import CompanionStorageService

#: The scenario categories.
CATEGORY_FIRST_MEETING = "first_meeting"
CATEGORY_COFFEE_SHOP = "coffee_shop"
CATEGORY_VIRTUAL_DATE = "virtual_date"
CATEGORY_ROMANTIC_EVENING = "romantic_evening"
CATEGORY_TRAVEL = "travel_story"
CATEGORY_DAILY_LIFE = "daily_life"
CATEGORY_FRIENDSHIP = "friendship"
CATEGORY_EMOTIONAL = "emotional_conversation"
CATEGORY_ADVENTURE = "adventure"
CATEGORY_MULTI = "multi_character"
CATEGORIES: tuple[str, ...] = (
    CATEGORY_FIRST_MEETING,
    CATEGORY_COFFEE_SHOP,
    CATEGORY_VIRTUAL_DATE,
    CATEGORY_ROMANTIC_EVENING,
    CATEGORY_TRAVEL,
    CATEGORY_DAILY_LIFE,
    CATEGORY_FRIENDSHIP,
    CATEGORY_EMOTIONAL,
    CATEGORY_ADVENTURE,
    CATEGORY_MULTI,
)

#: The readable category labels.
CATEGORY_LABELS: dict[str, str] = {
    CATEGORY_FIRST_MEETING: "First meeting",
    CATEGORY_COFFEE_SHOP: "Coffee shop meeting",
    CATEGORY_VIRTUAL_DATE: "Virtual date",
    CATEGORY_ROMANTIC_EVENING: "Romantic evening",
    CATEGORY_TRAVEL: "Travel story",
    CATEGORY_DAILY_LIFE: "Daily-life interactions",
    CATEGORY_FRIENDSHIP: "Friendship development",
    CATEGORY_EMOTIONAL: "Emotional conversations",
    CATEGORY_ADVENTURE: "Fictional adventures",
    CATEGORY_MULTI: "Multi-character stories",
}

#: The allowed content levels (``general`` unless the user raises it).
CONTENT_GENERAL = "general"
CONTENT_ROMANTIC = "romantic"
CONTENT_ADULT = "adult"
CONTENT_LEVELS: tuple[str, ...] = (CONTENT_GENERAL, CONTENT_ROMANTIC, CONTENT_ADULT)

#: The session states.
STATE_ACTIVE = "active"
STATE_PAUSED = "paused"
STATE_ENDED = "ended"
SESSION_STATES: tuple[str, ...] = (STATE_ACTIVE, STATE_PAUSED, STATE_ENDED)


class RoleplayError(RuntimeError):
    """Raised when a scenario or session operation cannot be carried out."""


@dataclass(frozen=True)
class ScenarioSummary:
    """A light listing of one scenario."""

    scenario_id: str
    title: str
    category: str
    category_label: str
    content_level: str
    participant_count: int
    session_count: int


class RoleplayEngine:
    """Create and manage scenarios and their running sessions."""

    def __init__(
        self,
        storage: CompanionStorageService,
        *,
        characters: CharacterManager | None = None,
        conversations: ConversationManager | None = None,
    ) -> None:
        self.storage = storage
        self.characters = characters if characters is not None else CharacterManager(storage)
        self.conversations = (
            conversations if conversations is not None else ConversationManager(storage)
        )

    # -- scenarios --------------------------------------------------------

    def create_scenario(
        self,
        title: str,
        *,
        category: str = CATEGORY_FIRST_MEETING,
        description: str = "",
        location: str = "",
        setting_time: str = "",
        background: str = "",
        starting_situation: str = "",
        objectives: tuple[str, ...] = (),
        narrative_style: str = "",
        content_level: str = CONTENT_GENERAL,
        boundaries: tuple[str, ...] = (),
        is_real_place: bool = False,
    ) -> Any:
        """Create a fictional scenario. The location is fictional unless flagged."""
        if not (title or "").strip():
            raise RoleplayError("a scenario needs a title")
        if category not in CATEGORIES:
            raise RoleplayError(f"unknown scenario category: {category!r}")
        if content_level not in CONTENT_LEVELS:
            raise RoleplayError(f"unknown content level: {content_level!r}")
        return self.storage.domain.create(
            "companion_scenarios",
            title=title.strip(),
            category=category,
            description=description,
            location=location,
            setting_time=setting_time,
            background=background,
            starting_situation=starting_situation,
            objectives=tuple(objectives),
            narrative_style=narrative_style,
            content_level=content_level,
            boundaries=tuple(boundaries),
            is_real_place=bool(is_real_place),
        )

    def get_scenario(self, scenario_id: str) -> Any:
        scenario = self.storage.domain.get("companion_scenarios", scenario_id)
        if scenario is None:
            raise RoleplayError(f"no scenario {scenario_id!r}")
        return scenario

    def scenarios(self, *, category: str | None = None) -> tuple[Any, ...]:
        rows = self.storage.domain.list_records(
            "companion_scenarios", order_by="created_at DESC, record_id"
        )
        if category is not None:
            rows = [row for row in rows if row.category == category]
        return tuple(rows)

    def summaries(self, *, category: str | None = None) -> tuple[ScenarioSummary, ...]:
        result = []
        for scenario in self.scenarios(category=category):
            result.append(
                ScenarioSummary(
                    scenario_id=scenario.record_id,
                    title=scenario.title,
                    category=scenario.category,
                    category_label=CATEGORY_LABELS.get(scenario.category, scenario.category),
                    content_level=scenario.content_level,
                    participant_count=len(self.participants(scenario.record_id)),
                    session_count=len(self.sessions(scenario.record_id)),
                )
            )
        return tuple(result)

    # -- participants -----------------------------------------------------

    def add_participant(self, scenario_id: str, character_id: str, *, role: str = "") -> Any:
        """Add a character to a scenario's cast (never twice)."""
        self.get_scenario(scenario_id)
        if self.storage.domain.get("companion_characters", character_id) is None:
            raise RoleplayError(f"no character {character_id!r}")
        for existing in self.participants(scenario_id):
            if existing.character_id == character_id:
                raise RoleplayError(f"character {character_id!r} is already in the scenario")
        order = len(self.participants(scenario_id))
        return self.storage.domain.create(
            "scenario_participants",
            scenario_id=scenario_id,
            character_id=character_id,
            role=role,
            order=order,
        )

    def participants(self, scenario_id: str) -> tuple[Any, ...]:
        rows = self.storage.domain.list_records(
            "scenario_participants",
            where="scenario_id = ?",
            params=(scenario_id,),
            order_by="sort_order, record_id",
        )
        return tuple(rows)

    def remove_participant(self, participant_id: str) -> Any:
        return self.storage.domain.deactivate("scenario_participants", participant_id)

    def cast(self, scenario_id: str) -> tuple[str, ...]:
        """The character ids in the scenario, in order."""
        return tuple(
            participant.character_id
            for participant in self.participants(scenario_id)
            if participant.active
        )

    # -- sessions ---------------------------------------------------------

    def start(
        self,
        scenario_id: str,
        *,
        character_id: str = "",
        conversation_id: str = "",
    ) -> Any:
        """Start a running session for a scenario.

        A conversation is opened for the session (a dating/roleplay scene is its
        own conversation) unless one is supplied, so the scene has somewhere to
        append its turns.
        """
        scenario = self.get_scenario(scenario_id)
        if not conversation_id:
            anchor = character_id or (self.cast(scenario_id)[0] if self.cast(scenario_id) else "")
            if not anchor:
                raise RoleplayError("a scenario needs a character to start a session")
            conversation = self.conversations.start(
                anchor,
                title=scenario.title,
                mode="dating_simulation"
                if scenario.category != CATEGORY_MULTI
                else "multi_character_roleplay",
                scenario_id=scenario_id,
            )
            conversation_id = conversation.record_id
        return self.storage.domain.create(
            "scenario_sessions",
            scenario_id=scenario_id,
            conversation_id=conversation_id,
            state=STATE_ACTIVE,
        )

    def sessions(self, scenario_id: str) -> tuple[Any, ...]:
        return tuple(
            self.storage.domain.list_records(
                "scenario_sessions",
                where="scenario_id = ?",
                params=(scenario_id,),
                order_by="created_at, record_id",
            )
        )

    def get_session(self, session_id: str) -> Any:
        session = self.storage.domain.get("scenario_sessions", session_id)
        if session is None:
            raise RoleplayError(f"no session {session_id!r}")
        return session

    def _set_state(self, session_id: str, state: str) -> Any:
        if state not in SESSION_STATES:
            raise RoleplayError(f"unknown session state: {state!r}")
        return self.storage.domain.update("scenario_sessions", session_id, state=state)

    def pause(self, session_id: str) -> Any:
        return self._set_state(session_id, STATE_PAUSED)

    def resume(self, session_id: str) -> Any:
        return self._set_state(session_id, STATE_ACTIVE)

    def end(self, session_id: str) -> Any:
        return self._set_state(session_id, STATE_ENDED)

    def branch(self, session_id: str) -> Any:
        """Fork a session: a **new** session that points back at the one it forked.

        The original is untouched, so a story's history is never rewritten.
        """
        original = self.get_session(session_id)
        return self.storage.domain.create(
            "scenario_sessions",
            scenario_id=original.scenario_id,
            conversation_id=original.conversation_id,
            state=STATE_ACTIVE,
            branch_of=session_id,
        )

    def restart(self, session_id: str) -> Any:
        """Restart a scene: a fresh session for the same scenario and cast."""
        original = self.get_session(session_id)
        cast = self.cast(original.scenario_id)
        return self.start(
            original.scenario_id,
            character_id=cast[0] if cast else "",
        )


__all__ = [
    "CATEGORIES",
    "CATEGORY_ADVENTURE",
    "CATEGORY_COFFEE_SHOP",
    "CATEGORY_DAILY_LIFE",
    "CATEGORY_EMOTIONAL",
    "CATEGORY_FIRST_MEETING",
    "CATEGORY_FRIENDSHIP",
    "CATEGORY_LABELS",
    "CATEGORY_MULTI",
    "CATEGORY_ROMANTIC_EVENING",
    "CATEGORY_TRAVEL",
    "CATEGORY_VIRTUAL_DATE",
    "CONTENT_ADULT",
    "CONTENT_GENERAL",
    "CONTENT_LEVELS",
    "CONTENT_ROMANTIC",
    "SESSION_STATES",
    "STATE_ACTIVE",
    "STATE_ENDED",
    "STATE_PAUSED",
    "RoleplayEngine",
    "RoleplayError",
    "ScenarioSummary",
]
