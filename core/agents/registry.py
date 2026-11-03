"""The Personality Registry: one addressable agent record per personality.

Every surface that talks to a character needs the same facts about it -- its
stable id, its name, its role and specialisation, its communication and reasoning
preferences, the capabilities it holds, the scope of its memory and learning, and
the model it prefers. Rather than each surface re-deriving those from a persona, a
configuration file, and an isolation policy, this registry assembles them **once**
into a :class:`PersonalityProfile` and hands it out by id.

It is a **view**, never a second source of truth:

* identity (owner, mouth, name, role, private) comes from
  :mod:`core.character_identity`;
* the editable configuration (style, model, voice, learning) comes from
  :mod:`core.character_registry` / :mod:`core.character_config`;
* the capability set comes from :mod:`core.agents.capabilities`;
* the memory and learning scope come from :mod:`core.learning_isolation` and
  :mod:`core.agents.memory_access` / :mod:`core.agents.learning_scope`.

So editing a character in the Characters window is visible here at once, and the
registry can never disagree with the window about who a character is.

A stable id is either vocabulary (``sources`` or ``jarvis``), resolved through
:func:`core.learning_isolation.owner_for`, so a caller never has to remember which
one it holds.

Dependency-free and Tk-free: it loads no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agents.capabilities import (
    AgentCapability,
    CapabilitySet,
    effective_capabilities,
)
from core.character_config import CharacterConfig
from core.character_identity import (
    IDENTITY_OWNERS,
    CharacterIdentity,
    identity_for,
)
from core.character_learning import persona_for
from core.learning_isolation import (
    CHARACTER_OWNERS,
    MAIN_MODEL_OWNER,
    is_owner,
    is_private,
    mouth_key,
    owner_for,
)

#: The version of the personality *record* shape (not the configuration's own
#: version, which lives in :class:`core.character_config.CharacterConfig`). Bumped
#: when the profile gains a field, so a report can say what shape it read.
PROFILE_VERSION = 1


@dataclass(frozen=True)
class PersonalityProfile:
    """One personality as an addressable agent -- the registry's unit.

    The fields answer, in order, the questions the task's "Personality Identity
    and Configuration" section asks: a stable id, a name, a description, a role
    and specialisation, communication style, reasoning and task preferences,
    allowed capabilities, memory and learning scope, model preference, and the
    configuration version.
    """

    #: The stable id (the owner/directory name, e.g. ``jarvis``).
    personality_id: str
    #: The chat mouth key (e.g. ``sources``), so a caller can address the mouth.
    mouth: str
    name: str
    description: str
    role: str
    specialization: str
    #: How it speaks (formality, length, language) -- rendered from the config.
    communication_style: dict[str, Any]
    #: How it reasons (its reasoning-style preference).
    reasoning_preferences: dict[str, Any]
    #: What tasks it prefers (its objectives, in its own words).
    task_preferences: tuple[str, ...]
    #: What it may do -- the effective, ceiling-bounded capability set.
    capabilities: CapabilitySet
    #: Whose memory it may read (its own first) -- the isolation policy's answer.
    memory_scope: tuple[str, ...]
    #: Whose learning it may read and whether it may contribute to training.
    learning_scope: tuple[str, ...]
    contributes_training: bool
    #: The model it prefers, if it names one (else the mouth's own default).
    model_preference: str
    #: The voice it prefers, if it names one.
    voice_preference: str
    private: bool
    #: The configuration version this profile was read from.
    config_version: int = 1
    #: The profile shape version.
    profile_version: int = PROFILE_VERSION

    def holds(self, capability: AgentCapability) -> bool:
        """True when this personality's capability set grants ``capability``."""
        return self.capabilities.allows(capability)

    def describe(self) -> str:
        role = self.specialization or self.role or "(no stated specialisation)"
        privacy = ", private records" if self.private else ""
        return "\n".join(
            [
                f"{self.name} ({self.personality_id} / {self.mouth}){privacy}",
                f"  role       : {self.role or '(none)'}",
                f"  specialises: {role}",
                f"  model      : {self.model_preference or '(mouth default)'}",
                f"  capabilities: {self.capabilities.describe()}",
                f"  memory scope : {', '.join(self.memory_scope)}",
                f"  learning scope: {', '.join(self.learning_scope)}",
            ]
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "personality_id": self.personality_id,
            "mouth": self.mouth,
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "specialization": self.specialization,
            "communication_style": dict(self.communication_style),
            "reasoning_preferences": dict(self.reasoning_preferences),
            "task_preferences": list(self.task_preferences),
            "capabilities": self.capabilities.to_primitive(),
            "memory_scope": list(self.memory_scope),
            "learning_scope": list(self.learning_scope),
            "contributes_training": self.contributes_training,
            "model_preference": self.model_preference,
            "voice_preference": self.voice_preference,
            "private": self.private,
            "config_version": self.config_version,
            "profile_version": self.profile_version,
        }


def _communication_style(config: CharacterConfig) -> dict[str, Any]:
    """How the personality speaks, read from its editable configuration."""
    personality = config.personality
    return {
        "formality": getattr(personality, "formality", ""),
        "response_length": getattr(personality, "response_length", ""),
        "communication_style": getattr(personality, "communication_style", ""),
        "emotional_expression": getattr(personality, "emotional_expression", ""),
        "preferred_language": getattr(personality, "preferred_language", ""),
    }


def _reasoning_preferences(config: CharacterConfig) -> dict[str, Any]:
    """How the personality reasons, read from its editable configuration."""
    personality = config.personality
    return {
        "reasoning_style": getattr(personality, "reasoning_style", ""),
        "knowledge_domain": getattr(personality, "knowledge_domain", ""),
    }


def _task_preferences(config: CharacterConfig) -> tuple[str, ...]:
    """The objectives the personality names as its own."""
    objectives = getattr(config.personality, "objectives", ())
    return tuple(str(item) for item in objectives)


def _memory_scope(owner: str) -> tuple[str, ...]:
    """Whose memory ``owner`` may read, its own first (the policy's answer)."""
    from core.learning_isolation import decide

    readable = [owner]
    for other in IDENTITY_OWNERS:
        if other == owner:
            continue
        if decide(owner, other).allowed:
            readable.append(other)
    return tuple(readable)


def _learning_scope(owner: str) -> tuple[str, ...]:
    """Whose *learning* ``owner`` may read (same policy as memory, kept explicit)."""
    return _memory_scope(owner)


@dataclass
class PersonalityRegistry:
    """The one place a personality is resolved by its stable id.

    ``registry`` is the character-configuration registry (injectable, so a test
    keeps its configuration under a temp folder); ``configs`` lets a caller inject
    a fixed mapping of configurations instead of reading files, so the registry is
    exercised with no disk.
    """

    registry: Any | None = None
    configs: dict[str, CharacterConfig] | None = None
    _cache: dict[str, PersonalityProfile] = field(default_factory=dict, repr=False)

    def _config_registry(self) -> Any:
        if self.registry is None:
            from core.character_registry import default_registry

            self.registry = default_registry()
        return self.registry

    def _config_for(self, owner: str) -> CharacterConfig:
        if self.configs is not None and owner in self.configs:
            return self.configs[owner]
        return self._config_registry().load(owner)  # type: ignore[no-any-return]

    def ids(self) -> tuple[str, ...]:
        """Every personality id this registry manages, in the project's order."""
        return IDENTITY_OWNERS

    def character_ids(self) -> tuple[str, ...]:
        """The seven characters' ids (not the project's own model)."""
        return CHARACTER_OWNERS

    def resolve(self, key: str) -> PersonalityProfile:
        """The profile for ``key`` (either vocabulary), built and cached.

        An unknown key resolves to the project's own model -- never to a made-up
        personality -- so a stale saved choice is harmless.
        """
        owner = owner_for(key)
        if not is_owner(owner):
            owner = MAIN_MODEL_OWNER
        cached = self._cache.get(owner)
        if cached is not None:
            return cached
        profile = self._build(owner)
        self._cache[owner] = profile
        return profile

    def _build(self, owner: str) -> PersonalityProfile:
        identity: CharacterIdentity = identity_for(owner, registry=self._config_registry())
        config = self._config_for(owner)
        persona = persona_for(identity.mouth)
        capabilities = effective_capabilities(owner)
        private = is_private(owner)
        scope = _memory_scope(owner)
        return PersonalityProfile(
            personality_id=owner,
            mouth=mouth_key(owner),
            name=identity.name,
            description=config.general.description or persona.identity,
            role=identity.role or persona.role,
            specialization=config.general.role or persona.role,
            communication_style=_communication_style(config),
            reasoning_preferences=_reasoning_preferences(config),
            task_preferences=_task_preferences(config),
            capabilities=capabilities,
            memory_scope=scope,
            learning_scope=_learning_scope(owner),
            contributes_training=(
                not private and capabilities.allows(AgentCapability.CONTRIBUTE_TRAINING)
            ),
            model_preference=config.model.key or config.model.model,
            voice_preference=config.voice.voice,
            private=private,
            config_version=config.version,
        )

    def all_profiles(self) -> tuple[PersonalityProfile, ...]:
        """Every personality's profile, in the project's order."""
        return tuple(self.resolve(owner) for owner in self.ids())

    def character_profiles(self) -> tuple[PersonalityProfile, ...]:
        """The seven characters' profiles."""
        return tuple(self.resolve(owner) for owner in self.character_ids())

    def project_model(self) -> PersonalityProfile:
        """The project's own model's profile."""
        return self.resolve(MAIN_MODEL_OWNER)

    def reload(self, key: str | None = None) -> None:
        """Drop the cache for ``key`` (or all), so an APPLY is in force at once."""
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(owner_for(key), None)

    def describe(self) -> str:
        """A one-line-per-personality report."""
        return "\n".join(profile.describe() for profile in self.all_profiles())


#: A lazily-built shared registry, so a caller that just wants the facts does not
#: build one per call. ``default_registry`` is the injectable seam.
_default: PersonalityRegistry | None = None


def default_registry() -> PersonalityRegistry:
    """The shared personality registry (built on first use)."""
    global _default
    if _default is None:
        _default = PersonalityRegistry()
    return _default


def profile_for(key: str, *, registry: PersonalityRegistry | None = None) -> PersonalityProfile:
    """The profile for ``key``, through the shared registry by default."""
    return (registry or default_registry()).resolve(key)


def describe_personalities(*, registry: PersonalityRegistry | None = None) -> str:
    """A report of every personality, for a CLI or a document."""
    return (registry or default_registry()).describe()


__all__ = [
    "PROFILE_VERSION",
    "PersonalityProfile",
    "PersonalityRegistry",
    "default_registry",
    "describe_personalities",
    "profile_for",
]
