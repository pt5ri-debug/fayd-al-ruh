"""Natasha -- Fayd Al-Ruh's conversational companion.

A self-contained layer that gives the project a companion character to talk with
through a local Qwen3 model: a stable personality, durable memory, independent
persistent sessions, a simulated dialogue tone, and an evolving (honestly
narrated) relationship story.

It is *not* the cognitive core. Nothing here writes to entities, relations,
evidence, beliefs, or the knowledge graph; the companion keeps its own SQLite
database, separate from the project's knowledge, so adding Natasha cannot disturb
the symbolic architecture the project is built around.

The public surface is small on purpose::

    from companion import CompanionAgent, CompanionConfig

    agent = CompanionAgent(CompanionConfig())
    session = agent.new_session("first chat")
    reply = agent.ask(session.id, "مرحبا يا ناتاشا")
    print(reply.text)
"""

from __future__ import annotations

from companion.companion_agent import (
    CompanionAgent,
    CompanionError,
    CompanionReply,
    ModelCaller,
    ModelStatus,
)
from companion.config import CompanionConfig, default_db_path
from companion.conversation_manager import ConversationManager, SessionInfo
from companion.emotion_state import DialogueEmotion, DialogueEmotionTracker
from companion.memory_manager import Memory, MemoryManager
from companion.personality import NAME, PersonalitySettings, NatashaPersonality, default_personality
from companion.prompt_builder import PromptBuilder
from companion.relationship_manager import RelationshipEvent, RelationshipManager

__all__ = [
    "CompanionAgent",
    "CompanionConfig",
    "CompanionError",
    "CompanionReply",
    "ConversationManager",
    "DialogueEmotion",
    "DialogueEmotionTracker",
    "Memory",
    "MemoryManager",
    "ModelCaller",
    "ModelStatus",
    "NAME",
    "NatashaPersonality",
    "PersonalitySettings",
    "PromptBuilder",
    "RelationshipEvent",
    "RelationshipManager",
    "SessionInfo",
    "default_db_path",
    "default_personality",
]
