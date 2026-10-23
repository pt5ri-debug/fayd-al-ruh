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
from companion.language import detect_language_of, resolve_reply_language
from companion.memory_manager import Memory, MemoryManager
from companion.model import (
    BACKEND_AUTO,
    BACKEND_GGUF,
    BACKEND_OLLAMA,
    CATALOG,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SPEC_KEY,
    CompanionModelSpec,
    CompanionModelUnavailable,
    LlamaCppNatashaClient,
    NatashaModelClient,
    NatashaModelStatus,
    OllamaNatashaClient,
    build_client,
    download_model,
    natasha_model_status,
    pull_model,
    resolve_backend,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from companion.personality import (
    LANGUAGE_MODES,
    NAME,
    NatashaPersonality,
    PersonalitySettings,
    default_personality,
)
from companion.polish import PolishResult, polish, polish_report
from companion.prompt_builder import PromptBuilder
from companion.relationship_manager import RelationshipEvent, RelationshipManager

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_GGUF",
    "BACKEND_OLLAMA",
    "CATALOG",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SPEC_KEY",
    "LANGUAGE_MODES",
    "NAME",
    "CompanionAgent",
    "CompanionConfig",
    "CompanionError",
    "CompanionModelSpec",
    "CompanionModelUnavailable",
    "CompanionReply",
    "ConversationManager",
    "DialogueEmotion",
    "DialogueEmotionTracker",
    "LlamaCppNatashaClient",
    "Memory",
    "MemoryManager",
    "ModelCaller",
    "ModelStatus",
    "NatashaModelClient",
    "NatashaModelStatus",
    "NatashaPersonality",
    "OllamaNatashaClient",
    "PersonalitySettings",
    "PolishResult",
    "PromptBuilder",
    "RelationshipEvent",
    "RelationshipManager",
    "SessionInfo",
    "build_client",
    "default_db_path",
    "default_personality",
    "detect_language_of",
    "download_model",
    "natasha_model_status",
    "polish",
    "polish_report",
    "pull_model",
    "resolve_backend",
    "resolve_reply_language",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
