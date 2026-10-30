"""Hakeem -- Fayd Al-Ruh's scholar of religion and history.

Hakeem (Arabic ``حكيم``, "the wise one") is a self-contained character layer that
gives the project a scholar to talk with: he discusses the Abrahamic faiths --
Islam, Christianity, and Judaism -- compares them fairly, and is at home in
history and the ancient civilisations. He has his **own** local model, in his own
folder (see :mod:`hakeem.model`), separate from the companion Natasha's -- he is a
character in his own right, on his own model.

Like the companion, it is *not* the cognitive core. Nothing here writes to
entities, relations, evidence, beliefs, or the knowledge graph; Hakeem keeps his
own SQLite database (a separate file from the companion's), so adding him cannot
disturb the symbolic architecture the project is built around.

The public surface is small on purpose::

    from hakeem import HakeemAgent, HakeemConfig

    agent = HakeemAgent(HakeemConfig())
    session = agent.new_session("first chat")
    reply = agent.ask(session.id, "قارن بين مفهوم الله في الأديان الثلاثة")
    print(reply.text)
"""

from __future__ import annotations

from hakeem.config import HakeemConfig, default_db_path
from hakeem.hakeem_agent import (
    HakeemAgent,
    HakeemError,
    HakeemReply,
    ModelCaller,
    ModelStatus,
)
from hakeem.model import (
    BACKEND_AUTO,
    BACKEND_GGUF,
    BACKEND_OLLAMA,
    CATALOG,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SPEC_KEY,
    HakeemModelClient,
    HakeemModelStatus,
    HakeemModelUnavailable,
    download_model,
    hakeem_model_status,
    pull_model,
    resolve_backend,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from hakeem.personality import (
    ARABIC_NAME,
    LANGUAGE_MODES,
    NAME,
    HakeemPersonality,
    PersonalitySettings,
    default_personality,
)
from hakeem.prompt_builder import PromptBuilder

__all__ = [
    "ARABIC_NAME",
    "BACKEND_AUTO",
    "BACKEND_GGUF",
    "BACKEND_OLLAMA",
    "CATALOG",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SPEC_KEY",
    "LANGUAGE_MODES",
    "NAME",
    "HakeemAgent",
    "HakeemConfig",
    "HakeemError",
    "HakeemModelClient",
    "HakeemModelStatus",
    "HakeemModelUnavailable",
    "HakeemPersonality",
    "HakeemReply",
    "ModelCaller",
    "ModelStatus",
    "PersonalitySettings",
    "PromptBuilder",
    "default_db_path",
    "default_personality",
    "download_model",
    "hakeem_model_status",
    "pull_model",
    "resolve_backend",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
