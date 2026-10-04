from __future__ import annotations

from core.model_interface import CallableModel, ModelRequest, ModelResponse

try:  # pragma: no cover - import path only
    from ai.ollama_client import OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None  # type: ignore[assignment,misc]


class OllamaModel(CallableModel):
    """Adapts the local Ollama client to the CognitiveModel interface.

    The client only receives a rendered prompt; it cannot touch the knowledge
    graph, mutate beliefs, or run tools.
    """

    def __init__(self, client=None, model_name: str | None = None) -> None:
        if client is None:
            if OllamaClient is None:  # pragma: no cover - defensive
                raise RuntimeError("OllamaClient is unavailable.")
            client = OllamaClient()
        super().__init__(str(model_name or getattr(client, "model", "ollama")), client.generate)

    def generate(self, request: ModelRequest) -> ModelResponse:
        return super().generate(request)
