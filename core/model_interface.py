from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class MultimodalInput:
    """A model input part; binary media stays outside the knowledge graph."""
    modality: str
    value: str
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.modality not in {"text", "image", "audio"}:
            raise ValueError("Modality must be text, image, or audio.")
        if not self.value.strip():
            raise ValueError("Model input cannot be empty.")


@dataclass(frozen=True)
class ModelRequest:
    inputs: tuple[MultimodalInput, ...]
    system_instruction: str = "Reason from evidence and state uncertainty."
    context: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelResponse:
    text: str
    confidence: float
    model_name: str
    metadata: dict[str, str] = field(default_factory=dict)


class CognitiveModel(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...


class CallableModel:
    """Adapter for any hosted or local model client supplied by the app."""

    def __init__(self, model_name: str, generate: Callable[[ModelRequest], str]) -> None:
        self.model_name = model_name
        self._generate = generate

    def generate(self, request: ModelRequest) -> ModelResponse:
        text = self._generate(request).strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return ModelResponse(text=text, confidence=0.5, model_name=self.model_name)
