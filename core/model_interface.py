from __future__ import annotations

import inspect
import json
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
    # When set, the model must return JSON conforming to this schema. The
    # controller uses it to force structured proposals rather than free text.
    response_schema: dict | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    confidence: float
    model_name: str
    metadata: dict[str, str] = field(default_factory=dict)


class CognitiveModel(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...


class CallableModel:
    """Adapter for any hosted or local model client supplied by the app.

    ``generate`` may accept either a :class:`ModelRequest` or a plain prompt
    string. The latter keeps simple prompt-style clients (for example the
    Ollama client) usable without a wrapper.
    """

    def __init__(
        self,
        model_name: str,
        generate: Callable[..., str],
        *,
        prompt_builder: Callable[[ModelRequest], str] | None = None,
    ) -> None:
        self.model_name = model_name
        self._generate = generate
        self._prompt_builder = prompt_builder
        self._expects_request = prompt_builder is None and self._accepts_model_request(generate)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._prompt_builder is not None:
            text = self._generate(self._prompt_builder(request))
        elif self._expects_request:
            text = self._generate(request)
        else:
            text = self._generate(self.render_prompt(request))
        text = text.strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return ModelResponse(text=text, confidence=0.5, model_name=self.model_name)

    @staticmethod
    def _accepts_model_request(generate: Callable[..., str]) -> bool:
        """Detect whether ``generate`` takes a ModelRequest or a prompt string."""
        try:
            signature = inspect.signature(generate)
        except (TypeError, ValueError):
            return False
        parameters = list(signature.parameters.values())
        if not parameters:
            return False
        annotation = parameters[0].annotation
        if annotation is ModelRequest or annotation == "ModelRequest":
            return True
        return annotation is inspect.Parameter.empty

    @staticmethod
    def render_prompt(request: ModelRequest) -> str:
        parts = [request.system_instruction]
        if request.response_schema is not None:
            parts.append(
                "Return only JSON matching this schema: "
                + json.dumps(request.response_schema, ensure_ascii=False, sort_keys=True)
            )
        parts.extend(f"Context: {item}" for item in request.context)
        parts.extend(part.value for part in request.inputs)
        return "\n".join(part for part in parts if part.strip())
