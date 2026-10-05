"""The in-project language model: a small transformer trained inside the repo.

Nothing here talks to an external server. :class:`LocalLanguageModel` adapts the
local transformer to the project's :class:`core.model_interface.CognitiveModel`
interface, so it can replace the Ollama client everywhere a model is accepted.
"""

from __future__ import annotations

from pathlib import Path

from core.model_interface import CallableModel, ModelRequest, ModelResponse
from ml.llm.generator import LocalGenerator
from ml.llm.transformer import TransformerLM

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
)


class LocalLanguageModel(CallableModel):
    """Serves the in-project transformer as a ``CognitiveModel``.

    The model only ever receives a rendered prompt string; it cannot touch the
    knowledge graph, mutate beliefs, or run tools.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
    ) -> None:
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if not path.exists():
            raise FileNotFoundError(
                f"No local language model at {path}. Train one with "
                "`python -m ml.llm` (see docs)."
            )
        self.model = TransformerLM.load(path)
        self.generator = LocalGenerator(self.model)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        super().__init__(f"fayd-local-llm:{path.stem}", self._generate_prompt)

    def _generate_prompt(self, prompt: str) -> str:
        return self.generator.generate(
            prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        return super().generate(request)


__all__ = ["DEFAULT_CHECKPOINT", "LocalLanguageModel"]
