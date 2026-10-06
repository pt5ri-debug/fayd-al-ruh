"""The in-project language model: a small transformer trained inside the repo.

Nothing here talks to an external server. :class:`LocalLanguageModel` adapts the
local transformer to the project's :class:`core.model_interface.CognitiveModel`
interface, so it can replace the Ollama client everywhere a model is accepted.
"""

from __future__ import annotations

from pathlib import Path

from core.model_interface import CallableModel, ModelRequest, ModelResponse
from core.paths import default_checkpoint
from ml.llm.generator import LocalGenerator
from ml.llm.schema import StructuredDecoder
from ml.llm.tokenizer import load_tokenizer
from ml.llm.transformer import TransformerLM

DEFAULT_CHECKPOINT = default_checkpoint()


class LocalLanguageModel(CallableModel):
    """Serves the in-project transformer as a ``CognitiveModel``.

    The model only ever receives a rendered prompt string; it cannot touch the
    knowledge graph, mutate beliefs, or run tools. When a ``.tokenizer.json``
    sits beside the checkpoint (written by the trainer), it is loaded so the
    tokenizer always matches the trained vocabulary.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        tokenizer: str | Path | None = None,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        seed: int | None = None,
        max_string_bytes: int = 32,
    ) -> None:
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if not path.exists():
            raise FileNotFoundError(
                f"No local language model at {path}. Train one with "
                "`python -m ml.llm` (see docs)."
            )
        self.model = TransformerLM.load(path)
        tokenizer_path = Path(tokenizer) if tokenizer is not None else path.with_suffix(".tokenizer.json")
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.generator = LocalGenerator(self.model, tokenizer=self.tokenizer)
        self.decoder = StructuredDecoder(self.model, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.max_string_bytes = max_string_bytes
        super().__init__(f"fayd-local-llm:{path.stem}", self._generate_prompt)

    def _generate_prompt(self, prompt: str) -> str:
        full = self.generator.generate(
            prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )
        # The generator returns prompt + continuation; a model response is the
        # continuation only, so callers never see their own prompt echoed back.
        return full[len(prompt):]

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.response_schema is not None:
            text = self.decoder.decode(
                self.render_prompt(request),
                seed=self.seed,
                temperature=self.temperature,
                max_string_bytes=self.max_string_bytes,
            ).strip()
            if not text:
                raise ValueError("Model returned an empty response.")
            return ModelResponse(text=text, confidence=0.5, model_name=self.model_name)
        return super().generate(request)


__all__ = ["DEFAULT_CHECKPOINT", "LocalLanguageModel"]
