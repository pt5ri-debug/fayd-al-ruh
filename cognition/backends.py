"""Concrete backends: the project's own model (student) and a local one (teacher).

These are the two real :class:`~cognition.teacher_student.ModelBackend`s the
system uses, built on what the project already has:

* :class:`FaydStudentBackend` -- the project's **own** trained transformer
  (``ml.llm``), loaded through :class:`ai.local_model.LocalLanguageModel`. This is
  the primary learner.
* :class:`LocalTeacherBackend` -- a real local model for teaching: a
  ``llama.cpp`` GGUF (Ultron / a shared model) or a reachable Ollama model. It is
  the *teacher*, not a replacement for the student.

Both are lazy: nothing is loaded until the backend is first asked to answer, and a
missing model makes :meth:`available` honestly ``False`` so the caller reports the
real next step instead of failing.

The task's rule is kept here: **the student is the learner and the owner of the
cognitive memory; the teacher is an assistant that suggests examples.** The
teacher backend never writes anywhere -- it only returns text.
"""

from __future__ import annotations

from pathlib import Path

from core.paths import default_checkpoint

from cognition.teacher_student import ModelBackend


class FaydStudentBackend:
    """The project's own trained model, as a :class:`ModelBackend`."""

    name = "fayd-al-ruh"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint is not None else default_checkpoint()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._seed = seed
        self._model = None

    def available(self) -> bool:
        return self._checkpoint.exists()

    def checkpoint(self) -> Path:
        return self._checkpoint

    def answer(self, question: str) -> str:
        model = self._load()
        if model is None:
            return ""
        from ai.mind import render_prompt

        response = model.generate(
            _request(render_prompt(question))
        )
        return response.text.strip()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.available():
            return None
        try:
            from ai.local_model import LocalLanguageModel

            self._model = LocalLanguageModel(
                self._checkpoint,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                seed=self._seed,
            )
        except Exception:
            self._model = None
        return self._model

    def describe(self) -> str:
        state = "ready" if self.available() else f"no checkpoint at {self._checkpoint}"
        return f"Fayd Al-Ruh student ({state})"


def _request(prompt: str):
    from core.model_interface import ModelRequest, MultimodalInput

    return ModelRequest(inputs=(MultimodalInput(modality="text", value=prompt),))


class LocalTeacherBackend:
    """A real local model used as the teacher, through the project's own engine.

    Tries, in order: a ``llama.cpp`` GGUF the project ships for a mouth, then a
    reachable Ollama model. Either way the answer is plain text and nothing is
    written; the teacher only proposes.
    """

    name = "teacher"

    def __init__(self, *, model_key: str | None = None, max_new_tokens: int = 128) -> None:
        self._model_key = model_key
        self._max_new_tokens = max_new_tokens
        self._engine = None
        self._ollama = None

    def available(self) -> bool:
        return self._load_engine() is not None or self._load_ollama() is not None

    def answer(self, question: str) -> str:
        engine = self._load_engine()
        if engine is not None:
            try:
                return str(engine.reply(question)).strip()
            except Exception:
                pass
        ollama = self._load_ollama()
        if ollama is not None:
            try:
                reply = ollama.chat([{"role": "user", "content": question}])
                return str(reply).strip()
            except Exception:
                pass
        return ""

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from core.ultron import UltronEngine

            engine = UltronEngine()
            if engine.is_available():
                self._engine = engine
        except Exception:
            self._engine = None
        return self._engine

    def _load_ollama(self):
        if self._ollama is not None:
            return self._ollama
        try:
            from core.coder_model import OllamaCoderClient, coder_status

            status = coder_status()
            if getattr(status, "server_reachable", False):
                self._ollama = OllamaCoderClient()
        except Exception:
            self._ollama = None
        return self._ollama

    def describe(self) -> str:
        if self._load_engine() is not None:
            return "teacher: a local llama.cpp model"
        if self._load_ollama() is not None:
            return "teacher: a local Ollama model"
        return "teacher: no local model reachable"


def build_backends() -> tuple[ModelBackend, ModelBackend | None]:
    """The default student and teacher backends for this machine."""
    student = FaydStudentBackend()
    teacher = LocalTeacherBackend()
    return student, (teacher if teacher.available() else None)


__all__ = [
    "FaydStudentBackend",
    "LocalTeacherBackend",
    "build_backends",
]
