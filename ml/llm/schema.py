"""Schema-guided decoding for the in-project language model.

A byte-level model rarely emits valid JSON by chance. This decoder fixes the
JSON structure from the project's ``MODEL_OUTPUT_SCHEMA`` and lets the model
propose only the *content* of each leaf (labels, descriptions, statements,
questions, step descriptions, and confidence values). Because the skeleton is
written by the decoder and the result is serialised with :func:`json.dumps`,
the output is always valid JSON, and it still has to pass
``core.structured_output.parse_structured_output`` afterwards.
"""

from __future__ import annotations

import json

import numpy as np

from ml.llm.tokenizer import BOS_ID, sample
from ml.llm.transformer import TransformerLM

_STRING_STOP = frozenset({ord('"'), ord("\n"), ord("\r"), ord("\t")})
_NUMBERS = ("0.4", "0.5", "0.6", "0.7", "0.8", "0.9")
_BOOLEANS = ("true", "false")


class StructuredDecoder:
    """Fills the schema's leaf values from the model, keeping the JSON valid."""

    def __init__(self, model: TransformerLM) -> None:
        self.model = model

    def _logits(self, prefix: bytes) -> np.ndarray:
        window = prefix[-self.model.config.block_size :]
        tokens = np.frombuffer(window, dtype=np.uint8).astype(np.int64)
        return self.model.forward(tokens)[0, -1]

    def _string(self, prefix: bytes, rng, temperature: float, max_bytes: int) -> str:
        """Sample a string leaf; stops at a closing quote or control character.

        The prefix stays raw bytes, so a multi-byte UTF-8 character is never
        split by decoding: partial sequences keep their context.
        """
        start = len(prefix)
        raw = bytearray(prefix)
        for _ in range(max_bytes):
            token = sample(self._logits(bytes(raw)), temperature, rng)
            if token == BOS_ID or token in _STRING_STOP:
                break
            raw.append(token)
        return bytes(raw[start:]).decode("utf-8", errors="ignore").strip()

    def _choice(self, prefix: bytes, options: tuple[str, ...], rng) -> str:
        """Pick one literal, restricted to the options the schema allows."""
        probs = self._logits(prefix)
        probs = np.exp(probs - probs.max())
        probs /= probs.sum()
        scores = np.array([probs[ord(option[0])] for option in options])
        total = float(scores.sum())
        if total <= 1e-12:
            return options[0]
        return options[int(rng.choice(len(options), p=scores / total))]

    def decode(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float = 0.8,
        max_string_bytes: int = 32,
    ) -> str:
        """Return a JSON object string whose leaves were proposed by the model."""
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        buf = (prompt.rstrip() + '\n{"concepts": [{"label": "').encode("utf-8")
        label = self._string(buf, rng, temperature, max_string_bytes)

        buf += label.encode("utf-8") + b'", "aliases": []}], "evidence": [{"description": "'
        description = self._string(buf, rng, temperature, max_string_bytes)

        buf += description.encode("utf-8") + b'", "source": "'
        source = self._string(buf, rng, temperature, max_string_bytes)

        buf += source.encode("utf-8") + b'", "supports": '
        supports = self._choice(buf, _BOOLEANS, rng)

        buf += supports.encode("utf-8") + b', "strength": '
        strength = self._choice(buf, _NUMBERS, rng)

        buf += strength.encode("utf-8") + b'}], "hypotheses": [{"statement": "'
        statement = self._string(buf, rng, temperature, max_string_bytes)

        buf += statement.encode("utf-8") + b'", "confidence": '
        confidence = self._choice(buf, _NUMBERS, rng)

        buf += confidence.encode("utf-8") + b'}], "questions": [{"text": "'
        question = self._string(buf, rng, temperature, max_string_bytes)

        buf += question.encode("utf-8") + b'", "topic": "'
        topic = self._string(buf, rng, temperature, max_string_bytes)

        buf += topic.encode("utf-8") + b'", "priority": '
        priority = self._choice(buf, _NUMBERS, rng)

        buf += priority.encode("utf-8") + b'}], "plans": [{"goal": "'
        goal = self._string(buf, rng, temperature, max_string_bytes)

        buf += goal.encode("utf-8") + b'", "priority": '
        plan_priority = self._choice(buf, _NUMBERS, rng)

        buf += plan_priority.encode("utf-8") + b', "steps": [{"description": "'
        first_step = self._string(buf, rng, temperature, max_string_bytes)

        buf += first_step.encode("utf-8") + b'"}, {"description": "'
        second_step = self._string(buf, rng, temperature, max_string_bytes)

        steps = [
            {"description": description}
            for description in (first_step, second_step)
            if description
        ]
        proposal: dict[str, object] = {
            "concepts": [{"label": label, "aliases": []}] if label else [],
            "evidence": (
                [
                    {
                        "description": description,
                        "source": source,
                        "supports": supports == "true",
                        "strength": float(strength),
                    }
                ]
                if description and source
                else []
            ),
            "hypotheses": (
                [{"statement": statement, "confidence": float(confidence)}] if statement else []
            ),
            "questions": (
                [{"text": question, "topic": topic, "priority": float(priority)}] if question else []
            ),
            "plans": (
                [{"goal": goal, "priority": float(plan_priority), "steps": steps}] if goal else []
            ),
        }
        return json.dumps(proposal, ensure_ascii=False)


__all__ = ["StructuredDecoder"]
