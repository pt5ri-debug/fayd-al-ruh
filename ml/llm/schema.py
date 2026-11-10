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

from ml.llm.tokenizer import BOS_ID, ByteTokenizer, sample
from ml.llm.transformer import KVCache, TransformerLM

_STRING_STOP = frozenset({'"', "\n", "\r", "\t"})
_NUMBERS = ("0.4", "0.5", "0.6", "0.7", "0.8", "0.9")
_BOOLEANS = ("true", "false")


class StructuredDecoder:
    """Fills the schema's leaf values from the model, keeping the JSON valid.

    Works with any tokenizer exposing ``encode``/``decode``: the running buffer
    is text, re-encoded before each step, so a multi-byte UTF-8 character is
    never split and the same code path serves the byte and BPE tokenizers.

    The forward pass is cached: the skeleton only ever *appends* text, so when the
    new encoding begins with the previous token ids (always true for the byte
    tokenizer, and the common case for BPE) only the new tokens are run through
    the model. When the encoding is not a simple extension -- a merge rewrote the
    boundary -- the cache is rebuilt from scratch, so the logits are exactly the
    ones a full recompute would give.
    """

    def __init__(self, model: TransformerLM, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()
        self._cache = KVCache()
        self._tokens: list[int] = []

    def _logits(self, prefix: str) -> np.ndarray:
        tokens = self.tokenizer.encode(prefix, bos=False)
        block = self.model.config.block_size
        if len(tokens) > block:
            # The window slides, which the append-only cache cannot express.
            self._cache = KVCache()
            self._tokens = []
            return self.model.forward(np.array(tokens[-block:], dtype=np.int64))[0, -1]
        reuse = (
            self._tokens
            and len(tokens) > len(self._tokens)
            and tokens[: len(self._tokens)] == self._tokens
        )
        if reuse:
            new = tokens[len(self._tokens) :]
            logits = self.model.forward_incremental(np.array(new, dtype=np.int64), self._cache)
            self._tokens = tokens
            return logits[0, -1]
        self._cache = KVCache()
        self._tokens = tokens
        return self.model.forward_incremental(np.array(tokens, dtype=np.int64), self._cache)[0, -1]

    def _string(self, prefix: str, rng, temperature: float, max_tokens: int) -> str:
        """Sample a string leaf; stops at a closing quote or control character."""
        raw = prefix
        for _ in range(max_tokens):
            token = sample(self._logits(raw), temperature, rng)
            if token == BOS_ID:
                break
            text = self.tokenizer.decode([token])
            if not text or any(character in _STRING_STOP for character in text):
                break
            raw += text
        return raw[len(prefix):].strip()

    def _choice(self, prefix: str, options: tuple[str, ...], rng) -> str:
        """Pick one literal, restricted to the options the schema allows."""
        probs = self._logits(prefix)
        probs = np.exp(probs - probs.max())
        probs /= probs.sum()
        scores = np.array(
            [probs[self.tokenizer.encode(option, bos=False)[0]] for option in options]
        )
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
        buf = prompt.rstrip() + '\n{"concepts": [{"label": "'
        label = self._string(buf, rng, temperature, max_string_bytes)

        buf += label + '", "aliases": []}], "evidence": [{"description": "'
        description = self._string(buf, rng, temperature, max_string_bytes)

        buf += description + '", "source": "'
        source = self._string(buf, rng, temperature, max_string_bytes)

        buf += source + '", "supports": '
        supports = self._choice(buf, _BOOLEANS, rng)

        buf += supports + ', "strength": '
        strength = self._choice(buf, _NUMBERS, rng)

        buf += strength + '}], "hypotheses": [{"statement": "'
        statement = self._string(buf, rng, temperature, max_string_bytes)

        buf += statement + '", "confidence": '
        confidence = self._choice(buf, _NUMBERS, rng)

        buf += confidence + '}], "questions": [{"text": "'
        question = self._string(buf, rng, temperature, max_string_bytes)

        buf += question + '", "topic": "'
        topic = self._string(buf, rng, temperature, max_string_bytes)

        buf += topic + '", "priority": '
        priority = self._choice(buf, _NUMBERS, rng)

        buf += priority + '}], "plans": [{"goal": "'
        goal = self._string(buf, rng, temperature, max_string_bytes)

        buf += goal + '", "priority": '
        plan_priority = self._choice(buf, _NUMBERS, rng)

        buf += plan_priority + ', "steps": [{"description": "'
        first_step = self._string(buf, rng, temperature, max_string_bytes)

        buf += first_step + '"}, {"description": "'
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
