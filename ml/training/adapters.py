"""The LoRA/PEFT adapter layer for Phase 6.

The task's rule is that the base model is **never** modified: a fine-tune produces
an **adapter** that is stored separately, and the base weights are left exactly as
they were. This module is that adapter.

Two backends, one interface:

* :class:`NumpyLoRAAdapter` -- a real low-rank adapter over the project's **own**
  numpy transformer (``ml.llm.transformer``). It freezes every base parameter,
  adds a rank-``r`` update ``B @ A`` to a chosen set of weight matrices, trains
  **only** ``A`` and ``B`` with the project's own backprop, and saves a small
  ``.npz`` that is applied at inference time. It needs **no** optional dependency
  and runs on a CPU, so it is the one method that always works here.
* :class:`PeftAdapterBackend` -- a thin wrapper over the Hugging Face
  ``peft``/``transformers`` path for a **safetensors** base model. It is imported
  lazily and reports honestly when the ``[training]`` extra is absent.

The interface both satisfy is :class:`AdapterBackend`: ``train``, ``save``,
``load``, ``apply``, ``parameter_count``, and ``describe``. The trainer drives the
interface, so a test injects a tiny backend and the whole lifecycle is exercised
with no model and no extra.

**The base is provably untouched**: :meth:`NumpyLoRAAdapter.base_digest` hashes the
base parameters before and after training, and the trainer records both -- an
adapter that changed the base would be a bug the gate catches.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ml.training.compatibility import METHOD_LORA_INPROCESS, METHOD_LORA_TRANSFORMERS

#: The base weight matrices a LoRA adapter may target, by the transformer's own
#: attribute names. The attention projections and the two MLP matrices are the
#: conventional choices; the layer norms and biases are left alone.
TARGET_SUFFIXES: tuple[str, ...] = ("attn_w", "proj_w", "fc_w", "fc2_w")


class AdapterError(RuntimeError):
    """An adapter could not be built, loaded, or applied."""


@dataclass(frozen=True)
class AdapterResult:
    """What a training call produced."""

    steps: int
    initial_loss: float
    final_loss: float
    trainable_parameters: int
    base_parameters: int
    cancelled: bool = False

    @property
    def improvement(self) -> float:
        return self.initial_loss - self.final_loss

    def describe(self) -> str:
        state = "cancelled" if self.cancelled else "finished"
        return (
            f"{state}: {self.steps} step(s), loss {self.initial_loss:.4f} -> "
            f"{self.final_loss:.4f} (Δ {self.improvement:+.4f}), "
            f"{self.trainable_parameters:,} trainable / {self.base_parameters:,} base"
        )


class AdapterBackend(Protocol):
    """The interface the trainer drives, whichever backend is in use."""

    method: str

    def train(
        self,
        examples: Sequence[str],
        *,
        config: Any,
        tokenizer: Any,
        on_step: Any = None,
        should_cancel: Any = None,
    ) -> AdapterResult: ...

    def save(self, directory: Path) -> Path: ...

    def load(self, directory: Path) -> None: ...

    def apply(self, model: Any) -> Any: ...

    def parameter_count(self) -> int: ...

    def describe(self) -> str: ...


# --------------------------------------------------------------- the numpy LoRA


def _digest_parameters(parameters: Sequence[np.ndarray]) -> str:
    """A stable digest of a parameter list, so a base can be proven unchanged."""
    hasher = hashlib.blake2b(digest_size=16)
    for array in parameters:
        hasher.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return hasher.hexdigest()


@dataclass
class _LoRAPair:
    """One low-rank update: ``B @ A`` added to a frozen weight matrix."""

    name: str
    a: np.ndarray
    b: np.ndarray
    scale: float
    base_shape: tuple[int, int]

    def delta(self) -> np.ndarray:
        return (self.b @ self.a) * self.scale

    @property
    def trainable_parameters(self) -> int:
        return int(self.a.size + self.b.size)


class NumpyLoRAAdapter:
    """A real LoRA adapter over the project's own numpy transformer.

    The base transformer's parameters are **frozen** -- this class keeps its own
    ``A``/``B`` matrices and never writes to the base -- and training updates only
    those matrices. Applying the adapter adds ``scale * (B @ A)`` to each targeted
    weight matrix, which is exactly LoRA with the base frozen.
    """

    method = METHOD_LORA_INPROCESS

    def __init__(
        self,
        *,
        rank: int = 4,
        alpha: int | None = None,
        dropout: float = 0.0,
        target_suffixes: Sequence[str] = TARGET_SUFFIXES,
        seed: int = 0,
    ) -> None:
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2
        self.dropout = dropout
        self.target_suffixes = tuple(target_suffixes)
        self.seed = seed
        self._pairs: dict[str, _LoRAPair] = {}
        self._rng = np.random.default_rng(seed)
        self._base_digest_before = ""

    # -- construction ------------------------------------------------------
    def attach(self, model: Any) -> None:
        """Build one ``A``/``B`` pair per targeted base weight matrix.

        ``A`` is a small random matrix (the down-projection) and ``B`` starts at
        zero, which is the standard LoRA initialisation: the adapter is exactly
        the identity update at step 0, so a freshly attached adapter changes
        nothing until it is trained.
        """
        self._base_digest_before = _digest_parameters(list(model.parameters()))
        self._pairs.clear()
        for name, matrix in self._iter_targets(model):
            out_features, in_features = matrix.shape
            a = self._rng.normal(0.0, 0.02, size=(self.rank, in_features)).astype(
                np.float32
            )
            b = np.zeros((out_features, self.rank), dtype=np.float32)
            self._pairs[name] = _LoRAPair(
                name=name,
                a=a,
                b=b,
                scale=self.alpha / self.rank,
                base_shape=(out_features, in_features),
            )
        if not self._pairs:
            raise AdapterError(
                "No weight matrix matched the target suffixes "
                f"{self.target_suffixes}; nothing to adapt."
            )

    def _iter_targets(self, model: Any) -> list[tuple[str, np.ndarray]]:
        targets: list[tuple[str, np.ndarray]] = []
        for index, layer in enumerate(getattr(model, "layers", [])):
            for suffix in self.target_suffixes:
                matrix = getattr(layer, suffix, None)
                if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                    targets.append((f"layers.{index}.{suffix}", matrix))
        return targets

    @property
    def trainable_parameters(self) -> int:
        return sum(pair.trainable_parameters for pair in self._pairs.values())

    def parameter_count(self) -> int:
        return self.trainable_parameters

    def base_parameters(self, model: Any) -> int:
        """The number of frozen base parameters (for an honest size comparison)."""
        return int(sum(int(p.size) for p in model.parameters()))

    @property
    def base_digest(self) -> str:
        """The digest captured when the adapter was attached."""
        return self._base_digest_before

    def base_unchanged(self, model: Any) -> bool:
        """True when the base parameters still hash to the digest captured at attach."""
        return _digest_parameters(list(model.parameters())) == self._base_digest_before

    # -- applying ----------------------------------------------------------
    def apply(self, model: Any) -> Any:
        """Add each trained update to the base weights, in place, and return the model.

        This is the *inference* step: a copy of the base is loaded and the adapter
        is folded onto it. The stored adapter never touches the base file.
        """
        for name, pair in self._pairs.items():
            matrix = self._resolve(model, name)
            if matrix.shape != pair.base_shape:
                raise AdapterError(
                    f"Base shape {matrix.shape} does not match the adapter's "
                    f"{pair.base_shape} for {name}."
                )
            matrix[...] = matrix + pair.delta().astype(matrix.dtype)
        return model

    def _resolve(self, model: Any, name: str) -> np.ndarray:
        _, index, suffix = name.split(".")
        layer = model.layers[int(index)]
        return getattr(layer, suffix)

    # -- training ----------------------------------------------------------
    def train(
        self,
        examples: Sequence[str],
        *,
        config: Any,
        tokenizer: Any,
        on_step: Any = None,
        should_cancel: Any = None,
    ) -> AdapterResult:
        """Train the adapter with the project's own forward/backward pass.

        The loss is the transformer's own next-token cross-entropy; the only
        parameters updated are the LoRA ``A``/``B`` matrices. A missing tokenizer
        or an empty dataset raises rather than reporting a fake success.
        """
        if not self._pairs:
            raise AdapterError("attach(model) must be called before train().")
        if tokenizer is None:
            raise AdapterError("A tokenizer is required to train.")
        stream = _encode_examples(examples, tokenizer)
        block = int(getattr(self.model_config, "block_size", 64)) if hasattr(self, "model_config") else 64
        if len(stream) <= block + 1:
            raise AdapterError(
                f"The training text is too short ({len(stream)} tokens) for a "
                f"block of {block}; add more examples."
            )
        return self._train_stream(stream, config=config, block=block, on_step=on_step, should_cancel=should_cancel)

    def _train_stream(
        self,
        stream: Sequence[int],
        *,
        config: Any,
        block: int,
        on_step: Any,
        should_cancel: Any,
    ) -> AdapterResult:
        model = self._model
        optimizer_state = {name: {"m": np.zeros_like(p.a), "v": np.zeros_like(p.a)} for name, p in self._pairs.items()}
        for name, pair in self._pairs.items():
            optimizer_state[name]["mb"] = np.zeros_like(pair.b)
            optimizer_state[name]["vb"] = np.zeros_like(pair.b)
        lr = float(config.learning_rate)
        batch_size = int(config.batch_size)
        max_steps = int(config.max_steps)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        initial_loss: float | None = None
        final_loss = 0.0
        steps = 0
        cancelled = False
        for step in range(1, max_steps + 1):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            batch_x, batch_y = _sample_block(stream, block, batch_size, self._rng)
            with _apply_adapter(self, model):
                logits = model.forward(batch_x)
                loss = model.loss(logits, batch_y)
                base_grads = model.backward(batch_y)
            if initial_loss is None:
                initial_loss = loss
            final_loss = loss
            grad_maps = self._adapter_grads(model, base_grads)
            for name, pair in self._pairs.items():
                g_a, g_b = grad_maps[name]
                state = optimizer_state[name]
                state["m"] = beta1 * state["m"] + (1 - beta1) * g_a
                state["v"] = beta2 * state["v"] + (1 - beta2) * (g_a * g_a)
                pair.a -= lr * (state["m"] / (1 - beta1**step)) / (
                    np.sqrt(state["v"] / (1 - beta2**step)) + eps
                )
                state["mb"] = beta1 * state["mb"] + (1 - beta1) * g_b
                state["vb"] = beta2 * state["vb"] + (1 - beta2) * (g_b * g_b)
                pair.b -= lr * (state["mb"] / (1 - beta1**step)) / (
                    np.sqrt(state["vb"] / (1 - beta2**step)) + eps
                )
            steps = step
            if on_step is not None:
                on_step(step, float(loss))
        return AdapterResult(
            steps=steps,
            initial_loss=float(initial_loss or 0.0),
            final_loss=float(final_loss),
            trainable_parameters=self.trainable_parameters,
            base_parameters=sum(int(m.size) for _n, m in self._iter_targets(model)),
            cancelled=cancelled,
        )

    def _adapter_grads(self, model: Any, base_grads: list[np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Derive each ``A``/``B`` gradient from the base weight gradient.

        For an update ``delta = scale * (B @ A)`` added to a frozen weight ``W``
        with incoming gradient ``g`` (``dL/dW``): ``dL/dB = scale * g @ A.T`` and
        ``dL/dA = scale * B.T @ g``. The base gradients the transformer already
        computed are exactly ``g``, so no second backward pass is needed.
        """
        # Map the transformer's flat gradient list back to names.
        named = _name_base_gradients(model, base_grads)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, pair in self._pairs.items():
            g = named.get(name)
            if g is None:
                out[name] = (np.zeros_like(pair.a), np.zeros_like(pair.b))
                continue
            scale = pair.scale
            g_b = scale * (g @ pair.a.T)
            g_a = scale * (pair.b.T @ g)
            out[name] = (g_a.astype(np.float32), g_b.astype(np.float32))
        return out

    # -- persistence -------------------------------------------------------
    def save(self, directory: Path) -> Path:
        """Write the adapter's own small file. The base is not written here."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for name, pair in self._pairs.items():
            arrays[f"{name}.A"] = pair.a
            arrays[f"{name}.B"] = pair.b
        target = directory / "adapter.npz"
        np.savez_compressed(target, **arrays)  # type: ignore[arg-type]
        meta = {
            "method": self.method,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": list(self._pairs),
            "trainable_parameters": self.trainable_parameters,
            "base_digest": self._base_digest_before,
            "seed": self.seed,
        }
        (directory / "adapter.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        return target

    def load(self, directory: Path) -> None:
        """Read a saved adapter back, rebuilding the pairs from the metadata."""
        directory = Path(directory)
        meta_path = directory / "adapter.json"
        arrays_path = directory / "adapter.npz"
        if not meta_path.is_file() or not arrays_path.is_file():
            raise AdapterError(f"No adapter found in {directory}.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("method") != self.method:
            raise AdapterError(
                f"The stored adapter is {meta.get('method')!r}, not {self.method!r}."
            )
        self.rank = int(meta["rank"])
        self.alpha = int(meta["alpha"])
        self.dropout = float(meta.get("dropout", 0.0))
        self._base_digest_before = str(meta.get("base_digest", ""))
        self._pairs.clear()
        with np.load(arrays_path) as data:
            for name in meta["targets"]:
                a = data[f"{name}.A"]
                b = data[f"{name}.B"]
                self._pairs[name] = _LoRAPair(
                    name=name,
                    a=a,
                    b=b,
                    scale=self.alpha / self.rank,
                    base_shape=(b.shape[0], a.shape[1]),
                )

    def describe(self) -> str:
        return (
            f"in-process LoRA (rank {self.rank}, alpha {self.alpha}) over "
            f"{len(self._pairs)} matrix(es), {self.trainable_parameters:,} trainable"
        )

    # -- the model is injected, never owned --------------------------------
    _model: Any = None
    model_config: Any = None

    def bind(self, model: Any) -> None:
        """Bind the base model this adapter adapts (the trainer's job)."""
        self._model = model
        self.model_config = getattr(model, "config", None)


def _name_base_gradients(model: Any, grads: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Map the transformer's flat gradient list onto ``layers.<i>.<suffix>`` names.

    ``TransformerLM.backward`` returns gradients in the order
    ``TransformerLM.parameters()`` lists them. For each layer that order is
    ``ln1_w, ln1_b, attn_w, attn_b, proj_w, proj_b, ln2_w, ln2_b, fc_w, fc_b,
    fc2_w, fc2_b``; the token/position embeddings and the final layer norm come
    before and after. Only the 2-D weight matrices are named, because those are
    the only ones an adapter targets.
    """
    named: dict[str, np.ndarray] = {}
    offset = 2  # token_embedding, position_embedding
    per_layer = 12
    layer_offsets = {"attn_w": 2, "proj_w": 4, "fc_w": 8, "fc2_w": 10}
    for index in range(len(getattr(model, "layers", []))):
        base = offset + index * per_layer
        for suffix, delta in layer_offsets.items():
            position = base + delta
            if position < len(grads):
                named[f"layers.{index}.{suffix}"] = grads[position]
    return named


class _apply_adapter:
    """A context manager that adds the adapter's update for the duration of a step.

    The transformer's forward pass reads the base weight matrices directly, so to
    train the adapter we add its current delta to each targeted matrix, run the
    pass, and subtract it again -- the base is restored exactly on exit, which is
    what keeps the frozen base frozen.
    """

    def __init__(self, adapter: NumpyLoRAAdapter, model: Any) -> None:
        self.adapter = adapter
        self.model = model
        self._saved: list[tuple[np.ndarray, np.ndarray]] = []

    def __enter__(self) -> None:
        # The exact pre-add value is kept, so the base is restored *bit for bit*
        # rather than by subtracting the delta again (which can leave a float
        # rounding residue behind).
        self._saved.clear()
        for name, pair in self.adapter._pairs.items():
            matrix = self.adapter._resolve(self.model, name)
            self._saved.append((matrix, matrix.copy()))
            matrix[...] = matrix + pair.delta().astype(matrix.dtype)

    def __exit__(self, *_exc: object) -> None:
        for matrix, original in self._saved:
            matrix[...] = original
        self._saved.clear()


def _encode_examples(examples: Sequence[str], tokenizer: Any) -> list[int]:
    """Encode a dataset's texts into one token stream, with a separator between."""
    stream: list[int] = []
    for text in examples:
        ids = _tokenize(tokenizer, text)
        stream.extend(ids)
        stream.append(_eos(tokenizer))
    return stream


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text))
    if hasattr(tokenizer, "token_ids"):
        return list(tokenizer.token_ids(text))
    raise AdapterError("The tokenizer exposes neither encode() nor token_ids().")


def _eos(tokenizer: Any) -> int:
    for attribute in ("eos_id", "eos_token_id", "vocab_size"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int):
            return value if attribute != "vocab_size" else max(0, value - 1)
    return 0


def _sample_block(
    stream: Sequence[int], block: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A batch of ``batch_size`` random windows of length ``block`` from the stream."""
    limit = len(stream) - block - 1
    starts = rng.integers(0, max(1, limit), size=batch_size)
    x = np.stack([np.asarray(stream[s : s + block], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(stream[s + 1 : s + block + 1], dtype=np.int64) for s in starts])
    return x, y


# --------------------------------------------------------------- the peft backend


#: The runner signature the PEFT backend drives: it receives the loaded base
#: (whatever the runtime produced), the prepared examples, the settings, and the
#: optional progress/cancel hooks, and returns the training metrics.
PeftRunner = Any


@dataclass
class PeftAdapterBackend:
    """The optional Hugging Face ``peft`` backend for a safetensors base model.

    Every heavy import (``torch``/``transformers``/``peft``) is lazy and every
    absence is reported as a real next step, so a base install neither imports nor
    needs them. ``runner`` is the injection seam: the default
    :class:`TransformersPeftRunner` builds a real ``peft`` ``LoraConfig`` over a
    ``transformers`` model, and a test injects its own runner, so the backend is
    exercised with no extra and no weights -- exactly the pattern the project's
    other optional engines use.

    The backend only ever produces a **separate adapter directory**; the base
    weights are loaded read-only and never written.
    """

    method: str = METHOD_LORA_TRANSFORMERS
    rank: int = 4
    alpha: int | None = None
    dropout: float = 0.05
    seed: int | None = None
    base_path: str = ""
    runner: PeftRunner = None
    _adapter_dir: Path | None = None

    def _require(self) -> None:
        from core.optional_deps import FEATURE_TRAINING, install_hint
        from core.optional_deps import status as feature_status

        state = feature_status(FEATURE_TRAINING)
        if not state.python_ok:
            raise AdapterError(
                "The transformers/peft training path needs the [training] extra: "
                f"{install_hint(FEATURE_TRAINING)}"
            )

    def _active_runner(self) -> PeftRunner:
        self._require()
        if self.runner is None:
            self.runner = TransformersPeftRunner(
                base_path=self.base_path,
                rank=self.rank,
                alpha=self.alpha if self.alpha is not None else self.rank * 2,
                dropout=self.dropout,
            )
        return self.runner

    def train(
        self,
        examples: Sequence[str],
        *,
        config: Any,
        tokenizer: Any,
        on_step: Any = None,
        should_cancel: Any = None,
    ) -> AdapterResult:
        if not examples:
            raise AdapterError("The transformers/peft path needs at least one example.")
        runner = self._active_runner()
        result = runner.train(
            list(examples), config=config, on_step=on_step, should_cancel=should_cancel
        )
        self._adapter_dir = Path(result.adapter_dir)
        return AdapterResult(
            steps=int(result.steps),
            initial_loss=float(result.initial_loss),
            final_loss=float(result.final_loss),
            trainable_parameters=int(result.trainable_parameters),
            base_parameters=int(result.base_parameters),
            cancelled=bool(getattr(result, "cancelled", False)),
        )

    def save(self, directory: Path) -> Path:
        """Copy the trained adapter into ``directory`` and return it."""
        if self._adapter_dir is None:
            raise AdapterError("Nothing to save: train() has not run.")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(self._adapter_dir.iterdir()):
            if entry.is_file():
                shutil.copy2(entry, target / entry.name)
        self._adapter_dir = target
        return target

    def load(self, directory: Path) -> None:
        path = Path(directory)
        if not path.is_dir():
            raise AdapterError(f"No adapter directory at {path}.")
        self._adapter_dir = path

    def apply(self, model: Any) -> Any:
        """Apply the saved adapter to a loaded base and return the model.

        Uses ``peft.PeftModel.from_pretrained`` on the adapter directory; the base
        model object is returned with the adapter merged in memory, and no base
        file is written.
        """
        self._require()
        if self._adapter_dir is None:
            raise AdapterError("No adapter is loaded; call load() first.")
        try:  # pragma: no cover - needs the extra
            from peft import PeftModel
        except Exception as error:  # pragma: no cover - needs the extra
            raise AdapterError(f"peft is not importable: {error}") from error
        return PeftModel.from_pretrained(model, str(self._adapter_dir))

    def parameter_count(self) -> int:
        runner = self.runner
        return int(getattr(runner, "trainable_parameters", 0) or 0)

    def describe(self) -> str:
        return "Hugging Face peft LoRA (needs the [training] extra and safetensors weights)"


@dataclass
class _PeftTrainOutcome:
    """What a PEFT runner produced: metrics and where the adapter was written."""

    steps: int
    initial_loss: float
    final_loss: float
    trainable_parameters: int
    base_parameters: int
    adapter_dir: str
    cancelled: bool = False


class TransformersPeftRunner:
    """A real ``transformers`` + ``peft`` LoRA run, imported only when it is used.

    ``transformers``/``peft``/``torch`` are imported lazily inside :meth:`train`,
    so a base install never touches them. The base model is loaded with
    ``torch_dtype`` matching the machine, wrapped with ``get_peft_model``, and
    trained with the project's own settings; only the adapter is written out.
    """

    def __init__(
        self,
        *,
        base_path: str = "",
        rank: int = 4,
        alpha: int | None = None,
        dropout: float = 0.05,
        target_modules: Sequence[str] | None = None,
    ) -> None:
        self.base_path = base_path
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2
        self.dropout = dropout
        self.target_modules = tuple(target_modules or ("q_proj", "v_proj"))
        self.trainable_parameters = 0

    def train(  # pragma: no cover - needs the extra and real weights
        self,
        examples: Sequence[str],
        *,
        config: Any,
        on_step: Any = None,
        should_cancel: Any = None,
    ) -> _PeftTrainOutcome:
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except Exception as error:
            raise AdapterError(
                "The transformers/peft runtime could not be imported "
                f"({error}); install requirements-training.txt."
            ) from error

        if not self.base_path:
            raise AdapterError("A safetensors base model path is required for PEFT training.")
        tokenizer = AutoTokenizer.from_pretrained(self.base_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_path, torch_dtype=torch.float32
        )
        lora = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        self.trainable_parameters = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        base_parameters = int(sum(p.numel() for p in model.parameters()))

        encodings = tokenizer(list(examples), return_tensors="pt", padding=True, truncation=True)
        encodings["labels"] = encodings["input_ids"].clone()
        dataset = _TorchDataset(encodings)

        out_dir = Path(getattr(config, "output_dir", "") or ".") / "peft-adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=int(config.batch_size),
            gradient_accumulation_steps=int(config.grad_accum),
            max_steps=int(config.max_steps),
            learning_rate=float(config.learning_rate),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )

        history: list[float] = []

        class _Callback:
            def on_log(self, _args, state, control, logs=None, **_kwargs):
                loss = float((logs or {}).get("loss", 0.0))
                history.append(loss)
                if on_step is not None:
                    on_step(state.global_step, loss)
                if should_cancel is not None and should_cancel():
                    control.should_training_stop = True

        from transformers import TrainerCallback

        class _CallbackImpl(TrainerCallback, _Callback):
            pass

        trainer = Trainer(
            model=model, args=args, train_dataset=dataset, callbacks=[_CallbackImpl()]
        )
        trainer.train()
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        initial = history[0] if history else 0.0
        final = history[-1] if history else 0.0
        return _PeftTrainOutcome(
            steps=len(history),
            initial_loss=initial,
            final_loss=final,
            trainable_parameters=self.trainable_parameters,
            base_parameters=base_parameters,
            adapter_dir=str(out_dir),
        )


class _TorchDataset:  # pragma: no cover - needs torch
    """A minimal ``torch`` dataset over tokenizer encodings."""

    def __init__(self, encodings: Any) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return int(next(iter(self.encodings.values())).shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {key: value[index] for key, value in self.encodings.items()}


def backend_for(method: str, **kwargs: Any) -> AdapterBackend:
    """The adapter backend for a training method.

    ``lora_inprocess`` (the default) always returns the numpy adapter; the
    transformers path returns the peft backend, which reports honestly when its
    extra is absent rather than pretending to be ready.
    """
    if method == METHOD_LORA_INPROCESS:
        return NumpyLoRAAdapter(**kwargs)
    if method == METHOD_LORA_TRANSFORMERS:
        return PeftAdapterBackend(**kwargs)
    raise AdapterError(f"No adapter backend for method {method!r}.")


def describe_adapters() -> str:
    """One readable line per adapter method, so a menu can show the choices."""
    return "\n".join(
        f"{method}: {summary}"
        for method, summary in (
            (METHOD_LORA_INPROCESS, "LoRA over the in-project transformer (CPU, no extra)"),
            (METHOD_LORA_TRANSFORMERS, "LoRA over a transformers model (needs the [training] extra)"),
            ("qlora", "QLoRA 4-bit (needs a CUDA GPU)"),
        )
    )


__all__ = [
    "TARGET_SUFFIXES",
    "AdapterBackend",
    "AdapterError",
    "AdapterResult",
    "NumpyLoRAAdapter",
    "PeftAdapterBackend",
    "TransformersPeftRunner",
    "backend_for",
    "describe_adapters",
]
