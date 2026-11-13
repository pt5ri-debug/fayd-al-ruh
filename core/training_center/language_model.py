"""The Language Model workspace: every way the project trains Fayd Al-Ruh's own model.

The project trains a small in-project transformer language model (``ml.llm``) and
ships the result as a committed checkpoint. That trainer grew a large surface over
time -- architecture presets, tokenizer kinds, corpus sources, an optimiser and
schedule, resumable sharded training, a full release run, ready-made weights to
import or continue, larger capacity tiers to build, and held-out evaluation -- and
each option used to be reachable only from its own scattered place (the CLI's
``llm``, ``release``, ``weights`` and ``checkpoints`` commands, the GUI sidebar's
training actions, and the ``cognition`` staged trainer).

This module is the **one engine** behind a single, professional surface: it drives
the project's **real** trainers and helpers -- it reimplements no training step --
and returns plain reports the window and the CLI both render, so the two can never
disagree and no option is silently lost.

Honest by construction:

* a control the machine cannot do names the **real next step** instead of failing
  with a raw traceback;
* nothing is trained or downloaded on import -- the heavy modules are imported
  inside the method that needs them;
* every option the trainer accepts is a field of :class:`LanguageModelSettings`,
  so the workspace is a superset of the scattered commands, never a subset;
* a run's own progress lines are captured and forwarded, so a long fit never
  looks frozen and the outcome is read from the run itself, never assumed.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from storage.json_store import JsonStore

from core.paths import checkpoints_dir

#: The checkpoint a Training Center run trains and continues. The same file
#: ``python -m ml.llm`` writes, so the two never diverge.
TRAIN_CHECKPOINT_NAME = "fayd-llm.npz"

#: The JsonStore namespace/key the language-model settings live under.
_NAMESPACE = "training_center"
_SETTINGS_KEY = "language_model"

#: The architecture presets ``ml.llm`` really offers, with a one-line description.
PRESETS: dict[str, str] = {
    "small": "3 layers / 96 wide -- the fast default, good for a quick check",
    "release-1": "4 layers / 128 wide, weight-tied -- the released capacity tier",
    "capable": "the larger tier for a machine with time to spare",
    "modern": "RMSNorm + RoPE + SwiGLU at the released width (the modern primitives)",
}

#: The tokenizer kinds, with a one-line description.
TOKENIZER_KINDS: dict[str, str] = {
    "byte": "raw bytes -- linear, always works, used for the full corpus",
    "bpe": "subword BPE -- ~3x smaller text, slower to train",
}

#: The learning-rate decay shapes.
LR_SCHEDULES: tuple[str, ...] = ("cosine", "linear", "constant")

#: The learned capacity tiers this build can train (fallback if the registry fails).
LEARNED_TIER_FALLBACK: tuple[str, ...] = ("learned-large", "learned-xl")

#: A short, readable note per corpus source, so the workspace explains each switch.
CORPUS_SOURCES: tuple[tuple[str, str], ...] = (
    ("include_runtime", "documents written at runtime (internet learning, feedback)"),
    ("include_characters", "the seven characters' own learning (facts and exchanges)"),
    (
        "include_character_corpus",
        "the characters' own knowledge corpora (opt-in per character, non-private only)",
    ),
    ("include_sessions", "the local mouths' own saved conversations (personal -- off by default)"),
)

_STEP_RE = re.compile(r"step\s+(\d+)\s+loss\s+([0-9.]+)")
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+[0-9. ]*%.*?loss\s+([0-9.]+)")
_SUMMARY_RE = re.compile(
    r"trained\s+(\d+)\s+steps\s+loss\s+([0-9.]+)\s*->\s*([0-9.]+)\s+val\s+([0-9.]+)"
    r"(?:\s+best-val\s+([0-9.]+)(?:\s+\(step\s+(\d+)\))?)?"
    r".*?params\s+(\d+).*?tokens\s+(\d+)"
)


def _checkpoint_path(root: Path | str | None = None) -> Path:
    """The checkpoint a Training Center run writes and continues."""
    if root is not None:
        return Path(root) / "ml" / "llm" / "checkpoints" / TRAIN_CHECKPOINT_NAME
    return checkpoints_dir() / TRAIN_CHECKPOINT_NAME


def _default_data_dir() -> Path:
    from core.paths import data_dir

    return data_dir()


@dataclass(frozen=True)
class LanguageModelSettings:
    """Every option a training run accepts -- the whole surface in one place.

    The defaults reproduce a serious **release** run (the good checkpoint a user
    actually wants), and every field maps onto a real argument of
    :func:`ml.llm.__main__.train_and_save` (or, for the release fields, of
    :class:`ml.release.TrainingConfig`), so the workspace can never offer a knob
    the trainer ignores.
    """

    # -- architecture ------------------------------------------------------
    preset: str = "release-1"
    tokenizer_kind: str = "byte"
    vocab_size: int = 1024

    # -- the run -----------------------------------------------------------
    steps: int = 4000
    repeats: int = 40
    structured_repeats: int = 60
    seed: int = 0
    learning_rate: float = 3e-3
    warmup: int | None = None
    max_grad_norm: float = 1.0
    val_every: int = 200
    log_every: int = 100

    # -- optimiser and schedule -------------------------------------------
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    restore_best: bool = True
    accum_steps: int = 1

    # -- corpus sources ----------------------------------------------------
    kb_max_chunks: int = 1500
    kb_repeats: int = 1
    include_runtime: bool = True
    include_characters: bool = True
    include_character_corpus: bool = False
    character_corpus_chunks: int = 2000
    include_sessions: bool = False

    # -- resuming and the sharded schedule --------------------------------
    resume: bool = True
    shard_count: int = 0

    # -- the whole-project release run ------------------------------------
    episodes: int = 120
    replay_batch: int = 32
    include_base: bool = True
    include_procedural: bool = True
    measure_perplexity: bool = False

    # -- generation --------------------------------------------------------
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    def to_primitive(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_primitive(cls, data: object) -> LanguageModelSettings:
        """Rebuild from a stored mapping, ignoring unknown or missing keys."""
        if not isinstance(data, dict):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_values(self, **changes: Any) -> LanguageModelSettings:
        return replace(self, **changes)

    def describe(self) -> str:
        lines = ["Language model — the settings in force"]
        lines.append(
            f"  architecture: {self.preset}  tokenizer {self.tokenizer_kind}/{self.vocab_size}"
        )
        lines.append(
            f"  run: {self.steps} steps  repeats {self.repeats}  seed {self.seed}  "
            f"lr {self.learning_rate}"
        )
        lines.append(
            f"  optimiser: weight-decay {self.weight_decay}  schedule {self.lr_schedule}  "
            f"restore-best {'on' if self.restore_best else 'off'}  accum {self.accum_steps}"
        )
        lines.append(
            f"  corpus: kb {self.kb_max_chunks} chunks  "
            f"runtime {'on' if self.include_runtime else 'off'}  "
            f"characters {'on' if self.include_characters else 'off'}  "
            f"character-corpus {'on' if self.include_character_corpus else 'off'}  "
            f"sessions {'on' if self.include_sessions else 'off'}"
        )
        lines.append(
            f"  resume {'on' if self.resume else 'off'}  "
            f"shards {self.shard_count or 'whole corpus'}"
        )
        return "\n".join(lines)


@dataclass
class TrainOutcome:
    """What one language-model training run did, read from the run itself."""

    ok: bool
    steps: int
    total_steps: int
    initial_loss: float
    final_loss: float
    best_val: float
    best_val_step: int | None
    parameters: int
    tokens: int
    checkpoint: str
    resumed: bool
    duration_seconds: float
    lines: tuple[str, ...] = ()
    error: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"The run failed: {self.error}"
        resumed = "resumed" if self.resumed else "from scratch"
        best = f"{self.best_val:.3f}"
        if self.best_val_step is not None:
            best += f" (step {self.best_val_step})"
        return (
            f"Trained {self.steps} steps ({resumed}) — loss "
            f"{self.initial_loss:.3f} → {self.final_loss:.3f}, best val {best}\n"
            f"  parameters {self.parameters:,}  tokens {self.tokens:,}  "
            f"total steps {self.total_steps}  {self.duration_seconds:.1f}s\n"
            f"  saved {self.checkpoint}"
        )

    def to_primitive(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("lines", None)
        return data


@dataclass
class CheckpointReport:
    """The state of the checkpoint a run continues, read from the file itself."""

    exists: bool
    path: str
    steps: int = 0
    size_bytes: int = 0
    parameters: int | None = None
    preset: str = ""
    tokenizer: str = ""
    architecture: dict[str, Any] = field(default_factory=dict)
    next_step: str = ""

    def describe(self) -> str:
        if not self.exists:
            return (
                f"No checkpoint at {self.path} yet.\n"
                f"  {self.next_step or 'Train one from scratch to create it.'}"
            )
        lines = [
            f"Checkpoint: {self.path}",
            f"  steps: {self.steps}  size: {self.size_bytes / 1024:.1f} KB",
        ]
        if self.parameters is not None:
            lines.append(
                f"  parameters: {self.parameters:,}  preset {self.preset}  "
                f"tokenizer {self.tokenizer or 'n/a'}"
            )
        if self.architecture:
            arch = "  ".join(f"{k}={v}" for k, v in self.architecture.items())
            lines.append(f"  architecture: {arch}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Held-out evaluation of the checkpoint: loss, perplexity, bits per byte."""

    ok: bool
    checkpoint: str
    train_ppl: float = 0.0
    validation_ppl: float = 0.0
    test_ppl: float = 0.0
    bits_per_byte: float = 0.0
    test_tokens: int = 0
    error: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"Could not evaluate the checkpoint: {self.error}"
        return (
            f"Held-out evaluation of {self.checkpoint}\n"
            f"  perplexity: train {self.train_ppl:.3f}  validation "
            f"{self.validation_ppl:.3f}  test {self.test_ppl:.3f}\n"
            f"  bits per byte {self.bits_per_byte:.4f}  over {self.test_tokens:,} test tokens"
        )

    def to_primitive(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightActionReport:
    """The result of installing, importing, fetching, or building weights."""

    ok: bool
    kind: str
    message: str
    path: str = ""
    command: str = ""
    backup: str = ""

    def describe(self) -> str:
        head = "Done" if self.ok else "Not done"
        lines = [f"{head}: {self.message}"]
        if self.path:
            lines.append(f"  path: {self.path}")
        if self.backup:
            lines.append(f"  previous file kept as: {self.backup}")
        if self.command:
            lines.append(f"  continue with: {self.command}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReleaseOutcome:
    """What a whole-project release run produced, from the manifest it wrote."""

    ok: bool
    manifest: dict[str, Any]
    manifest_path: str = ""
    error: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"The release run failed: {self.error}"
        from ml.release import format_manifest

        body = format_manifest(self.manifest)
        if self.manifest_path:
            body += f"\n\nManifest written to {self.manifest_path}"
        return body

    def to_primitive(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "manifest": self.manifest,
            "manifest_path": self.manifest_path,
        }


class _LineWriter(io.TextIOBase):
    """Capture a trainer's own output and forward it line by line.

    The trainer prints its progress and summary to stdout/stderr. Redirecting
    through this writer keeps that output out of the caller's own stream and lets
    every line reach the caller's live log, so a run's detail is visible while it
    happens rather than only at the end. Each line is also offered to an optional
    step callback, so live progress is read from the run's own words.
    """

    def __init__(self, sink: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self._sink = sink
        self._buffer = ""
        self.lines: list[str] = []

    def writable(self) -> bool:  # pragma: no cover - io contract
        return True

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            self._record(line)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - io contract
        if self._buffer:
            self._record(self._buffer)
            self._buffer = ""

    def _record(self, line: str) -> None:
        stripped = line.rstrip()
        if not stripped:
            return
        self.lines.append(stripped)
        if self._sink is not None:
            self._sink(stripped)


@contextmanager
def _capture(writer: io.TextIOBase) -> Iterator[None]:
    """Redirect stdout and stderr to ``writer`` for the duration of a block."""
    with redirect_stdout(writer), redirect_stderr(writer):  # type: ignore[arg-type]
        yield


class LanguageModelEngine:
    """Drive the project's real language-model trainer and its helpers.

    Every collaborator may be injected, so the whole engine is exercised with no
    weights and no long run: ``trainer`` replaces :func:`train_and_save`,
    ``release_runner`` replaces :func:`ml.release.train_release`, and ``store``
    replaces the JSON store.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        checkpoint: Path | str | None = None,
        store: JsonStore | None = None,
        trainer: Callable[..., Any] | None = None,
        release_runner: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self._checkpoint = Path(checkpoint) if checkpoint is not None else _checkpoint_path(root)
        self._store = store or JsonStore(self.root or _default_data_dir())
        self._trainer = trainer
        self._release_runner = release_runner

    # -- the checkpoint ----------------------------------------------------
    @property
    def checkpoint(self) -> Path:
        return self._checkpoint

    def checkpoint_report(self) -> CheckpointReport:
        """Read the checkpoint's own step count and architecture."""
        path = self._checkpoint
        if not path.exists():
            return CheckpointReport(
                exists=False,
                path=str(path),
                next_step="Train one from scratch with a Language Model run to create it.",
            )
        steps = 0
        try:
            from ml.llm.state import load_step_count, state_path

            steps = int(load_step_count(state_path(path)) or 0)
        except Exception:  # a sidecar may be absent; the checkpoint is still real
            steps = 0
        parameters: int | None = None
        preset = ""
        architecture: dict[str, Any] = {}
        try:
            from ml.llm.transformer import TransformerLM

            model = TransformerLM.load(path)
            parameters = int(sum(p.size for p in model.parameters()))
            config = model.config
            preset = _match_preset(config)
            architecture = {
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "block_size": config.block_size,
            }
        except Exception:
            pass
        tokenizer = ""
        tokenizer_file = path.with_suffix(".tokenizer.json")
        if tokenizer_file.exists():
            try:
                import json

                data = json.loads(tokenizer_file.read_text(encoding="utf-8"))
                kind = data.get("kind", "byte")
                vocab = len(data.get("merges", [])) + 257 if kind == "bpe" else 257
                tokenizer = f"{kind}/{vocab}"
            except Exception:
                tokenizer = ""
        return CheckpointReport(
            exists=True,
            path=str(path),
            steps=steps,
            size_bytes=path.stat().st_size,
            parameters=parameters,
            preset=preset,
            tokenizer=tokenizer,
            architecture=architecture,
        )

    # -- settings ----------------------------------------------------------
    def settings(self) -> LanguageModelSettings:
        """The saved settings merged over the shipped defaults."""
        return LanguageModelSettings.from_primitive(self._store.load(_NAMESPACE, _SETTINGS_KEY))

    def save_settings(self, settings: LanguageModelSettings) -> LanguageModelSettings:
        self._store.save(_NAMESPACE, _SETTINGS_KEY, settings.to_primitive())
        return settings

    def set_setting(self, key: str, value: Any) -> LanguageModelSettings:
        """Persist one field; an unknown field is refused rather than stored."""
        if key not in LanguageModelSettings.__dataclass_fields__:
            raise KeyError(
                f"Unknown language-model setting {key!r}; expected one of "
                f"{tuple(LanguageModelSettings.__dataclass_fields__)}."
            )
        return self.save_settings(self.settings().with_values(**{key: value}))

    def reset_settings(self) -> LanguageModelSettings:
        return self.save_settings(LanguageModelSettings())

    # -- choice lists for a dropdown --------------------------------------
    def preset_choices(self) -> tuple[tuple[str, str], ...]:
        return tuple(PRESETS.items())

    def tokenizer_choices(self) -> tuple[tuple[str, str], ...]:
        return tuple(TOKENIZER_KINDS.items())

    def lr_schedule_choices(self) -> tuple[str, ...]:
        return LR_SCHEDULES

    def learned_tier_choices(self) -> tuple[str, ...]:
        try:
            from core.learned_weights import tier_keys

            return tuple(tier_keys())
        except Exception:
            return LEARNED_TIER_FALLBACK

    def pack_choices(self) -> tuple[tuple[str, str], ...]:
        """The committed weights-pack entries a user may install and continue."""
        try:
            from core.weights_pack import load_pack

            pack = load_pack()
            return tuple((entry.key, f"{entry.title} ({entry.kind})") for entry in pack.entries)
        except Exception:
            return ()

    # -- the corpus --------------------------------------------------------
    def corpus_summary(self, settings: LanguageModelSettings | None = None) -> dict[str, Any]:
        """How much text a run would train on, read from the real corpus."""
        settings = settings or self.settings()
        from ml.release import TrainingConfig, corpus_summary

        config = TrainingConfig(
            llm_steps=settings.steps,
            llm_repeats=settings.repeats,
            llm_structured_repeats=settings.structured_repeats,
            seed=settings.seed,
            llm_preset=settings.preset,
            tokenizer_kind=settings.tokenizer_kind,
            tokenizer_vocab_size=settings.vocab_size,
            kb_max_chunks=settings.kb_max_chunks,
            kb_repeats=settings.kb_repeats,
            include_runtime=settings.include_runtime,
            include_sessions=settings.include_sessions,
            include_characters=settings.include_characters,
            include_character_knowledge=settings.include_character_corpus,
            character_knowledge_chunks=settings.character_corpus_chunks,
        )
        return corpus_summary(config)

    def describe_corpus(self, settings: LanguageModelSettings | None = None) -> str:
        counts = self.corpus_summary(settings)
        lines = ["Language model — the training corpus"]
        for key in sorted(counts):
            value = counts[key]
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for inner, inner_value in sorted(value.items()):
                    lines.append(f"    {inner}: {inner_value}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    # -- training ----------------------------------------------------------
    def _trainer_fn(self) -> Callable[..., Any]:
        if self._trainer is not None:
            return self._trainer
        from ml.llm.__main__ import train_and_save

        return train_and_save

    def train(
        self,
        settings: LanguageModelSettings | None = None,
        *,
        on_line: Callable[[str], None] | None = None,
        on_step: Callable[[int, float, int], None] | None = None,
    ) -> TrainOutcome:
        """Train (or continue) the checkpoint, streaming the run's own lines.

        ``on_line`` receives every line the trainer prints, and ``on_step``
        receives ``(step, loss, max_steps)`` as the fit advances -- parsed from
        the run's own progress output, so the window shows live progress without
        a second progress implementation.
        """
        settings = settings or self.settings()
        step_parser = _StepParser(on_step)
        writer = _LineWriter(_compose_sink(on_line, step_parser))
        checkpoint = self._checkpoint
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        try:
            with _capture(writer):
                self._trainer_fn()(
                    steps=settings.steps,
                    repeats=settings.repeats,
                    structured_repeats=settings.structured_repeats,
                    seed=settings.seed,
                    checkpoint=checkpoint,
                    tokenizer_kind=settings.tokenizer_kind,
                    vocab_size=settings.vocab_size,
                    preset=settings.preset,
                    learning_rate=settings.learning_rate,
                    warmup=settings.warmup,
                    max_grad_norm=settings.max_grad_norm,
                    val_every=settings.val_every,
                    log_every=settings.log_every,
                    kb_max_chunks=settings.kb_max_chunks,
                    kb_repeats=settings.kb_repeats,
                    include_runtime=settings.include_runtime,
                    include_sessions=settings.include_sessions,
                    include_characters=settings.include_characters,
                    include_character_knowledge=settings.include_character_corpus,
                    character_knowledge_chunks=settings.character_corpus_chunks,
                    weight_decay=settings.weight_decay,
                    lr_schedule=settings.lr_schedule,
                    restore_best=settings.restore_best,
                    accum_steps=settings.accum_steps,
                    resume=settings.resume,
                    shard_count=settings.shard_count,
                    verbose=True,
                    progress_every=0,
                )
        except Exception as error:  # handed back honestly, never swallowed
            return TrainOutcome(
                ok=False, steps=settings.steps, total_steps=0, initial_loss=0.0,
                final_loss=0.0, best_val=float("inf"), best_val_step=None,
                parameters=0, tokens=0, checkpoint=str(checkpoint), resumed=False,
                duration_seconds=time.monotonic() - started,
                lines=tuple(writer.lines), error=f"{type(error).__name__}: {error}",
            )

        duration = time.monotonic() - started
        return self._outcome_from(settings, duration, writer.lines)

    def _outcome_from(
        self,
        settings: LanguageModelSettings,
        duration: float,
        lines: Iterable[str],
    ) -> TrainOutcome:
        summary = _parse_summary(lines)
        total_steps = settings.steps
        try:
            from ml.llm.state import load_step_count, state_path

            total_steps = int(load_step_count(state_path(self._checkpoint)) or settings.steps)
        except Exception:
            total_steps = settings.steps
        return TrainOutcome(
            ok=True,
            steps=settings.steps,
            total_steps=total_steps,
            initial_loss=summary.get("initial", 0.0),
            final_loss=summary.get("final", 0.0),
            best_val=summary.get("best", float("inf")),
            best_val_step=summary.get("best_step"),
            parameters=summary.get("parameters", 0),
            tokens=summary.get("tokens", 0),
            checkpoint=str(self._checkpoint),
            resumed=bool(settings.resume and total_steps > settings.steps),
            duration_seconds=duration,
            lines=tuple(lines),
        )

    # -- generation --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        settings: LanguageModelSettings | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Sample a continuation from the checkpoint (a real, greedy-able sample)."""
        from ml.llm.generator import LocalGenerator
        from ml.llm.tokenizer import load_tokenizer
        from ml.llm.transformer import TransformerLM

        settings = settings or self.settings()
        if not self._checkpoint.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self._checkpoint}; train one first, then sample."
            )
        model = TransformerLM.load(self._checkpoint)
        tokenizer_file = self._checkpoint.with_suffix(".tokenizer.json")
        tokenizer = load_tokenizer(tokenizer_file) if tokenizer_file.exists() else None
        generator = LocalGenerator(model, tokenizer)
        return generator.generate(
            prompt,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else settings.max_new_tokens,
            temperature=temperature if temperature is not None else settings.temperature,
            seed=seed,
            top_k=settings.top_k,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
        )

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        settings: LanguageModelSettings | None = None,
        *,
        on_line: Callable[[str], None] | None = None,
    ) -> EvaluationReport:
        """Measure the checkpoint on held-out data (loss, perplexity, bits/byte)."""
        settings = settings or self.settings()
        if not self._checkpoint.exists():
            return EvaluationReport(
                ok=False, checkpoint=str(self._checkpoint),
                error=f"No checkpoint at {self._checkpoint}; train one first.",
            )
        writer = _LineWriter(on_line)
        try:
            from ml.llm.corpus import corpus_units
            from ml.llm.evaluate import evaluate_checkpoint
            from ml.llm.tokenizer import load_tokenizer

            tokenizer_file = self._checkpoint.with_suffix(".tokenizer.json")
            tokenizer = load_tokenizer(tokenizer_file) if tokenizer_file.exists() else None
            with _capture(writer):
                report = evaluate_checkpoint(
                    self._checkpoint,
                    corpus_units(
                        include_sessions=settings.include_sessions,
                        include_characters=settings.include_characters,
                        include_character_knowledge=settings.include_character_corpus,
                        character_knowledge_chunks=settings.character_corpus_chunks,
                    ),
                    tokenizer=tokenizer,
                    seed=settings.seed,
                )
        except Exception as error:
            return EvaluationReport(
                ok=False, checkpoint=str(self._checkpoint),
                error=f"{type(error).__name__}: {error}",
            )
        return EvaluationReport(
            ok=True,
            checkpoint=str(self._checkpoint),
            train_ppl=float(getattr(report, "train_ppl", 0.0)),
            validation_ppl=float(getattr(report, "validation_ppl", 0.0)),
            test_ppl=float(getattr(report, "test_ppl", 0.0)),
            bits_per_byte=float(getattr(report, "test_bits_per_byte", 0.0)),
            test_tokens=int(getattr(report, "test_tokens", 0)),
        )

    # -- ready-made weights ------------------------------------------------
    def import_checkpoint(self, path: str | Path, *, force: bool = False) -> WeightActionReport:
        """Install ready-made ``.npz`` weights into the trainer's folder."""
        try:
            from core.checkpoint_import import install_weights

            outcome = install_weights(Path(path), destination=self._checkpoint, force=force)
        except Exception as error:
            return WeightActionReport(
                ok=False, kind="language", message=f"{type(error).__name__}: {error}"
            )
        return WeightActionReport(
            ok=bool(getattr(outcome, "installed", False)),
            kind=str(getattr(getattr(outcome, "info", None), "kind", "language")),
            message=outcome.describe(),
            path=str(getattr(outcome, "destination", "") or ""),
            backup=str(getattr(outcome, "backup", "") or ""),
            command="python -m cli training-center lm-train --resume",
        )

    def install_pack_entry(self, key: str, *, force: bool = False) -> WeightActionReport:
        """Install a committed weights-pack entry into this engine's checkpoint folder.

        The entry is **verified first** (digest, kind, architecture), so a tampered
        or mismatched file is refused and nothing is written. The destination is the
        engine's own checkpoint, so a run under a temporary data root is
        self-contained instead of reaching into the default one.
        """
        try:
            from core.checkpoint_import import install_weights
            from core.weights_pack import load_pack, verify_entry

            pack = load_pack()
            entry = next((item for item in pack.entries if item.key == key), None)
            if entry is None:
                return WeightActionReport(
                    ok=False, kind="pack",
                    message=f"No pack entry {key!r}; the pack offers "
                    + ", ".join(item.key for item in pack.entries),
                )
            verification = verify_entry(entry)
            if not verification.ok:
                return WeightActionReport(ok=False, kind="pack", message=verification.detail)
            outcome = install_weights(
                entry.path, kind=entry.kind, destination=self._checkpoint, force=force
            )
        except Exception as error:
            return WeightActionReport(
                ok=False, kind="pack", message=f"{type(error).__name__}: {error}"
            )
        return WeightActionReport(
            ok=bool(getattr(outcome, "installed", False)),
            kind="pack",
            message=outcome.describe(),
            path=str(getattr(outcome, "destination", "") or ""),
            backup=str(getattr(outcome, "backup", "") or ""),
            command="python -m cli training-center lm-train --resume",
        )

    def fetch_weights(self, url: str, *, approved: bool = False) -> WeightActionReport:
        """Fetch weights over HTTPS (governed: needs ``approved=True``)."""
        if not approved:
            return WeightActionReport(
                ok=False, kind="language",
                message="Fetching weights needs approval; re-run with --approve.",
            )
        try:
            from core.checkpoint_import import fetch_weights

            outcome = fetch_weights(url, approved=True)
        except Exception as error:
            return WeightActionReport(
                ok=False, kind="language", message=f"{type(error).__name__}: {error}"
            )
        return WeightActionReport(
            ok=bool(getattr(outcome, "installed", False)),
            kind=str(getattr(getattr(outcome, "info", None), "kind", "language")),
            message=outcome.describe(),
            path=str(getattr(outcome, "destination", "") or ""),
            command="python -m cli training-center lm-train --resume",
        )

    def build_learned(
        self,
        tier: str,
        *,
        force: bool = False,
        on_line: Callable[[str], None] | None = None,
    ) -> WeightActionReport:
        """Train one larger capacity tier into the pack folder, with provenance."""
        writer = _LineWriter(on_line)
        try:
            from core.learned_weights import build_learned

            with _capture(writer):
                outcome = build_learned(tier, force=force, verbose=True)
        except Exception as error:
            return WeightActionReport(
                ok=False, kind="learned", message=f"{type(error).__name__}: {error}"
            )
        return WeightActionReport(
            ok=bool(getattr(outcome, "built", False)),
            kind="learned",
            message=outcome.describe(),
            path=str(getattr(outcome, "checkpoint", "")),
            command="python -m cli training-center lm-train --resume",
        )

    def list_learned(self) -> tuple[dict[str, Any], ...]:
        try:
            from core.learned_weights import list_learned

            return tuple(entry.to_primitive() for entry in list_learned())
        except Exception:
            return ()

    def describe_learned(self) -> str:
        try:
            from core.learned_weights import describe_learned, describe_tiers

            return describe_tiers() + "\n\n" + describe_learned()
        except Exception as error:
            return f"Could not read the learned tiers: {type(error).__name__}: {error}"

    # -- the whole-project release run ------------------------------------
    def release(
        self,
        settings: LanguageModelSettings | None = None,
        *,
        on_line: Callable[[str], None] | None = None,
    ) -> ReleaseOutcome:
        """Train every learning layer and write the release manifest."""
        settings = settings or self.settings()
        writer = _LineWriter(on_line)
        try:
            if self._release_runner is not None:
                with _capture(writer):
                    manifest = self._release_runner(settings)
            else:
                from ml.release import TrainingConfig, train_release

                config = TrainingConfig(
                    episodes=settings.episodes,
                    llm_steps=settings.steps,
                    llm_repeats=settings.repeats,
                    llm_structured_repeats=settings.structured_repeats,
                    seed=settings.seed,
                    replay_batch=settings.replay_batch,
                    include_base=settings.include_base,
                    llm_preset=settings.preset,
                    tokenizer_kind=settings.tokenizer_kind,
                    tokenizer_vocab_size=settings.vocab_size,
                    kb_max_chunks=settings.kb_max_chunks,
                    kb_repeats=settings.kb_repeats,
                    include_runtime=settings.include_runtime,
                    include_sessions=settings.include_sessions,
                    include_characters=settings.include_characters,
                    include_character_knowledge=settings.include_character_corpus,
                    character_knowledge_chunks=settings.character_corpus_chunks,
                    include_procedural=settings.include_procedural,
                    measure_perplexity=settings.measure_perplexity,
                    resume=settings.resume,
                    shard_count=settings.shard_count,
                    weight_decay=settings.weight_decay,
                    lr_schedule=settings.lr_schedule,
                    restore_best=settings.restore_best,
                    accum_steps=settings.accum_steps,
                    max_grad_norm=settings.max_grad_norm,
                )
                with _capture(writer):
                    manifest = train_release(config, verbose=False)
        except Exception as error:
            return ReleaseOutcome(
                ok=False, manifest={}, error=f"{type(error).__name__}: {error}"
            )
        return ReleaseOutcome(
            ok=True,
            manifest=manifest,
            manifest_path=str(manifest.get("manifest_path", "")),
        )

    # -- the exact command a control runs ---------------------------------
    def plan(self, settings: LanguageModelSettings | None = None) -> str:
        """The real headless command equivalent to the settings in force."""
        settings = settings or self.settings()
        parts = [
            "python -m cli training-center lm-train",
            f"--preset {settings.preset}",
            f"--tokenizer-kind {settings.tokenizer_kind}",
            f"--steps {settings.steps}",
            f"--repeats {settings.repeats}",
            f"--seed {settings.seed}",
            f"--weight-decay {settings.weight_decay}",
            f"--lr-schedule {settings.lr_schedule}",
            f"--accum-steps {settings.accum_steps}",
            f"--kb-max-chunks {settings.kb_max_chunks}",
        ]
        if settings.shard_count:
            parts.append(f"--shards {settings.shard_count}")
        if not settings.resume:
            parts.append("--fresh")
        return " ".join(parts)


def _compose_sink(
    on_line: Callable[[str], None] | None, parser: _StepParser
) -> Callable[[str], None]:
    def sink(line: str) -> None:
        parser(line)
        if on_line is not None:
            on_line(line)

    return sink


class _StepParser:
    """Read live ``(step, loss)`` pairs out of the trainer's own progress lines."""

    def __init__(self, on_step: Callable[[int, float, int], None] | None) -> None:
        self._on_step = on_step
        self._total = 0
        self._last: tuple[int, float] | None = None

    def __call__(self, line: str) -> None:
        if self._on_step is None:
            return
        match = _PROGRESS_RE.search(line) or _STEP_RE.search(line)
        if match is None:
            return
        groups = match.groups()
        if len(groups) == 3:
            step, total, loss = int(groups[0]), int(groups[1]), float(groups[2])
            self._total = total or self._total
        else:
            step, loss = int(groups[0]), float(groups[1])
        # The trainer reports the same step twice (its progress line and its log
        # line); the second copy is dropped so the live stream stays readable.
        if self._last == (step, loss):
            return
        self._last = (step, loss)
        self._on_step(step, loss, self._total or step)


def _parse_summary(lines: Iterable[str]) -> dict[str, Any]:
    """Read the trainer's final summary line into the numbers it reported."""
    for line in lines:
        match = _SUMMARY_RE.search(line)
        if match is None:
            continue
        (
            _steps, initial, final, val, best_val, best_step, params, tokens,
        ) = match.groups()
        best = float(best_val) if best_val is not None else float(val)
        return {
            "initial": float(initial),
            "final": float(final),
            "best": best,
            "best_step": int(best_step) if best_step is not None else None,
            "parameters": int(params),
            "tokens": int(tokens),
        }
    return {}


def _match_preset(config: Any) -> str:
    try:
        from ml.release import _match_preset as match

        return match(config) or ""
    except Exception:
        return ""


def language_model_engine(root: Path | str | None = None, **kwargs: Any) -> LanguageModelEngine:
    """An engine over the project's own data folder (or ``root``)."""
    return LanguageModelEngine(root, **kwargs)


__all__ = [
    "CORPUS_SOURCES",
    "LR_SCHEDULES",
    "PRESETS",
    "TOKENIZER_KINDS",
    "TRAIN_CHECKPOINT_NAME",
    "CheckpointReport",
    "EvaluationReport",
    "LanguageModelEngine",
    "LanguageModelSettings",
    "ReleaseOutcome",
    "TrainOutcome",
    "WeightActionReport",
    "language_model_engine",
]
