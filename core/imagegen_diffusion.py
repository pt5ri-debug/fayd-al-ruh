"""Optional learned text-to-image backend: a real diffusion model on CPU.

The default :class:`~core.imagegen.ProceduralImageGenerator` is honest but is a
*renderer*, not a model. This module adds the real thing: a Diffusers diffusion
pipeline (default ``stabilityai/sd-turbo``) that turns a prompt into a picture the
way a learned text-to-image model does. It is entirely optional and follows the
project's hard rules:

- **Optional extra.** ``torch``/``diffusers`` are imported lazily, only when a
  generation actually runs, so a base install keeps working and importing this
  module never pulls a heavy dependency.
- **Human approval.** Downloading multi-gigabyte weights is an external action
  that leaves the machine, so it runs only through the project's permission and
  human-approval gates (:mod:`core.permissions`, :mod:`core.human_approval`),
  denied by default, exactly like a web search. A cached model needs no download
  and so needs no approval.
- **Honesty.** Every :class:`~core.imagegen.GeneratedImage` it returns carries
  ``learned=True`` and a description that names the model and the step count. When
  the extra or the weights are missing it raises :class:`DiffusionUnavailable`
  with a clear message, and the caller can fall back to the procedural renderer.
- **One home.** The weights live under the writable data folder
  (:func:`core.paths.models_dir`), git-ignored, never in the repository.

The heavy work is delegated to a :class:`DiffusionPipelineRunner`; the default
runner is a thin Diffusers wrapper, and a test can supply its own runner without
installing torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.human_approval import HumanApprovalGate
from core.imagegen import (
    GeneratedImage,
    ImageRequest,
    encode_png,
)
from core.paths import models_dir
from core.permissions import Capability, PermissionPolicy

#: The default model: SD 2.1-turbo, a 1-4 step model small enough for CPU use.
DEFAULT_MODEL = "stabilityai/sd-turbo"
#: The external-tool name the permission/approval gates see for a download.
DOWNLOAD_TOOL = "image_model_download"
#: A safe ceiling for a single generation, so a mistyped request cannot hang.
MAX_STEPS = 50


class DiffusionUnavailable(RuntimeError):
    """Raised when the learned backend cannot run (no extra, no weights, no GPU)."""


@runtime_checkable
class DiffusionPipelineRunner(Protocol):
    """The narrow seam a real Diffusers pipeline (or a test) implements."""

    def __call__(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> Sequence[bytes]:
        """Render ``prompt`` and return one RGB row (``width * 3`` bytes) per row."""
        ...


def _require_extra() -> tuple[Any, Any]:
    """Import Diffusers lazily; raise :class:`DiffusionUnavailable` when absent."""
    try:
        import diffusers  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the install
        raise DiffusionUnavailable(
            "The learned text-to-image backend needs the optional image extra. "
            "Install it with:  pip install .[image]"
        ) from error
    return diffusers, torch


def _cache_dir(root: Path | None) -> Path:
    return Path(root) if root is not None else models_dir()


def _model_cache_name(model: str) -> str:
    """A filesystem-safe cache folder name for a HuggingFace repo id."""
    return model.replace("/", "__")


def model_is_cached(model: str, *, root: Path | None = None) -> bool:
    """True when the model's weights are already on disk (so no download is needed)."""
    directory = _cache_dir(root) / _model_cache_name(model)
    if not directory.is_dir():
        return False
    # ``model_index.json`` is written by ``save_pretrained``; a partial download
    # that never finished leaves the folder without it.
    return (directory / "model_index.json").is_file()


def download_model(
    model: str = DEFAULT_MODEL,
    *,
    root: Path | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download and cache the model weights, through the permission/approval gates.

    The download leaves the machine, so it is treated as an external tool: denied
    by default, and it runs only when an approving :class:`PermissionPolicy` *and*
    a live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the
    CLI's convenience that records a human approval). Returns the cache directory.
    """
    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({DOWNLOAD_TOOL}),
        )
        approvals = approvals or _approve(DOWNLOAD_TOOL)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, DOWNLOAD_TOOL)
    if not decision.allowed:
        raise DiffusionUnavailable(
            f"Downloading {model!r} is denied by policy ({decision.reason})."
        )
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise DiffusionUnavailable(
            f"Downloading {model!r} needs human approval ({approval.reason})."
        )

    target = _cache_dir(root) / _model_cache_name(model)
    if model_is_cached(model, root=root):
        return target
    diffusers, _torch = _require_extra()
    if progress is not None:
        progress(f"Downloading {model} to {target} (this is a one-time ~2.5 GB download)…")
    target.mkdir(parents=True, exist_ok=True)
    try:
        pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(model)
        pipeline.save_pretrained(str(target))
    except Exception as error:  # noqa: BLE001 - surfaced as an honest message
        raise DiffusionUnavailable(f"Could not download {model!r}: {error}") from error
    if progress is not None:
        progress(f"Saved {model} to {target}.")
    return target


def _approve(tool: str) -> HumanApprovalGate:
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


def _default_runner(model_dir: str) -> DiffusionPipelineRunner:
    """Build the real Diffusers runner for a cached model directory."""

    def run(
        model: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> Sequence[bytes]:
        diffusers, torch = _require_extra()
        pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(model_dir)
        pipeline.set_progress_bar_config(disable=True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "generator": generator,
            "width": width,
            "height": height,
        }
        # Turbo models run with guidance off; a normal model uses the guidance.
        if guidance > 0:
            kwargs["guidance_scale"] = guidance
        result = pipeline(**kwargs)
        image = result.images[0].convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height))
        pixels = image.tobytes()
        return [pixels[offset : offset + width * 3] for offset in range(0, len(pixels), width * 3)]

    return run


class DiffusionImageGenerator:
    """A learned text-to-image backend behind the :class:`ImageGenerator` protocol."""

    name = "diffusion"
    learned = True

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        steps: int = 1,
        guidance: float = 0.0,
        root: Path | str | None = None,
        runner: DiffusionPipelineRunner | None = None,
        require_cached: bool = True,
    ) -> None:
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be in [1, {MAX_STEPS}].")
        if guidance < 0:
            raise ValueError("guidance cannot be negative.")
        self.model = model
        self.steps = steps
        self.guidance = guidance
        self._root = Path(root) if root is not None else None
        self._runner = runner
        self._require_cached = require_cached

    @property
    def model_dir(self) -> Path:
        return _cache_dir(self._root) / _model_cache_name(self.model)

    def is_available(self) -> bool:
        """True when the extra is installed and the weights are cached locally."""
        if self._runner is not None:
            return True
        try:
            _require_extra()
        except DiffusionUnavailable:
            return False
        return model_is_cached(self.model, root=self._root)

    def generate(self, request: ImageRequest) -> GeneratedImage:
        if self._runner is not None:
            rows = self._runner(
                self.model,
                request.prompt,
                request.width,
                request.height,
                self.steps,
                self.guidance,
                request.seed if request.seed is not None else _prompt_seed(request.prompt),
            )
            return self._to_image(request, rows)

        _require_extra()
        if self._require_cached and not model_is_cached(self.model, root=self._root):
            raise DiffusionUnavailable(
                f"The model {self.model!r} is not cached yet. Download it first "
                f"(python -m cli image download --approve)."
            )
        runner = _default_runner(str(self.model_dir))
        seed = request.seed if request.seed is not None else _prompt_seed(request.prompt)
        rows = runner(
            self.model, request.prompt, request.width, request.height, self.steps, self.guidance, seed
        )
        return self._to_image(request, rows)

    def _to_image(self, request: ImageRequest, rows: Sequence[bytes]) -> GeneratedImage:
        png = encode_png(request.width, request.height, rows)
        seed = request.seed if request.seed is not None else _prompt_seed(request.prompt)
        return GeneratedImage(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            png=png,
            backend=self.name,
            seed=seed,
            learned=True,
            description=(
                f"Learned text-to-image image {request.width}x{request.height} from the "
                f"diffusion model {self.model!r} ({self.steps} step(s), guidance "
                f"{self.guidance:g}, seed {seed})."
            ),
        )


def _prompt_seed(prompt: str) -> int:
    from core.imagegen import _seed_for

    return _seed_for(prompt, None)


def diffusion_available(
    model: str = DEFAULT_MODEL, *, root: Path | str | None = None
) -> bool:
    """True when the learned backend can run *now* (extra installed, weights cached)."""
    generator = DiffusionImageGenerator(model=model, root=root)
    return generator.is_available()


__all__ = [
    "DEFAULT_MODEL",
    "DOWNLOAD_TOOL",
    "MAX_STEPS",
    "DiffusionImageGenerator",
    "DiffusionPipelineRunner",
    "DiffusionUnavailable",
    "diffusion_available",
    "download_model",
    "model_is_cached",
]
