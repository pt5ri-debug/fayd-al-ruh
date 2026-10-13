"""Optional learned text-to-image backend: a real diffusion model, tuned for
photorealistic, clean pictures.

The default :class:`~core.imagegen.ProceduralImageGenerator` is honest but is a
*renderer*, not a model. This module adds the real thing: a Diffusers diffusion
pipeline that turns a prompt into a picture the way a learned text-to-image model
does. The goal here is *photographic realism*, not a sketch: the default preset
uses a photorealistic SD 1.5 fine-tune (``Lykon/dreamshaper-8-lcm``), a negative
prompt that suppresses the usual "looks drawn" artefacts, and a real sampler
setting (LCM, guidance 1.5) rather than a single no-guidance step. A
:class:`~core.imagegen.ImageRequest` with a size is snapped to the model's 64px
grid and the result is resampled back with a high-quality filter, so the output
is crisp rather than blocky.

It is entirely optional and follows the project's hard rules:

- **Optional extra.** ``torch``/``diffusers`` are imported lazily, only when a
  generation actually runs, so a base install keeps working and importing this
  module never pulls a heavy dependency.
- **Human approval.** Downloading multi-gigabyte weights is an external action
  that leaves the machine, so it runs only through the project's permission and
  human-approval gates (:mod:`core.permissions`, :mod:`core.human_approval`),
  denied by default, exactly like a web search. A cached model needs no download
  and so needs no approval.
- **Honesty.** Every :class:`~core.imagegen.GeneratedImage` it returns carries
  ``learned=True`` and a description that names the model, the preset, and the
  step count. When the extra or the weights are missing it raises
  :class:`DiffusionUnavailable` with a clear message, and the caller can fall back
  to the procedural renderer.
- **One home.** The weights live under the writable data folder
  (:func:`core.paths.models_dir`), git-ignored, never in the repository.

Realism has a cost on CPU: the default preset is small (an SD 1.5 model, ~4 GB)
but a full-quality generation is tens of seconds to a few minutes per picture.
The ``fast`` preset trades realism for speed. A CUDA build of torch is much
faster; the project installs the CPU wheel by default so a small model needs no
GPU.

The heavy work is delegated to a :class:`DiffusionPipelineRunner`; the default
runner is a thin Diffusers wrapper, and a test can supply its own runner without
installing torch.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.human_approval import HumanApprovalGate
from core.imagegen import (
    GeneratedImage,
    ImageRequest,
    encode_png,
)
from core.optional_deps import FEATURE_IMAGE, pip_install_command
from core.paths import models_dir
from core.permissions import Capability, PermissionPolicy

#: The external-tool name the permission/approval gates see for a download.
DOWNLOAD_TOOL = "image_model_download"
#: A safe ceiling for a single generation, so a mistyped request cannot hang.
MAX_STEPS = 50
#: Diffusion needs a multiple of this; snapping avoids the tell-tale soft/striped
#: edges a non-multiple size produces.
BASE_RESOLUTION = 64

#: A negative prompt that suppresses the usual diffusion artefacts (the "looks
#: like a drawing" tells) so the default output reads as a photograph.
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, "
    "low quality, blurry, grainy, noisy, jpeg artifacts, deformed, disfigured, "
    "bad anatomy, extra limbs, watermark, text, signature, oversaturated"
)


class DiffusionUnavailable(RuntimeError):
    """Raised when the learned backend cannot run (no extra, no weights, no GPU)."""


@dataclass(frozen=True)
class ImageModelPreset:
    """A named quality/speed trade-off for the learned backend.

    A preset bundles the model id, the sampler settings, and a negative prompt,
    so "photorealistic and clean" is the *default* rather than something the user
    has to assemble from flags.
    """

    name: str
    title: str
    model: str
    steps: int
    guidance: float
    negative_prompt: str
    summary: str
    #: The sampler to install, e.g. ``"lcm"`` for an LCM model (empty = default).
    scheduler: str = ""


#: The presets, keyed by name. ``realistic`` is the default: photorealistic.
PRESETS: dict[str, ImageModelPreset] = {
    "fast": ImageModelPreset(
        name="fast",
        title="Fast",
        model="stabilityai/sd-turbo",
        steps=2,
        guidance=0.0,
        negative_prompt="",
        summary="Fastest: SD 2.1-turbo, a couple of steps. Prefer speed over detail.",
    ),
    "realistic": ImageModelPreset(
        name="realistic",
        title="Photorealistic (default)",
        model="Lykon/dreamshaper-8-lcm",
        steps=6,
        guidance=1.5,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        summary="Photorealistic: a fine-tuned SD 1.5, LCM-sampled in a few steps.",
        scheduler="lcm",
    ),
    "detailed": ImageModelPreset(
        name="detailed",
        title="Most detailed",
        model="Lykon/dreamshaper-8",
        steps=30,
        guidance=7.0,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        summary="Highest detail: full 30-step sampling. Slow on CPU.",
    ),
}

#: The default preset: photorealistic.
DEFAULT_PRESET = "realistic"
#: The default model, taken from the default preset.
DEFAULT_MODEL = PRESETS[DEFAULT_PRESET].model


def resolve_preset(name: str | None) -> ImageModelPreset:
    """Return the named preset, or the default when ``name`` is ``None``/empty."""
    key = (name or DEFAULT_PRESET).strip().lower()
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(
            f"Unknown image preset {name!r}; choose one of {', '.join(PRESETS)}."
        ) from None


def preset_names() -> tuple[str, ...]:
    """The preset names, in a stable order (``fast`` first, the default flagged)."""
    return tuple(PRESETS)


def preset_summaries() -> tuple[tuple[str, str, str], ...]:
    """``(name, model, summary)`` per preset, for a menu or a status line."""
    return tuple((p.name, p.model, p.summary) for p in PRESETS.values())


@runtime_checkable
class DiffusionPipelineRunner(Protocol):
    """The narrow seam a real Diffusers pipeline (or a test) implements."""

    def __call__(
        self,
        model: str,
        prompt: str,
        negative_prompt: str,
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
            f"Install it with:  {pip_install_command(FEATURE_IMAGE)}"
        ) from error
    return diffusers, torch


def _cache_dir(root: Path | None) -> Path:
    return Path(root) if root is not None else models_dir()


def _model_cache_name(model: str) -> str:
    """A filesystem-safe cache folder name for a HuggingFace repo id."""
    return model.replace("/", "__")


def _snap(value: int, grid: int = BASE_RESOLUTION) -> int:
    """Round ``value`` to the nearest multiple of ``grid`` (at least one grid)."""
    return max(grid, int(round(value / grid)) * grid)


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
        progress(f"Downloading {model} to {target} (this is a one-time ~4 GB download)…")
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


def _accepts(pipeline: Any, name: str) -> bool:
    """True when a pipeline's ``__call__`` takes ``name`` (older/newer differ)."""
    try:
        return name in set(inspect.signature(pipeline.__call__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False


def _load_pipeline(diffusers: Any, model_dir: str, torch: Any) -> Any:
    """Load a text-to-image pipeline in float32, tolerating the dtype rename.

    Diffusers renamed ``torch_dtype`` to ``dtype`` (1.0); try the new name first
    and fall back, so the backend works across versions.
    """
    for keyword in ("dtype", "torch_dtype"):
        try:
            return diffusers.AutoPipelineForText2Image.from_pretrained(
                model_dir, **{keyword: torch.float32}
            )
        except TypeError:
            continue
    return diffusers.AutoPipelineForText2Image.from_pretrained(model_dir)


def _install_scheduler(pipeline: Any, diffusers: Any, name: str) -> None:
    """Swap in the sampler a preset needs (e.g. LCM for an LCM model)."""
    if not name:
        return
    if name == "lcm" and hasattr(diffusers, "LCMScheduler"):
        pipeline.scheduler = diffusers.LCMScheduler.from_config(pipeline.scheduler.config)


def _default_runner(model_dir: str, scheduler: str = "") -> DiffusionPipelineRunner:
    """Build the real Diffusers runner for a cached model directory."""

    def run(
        model: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> Sequence[bytes]:
        diffusers, torch = _require_extra()
        pipeline = _load_pipeline(diffusers, model_dir, torch)
        _install_scheduler(pipeline, diffusers, scheduler)
        pipeline.set_progress_bar_config(disable=True)
        # CPU memory: attention slicing trades a little speed for a lot of headroom,
        # so a photorealistic model fits without exhausting RAM.
        try:
            pipeline.enable_attention_slicing()
        except Exception:  # pragma: no cover - pipelines differ across versions
            pass
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Snap to the model's grid for crisp output, then resample back to the
        # requested size so the returned rows match the request exactly.
        gen_width, gen_height = _snap(width), _snap(height)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "generator": generator,
            "width": gen_width,
            "height": gen_height,
        }
        # Turbo models run with guidance off; a normal model uses the guidance.
        if guidance > 0:
            kwargs["guidance_scale"] = guidance
        if negative_prompt and _accepts(pipeline, "negative_prompt"):
            kwargs["negative_prompt"] = negative_prompt
        result = pipeline(**kwargs)
        image = result.images[0].convert("RGB")
        if image.size != (width, height):
            from PIL import Image as _PILImage

            image = image.resize((width, height), _PILImage.Resampling.LANCZOS)
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
        model: str | None = None,
        preset: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        root: Path | str | None = None,
        runner: DiffusionPipelineRunner | None = None,
        require_cached: bool = True,
    ) -> None:
        self.preset = resolve_preset(preset)
        self.model = model or self.preset.model
        self.steps = self.preset.steps if steps is None else steps
        self.guidance = self.preset.guidance if guidance is None else guidance
        self.negative_prompt = (
            self.preset.negative_prompt if negative_prompt is None else negative_prompt
        )
        if not 1 <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be in [1, {MAX_STEPS}].")
        if self.guidance < 0:
            raise ValueError("guidance cannot be negative.")
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
        seed = request.seed if request.seed is not None else _prompt_seed(request.prompt)
        negative = (
            self.negative_prompt
            if request.negative_prompt is None
            else request.negative_prompt
        )
        if self._runner is not None:
            rows = self._runner(
                self.model,
                request.prompt,
                negative,
                request.width,
                request.height,
                self.steps,
                self.guidance,
                seed,
            )
            return self._to_image(request, rows)

        _require_extra()
        if self._require_cached and not model_is_cached(self.model, root=self._root):
            raise DiffusionUnavailable(
                f"The model {self.model!r} is not cached yet. Download it first "
                f"(python -m cli image download --approve)."
            )
        runner = _default_runner(str(self.model_dir), self.preset.scheduler)
        rows = runner(
            self.model,
            request.prompt,
            negative,
            request.width,
            request.height,
            self.steps,
            self.guidance,
            seed,
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
                f"Learned photorealistic image {request.width}x{request.height} from the "
                f"diffusion model {self.model!r} (preset {self.preset.name!r}, "
                f"{self.steps} step(s), guidance {self.guidance:g}, seed {seed})."
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
    "BASE_RESOLUTION",
    "DEFAULT_MODEL",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_PRESET",
    "DOWNLOAD_TOOL",
    "MAX_STEPS",
    "PRESETS",
    "DiffusionImageGenerator",
    "DiffusionPipelineRunner",
    "DiffusionUnavailable",
    "ImageModelPreset",
    "diffusion_available",
    "download_model",
    "model_is_cached",
    "preset_names",
    "preset_summaries",
    "resolve_preset",
]
