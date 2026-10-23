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

from core.best_effort import best_effort
from core.human_approval import HumanApprovalGate
from core.imageedit import ImageEditError, Raster
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


#: The instruction-editing model (InstructPix2Pix): it changes an existing picture
#: from a written instruction ("make him smile", "change the pose to a wave"), the
#: way the presets above turn a prompt into a *new* picture.
EDIT_MODEL = "timbrooks/instruct-pix2pix"
#: The default sampler settings for the editing model. InstructPix2Pix follows the
#: instruction through ``guidance_scale`` and how far it may move from the input
#: through ``image_guidance_scale`` (lower = a bigger change). Twelve steps keeps a
#: CPU edit to a few minutes at the 512px cap rather than the ten a 20-step run takes.
DEFAULT_EDIT_STEPS = 12
DEFAULT_EDIT_GUIDANCE = 7.5
DEFAULT_EDIT_IMAGE_GUIDANCE = 1.5
#: A negative prompt for the editing model, to hold the subject together.
DEFAULT_EDIT_NEGATIVE_PROMPT = (
    "deformed, disfigured, bad anatomy, extra limbs, extra fingers, blurry, "
    "low quality, jpeg artifacts, watermark, text"
)


@dataclass(frozen=True)
class ImageEditPreset:
    """A named way to steer the instruction-editing model.

    An edit has two dials: how strongly the model obeys the *text* instruction
    (``guidance``) and how far it may move from the *input image*
    (``image_guidance``; lower means a bigger change, so a whole new pose or a
    large change in detail, higher means a light touch). A preset bundles both so
    "faithful light edit" and "change the movement" are one choice, not two flags.
    """

    name: str
    title: str
    guidance: float
    image_guidance: float
    summary: str


#: The editing presets, keyed by name. ``balanced`` is the default.
EDIT_PRESETS: dict[str, ImageEditPreset] = {
    "subtle": ImageEditPreset(
        name="subtle",
        title="Subtle (keep the image close)",
        guidance=7.5,
        image_guidance=2.0,
        summary="A light change: expression, colour, small details. Stays close to your picture.",
    ),
    "balanced": ImageEditPreset(
        name="balanced",
        title="Balanced (default)",
        guidance=7.5,
        image_guidance=1.5,
        summary="A clear change: make someone smile, change an object, restyle a region.",
    ),
    "strong": ImageEditPreset(
        name="strong",
        title="Strong (change the pose or movement)",
        guidance=8.5,
        image_guidance=1.0,
        summary="A large change: a new pose, a different movement, a big change in detail.",
    ),
}

#: The default editing preset.
DEFAULT_EDIT_PRESET = "balanced"


def resolve_edit_preset(name: str | None) -> ImageEditPreset:
    """Return the named editing preset, or the default when ``name`` is empty."""
    key = (name or DEFAULT_EDIT_PRESET).strip().lower()
    try:
        return EDIT_PRESETS[key]
    except KeyError:
        raise ValueError(
            f"Unknown edit preset {name!r}; choose one of {', '.join(EDIT_PRESETS)}."
        ) from None


def edit_preset_names() -> tuple[str, ...]:
    """The editing preset names, in a stable order."""
    return tuple(EDIT_PRESETS)


def edit_preset_summaries() -> tuple[tuple[str, float, float, str], ...]:
    """``(name, guidance, image_guidance, summary)`` per editing preset."""
    return tuple(
        (p.name, p.guidance, p.image_guidance, p.summary) for p in EDIT_PRESETS.values()
    )


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
    return max(grid, round(value / grid) * grid)


def model_is_cached(model: str, *, root: Path | None = None) -> bool:
    """True when the model's weights are already on disk (so no download is needed)."""
    directory = _cache_dir(root) / _model_cache_name(model)
    if not directory.is_dir():
        return False
    # ``model_index.json`` is written by ``save_pretrained``; a partial download
    # that never finished leaves the folder without it.
    return (directory / "model_index.json").is_file()


def _pipeline_class(diffusers: Any, kind: str) -> Any:
    """The Diffusers pipeline class for a model kind.

    ``text2image`` turns a prompt into a new picture; ``instruct``
    (InstructPix2Pix) changes an existing picture from an instruction.
    InstructPix2Pix has no ``AutoPipelineFor*`` mapping, so its concrete class is
    used directly.
    """
    if kind == "instruct":
        return diffusers.StableDiffusionInstructPix2PixPipeline
    return diffusers.AutoPipelineForText2Image


def download_model(
    model: str = DEFAULT_MODEL,
    *,
    root: Path | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Callable[[str], None] | None = None,
    kind: str = "text2image",
) -> Path:
    """Download and cache the model weights, through the permission/approval gates.

    The download leaves the machine, so it is treated as an external tool: denied
    by default, and it runs only when an approving :class:`PermissionPolicy` *and*
    a live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the
    CLI's convenience that records a human approval). Returns the cache directory.
    ``kind`` selects the pipeline family (``text2image`` or ``instruct``).
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
        factory = _pipeline_class(diffusers, kind)
        pipeline = factory.from_pretrained(model)
        pipeline.save_pretrained(str(target))
    except Exception as error:
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

    ``safety_checker=None`` mirrors :func:`_load_edit_pipeline`: the checker blanks
    a picture it misreads (an abstract or stylised prompt is a common false
    positive) to a solid black frame instead of drawing it. The default
    ``realistic`` preset's model ships a real checker, so without this the window's
    preview shows black. Diffusers renamed ``torch_dtype`` to ``dtype`` (1.0); try
    the new name first and fall back, so the backend works across versions.
    """
    for keyword in ("dtype", "torch_dtype"):
        try:
            return diffusers.AutoPipelineForText2Image.from_pretrained(
                model_dir, safety_checker=None, **{keyword: torch.float32}
            )
        except TypeError:
            continue
    return diffusers.AutoPipelineForText2Image.from_pretrained(
        model_dir, safety_checker=None
    )


def _install_scheduler(pipeline: Any, diffusers: Any, name: str) -> None:
    """Swap in the sampler a preset needs (e.g. LCM for an LCM model)."""
    if not name:
        return
    if name == "lcm" and hasattr(diffusers, "LCMScheduler"):
        pipeline.scheduler = diffusers.LCMScheduler.from_config(pipeline.scheduler.config)


def _load_edit_pipeline(diffusers: Any, model_dir: str, torch: Any) -> Any:
    """Load an InstructPix2Pix pipeline in float32, tolerating the dtype rename.

    ``safety_checker=None`` follows the model's own documentation: the checker
    blanks a picture it misreads (a stylised or abstract input is a common false
    positive) instead of editing it, which would silently destroy the user's image.
    The editor only ever transforms an image the user supplied.
    """
    for keyword in ("dtype", "torch_dtype"):
        try:
            return diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_dir, safety_checker=None, **{keyword: torch.float32}
            )
        except TypeError:
            continue
    return diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_dir, safety_checker=None
    )


def _work_size(width: int, height: int, *, cap: int = 512, grid: int = 64) -> tuple[int, int]:
    """The resolution the model runs at: the input's, capped, snapped to the grid.

    InstructPix2Pix is trained at 512px; running a large photo at full size would
    exhaust CPU memory and take many minutes for no gain, so the longest side is
    capped (the result is resized back to the input's size).
    """
    scale = min(1.0, cap / max(width, height, 1))
    work_w = _snap(max(grid, round(width * scale)), grid)
    work_h = _snap(max(grid, round(height * scale)), grid)
    return work_w, work_h


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
        with best_effort("enable diffusion attention slicing"):
            pipeline.enable_attention_slicing()
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


@runtime_checkable
class ImageEditRunner(Protocol):
    """The narrow seam a real InstructPix2Pix pipeline (or a test) implements.

    Unlike :class:`DiffusionPipelineRunner`, it receives the *source* image's RGB
    rows and returns the edited image's rows, so the model changes a picture the
    user already has rather than inventing one from a prompt.
    """

    def __call__(
        self,
        model: str,
        image: Sequence[bytes],
        instruction: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        image_guidance: float,
        seed: int,
    ) -> Sequence[bytes]:
        """Edit ``image`` (one RGB row per entry) and return the result's rows."""
        ...


def _default_edit_runner(model_dir: str) -> ImageEditRunner:
    """Build the real Diffusers runner for a cached InstructPix2Pix model."""

    def run(
        model: str,
        image: Sequence[bytes],
        instruction: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        image_guidance: float,
        seed: int,
    ) -> Sequence[bytes]:
        from PIL import Image as _PILImage

        diffusers, torch = _require_extra()
        pipeline = _load_edit_pipeline(diffusers, model_dir, torch)
        pipeline.set_progress_bar_config(disable=True)
        with best_effort("enable diffusion attention slicing"):
            pipeline.enable_attention_slicing()
        source = _PILImage.frombytes("RGB", (width, height), b"".join(bytes(row) for row in image))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        work_w, work_h = _work_size(width, height)
        if (work_w, work_h) != (width, height):
            source = source.resize((work_w, work_h), _PILImage.Resampling.LANCZOS)
        result = pipeline(
            instruction,
            image=source,
            num_inference_steps=steps,
            guidance_scale=guidance,
            image_guidance_scale=image_guidance,
            negative_prompt=negative_prompt or None,
            generator=generator,
        )
        edited = result.images[0].convert("RGB")
        if edited.size != (width, height):
            edited = edited.resize((width, height), _PILImage.Resampling.LANCZOS)
        pixels = edited.tobytes()
        return [
            pixels[offset : offset + width * 3]
            for offset in range(0, len(pixels), width * 3)
        ]

    return run


class InstructionImageEditor:
    """A learned *image-editing* backend: change a picture from a written instruction.

    This is the counterpart of :class:`DiffusionImageGenerator`. It runs an
    InstructPix2Pix model, which takes an image and an instruction ("make him
    smile", "change the pose to a wave") and re-synthesises the picture. It is the
    only way to make a *semantic* change -- an expression, a pose, the movement --
    rather than the deterministic pixel operations in :mod:`core.imageedit`.
    """

    name = "instruct"
    learned = True

    def __init__(
        self,
        *,
        model: str | None = None,
        preset: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        image_guidance: float | None = None,
        negative_prompt: str | None = None,
        root: Path | str | None = None,
        runner: ImageEditRunner | None = None,
        require_cached: bool = True,
    ) -> None:
        self.preset = resolve_edit_preset(preset)
        self.model = model or EDIT_MODEL
        self.steps = DEFAULT_EDIT_STEPS if steps is None else steps
        self.guidance = self.preset.guidance if guidance is None else guidance
        self.image_guidance = (
            self.preset.image_guidance if image_guidance is None else image_guidance
        )
        self.negative_prompt = (
            DEFAULT_EDIT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt
        )
        if not 1 <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be in [1, {MAX_STEPS}].")
        if self.guidance < 0 or self.image_guidance < 0:
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

    def edit(
        self,
        raster: Raster,
        instruction: str,
        *,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> Raster:
        """Apply ``instruction`` to ``raster`` and return the edited image.

        Raises :class:`~core.imageedit.ImageEditError` for an empty instruction,
        and :class:`DiffusionUnavailable` when the model cannot run.
        """
        text = (instruction or "").strip()
        if not text:
            raise ImageEditError("Write an instruction first.")
        width, height = raster.width, raster.height
        chosen_seed = seed if seed is not None else _prompt_seed(text)
        negative = self.negative_prompt if negative_prompt is None else negative_prompt
        rows = _rows_from_raster(raster)

        if self._runner is not None:
            out = self._runner(
                self.model, rows, text, negative, width, height,
                self.steps, self.guidance, self.image_guidance, chosen_seed,
            )
            return _raster_from_rows(width, height, out)

        _require_extra()
        if self._require_cached and not model_is_cached(self.model, root=self._root):
            raise DiffusionUnavailable(
                f"The editing model {self.model!r} is not cached yet. Download it "
                "first (python -m cli image edit-model --approve)."
            )
        runner = _default_edit_runner(str(self.model_dir))
        out = runner(
            self.model, rows, text, negative, width, height,
            self.steps, self.guidance, self.image_guidance, chosen_seed,
        )
        return _raster_from_rows(width, height, out)

    def describe(self, instruction: str) -> str:
        """An honest one-line description of what produced (or would produce) an edit."""
        return (
            f"Edited with the learned model {self.model!r} (preset {self.preset.name!r}, "
            f"{self.steps} step(s), guidance {self.guidance:g}, "
            f"image guidance {self.image_guidance:g})."
        )


def _rows_from_raster(raster: Raster) -> list[bytes]:
    """The raster's RGB rows, the shape a diffusion runner expects."""
    stride = raster.width * 3
    data = bytes(raster.data)
    return [data[offset : offset + stride] for offset in range(0, len(data), stride)]


def _raster_from_rows(width: int, height: int, rows: Sequence[bytes]) -> Raster:
    """Build a raster from a runner's rows, checking the shape so a bad runner fails."""
    data = bytearray()
    for row in rows:
        data.extend(bytes(row))
    if len(data) != width * height * 3:
        raise ImageEditError(
            f"The editing model returned {len(data)} bytes, expected {width * height * 3}."
        )
    return Raster(width, height, data)


def instruction_editor_available(
    model: str = EDIT_MODEL, *, root: Path | str | None = None
) -> bool:
    """True when the editing backend can run *now* (extra installed, weights cached)."""
    editor = InstructionImageEditor(model=model, root=root)
    return editor.is_available()


def download_edit_model(
    model: str = EDIT_MODEL,
    *,
    root: Path | None = None,
    approved: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download the instruction-editing model's weights (external action; needs approval)."""
    return download_model(
        model, root=root, approved=approved, progress=progress, kind="instruct"
    )


def diffusion_available(
    model: str = DEFAULT_MODEL, *, root: Path | str | None = None
) -> bool:
    """True when the learned backend can run *now* (extra installed, weights cached)."""
    generator = DiffusionImageGenerator(model=model, root=root)
    return generator.is_available()


__all__ = [
    "BASE_RESOLUTION",
    "DEFAULT_EDIT_GUIDANCE",
    "DEFAULT_EDIT_IMAGE_GUIDANCE",
    "DEFAULT_EDIT_NEGATIVE_PROMPT",
    "DEFAULT_EDIT_PRESET",
    "DEFAULT_EDIT_STEPS",
    "DEFAULT_MODEL",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_PRESET",
    "DOWNLOAD_TOOL",
    "EDIT_MODEL",
    "EDIT_PRESETS",
    "MAX_STEPS",
    "PRESETS",
    "DiffusionImageGenerator",
    "DiffusionPipelineRunner",
    "DiffusionUnavailable",
    "ImageEditPreset",
    "ImageEditRunner",
    "ImageModelPreset",
    "InstructionImageEditor",
    "diffusion_available",
    "download_edit_model",
    "download_model",
    "edit_preset_names",
    "edit_preset_summaries",
    "instruction_editor_available",
    "model_is_cached",
    "preset_names",
    "preset_summaries",
    "resolve_edit_preset",
    "resolve_preset",
]
