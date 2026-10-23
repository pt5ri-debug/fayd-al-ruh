"""Text-to-image: the generator, the service, and the CLI (no display needed).

These drive the real code paths -- the dependency-free PNG encoder, the
procedural renderer, the service's prompt parsing and saving, and the CLI's
``image generate`` -- with no mocks. Every test names the property it pins:
determinism, honest labelling, bounded sizes, and file output that is a valid
PNG.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
from cli.image import main as image_main
from core.imagegen import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_DIMENSION,
    ImageRequest,
    ProceduralImageGenerator,
    default_image_dir,
    encode_png,
    image_filename,
    load_image_generator,
)
from gui.image_gen import ImageGenerationService, parse_size

# ---------------------------------------------------------------------------
# PNG encoder
# ---------------------------------------------------------------------------


def test_encode_png_is_a_valid_png_with_the_right_dimensions():
    rows = [bytes([(x * 7) % 256 for x in range(4 * 3)]) for _ in range(4)]
    png = encode_png(4, 4, rows)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # First chunk after the signature is IHDR; parse width/height out of it.
    length = struct.unpack(">I", png[8:12])[0]
    assert png[12:16] == b"IHDR"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (4, 4)
    assert length == 13


def test_encode_png_round_trips_the_pixels():
    """Decompress the IDAT and check the filter byte and the exact RGB bytes."""
    rows = [bytes([1, 2, 3, 4, 5, 6]), bytes([7, 8, 9, 10, 11, 12])]
    png = encode_png(2, 2, rows)
    # Walk the chunks to find IDAT, then inflate it.
    index = 8
    idat = b""
    while index < len(png):
        length = struct.unpack(">I", png[index : index + 4])[0]
        tag = png[index + 4 : index + 8]
        data = png[index + 8 : index + 8 + length]
        if tag == b"IDAT":
            idat += data
        index += 12 + length
    raw = zlib.decompress(idat)
    assert raw == b"\x00" + rows[0] + b"\x00" + rows[1]


def test_encode_png_rejects_wrong_row_length_and_count():
    with pytest.raises(ValueError):
        encode_png(2, 1, [b"\x00\x00\x00"])  # 1 byte per row for width 2
    with pytest.raises(ValueError):
        encode_png(1, 2, [b"\x00\x00\x00"])  # only one row for height 2


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------


def test_request_rejects_empty_prompt_and_out_of_range_size():
    with pytest.raises(ValueError):
        ImageRequest(prompt="   ")
    with pytest.raises(ValueError):
        ImageRequest(prompt="ok", width=0)
    with pytest.raises(ValueError):
        ImageRequest(prompt="ok", height=MAX_DIMENSION + 1)


# ---------------------------------------------------------------------------
# procedural renderer
# ---------------------------------------------------------------------------


def test_same_prompt_and_seed_is_identical():
    generator = ProceduralImageGenerator()
    first = generator.generate(ImageRequest("a red spiral", width=32, height=24))
    second = generator.generate(ImageRequest("a red spiral", width=32, height=24))
    assert first.png == second.png
    assert first.seed == second.seed


def test_different_prompt_or_seed_changes_the_image():
    generator = ProceduralImageGenerator()
    base = generator.generate(ImageRequest("a red spiral", width=32, height=24, seed=1))
    other_prompt = generator.generate(ImageRequest("a blue spiral", width=32, height=24, seed=1))
    other_seed = generator.generate(ImageRequest("a red spiral", width=32, height=24, seed=2))
    assert base.png != other_prompt.png
    assert base.png != other_seed.png


def test_renderer_is_honest_about_being_procedural():
    image = ProceduralImageGenerator().generate(ImageRequest("anything"))
    assert image.learned is False
    assert "not a learned depiction" in image.description
    assert image.backend == "procedural"
    assert image.width == DEFAULT_WIDTH and image.height == DEFAULT_HEIGHT


def test_generated_image_saves_to_disk(tmp_path):
    image = ProceduralImageGenerator().generate(ImageRequest("save me", width=16, height=16))
    target = image.save(tmp_path / "nested" / "out.png")
    assert target.exists()
    assert target.read_bytes() == image.png
    assert image.size_bytes == len(image.png)


# ---------------------------------------------------------------------------
# backend registry
# ---------------------------------------------------------------------------


def test_load_generator_and_reject_unknown():
    assert load_image_generator("procedural").name == "procedural"
    with pytest.raises(ValueError):
        load_image_generator("stable-diffusion")


def test_default_image_dir_and_filename_are_stable(tmp_path):
    assert default_image_dir(tmp_path) == tmp_path
    name = image_filename("A prompt", 123)
    assert name.startswith("image-") and name.endswith(".png")
    assert image_filename("a PROMPT", 123) == name  # case-insensitive


# ---------------------------------------------------------------------------
# service: prompt parsing and saving
# ---------------------------------------------------------------------------


def test_parse_size_reads_a_trailing_dimension():
    assert parse_size("a red spiral 768x512") == ("a red spiral", 768, 512)
    assert parse_size("wide scene 1024×768") == ("wide scene", 1024, 768)
    assert parse_size("no size here") == ("no size here", DEFAULT_WIDTH, DEFAULT_HEIGHT)


def test_service_generates_and_saves_under_the_root(tmp_path):
    service = ImageGenerationService(root=tmp_path / "images")
    report = service.generate("a quiet sea 128x96")
    assert report.ok
    assert (report.width, report.height) == (128, 96)
    saved = Path(report.path)
    assert saved.exists() and saved.is_relative_to(tmp_path)
    assert saved.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert report.learned is False
    assert "Image saved to" in report.summary()


def test_service_honours_explicit_size_and_output(tmp_path):
    service = ImageGenerationService(root=tmp_path)
    target = tmp_path / "chosen.png"
    report = service.generate("prompt", width=40, height=20, output=target)
    assert report.ok and report.path == str(target)
    assert (report.width, report.height) == (40, 20)
    assert target.exists()


def test_service_reports_a_bad_request_without_raising(tmp_path):
    service = ImageGenerationService(root=tmp_path)
    report = service.generate("   ")
    assert report.ok is False
    assert "prompt" in report.message.casefold()
    assert report.summary().startswith("Image generation failed")


def test_service_reports_a_bad_size_without_raising(tmp_path):
    service = ImageGenerationService(root=tmp_path)
    report = service.generate("ok", width=0)
    assert report.ok is False
    assert "width" in report.message


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_generate_writes_a_file(tmp_path, capsys):
    code = image_main(
        ["--output-dir", str(tmp_path), "generate", "a test prompt 64x48"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Image saved to" in out
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 1
    assert files[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_generate_json_reports_honesty(tmp_path, capsys):
    import json

    code = image_main(["--output-dir", str(tmp_path), "generate", "p", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["learned"] is False
    assert payload["backend"] == "procedural"


def test_cli_generate_rejects_an_empty_prompt(tmp_path, capsys):
    code = image_main(["--output-dir", str(tmp_path), "generate", "   "])
    assert code == 1
    assert "prompt" in capsys.readouterr().out.casefold()


def test_cli_backends_lists_the_procedural_renderer(capsys):
    assert image_main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "procedural: available" in out
    assert "not a model" in out
    assert "diffusion:" in out


def test_cli_download_without_approve_is_refused(capsys):
    assert image_main(["download"]) == 1
    assert "--approve" in capsys.readouterr().out


def test_cli_seed_reproduces_the_same_file(tmp_path):
    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    image_main(["--output-dir", str(tmp_path), "generate", "same", "--seed", "9", "--output", str(first)])
    image_main(["--output-dir", str(tmp_path), "generate", "same", "--seed", "9", "--output", str(second)])
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# wiring: the GUI service and the CLI menu expose the feature
# ---------------------------------------------------------------------------


def test_chat_service_exposes_the_image_engine(tmp_path):
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path, image_store_dir=tmp_path / "images")
    assert service.image_gen.backend == "procedural"
    assert service.image_gen.learned is False


def test_cli_menu_and_dispatch_include_image():
    from cli.__main__ import _COMMANDS
    from cli.menu import MENU, visible_menu

    assert any(label == "Generate an image from text" for label, _action, _args in MENU)
    assert any(label == "Generate an image from text" for label, _a, _b in visible_menu())
    assert "image" in _COMMANDS


# ---------------------------------------------------------------------------
# optional learned backend (a fake runner stands in for Diffusers: no torch)
# ---------------------------------------------------------------------------


class _FakeRunner:
    """A stand-in diffusion runner: fills the image with one flat colour."""

    def __init__(self, colour: tuple[int, int, int] = (10, 20, 30)) -> None:
        self.colour = colour
        self.calls: list[tuple] = []

    def __call__(self, model, prompt, negative_prompt, width, height, steps, guidance, seed):
        self.calls.append(
            (model, prompt, negative_prompt, width, height, steps, guidance, seed)
        )
        row = bytes(self.colour) * width
        return [row for _ in range(height)]


def test_diffusion_generator_labels_output_as_learned(tmp_path):
    from core.imagegen_diffusion import DEFAULT_MODEL, DiffusionImageGenerator

    runner = _FakeRunner()
    generator = DiffusionImageGenerator(root=tmp_path, runner=runner, steps=4, guidance=0.0)
    image = generator.generate(ImageRequest("a drilling rig at dusk", width=8, height=6))
    assert image.learned is True
    assert image.backend == "diffusion"
    assert DEFAULT_MODEL in image.description
    assert "4 step" in image.description
    assert "realistic" in image.description  # the photorealistic preset is named
    assert image.width == 8 and image.height == 6
    # A valid PNG whose pixels are the runner's colour.
    assert image.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert runner.calls[0][1] == "a drilling rig at dusk"
    assert runner.calls[0][5] == 4


def test_realistic_preset_is_the_default_and_carries_a_negative_prompt():
    from core.imagegen_diffusion import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PRESET, PRESETS

    assert DEFAULT_PRESET == "realistic"
    preset = PRESETS[DEFAULT_PRESET]
    # Photorealistic by default: a real sampler, not one no-guidance step.
    assert preset.steps >= 4
    assert preset.guidance > 0
    assert preset.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert "cartoon" in preset.negative_prompt  # steers away from the "looks drawn" tells


def test_resolve_preset_rejects_an_unknown_name():
    import pytest
    from core.imagegen_diffusion import resolve_preset

    assert resolve_preset(None).name == "realistic"
    assert resolve_preset("FAST").name == "fast"
    with pytest.raises(ValueError):
        resolve_preset("nope")


def test_generator_uses_the_preset_and_forwards_the_negative_prompt(tmp_path):
    from core.imagegen_diffusion import DiffusionImageGenerator

    runner = _FakeRunner()
    generator = DiffusionImageGenerator(root=tmp_path, runner=runner, preset="detailed")
    generator.generate(ImageRequest("a portrait"))
    _model, _prompt, negative, _w, _h, steps, guidance, _seed = runner.calls[0]
    assert steps == 30 and guidance == 7.0
    assert "cartoon" in negative
    # An explicit negative prompt overrides the preset's default.
    generator.generate(ImageRequest("a portrait", negative_prompt="rain"))
    assert runner.calls[1][2] == "rain"


def test_explicit_settings_override_the_preset(tmp_path):
    from core.imagegen_diffusion import DiffusionImageGenerator

    generator = DiffusionImageGenerator(
        root=tmp_path, runner=_FakeRunner(), preset="realistic", steps=1, guidance=0.0
    )
    assert generator.steps == 1
    assert generator.guidance == 0.0


def test_preset_names_and_summaries_are_stable():
    from core.imagegen_diffusion import preset_names, preset_summaries

    assert preset_names() == ("fast", "realistic", "detailed")
    summaries = {name: model for name, model, _summary in preset_summaries()}
    assert summaries["realistic"] == "Lykon/dreamshaper-8-lcm"
    assert "photorealistic" in {
        name: summary for name, _model, summary in preset_summaries()
    }["realistic"].lower()


def test_the_realistic_preset_installs_an_lcm_scheduler():
    from core.imagegen_diffusion import PRESETS, _install_scheduler

    assert PRESETS["realistic"].scheduler == "lcm"

    class _Scheduler:
        def __init__(self) -> None:
            self.config = {"_class_name": "PNDMScheduler"}

    class _Pipeline:
        def __init__(self) -> None:
            self.scheduler = _Scheduler()

    class _LCM(_Scheduler):
        @classmethod
        def from_config(cls, config):
            instance = cls()
            instance.config = config
            return instance

    class _Diffusers:
        LCMScheduler = _LCM

    pipeline = _Pipeline()
    _install_scheduler(pipeline, _Diffusers, "lcm")
    assert isinstance(pipeline.scheduler, _LCM)

    # An empty scheduler name leaves the pipeline's own sampler alone.
    untouched = _Pipeline()
    original = untouched.scheduler
    _install_scheduler(untouched, _Diffusers, "")
    assert untouched.scheduler is original


def test_text_to_image_loader_disables_the_safety_checker():
    """The rendered pipeline is built with ``safety_checker=None``.

    The default ``realistic`` preset's model ships a real checker, which blanks a
    picture it misreads (a stylised prompt is a common false positive) to solid
    black. Loading it would make the window's preview black, so the loader must
    disable it -- and must keep doing so across the diffusers ``dtype`` rename.
    """
    from core.imagegen_diffusion import _load_pipeline

    calls: list[dict] = []

    class _Pipeline:
        pass

    class _Auto:
        @staticmethod
        def from_pretrained(model_dir, **kwargs):
            calls.append({"model_dir": model_dir, **kwargs})
            return _Pipeline()

    class _Diffusers:
        AutoPipelineForText2Image = _Auto

    class _Torch:
        float32 = "float32"

    _load_pipeline(_Diffusers, "some/model", _Torch)
    assert len(calls) == 1
    assert calls[0]["safety_checker"] is None
    assert calls[0]["dtype"] == "float32"

    # When the installed diffusers rejects ``dtype`` (the pre-1.0 name), the loader
    # falls back to ``torch_dtype``, still disabling the checker.
    calls.clear()

    class _OldAuto:
        @staticmethod
        def from_pretrained(model_dir, **kwargs):
            if "dtype" in kwargs:
                raise TypeError("unexpected keyword argument 'dtype'")
            calls.append({"model_dir": model_dir, **kwargs})
            return _Pipeline()

    class _OldDiffusers:
        AutoPipelineForText2Image = _OldAuto

    _load_pipeline(_OldDiffusers, "some/model", _Torch)
    assert len(calls) == 1
    assert calls[0]["safety_checker"] is None
    assert calls[0]["torch_dtype"] == "float32"


def test_diffusion_is_available_with_an_injected_runner(tmp_path):
    """An injected runner bypasses the extra check, so tests need no torch."""
    from core.imagegen_diffusion import DiffusionImageGenerator

    assert DiffusionImageGenerator(root=tmp_path, runner=_FakeRunner()).is_available()


def test_diffusion_without_the_extra_is_unavailable(tmp_path):
    """On a base install (no torch) the real backend reports itself unavailable."""
    from core.imagegen_diffusion import DiffusionImageGenerator

    generator = DiffusionImageGenerator(root=tmp_path)  # no runner, no extra
    # Either the extra is absent (unavailable) or present but weights uncached.
    assert generator.is_available() is False


def test_diffusion_unavailable_names_the_install_command(tmp_path):
    from core.imagegen_diffusion import DiffusionImageGenerator, DiffusionUnavailable

    generator = DiffusionImageGenerator(root=tmp_path)
    try:
        generator.generate(ImageRequest("anything"))
    except DiffusionUnavailable as error:
        # The message must point at a command that works from a source checkout,
        # not the unpublished ``pip install .[image]``.
        assert "requirements-image.txt" in str(error) or "download" in str(error)
    else:  # pragma: no cover - only when the extra is installed and cached
        pytest.skip("the image extra is installed and the model is cached")


def test_load_image_generator_builds_the_diffusion_backend(tmp_path):
    runner = _FakeRunner()
    generator = load_image_generator("diffusion", root=tmp_path, runner=runner, steps=2)
    assert generator.name == "diffusion"
    assert getattr(generator, "learned", False) is True


def test_backend_status_lists_both_backends():
    from core.imagegen import image_backend_status

    status = {name: available for name, available, _note in image_backend_status()}
    assert status["procedural"] is True
    assert "diffusion" in status  # availability depends on the install


def test_service_for_backend_keeps_the_same_root(tmp_path):
    service = ImageGenerationService(root=tmp_path / "images")
    other = service.for_backend("diffusion")
    assert other.root == service.root
    assert other.backend == "diffusion"


def test_service_reports_unavailable_diffusion_without_raising(tmp_path):
    service = ImageGenerationService(root=tmp_path, backend="diffusion")
    report = service.generate("a prompt")
    if report.ok:  # pragma: no cover - only when the real model is cached
        pytest.skip("the learned model is cached and usable")
    assert report.ok is False
    assert report.learned is True


def test_service_forwards_preset_and_negative_prompt_to_the_generator(tmp_path):
    from core.imagegen_diffusion import DiffusionImageGenerator

    runner = _FakeRunner()
    generator = DiffusionImageGenerator(root=tmp_path, runner=runner, preset="detailed")
    service = ImageGenerationService(root=tmp_path, generator=generator)
    report = service.generate("a portrait", negative_prompt="rain, fog")
    assert report.ok
    _model, _prompt, negative, _w, _h, steps, _g, _seed = runner.calls[0]
    assert negative == "rain, fog"
    assert steps == 30  # the preset's step count, not a single step


def test_service_download_model_accepts_a_preset(tmp_path):
    service = ImageGenerationService(root=tmp_path, backend="diffusion")
    report = service.download_model(approved=False, preset="detailed")
    # Denied by default (no approval), but the preset must resolve to a real model
    # rather than raising an unknown-preset error.
    assert report.ok is False
    assert "preset" not in report.message.lower()


def test_cli_presets_lists_the_photorealistic_default(capsys):
    assert image_main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "realistic (default)" in out
    assert "fast" in out and "detailed" in out


def test_cli_generate_accepts_preset_and_negative(tmp_path):
    # On a base install the learned backend is unavailable, so this exercises the
    # flag plumbing and the honest failure -- not a real generation.
    code = image_main(
        [
            "--output-dir",
            str(tmp_path),
            "generate",
            "a prompt",
            "--backend",
            "diffusion",
            "--preset",
            "fast",
            "--negative",
            "blur",
        ]
    )
    assert code == 1  # no extra/weights in the test environment
