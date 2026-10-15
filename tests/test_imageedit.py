"""Image editing: the decoder, the instruction interpreter, the service, the CLI.

These drive the real code paths -- the standard-library PNG/PPM decoders, the
deterministic pixel operations, the service's file handling, and the CLI's
``image edit`` -- with no mocks. Every test names the property it pins: exact
decoding, order-preserving instructions, honest "unknown words", bounded sizes,
and output that is a valid PNG. The window is exercised separately in
``tests/test_gui_design.py`` (it needs a display).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from cli.image import main as image_main
from core.imagegen import ImageRequest, ProceduralImageGenerator
from core.imageedit import (
    EditOp,
    EditPlan,
    ImageEditError,
    Raster,
    apply_op,
    apply_plan,
    decode_image,
    decode_image_bytes,
    plan_edits,
)
from gui.image_edit import ImageEditService


def _png_rows(png: bytes) -> tuple[int, int, bytes]:
    """Return (width, height, inflated IDAT bytes) for a PNG this project wrote."""
    index = 8
    width = height = 0
    idat = b""
    while index < len(png):
        length = struct.unpack(">I", png[index : index + 4])[0]
        tag = png[index + 4 : index + 8]
        data = png[index + 8 : index + 8 + length]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
        index += 12 + length
    return width, height, zlib.decompress(idat)


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# the raster and its encoders
# ---------------------------------------------------------------------------


def test_raster_rejects_bad_dimensions_and_pixel_count():
    with pytest.raises(ImageEditError):
        Raster(0, 4, bytearray())
    with pytest.raises(ImageEditError):
        Raster(2, 2, bytearray(3))  # needs 2*2*3 = 12 bytes


def test_solid_raster_reads_back_its_colour():
    raster = Raster.solid(3, 2, (10, 20, 30))
    assert raster.width == 3 and raster.height == 2
    assert raster.pixel(2, 1) == (10, 20, 30)
    assert list(raster.pixels()) == [(10, 20, 30)] * 6


def test_to_png_is_a_valid_png_of_the_right_size():
    raster = Raster.solid(5, 4, (1, 2, 3))
    png = raster.to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, _idat = _png_rows(png)
    assert (width, height) == (5, 4)


def test_to_png_round_trips_through_the_decoder():
    raster = Raster(2, 1, bytearray([1, 2, 3, 250, 251, 252]))
    decoded = decode_image_bytes(raster.to_png())
    assert decoded.width == 2 and decoded.height == 1
    assert decoded.pixel(0, 0) == (1, 2, 3)
    assert decoded.pixel(1, 0) == (250, 251, 252)


def test_raster_bridges_to_and_from_the_vision_buffer():
    raster = Raster.solid(2, 2, (7, 8, 9))
    buffer = raster.to_image_buffer()
    assert (buffer.width, buffer.height) == (2, 2)
    assert buffer.pixels[0] == (7, 8, 9)
    back = Raster.from_image_buffer(buffer)
    assert back.data == raster.data


# ---------------------------------------------------------------------------
# decoding: PPM, project PNG, and errors
# ---------------------------------------------------------------------------


def test_decode_ppm_reads_the_pixels():
    ppm = Raster.solid(3, 2, (4, 5, 6)).to_image_buffer().to_ppm()
    decoded = decode_image_bytes(ppm)
    assert (decoded.width, decoded.height) == (3, 2)
    assert decoded.pixel(1, 1) == (4, 5, 6)


def test_decode_ppm_rejects_a_truncated_payload():
    payload = b"P6\n2 2\n255\n" + bytes(3)  # needs 12 bytes of pixels
    with pytest.raises(ImageEditError, match="truncated"):
        decode_image_bytes(payload)


def test_decode_rejects_an_unknown_format():
    with pytest.raises(ImageEditError, match="Unsupported image format"):
        decode_image_bytes(b"GIF89a not supported here")


def test_decode_project_png_round_trips_the_generator_output(tmp_path):
    """The procedural generator writes a PNG the editor can read back exactly."""
    generator = ProceduralImageGenerator()
    image = generator.generate(ImageRequest(prompt="a red spiral", width=48, height=32))
    decoded = decode_image_bytes(image.png)
    assert (decoded.width, decoded.height) == (48, 32)
    # Re-encoding then decoding is stable.
    again = decode_image_bytes(decoded.to_png())
    assert again.data == decoded.data


def test_decode_reads_a_file_from_disk(tmp_path):
    path = tmp_path / "shot.png"
    Raster.solid(6, 4, (30, 60, 90)).save_png(path)
    decoded = decode_image(path)
    assert (decoded.width, decoded.height) == (6, 4)
    assert decoded.pixel(0, 0) == (30, 60, 90)


@pytest.mark.skipif(not _pillow_available(), reason="Pillow not installed")
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P", "LA", "1"])
def test_decode_reads_the_png_colour_types_pillow_writes(tmp_path, mode):
    """Pillow writes PNGs with real scanline filters; the decoder must undo them."""
    from PIL import Image

    source = Image.new(mode, (7, 5), 0)
    path = tmp_path / f"mode-{mode}.png"
    source.save(path)
    decoded = decode_image(path)
    assert (decoded.width, decoded.height) == (7, 5)
    # Every decoded pixel is fully opaque (alpha, if any, is composited over white).
    assert all(0 <= channel <= 255 for pixel in decoded.pixels() for channel in pixel)


@pytest.mark.skipif(not _pillow_available(), reason="Pillow not installed")
def test_decode_composites_alpha_over_white(tmp_path):
    from PIL import Image

    source = Image.new("RGBA", (2, 1), (255, 0, 0, 0))  # fully transparent red
    path = tmp_path / "clear.png"
    source.save(path)
    decoded = decode_image(path)
    assert decoded.pixel(0, 0) == (255, 255, 255)


def test_decode_rejects_an_interlaced_png(tmp_path):
    # A hand-built 1x1 interlaced PNG: the decoder refuses Adam7 clearly.
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    with pytest.raises(ImageEditError, match="Interlaced"):
        decode_image_bytes(png)


def test_decode_reports_a_truncated_png():
    with pytest.raises(ImageEditError):
        decode_image_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00")


# ---------------------------------------------------------------------------
# the instruction interpreter
# ---------------------------------------------------------------------------


def test_plan_is_empty_for_blank_text():
    plan = plan_edits("   ")
    assert plan.ops == () and plan.unknown == ()
    assert plan.ok is False
    assert "No operation" in plan.describe()


def test_plan_orders_operations_by_the_words():
    plan = plan_edits("rotate 90 and brighten")
    assert [op.kind for op in plan.ops] == ["rotate", "brightness"]
    # Reversing the words reverses the operations.
    reversed_plan = plan_edits("brighten and rotate 90")
    assert [op.kind for op in reversed_plan.ops] == ["brightness", "rotate"]


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("rotate 90", "rotate"),
        ("rotate left", "rotate"),
        ("flip horizontally", "flip_h"),
        ("mirror vertically", "flip_v"),
        ("resize to 512x512", "resize"),
        ("scale 2", "resize"),
        ("crop to a square", "crop_square"),
        ("crop to 100x50", "crop"),
        ("brighten", "brightness"),
        ("reduce brightness by 20%", "brightness"),
        ("more contrast", "contrast"),
        ("desaturate", "saturation"),
        ("grayscale", "grayscale"),
        ("black and white", "grayscale"),
        ("invert", "invert"),
        ("sepia", "sepia"),
        ("warm", "warm"),
        ("cool", "cool"),
        ("blur 3", "blur"),
        ("sharpen", "sharpen"),
        ("threshold at 100", "threshold"),
    ],
)
def test_plan_recognises_the_documented_vocabulary(text, kind):
    plan = plan_edits(text)
    assert [op.kind for op in plan.ops] == [kind]
    assert plan.unknown == ()


def test_plan_reports_words_it_did_not_understand():
    plan = plan_edits("make the background purple")
    assert plan.ops == ()
    assert set(plan.unknown) == {"background", "purple"}


def test_plan_keeps_understanding_and_reports_the_rest():
    plan = plan_edits("rotate 90 and paint the sky green")
    assert [op.kind for op in plan.ops] == ["rotate"]
    assert "sky" in plan.unknown and "green" in plan.unknown


def test_brighten_and_darken_read_the_direction():
    assert plan_edits("brighten by 20%").ops[0].params["factor"] == pytest.approx(1.2)
    assert plan_edits("reduce brightness by 20%").ops[0].params["factor"] == pytest.approx(0.8)
    assert plan_edits("darken").ops[0].params["factor"] < 1


def test_rotate_reads_degrees_and_direction():
    assert plan_edits("rotate 180").ops[0].params["degrees"] == 180
    assert plan_edits("rotate right").ops[0].params["degrees"] == 90
    assert plan_edits("rotate left").ops[0].params["degrees"] == 270


# ---------------------------------------------------------------------------
# the pixel operations
# ---------------------------------------------------------------------------


def test_rotate_90_swaps_the_dimensions_and_moves_a_corner():
    raster = Raster(2, 1, bytearray([1, 2, 3, 4, 5, 6]))  # left pixel then right
    rotated = apply_op(raster, EditOp("rotate", {"degrees": 90}))
    assert (rotated.width, rotated.height) == (1, 2)
    # Clockwise: the left edge becomes the top, so the left pixel stays on top.
    assert rotated.pixel(0, 0) == (1, 2, 3)
    assert rotated.pixel(0, 1) == (4, 5, 6)
    # Counter-clockwise sends the left edge to the bottom.
    counter = apply_op(raster, EditOp("rotate", {"degrees": 270}))
    assert counter.pixel(0, 0) == (4, 5, 6)
    assert counter.pixel(0, 1) == (1, 2, 3)


def test_rotate_180_and_360_are_identity_or_reverse():
    raster = Raster(2, 1, bytearray([1, 2, 3, 4, 5, 6]))
    flipped = apply_op(raster, EditOp("rotate", {"degrees": 180}))
    assert flipped.pixel(0, 0) == (4, 5, 6)
    assert flipped.pixel(1, 0) == (1, 2, 3)
    same = apply_op(raster, EditOp("rotate", {"degrees": 360}))
    assert same.data == raster.data


def test_flip_horizontal_and_vertical():
    raster = Raster(2, 1, bytearray([1, 2, 3, 4, 5, 6]))
    assert apply_op(raster, EditOp("flip_h")).pixel(0, 0) == (4, 5, 6)
    tall = Raster(1, 2, bytearray([1, 2, 3, 4, 5, 6]))
    assert apply_op(tall, EditOp("flip_v")).pixel(0, 0) == (4, 5, 6)


def test_resize_scales_the_dimensions():
    raster = Raster.solid(4, 2, (10, 10, 10))
    out = apply_op(raster, EditOp("resize", {"width": 8, "height": 4}))
    assert (out.width, out.height) == (8, 4)
    scaled = apply_op(raster, EditOp("resize", {"scale": 0.5}))
    assert (scaled.width, scaled.height) == (2, 1)


def test_resize_is_bounded_by_the_max_dimension():
    from core.imagegen import MAX_DIMENSION

    raster = Raster.solid(2, 2, (0, 0, 0))
    out = apply_op(raster, EditOp("resize", {"width": 99_999, "height": 99_999}))
    assert out.width == MAX_DIMENSION and out.height == MAX_DIMENSION


def test_crop_centres_and_clamps_to_the_source():
    data = bytearray(channel for value in range(16) for channel in (value, value, value))
    raster = Raster(4, 4, data)
    out = apply_op(raster, EditOp("crop", {"width": 2, "height": 2}))
    assert (out.width, out.height) == (2, 2)
    # The centre of a 4x4 is pixels (1,1)-(2,2), whose value is 5 and 6.
    assert out.pixel(0, 0) == (5, 5, 5)
    # A crop larger than the source is clamped, not an error.
    big = apply_op(raster, EditOp("crop", {"width": 99, "height": 99}))
    assert (big.width, big.height) == (4, 4)


def test_grayscale_makes_every_channel_equal():
    raster = Raster.solid(1, 1, (200, 100, 50))
    out = apply_op(raster, EditOp("grayscale"))
    red, green, blue = out.pixel(0, 0)
    assert red == green == blue


def test_invert_flips_every_channel():
    raster = Raster.solid(1, 1, (0, 128, 255))
    assert apply_op(raster, EditOp("invert")).pixel(0, 0) == (255, 127, 0)


def test_brightness_and_threshold_change_the_pixels():
    raster = Raster.solid(1, 1, (100, 100, 100))
    brighter = apply_op(raster, EditOp("brightness", {"factor": 1.5}))
    assert brighter.pixel(0, 0)[0] > 100
    black = apply_op(raster, EditOp("threshold", {"level": 200}))
    assert black.pixel(0, 0) == (0, 0, 0)


def test_blur_smooths_an_edge_but_keeps_the_size():
    left = (0, 0, 0)
    right = (255, 255, 255)
    raster = Raster(2, 1, bytearray(left + right))
    blurred = apply_op(raster, EditOp("blur", {"radius": 1}))
    assert (blurred.width, blurred.height) == (2, 1)
    assert 0 < blurred.pixel(0, 0)[0] < 255


def test_unknown_operation_is_refused():
    with pytest.raises(ImageEditError, match="Unknown edit operation"):
        apply_op(Raster.solid(1, 1, (0, 0, 0)), EditOp("teleport"))


def test_apply_plan_runs_every_operation_in_order():
    raster = Raster.solid(4, 2, (10, 10, 10))
    plan = EditPlan("x", (EditOp("rotate", {"degrees": 90}), EditOp("grayscale")))
    out = apply_plan(raster, plan)
    assert (out.width, out.height) == (2, 4)


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


def _source_image(tmp_path: Path) -> Path:
    generator = ProceduralImageGenerator()
    image = generator.generate(ImageRequest(prompt="a blue rig", width=40, height=24))
    path = tmp_path / "source.png"
    path.write_bytes(image.png)
    return path


def test_service_loads_and_edits_an_image(tmp_path):
    source = _source_image(tmp_path)
    service = ImageEditService(root=tmp_path / "images")
    session = service.load(source)
    assert (session.raster.width, session.raster.height) == (40, 24)

    report = service.apply(session, "rotate 90 and grayscale")
    assert report.ok
    assert report.learned is False
    assert report.operations == ("rotate 90°", "grayscale")
    assert (report.width, report.height) == (24, 40)
    assert Path(report.path).exists()
    # The session advanced to the result.
    assert (session.raster.width, session.raster.height) == (24, 40)


def test_service_writes_under_its_root_by_default(tmp_path):
    source = _source_image(tmp_path)
    root = tmp_path / "images"
    service = ImageEditService(root=root)
    session = service.load(source)
    report = service.apply(session, "invert")
    assert Path(report.path).parent == root
    assert report.path.endswith("source-edited-1.png")


def test_service_honours_an_explicit_output(tmp_path):
    source = _source_image(tmp_path)
    service = ImageEditService(root=tmp_path / "images")
    session = service.load(source)
    target = tmp_path / "chosen.png"
    report = service.apply(session, "sepia", output=target)
    assert Path(report.path) == target
    assert target.exists()


def test_service_chains_edits_and_numbers_the_files(tmp_path):
    source = _source_image(tmp_path)
    service = ImageEditService(root=tmp_path / "images")
    session = service.load(source)
    first = service.apply(session, "rotate 90")
    second = service.apply(session, "flip horizontally")
    assert first.ok and second.ok
    assert Path(second.path).name.endswith("-edited-2.png")
    assert session.applied == ("rotate 90°", "flip horizontally")


def test_service_refuses_an_unrecognised_instruction_without_changing_the_image(tmp_path):
    source = _source_image(tmp_path)
    service = ImageEditService(root=tmp_path / "images")
    session = service.load(source)
    before = session.raster.copy()
    report = service.apply(session, "paint the sky purple")
    assert report.ok is False
    assert set(report.unknown) == {"paint", "sky", "purple"}
    assert session.raster.data == before.data  # untouched


def test_service_reports_a_bad_file_without_raising(tmp_path):
    bad = tmp_path / "not-an-image.txt"
    bad.write_text("hello")
    service = ImageEditService(root=tmp_path / "images")
    with pytest.raises(ImageEditError):
        service.load(bad)


def test_service_save_copy_and_working_file(tmp_path):
    source = _source_image(tmp_path)
    service = ImageEditService(root=tmp_path / "images")
    session = service.load(source)
    service.apply(session, "grayscale")
    copy = tmp_path / "copy.png"
    report = service.save_copy(session, copy)
    assert report.ok and copy.exists()
    working = service.render_working(session)
    assert Path(working).exists()
    assert Path(working).name.endswith("-working.png")


def test_report_summary_states_honesty(tmp_path):
    source = _source_image(tmp_path)
    service = ImageEditService(root=tmp_path / "images")
    session = service.load(source)
    report = service.apply(session, "blur 2")
    summary = report.summary()
    assert "not a learned model" in summary
    assert "blur" in summary


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def test_cli_edit_writes_a_file(tmp_path, capsys):
    source = _source_image(tmp_path)
    code = image_main(
        ["--output-dir", str(tmp_path / "out"), "edit", str(source), "rotate 90 and invert"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Edited image saved" in out
    assert (tmp_path / "out" / "source-edited-1.png").exists()


def test_cli_edit_json_reports_honesty(tmp_path, capsys):
    import json

    source = _source_image(tmp_path)
    code = image_main(
        ["--output-dir", str(tmp_path / "out"), "edit", str(source), "grayscale", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["learned"] is False
    assert payload["operations"] == ["grayscale"]


def test_cli_edit_refuses_an_unrecognised_instruction(tmp_path, capsys):
    source = _source_image(tmp_path)
    code = image_main(
        ["--output-dir", str(tmp_path / "out"), "edit", str(source), "make it purple"]
    )
    assert code == 1
    assert "could not find an edit" in capsys.readouterr().out


def test_cli_edit_reports_a_missing_file(tmp_path, capsys):
    code = image_main(
        ["--output-dir", str(tmp_path / "out"), "edit", str(tmp_path / "nope.png"), "invert"]
    )
    assert code == 1
    assert "could not open" in capsys.readouterr().out
