"""Reading a model file's facts without loading it.

Every test writes its **own** tiny model file into ``tmp_path``: a real GGUF
header with a real metadata block, and a real numpy archive for the checkpoint.
Nothing is downloaded, nothing is loaded, and no GPU is touched.
"""

from __future__ import annotations

import ast
import struct
import zipfile

import pytest
from core.model_files import (
    FORMAT_FAYD_CHECKPOINT,
    FORMAT_GGUF,
    FORMAT_UNKNOWN,
    INTEGRITY_MAGIC_MISMATCH,
    INTEGRITY_MISSING,
    INTEGRITY_OK,
    INTEGRITY_TRUNCATED,
    INTEGRITY_UNREADABLE,
    ModelFileError,
    detect_format,
    digest_of,
    discover_models,
    is_supported_suffix,
    probe_file,
    read_checkpoint_info,
    read_gguf_info,
    validate_model_path,
)

# --------------------------------------------------------------- GGUF fixtures


def _gguf_key(text: str) -> bytes:
    """A GGUF metadata key: its length, then its bytes. Keys carry no type tag."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_string(text: str) -> bytes:
    """A GGUF string *value*: its type tag, its length, then the bytes."""
    raw = text.encode("utf-8")
    return struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw


def _gguf_uint32(value: int) -> bytes:
    return struct.pack("<I", 4) + struct.pack("<I", value)


def _gguf_int32(value: int) -> bytes:
    """A GGUF int32 value. The test helper keeps its values inside int32."""
    return struct.pack("<I", 5) + struct.pack("<i", value)


def write_gguf(
    path,
    *,
    name: str = "Test Model",
    architecture: str = "llama",
    file_type: int = 15,
    context_length: int | None = 8192,
    parameter_count: int | None = None,
    license_name: str = "",
    tensor_count: int = 3,
    extra_keys: dict[str, bytes] | None = None,
) -> None:
    """Write a minimal but *real* GGUF header: magic, counts, a metadata block.

    The tensor data is not written, because nothing in the inspector reads it --
    which is exactly the property these tests pin.
    """
    entries: list[tuple[str, bytes]] = [
        ("general.name", _gguf_string(name)),
        ("general.architecture", _gguf_string(architecture)),
        ("general.file_type", _gguf_uint32(file_type)),
    ]
    if context_length is not None:
        entries.append((f"{architecture}.context_length", _gguf_uint32(context_length)))
    if parameter_count is not None:
        entries.append(("general.parameter_count", _gguf_int32(parameter_count)))
    if license_name:
        entries.append(("general.license", _gguf_string(license_name)))
    for key, payload in (extra_keys or {}).items():
        entries.append((key, payload))

    body = b"".join(_gguf_key(key) + payload for key, payload in entries)
    header = b"GGUF" + struct.pack("<IQQ", 3, tensor_count, len(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)


def write_checkpoint(path, *, values=(257.0, 4.0, 8.0, 128.0, 128.0, 0.1, 1.0), arrays=52):
    """Write a real ``.npz`` with a real numpy ``config`` array, stdlib only."""
    header = repr(
        {"descr": "<f8", "fortran_order": False, "shape": (len(values),)}
    ).encode("latin1")
    # The npy header must be padded so the data starts 64-byte aligned; the
    # parser accepts the shape it is given, so a simple pad is enough here.
    pad = 64 - ((10 + len(header) + 1) % 64)
    payload = (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header) + pad + 1)
        + header
        + b" " * (pad + 1)
        + struct.pack(f"<{len(values)}d", *values)
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.npy", payload)
        for index in range(arrays):
            archive.writestr(f"p{index}.npy", b"\x93NUMPY stub")


# ------------------------------------------------------------------- GGUF read


def test_reads_a_gguf_header_and_its_metadata(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target, name="Fayd Test", architecture="qwen2", file_type=15)
    info = read_gguf_info(target)
    assert info is not None
    assert info.version == 3
    assert info.tensor_count == 3
    assert info.name == "Fayd Test"
    assert info.architecture == "qwen2"
    assert info.quantization == "Q4_K_M"
    assert info.context_length == 8192


def test_an_unknown_quantisation_stays_unknown(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target, file_type=999)
    info = read_gguf_info(target)
    assert info is not None
    assert info.quantization == ""


def test_a_missing_context_length_is_reported_as_unknown(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target, context_length=None)
    info = read_gguf_info(target)
    assert info is not None
    assert info.context_length is None


def test_reads_the_declared_parameter_count_and_license(tmp_path):
    target = tmp_path / "model.gguf"
    # ``general.parameter_count`` is a GGUF int32 in this helper, so the value
    # stays inside int32; the reader must not narrow or widen it.
    write_gguf(target, parameter_count=2_000_000_000, license_name="apache-2.0")
    info = read_gguf_info(target)
    assert info is not None
    assert info.parameter_count == 2_000_000_000
    assert info.license == "apache-2.0"


def test_a_non_gguf_file_reads_as_none(tmp_path):
    target = tmp_path / "notes.gguf"
    target.write_bytes(b"this is not a model at all")
    assert read_gguf_info(target) is None


def test_an_implausible_header_is_refused(tmp_path):
    target = tmp_path / "huge.gguf"
    target.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 999_999_999, 999_999_999))
    assert read_gguf_info(target) is None


def test_a_header_that_ends_mid_block_is_partial(tmp_path):
    target = tmp_path / "cut.gguf"
    write_gguf(target)
    data = target.read_bytes()
    # Keep the magic and the three counts (24 bytes), cut the whole metadata
    # block. The counts are readable, so this is still a GGUF -- but nothing in
    # the block can be read, so every fact stays unknown.
    target.write_bytes(data[:24])
    info = read_gguf_info(target)
    assert info is not None
    assert info.partial is True
    assert info.name == ""
    assert info.architecture == ""


def test_a_header_cut_inside_a_later_entry_keeps_the_earlier_facts(tmp_path):
    """What was read before the cut is kept; what was not is left unknown."""
    target = tmp_path / "cut.gguf"
    write_gguf(target, context_length=None, parameter_count=None, license_name="")
    data = target.read_bytes()
    # Cut inside the last entry (``general.file_type``), so name and architecture
    # were already read and only the quantisation is lost.
    target.write_bytes(data[: len(data) - 4])
    info = read_gguf_info(target)
    assert info is not None
    assert info.partial is True
    assert info.name == "Test Model"
    assert info.architecture == "llama"
    assert info.quantization == ""


# ------------------------------------------------------------------- discovery


def test_detects_format_from_the_contents_not_the_name(tmp_path):
    gguf = tmp_path / "real.gguf"
    write_gguf(gguf)
    assert detect_format(gguf) == FORMAT_GGUF

    checkpoint = tmp_path / "trained.bin"
    write_checkpoint(checkpoint)
    # A checkpoint named `.bin` is still a checkpoint: the bytes decide.
    assert detect_format(checkpoint) == FORMAT_FAYD_CHECKPOINT

    renamed = tmp_path / "liar.gguf"
    write_checkpoint(renamed)
    assert detect_format(renamed) == FORMAT_FAYD_CHECKPOINT


def test_an_unknown_file_is_reported_unknown(tmp_path):
    target = tmp_path / "thing.gguf"
    target.write_bytes(b"nope")
    assert detect_format(target) == FORMAT_UNKNOWN


def test_discovery_finds_both_formats_and_skips_the_rest(tmp_path):
    write_gguf(tmp_path / "a.gguf")
    write_gguf(tmp_path / "nested" / "b.gguf")
    write_checkpoint(tmp_path / "c.npz")
    (tmp_path / "notes.txt").write_text("hello")
    report = discover_models(tmp_path)
    assert len(report) == 3
    assert len(report.by_format(FORMAT_GGUF)) == 2
    assert len(report.by_format(FORMAT_FAYD_CHECKPOINT)) == 1
    assert "GGUF" in report.summary()


def test_discovery_skips_hidden_and_vendor_folders(tmp_path):
    write_gguf(tmp_path / "keep.gguf")
    write_gguf(tmp_path / ".hidden" / "skip.gguf")
    write_gguf(tmp_path / "node_modules" / "skip.gguf")
    report = discover_models(tmp_path)
    assert len(report) == 1
    assert report.probes[0].path.name == "keep.gguf"


def test_discovery_is_bounded_and_says_so(tmp_path):
    for index in range(6):
        write_gguf(tmp_path / f"m{index}.gguf")
    report = discover_models(tmp_path, limit=3)
    assert len(report) == 3
    assert report.truncated is True


def test_discovery_of_nothing_is_an_honest_empty(tmp_path):
    report = discover_models(tmp_path)
    assert len(report) == 0
    assert report.truncated is False
    assert "No model files" in report.summary()


def test_discovery_accepts_a_list_of_roots(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    write_gguf(first / "a.gguf")
    write_gguf(second / "b.gguf")
    report = discover_models([first, second])
    assert len(report) == 2


def test_discovery_accepts_no_roots(tmp_path):
    report = discover_models()
    assert len(report) == 0
    assert report.roots == ()


# ---------------------------------------------------------------- probe_file


def test_probe_of_a_good_gguf_is_usable(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target, name="Probe Me")
    probe = probe_file(target)
    assert probe.usable is True
    assert probe.integrity == INTEGRITY_OK
    assert probe.format == FORMAT_GGUF
    assert probe.size_bytes > 0
    assert probe.gguf is not None
    assert probe.gguf.name == "Probe Me"


def test_probe_of_a_missing_file_reports_missing(tmp_path):
    probe = probe_file(tmp_path / "gone.gguf")
    assert probe.exists is False
    assert probe.integrity == INTEGRITY_MISSING
    assert probe.usable is False
    assert "no file" in probe.detail.lower()


def test_probe_of_a_folder_reports_unreadable(tmp_path):
    folder = tmp_path / "a-folder.gguf"
    folder.mkdir()
    probe = probe_file(folder)
    assert probe.integrity == INTEGRITY_UNREADABLE
    assert "folder" in probe.detail


def test_probe_of_a_truncated_gguf_is_refused(tmp_path):
    target = tmp_path / "cut.gguf"
    target.write_bytes(b"GGUF" + b"\x03\x00\x00\x00")
    probe = probe_file(target)
    assert probe.integrity == INTEGRITY_TRUNCATED
    assert probe.usable is False
    assert "truncated" in probe.detail.lower()


def test_probe_of_a_wrong_magic_file_is_refused(tmp_path):
    target = tmp_path / "wrong.gguf"
    target.write_bytes(b"ELF\x7f not a model")
    probe = probe_file(target)
    assert probe.integrity == INTEGRITY_MAGIC_MISMATCH
    assert probe.format == FORMAT_UNKNOWN
    assert "not a GGUF" in probe.detail


def test_probe_can_compute_a_digest_when_asked(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target)
    without = probe_file(target)
    assert without.digest == ""
    with_digest = probe_file(target, digest=True)
    assert with_digest.digest
    assert with_digest.digest_algorithm == "blake2b"
    assert with_digest.digest == digest_of(target)


def test_probe_serialises_to_a_stable_mapping(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target)
    first = probe_file(target).to_primitive()
    second = probe_file(target).to_primitive()
    assert first == second
    assert first["format"] == FORMAT_GGUF
    assert isinstance(first["gguf"], dict)


def test_suffix_hint_never_claims_a_format(tmp_path):
    assert is_supported_suffix("x.gguf") is True
    assert is_supported_suffix("x.npz") is True
    assert is_supported_suffix("x.txt") is False


# --------------------------------------------------------------- checkpoint


def test_reads_a_trained_checkpoint_without_numpy(tmp_path):
    target = tmp_path / "fayd-llm.npz"
    write_checkpoint(target, values=(257.0, 4.0, 8.0, 128.0, 128.0, 0.1, 1.0), arrays=52)
    info = read_checkpoint_info(target)
    assert info is not None
    assert info.vocab_size == 257
    assert info.n_layer == 4
    assert info.n_head == 8
    assert info.n_embd == 128
    assert info.block_size == 128
    assert info.dropout == pytest.approx(0.1)
    assert info.tie_weights is True
    assert info.parameter_arrays == 52
    assert info.parameter_count is not None and info.parameter_count > 0


def test_an_older_checkpoint_without_dropout_still_reads(tmp_path):
    target = tmp_path / "old.npz"
    write_checkpoint(target, values=(257.0, 4.0, 8.0, 128.0, 128.0), arrays=3)
    info = read_checkpoint_info(target)
    assert info is not None
    assert info.vocab_size == 257
    assert info.dropout is None
    assert info.tie_weights is None


def test_a_zip_that_is_not_a_checkpoint_reads_as_none(tmp_path):
    target = tmp_path / "archive.npz"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("something.txt", "not a checkpoint")
    assert read_checkpoint_info(target) is None


def test_a_corrupt_config_array_reads_as_none(tmp_path):
    target = tmp_path / "broken.npz"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("config.npy", b"\x93NUMPY\x01\x00\x10\x00garbage-not-a-dict")
    assert read_checkpoint_info(target) is None


def test_a_checkpoint_probes_as_usable(tmp_path):
    target = tmp_path / "fayd-llm.npz"
    write_checkpoint(target)
    probe = probe_file(target)
    assert probe.usable is True
    assert probe.format == FORMAT_FAYD_CHECKPOINT
    assert probe.checkpoint is not None
    assert probe.checkpoint.n_layer == 4


# --------------------------------------------------------------- validation


def test_validate_accepts_a_real_model_and_returns_the_probe(tmp_path):
    target = tmp_path / "model.gguf"
    write_gguf(target)
    probe = validate_model_path(target)
    assert probe.usable is True


def test_validate_refuses_a_missing_file_with_the_reason(tmp_path):
    with pytest.raises(ModelFileError) as caught:
        validate_model_path(tmp_path / "nope.gguf")
    assert "no file" in str(caught.value).lower()


def test_validate_refuses_a_non_model_file(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hello")
    with pytest.raises(ModelFileError) as caught:
        validate_model_path(target)
    assert "not a GGUF" in str(caught.value)


def test_validate_refuses_a_truncated_gguf(tmp_path):
    target = tmp_path / "cut.gguf"
    target.write_bytes(b"GGUF\x03\x00\x00\x00")
    with pytest.raises(ModelFileError):
        validate_model_path(target)


def test_reading_a_gguf_never_reads_the_tensor_data(tmp_path):
    """A header-only file is enough, which proves the weights are not read."""
    target = tmp_path / "header-only.gguf"
    write_gguf(target, tensor_count=500)
    info = read_gguf_info(target)
    assert info is not None
    assert info.tensor_count == 500
    # The file is a few hundred bytes: no tensor payload exists at all.
    assert target.stat().st_size < 4096


def test_npy_parser_refuses_a_non_float64_array():
    from core.model_files import _parse_npy_float64

    header = repr({"descr": "<i4", "fortran_order": False, "shape": (3,)}).encode("latin1")
    payload = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header) + 1) + header + b" " + b"\x00" * 12
    assert _parse_npy_float64(payload) is None


def test_npy_parser_refuses_a_fortran_order_array():
    from core.model_files import _parse_npy_float64

    header = repr({"descr": "<f8", "fortran_order": True, "shape": (3,)}).encode("latin1")
    payload = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header) + 1) + header + b" " + b"\x00" * 24
    assert _parse_npy_float64(payload) is None


def test_npy_parser_refuses_a_short_payload():
    from core.model_files import _parse_npy_float64

    assert _parse_npy_float64(b"") is None
    assert _parse_npy_float64(b"\x93NUMPY\x01\x00") is None


def test_the_header_literal_round_trips_as_a_dict():
    """Guards the assumption the checkpoint parser rests on."""
    header = repr({"descr": "<f8", "fortran_order": False, "shape": (7,)}).encode("latin1")
    parsed = ast.literal_eval(header.decode("latin1"))
    assert parsed["descr"] == "<f8"
