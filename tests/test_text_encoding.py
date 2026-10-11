"""Tests for safe text-encoding detection and strict decoding."""

from __future__ import annotations

import pytest

from core.text_encoding import (
    CANDIDATE_ENCODINGS,
    EncodingDetectionError,
    decode_bytes,
    detect_encoding,
)


def test_utf8_is_detected():
    assert detect_encoding("plain ascii".encode("utf-8")) == "utf-8"


def test_utf8_arabic_round_trips_without_replacement():
    raw = "معادلة ضغط الخزان".encode("utf-8")
    decoded = decode_bytes(raw)
    assert decoded.encoding == "utf-8"
    assert decoded.text == "معادلة ضغط الخزان"
    assert "\ufffd" not in decoded.text


def test_utf8_bom_is_honoured_and_stripped():
    raw = "\ufeffhello".encode("utf-8")
    decoded = decode_bytes(raw)
    assert decoded.encoding == "utf-8-sig"
    assert decoded.text == "hello"


def test_cp1256_arabic_is_detected():
    text = "ضغط البئر"
    raw = text.encode("cp1256")
    assert detect_encoding(raw) == "cp1256"
    assert decode_bytes(raw).text == text


def test_utf16_with_bom_is_detected():
    raw = "hello عالم".encode("utf-16")  # includes a BOM
    decoded = decode_bytes(raw)
    assert decoded.encoding in {"utf-16-le", "utf-16-be"}
    assert decoded.text.lstrip("\ufeff") == "hello عالم"


def test_undecodable_bytes_raise_instead_of_replacing():
    # A UTF-16 stream truncated mid-character cannot decode cleanly as anything;
    # the detector must refuse rather than hand back U+FFFD-filled text.
    raw = "hello".encode("utf-16-le")[:-1]
    with pytest.raises(EncodingDetectionError):
        detect_encoding(raw)
    with pytest.raises(EncodingDetectionError):
        decode_bytes(raw)


def test_wrong_explicit_encoding_raises():
    with pytest.raises(EncodingDetectionError):
        decode_bytes(b"\xff\xfe\x00\x00", encoding="ascii")


def test_candidate_list_is_ordered_and_utf8_first():
    assert CANDIDATE_ENCODINGS[0] == "utf-8"


def test_empty_input_defaults_to_utf8():
    assert detect_encoding(b"") == "utf-8"
    assert decode_bytes(b"").text == ""


def test_truncated_prefix_tolerates_a_partial_final_character():
    raw = "مرحبا".encode("utf-8")[:-1]  # drop the last byte of the final letter
    assert detect_encoding(raw, allow_truncated=True) == "utf-8"
    with pytest.raises(EncodingDetectionError):
        detect_encoding(raw)
