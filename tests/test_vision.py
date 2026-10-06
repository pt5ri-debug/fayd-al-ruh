"""Tests for optional multimodal perception."""

from __future__ import annotations

import pytest

from core.assistant import GroundedAssistant
from core.documents import build_document_store
from core.ontology import Ontology
from core.domain_knowledge import seed_ontology
from core.curriculum_knowledge import seed_curriculum
from core.rag import RagRetriever
from core.vision import (
    HeuristicVisionEncoder,
    ImageBuffer,
    MultimodalAssistant,
    extract_features,
    load_vision_encoder,
)


def _assistant():
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=ontology))


def test_image_buffer_validates_shape():
    with pytest.raises(ValueError):
        ImageBuffer(0, 1, ())
    with pytest.raises(ValueError):
        ImageBuffer(2, 2, ((0, 0, 0),))
    with pytest.raises(ValueError):
        ImageBuffer(1, 1, ((300, 0, 0),))


def test_ppm_round_trip():
    image = ImageBuffer(2, 1, ((255, 0, 0), (0, 128, 255)))
    restored = ImageBuffer.from_ppm(image.to_ppm())
    assert restored == image


def test_ppm_round_trip_through_file(tmp_path):
    image = ImageBuffer.solid(3, 2, (10, 20, 30))
    path = tmp_path / "img.ppm"
    image.write_ppm(path)
    assert ImageBuffer.read_ppm(path) == image


def test_ppm_rejects_bad_header_and_truncation():
    with pytest.raises(ValueError):
        ImageBuffer.from_ppm(b"P5\n1 1\n255\n\x00")
    with pytest.raises(ValueError):
        ImageBuffer.from_ppm(b"P6\n2 2\n255\n\x00\x01")


def test_features_of_a_solid_image():
    features = extract_features(ImageBuffer.solid(4, 4, (100, 100, 100)))
    assert features.brightness == pytest.approx(100 / 255, abs=0.01)
    assert features.saturation == 0.0
    assert features.edge_density == 0.0
    assert features.width == 4 and features.height == 4


def test_features_detect_edges_and_orientation():
    # A left-black / right-white split has one strong vertical edge per row.
    pixels = tuple(
        (0, 0, 0) if column < 4 else (255, 255, 255)
        for _row in range(4)
        for column in range(8)
    )
    features = extract_features(ImageBuffer(8, 4, pixels))
    assert features.edge_density > 0.0
    assert features.is_landscape


def test_heuristic_encoder_describes_measurables():
    description = HeuristicVisionEncoder().describe(ImageBuffer.solid(8, 4, (200, 200, 200)))
    assert "8x4" in description
    assert "Landscape" in description


def test_load_vision_encoder_rejects_unknown():
    assert isinstance(load_vision_encoder("heuristic"), HeuristicVisionEncoder)
    with pytest.raises(ValueError):
        load_vision_encoder("clip")


def test_multimodal_assistant_answers_with_description():
    multimodal = MultimodalAssistant(_assistant())
    image = ImageBuffer.solid(6, 6, (120, 120, 120))
    result = multimodal.ask_about_image("describe the image", image)
    assert result.description
    assert result.answer
    assert "describe the image" in result.question
