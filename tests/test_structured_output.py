import json

import pytest
from core.structured_output import (
    MODEL_OUTPUT_SCHEMA,
    StructuredOutputError,
    parse_structured_output,
)


def _payload(**overrides) -> str:
    base: dict = {"concepts": [], "evidence": [], "hypotheses": [], "questions": [], "plans": []}
    base.update(overrides)
    return json.dumps(base)


def test_schema_lists_all_required_sections():
    assert set(MODEL_OUTPUT_SCHEMA["required"]) == {
        "concepts", "evidence", "hypotheses", "questions", "plans",
    }
    assert MODEL_OUTPUT_SCHEMA["additionalProperties"] is False


def test_parse_full_proposal():
    proposal = parse_structured_output(_payload(
        concepts=[{"label": "rain", "aliases": ["precipitation"]}],
        evidence=[{"description": "ground wet", "source": "sensor", "supports": True, "strength": 0.8}],
        hypotheses=[{"statement": "rain wets ground", "confidence": 0.7}],
        questions=[{"text": "does rain wet the ground?", "topic": "causality", "priority": 0.6}],
        plans=[{"goal": "confirm", "priority": 0.9, "steps": [
            {"description": "observe"},
            {"description": "measure", "depends_on": ["observe"]},
        ]}],
    ))

    assert proposal.concepts[0].label == "rain"
    assert proposal.concepts[0].aliases == ("precipitation",)
    assert proposal.evidence[0].strength == 0.8
    assert proposal.hypotheses[0].confidence == 0.7
    assert proposal.questions[0].topic == "causality"
    assert [step.description for step in proposal.plans[0].steps] == ["observe", "measure"]
    assert proposal.plans[0].steps[1].depends_on == ("observe",)


def test_parse_extracts_json_from_markdown_fence():
    proposal = parse_structured_output("Here you go:\n```json\n" + _payload(concepts=[{"label": "x"}]) + "\n```")
    assert proposal.concepts[0].label == "x"


def test_parse_extracts_json_from_surrounding_prose():
    proposal = parse_structured_output("Sure. " + _payload(hypotheses=[{"statement": "s"}]) + " Done.")
    assert proposal.hypotheses[0].statement == "s"


def test_missing_sections_default_to_empty():
    proposal = parse_structured_output('{"concepts": [{"label": "a"}]}')
    assert proposal.is_empty is False
    assert proposal.evidence == [] and proposal.plans == []


def test_confidence_is_clamped():
    proposal = parse_structured_output(_payload(hypotheses=[{"statement": "s", "confidence": 5.0}]))
    assert proposal.hypotheses[0].confidence == 1.0


def test_non_json_output_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output("just prose, no json here")


@pytest.mark.parametrize("forbidden", ["beliefs", "tools", "tool_calls", "actions", "execute", "shell"])
def test_forbidden_keys_are_rejected(forbidden):
    with pytest.raises(StructuredOutputError):
        parse_structured_output(json.dumps({forbidden: [], "evidence": []}))


def test_forbidden_key_nested_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(_payload(plans=[{"goal": "g", "steps": [{"description": "d", "tool_calls": []}]}]))


def test_wrong_section_type_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(json.dumps({"concepts": "not a list"}))


def test_empty_concept_label_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(_payload(concepts=[{"label": "  "}]))


def test_non_numeric_confidence_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(_payload(hypotheses=[{"statement": "s", "confidence": "high"}]))
