import pytest

from core.belief import Belief
from core.evidence import Evidence
from core.observation import Observation
from core.prediction import Prediction
from core.provenance import (
    ProvenanceLedger,
    ProvenanceRecord,
    ProvenanceStage,
)


def test_from_artifact_reads_id_source_and_confidence():
    observation = Observation("Rain falls.", "weather_sensor", 0.9)
    record = ProvenanceRecord.from_artifact(
        observation, ProvenanceStage.OBSERVATION, context_id="CTX-1"
    )

    assert record.entity_id == observation.id
    assert record.stage is ProvenanceStage.OBSERVATION
    assert record.source == "weather_sensor"
    assert record.confidence == 0.9
    assert record.context_id == "CTX-1"
    assert record.id.startswith("PV-")


def test_from_artifact_uses_strength_when_confidence_is_absent():
    evidence = Evidence("Rain fell.", "weather_sensor", strength=0.7)
    record = ProvenanceRecord.from_artifact(evidence, ProvenanceStage.EVIDENCE)

    assert record.entity_id == evidence.id
    assert record.confidence == 0.7


def test_from_artifact_defaults_source_when_artifact_has_none():
    belief = Belief("Rain makes ground wet.", 0.5)
    record = ProvenanceRecord.from_artifact(belief, ProvenanceStage.BELIEF)

    assert record.source == "internal"
    assert record.confidence == 0.5


def test_from_artifact_requires_an_id():
    with pytest.raises(ValueError):
        ProvenanceRecord.from_artifact(object(), ProvenanceStage.BELIEF)


def test_record_validates_required_fields():
    with pytest.raises(ValueError):
        ProvenanceRecord("", ProvenanceStage.BELIEF, "source", 0.5)
    with pytest.raises(ValueError):
        ProvenanceRecord("BL-1", ProvenanceStage.BELIEF, "  ", 0.5)
    with pytest.raises(ValueError):
        ProvenanceRecord("BL-1", ProvenanceStage.BELIEF, "source", 1.5)


def test_record_deduplicates_evidence_and_parent_ids():
    record = ProvenanceRecord(
        "PRD-1",
        ProvenanceStage.PREDICTION,
        "pattern",
        0.5,
        evidence_ids=["EV-1", "EV-1", "EV-2"],
        parent_ids=["HYP-1", "HYP-1"],
    )

    assert record.evidence_ids == ("EV-1", "EV-2")
    assert record.parent_ids == ("HYP-1",)


def test_record_round_trips_through_dict():
    record = ProvenanceRecord.from_artifact(
        Prediction("It may rain.", 0.6, "HYP-1"),
        ProvenanceStage.PREDICTION,
        source="pattern:rain",
        context_id="CTX-1",
        evidence_ids=["EV-1"],
        parent_ids=["HYP-1"],
    )

    restored = ProvenanceRecord.from_dict(record.to_dict())

    assert restored.id == record.id
    assert restored.entity_id == record.entity_id
    assert restored.stage is ProvenanceStage.PREDICTION
    assert restored.source == "pattern:rain"
    assert restored.context_id == "CTX-1"
    assert restored.evidence_ids == ("EV-1",)
    assert restored.parent_ids == ("HYP-1",)
    assert restored.recorded_at == record.recorded_at


def test_ledger_tracks_history_and_latest():
    ledger = ProvenanceLedger()
    belief = Belief("Rain makes ground wet.", 0.5)

    first = ProvenanceRecord.from_artifact(belief, ProvenanceStage.BELIEF)
    second = ProvenanceRecord.from_artifact(belief, ProvenanceStage.LEARNING)
    ledger.record(first)
    ledger.record(second)

    assert ledger.get(belief.id) is second
    assert ledger.require(belief.id) is second
    assert ledger.history(belief.id) == [first, second]
    assert len(ledger) == 2
    assert belief.id in ledger


def test_ledger_queries_by_context_and_stage():
    ledger = ProvenanceLedger()
    ledger.record(
        ProvenanceRecord.from_artifact(
            Observation("Rain falls.", "sensor"),
            ProvenanceStage.OBSERVATION,
            context_id="CTX-1",
        )
    )
    ledger.record(
        ProvenanceRecord.from_artifact(
            Evidence("Rain fell.", "sensor"),
            ProvenanceStage.EVIDENCE,
            context_id="CTX-1",
        )
    )
    ledger.record(
        ProvenanceRecord.from_artifact(
            Observation("Sun shines.", "sensor"),
            ProvenanceStage.OBSERVATION,
            context_id="CTX-2",
        )
    )

    assert len(ledger.for_context("CTX-1")) == 2
    assert len(ledger.for_stage(ProvenanceStage.OBSERVATION)) == 2
    with pytest.raises(KeyError):
        ledger.require("missing")


def test_ledger_dump_and_restore_round_trip():
    ledger = ProvenanceLedger()
    observation = Observation("Rain falls.", "weather_sensor", 0.9)
    ledger.record(
        ProvenanceRecord.from_artifact(
            observation,
            ProvenanceStage.OBSERVATION,
            context_id="CTX-1",
            evidence_ids=["EV-1"],
        )
    )

    restored = ProvenanceLedger.restore(ledger.dump())

    assert len(restored) == len(ledger)
    record = restored.require(observation.id)
    assert record.context_id == "CTX-1"
    assert record.evidence_ids == ("EV-1",)
    assert record.source == "weather_sensor"
