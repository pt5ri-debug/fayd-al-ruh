"""Central provenance must not lose context across the cognitive pipeline.

Each transition records one ProvenanceRecord carrying identity, time, source,
evidence, confidence, and context. This test walks Observation -> Evidence ->
Belief -> Prediction -> PredictionError -> Learning and proves the context id
survives every hop, including a snapshot/restore round trip.
"""

import pytest
from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.observation import Observation
from core.provenance import ProvenanceStage
from storage.json_store import JsonStore

pytestmark = pytest.mark.integration


def _run_episode(controller: CognitiveController):
    belief = Belief("Rain makes the ground wet.", 0.5)

    # A repeated observation lets the pattern detector produce a prediction.
    controller.process(
        Observation("Rain falls on the ground.", "weather_sensor", 0.9),
        belief,
        EvidencePolarity.SUPPORTS,
    )
    observed = Observation("Rain falls on the ground.", "weather_sensor", 0.9)
    cycle = controller.process(observed, belief, EvidencePolarity.SUPPORTS)

    prediction = cycle.loop_result.predictions[0]
    evaluated = controller.evaluate_prediction(prediction.id, belief, actual_result=1.0)
    error = evaluated.loop_result.prediction_errors[0]
    controller.learn_from_prediction_error(belief, error, action="observe")

    return belief, observed, cycle, prediction, error


def test_context_survives_every_transition():
    controller = CognitiveController()
    belief, observed, cycle, prediction, error = _run_episode(controller)
    context_id = cycle.context_id
    evidence_id = cycle.loop_result.evidence_id

    # Observation, evidence, belief, and prediction all share one context.
    for artifact in (observed, controller.graph.evidence[evidence_id], belief, prediction):
        record = controller.provenance_for(artifact.id)
        assert record is not None, f"no provenance for {artifact.id}"
        assert record.context_id == context_id

    assert controller.provenance_for(observed.id).stage is ProvenanceStage.OBSERVATION
    assert controller.provenance_for(evidence_id).stage is ProvenanceStage.EVIDENCE
    assert controller.provenance_for(prediction.id).stage is ProvenanceStage.PREDICTION

    # The prediction error and the learning update keep the same context.
    assert controller.provenance_for(error.id).context_id == context_id
    assert controller.provenance_for(error.id).stage is ProvenanceStage.PREDICTION_ERROR

    learning = controller.provenance_for(belief.id)
    assert learning.stage is ProvenanceStage.LEARNING
    assert learning.context_id == context_id
    # The belief now has a creation record and a learning record in one chain.
    stages = [record.stage for record in controller.provenance_history(belief.id)]
    assert ProvenanceStage.BELIEF in stages
    assert ProvenanceStage.LEARNING in stages


def test_provenance_links_evidence_and_parents():
    controller = CognitiveController()
    belief, observed, cycle, prediction, error = _run_episode(controller)
    evidence_id = cycle.loop_result.evidence_id

    evidence_record = controller.provenance_for(evidence_id)
    assert observed.id in evidence_record.parent_ids
    assert evidence_id in evidence_record.evidence_ids

    prediction_record = controller.provenance_for(prediction.id)
    assert belief.id in prediction_record.parent_ids
    assert evidence_id in prediction_record.evidence_ids

    error_record = controller.provenance_for(error.id)
    assert prediction.id in error_record.parent_ids

    learning_record = controller.provenance_for(belief.id)
    assert error.id in learning_record.parent_ids


def test_provenance_survives_snapshot_restore(tmp_path):
    controller = CognitiveController()
    belief, observed, cycle, prediction, error = _run_episode(controller)
    context_id = cycle.context_id
    before = controller.provenance.dump()

    controller.create_snapshot("episode", JsonStore(tmp_path))
    restored = CognitiveController()
    assert restored.restore_snapshot("episode", JsonStore(tmp_path)) is True

    assert restored.provenance.dump() == before
    assert restored.provenance_for(observed.id).context_id == context_id
    assert restored.provenance_for(error.id).context_id == context_id
    assert restored.provenance_for(prediction.id).stage is ProvenanceStage.PREDICTION
    assert restored.provenance_for(belief.id).stage is ProvenanceStage.LEARNING
