"""Identity preservation across the full cognitive pipeline.

Every transition must keep the same observation id, evidence ids, belief id,
prediction id, prediction-error id, and causal-link id, including through a
snapshot/restore round trip.
"""

import pytest
from core.belief import Belief
from core.causal_link import CausalLink
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.observation import Observation
from storage.json_store import JsonStore

pytestmark = pytest.mark.integration


def test_identity_survives_every_transition_and_snapshot(tmp_path):
    controller = CognitiveController()
    belief = Belief("Rain makes the ground wet.", .5)
    observation = Observation("Rain falls on the ground.", "weather_sensor", .9)

    # observation -> evidence -> context (a repeated observation yields a prediction)
    controller.process(Observation("Rain falls on the ground.", "weather_sensor", .9), belief, EvidencePolarity.SUPPORTS)
    cycle = controller.process(observation, belief, EvidencePolarity.SUPPORTS)
    context = controller.contexts.contexts[cycle.context_id]
    prediction = cycle.loop_result.predictions[0]

    assert context.observation_id == observation.id
    assert context.belief_id == belief.id
    assert context.evidence_ids == [cycle.loop_result.evidence_id]
    assert context.prediction_ids == [prediction.id]

    # prediction -> prediction error -> learning target
    evaluated = controller.evaluate_prediction(prediction.id, belief, actual_result=1.0)
    error = evaluated.loop_result.prediction_errors[0]
    controller.learn_from_prediction_error(belief, error, action="observe")

    assert context.prediction_error_ids == [error.id]
    assert context.learning_target_ids == [belief.id]

    # evidence -> causal link attached back to the originating context
    evidence_ids = list(context.evidence_ids)
    link = CausalLink("rain", "wet_ground", .9, evidence_ids)
    controller.add_causal_link(link)

    assert link.id in context.causal_link_ids
    assert controller.validate_contexts() == []

    # snapshot -> restore keeps every id and every cross-reference intact
    controller.create_snapshot("episode", JsonStore(tmp_path))
    restored = CognitiveController()
    assert restored.restore_snapshot("episode", JsonStore(tmp_path)) is True

    restored_context = restored.contexts.contexts[cycle.context_id]
    assert restored_context.observation_id == observation.id
    assert restored_context.belief_id == belief.id
    assert restored_context.evidence_ids == evidence_ids
    assert restored_context.prediction_ids == [prediction.id]
    assert restored_context.prediction_error_ids == [error.id]
    assert restored_context.learning_target_ids == [belief.id]
    assert restored_context.causal_link_ids == [link.id]
    assert link.id in restored.causal_graph.links
    assert restored_context.evidence_ids == list(restored.causal_graph.links[link.id].evidence_ids)
    assert restored.validate_contexts() == []
