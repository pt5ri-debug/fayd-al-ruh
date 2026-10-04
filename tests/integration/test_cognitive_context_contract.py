import pytest

from core.belief import Belief
from core.causal_link import CausalLink
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.observation import Observation


pytestmark = pytest.mark.integration


def test_context_preserves_identity_evidence_and_causality_across_learning():
    controller = CognitiveController()
    belief = Belief("Rain makes ground wet.", .5)

    controller.process(
        Observation("Rain falls.", "weather_sensor", .9),
        belief,
        EvidencePolarity.SUPPORTS,
    )
    observed = Observation("Rain falls.", "weather_sensor", .9)
    result = controller.process(observed, belief, EvidencePolarity.SUPPORTS)

    context = controller.contexts.contexts[result.context_id]
    assert context.observation_id == observed.id
    assert context.source == "weather_sensor"
    assert context.belief_id == belief.id
    assert len(context.evidence_ids) == 1
    assert context.prediction_ids == [result.loop_result.predictions[0].id]

    evaluated = controller.evaluate_prediction(
        result.loop_result.predictions[0].id,
        belief,
        actual_result=1.0,
    )
    error = evaluated.loop_result.prediction_errors[0]
    controller.learn_from_prediction_error(belief, error, action="observe")
    controller.add_causal_link(CausalLink("rain", "wet_ground", .9, context.evidence_ids))

    assert context.prediction_error_ids == [error.id]
    assert context.learning_target_ids == [belief.id]
    assert len(context.causal_link_ids) == 1
    assert controller.validate_contexts() == []
