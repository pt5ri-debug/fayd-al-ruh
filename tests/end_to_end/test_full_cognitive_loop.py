from core.action import Action
from core.belief import Belief
from core.causal_link import CausalLink
from core.cognitive_controller import CognitiveController
from core.environment import Environment
from core.evidence_processor import EvidencePolarity
from core.state import State
import pytest


pytestmark = pytest.mark.end_to_end


def test_full_cognitive_loop():
    controller = CognitiveController()
    environment = Environment(State(frozenset({"rain"})))
    belief = Belief("Rain makes ground wet.", .5)

    # Environment -> observation -> memory -> repeated pattern -> hypothesis -> prediction.
    controller.process(environment.observe(), belief, EvidencePolarity.SUPPORTS)
    second = controller.process(environment.observe(), belief, EvidencePolarity.SUPPORTS)
    assert controller.memory.episodic_memory.episodes
    assert second.loop_result.patterns and second.loop_result.hypotheses and second.loop_result.predictions

    # Action -> changed environment -> new observation -> prediction error -> learning.
    environment.apply(Action("wet_ground", frozenset({"rain"}), frozenset({"wet_ground"})))
    observed = environment.observe(result=1.0)
    learned = controller.evaluate_prediction(second.loop_result.predictions[0].id, belief, observed.data["result"])
    assert learned.loop_result.prediction_errors
    controller.learn_from_prediction_error(belief, learned.loop_result.prediction_errors[0], action="wet_ground")

    # Belief revision + causal update + knowledge graph + subsequent prediction.
    controller.revise_belief(belief.id)
    evidence_ids = [item.id for item in controller.graph.get_belief_evidence(belief.id)]
    controller.add_causal_link(CausalLink("rain", "wet_ground", .9, evidence_ids))
    third = controller.process(observed, belief, EvidencePolarity.SUPPORTS)

    assert controller.graph.get_belief(belief.id) is belief
    assert controller.causal_graph.link_count == 1
    assert third.loop_result.predictions
    assert controller.learner.history.events
    assert controller.validate_contexts() == []
