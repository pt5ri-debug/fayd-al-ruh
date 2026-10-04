"""Small, safe local demonstration of the advanced cognitive components."""

from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.goal import Goal, PlanStep
from core.observation import Observation


def main() -> None:
    controller = CognitiveController()
    belief = Belief("Rain makes the garden wet.")

    controller.process(
        Observation("Rain falls on the garden.", "demo_sensor"),
        belief,
        EvidencePolarity.SUPPORTS,
    )
    result = controller.process(
        Observation("The garden receives rain again.", "demo_sensor"),
        belief,
        EvidencePolarity.SUPPORTS,
    )

    goal = Goal("Verify the belief from further observations.", priority=0.8)
    plan = controller.order_plan_steps(
        goal,
        [
            PlanStep("collect a new observation"),
            PlanStep("evaluate the prediction", ("collect a new observation",)),
        ],
    )

    print(f"Working memories recalled: {len(result.working_memory)}")
    print(f"Plan: {[step.description for step in plan]}")
    print(f"Quality: {controller.assess_quality()}")


if __name__ == "__main__":
    main()
