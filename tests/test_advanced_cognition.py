from core.cognitive_controller import CognitiveController
from core.goals import GoalPlanner, PlanStep, TaskGoal
from core.model_interface import CallableModel, ModelRequest, MultimodalInput
from core.safe_learning import LearningSandbox, SandboxResult
from core.semantic_memory import SemanticMemoryIndex


def test_semantic_index_returns_related_item():
    index = SemanticMemoryIndex()
    index.add("weather", "rain clouds water")
    index.add("robot", "motor steel machine")

    assert index.search("rain water", limit=1)[0].item_id == "weather"


def test_goal_planner_orders_dependencies():
    goal = TaskGoal("Learn safely")
    steps = [PlanStep("evaluate"), PlanStep("learn", ("evaluate",))]

    assert [step.description for step in GoalPlanner().plan(goal, steps)] == ["evaluate", "learn"]


def test_sandbox_runs_only_registered_experiment():
    sandbox = LearningSandbox()
    sandbox.register("reward", lambda state: SandboxResult("reward", state["score"], "ok"))

    assert sandbox.run("reward", {"score": 0.5}).reward == 0.5


def test_controller_model_adapter_receives_context():
    controller = CognitiveController()
    received = []
    model = CallableModel("test", lambda request: received.append(request) or "reasoned")
    response = controller.reason(model, ModelRequest((MultimodalInput("text", "hello"),)))

    assert response.text == "reasoned"
    assert received[0].inputs[0].value == "hello"
    assert controller.assess_quality().logical_consistency == 1.0
