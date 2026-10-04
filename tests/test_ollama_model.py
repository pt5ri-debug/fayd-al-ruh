from ai.ollama_model import OllamaModel
from core.model_interface import CallableModel, ModelRequest, MultimodalInput
from core.structured_output import MODEL_OUTPUT_SCHEMA


class _FakeOllamaClient:
    """Mimics OllamaClient's ``generate(prompt: str) -> str`` contract."""

    model = "qwen3:4b"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "{\"concepts\": [], \"evidence\": [], \"hypotheses\": [], \"questions\": [], \"plans\": []}"


def test_ollama_model_adapts_client_to_cognitive_model():
    client = _FakeOllamaClient()
    model = OllamaModel(client)
    request = ModelRequest((MultimodalInput("text", "analyze this"),), response_schema=MODEL_OUTPUT_SCHEMA)

    response = model.generate(request)

    assert response.model_name == "qwen3:4b"
    assert client.prompts and "analyze this" in client.prompts[0]
    assert "schema" in client.prompts[0].lower()


def test_callable_model_render_prompt_includes_schema_and_context():
    prompt = CallableModel.render_prompt(
        ModelRequest(
            (MultimodalInput("text", "question"),),
            system_instruction="Be careful.",
            context=("prior fact",),
            response_schema=MODEL_OUTPUT_SCHEMA,
        )
    )

    assert "Be careful." in prompt
    assert "Context: prior fact" in prompt
    assert "question" in prompt
    assert "concepts" in prompt
