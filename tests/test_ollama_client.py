import os

import pytest

from ai.ollama_client import OllamaClient


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="requires a running local Ollama server; set RUN_OLLAMA_INTEGRATION=1",
)
def test_qwen_connection():

    client = OllamaClient()

    response = client.generate(
        "أجب بجملة واحدة فقط: ما هو الذكاء الاصطناعي؟"
    )

    print("\nQWEN RESPONSE:")
    print(response)

    assert response
