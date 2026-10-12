from app.llm.base import LLMRequest
from app.llm.gateway import LLMGateway
from app.llm.providers.mock import MockLLMProvider


def test_mock_llm_gateway():
    provider = MockLLMProvider()
    gateway = LLMGateway(provider)

    request = LLMRequest(
        prompt="What is machine learning?"
    )

    response = gateway.generate(request)

    assert response.provider == "mock"
    assert response.model == "mock-model"
    assert "What is machine learning?" in response.content


def test_llm_request_with_system_prompt():
    provider = MockLLMProvider()
    gateway = LLMGateway(provider)

    request = LLMRequest(
        prompt="Explain recursion.",
        system_prompt="You are a helpful assistant.",
        temperature=0.5,
        max_tokens=100
    )

    response = gateway.generate(request)

    assert response.provider == "mock"
    assert response.content is not None