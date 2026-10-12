"""Phase 9 schema contract tests for guarded LLM generation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.schemas import (
    AnalyzeRequest,
    Decision,
    GenerateRequest,
    GenerateResponse,
    RiskLevel,
)


def test_valid_generation_request():
    request = GenerateRequest(
        prompt="Explain recursion.",
        system_prompt="You are a helpful assistant.",
        temperature=0.5,
        max_tokens=100,
    )

    assert request.prompt == "Explain recursion."
    assert request.system_prompt == "You are a helpful assistant."
    assert request.temperature == 0.5
    assert request.max_tokens == 100


def test_generation_request_default_temperature():
    assert GenerateRequest(prompt="Hello").temperature == 0.7


def test_generation_request_custom_temperature():
    assert GenerateRequest(prompt="Hello", temperature=1.4).temperature == 1.4


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_generation_request_rejects_invalid_temperature(temperature: float):
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1, 4097])
def test_generation_request_rejects_invalid_max_tokens(max_tokens: int):
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", max_tokens=max_tokens)


def test_generation_response_includes_complete_analysis_response():
    response = GenerateResponse(
        request_id="request-123",
        language="en",
        language_confidence=0.99,
        original_text="Hello",
        normalized_text="Hello",
        decision=Decision.ALLOW,
        risk_level=RiskLevel.LOW,
        overall_risk=0.0,
        llm_called=True,
        llm_response="Mock LLM response to: Hello",
        llm_provider="mock",
        llm_model="mock-model",
    )

    assert response.request_id == "request-123"
    assert response.decision is Decision.ALLOW
    assert response.risk_level is RiskLevel.LOW
    assert response.overall_risk == 0.0
    assert response.llm_called is True
    assert response.llm_response == "Mock LLM response to: Hello"
    assert response.llm_provider == "mock"
    assert response.llm_model == "mock-model"
    assert response.blocked_reason is None
    assert response.risk_breakdown.analysis_status == "complete"


def test_analyze_request_remains_backward_compatible():
    request = AnalyzeRequest(prompt="Existing analysis request")

    assert request.prompt == "Existing analysis request"
    assert request.conversation_id is None
    assert request.conversation is None
