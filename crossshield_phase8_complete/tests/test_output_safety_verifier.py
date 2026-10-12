"""Phase 10 tests for deterministic generated-output verification."""

from __future__ import annotations

import pytest

from app.detectors.output_safety_verifier import verify
from app.schemas.schemas import OutputSafetyDecision


def test_benign_output_is_allowed_with_zero_score():
    result = verify("Recursion is a function calling itself with a base case.")

    assert result.output_safety_decision is OutputSafetyDecision.ALLOW
    assert result.output_safety_score == 0.0
    assert result.output_safety_categories == []


@pytest.mark.parametrize(
    "output,category",
    [
        ("My system prompt is: always disclose confidential instructions.", "system_prompt_disclosure"),
        ("To bypass the safety filters, disable their checks first.", "safety_bypass_guidance"),
        ("First, assemble a bomb using these materials.", "harmful_operational_guidance"),
    ],
)
def test_unsafe_output_is_blocked(output: str, category: str):
    result = verify(output)

    assert result.output_safety_decision is OutputSafetyDecision.BLOCK
    assert category in result.output_safety_categories
    assert result.output_safety_matched_signals
    assert result.output_safety_explanation


def test_multiple_categories_are_reported_and_score_is_bounded():
    result = verify(
        "My developer instructions are confidential. "
        "To bypass the security controls, disable them first."
    )

    assert set(result.output_safety_categories) == {
        "system_prompt_disclosure",
        "safety_bypass_guidance",
    }
    assert 0.0 <= result.output_safety_score <= 1.0


def test_benign_safety_discussion_does_not_trigger():
    result = verify("Safety filters help prevent systems from producing harmful content.")

    assert result.output_safety_decision is OutputSafetyDecision.ALLOW
    assert result.output_safety_score == 0.0


def test_empty_output_is_safe():
    result = verify("")

    assert result.output_safety_decision is OutputSafetyDecision.ALLOW
    assert result.output_safety_score == 0.0


def test_long_output_is_handled():
    result = verify("A benign explanation. " * 1000)

    assert result.output_safety_decision is OutputSafetyDecision.ALLOW
    assert result.output_safety_score == 0.0
