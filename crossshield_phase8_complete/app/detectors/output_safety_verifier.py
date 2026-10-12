"""Phase 10 deterministic verification of generated LLM output only."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.detectors.output_safety_rules import ALL_OUTPUT_SAFETY_RULES, OutputSafetyRule
from app.schemas.schemas import OutputSafetyDecision


OUTPUT_SAFETY_BLOCK_THRESHOLD = 0.5


@dataclass
class OutputSafetyResult:
    output_safety_score: float
    output_safety_categories: list[str] = field(default_factory=list)
    output_safety_matched_signals: list[str] = field(default_factory=list)
    output_safety_explanation: list[str] = field(default_factory=list)
    output_safety_decision: OutputSafetyDecision = OutputSafetyDecision.ALLOW


def _rules_matching(output: str) -> list[OutputSafetyRule]:
    return [
        rule
        for rule in ALL_OUTPUT_SAFETY_RULES
        if any(pattern.search(output) for pattern in rule.patterns)
    ]


def verify(output: str) -> OutputSafetyResult:
    """Return a bounded, explainable output-only safety decision."""
    if not output.strip():
        return OutputSafetyResult(output_safety_score=0.0)

    matched_rules = _rules_matching(output)
    category_weights: dict[str, float] = {}
    for rule in matched_rules:
        category_weights[rule.category] = max(
            category_weights.get(rule.category, 0.0), rule.weight
        )

    score = max(0.0, min(1.0, sum(category_weights.values())))
    decision = (
        OutputSafetyDecision.BLOCK
        if score >= OUTPUT_SAFETY_BLOCK_THRESHOLD
        else OutputSafetyDecision.ALLOW
    )

    categories: list[str] = []
    signals: list[str] = []
    explanations: list[str] = []
    for rule in matched_rules:
        if rule.category not in categories:
            categories.append(rule.category)
        signals.append(rule.id)
        explanations.append(rule.explanation)

    return OutputSafetyResult(
        output_safety_score=score,
        output_safety_categories=categories,
        output_safety_matched_signals=signals,
        output_safety_explanation=explanations,
        output_safety_decision=decision,
    )
