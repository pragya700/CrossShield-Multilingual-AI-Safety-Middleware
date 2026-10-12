"""Narrow deterministic rules for Phase 10 generated-output verification."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputSafetyRule:
    id: str
    category: str
    patterns: tuple[re.Pattern[str], ...]
    weight: float
    explanation: str


def _p(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in patterns)


ALL_OUTPUT_SAFETY_RULES: tuple[OutputSafetyRule, ...] = (
    OutputSafetyRule(
        id="output_system_prompt_disclosure_001",
        category="system_prompt_disclosure",
        patterns=_p(
            r"\b(?:my|the)\s+(?:system|developer|hidden)\s+(?:prompt|instructions?)\s+"
            r"(?:is|are|says?|read|contain)\b",
        ),
        weight=0.8,
        explanation="Generated output appears to disclose system or developer instructions.",
    ),
    OutputSafetyRule(
        id="output_safety_bypass_guidance_001",
        category="safety_bypass_guidance",
        patterns=_p(
            r"\b(?:to|you can|first|step\s*\d+\s*:)\s+"
            r"(?:bypass|disable|turn\s+off|remove)\s+(?:the\s+)?"
            r"(?:safety|security)\s+(?:filters?|restrictions?|controls?|guardrails?)\b",
        ),
        weight=0.75,
        explanation="Generated output provides operational guidance to bypass safety controls.",
    ),
    OutputSafetyRule(
        id="output_harmful_operational_guidance_001",
        category="harmful_operational_guidance",
        patterns=_p(
            r"\b(?:to|you can|first|then|step\s*\d+\s*:)\b.{0,80}\b"
            r"(?:make|build|assemble|construct)\s+(?:an?\s+)?"
            r"(?:bomb|explosive|weapon|incendiary\s+device)\b",
        ),
        weight=0.8,
        explanation="Generated output provides operational guidance for harmful physical action.",
    ),
)
