"""
Pydantic models for CrossShield's public API contracts.

These schemas are intentionally verbose in Phase 1 (even though most scores
are stubbed) because the whole point of the MVP is to lock down the response
shape early — later phases fill in real detector logic behind this same
contract instead of changing it.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_MONITOR = "ALLOW_MONITOR"
    FLAG_VERIFY = "FLAG_VERIFY"
    BLOCK = "BLOCK"


class OutputSafetyDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationTurn(BaseModel):
    role: ConversationRole
    content: str = Field(..., min_length=1, max_length=8000)


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional ID to associate this prompt with prior turns. "
        "Not used by any detector; kept for client-side correlation only.",
    )
    # Phase 7: optional PRIOR conversation turns (NOT including `prompt`,
    # which is always treated as the final/current user turn). Omitting
    # this field entirely keeps every pre-Phase-7 request exactly as valid
    # as before -- this is a purely additive, backward-compatible change.
    conversation: Optional[List[ConversationTurn]] = Field(
        default=None,
        max_length=50,
        description="Optional prior conversation turns for Phase 7 context/"
        "reasoning analysis. Omit for a plain single-prompt request.",
    )


class GenerateRequest(AnalyzeRequest):
    """Analysis request plus the generation controls for Phase 9."""

    system_prompt: Optional[str] = Field(default=None, max_length=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)


class SemanticMatchInfo(BaseModel):
    """One semantic match: similarity to a known reference (NOT a probability)."""

    reference_id: str
    category: str
    similarity: float
    segment_index: int = 0


class CrossLingualMatchInfo(BaseModel):
    """One Phase 6 match: similarity to an aligned cross-lingual reference
    group (NOT a probability). `languages` lists every language that group's
    intent is expressed in within the reference dataset."""

    group_id: str
    reference_id: str
    category: str
    language: str
    similarity: float
    segment_index: int = 0


class ContextReasoningMatchInfo(BaseModel):
    """One Phase 7 match: a cross-turn contextual/reasoning pattern. `turn_indices`
    are 0-based positions into the analyzed user-turn sequence (prior
    conversation turns in order, then the current `prompt` last)."""

    category: str
    description: str
    turn_indices: List[int] = Field(default_factory=list)


class RiskSignalBreakdownInfo(BaseModel):
    """Explainable contribution of one Phase 8 risk signal."""

    name: str
    score: Optional[float] = None
    configured_weight: float = 0.0
    effective_weight: float = 0.0
    contribution: float = 0.0
    available: bool = False
    state: str = "disabled"  # available | disabled | unavailable
    reason: Optional[str] = None


class RiskBreakdownInfo(BaseModel):
    """Public Phase 8 risk aggregation breakdown."""

    analysis_status: str = "complete"  # complete | degraded
    unavailable_signals: List[str] = Field(default_factory=list)
    signals: List[RiskSignalBreakdownInfo] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    request_id: str
    language: str
    language_confidence: float
    languages_detected: List[str] = Field(default_factory=list)
    code_switching: bool = False

    original_text: str
    normalized_text: str
    transformations: List[str] = Field(default_factory=list)
    has_invisible_characters: bool = False
    has_repeated_characters: bool = False
    possible_transliteration_signal: bool = False

    prompt_risk: float = 0.0
    jailbreak_score: float = 0.0
    crosslingual_score: float = 0.0
    context_score: float = 0.0

    overall_risk: float = 0.0
    risk_breakdown: RiskBreakdownInfo = Field(default_factory=RiskBreakdownInfo)
    risk_analysis_status: str = "complete"
    unavailable_risk_signals: List[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    # `attack_categories` is the union of rule and semantic categories;
    # `matched_signals` are rule ids; `explanation` covers both detectors.
    attack_categories: List[str] = Field(default_factory=list)
    matched_signals: List[str] = Field(default_factory=list)
    explanation: List[str] = Field(default_factory=list)

    # Phase 5: the two detectors are exposed separately so the rule-based
    # baseline stays comparable. `jailbreak_score` (above) is the COMBINED score.
    rule_jailbreak_score: float = 0.0
    rule_attack_categories: List[str] = Field(default_factory=list)
    semantic_status: str = "disabled"  # "enabled" | "disabled"
    # None (not 0.0) when the semantic detector did not run.
    semantic_jailbreak_score: Optional[float] = None
    semantic_attack_categories: List[str] = Field(default_factory=list)
    semantic_matches: List[SemanticMatchInfo] = Field(default_factory=list)
    # Raw best similarity to an attack reference, for threshold calibration.
    semantic_max_similarity: Optional[float] = None
    semantic_hard_negative_suppressed: Optional[bool] = None
    semantic_model: Optional[str] = None

    # Phase 6: cross-lingual semantic safety analysis. `crosslingual_score`
    # (above, no underscore -- the original Phase-1 placeholder field name,
    # kept unchanged rather than introducing a second near-duplicate field)
    # is now the REAL Phase 6 score feeding the risk engine's existing
    # weight_crosslingual_risk, exactly mirroring how `jailbreak_score` is
    # the combined Phase 4+5 score. The fields below (new in Phase 6, hence
    # the underscore) carry the structured detail behind that number.
    cross_lingual_status: str = "disabled"  # "enabled" | "disabled"
    cross_lingual_attack_categories: List[str] = Field(default_factory=list)
    cross_lingual_matches: List[CrossLingualMatchInfo] = Field(default_factory=list)
    # Every language the best-matched aligned group's intent is confirmed to
    # be expressed in, per the reference dataset (not a language-detection
    # result -- see language/languages_detected above for that).
    cross_lingual_languages: List[str] = Field(default_factory=list)
    cross_lingual_max_similarity: Optional[float] = None
    cross_lingual_suppressed: Optional[bool] = None
    cross_lingual_model: Optional[str] = None

    # Phase 7: deterministic context/reasoning analysis across conversation
    # turns. `context_score` (above, no underscore -- the original Phase-1
    # placeholder field name, unchanged) is now the REAL Phase 7 score
    # feeding the risk engine's existing weight_context_risk, exactly
    # mirroring crosslingual_score's role for Phase 6. Unlike Phases 5-6,
    # this analyzer has no model to download and therefore no "unavailable"
    # failure state -- context_reasoning_status only ever reflects the
    # config toggle.
    context_reasoning_status: str = "disabled"  # "enabled" | "disabled"
    context_reasoning_categories: List[str] = Field(default_factory=list)
    context_reasoning_matches: List[ContextReasoningMatchInfo] = Field(default_factory=list)
    context_reasoning_turns_analyzed: int = 0

    decision: Decision = Decision.ALLOW

    latency_ms: float = 0.0
    notes: Optional[str] = None


class GenerateResponse(AnalyzeResponse):
    """Full analysis result together with Phase 9 generation metadata."""

    llm_called: bool
    llm_response: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    blocked_reason: Optional[str] = None
    output_safety_score: float = 0.0
    output_safety_categories: List[str] = Field(default_factory=list)
    output_safety_matched_signals: List[str] = Field(default_factory=list)
    output_safety_explanation: List[str] = Field(default_factory=list)
    output_safety_decision: OutputSafetyDecision = OutputSafetyDecision.ALLOW
