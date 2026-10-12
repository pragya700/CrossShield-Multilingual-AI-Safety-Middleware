from typing import Any

from pydantic import BaseModel, Field


class DashboardSummary(BaseModel):
    total_requests: int
    completed: int
    input_blocked: int
    output_blocked: int
    llm_errors: int

    risk_level_distribution: dict[str, int]
    input_decision_distribution: dict[str, int]
    output_safety_distribution: dict[str, int]
    language_distribution: dict[str, int]

    top_attack_categories: list[Any]
    top_output_safety_categories: list[Any]

    average_latency_ms: float
    average_overall_risk: float


class DashboardEvent(BaseModel):
    occurred_at: str
    request_id: str
    outcome: str

    language: str | None = None

    input_risk_level: str | None = None
    input_decision: str | None = None
    overall_risk: float | None = None

    llm_called: bool
    llm_provider: str | None = None
    llm_model: str | None = None

    output_safety_checked: bool
    output_safety_decision: str | None = None
    output_safety_score: float | None = None

    attack_categories: list[str] = Field(default_factory=list)
    matched_signals: list[str] = Field(default_factory=list)

    output_categories: list[str] = Field(default_factory=list)
    output_matched_signals: list[str] = Field(default_factory=list)

    blocked_reason: str | None = None
    request_latency_ms: float | None = None


class DashboardTrendPoint(BaseModel):
    bucket: str
    request_count: int
    average_overall_risk: float
    blocked_count: int