"""Internal metadata-only audit event contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.schemas.schemas import AnalyzeResponse


@dataclass(frozen=True)
class SecurityAuditEvent:
    event_id: str
    request_id: str
    occurred_at: str
    endpoint: str
    language: str
    input_risk_level: str
    input_decision: str
    overall_risk: float
    risk_analysis_status: str
    risk_breakdown: dict[str, Any]
    attack_categories: list[str] = field(default_factory=list)
    matched_signals: list[str] = field(default_factory=list)
    llm_called: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    output_safety_checked: bool = False
    output_safety_decision: Optional[str] = None
    output_safety_score: Optional[float] = None
    output_categories: list[str] = field(default_factory=list)
    output_matched_signals: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    request_latency_ms: float = 0.0
    outcome: str = "completed"

    @classmethod
    def for_generate(
        cls,
        analysis: AnalyzeResponse,
        *,
        llm_called: bool,
        llm_provider: Optional[str],
        llm_model: Optional[str],
        output_safety_checked: bool,
        output_safety_decision: Optional[str],
        output_safety_score: Optional[float],
        output_categories: list[str],
        output_matched_signals: list[str],
        blocked_reason: Optional[str],
        request_latency_ms: float,
        outcome: str,
    ) -> "SecurityAuditEvent":
        return cls(
            event_id=str(uuid4()),
            request_id=analysis.request_id,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            endpoint="/generate",
            language=analysis.language,
            input_risk_level=analysis.risk_level.value,
            input_decision=analysis.decision.value,
            overall_risk=analysis.overall_risk,
            risk_analysis_status=analysis.risk_analysis_status,
            risk_breakdown=analysis.risk_breakdown.model_dump(mode="json"),
            attack_categories=list(analysis.attack_categories),
            matched_signals=list(analysis.matched_signals),
            llm_called=llm_called,
            llm_provider=llm_provider,
            llm_model=llm_model,
            output_safety_checked=output_safety_checked,
            output_safety_decision=output_safety_decision,
            output_safety_score=output_safety_score,
            output_categories=list(output_categories),
            output_matched_signals=list(output_matched_signals),
            blocked_reason=blocked_reason,
            request_latency_ms=request_latency_ms,
            outcome=outcome,
        )
