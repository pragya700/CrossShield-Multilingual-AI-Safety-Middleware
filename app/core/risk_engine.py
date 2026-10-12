"""CrossShield Phase 8 risk aggregation and decision policy.

The risk engine keeps the four real safety signals independently visible:
rule, semantic, cross-lingual, and context/reasoning. The legacy combined
``jailbreak_score`` remains a Phase 5 API field, but Phase 8 does not use it
as a substitute for its two underlying signals.

``None`` means a detector did not provide a result. ``0.0`` means it ran and
found no risk.

Available weights are normalized so an unavailable detector does not silently
contribute a zero.

Phase 8 additionally applies a safety-floor rule:

    final_score = max(weighted_score, strongest_available_signal)

This prevents a high-confidence signal from being diluted merely because
another detector returned a low score. The weighted score remains useful for
combining multiple independent signals, while the strongest signal preserves
high-severity evidence.

Defaults are engineering baselines, not empirically optimized values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.core.config import Settings
from app.schemas.schemas import Decision, RiskLevel


@dataclass(frozen=True)
class RiskSignal:
    name: str
    score: float | None
    configured_weight: float
    state: str  # available | disabled | unavailable
    reason: str | None = None


@dataclass(frozen=True)
class RiskSignalBreakdown:
    name: str
    score: float | None
    configured_weight: float
    effective_weight: float
    contribution: float
    available: bool
    state: str
    reason: str | None = None


@dataclass(frozen=True)
class RiskBreakdown:
    overall_score: float
    signals: tuple[RiskSignalBreakdown, ...]
    unavailable_signals: tuple[str, ...]
    analysis_status: str


def _clamp_score(score: float) -> float:
    """Keep every risk score inside the [0, 1] interval."""
    return max(0.0, min(1.0, float(score)))


def combine_jailbreak_scores(
    rule_score: float,
    semantic_score: float | None,
    settings: Settings,
) -> float:
    """Preserve the Phase 5 combined jailbreak-score contract."""
    rule = _clamp_score(rule_score)

    if semantic_score is None:
        return rule

    semantic = _clamp_score(semantic_score)

    return _clamp_score(
        settings.rule_jailbreak_weight * rule
        + settings.semantic_jailbreak_weight * semantic
    )


def build_risk_breakdown(
    *,
    rule_score: float | None,
    semantic_score: float | None,
    crosslingual_score: float | None,
    context_score: float | None,
    settings: Settings,
    signal_states: Mapping[str, str] | None = None,
    signal_reasons: Mapping[str, str] | None = None,
) -> RiskBreakdown:
    """Aggregate the four Phase 8 safety signals.

    The unfinished Prompt Risk Analyzer is deliberately not part of the
    aggregation yet. Its old configuration field remains in Settings for
    backward compatibility, but no fake zero is inserted into the score.

    Two calculations are performed:

    1. Weighted aggregation:
       available detector scores are combined using their configured weights,
       normalized over the detectors that actually ran.

    2. Safety floor:
       the strongest available detector score is preserved so that a
       high-confidence signal cannot be diluted by another detector that
       simply returned a low or zero score.

    Therefore:

        weighted_score = 0.50
        strongest_signal = 1.00
        final_score = 1.00

    This keeps the multi-signal weighted model while preventing severe
    individual evidence from disappearing inside the average.
    """

    states = signal_states or {}
    reasons = signal_reasons or {}

    signals = (
        RiskSignal(
            "rule",
            rule_score,
            settings.weight_rule_risk,
            states.get(
                "rule",
                "available" if rule_score is not None else "disabled",
            ),
            reasons.get("rule"),
        ),
        RiskSignal(
            "semantic",
            semantic_score,
            settings.weight_semantic_risk,
            states.get(
                "semantic",
                "available" if semantic_score is not None else "disabled",
            ),
            reasons.get("semantic"),
        ),
        RiskSignal(
            "cross_lingual",
            crosslingual_score,
            settings.weight_cross_lingual_risk,
            states.get(
                "cross_lingual",
                "available"
                if crosslingual_score is not None
                else "disabled",
            ),
            reasons.get("cross_lingual"),
        ),
        RiskSignal(
            "context",
            context_score,
            settings.weight_context_risk,
            states.get(
                "context",
                "available" if context_score is not None else "disabled",
            ),
            reasons.get("context"),
        ),
    )

    # Only signals that actually ran and have a positive configured weight
    # participate in weighted aggregation.
    available = [
        signal
        for signal in signals
        if (
            signal.state == "available"
            and signal.score is not None
            and signal.configured_weight > 0
        )
    ]

    configured_total = sum(
        signal.configured_weight for signal in available
    )

    breakdowns: list[RiskSignalBreakdown] = []
    unavailable: list[str] = []

    weighted_overall = 0.0
    strongest_signal_score = 0.0

    for signal in signals:
        if signal.state != "available" or signal.score is None:
            if signal.state == "unavailable":
                unavailable.append(signal.name)

            breakdowns.append(
                RiskSignalBreakdown(
                    name=signal.name,
                    score=None,
                    configured_weight=signal.configured_weight,
                    effective_weight=0.0,
                    contribution=0.0,
                    available=False,
                    state=signal.state,
                    reason=signal.reason,
                )
            )

            continue

        score = _clamp_score(signal.score)

        # Keep track of the strongest real detector evidence.
        strongest_signal_score = max(
            strongest_signal_score,
            score,
        )

        effective_weight = (
            signal.configured_weight / configured_total
            if configured_total
            else 0.0
        )

        contribution = score * effective_weight
        weighted_overall += contribution

        breakdowns.append(
            RiskSignalBreakdown(
                name=signal.name,
                score=score,
                configured_weight=signal.configured_weight,
                effective_weight=effective_weight,
                contribution=contribution,
                available=True,
                state="available",
                reason=signal.reason,
            )
        )

    # ---------------------------------------------------------------
    # Phase 8 safety floor
    # ---------------------------------------------------------------
    #
    # Do not allow a high-confidence individual signal to disappear
    # inside the weighted average.
    #
    # Example:
    #
    #   rule    = 1.0
    #   context = 0.0
    #
    #   weighted score   = 0.5
    #   strongest signal = 1.0
    #
    #   final score      = 1.0
    #
    final_overall = max(
        weighted_overall,
        strongest_signal_score,
    )

    final_overall = _clamp_score(final_overall)

    return RiskBreakdown(
        overall_score=final_overall,
        signals=tuple(breakdowns),
        unavailable_signals=tuple(unavailable),
        analysis_status="degraded" if unavailable else "complete",
    )


def compute_overall_risk(
    prompt_risk: float | None,
    jailbreak_score: float | None,
    crosslingual_score: float | None,
    context_score: float | None,
    settings: Settings,
) -> float:
    """Legacy-compatible wrapper using combined jailbreak as one signal.

    New Phase 8 route code should use ``build_risk_breakdown`` directly so
    rule and semantic signals remain independently weighted.
    """
    return build_risk_breakdown(
        rule_score=jailbreak_score,
        semantic_score=None,
        crosslingual_score=crosslingual_score,
        context_score=context_score,
        settings=settings,
        signal_states={
            "rule": (
                "available"
                if jailbreak_score is not None
                else "disabled"
            ),
            "semantic": "disabled",
        },
    ).overall_score


def risk_level_for(
    score: float,
    settings: Settings,
) -> RiskLevel:
    """Map an overall risk score to the configured risk level."""
    score = _clamp_score(score)

    if score >= settings.threshold_critical:
        return RiskLevel.CRITICAL

    if score >= settings.threshold_high:
        return RiskLevel.HIGH

    if score >= settings.threshold_medium:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def decision_for(level: RiskLevel) -> Decision:
    """Map a risk level to the operational CrossShield decision."""
    return {
        RiskLevel.LOW: Decision.ALLOW,
        RiskLevel.MEDIUM: Decision.ALLOW_MONITOR,
        RiskLevel.HIGH: Decision.FLAG_VERIFY,
        RiskLevel.CRITICAL: Decision.BLOCK,
    }[level]