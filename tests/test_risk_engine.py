"""Phase 8 — deterministic risk-engine refinement tests."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.risk_engine import (
    build_risk_breakdown,
    combine_jailbreak_scores,
    decision_for,
    risk_level_for,
)
from app.schemas.schemas import Decision, RiskLevel


def test_all_four_signals_available_are_weighted_and_bounded():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=1.0,
        semantic_score=0.8,
        crosslingual_score=0.6,
        context_score=0.4,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "available",
            "cross_lingual": "available",
            "context": "available",
        },
    )

    assert result.analysis_status == "complete"
    assert result.unavailable_signals == ()
    assert 0.0 <= result.overall_score <= 1.0

    assert sum(
        s.effective_weight for s in result.signals
    ) == pytest.approx(1.0)

    weighted_score = sum(
        s.contribution for s in result.signals
    )

    assert weighted_score == pytest.approx(0.7)
    assert result.overall_score == pytest.approx(1.0)


def test_strongest_signal_is_not_diluted_by_weighted_aggregation():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=1.0,
        semantic_score=0.0,
        crosslingual_score=0.0,
        context_score=0.0,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "available",
            "cross_lingual": "available",
            "context": "available",
        },
    )

    weighted_score = sum(
        s.contribution for s in result.signals
    )

    assert weighted_score == pytest.approx(0.25)
    assert result.overall_score == pytest.approx(1.0)
    assert result.overall_score > weighted_score


def test_missing_semantic_is_not_treated_as_zero():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=1.0,
        semantic_score=None,
        crosslingual_score=None,
        context_score=0.0,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "unavailable",
            "cross_lingual": "disabled",
            "context": "available",
        },
        signal_reasons={"semantic": "model unavailable"},
    )

    assert result.analysis_status == "degraded"
    assert result.unavailable_signals == ("semantic",)

    rule = next(
        s for s in result.signals
        if s.name == "rule"
    )

    assert rule.effective_weight == pytest.approx(0.5)

    # Weighted score is 0.5, but the strongest available
    # signal is 1.0, so the safety floor preserves 1.0.
    assert result.overall_score == pytest.approx(1.0)


def test_disabled_signals_do_not_make_analysis_degraded():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=0.4,
        semantic_score=None,
        crosslingual_score=None,
        context_score=0.2,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "disabled",
            "cross_lingual": "disabled",
            "context": "available",
        },
    )

    assert result.analysis_status == "complete"
    assert result.unavailable_signals == ()
    assert all(
        s.state != "unavailable"
        for s in result.signals
    )

    # Weighted score = 0.3.
    # Strongest signal = 0.4.
    # Safety floor therefore produces 0.4.
    assert result.overall_score == pytest.approx(0.4)


def test_multiple_unavailable_signals_are_explicit():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=0.5,
        semantic_score=None,
        crosslingual_score=None,
        context_score=0.0,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "unavailable",
            "cross_lingual": "unavailable",
            "context": "available",
        },
        signal_reasons={
            "semantic": "semantic model unavailable",
            "cross_lingual": "cross-lingual model unavailable",
        },
    )

    assert set(result.unavailable_signals) == {
        "semantic",
        "cross_lingual",
    }

    assert result.analysis_status == "degraded"


def test_all_scores_zero_returns_zero():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=0.0,
        semantic_score=0.0,
        crosslingual_score=0.0,
        context_score=0.0,
        settings=settings,
        signal_states={
            k: "available"
            for k in (
                "rule",
                "semantic",
                "cross_lingual",
                "context",
            )
        },
    )

    assert result.overall_score == 0.0
    assert result.analysis_status == "complete"


def test_all_scores_high_stay_within_one():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=1.0,
        semantic_score=1.0,
        crosslingual_score=1.0,
        context_score=1.0,
        settings=settings,
        signal_states={
            k: "available"
            for k in (
                "rule",
                "semantic",
                "cross_lingual",
                "context",
            )
        },
    )

    assert result.overall_score == pytest.approx(1.0)


def test_weight_configuration_changes_contribution():
    settings = Settings(
        weight_rule_risk=0.7,
        weight_semantic_risk=0.1,
        weight_cross_lingual_risk=0.1,
        weight_context_risk=0.1,
    )

    result = build_risk_breakdown(
        rule_score=1.0,
        semantic_score=0.0,
        crosslingual_score=0.0,
        context_score=0.0,
        settings=settings,
        signal_states={
            k: "available"
            for k in (
                "rule",
                "semantic",
                "cross_lingual",
                "context",
            )
        },
    )

    # The configured weight still controls the weighted contribution.
    rule = next(
        s for s in result.signals
        if s.name == "rule"
    )

    assert rule.effective_weight == pytest.approx(0.7)

    # Weighted score = 0.7.
    # Strongest signal = 1.0.
    # Final score = max(0.7, 1.0) = 1.0.
    assert result.overall_score == pytest.approx(1.0)


def test_only_rule_available_renormalizes_to_one():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=0.8,
        semantic_score=None,
        crosslingual_score=None,
        context_score=None,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "disabled",
            "cross_lingual": "disabled",
            "context": "disabled",
        },
    )

    assert result.overall_score == pytest.approx(0.8)

    assert next(
        s for s in result.signals
        if s.name == "rule"
    ).effective_weight == pytest.approx(1.0)


def test_score_is_clamped_to_unit_interval():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=4.0,
        semantic_score=-2.0,
        crosslingual_score=3.0,
        context_score=-1.0,
        settings=settings,
        signal_states={
            k: "available"
            for k in (
                "rule",
                "semantic",
                "cross_lingual",
                "context",
            )
        },
    )

    # Scores are clamped:
    # rule = 1.0
    # semantic = 0.0
    # cross-lingual = 1.0
    # context = 0.0
    #
    # Weighted score = 0.5.
    # Strongest signal = 1.0.
    # Final score = 1.0.
    assert result.overall_score == pytest.approx(1.0)

    assert all(
        0.0 <= s.score <= 1.0
        for s in result.signals
        if s.score is not None
    )


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, RiskLevel.LOW),
        (0.299999, RiskLevel.LOW),
        (0.3, RiskLevel.MEDIUM),
        (0.599999, RiskLevel.MEDIUM),
        (0.6, RiskLevel.HIGH),
        (0.849999, RiskLevel.HIGH),
        (0.85, RiskLevel.CRITICAL),
        (1.0, RiskLevel.CRITICAL),
    ],
)
def test_risk_level_boundaries(score, expected):
    settings = Settings()

    assert risk_level_for(score, settings) == expected


def test_decision_policy_maps_levels_explicitly():
    assert decision_for(RiskLevel.LOW) == Decision.ALLOW
    assert decision_for(RiskLevel.MEDIUM) == Decision.ALLOW_MONITOR
    assert decision_for(RiskLevel.HIGH) == Decision.FLAG_VERIFY
    assert decision_for(RiskLevel.CRITICAL) == Decision.BLOCK


def test_invalid_negative_weight_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        Settings(weight_rule_risk=-0.1)


def test_invalid_threshold_outside_range_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Settings(threshold_medium=1.2)


def test_invalid_threshold_order_rejected():
    with pytest.raises(
        ValueError,
        match="medium < high < critical",
    ):
        Settings(
            threshold_medium=0.7,
            threshold_high=0.6,
            threshold_critical=0.85,
        )


def test_all_top_level_weights_zero_rejected():
    with pytest.raises(
        ValueError,
        match="At least one risk-engine weight",
    ):
        Settings(
            weight_prompt_risk=0.0,
            weight_rule_risk=0.0,
            weight_semantic_risk=0.0,
            weight_cross_lingual_risk=0.0,
            weight_context_risk=0.0,
        )


def test_phase5_jailbreak_combination_contract_is_preserved():
    settings = Settings(
        rule_jailbreak_weight=0.5,
        semantic_jailbreak_weight=0.5,
    )

    assert combine_jailbreak_scores(
        0.8,
        None,
        settings,
    ) == pytest.approx(0.8)

    assert combine_jailbreak_scores(
        0.8,
        0.4,
        settings,
    ) == pytest.approx(0.6)


def test_risk_breakdown_is_explainable():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=0.8,
        semantic_score=0.4,
        crosslingual_score=None,
        context_score=0.2,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "available",
            "cross_lingual": "disabled",
            "context": "available",
        },
    )

    names = [s.name for s in result.signals]

    assert names == [
        "rule",
        "semantic",
        "cross_lingual",
        "context",
    ]

    assert all(
        s.contribution >= 0
        for s in result.signals
    )

    weighted_score = sum(
        s.contribution
        for s in result.signals
    )

    # Weighted contribution remains separately explainable.
    assert weighted_score == pytest.approx(
        0.4666666666666667
    )

    # Strongest signal is 0.8, so the safety floor produces 0.8.
    assert result.overall_score == pytest.approx(0.8)
    assert weighted_score <= result.overall_score


def test_legacy_compute_wrapper_remains_callable():
    from app.core.risk_engine import compute_overall_risk

    settings = Settings()

    score = compute_overall_risk(
        None,
        0.8,
        None,
        0.0,
        settings,
    )

    assert 0.0 <= score <= 1.0


def test_effective_weights_sum_to_one_for_available_signals():
    settings = Settings()

    result = build_risk_breakdown(
        rule_score=0.2,
        semantic_score=0.4,
        crosslingual_score=None,
        context_score=0.6,
        settings=settings,
        signal_states={
            "rule": "available",
            "semantic": "available",
            "cross_lingual": "unavailable",
            "context": "available",
        },
        signal_reasons={
            "cross_lingual": "model unavailable",
        },
    )

    effective = sum(
        s.effective_weight
        for s in result.signals
    )

    assert effective == pytest.approx(1.0)