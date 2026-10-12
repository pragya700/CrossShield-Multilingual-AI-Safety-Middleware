from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.audit.security_event import SecurityAuditEvent
from app.audit.sqlite_store import SQLiteAuditStore
from app.core.config import get_settings
from app.core.risk_engine import (
    combine_jailbreak_scores,
    build_risk_breakdown,
    decision_for,
    risk_level_for,
)
from app.detectors.context_reasoning_analyzer import ConversationMessage
from app.detectors.context_reasoning_analyzer import analyze as analyze_context_reasoning
from app.detectors.cross_lingual_analyzer import get_cross_lingual_analyzer
from app.detectors.jailbreak_detector import detect as detect_jailbreak
from app.detectors.language_detector import detect_language
from app.detectors.normalizer import normalize
from app.detectors.output_safety_verifier import verify as verify_output_safety
from app.detectors.semantic_jailbreak_detector import (
    SemanticModelUnavailableError,
    get_semantic_detector,
)
from app.llm.base import LLMRequest
from app.llm.gateway import LLMGateway
from app.llm.providers.mock import MockLLMProvider
from app.schemas.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Decision,
    GenerateRequest,
    GenerateResponse,
    RiskBreakdownInfo,
    RiskSignalBreakdownInfo,
    ContextReasoningMatchInfo,
    CrossLingualMatchInfo,
    SemanticMatchInfo,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _persist_generate_audit(
    analysis: AnalyzeResponse,
    *,
    llm_called: bool,
    llm_provider: str | None,
    llm_model: str | None,
    output_safety_checked: bool,
    output_safety_decision: str | None,
    output_safety_score: float | None,
    output_categories: list[str],
    output_matched_signals: list[str],
    blocked_reason: str | None,
    request_latency_ms: float,
    outcome: str,
) -> None:
    """Persist metadata only; audit failures must never affect the request."""
    settings = get_settings()
    if not settings.security_audit_enabled:
        return

    event = SecurityAuditEvent.for_generate(
        analysis,
        llm_called=llm_called,
        llm_provider=llm_provider,
        llm_model=llm_model,
        output_safety_checked=output_safety_checked,
        output_safety_decision=output_safety_decision,
        output_safety_score=output_safety_score,
        output_categories=output_categories,
        output_matched_signals=output_matched_signals,
        blocked_reason=blocked_reason,
        request_latency_ms=request_latency_ms,
        outcome=outcome,
    )
    try:
        SQLiteAuditStore(settings.security_audit_db_path).write_event(event)
    except Exception as exc:
        logger.warning("Security audit persistence failed: %s", type(exc).__name__)


def _analyze_request(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the unchanged Phase 1-8 analysis pipeline for either endpoint."""
    settings = get_settings()
    start = time.perf_counter()

    lang_result = detect_language(request.prompt)
    norm_result = normalize(request.prompt)
    jailbreak_result = detect_jailbreak(request.prompt, norm_result.normalized_text)

    # Phase 5: the semantic detector runs alongside (never instead of) the
    # rule-based one. If it is enabled but its model cannot be loaded we fail
    # loudly with 503 rather than report a fabricated semantic score of 0.0.
    semantic_result = None
    unavailable_reasons: dict[str, str] = {}
    if settings.semantic_enabled:
        try:
            semantic_result = get_semantic_detector().detect(
                request.prompt, norm_result.normalized_text
            )
        except SemanticModelUnavailableError as exc:
            if not settings.risk_degraded_mode_enabled:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "semantic_model_unavailable", "message": str(exc)},
                ) from exc
            unavailable_reasons["semantic"] = f"Semantic model unavailable: {exc}"

    # Phase 6: cross-lingual semantic safety analysis. Only attempted when
    # semantic detection is ALSO enabled -- it reuses Phase 5's already-
    # loaded embedder (see cross_lingual_analyzer.build_cross_lingual_analyzer)
    # rather than loading a second copy of the model, so without Phase 5
    # running there is nothing for it to reuse. Same fail-loud contract as
    # Phase 5: a 503 with a clear reason, never a fabricated 0.0.
    cross_lingual_result = None
    if settings.semantic_enabled and settings.cross_lingual_enabled and semantic_result is not None:
        try:
            cross_lingual_result = get_cross_lingual_analyzer().detect(
                request.prompt, norm_result.normalized_text
            )
        except SemanticModelUnavailableError as exc:
            if not settings.risk_degraded_mode_enabled:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "semantic_model_unavailable", "message": str(exc)},
                ) from exc
            unavailable_reasons["cross_lingual"] = f"Cross-lingual model unavailable: {exc}"
        except Exception as exc:  # e.g. ReferenceDataError from a bad dataset
            if not settings.risk_degraded_mode_enabled:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "cross_lingual_unavailable", "message": str(exc)},
                ) from exc
            unavailable_reasons["cross_lingual"] = f"Cross-lingual analysis unavailable: {exc}"
    elif settings.semantic_enabled and settings.cross_lingual_enabled and semantic_result is None and settings.risk_degraded_mode_enabled:
        unavailable_reasons["cross_lingual"] = "Cross-lingual analysis requires the Phase 5 semantic embedder."

    # Phase 7: context/reasoning analysis across conversation turns. Pure
    # local deterministic logic -- no model, so no 503/unavailable path
    # exists for this one (see context_reasoning_analyzer.py docstring).
    context_result = None
    if settings.context_reasoning_enabled:
        conversation_messages = (
            [ConversationMessage(role=t.role.value, content=t.content) for t in request.conversation]
            if request.conversation
            else None
        )
        context_result = analyze_context_reasoning(
            request.prompt, norm_result.normalized_text, conversation_messages
        )

    rule_score = jailbreak_result.jailbreak_score
    semantic_score = semantic_result.semantic_jailbreak_score if semantic_result else None

    # Real signals so far: language, normalization, rule score, semantic
    # score (when enabled), cross-lingual score (when enabled), context/
    # reasoning score (when enabled). prompt_risk (a separate, still
    # unbuilt Prompt Risk Analyzer) remains an honest 0.0 placeholder.
    # Phase 8: preserve the public prompt_risk field as 0.0 for backward
    # compatibility, but pass None into the risk engine because the dedicated
    # Prompt Risk Analyzer has not been implemented yet. None means
    # "unavailable", not "evaluated and safe".
    prompt_risk = 0.0
    jailbreak_score = combine_jailbreak_scores(rule_score, semantic_score, settings)
    crosslingual_score = (
        cross_lingual_result.cross_lingual_score if cross_lingual_result is not None else 0.0
    )
    context_score = context_result.context_reasoning_score if context_result is not None else 0.0

    signal_states = {
        "rule": "available",
        "semantic": (
            "unavailable" if "semantic" in unavailable_reasons
            else "available" if semantic_result is not None
            else "disabled"
        ),
        "cross_lingual": (
            "unavailable" if "cross_lingual" in unavailable_reasons
            else "available" if cross_lingual_result is not None
            else "disabled"
        ),
        "context": "available" if context_result is not None else "disabled",
    }

    risk_breakdown = build_risk_breakdown(
        rule_score=rule_score,
        semantic_score=semantic_score,
        crosslingual_score=(
            cross_lingual_result.cross_lingual_score
            if cross_lingual_result is not None else None
        ),
        context_score=(
            context_result.context_reasoning_score
            if context_result is not None else None
        ),
        settings=settings,
        signal_reasons=unavailable_reasons,
        signal_states=signal_states,
    )

    attack_categories = list(jailbreak_result.attack_categories)
    explanation = list(jailbreak_result.explanation)
    if semantic_result is not None:
        for category in semantic_result.attack_categories:
            if category not in attack_categories:
                attack_categories.append(category)
        explanation.extend(semantic_result.explanation)
    if cross_lingual_result is not None:
        for category in cross_lingual_result.attack_categories:
            if category not in attack_categories:
                attack_categories.append(category)
        explanation.extend(cross_lingual_result.explanation)
    if context_result is not None:
        for category in context_result.categories:
            if category not in attack_categories:
                attack_categories.append(category)
        explanation.extend(context_result.explanation)

    overall_risk = risk_breakdown.overall_score
    level = risk_level_for(overall_risk, settings)
    decision = decision_for(level)

    latency_ms = (time.perf_counter() - start) * 1000

    notes = (
        "Real signals: language (Phase 2), normalization (Phase 3), rule-based "
        "jailbreak score (Phase 4)"
        + (", semantic jailbreak score (Phase 5)" if semantic_result else "")
        + (", cross-lingual semantic score (Phase 6)" if cross_lingual_result else "")
        + (", context/reasoning score (Phase 7)" if context_result else "")
        + ". "
        + (
            "Semantic detection is disabled, so jailbreak_score equals the rule score. "
            if not semantic_result
            else ""
        )
        + (
            "Cross-lingual analysis is disabled, so crosslingual_score is 0.0. "
            if not cross_lingual_result
            else ""
        )
        + (
            "Context/reasoning analysis is disabled, so context_score is 0.0. "
            if not context_result
            else ""
        )
        + "Phase 8 risk aggregation keeps rule, semantic, cross-lingual, and "
        "context signals separate; unavailable signals are not treated as zero. "
        + ("Analysis is in explicit degraded mode. " if risk_breakdown.analysis_status == "degraded" else "")
        + "prompt_risk is still a 0.0 public placeholder pending a future Prompt Risk "
        "Analyzer. Semantic and cross-lingual values are similarity scores, "
        "not probabilities. Context/reasoning is deterministic rule-based "
        "pattern matching across turns, not a reasoning model."
    )

    return AnalyzeResponse(
        request_id=str(uuid.uuid4()),
        language=lang_result.primary_language,
        language_confidence=lang_result.confidence,
        languages_detected=lang_result.languages_detected,
        code_switching=lang_result.code_switching,
        original_text=norm_result.original_text,
        normalized_text=norm_result.normalized_text,
        transformations=norm_result.transformations,
        has_invisible_characters=norm_result.has_invisible_characters,
        has_repeated_characters=norm_result.has_repeated_characters,
        possible_transliteration_signal=norm_result.possible_transliteration_signal,
        prompt_risk=prompt_risk,
        jailbreak_score=jailbreak_score,
        crosslingual_score=crosslingual_score,
        context_score=context_score,
        overall_risk=overall_risk,
        risk_breakdown=RiskBreakdownInfo(
            analysis_status=risk_breakdown.analysis_status,
            unavailable_signals=list(risk_breakdown.unavailable_signals),
            signals=[
                RiskSignalBreakdownInfo(
                    name=s.name,
                    score=s.score,
                    configured_weight=s.configured_weight,
                    effective_weight=s.effective_weight,
                    contribution=s.contribution,
                    available=s.available,
                    state=s.state,
                    reason=s.reason,
                )
                for s in risk_breakdown.signals
            ],
        ),
        risk_analysis_status=risk_breakdown.analysis_status,
        unavailable_risk_signals=list(risk_breakdown.unavailable_signals),
        risk_level=level,
        attack_categories=attack_categories,
        matched_signals=jailbreak_result.matched_signals,
        explanation=explanation,
        rule_jailbreak_score=rule_score,
        rule_attack_categories=list(jailbreak_result.attack_categories),
        semantic_status="enabled" if semantic_result is not None else "disabled",
        semantic_jailbreak_score=semantic_score,
        semantic_attack_categories=(
            list(semantic_result.attack_categories) if semantic_result else []
        ),
        semantic_matches=(
            [
                SemanticMatchInfo(
                    reference_id=m.reference_id,
                    category=m.category,
                    similarity=m.similarity,
                    segment_index=m.segment_index,
                )
                for m in semantic_result.matches
            ]
            if semantic_result
            else []
        ),
        semantic_max_similarity=(
            semantic_result.max_similarity if semantic_result else None
        ),
        semantic_hard_negative_suppressed=(
            semantic_result.hard_negative_suppressed if semantic_result else None
        ),
        semantic_model=settings.semantic_model_name if semantic_result else None,
        cross_lingual_status="enabled" if cross_lingual_result is not None else "disabled",
        cross_lingual_attack_categories=(
            list(cross_lingual_result.attack_categories) if cross_lingual_result else []
        ),
        cross_lingual_matches=(
            [
                CrossLingualMatchInfo(
                    group_id=m.group_id,
                    reference_id=m.reference_id,
                    category=m.category,
                    language=m.language,
                    similarity=m.similarity,
                    segment_index=m.segment_index,
                )
                for m in cross_lingual_result.matches
            ]
            if cross_lingual_result
            else []
        ),
        cross_lingual_languages=(
            list(cross_lingual_result.languages) if cross_lingual_result else []
        ),
        cross_lingual_max_similarity=(
            cross_lingual_result.max_similarity if cross_lingual_result else None
        ),
        cross_lingual_suppressed=(
            cross_lingual_result.suppressed if cross_lingual_result else None
        ),
        cross_lingual_model=settings.semantic_model_name if cross_lingual_result else None,
        context_reasoning_status="enabled" if context_result is not None else "disabled",
        context_reasoning_categories=(
            list(context_result.categories) if context_result else []
        ),
        context_reasoning_matches=(
            [
                ContextReasoningMatchInfo(
                    category=m.category,
                    description=m.description,
                    turn_indices=list(m.turn_indices),
                )
                for m in context_result.matches
            ]
            if context_result
            else []
        ),
        context_reasoning_turns_analyzed=(
            context_result.turns_analyzed if context_result else 0
        ),
        decision=decision,
        latency_ms=latency_ms,
        notes=notes,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return _analyze_request(request)


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    request_start = time.perf_counter()
    analysis = _analyze_request(request)
    response_data = analysis.model_dump()

    if analysis.decision is Decision.BLOCK:
        response = GenerateResponse(
            **response_data,
            llm_called=False,
            llm_response=None,
            llm_provider=None,
            llm_model=None,
            blocked_reason=(
                "Generation blocked because CrossShield assigned a BLOCK decision."
            ),
        )
        _persist_generate_audit(
            analysis,
            llm_called=False,
            llm_provider=None,
            llm_model=None,
            output_safety_checked=False,
            output_safety_decision=None,
            output_safety_score=None,
            output_categories=[],
            output_matched_signals=[],
            blocked_reason=response.blocked_reason,
            request_latency_ms=(time.perf_counter() - request_start) * 1000,
            outcome="input_blocked",
        )
        return response

    try:
        llm_response = LLMGateway(MockLLMProvider()).generate(
            LLMRequest(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        )
    except Exception as exc:
        _persist_generate_audit(
            analysis,
            llm_called=True,
            llm_provider=None,
            llm_model=None,
            output_safety_checked=False,
            output_safety_decision=None,
            output_safety_score=None,
            output_categories=[],
            output_matched_signals=[],
            blocked_reason=None,
            request_latency_ms=(time.perf_counter() - request_start) * 1000,
            outcome="llm_error",
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "llm_generation_failed",
                "message": "Mock LLM generation failed.",
            },
        ) from exc

    output_safety = verify_output_safety(llm_response.content)
    output_blocked = output_safety.output_safety_decision.value == "BLOCK"

    response = GenerateResponse(
        **response_data,
        llm_called=True,
        llm_response=None if output_blocked else llm_response.content,
        llm_provider=llm_response.provider,
        llm_model=llm_response.model,
        blocked_reason=(
            "Generated response blocked by output safety verification."
            if output_blocked
            else None
        ),
        output_safety_score=output_safety.output_safety_score,
        output_safety_categories=output_safety.output_safety_categories,
        output_safety_matched_signals=output_safety.output_safety_matched_signals,
        output_safety_explanation=output_safety.output_safety_explanation,
        output_safety_decision=output_safety.output_safety_decision,
    )
    _persist_generate_audit(
        analysis,
        llm_called=True,
        llm_provider=llm_response.provider,
        llm_model=llm_response.model,
        output_safety_checked=True,
        output_safety_decision=output_safety.output_safety_decision.value,
        output_safety_score=output_safety.output_safety_score,
        output_categories=output_safety.output_safety_categories,
        output_matched_signals=output_safety.output_safety_matched_signals,
        blocked_reason=response.blocked_reason,
        request_latency_ms=(time.perf_counter() - request_start) * 1000,
        outcome="output_blocked" if output_blocked else "completed",
    )
    return response


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "crossshield"}
