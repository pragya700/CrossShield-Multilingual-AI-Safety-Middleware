"""
Central, environment-driven configuration.

Nothing about risk thresholds or weights should be hard-coded inside
detector/engine modules — they all read from here so that later phases
(tuning, experiments) only touch this file.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, field_validator, model_validator


class Settings(BaseModel):
    app_name: str = "CrossShield"

    env: str = os.getenv("CROSSSHIELD_ENV", "development")

    # Supported languages for Phase 1+ (extend via config, not code rewrites)
    supported_languages: list[str] = ["en", "hi", "ta", "te", "bn"]

    # Risk thresholds (UNVALIDATED placeholders — to be tuned experimentally
    # in Phase 13's evaluation). Overall risk in [0, 1].
    threshold_medium: float = float(
        os.getenv("CROSSSHIELD_THRESHOLD_MEDIUM", 0.3)
    )

    threshold_high: float = float(
        os.getenv("CROSSSHIELD_THRESHOLD_HIGH", 0.6)
    )

    threshold_critical: float = float(
        os.getenv("CROSSSHIELD_THRESHOLD_CRITICAL", 0.85)
    )

    # Risk engine weights
    weight_prompt_risk: float = float(
        os.getenv("CROSSSHIELD_WEIGHT_PROMPT_RISK", 0.35)
    )

    weight_rule_risk: float = float(
        os.getenv("CROSSSHIELD_WEIGHT_RULE_RISK", 0.25)
    )

    weight_semantic_risk: float = float(
        os.getenv("CROSSSHIELD_WEIGHT_SEMANTIC_RISK", 0.25)
    )

    weight_cross_lingual_risk: float = float(
        os.getenv("CROSSSHIELD_WEIGHT_CROSS_LINGUAL_RISK", 0.25)
    )

    weight_context_risk: float = float(
        os.getenv("CROSSSHIELD_WEIGHT_CONTEXT_RISK", 0.25)
    )

    # Legacy Phase-1 name retained for compatibility
    weight_jailbreak_risk: float = float(
        os.getenv("CROSSSHIELD_WEIGHT_JAILBREAK_RISK", 0.35)
    )

    # Phase 8 risk-engine policy
    risk_degraded_mode_enabled: bool = (
        os.getenv(
            "CROSSSHIELD_RISK_DEGRADED_MODE_ENABLED",
            "false",
        ).lower()
        == "true"
    )

    # ------------------------------------------------------------------
    # Phase 5: semantic jailbreak detection
    # ------------------------------------------------------------------

    semantic_enabled: bool = (
        os.getenv(
            "CROSSSHIELD_SEMANTIC_ENABLED",
            "false",
        ).lower()
        == "true"
    )

    semantic_model_name: str = os.getenv(
        "CROSSSHIELD_SEMANTIC_MODEL_NAME",
        "sentence-transformers/LaBSE",
    )

    semantic_device: str = os.getenv(
        "CROSSSHIELD_SEMANTIC_DEVICE",
        "cpu",
    )

    semantic_local_files_only: bool = (
        os.getenv(
            "CROSSSHIELD_SEMANTIC_LOCAL_FILES_ONLY",
            "false",
        ).lower()
        == "true"
    )

    semantic_text_prefix: str = os.getenv(
        "CROSSSHIELD_SEMANTIC_TEXT_PREFIX",
        "",
    )

    semantic_threshold: float = float(
        os.getenv(
            "CROSSSHIELD_SEMANTIC_THRESHOLD",
            0.70,
        )
    )

    semantic_hard_negative_margin: float = float(
        os.getenv(
            "CROSSSHIELD_SEMANTIC_HARD_NEGATIVE_MARGIN",
            0.0,
        )
    )

    semantic_max_segments: int = int(
        os.getenv(
            "CROSSSHIELD_SEMANTIC_MAX_SEGMENTS",
            32,
        )
    )

    semantic_top_matches: int = int(
        os.getenv(
            "CROSSSHIELD_SEMANTIC_TOP_MATCHES",
            3,
        )
    )

    semantic_reference_path: str = os.getenv(
        "CROSSSHIELD_SEMANTIC_REFERENCE_PATH",
        "",
    )

    semantic_hard_negative_path: str = os.getenv(
        "CROSSSHIELD_SEMANTIC_HARD_NEGATIVE_PATH",
        "",
    )

    rule_jailbreak_weight: float = float(
        os.getenv(
            "CROSSSHIELD_RULE_JAILBREAK_WEIGHT",
            0.5,
        )
    )

    semantic_jailbreak_weight: float = float(
        os.getenv(
            "CROSSSHIELD_SEMANTIC_JAILBREAK_WEIGHT",
            0.5,
        )
    )

    # ------------------------------------------------------------------
    # Phase 6: cross-lingual semantic safety analysis
    # ------------------------------------------------------------------

    cross_lingual_enabled: bool = (
        os.getenv(
            "CROSSSHIELD_CROSS_LINGUAL_ENABLED",
            "false",
        ).lower()
        == "true"
    )

    cross_lingual_threshold: float = float(
        os.getenv(
            "CROSSSHIELD_CROSS_LINGUAL_THRESHOLD",
            0.70,
        )
    )

    cross_lingual_margin: float = float(
        os.getenv(
            "CROSSSHIELD_CROSS_LINGUAL_MARGIN",
            0.0,
        )
    )

    cross_lingual_max_segments: int = int(
        os.getenv(
            "CROSSSHIELD_CROSS_LINGUAL_MAX_SEGMENTS",
            32,
        )
    )

    cross_lingual_top_matches: int = int(
        os.getenv(
            "CROSSSHIELD_CROSS_LINGUAL_TOP_MATCHES",
            3,
        )
    )

    cross_lingual_reference_path: str = os.getenv(
        "CROSSSHIELD_CROSS_LINGUAL_REFERENCE_PATH",
        "",
    )

    # ------------------------------------------------------------------
    # Phase 7: context/reasoning analysis
    # ------------------------------------------------------------------

    context_reasoning_enabled: bool = (
        os.getenv(
            "CROSSSHIELD_CONTEXT_REASONING_ENABLED",
            "true",
        ).lower()
        == "true"
    )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    @field_validator(
        "threshold_medium",
        "threshold_high",
        "threshold_critical",
        "weight_prompt_risk",
        "weight_jailbreak_risk",
        "weight_rule_risk",
        "weight_semantic_risk",
        "weight_cross_lingual_risk",
        "weight_context_risk",
    )
    @classmethod
    def validate_non_negative_or_bounded(cls, value: float, info):
        if info.field_name.startswith("threshold_") and not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{info.field_name} must be between 0 and 1"
            )

        if info.field_name.startswith("weight_") and value < 0.0:
            raise ValueError(
                f"{info.field_name} must be non-negative"
            )

        return value

    @model_validator(mode="after")
    def validate_risk_configuration(self):
        if not (
            self.threshold_medium
            < self.threshold_high
            < self.threshold_critical
        ):
            raise ValueError(
                "Risk thresholds must satisfy medium < high < critical"
            )

        if (
            self.weight_prompt_risk
            + self.weight_rule_risk
            + self.weight_semantic_risk
            + self.weight_cross_lingual_risk
            + self.weight_context_risk
        ) <= 0:
            raise ValueError(
                "At least one risk-engine weight must be greater than zero"
            )

        if (
            self.rule_jailbreak_weight < 0
            or self.semantic_jailbreak_weight < 0
        ):
            raise ValueError(
                "Jailbreak-combination weights must be non-negative"
            )

        if (
            self.rule_jailbreak_weight
            + self.semantic_jailbreak_weight
            <= 0
        ):
            raise ValueError(
                "At least one jailbreak-combination weight must be greater than zero"
            )

        return self

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_raw_prompts: bool = (
        os.getenv(
            "CROSSSHIELD_LOG_RAW_PROMPTS",
            "false",
        ).lower()
        == "true"
    )

    # ------------------------------------------------------------------
    # Phase 11: local, metadata-only security auditing
    # ------------------------------------------------------------------

    security_audit_enabled: bool = (
        os.getenv(
            "CROSSSHIELD_SECURITY_AUDIT_ENABLED",
            "true",
        ).lower()
        == "true"
    )

    security_audit_db_path: str = os.getenv(
        "CROSSSHIELD_SECURITY_AUDIT_DB_PATH",
        "data/crossshield_audit.db",
    )

    # ------------------------------------------------------------------
    # Phase 12: dashboard configuration
    # ------------------------------------------------------------------

    # TEMPORARILY ENABLED FOR DASHBOARD TESTING.
    dashboard_enabled: bool = (
        os.getenv(
            "CROSSSHIELD_DASHBOARD_ENABLED",
            "true",
        ).lower()
        == "true"
    )

    dashboard_max_events: int = int(
        os.getenv(
            "CROSSSHIELD_DASHBOARD_MAX_EVENTS",
            "100",
        )
    )

    dashboard_max_hours: int = int(
        os.getenv(
            "CROSSSHIELD_DASHBOARD_MAX_HOURS",
            "168",
        )
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()