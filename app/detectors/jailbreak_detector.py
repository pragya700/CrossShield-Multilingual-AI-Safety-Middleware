"""
Jailbreak Detector — Phase 4 (rule-based baseline).

This is a transparent, explainable BASELINE, not the project's final
detection strategy. It exists so later, stronger detectors (Phase 5:
semantic/embedding-based; Phase 6: cross-lingual semantic) have something
concrete to be measured against. See jailbreak_rules.py for the pattern
content, weights, and their documented limitations.

WHAT THIS DOES NOT DO:
- No semantic understanding -- purely regex pattern matching against the
  normalized prompt text.
- No claim of catching every jailbreak, in any language. It catches the
  specific phrasings encoded in jailbreak_rules.py and close variants only.
- No ML, no embeddings, no external calls -- fully local and deterministic,
  safe to run synchronously on every request.

SCORING (BASELINE CONFIGURATION, not scientifically tuned):
Each rule that matches contributes its `weight`. To avoid one attack
category being over-counted just because it has several overlapping
patterns, rules are first aggregated per category (taking the *maximum*
weight among rules that matched within that category), and only then are
those per-category scores summed across categories and capped at 1.0. This
means a prompt matching 3 different attack categories scores higher than
one matching 3 overlapping patterns within a single category -- multiple
independent attack vectors are treated as more suspicious than one
strongly-worded one, which is a deliberate (documented) design choice, not
a mathematically derived optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.detectors.jailbreak_rules import ALL_RULES, Rule


@dataclass
class JailbreakResult:
    jailbreak_score: float
    attack_categories: list[str] = field(default_factory=list)
    matched_signals: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)


def _rules_matching(text: str) -> list[Rule]:
    matched: list[Rule] = []
    for rule in ALL_RULES:
        if any(pattern.search(text) for pattern in rule.patterns):
            matched.append(rule)
    return matched


def detect(prompt: str, normalized_prompt: str, context: list[str] | None = None) -> JailbreakResult:
    """
    Detects rule-based jailbreak signals in a prompt.

    `prompt` (the original, unmodified text) is accepted for interface
    symmetry with the rest of the pipeline and for potential future use
    (e.g. a rule that specifically cares about pre-normalization form), but
    Phase 4's rules match against `normalized_prompt` -- matching post-
    normalization means whitespace/repeated-character/basic-obfuscation
    tricks handled in Phase 3 (e.g. "i g n o r e   p r e v i o u s") are
    already cleaned up by the time these patterns run.

    `context` (prior conversation turns) is accepted but NOT used yet --
    multi-turn/context-aware analysis is Phase 7's job. It's in the
    signature now so the API contract for this function doesn't need to
    change when that phase lands.
    """
    del prompt, context  # not used by the Phase 4 baseline; see docstring

    if not normalized_prompt.strip():
        return JailbreakResult(jailbreak_score=0.0)

    matched_rules = _rules_matching(normalized_prompt)
    if not matched_rules:
        return JailbreakResult(jailbreak_score=0.0)

    category_best_weight: dict[str, float] = {}
    for rule in matched_rules:
        current_best = category_best_weight.get(rule.category, 0.0)
        if rule.weight > current_best:
            category_best_weight[rule.category] = rule.weight

    jailbreak_score = min(1.0, sum(category_best_weight.values()))

    # Preserve first-match order for readability rather than sorting
    # alphabetically -- the order rules matched in is itself informative.
    attack_categories: list[str] = []
    matched_signals: list[str] = []
    explanation: list[str] = []
    for rule in matched_rules:
        if rule.category not in attack_categories:
            attack_categories.append(rule.category)
        matched_signals.append(rule.id)
        explanation.append(rule.explanation)

    return JailbreakResult(
        jailbreak_score=jailbreak_score,
        attack_categories=attack_categories,
        matched_signals=matched_signals,
        explanation=explanation,
    )
