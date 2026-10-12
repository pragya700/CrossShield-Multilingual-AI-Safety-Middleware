"""
Context / Reasoning Analyzer — Phase 7.

Phases 4-6 all score ONE prompt in isolation. Phase 7 is the first detector
that looks at the *relationship between turns* in a conversation: whether an
earlier, benign-looking message sets up a later one, whether risk climbs
turn over turn, whether a later message tries to override or contradict an
earlier instruction, and so on. It is deliberately a SEPARATE module (not
folded into jailbreak_detector.py) so each detector stays small and the
single-prompt detectors are untouched.

THIS IS A DETERMINISTIC, RULE-BASED FIRST IMPLEMENTATION, not a reasoning
model. It does not "understand" a conversation; it looks for a small set of
hand-specified structural patterns across turns, each independently
explainable. See CONTEXT_CATEGORY_WEIGHTS below and the README for the full,
honest list of what this does and does not catch.

CATEGORIES IMPLEMENTED (only these -- no others are claimed):
    context_reset                    - "ignore everything above", "start a
                                        new conversation", etc. (single turn)
    instruction_hierarchy_manipulation - "the previous instruction no longer
                                        applies", "treat my next message as
                                        highest priority", etc. (single turn)
    hypothetical_bypass              - a hypothetical/fictional framing
                                        phrase co-occurring (same turn or
                                        across turns) with an actual bypass-
                                        intent phrase. A hypothetical
                                        question ALONE never fires this.
    role_escalation                  - an earlier benign role-play/"act as"
                                        turn followed by a LATER turn
                                        escalating that role into "no
                                        rules"/"unrestricted" framing.
    indirect_instruction_chain       - an earlier "explain how an attacker
                                        would think" turn followed by a
                                        LATER "now convert that into exact
                                        instructions" turn.
    conflicting_instruction_manipulation - an earlier "always do X" turn
                                        followed by a LATER "actually,
                                        ignore what I said, do Y instead"
                                        turn.
    gradual_escalation                - reuses Phase 4's per-turn rule score
                                        (not a new keyword system) across 3+
                                        user turns: the conversation must
                                        start at 0.0 risk, end above 0.0, and
                                        never have a later turn score lower
                                        than an earlier one (non-decreasing).
    multi_step_jailbreak              - meta-category: fires when 2+ of the
                                        other categories above fire in the
                                        SAME conversation, i.e. a coordinated
                                        multi-pattern attempt rather than one
                                        isolated signal.

FALSE-POSITIVE PRINCIPLE (see README "Important false-positive requirement"):
multi-turn conversation, role-play, fiction, a hypothetical question, complex
reasoning, or simply having many messages NEVER by themselves produce a
signal. Every category above requires a specific structural co-occurrence
or order relationship, not just the presence of a topic or keyword.

SINGLE-PROMPT REQUESTS: when no `conversation` is supplied, the analyzer
still runs (it treats the request as a 1-turn conversation), but by
construction every category except context_reset/instruction_hierarchy_
manipulation/hypothetical_bypass requires 2+ user turns to fire at all, and
those three single-turn-capable categories look for specific structural
phrases, not generic jailbreak language already covered by Phase 4. In
practice this means an isolated single prompt usually scores 0.0 here even
when Phase 4/5/6 flag it -- that is correct, not a bug: Phase 7's whole
point is signals that only exist ACROSS turns.

NO MODEL, NO "UNAVAILABLE" STATE: unlike Phases 5-6, this analyzer has
nothing to download and nothing that can be network-unavailable -- it is
pure local string/regex logic reusing Phase 3's normalizer and Phase 4's
rule detector (both already local and deterministic). The `context_
reasoning_status` field therefore only ever reflects the config toggle
("enabled"/"disabled"), never a failure state.

MULTILINGUAL LIMITATION (stated plainly, not glossed over): every pattern
below is English-only. A conversation conducted entirely in Hindi, Tamil,
Telugu, or Bengali will not trigger context_reset / instruction_hierarchy_
manipulation / hypothetical_bypass / role_escalation / indirect_instruction_
chain / conflicting_instruction_manipulation, because none of their phrases
are translated. gradual_escalation and multi_step_jailbreak partially
inherit whatever multilingual coverage Phase 4's rule detector already has
(which is itself a small, documented, non-comprehensive set -- see Phase 4
README section) since they reuse it directly, but nothing new was added
here. This is Phase 7's most significant limitation and is not claimed to
be solved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.detectors.jailbreak_detector import detect as detect_jailbreak_rules
from app.detectors.normalizer import normalize

# BASELINE CONFIGURATION -- explicit placeholders, not claimed optimal.
# Same per-category-then-sum-capped-at-1.0 pattern Phase 4 uses.
CONTEXT_CATEGORY_WEIGHTS: dict[str, float] = {
    "context_reset": 0.35,
    "instruction_hierarchy_manipulation": 0.40,
    "role_escalation": 0.40,
    "hypothetical_bypass": 0.35,
    "gradual_escalation": 0.35,
    "indirect_instruction_chain": 0.35,
    "conflicting_instruction_manipulation": 0.30,
    "multi_step_jailbreak": 0.45,
}


def _p(*patterns: str) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


_CONTEXT_RESET_PATTERNS = _p(
    r"\bignore everything (above|before)\b",
    r"\bstart (a )?new conversation\b",
    r"\bfrom (this point( onward)?|now on)\s*,?\s*(the )?previous (rules|instructions) (do not|don't) exist\b",
    r"\bprevious (rules|instructions) no longer (exist|apply)\b",
)

_INSTRUCTION_HIERARCHY_PATTERNS = _p(
    r"\bthe previous (system )?instructions? no longer applies?\b",
    r"\btreat my next message as (the )?highest priority\b",
    r"\bforget all previous (restrictions|instructions|rules)\b",
    r"\bthe developer instructions? should be ignored\b",
    r"\b(system|developer) instructions? (no longer|does not|do not) apply\b",
)

_HYPOTHETICAL_FRAMING_PATTERNS = _p(
    r"\bfor a fictional scenario\b",
    r"\bpretend you are an? unrestricted\b",
    r"\bin a hypothetical world where safety rules (don't|do not) exist\b",
    r"\bhypothetically\b",
    r"\bimagine (you are|a world where)\b",
)
_BYPASS_INTENT_PATTERNS = _p(
    r"\bignore (those |the )?safety (polic(y|ies)|rules|restrictions)\b",
    r"\bgive (me )?(the )?(real|actual|exact|operational) instructions\b",
    r"\bwithout (any )?(restrictions|limitations|safety)\b",
    r"\bsafety (polic(y|ies)|rules|restrictions) (would normally )?prevent\b",
)

_EARLY_ROLE_PATTERNS = _p(
    r"\b(let'?s|let us) (role.?play|act out)\b",
    r"\bact as an?\b",
    r"\bpretend (you are|to be) an?\b",
)
_ESCALATED_ROLE_PATTERNS = _p(
    r"\byou (are|have become) now an? unrestricted\b",
    r"\bwith no (rules|restrictions|limitations)\b",
    r"\bno (safety )?restrictions? (apply|anymore)\b",
    r"\bunrestricted (ai|assistant|system)\b",
)

_INDIRECT_CHAIN_PART_A_PATTERNS = _p(
    r"\bexplain how an? (attacker|hacker|malicious actor) would (think|act|approach)\b",
    r"\bdescribe how (someone|an attacker) would\b",
)
_INDIRECT_CHAIN_PART_B_PATTERNS = _p(
    r"\b(now )?(convert|turn) (that|this|it) into (exact|operational|actionable|step.by.step) instructions\b",
    r"\bnow (give|provide|write) (the )?(exact|operational) (steps|instructions)\b",
)

_CONFLICT_ESTABLISH_PATTERNS = _p(
    r"\byou must always\b",
    r"\balways (do|follow|obey)\b",
)
_CONFLICT_OVERRIDE_PATTERNS = _p(
    r"\bactually,?\s*ignore what i said\b",
    r"\bdisregard that,?\s*instead\b",
    r"\bbut actually do the opposite\b",
    r"\bignore (that|what i just said)\b",
)

# Supplementary per-turn risk signal for gradual_escalation ONLY, local to
# this file -- NOT added to Phase 4's own rule set (jailbreak_rules.py is
# intentionally left untouched). Phase 4's patterns are tuned for single-
# prompt precision and use a specific word list (rules/instructions/prompts/
# commands) that does not include every escalation-indicative synonym (e.g.
# "restrictions"); this small supplementary set exists only to let the
# escalation TREND detector below see a turn-over-turn rise even when the
# wording uses a near-synonym Phase 4 doesn't cover. It intentionally does
# NOT duplicate Phase 4's full rule set -- just two phrases needed for the
# escalation pattern specifically.
_ESCALATION_SUPPLEMENTARY_PATTERNS = _p(
    r"\bignore (the |all |those )?(previous |prior |earlier )?restrictions?\b",
    r"\b(give|provide|write) (me |the )?(exact|actual|real) (harmful|dangerous|unsafe) (instructions?|steps)\b",
)


def _turn_risk_score(text: str) -> float:
    """Per-turn risk used only for the gradual_escalation trend check.
    Primary signal is Phase 4's real rule score (reused, not duplicated);
    the small supplementary set above fills one specific gap -- see comment
    there."""
    norm = normalize(text)
    rule_score = detect_jailbreak_rules(text, norm.normalized_text).jailbreak_score
    supplementary = 0.3 if _any_match(_ESCALATION_SUPPLEMENTARY_PATTERNS, text) else 0.0
    return max(rule_score, supplementary)


@dataclass(frozen=True)
class ConversationMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ContextMatch:
    category: str
    description: str
    turn_indices: list[int] = field(default_factory=list)


@dataclass
class ContextReasoningResult:
    context_reasoning_score: float
    categories: list[str] = field(default_factory=list)
    matches: list[ContextMatch] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    turns_analyzed: int = 0


def _any_match(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _first_match_index(patterns: Sequence[re.Pattern[str]], texts: Sequence[str]) -> int | None:
    for i, text in enumerate(texts):
        if _any_match(patterns, text):
            return i
    return None


def _all_match_indices(patterns: Sequence[re.Pattern[str]], texts: Sequence[str]) -> list[int]:
    return [i for i, text in enumerate(texts) if _any_match(patterns, text)]


# -- single-turn categories ---------------------------------------------


def _detect_context_reset(texts: Sequence[str]) -> ContextMatch | None:
    indices = _all_match_indices(_CONTEXT_RESET_PATTERNS, texts)
    if not indices:
        return None
    return ContextMatch(
        "context_reset",
        "A message attempts to discard prior context or restrictions outright "
        "(e.g. 'ignore everything above', 'start a new conversation').",
        indices,
    )


def _detect_instruction_hierarchy_manipulation(texts: Sequence[str]) -> ContextMatch | None:
    indices = _all_match_indices(_INSTRUCTION_HIERARCHY_PATTERNS, texts)
    if not indices:
        return None
    return ContextMatch(
        "instruction_hierarchy_manipulation",
        "A message claims prior system/developer instructions no longer apply "
        "or asserts a new, higher-priority instruction.",
        indices,
    )


# -- cross-turn (or same-turn co-occurrence) categories -------------------


def _detect_hypothetical_bypass(texts: Sequence[str]) -> ContextMatch | None:
    framing_indices = _all_match_indices(_HYPOTHETICAL_FRAMING_PATTERNS, texts)
    bypass_indices = _all_match_indices(_BYPASS_INTENT_PATTERNS, texts)
    if not framing_indices or not bypass_indices:
        return None
    # Require the framing to appear at or before a bypass-intent turn --
    # "set up the hypothetical, then ask for the bypass" is the pattern of
    # concern; a bypass phrase appearing before any hypothetical framing
    # existed is not this pattern (Phase 4 already covers a bare bypass
    # phrase on its own).
    first_framing = min(framing_indices)
    relevant_bypass = [i for i in bypass_indices if i >= first_framing]
    if not relevant_bypass:
        return None
    indices = sorted(set([first_framing, *relevant_bypass]))
    return ContextMatch(
        "hypothetical_bypass",
        "A hypothetical/fictional framing is followed by an actual request to "
        "bypass safety behavior -- a hypothetical framing alone never triggers "
        "this; an accompanying concrete bypass request is required.",
        indices,
    )


def _detect_role_escalation(texts: Sequence[str]) -> ContextMatch | None:
    early_indices = _all_match_indices(_EARLY_ROLE_PATTERNS, texts)
    escalated_indices = _all_match_indices(_ESCALATED_ROLE_PATTERNS, texts)
    if not early_indices or not escalated_indices:
        return None
    first_role = min(early_indices)
    later_escalation = [i for i in escalated_indices if i > first_role]
    if not later_escalation:
        return None
    indices = sorted(set([first_role, *later_escalation]))
    return ContextMatch(
        "role_escalation",
        "An earlier benign role-play/persona request is followed by a later "
        "turn escalating that role into an 'unrestricted'/'no rules' framing.",
        indices,
    )


def _detect_indirect_instruction_chain(texts: Sequence[str]) -> ContextMatch | None:
    part_a = _all_match_indices(_INDIRECT_CHAIN_PART_A_PATTERNS, texts)
    part_b = _all_match_indices(_INDIRECT_CHAIN_PART_B_PATTERNS, texts)
    if not part_a or not part_b:
        return None
    first_a = min(part_a)
    later_b = [i for i in part_b if i > first_a]
    if not later_b:
        return None
    indices = sorted(set([first_a, *later_b]))
    return ContextMatch(
        "indirect_instruction_chain",
        "An earlier turn asks how a malicious actor would approach something, "
        "and a later turn asks to convert that explanation into concrete, "
        "actionable instructions.",
        indices,
    )


def _detect_conflicting_instruction_manipulation(texts: Sequence[str]) -> ContextMatch | None:
    establish = _all_match_indices(_CONFLICT_ESTABLISH_PATTERNS, texts)
    override = _all_match_indices(_CONFLICT_OVERRIDE_PATTERNS, texts)
    if not establish or not override:
        return None
    first_establish = min(establish)
    later_override = [i for i in override if i > first_establish]
    if not later_override:
        return None
    indices = sorted(set([first_establish, *later_override]))
    return ContextMatch(
        "conflicting_instruction_manipulation",
        "An earlier turn establishes a standing directive, and a later turn "
        "explicitly tells the model to disregard it in favor of something else.",
        indices,
    )


def _detect_gradual_escalation(texts: Sequence[str]) -> ContextMatch | None:
    if len(texts) < 3:
        return None
    scores = [_turn_risk_score(text) for text in texts]
    if scores[0] != 0.0 or scores[-1] <= 0.0:
        return None
    non_decreasing = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    if not non_decreasing:
        return None
    return ContextMatch(
        "gradual_escalation",
        f"Per-turn risk is non-decreasing across all {len(texts)} user turns, "
        f"starting benign (0.0) and ending risky ({scores[-1]:.2f}) -- a "
        "trend, not a single-turn match.",
        list(range(len(texts))),
    )


def analyze(
    prompt: str,
    normalized_prompt: str,
    conversation: Sequence[ConversationMessage] | None = None,
) -> ContextReasoningResult:
    """
    Analyzes `conversation` (prior turns, NOT including the current prompt)
    plus `prompt` (always treated as the final, current user turn) for
    cross-turn contextual/reasoning jailbreak patterns.

    `normalized_prompt` is accepted for interface consistency with the other
    detectors but each turn (including the final prompt) is independently
    re-normalized internally via Phase 3's normalize(), since prior turns
    arrive raw and need the same treatment for the regex patterns here to
    behave consistently turn-to-turn.
    """
    del normalized_prompt  # see docstring

    user_texts: list[str] = []
    for message in conversation or ():
        if message.role == "user":
            user_texts.append(normalize(message.content).normalized_text)
    user_texts.append(normalize(prompt).normalized_text)  # current turn is always last

    detectors = [
        _detect_context_reset,
        _detect_instruction_hierarchy_manipulation,
        _detect_hypothetical_bypass,
        _detect_role_escalation,
        _detect_indirect_instruction_chain,
        _detect_conflicting_instruction_manipulation,
        _detect_gradual_escalation,
    ]
    matches: list[ContextMatch] = []
    for detector in detectors:
        result = detector(user_texts)
        if result is not None:
            matches.append(result)

    if len(matches) >= 2:
        combined_indices = sorted({i for m in matches for i in m.turn_indices})
        matches.append(
            ContextMatch(
                "multi_step_jailbreak",
                f"{len(matches)} independent contextual patterns "
                f"({', '.join(m.category for m in matches)}) fired together in "
                "the same conversation, indicating a coordinated multi-step "
                "attempt rather than one isolated signal.",
                combined_indices,
            )
        )

    categories = [m.category for m in matches]
    score = min(1.0, sum(CONTEXT_CATEGORY_WEIGHTS[c] for c in categories))
    explanation = [m.description for m in matches]

    return ContextReasoningResult(
        context_reasoning_score=score,
        categories=categories,
        matches=matches,
        explanation=explanation,
        turns_analyzed=len(user_texts),
    )
