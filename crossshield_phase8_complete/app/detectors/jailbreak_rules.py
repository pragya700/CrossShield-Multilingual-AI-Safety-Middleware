"""
Phase 4 — Rule definitions for the rule-based jailbreak detection baseline.

These rules are intentionally transparent and deterministic.
They provide the lexical baseline for later semantic and
cross-lingual detectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    patterns: tuple[re.Pattern[str], ...]
    weight: float
    explanation: str
    language: str = "en"


def _p(*patterns: str) -> tuple[re.Pattern[str], ...]:
    """Compile Unicode-aware case-insensitive regex patterns."""
    return tuple(
        re.compile(
            pattern,
            re.IGNORECASE | re.UNICODE
        )
        for pattern in patterns
    )


# =============================================================================
# ENGLISH
# =============================================================================

_ENGLISH_RULES: list[Rule] = [

    # -------------------------------------------------------------------------
    # Instruction Override
    # -------------------------------------------------------------------------

    Rule(
        id="instruction_override_en_001",
        category="instruction_override",
        patterns=_p(
            r"\bignore\s+(all\s+)?(the\s+)?"
            r"(previous|prior|above|earlier)\s+"
            r"(instructions?|rules?|prompts?|commands?)\b",

            r"\bdisregard\s+(all\s+)?(the\s+)?"
            r"(previous|prior|above|earlier)\s+"
            r"(instructions?|rules?)\b",

            r"\bforget\s+(everything|all)\s+"
            r"(you\s+)?(were\s+)?"
            r"(told|instructed|taught)\b",

            r"\boverride\s+(your|the)\s+"
            r"(previous\s+)?instructions?\b",
        ),
        weight=0.4,
        explanation=(
            "Detected an attempt to override or discard "
            "previously given instructions."
        ),
    ),

    # -------------------------------------------------------------------------
    # System Prompt Manipulation
    # -------------------------------------------------------------------------

    Rule(
        id="system_prompt_manipulation_en_001",
        category="system_prompt_manipulation",
        patterns=_p(
            r"\b(reveal|show|print|repeat|output)\s+"
            r"(me\s+)?your\s+"
            r"(?:(?:system|developer|hidden)\s+){1,2}"
            r"(prompt|instructions?)\b",

            r"\bwhat\s+(is|are)\s+your\s+"
            r"(system|initial|original)\s+"
            r"(prompt|instructions?)\b",

            r"\brepeat\s+(the\s+)?"
            r"(system\s+)?prompt\s+"
            r"(above|verbatim|exactly)\b",
        ),
        weight=0.35,
        explanation=(
            "Detected an attempt to reveal or manipulate "
            "the system/developer prompt."
        ),
    ),

    # -------------------------------------------------------------------------
    # Safety Bypass
    # -------------------------------------------------------------------------

    Rule(
        id="safety_bypass_en_001",
        category="safety_bypass",
        patterns=_p(
            r"\b(bypass|disable|remove|turn\s+off)\s+"
            r"(the\s+)?(all\s+)?safety\s+"
            r"(rules?|restrictions?|filters?|guidelines?|"
            r"features?|mechanisms?)\b",

            r"\bact\s+(completely\s+)?"
            r"without\s+(any\s+)?"
            r"(restrictions?|rules?|filters?|limitations?)\b",

            r"\bwithout\s+any\s+"
            r"(content\s+)?"
            r"(restrictions?|filters?|limitations?|moderation)\b",

            r"\bno\s+(ethical|moral|safety)\s+"
            r"(guidelines?|restrictions?|limitations?)\s+"
            r"(apply|whatsoever)\b",
        ),
        weight=0.45,
        explanation=(
            "Detected language suggesting an attempt to bypass "
            "or disable safety restrictions."
        ),
    ),

    # -------------------------------------------------------------------------
    # Role Manipulation
    # -------------------------------------------------------------------------

    Rule(
        id="role_manipulation_en_001",
        category="role_manipulation",
        patterns=_p(
            r"\byou\s+are\s+now\s+"
            r"(DAN|an?\s+unrestricted\s+AI|"
            r"a\s+jailbroken\s+AI)\b",

            r"\b(pretend|imagine)\s+"
            r"(you\s+are|to\s+be)\s+"
            r"an?\s+AI\s+"
            r"(with\s+no|without)\s+"
            r"(restrictions?|rules?|filters?|limitations?)\b",

            r"\bact\s+as\s+an?\s+AI\s+"
            r"(that\s+has\s+no|without)\s+"
            r"(restrictions?|rules?|filters?)\b",

            r"\byou\s+"
            r"(have\s+no|do\s+not\s+have)\s+"
            r"(restrictions?|rules?|guidelines?)\s+"
            r"(anymore|now|in\s+this\s+mode)\b",

            r"\bpretend\s+you\s+are\s+an?\s+"
            r"(unrestricted|uncensored|unfiltered)\s+"
            r"(assistant|AI|model)\b",
        ),
        weight=0.4,
        explanation=(
            "Detected an attempt to make the model adopt "
            "an unrestricted or conflicting role."
        ),
    ),

    # -------------------------------------------------------------------------
    # Context Manipulation
    # -------------------------------------------------------------------------

    Rule(
        id="context_manipulation_en_001",
        category="context_manipulation",
        patterns=_p(
            r"\bfor\s+"
            r"(this\s+conversation|the\s+rest\s+of\s+this\s+chat)"
            r"\s*,?\s*"
            r"(the\s+)?"
            r"(rules?|instructions?|guidelines?)\s+"
            r"(do\s+not|don't)\s+apply\b",

            r"\bnew\s+(system\s+)?instructions?\s+"
            r"(supersede|override|replace)\s+"
            r"(the\s+)?"
            r"(previous|old|earlier)\b",

            r"\bfrom\s+now\s+on\s*,?\s*"
            r"(you\s+)?"
            r"(will|must)\s+"
            r"(ignore|disregard)\s+"
            r"(all\s+)?"
            r"(previous|prior)\b",

            r"\b(previous|prior|earlier)\s+instructions?\s+"
            r"(no\s+longer|no\s+more)\s+"
            r"(apply|matter|exist)\b",

            r"\bstart\s+over\s+and\s+"
            r"(obey|follow)\s+"
            r"(this|the)\s+new\s+"
            r"(instruction|rule|hierarchy)\b",
        ),
        weight=0.35,
        explanation=(
            "Detected an attempt to manipulate instruction "
            "hierarchy or conversational context."
        ),
    ),

    # -------------------------------------------------------------------------
    # Jailbreak Framing
    # -------------------------------------------------------------------------

    Rule(
        id="jailbreak_framing_en_001",
        category="jailbreak_framing",
        patterns=_p(
            r"\bdo\s+anything\s+now\b",
            r"\bjailbreak\s+(mode|prompt)\b",
            r"\bdeveloper\s+mode\s+"
            r"(enabled|activated|on)\b",
            r"\b(unfiltered|uncensored)\s+"
            r"(mode|response|output)\s+"
            r"(enabled|activated|on)?\b",
        ),
        weight=0.4,
        explanation=(
            "Detected phrasing commonly associated with "
            "known jailbreak framings."
        ),
    ),

    # -------------------------------------------------------------------------
    # Prompt Injection
    # -------------------------------------------------------------------------

    Rule(
        id="prompt_injection_en_001",
        category="prompt_injection",
        patterns=_p(
            r"\bignore\s+the\s+"
            r"(above|previous)\s+"
            r"(text|content|document|context)\s+and\b",

            r"\bnew\s+instructions?\s*:\s*\S",

            r"\bthe\s+"
            r"(text|content)\s+above\s+is\s+"
            r"(a\s+)?"
            r"(test|fake|irrelevant|not\s+real)\b",
        ),
        weight=0.4,
        explanation=(
            "Detected an injected instruction attempting "
            "to override the surrounding task/context."
        ),
    ),
]


# =============================================================================
# HINDI
# =============================================================================

_HINDI_RULES: list[Rule] = [

    Rule(
        id="instruction_override_hi_001",
        category="instruction_override",
        language="hi",
        patterns=_p(
            r"पिछले\s+निर्देशों?\s+को\s+अनदेखा\s+कर",
            r"पहले\s+के\s+निर्देशों?\s+को\s+भूल\s+जा",
            r"सभी\s+पिछले\s+निर्देशों?\s+को\s+अनदेखा\s+कर",
            r"पिछली\s+सभी\s+निर्देशों?\s+को\s+अनदेखा\s+कर",
        ),
        weight=0.4,
        explanation=(
            "Detected Hindi phrasing suggesting previous "
            "instructions should be ignored."
        ),
    ),

    Rule(
        id="system_prompt_manipulation_hi_001",
        category="system_prompt_manipulation",
        language="hi",
        patterns=_p(
            r"छिपे\s+हुए\s+सिस्टम\s+निर्देशों?\s+को\s+"
            r"(दिखाओ|बताओ|प्रकट\s+करो|उजागर\s+करो)",

            r"सिस्टम\s+निर्देशों?\s+को\s+"
            r"(दिखाओ|बताओ|प्रकट\s+करो|दोहराओ)",

            r"सिस्टम\s+प्रॉम्प्ट\s+"
            r"(दिखाओ|बताओ|प्रकट\s+करो|दोहराओ)",
        ),
        weight=0.35,
        explanation=(
            "Detected Hindi phrasing requesting disclosure "
            "of system instructions or prompts."
        ),
    ),

    Rule(
        id="safety_bypass_hi_001",
        category="safety_bypass",
        language="hi",
        patterns=_p(
            r"सुरक्षा\s+नियमों?\s+को\s+"
            r"(दरकिनार|बायपास|बंद)\s+कर",

            r"सुरक्षा\s+नियमों?\s+को\s+बंद\s+कर",

            r"सुरक्षा\s+प्रतिबंधों?\s+के\s+बिना",

            r"बिना\s+किसी\s+प्रतिबंध\s+के\s+जवाब\s+दो",
        ),
        weight=0.45,
        explanation=(
            "Detected Hindi phrasing suggesting safety "
            "rules or restrictions should be bypassed."
        ),
    ),

    Rule(
        id="role_manipulation_hi_001",
        category="role_manipulation",
        language="hi",
        patterns=_p(
            r"ऐसे\s+सहायक\s+की\s+भूमिका\s+निभाओ"
            r".{0,80}"
            r"(कोई\s+सुरक्षा\s+नीति|सुरक्षा\s+नियम|प्रतिबंध)\s*"
            r"(नहीं|न\s+हो)",

            r"नियंत्रणहीन\s+सहायक\s+की\s+भूमिका\s+निभाओ",

            r"बिना\s+किसी\s+सुरक्षा\s+नीति\s+वाले\s+"
            r"सहायक\s+की\s+तरह\s+व्यवहार\s+करो",
        ),
        weight=0.4,
        explanation=(
            "Detected Hindi role manipulation attempting "
            "to create an unrestricted assistant."
        ),
    ),

    Rule(
        id="context_manipulation_hi_001",
        category="context_manipulation",
        language="hi",
        patterns=_p(
            r"ऊपर\s+दिए\s+गए\s+सभी\s+निर्देश\s+अब\s+अमान्य\s+हैं",

            r"पिछले\s+निर्देश\s+अब\s+लागू\s+नहीं\s+हैं",

            r"नए\s+नियमों?\s+को\s+सर्वोच्च\s+प्राथमिकता\s+दो",

            r"इन\s+नए\s+निर्देशों?\s+को\s+सबसे\s+अधिक\s+प्राथमिकता\s+दो",
        ),
        weight=0.35,
        explanation=(
            "Detected Hindi manipulation of instruction hierarchy "
            "or conversational context."
        ),
    ),
]


# =============================================================================
# TAMIL
# =============================================================================

_TAMIL_RULES: list[Rule] = [

    Rule(
        id="instruction_override_ta_001",
        category="instruction_override",
        language="ta",
        patterns=_p(
            r"முந்தைய\s+அனைத்து\s+வழிமுறைகளையும்\s+புறக்கணி",
            r"முந்தைய\s+வழிமுறைகளை\s+புறக்கணி",
            r"முந்தைய\s+அனைத்து\s+வழிமுறைகளையும்\s+புறக்கணித்து",
            r"முந்தைய\s+வழிமுறைகளை\s+புறக்கணித்து",
        ),
        weight=0.4,
        explanation=(
            "Detected Tamil phrasing suggesting previous "
            "instructions should be ignored."
        ),
    ),

    Rule(
        id="system_prompt_manipulation_ta_001",
        category="system_prompt_manipulation",
        language="ta",
        patterns=_p(
            r"மறைக்கப்பட்ட\s+கணினி\s+வழிமுறைகளை\s+"
            r"(வெளிப்படுத்து|காட்டி|கூறு|மீண்டும்\s+கூறு)",

            r"உங்கள்\s+மறைக்கப்பட்ட\s+கணினி\s+வழிமுறைகளை\s+"
            r"(வெளிப்படுத்து|கூறு|மீண்டும்\s+கூறு)",

            r"கணினி\s+வழிமுறைகளை\s+"
            r"(வெளிப்படுத்து|காட்டு|கூறு)",
        ),
        weight=0.35,
        explanation=(
            "Detected Tamil phrasing requesting disclosure "
            "of system instructions."
        ),
    ),

    Rule(
        id="safety_bypass_ta_001",
        category="safety_bypass",
        language="ta",
        patterns=_p(
            r"பாதுகாப்பு\s+விதிகளை\s+"
            r"(மீறு|முடக்கு|புறக்கணி)",

            r"பாதுகாப்பு\s+விதிகளை\s+முடக்கி",

            r"எந்த\s+கட்டுப்பாடுகளும்\s+இல்லாமல்",

            r"எந்த\s+வரம்புகளும்\s+இல்லாமல்",
        ),
        weight=0.45,
        explanation=(
            "Detected Tamil phrasing suggesting safety "
            "rules or restrictions should be bypassed."
        ),
    ),

    Rule(
        id="role_manipulation_ta_001",
        category="role_manipulation",
        language="ta",
        patterns=_p(
            r"கட்டுப்பாடற்ற\s+உதவியாளராக\s+நடிக்கவும்",

            r"எந்த\s+பாதுகாப்பு\s+கொள்கைகளும்\s+இல்லாத\s+"
            r"கட்டுப்பாடற்ற\s+உதவியாளராக",

            r"பாதுகாப்பு\s+கொள்கைகள்\s+இல்லாத\s+"
            r"உதவியாளராக\s+நடிக்கவும்",
        ),
        weight=0.4,
        explanation=(
            "Detected Tamil role manipulation attempting "
            "to create an unrestricted assistant."
        ),
    ),

    Rule(
        id="context_manipulation_ta_001",
        category="context_manipulation",
        language="ta",
        patterns=_p(
            r"முந்தைய\s+வழிமுறைகள்\s+இனி\s+செல்லாது",

            r"முந்தைய\s+அனைத்து\s+வழிமுறைகளும்\s+இனி\s+செல்லாது",

            r"இந்த\s+புதிய\s+வழிமுறைகளுக்கு\s+முன்னுரிமை\s+கொடுக்கவும்",

            r"புதிய\s+வழிமுறைகளுக்கு\s+அதிக\s+முன்னுரிமை",
        ),
        weight=0.35,
        explanation=(
            "Detected Tamil manipulation of instruction hierarchy "
            "or conversational context."
        ),
    ),
]


# =============================================================================
# TELUGU
# =============================================================================

_TELUGU_RULES: list[Rule] = [

    Rule(
        id="instruction_override_te_001",
        category="instruction_override",
        language="te",
        patterns=_p(
            r"మునుపటి\s+అన్ని\s+సూచనలను\s+విస్మరించి",
            r"మునుపటి\s+సూచనలను\s+విస్మరించ",
            r"మునుపటి\s+అన్ని\s+సూచనలను\s+విస్మరించ",
            r"గత\s+సూచనలను\s+విస్మరించ",
        ),
        weight=0.4,
        explanation=(
            "Detected Telugu phrasing suggesting previous "
            "instructions should be ignored."
        ),
    ),

    Rule(
        id="system_prompt_manipulation_te_001",
        category="system_prompt_manipulation",
        language="te",
        patterns=_p(
            r"మీ\s+దాచిన\s+సిస్టమ్\s+సూచనలను\s+"
            r"(వెల్లడించి|వెల్లడించండి|చూపించి|చెప్పి)",

            r"దాచిన\s+సిస్టమ్\s+సూచనలను\s+"
            r"(వెల్లడించండి|చూపించండి|పునరావృతం\s+చేయండి)",

            r"సిస్టమ్\s+సూచనలను\s+"
            r"(వెల్లడించండి|చూపించండి|పునరావృతం\s+చేయండి)",
        ),
        weight=0.35,
        explanation=(
            "Detected Telugu phrasing requesting disclosure "
            "of system instructions."
        ),
    ),

    Rule(
        id="safety_bypass_te_001",
        category="safety_bypass",
        language="te",
        patterns=_p(
            r"మీ\s+భద్రతా\s+నియమాలను\s+"
            r"(నిలిపివేసి|నిలిపివేయండి|దాటవేసి|మార్చి)",

            r"భద్రతా\s+నియమాలను\s+"
            r"(నిలిపివేయండి|దాటవేయండి|విస్మరించండి)",

            r"ఎలాంటి\s+పరిమితులు\s+లేకుండా",

            r"ఏ\s+పరిమితులు\s+లేకుండా",
        ),
        weight=0.45,
        explanation=(
            "Detected Telugu phrasing suggesting safety "
            "rules or restrictions should be bypassed."
        ),
    ),

    Rule(
        id="role_manipulation_te_001",
        category="role_manipulation",
        language="te",
        patterns=_p(
            r"ఎటువంటి\s+భద్రతా\s+విధానాలు\s+లేని\s+"
            r"నియంత్రణలేని\s+సహాయకుడిగా\s+నటించండి",

            r"నియంత్రణలేని\s+సహాయకుడిగా\s+నటించండి",

            r"భద్రతా\s+విధానాలు\s+లేని\s+"
            r"సహాయకుడిగా\s+నటించండి",
        ),
        weight=0.4,
        explanation=(
            "Detected Telugu role manipulation attempting "
            "to create an unrestricted assistant."
        ),
    ),

    Rule(
        id="context_manipulation_te_001",
        category="context_manipulation",
        language="te",
        patterns=_p(
            r"మునుపటి\s+సూచనలు\s+ఇకపై\s+వర్తించవు",

            r"మునుపటి\s+అన్ని\s+సూచనలు\s+ఇకపై\s+వర్తించవు",

            r"ఈ\s+కొత్త\s+సూచనలకు\s+అత్యధిక\s+ప్రాధాన్యత\s+ఇవ్వండి",

            r"కొత్త\s+సూచనలకు\s+అత్యధిక\s+ప్రాధాన్యత",
        ),
        weight=0.35,
        explanation=(
            "Detected Telugu manipulation of instruction hierarchy "
            "or conversational context."
        ),
    ),
]


# =============================================================================
# BENGALI
# =============================================================================

_BENGALI_RULES: list[Rule] = [

    Rule(
        id="instruction_override_bn_001",
        category="instruction_override",
        language="bn",
        patterns=_p(
            r"আগের\s+সব\s+নির্দেশনা\s+উপেক্ষা\s+করুন",
            r"আগের\s+সব\s+নির্দেশনা\s+উপেক্ষা\s+করে",
            r"পূর্ববর্তী\s+নির্দেশাবলী\s+উপেক্ষা\s+কর",
            r"পূর্বের\s+সব\s+নির্দেশনা\s+উপেক্ষা\s+করুন",
        ),
        weight=0.4,
        explanation=(
            "Detected Bengali phrasing suggesting previous "
            "instructions should be ignored."
        ),
    ),

    Rule(
        id="system_prompt_manipulation_bn_001",
        category="system_prompt_manipulation",
        language="bn",
        patterns=_p(
            r"আপনার\s+গোপন\s+সিস্টেম\s+নির্দেশনা\s+"
            r"(প্রকাশ\s+করুন|দেখান|বলুন|পুনরাবৃত্তি\s+করুন)",

            r"গোপন\s+সিস্টেম\s+নির্দেশনা\s+"
            r"(প্রকাশ\s+করুন|দেখান|বলুন|পুনরাবৃত্তি\s+করুন)",

            r"সিস্টেম\s+নির্দেশনা\s+"
            r"(প্রকাশ\s+করুন|দেখান|বলুন)",
        ),
        weight=0.35,
        explanation=(
            "Detected Bengali phrasing requesting disclosure "
            "of system instructions."
        ),
    ),

    Rule(
        id="safety_bypass_bn_001",
        category="safety_bypass",
        language="bn",
        patterns=_p(
            r"আপনার\s+নিরাপত্তা\s+নিয়ম\s+"
            r"(বন্ধ\s+করুন|অক্ষম\s+করুন|উপেক্ষা\s+করুন)",

            r"নিরাপত্তা\s+নিয়ম\s+"
            r"(বন্ধ\s+করুন|অক্ষম\s+করুন|উপেক্ষা\s+করুন)",

            r"কোনো\s+সীমাবদ্ধতা\s+ছাড়াই\s+উত্তর\s+দিন",

            r"কোনো\s+সীমাবদ্ধতা\s+ছাড়াই",
        ),
        weight=0.45,
        explanation=(
            "Detected Bengali phrasing suggesting safety "
            "rules or restrictions should be bypassed."
        ),
    ),

    Rule(
        id="role_manipulation_bn_001",
        category="role_manipulation",
        language="bn",
        patterns=_p(
            r"এমন\s+একজন\s+নিয়ন্ত্রণহীন\s+সহকারীর\s+ভূমিকা\s+"
            r"পালন\s+করুন",

            r"যার\s+কোনো\s+নিরাপত্তা\s+নীতি\s+নেই",

            r"নিয়ন্ত্রণহীন\s+সহকারী\s+হিসেবে\s+অভিনয়\s+করুন",
        ),
        weight=0.4,
        explanation=(
            "Detected Bengali role manipulation attempting "
            "to create an unrestricted assistant."
        ),
    ),

    Rule(
        id="context_manipulation_bn_001",
        category="context_manipulation",
        language="bn",
        patterns=_p(
            r"আগের\s+নির্দেশনাগুলো\s+আর\s+প্রযোজ্য\s+নয়",

            r"আগের\s+সব\s+নির্দেশনা\s+আর\s+প্রযোজ্য\s+নয়",

            r"এই\s+নতুন\s+নির্দেশনাগুলোকে\s+সর্বোচ্চ\s+অগ্রাধিকার\s+দিন",

            r"নতুন\s+নির্দেশনাগুলোকে\s+সর্বোচ্চ\s+অগ্রাধিকার",
        ),
        weight=0.35,
        explanation=(
            "Detected Bengali manipulation of instruction hierarchy "
            "or conversational context."
        ),
    ),
]


# =============================================================================
# MIXED LANGUAGE
# =============================================================================

_MIXED_RULES: list[Rule] = [

    Rule(
        id="safety_bypass_mixed_001",
        category="safety_bypass",
        language="mixed",
        patterns=_p(
            r"safety\s+rules?\s+(ko|को)\s+"
            r"(ignore|bypass)\s+kar[o0]\b",

            r"safety\s+rules?\s+(को)\s+"
            r"(bypass|ignore)\s+करो",
        ),
        weight=0.4,
        explanation=(
            "Detected a romanized/mixed Hindi-English phrase "
            "suggesting safety rules should be ignored or bypassed."
        ),
    ),

    Rule(
        id="instruction_override_mixed_001",
        category="instruction_override",
        language="mixed",
        patterns=_p(
            r"(previous\s+instructions?)-?"
            r"(ஐ)?\s*"
            r"ignore\s*"
            r"(பண்ணு|pannu)",
        ),
        weight=0.35,
        explanation=(
            "Detected a romanized/mixed Tamil-English phrase "
            "suggesting previous instructions should be ignored."
        ),
    ),
]


# =============================================================================
# ALL RULES
# =============================================================================

ALL_RULES: tuple[Rule, ...] = tuple(
    _ENGLISH_RULES
    + _HINDI_RULES
    + _TAMIL_RULES
    + _TELUGU_RULES
    + _BENGALI_RULES
    + _MIXED_RULES
)