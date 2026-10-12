"""
Phase 2 tests: the real Lingua-backed language detector.

Covers the 13 required cases: one clean sentence per supported language,
four Latin+Indic mixed-language cases, and edge cases (empty, very short,
numeric/symbol-only, long multilingual input) where the detector must not
crash and must not invent a language it isn't confident about.
"""

from __future__ import annotations

from app.detectors.language_detector import detect_language

# --- 1-5: one clean sentence per supported language -------------------------


def test_english_sentence():
    result = detect_language("What is machine learning?")
    assert result.primary_language == "en"
    assert result.code_switching is False
    assert result.confidence > 0.9


def test_hindi_sentence():
    result = detect_language("मशीन लर्निंग क्या है?")
    assert result.primary_language == "hi"
    assert result.code_switching is False
    assert result.confidence > 0.9


def test_tamil_sentence():
    result = detect_language("இயந்திர கற்றல் என்றால் என்ன?")
    assert result.primary_language == "ta"
    assert result.code_switching is False
    assert result.confidence > 0.9


def test_telugu_sentence():
    result = detect_language("మెషిన్ లెర్నింగ్ అంటే ఏమిటి?")
    assert result.primary_language == "te"
    assert result.code_switching is False
    assert result.confidence > 0.9


def test_bengali_sentence():
    result = detect_language("মেশিন লার্নিং কী?")
    assert result.primary_language == "bn"
    assert result.code_switching is False
    assert result.confidence > 0.9


# --- 6-9: mixed-language (code-switching) input -----------------------------


def test_mixed_english_hindi():
    result = detect_language("How are you आज?")
    assert result.code_switching is True
    assert set(result.languages_detected) == {"en", "hi"}


def test_mixed_english_hindi_reversed_dominance():
    result = detect_language("Machine learning क्या है?")
    assert result.code_switching is True
    assert set(result.languages_detected) == {"en", "hi"}


def test_mixed_english_tamil():
    result = detect_language("AI என்றால் என்ன?")
    assert result.code_switching is True
    assert set(result.languages_detected) == {"en", "ta"}


def test_mixed_english_telugu():
    # Telugu-script sentence with an English acronym prefixed.
    result = detect_language("ML అంటే ఏమిటి?")
    assert result.code_switching is True
    assert "te" in result.languages_detected


def test_mixed_english_bengali():
    result = detect_language("AI মানে কী?")
    assert result.code_switching is True
    assert "bn" in result.languages_detected


# --- 10-13: edge cases -------------------------------------------------------


def test_empty_input_returns_unknown_not_a_crash():
    result = detect_language("")
    assert result.primary_language == "und"
    assert result.confidence == 0.0
    assert result.languages_detected == []
    assert result.code_switching is False


def test_very_short_input_does_not_crash():
    # A single ASCII word is inherently ambiguous for a real language ID
    # model; we only assert it returns a well-formed result, not a specific
    # language, since 100% correctness on one word is not a realistic claim.
    result = detect_language("hi")
    assert isinstance(result.primary_language, str)
    assert 0.0 <= result.confidence <= 1.0


def test_numeric_and_symbol_only_input_returns_unknown():
    result = detect_language("12345 !!! ---")
    assert result.primary_language == "und"
    assert result.confidence == 0.0
    assert result.code_switching is False


def test_long_multilingual_input_does_not_crash():
    long_prompt = (
        "This is a long English passage about artificial intelligence. " * 5
        + "मशीन लर्निंग एक बहुत ही रोचक विषय है और इसका उपयोग बढ़ रहा है। " * 5
        + "இயந்திர கற்றல் இப்போது பரவலாக பயன்படுத்தப்படுகிறது. " * 5
    )
    result = detect_language(long_prompt)
    assert result.code_switching is True
    assert {"en", "hi", "ta"}.issubset(set(result.languages_detected))


# --- unsupported script: reported honestly as "und", not force-classified --


def test_unsupported_script_returns_unknown_not_a_guess():
    result = detect_language("こんにちは")  # Japanese: outside declared scope
    assert result.primary_language == "und"
    assert result.confidence == 0.0
