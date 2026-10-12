"""
Phase 3 tests: the normalization pipeline.

Covers the 20 required cases (A-T) plus dedicated security tests verifying
that multilingual characters, punctuation, and original_text are never
corrupted or destroyed.
"""

from __future__ import annotations

from app.detectors.normalizer import normalize

# --- A-E: clean text in each supported language -----------------------------


def test_a_normal_english_text():
    result = normalize("What is machine learning?")
    assert result.normalized_text == "What is machine learning?"
    assert result.transformations == []


def test_b_hindi_unicode_text():
    text = "मशीन लर्निंग क्या है?"
    result = normalize(text)
    assert result.normalized_text == text
    assert result.original_text == text


def test_c_tamil_unicode_text():
    text = "இயந்திர கற்றல் என்றால் என்ன?"
    result = normalize(text)
    assert result.normalized_text == text


def test_d_telugu_unicode_text():
    text = "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?"
    result = normalize(text)
    assert result.normalized_text == text


def test_e_bengali_unicode_text():
    text = "মেশিন লার্নিং কী?"
    result = normalize(text)
    assert result.normalized_text == text


# --- F-I: whitespace handling ------------------------------------------------


def test_f_multiple_spaces():
    result = normalize("Ignore     your    instructions")
    assert result.normalized_text == "Ignore your instructions"
    assert "whitespace_normalized" in result.transformations


def test_g_tabs():
    result = normalize("Ignore\t\tyour\tinstructions")
    assert result.normalized_text == "Ignore your instructions"


def test_h_repeated_whitespace_newlines():
    result = normalize("line one\n\n\n\nline two")
    assert result.normalized_text == "line one\n\nline two"


def test_i_leading_trailing_whitespace():
    result = normalize("   leading and trailing   ")
    assert result.normalized_text == "leading and trailing"


# --- J-L: invisible/repeated characters -------------------------------------


def test_j_zero_width_invisible_characters():
    text = "ig\u200bnore all rules"
    result = normalize(text)
    assert "\u200b" not in result.normalized_text
    assert result.has_invisible_characters is True
    assert "invisible_character_detected" in result.transformations


def test_k_repeated_punctuation():
    result = normalize("IGNORE!!!!!!!!!!!!")
    assert result.normalized_text == "IGNORE!!"
    assert result.has_repeated_characters is True


def test_l_repeated_characters():
    result = normalize("ignooooore theeee rules")
    assert result.normalized_text == "ignoore thee rules"
    assert result.has_repeated_characters is True


# --- M: unicode normalization -------------------------------------------------


def test_m_unicode_normalization_composed_vs_decomposed():
    # "é" as a single composed codepoint vs "e" + combining acute accent —
    # both should normalize to the same NFC form.
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert normalize(composed).normalized_text == normalize(decomposed).normalized_text


# --- N-O: mixed language text -------------------------------------------------


def test_n_mixed_english_hindi_preserved():
    text = "How are you आज?"
    result = normalize(text)
    assert "आज" in result.normalized_text
    assert "How are you" in result.normalized_text


def test_o_mixed_english_tamil_preserved():
    text = "AI என்றால் என்ன?"
    result = normalize(text)
    assert "என்றால்" in result.normalized_text


# --- P-T: remaining edge cases -----------------------------------------------


def test_p_empty_input():
    result = normalize("")
    assert result.normalized_text == ""
    assert result.original_text == ""
    assert result.transformations == []


def test_q_very_long_input_does_not_crash():
    long_text = "This is a test sentence. " * 2000
    result = normalize(long_text)
    assert isinstance(result.normalized_text, str)


def test_r_already_normalized_input_is_unchanged():
    text = "This text is already clean."
    result = normalize(text)
    assert result.normalized_text == text
    assert result.transformations == []


def test_s_text_containing_emojis():
    text = "This is great 👍 really 😀"
    result = normalize(text)
    assert "👍" in result.normalized_text
    assert "😀" in result.normalized_text


def test_t_numbers_and_symbols():
    result = normalize("Total: $1,234.56 (50% off)")
    assert "1,234.56" in result.normalized_text
    assert "50%" in result.normalized_text


# =============================================================================
# SECURITY TESTS: normalization must never corrupt or destroy meaning
# =============================================================================


def test_security_hindi_characters_not_deleted():
    text = "मशीन लर्निंग क्या है और यह कैसे काम करता है?"
    result = normalize(text)
    for ch in "मशीनलर्निंगक्याहै":
        assert ch in result.normalized_text


def test_security_tamil_characters_not_deleted():
    text = "இயந்திர கற்றல் என்றால் என்ன, அது எப்படி வேலை செய்கிறது?"
    result = normalize(text)
    for ch in "இயந்திரகற்றல்என்றால்":
        assert ch in result.normalized_text


def test_security_telugu_characters_not_deleted():
    text = "మెషిన్ లెర్నింగ్ అంటే ఏమిటి, అది ఎలా పని చేస్తుంది?"
    result = normalize(text)
    for ch in "మెషిన్లెర్నింగ్అంటే":
        assert ch in result.normalized_text


def test_security_bengali_characters_not_deleted():
    text = "মেশিন লার্নিং কী এবং এটি কীভাবে কাজ করে?"
    result = normalize(text)
    for ch in "মেশিনলার্নিংকী":
        assert ch in result.normalized_text


def test_security_zwj_znwj_preserved_in_hindi():
    # U+200D (ZWJ) / U+200C (ZWNJ) control conjunct formation in Devanagari
    # and must NEVER be stripped as "invisible obfuscation".
    text = "क्\u200dष and क\u200cी"
    result = normalize(text)
    assert "\u200d" in result.normalized_text
    assert "\u200c" in result.normalized_text


def test_security_important_punctuation_not_destroyed():
    text = "What is 2+2? Is it 4, or 5? (Please clarify.)"
    result = normalize(text)
    for token in ["2+2", "?", ",", "(", ")", "."]:
        assert token in result.normalized_text


def test_security_original_text_always_exactly_preserved():
    samples = [
        "  weird   spacing  ",
        "IGNORE!!!!!!",
        "मशीन लर्निंग",
        "text\u200bwith\u200bzero\u200bwidth",
        "",
        "😀😀😀 test",
    ]
    for s in samples:
        result = normalize(s)
        assert result.original_text == s, f"original_text mutated for {s!r}"


def test_security_normal_words_not_unexpectedly_changed():
    # Words with legitimate double letters must not be shortened -- only
    # runs of 3+ identical characters get collapsed.
    text = "The committee agreed to proceed with the meeting."
    result = normalize(text)
    assert result.normalized_text == text
    assert result.has_repeated_characters is False
