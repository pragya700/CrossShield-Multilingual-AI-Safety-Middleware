"""
Language Detection — Phase 2.

Real multilingual language identification using Lingua
(https://github.com/pemistahl/lingua-py — package name
`lingua-language-detector`).

WHY LINGUA:
- Documented strength on short text and single words/phrases.
- Native support for all five of our initial languages:
  English, Hindi, Tamil, Telugu, Bengali.
- Fully offline, with no external API/service call.
- Confidence values are produced by Lingua over the configured
  language set.

KNOWN LIMITATION — mixed-language / code-switching:
Lingua ships an experimental `detect_multiple_languages_of` method,
but testing it against this project's required mixed-language examples
showed that it does not reliably split short mixed sentences.

Because our five supported languages use distinct Unicode scripts
(Latin for English; Devanagari, Bengali, Tamil, and Telugu for the
other four), this module segments the input by Unicode script first,
then runs Lingua's confidence computation independently on each segment.

This is our aggregation choice, not a Lingua feature.

It does NOT solve fully-romanized code-switching such as:
"Safety rules ko ignore karo."

That case requires a different approach and is left for later work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lingua import Language, LanguageDetectorBuilder

from app.core.config import get_settings


# ISO-639-1-ish code <-> Lingua Language enum
_LANGUAGE_CODE_MAP: dict[str, Language] = {
    "en": Language.ENGLISH,
    "hi": Language.HINDI,
    "ta": Language.TAMIL,
    "te": Language.TELUGU,
    "bn": Language.BENGALI,
}

_CODE_BY_LANGUAGE: dict[Language, str] = {
    language: code
    for code, language in _LANGUAGE_CODE_MAP.items()
}

_UNKNOWN_CODE = "und"


# Unicode code point ranges for supported scripts
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "bn": (0x0980, 0x09FF),  # Bengali
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
}


@dataclass
class LanguageResult:
    primary_language: str
    confidence: float
    languages_detected: list[str] = field(default_factory=list)
    code_switching: bool = False


def _build_detector():
    settings = get_settings()

    languages = [
        _LANGUAGE_CODE_MAP[code]
        for code in settings.supported_languages
        if code in _LANGUAGE_CODE_MAP
    ]

    if len(languages) < 2:
        raise ValueError(
            "CrossShield language detection needs at least 2 configured "
            "supported_languages that Lingua recognizes."
        )

    return LanguageDetectorBuilder.from_languages(*languages).build()


# Build once when the application starts.
# Lingua's detector can then be reused for requests.
_detector = _build_detector()


def _char_script(ch: str) -> str | None:
    """
    Returns:
        'en'    -> ASCII/Latin
        'hi'    -> Devanagari
        'bn'    -> Bengali
        'ta'    -> Tamil
        'te'    -> Telugu
        'other' -> alphabetic character from another script
        None    -> non-alphabetic character
    """

    if not ch.isalpha():
        return None

    if ch.isascii():
        return "en"

    cp = ord(ch)

    for code, (lo, hi) in _SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return code

    return "other"


def _segment_by_script(text: str) -> list[tuple[str, str]]:
    """
    Split text into contiguous script-based runs.

    Example:

    "Hello आज कैसे हो"

    may become approximately:

    [
        ("en", "Hello"),
        ("hi", "आज कैसे हो")
    ]
    """

    scripts = [_char_script(c) for c in text]

    first_known = next(
        (script for script in scripts if script is not None),
        None,
    )

    filled: list[str | None] = []

    current = first_known

    for script in scripts:
        if script is not None:
            current = script

        filled.append(current)

    runs: list[tuple[str, str]] = []

    run_script: str | None = None
    run_start = 0

    for i, script in enumerate(filled):

        if script != run_script:

            if run_script is not None:
                segment_text = text[run_start:i].strip()

                if segment_text:
                    runs.append(
                        (run_script, segment_text)
                    )

            run_script = script
            run_start = i

    if run_script is not None:
        segment_text = text[run_start:].strip()

        if segment_text:
            runs.append(
                (run_script, segment_text)
            )

    return runs


def _confidence_for(
    text: str,
    language: Language,
) -> float:

    for confidence_value in (
        _detector.compute_language_confidence_values(text)
    ):

        if confidence_value.language == language:
            return confidence_value.value

    return 0.0


def _detect_run(
    script_code: str,
    run_text: str,
) -> tuple[str, float, int]:
    """
    Returns:

    (
        language_code,
        confidence,
        word_count
    )
    """

    word_count = len(run_text.split())

    # Unsupported alphabetic script
    if script_code == "other":
        return (
            _UNKNOWN_CODE,
            0.0,
            word_count,
        )

    # These scripts uniquely identify one of our supported languages.
    if script_code in ("hi", "bn", "ta", "te"):

        language = _LANGUAGE_CODE_MAP[script_code]

        return (
            script_code,
            _confidence_for(run_text, language),
            word_count,
        )

    # Latin script.
    # English is currently our only Latin-script target language,
    # but we still ask Lingua instead of hard-coding English.
    detected = _detector.detect_language_of(run_text)

    if detected is None:
        return (
            _UNKNOWN_CODE,
            0.0,
            word_count,
        )

    code = _CODE_BY_LANGUAGE.get(
        detected,
        _UNKNOWN_CODE,
    )

    return (
        code,
        _confidence_for(run_text, detected),
        word_count,
    )


def detect_language(text: str) -> LanguageResult:

    # Empty / whitespace / numbers / symbols
    if not text or not any(
        character.isalpha()
        for character in text
    ):

        return LanguageResult(
            primary_language=_UNKNOWN_CODE,
            confidence=0.0,
            languages_detected=[],
            code_switching=False,
        )

    runs = _segment_by_script(text)

    detected_runs = [
        (
            _detect_run(script, run_text),
            run_text,
        )
        for script, run_text in runs
    ]

    # Calculate total number of words.
    total_words = (
        sum(
            word_count
            for (_, _, word_count), _ in detected_runs
        )
        or 1
    )

    # Weighted confidence across detected script runs.
    weighted_confidence = sum(
        confidence * word_count
        for (_, confidence, word_count), _ in detected_runs
    ) / total_words

    # Primary language = language covering the largest number of words.
    primary_language = max(
        detected_runs,
        key=lambda item: item[0][2],
    )[0][0]

    # Keep unique detected languages.
    languages_detected: list[str] = []

    for (code, _, _), _ in detected_runs:

        if code not in languages_detected:
            languages_detected.append(code)

    code_switching = len(languages_detected) > 1

    return LanguageResult(
        primary_language=primary_language,
        confidence=round(
            weighted_confidence,
            4,
        ),
        languages_detected=languages_detected,
        code_switching=code_switching,
    )