"""
Text Normalization — Phase 3.

Turns a raw prompt into a consistent representation for later security
analysis, without ever discarding the original. Pipeline (each stage is a
small, independently testable function):

    original text
        -> unicode normalization (NFC)
        -> whitespace normalization (line-aware)
        -> control / invisible character handling
        -> repeated character analysis
        -> basic obfuscation normalization (spaced/dotted-out Latin letters)
        -> normalized_text

SECURITY PRINCIPLE: `original_text` is never modified or discarded. Every
stage operates on a working copy, and every stage that actually changes
something records a tag in `transformations` instead of changing things
silently. Later phases (jailbreak/cross-lingual/context detectors) can
compare original vs. normalized, and can inspect which transformations
fired, e.g. to treat "heavily obfuscated" prompts as more suspicious --
that judgment call belongs to those phases, not to this one.

DESIGN DECISIONS WORTH KNOWING ABOUT:

- NFC, not NFKC. NFKC would also collapse "compatibility" variants (e.g.
  fullwidth Latin letters, some ligatures) into their canonical form, which
  would incidentally defeat one known obfuscation trick (fullwidth-letter
  keyword evasion) -- but it also risks subtly altering legitimate text in
  scripts we haven't specifically vetted it against. Per the "don't blindly
  apply transformations that could change semantic meaning" requirement,
  this phase uses the more conservative NFC. Fullwidth/homoglyph
  normalization is a documented candidate for a later, more deliberate pass
  (see Known Limitations in the README) rather than being folded in here.

- Zero-width joiner (U+200D) and zero-width non-joiner (U+200C) are
  deliberately NEVER treated as suspicious or stripped. They are ordinary,
  meaning-bearing characters in Hindi/Bengali/Tamil/Telugu text (they
  control conjunct/half-form rendering), not obfuscation. Other zero-width
  and bidi-control characters, which have no legitimate role in an LLM
  prompt, are stripped from normalized_text (never from original_text) and
  recorded as a transformation.

- Repeated-character collapsing caps a run at 2 repeats, not 1, so
  legitimately doubled letters ("kitten", "sheep") are left alone; only
  runs of 3 or more identical characters in a row are shortened.

- The spaced/dotted-letter obfuscation pass ("i g n o r e", "i.g.n.o.r.e")
  is scoped to Latin letters only. Indic-script conjunct/syllable
  boundaries are complex enough that a naive single-character heuristic
  risks corrupting legitimate text, so it is intentionally not applied
  there yet.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Zero-width / bidi-control characters with no legitimate role in prompt
# text. Deliberately excludes U+200C (ZWNJ) and U+200D (ZWJ), which are
# meaningful in Devanagari/Bengali/Tamil/Telugu text.
_INVISIBLE_STRIP_CHARS: frozenset[str] = frozenset(
    "\u200b"  # ZERO WIDTH SPACE
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u2060"  # WORD JOINER
    "\u00ad"  # SOFT HYPHEN
    "\u202a\u202b\u202c\u202d\u202e"  # bidi embedding/override
    "\u2066\u2067\u2068\u2069"  # bidi isolates
)

_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.UNICODE)

# Three or more single Latin letters, each separated by one space/dot/dash/
# underscore, e.g. "i g n o r e" or "i.g.n.o.r.e" or "i-g-n-o-r-e".
_SPACED_LATIN_LETTERS_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z][ .\-_]){2,}[A-Za-z](?![A-Za-z])"
)


@dataclass
class NormalizationResult:
    original_text: str
    normalized_text: str
    transformations: list[str] = field(default_factory=list)
    has_invisible_characters: bool = False
    has_repeated_characters: bool = False
    # Structural signal only (text is entirely Latin-script letters): NOT a
    # language or transliteration judgement. That analysis is Phase 6's job;
    # this just gives it something cheap to key off later.
    possible_transliteration_signal: bool = False


def _normalize_unicode(text: str) -> tuple[str, bool]:
    result = unicodedata.normalize("NFC", text)
    return result, result != text


def _normalize_whitespace(text: str) -> tuple[str, bool]:
    original = text
    # Normalize line endings first so \r\n / \r don't create phantom lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    lines = [_HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip() for line in lines]
    text = "\n".join(lines)
    # Collapse runs of 3+ blank lines down to a single blank line (one
    # paragraph break), rather than deleting line structure entirely --
    # later contextual/multi-step analysis may care about paragraph breaks.
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    text = text.strip()
    return text, text != original


def _handle_invisible_characters(text: str) -> tuple[str, bool, bool]:
    """Returns (text_with_suspicious_invisibles_removed, found_any, changed)."""
    found = any(ch in _INVISIBLE_STRIP_CHARS for ch in text)
    if not found:
        # Still check for other non-whitespace control characters (Cc)
        # outside \t/\n/\r -- these have no place in prompt text either.
        found = any(
            unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in text
        )
    if not found:
        return text, False, False

    cleaned_chars = [
        ch
        for ch in text
        if ch not in _INVISIBLE_STRIP_CHARS
        and not (unicodedata.category(ch) == "Cc" and ch not in "\t\n\r")
    ]
    cleaned = "".join(cleaned_chars)
    return cleaned, True, cleaned != text


def _normalize_repeated_characters(text: str) -> tuple[str, bool, bool]:
    """Returns (normalized_text, found_repetition, changed)."""
    found = _REPEATED_CHAR_RE.search(text) is not None
    if not found:
        return text, False, False
    collapsed = _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * 2, text)
    return collapsed, True, collapsed != text


def _normalize_basic_obfuscation(text: str) -> tuple[str, bool]:
    def _collapse(match: "re.Match[str]") -> str:
        return re.sub(r"[ .\-_]", "", match.group(0))

    result = _SPACED_LATIN_LETTERS_RE.sub(_collapse, text)
    return result, result != text


def _is_latin_only(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return all(ch.isascii() for ch in letters)


def normalize(text: str) -> NormalizationResult:
    original_text = text
    transformations: list[str] = []

    working, changed = _normalize_unicode(text)
    if changed:
        transformations.append("unicode_normalized")

    working, changed = _normalize_whitespace(working)
    if changed:
        transformations.append("whitespace_normalized")

    working, has_invisible, changed = _handle_invisible_characters(working)
    if has_invisible:
        transformations.append("invisible_character_detected")
    if changed:
        transformations.append("invisible_character_removed")

    working, has_repeats, changed = _normalize_repeated_characters(working)
    if has_repeats:
        transformations.append("repeated_character_detected")
    if changed:
        transformations.append("repeated_character_normalized")

    working, changed = _normalize_basic_obfuscation(working)
    if changed:
        transformations.append("basic_obfuscation_normalized")

    return NormalizationResult(
        original_text=original_text,
        normalized_text=working,
        transformations=transformations,
        has_invisible_characters=has_invisible,
        has_repeated_characters=has_repeats,
        possible_transliteration_signal=_is_latin_only(working),
    )
