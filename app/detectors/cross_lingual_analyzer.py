"""
Cross-Lingual Semantic Analyzer — Phase 6.

Complements, and does NOT replace, the Phase 5 semantic detector
(semantic_jailbreak_detector.py). Phase 5 asks "is this prompt semantically
close to ANY known jailbreak-intent reference?" using one large, mixed-
language reference pool. Phase 6 asks a narrower, distinct question: "is
this prompt's intent the SAME as a specific risky (or benign) intent that we
have deliberately expressed in all five supported languages, and does the
model actually place those aligned translations near each other?" -- i.e.
whether unsafe (or benign) semantic intent remains stable across language
representations, not just whether any single-language reference happens to
match.

HOW IT WORKS
    prompt -> split into sentence segments (reuses Phase 5's segmentation)
           -> embed (the SAME cached multilingual model Phase 5 loaded --
              see build_cross_lingual_analyzer, which never constructs a
              second embedder)
           -> cosine similarity vs. cached ALIGNED GROUP embeddings
              (datasets/cross_lingual_reference_examples.json)
           -> best-matching group, gated by a threshold and a benign-group
              contrast (mirrors Phase 5's hard-negative gate, but here the
              "negative" set is the benign-group examples in this same
              dataset, not a separate hard-negative file)
           -> cross-lingual score + which languages that group spans

WHAT THE SCORE IS (AND IS NOT)
Exactly as in Phase 5: `cross_lingual_score` is a cosine *similarity*, not a
probability, and a high score does not by itself mean "this is a jailbreak"
-- it means "this prompt's intent matches a group we have confirmed (by
construction) expresses the same idea in multiple languages, and that group
happens to be labeled jailbreak_risk". A multilingual, translated, mixed, or
Romanized prompt is NEVER scored as risky merely for being non-English,
mixed, or similar across languages -- only matching a jailbreak_risk group
(net of the benign-group contrast) contributes to the score. Matching a
benign group contributes nothing extra; it exists to make the gate honest.

ALIGNMENT QUALITY (the Phase 6-specific measurement)
Separately from per-request scoring, `alignment_report()` reports, for every
group with 2+ languages, the pairwise cosine similarity between each
language's example and the group's English anchor (when present). This is
the direct, measurable answer to "does the embedding model actually place
same-intent translations near each other", computed once when the model is
available -- not claimed, not assumed. It requires the real model and is
NOT available in this project's development sandbox (see README).

FALSE-POSITIVE CONTROL
Mirrors Phase 5 exactly: a prompt matching a jailbreak_risk group only
counts if that match beats the best *benign-group* match for the same
segment by more than `margin`. This is what keeps "same question, different
language" or "a translated benign question" from being treated as risky
purely for being multilingual or cross-lingually similar.

FAILURE BEHAVIOUR
Mirrors Phase 5: if the shared embedder cannot be loaded,
`SemanticModelUnavailableError` propagates (it is the SAME exception type
Phase 5 raises, since it is the SAME underlying model). If the Phase 6
dataset itself is missing/malformed, `ReferenceDataError` propagates (reusing
Phase 5's loader and exception type). Neither path ever substitutes a fake
0.0 score.

NOT IN SCOPE HERE
- Real-time machine translation of the incoming prompt (there is no
  translation step; the prompt is compared directly, in whatever language
  it is in, against the pre-built aligned reference embeddings).
- A general romanized/transliteration solution: the "mixed" entries in the
  Phase 6 dataset are the same narrow, example-scoped romanized phrasings
  already documented as limited in Phase 4/5 -- not a new capability.
- Multi-turn/context analysis (Phase 7). `context` is accepted for
  interface parity and ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.config import Settings, get_settings
from app.detectors.semantic_jailbreak_detector import (
    Embedder,
    ReferenceDataError,
    SemanticModelUnavailableError,
    get_semantic_detector,
    split_segments,
)

logger = logging.getLogger(__name__)

JAILBREAK_RISK = "jailbreak_risk"
BENIGN = "benign"
_ALLOWED_RISK_LABELS = frozenset({JAILBREAK_RISK, BENIGN})

_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
DEFAULT_CROSS_LINGUAL_PATH = _DATASETS_DIR / "cross_lingual_reference_examples.json"


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossLingualExample:
    id: str
    group_id: str
    risk_label: str  # "jailbreak_risk" | "benign"
    category: str
    language: str
    text: str


@dataclass(frozen=True)
class CrossLingualMatch:
    group_id: str
    reference_id: str
    category: str
    language: str
    similarity: float
    segment_index: int  # 0 is the whole prompt when it was split into segments


@dataclass
class CrossLingualResult:
    cross_lingual_score: float
    attack_categories: list[str] = field(default_factory=list)
    matches: list[CrossLingualMatch] = field(default_factory=list)
    # Every language present in the best-matched group's aligned set -- the
    # direct answer to "which language representations share this intent".
    languages: list[str] = field(default_factory=list)
    max_similarity: float = 0.0
    suppressed: bool = False
    explanation: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Dataset loading (reuses Phase 5's generic loader for the per-entry schema,
# then validates the two Phase-6-specific fields: group_id and risk_label)
# ----------------------------------------------------------------------------


def load_cross_lingual_examples(path: Path) -> list[CrossLingualExample]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReferenceDataError(f"Cross-lingual dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReferenceDataError(f"Cross-lingual dataset is not valid JSON ({path}): {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ReferenceDataError(f"Cross-lingual dataset must be a non-empty JSON list: {path}")

    examples: list[CrossLingualExample] = []
    seen_ids: set[str] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ReferenceDataError(f"{path}: entry #{position} is not an object")
        values: dict[str, str] = {}
        for key in ("id", "group_id", "risk_label", "category", "language", "text"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ReferenceDataError(f"{path}: entry #{position} has missing/empty '{key}'")
            values[key] = value.strip()
        if values["id"] in seen_ids:
            raise ReferenceDataError(f"{path}: duplicate id '{values['id']}'")
        if values["risk_label"] not in _ALLOWED_RISK_LABELS:
            raise ReferenceDataError(
                f"{path}: entry '{values['id']}' has unknown risk_label "
                f"'{values['risk_label']}' (must be one of {sorted(_ALLOWED_RISK_LABELS)})"
            )
        seen_ids.add(values["id"])
        examples.append(CrossLingualExample(**values))
    return examples


# ----------------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------------


class CrossLingualAnalyzer:
    def __init__(
        self,
        embedder: Embedder,
        examples: Sequence[CrossLingualExample],
        *,
        threshold: float,
        margin: float = 0.0,
        max_segments: int = 32,
        top_matches: int = 3,
    ) -> None:
        risk_examples = [e for e in examples if e.risk_label == JAILBREAK_RISK]
        benign_examples = [e for e in examples if e.risk_label == BENIGN]
        if not risk_examples:
            raise ReferenceDataError("At least one jailbreak_risk cross-lingual example is required.")
        if not benign_examples:
            raise ReferenceDataError(
                "At least one benign cross-lingual example is required (used as the "
                "false-positive contrast set, the same role Phase 5's hard negatives play)."
            )

        self._embedder = embedder
        self._risk_examples = risk_examples
        self._benign_examples = benign_examples
        self._threshold = threshold
        self._margin = margin
        self._max_segments = max_segments
        self._top_matches = max(1, top_matches)

        # group_id -> every language present in that group (across BOTH risk
        # and benign examples sharing that group_id, though in practice a
        # group is entirely one risk_label by dataset convention).
        self._group_languages: dict[str, list[str]] = {}
        for example in examples:
            langs = self._group_languages.setdefault(example.group_id, [])
            if example.language not in langs:
                langs.append(example.language)

        self._risk_matrix: np.ndarray | None = None
        self._benign_matrix: np.ndarray | None = None
        self._init_lock = threading.Lock()

        canonical = json.dumps(
            [[e.id, e.group_id, e.risk_label, e.category, e.language, e.text] for e in examples],
            ensure_ascii=False,
        )
        self._dataset_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    # -- introspection --------------------------------------------------

    def info(self) -> dict[str, object]:
        return {
            "model": self._embedder.model_name,
            "threshold": self._threshold,
            "margin": self._margin,
            "n_risk_examples": len(self._risk_examples),
            "n_benign_examples": len(self._benign_examples),
            "n_groups": len(self._group_languages),
            "dataset_fingerprint": self._dataset_fingerprint,
        }

    def alignment_report(self) -> list[dict[str, object]]:
        """
        For every group with 2+ languages and an English ('en') anchor, reports
        the cosine similarity between that anchor and each other language's
        aligned example. This is the direct, measurable answer to Phase 6's
        core question ("does the model place same-intent translations near
        each other?"). Requires the real embedder (raises
        SemanticModelUnavailableError if it cannot be loaded) and is computed
        fresh each call -- it is a diagnostic/calibration tool
        (scripts/compare_detectors.py-style usage), not part of the hot
        per-request detect() path.
        """
        by_group: dict[str, list[CrossLingualExample]] = {}
        for example in self._risk_examples + self._benign_examples:
            by_group.setdefault(example.group_id, []).append(example)

        report: list[dict[str, object]] = []
        for group_id, members in by_group.items():
            anchor = next((m for m in members if m.language == "en"), None)
            others = [m for m in members if m is not anchor]
            if anchor is None or not others:
                continue
            texts = [anchor.text] + [m.text for m in others]
            vectors = self._embedder.encode(texts)
            anchor_vec, other_vecs = vectors[0], vectors[1:]
            similarities = other_vecs @ anchor_vec
            report.append(
                {
                    "group_id": group_id,
                    "risk_label": members[0].risk_label,
                    "anchor_language": "en",
                    "per_language_similarity_to_anchor": {
                        m.language: round(float(sim), 4) for m, sim in zip(others, similarities)
                    },
                }
            )
        return report

    # -- reference cache --------------------------------------------------

    def _ensure_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        if self._risk_matrix is not None:
            return self._risk_matrix, self._benign_matrix  # type: ignore[return-value]
        with self._init_lock:
            if self._risk_matrix is None:
                risk_matrix = self._embedder.encode([e.text for e in self._risk_examples])
                benign_matrix = self._embedder.encode([e.text for e in self._benign_examples])
                if risk_matrix.shape[0] != len(self._risk_examples):
                    raise RuntimeError("Embedder returned a wrong number of risk-example embeddings.")
                if benign_matrix.shape[0] != len(self._benign_examples):
                    raise RuntimeError("Embedder returned a wrong number of benign-example embeddings.")
                self._benign_matrix = benign_matrix
                self._risk_matrix = risk_matrix  # set last: readers test this one
        return self._risk_matrix, self._benign_matrix  # type: ignore[return-value]

    # -- inference ----------------------------------------------------------

    def detect(
        self,
        prompt: str,
        normalized_prompt: str,
        context: list[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> CrossLingualResult:
        """
        Matches `normalized_prompt` against the aligned cross-lingual groups.
        Same interface shape as the Phase 4/5 detectors for pipeline
        consistency; `prompt` and `context` are accepted but unused here
        (same reasons as Phase 5 -- see module docstring).
        """
        del prompt, context
        active_threshold = self._threshold if threshold is None else threshold
        segments = split_segments(normalized_prompt, self._max_segments)
        if not segments:
            return CrossLingualResult(cross_lingual_score=0.0)

        risk_matrix, benign_matrix = self._ensure_embeddings()
        segment_vectors = self._embedder.encode(segments)
        risk_similarities = segment_vectors @ risk_matrix.T  # (segments, risk_examples)
        benign_similarities = segment_vectors @ benign_matrix.T  # (segments, benign_examples)
        best_benign_per_segment = benign_similarities.max(axis=1)

        max_similarity = float(risk_similarities.max())
        best_by_group: dict[str, CrossLingualMatch] = {}
        suppressed_segment = False

        for s in range(len(segments)):
            row = risk_similarities[s]
            if float(row.max()) < active_threshold:
                continue
            best_idx = int(np.argmax(row))
            similarity = float(row[best_idx])
            if similarity >= active_threshold and similarity > float(best_benign_per_segment[s]) + self._margin:
                example = self._risk_examples[best_idx]
                current = best_by_group.get(example.group_id)
                if current is None or similarity > current.similarity:
                    best_by_group[example.group_id] = CrossLingualMatch(
                        group_id=example.group_id,
                        reference_id=example.id,
                        category=example.category,
                        language=example.language,
                        similarity=round(min(1.0, max(0.0, similarity)), 4),
                        segment_index=s,
                    )
            else:
                suppressed_segment = True

        matches = sorted(best_by_group.values(), key=lambda m: m.similarity, reverse=True)
        categories = [m.category for m in matches]
        score = matches[0].similarity if matches else 0.0
        suppressed = suppressed_segment and not matches

        languages: list[str] = []
        for match in matches:
            for lang in self._group_languages.get(match.group_id, []):
                if lang not in languages:
                    languages.append(lang)

        explanation = [
            f"Semantically matches the '{m.group_id}' cross-lingual reference group "
            f"(category '{m.category}', matched via its {m.language} example; cosine "
            f"similarity {m.similarity:.2f} >= threshold {active_threshold:.2f}; a "
            "similarity score, not a probability). This group's intent is expressed "
            f"across: {', '.join(self._group_languages.get(m.group_id, []))}."
            for m in matches
        ]
        if suppressed:
            explanation.append(
                "Text was semantically close to a jailbreak-risk cross-lingual group "
                "but closer to a benign aligned group expressing a different intent in "
                "the same languages, so it was not flagged."
            )

        return CrossLingualResult(
            cross_lingual_score=score,
            attack_categories=categories,
            matches=matches[: self._top_matches],
            languages=languages,
            max_similarity=round(min(1.0, max(-1.0, max_similarity)), 4),
            suppressed=suppressed,
            explanation=explanation,
        )


# ----------------------------------------------------------------------------
# Construction / process-wide singleton
# ----------------------------------------------------------------------------


def build_cross_lingual_analyzer(
    settings: Settings, embedder: Embedder | None = None
) -> CrossLingualAnalyzer:
    path = (
        Path(settings.cross_lingual_reference_path)
        if settings.cross_lingual_reference_path
        else DEFAULT_CROSS_LINGUAL_PATH
    )
    examples = load_cross_lingual_examples(path)
    if embedder is None:
        # Reuse the SAME loaded embedder Phase 5 uses -- never construct a
        # second SentenceTransformerEmbedder (and therefore never load a
        # second copy of the model into memory). Phase 6 is only meaningful
        # when Phase 5 is enabled, since there is otherwise no embedder to
        # reuse; build_cross_lingual_analyzer is only called from the API
        # layer when both settings.semantic_enabled and
        # settings.cross_lingual_enabled are true (see routes.py).
        embedder = get_semantic_detector().embedder
    return CrossLingualAnalyzer(
        embedder,
        examples,
        threshold=settings.cross_lingual_threshold,
        margin=settings.cross_lingual_margin,
        max_segments=settings.cross_lingual_max_segments,
        top_matches=settings.cross_lingual_top_matches,
    )


@lru_cache(maxsize=1)
def get_cross_lingual_analyzer() -> CrossLingualAnalyzer:
    """One analyzer per process, reusing Phase 5's already-cached embedder.
    Tests can reset it with `get_cross_lingual_analyzer.cache_clear()`."""
    return build_cross_lingual_analyzer(get_settings())
