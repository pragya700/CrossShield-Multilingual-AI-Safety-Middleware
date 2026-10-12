"""
Semantic Jailbreak Detector — Phase 5 (embedding-similarity baseline).

Complements, and does NOT replace, the Phase 4 rule-based detector
(jailbreak_detector.py), which stays available as the comparison baseline.

HOW IT WORKS
    prompt -> split into sentence segments -> embed (multilingual model)
           -> cosine similarity vs. cached reference embeddings
           -> per-category best match, gated by a threshold and a
              hard-negative contrast -> semantic score + categories

WHAT THE SCORE IS (AND IS NOT)
`semantic_jailbreak_score` is a cosine *similarity* between the prompt and a
known jailbreak-intent reference, reported only when it passes the gate below
(otherwise 0.0, meaning "ran, nothing flagged"). It is NOT a probability:
0.81 does not mean "81% chance of being a jailbreak". The threshold is a
configurable PLACEHOLDER (see app/core/config.py) that has not been calibrated
on data. Different embedding models spread their scores very differently, so a
threshold is only meaningful for the model it was tuned with.

FALSE-POSITIVE CONTROL (hard negatives)
Text that merely *discusses* jailbreaks ("What does jailbreak mean in AI
security?") is embedded very close to real attacks because the vocabulary
overlaps. A bare similarity threshold cannot separate the two. So a second
reference set of benign look-alikes ("hard negatives") is embedded as well, and
an attack match only counts if it beats the best hard-negative match for the
same segment. Because this comparison is *relative*, it does not depend on the
absolute score scale of the model. Whether it actually separates discussion
from attack in practice has NOT been measured yet: see README.

SEGMENTATION
Embedding a whole long prompt dilutes a short attack sentence among benign
text, and embedding models truncate long inputs. So the full text and each
sentence are embedded and the best match wins. Only the first
`max_segments` segments are embedded; the rule-based detector still scans the
whole prompt.

FAILURE BEHAVIOUR
If the model cannot be loaded, `SemanticModelUnavailableError` is raised with an
actionable message. The detector never substitutes a 0.0 score for a failure.

NOT IN SCOPE HERE
Multi-turn/context analysis (Phase 7) and dedicated cross-lingual alignment
study (Phase 6). `context` is accepted for interface parity and ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

KNOWN_ATTACK_CATEGORIES: frozenset[str] = frozenset(
    {
        "instruction_override",
        "system_prompt_manipulation",
        "safety_bypass",
        "role_manipulation",
        "context_manipulation",
        "jailbreak_framing",
        "prompt_injection",
    }
)
HARD_NEGATIVE_CATEGORY = "hard_negative"

_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
DEFAULT_REFERENCE_PATH = _DATASETS_DIR / "jailbreak_reference_examples.json"
DEFAULT_HARD_NEGATIVE_PATH = _DATASETS_DIR / "jailbreak_hard_negatives.json"

# Sentence boundaries incl. the Devanagari/Bengali danda (U+0964) and double danda.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u0964\u0965])\s+|\n+")


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class SemanticModelUnavailableError(RuntimeError):
    """The embedding model could not be loaded. Never mapped to a fake score."""


class ReferenceDataError(ValueError):
    """A reference/hard-negative dataset file is missing or malformed."""


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceExample:
    id: str
    category: str
    language: str
    text: str


@dataclass(frozen=True)
class SemanticMatch:
    reference_id: str
    category: str
    similarity: float
    segment_index: int  # 0 is the whole prompt when it was split into segments


@dataclass
class SemanticResult:
    semantic_jailbreak_score: float
    attack_categories: list[str] = field(default_factory=list)
    matches: list[SemanticMatch] = field(default_factory=list)
    # Best raw attack-reference similarity over all segments, whether or not it
    # passed the gate. Exposed for threshold calibration; not a risk score.
    max_similarity: float = 0.0
    # True when an attack-like segment passed the threshold but was closer to a
    # benign hard-negative reference, so nothing was flagged.
    hard_negative_suppressed: bool = False
    explanation: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Embedders (model loading kept apart from inference logic)
# ----------------------------------------------------------------------------


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised embeddings."""
        ...


class SentenceTransformerEmbedder:
    """
    Lazy, thread-safe wrapper around a sentence-transformers model.

    Construction is free: the model is loaded on first `encode` call and then
    reused for the life of the process. A failed load is remembered so a
    misconfigured deployment fails fast instead of retrying a slow network
    download on every request; restart the process after fixing the cause.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
        text_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._local_files_only = local_files_only
        self._text_prefix = text_prefix
        self._batch_size = batch_size
        self._model = None
        self._load_failure: str | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _unavailable(self, exc: BaseException) -> SemanticModelUnavailableError:
        detail = f"{type(exc).__name__}: {str(exc)[:300]}"
        return SemanticModelUnavailableError(
            f"Semantic model '{self._model_name}' could not be loaded ({detail}). "
            "If this is the first run, download it once with "
            "`python scripts/download_semantic_model.py` (needs internet), or point "
            "CROSSSHIELD_SEMANTIC_MODEL_NAME at a local model directory. To run "
            "without the semantic detector set CROSSSHIELD_SEMANTIC_ENABLED=false "
            "(semantic fields are then null and the combined score equals the rule "
            "score). A failed load is remembered until the process restarts."
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failure is not None:
                raise SemanticModelUnavailableError(self._load_failure)
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading semantic model %s (device=%s)", self._model_name, self._device)
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    local_files_only=self._local_files_only,
                )
            except Exception as exc:  # broad on purpose: import, network, disk, format
                error = self._unavailable(exc)
                self._load_failure = str(error)
                raise error from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._ensure_loaded()
        prefixed = [self._text_prefix + t for t in texts]
        vectors = model.encode(
            prefixed,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------


def load_reference_examples(path: Path, *, allowed_categories: frozenset[str]) -> list[ReferenceExample]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReferenceDataError(f"Reference dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReferenceDataError(f"Reference dataset is not valid JSON ({path}): {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ReferenceDataError(f"Reference dataset must be a non-empty JSON list: {path}")

    examples: list[ReferenceExample] = []
    seen_ids: set[str] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ReferenceDataError(f"{path}: entry #{position} is not an object")
        values: dict[str, str] = {}
        for key in ("id", "category", "language", "text"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ReferenceDataError(f"{path}: entry #{position} has missing/empty '{key}'")
            values[key] = value.strip()
        if values["id"] in seen_ids:
            raise ReferenceDataError(f"{path}: duplicate id '{values['id']}'")
        if values["category"] not in allowed_categories:
            raise ReferenceDataError(
                f"{path}: entry '{values['id']}' has unknown category '{values['category']}'"
            )
        seen_ids.add(values["id"])
        examples.append(ReferenceExample(**values))
    return examples


def split_segments(text: str, max_segments: int) -> list[str]:
    """Whole text first, then its sentences (only when there is more than one)."""
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text)]
    sentences = [s for s in sentences if s and any(ch.isalpha() for ch in s)]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [text]
    return ([text] + sentences)[: max(1, max_segments)]


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------


class SemanticJailbreakDetector:
    def __init__(
        self,
        embedder: Embedder,
        references: Sequence[ReferenceExample],
        hard_negatives: Sequence[ReferenceExample] = (),
        *,
        threshold: float,
        hard_negative_margin: float = 0.0,
        max_segments: int = 32,
        top_matches: int = 3,
    ) -> None:
        if not references:
            raise ReferenceDataError("At least one attack reference example is required.")
        self._embedder = embedder
        self._references = list(references)
        self._hard_negatives = list(hard_negatives)
        self._threshold = threshold
        self._margin = hard_negative_margin
        self._max_segments = max_segments
        self._top_matches = max(1, top_matches)

        self._category_indices: dict[str, np.ndarray] = {}
        for category in dict.fromkeys(r.category for r in self._references):
            self._category_indices[category] = np.array(
                [i for i, r in enumerate(self._references) if r.category == category]
            )

        self._reference_matrix: np.ndarray | None = None
        self._negative_matrix: np.ndarray | None = None
        self._init_lock = threading.Lock()

        canonical = json.dumps(
            [[r.id, r.category, r.text] for r in self._references + self._hard_negatives],
            ensure_ascii=False,
        )
        self._dataset_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    # -- introspection ------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        """Exposes the already-constructed embedder so other detectors (Phase
        6's cross-lingual analyzer) can reuse the same loaded model instead of
        constructing and loading a second copy of it."""
        return self._embedder

    def info(self) -> dict[str, object]:
        return {
            "model": self._embedder.model_name,
            "threshold": self._threshold,
            "hard_negative_margin": self._margin,
            "n_references": len(self._references),
            "n_hard_negatives": len(self._hard_negatives),
            "dataset_fingerprint": self._dataset_fingerprint,
        }

    # -- reference cache ----------------------------------------------------

    def _ensure_reference_embeddings(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self._reference_matrix is not None:
            return self._reference_matrix, self._negative_matrix
        with self._init_lock:
            if self._reference_matrix is None:
                matrix = self._embedder.encode([r.text for r in self._references])
                if matrix.shape[0] != len(self._references):
                    raise RuntimeError("Embedder returned a wrong number of reference embeddings.")
                negatives = None
                if self._hard_negatives:
                    negatives = self._embedder.encode([r.text for r in self._hard_negatives])
                    if negatives.shape[0] != len(self._hard_negatives):
                        raise RuntimeError("Embedder returned a wrong number of hard-negative embeddings.")
                self._negative_matrix = negatives
                self._reference_matrix = matrix  # set last: readers test this one
        return self._reference_matrix, self._negative_matrix

    # -- inference ----------------------------------------------------------

    def detect(
        self,
        prompt: str,
        normalized_prompt: str,
        context: list[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> SemanticResult:
        """
        Match `normalized_prompt` against the reference sets. `prompt` (original)
        and `context` are accepted for interface parity with the rule detector
        and are not used in Phase 5. `threshold` overrides the configured
        threshold for this call only; it exists so calibration scripts can sweep
        thresholds without rebuilding the detector or re-embedding references.
        """
        del prompt, context
        active_threshold = self._threshold if threshold is None else threshold
        segments = split_segments(normalized_prompt, self._max_segments)
        if not segments:
            # Nothing linguistic to embed. This is "no signal", not a failure,
            # and deliberately does not touch (or require) the model.
            return SemanticResult(semantic_jailbreak_score=0.0)

        reference_matrix, negative_matrix = self._ensure_reference_embeddings()
        segment_vectors = self._embedder.encode(segments)
        similarities = segment_vectors @ reference_matrix.T  # (segments, references)
        if negative_matrix is not None:
            best_negative = (segment_vectors @ negative_matrix.T).max(axis=1)
        else:
            best_negative = np.full(len(segments), -np.inf, dtype=np.float32)

        max_similarity = float(similarities.max())
        best_by_category: dict[str, SemanticMatch] = {}
        suppressed_segment = False

        for s in range(len(segments)):
            row = similarities[s]
            if float(row.max()) < active_threshold:
                continue
            flagged_here = False
            for category, indices in self._category_indices.items():
                local = int(np.argmax(row[indices]))
                similarity = float(row[indices[local]])
                if similarity >= active_threshold and similarity > float(best_negative[s]) + self._margin:
                    flagged_here = True
                    current = best_by_category.get(category)
                    if current is None or similarity > current.similarity:
                        best_by_category[category] = SemanticMatch(
                            reference_id=self._references[int(indices[local])].id,
                            category=category,
                            similarity=round(min(1.0, max(0.0, similarity)), 4),
                            segment_index=s,
                        )
            if not flagged_here:
                suppressed_segment = True

        matches = sorted(best_by_category.values(), key=lambda m: m.similarity, reverse=True)
        categories = [m.category for m in matches]
        score = matches[0].similarity if matches else 0.0
        suppressed = suppressed_segment and not matches

        explanation = [
            f"Semantically similar to a known '{m.category}' reference "
            f"(cosine similarity {m.similarity:.2f} >= threshold {active_threshold:.2f}; "
            "a similarity score, not a probability)."
            for m in matches
        ]
        if suppressed:
            explanation.append(
                "Text was semantically close to an attack reference but closer to a benign "
                "hard-negative (discussion) reference, so it was not flagged."
            )

        return SemanticResult(
            semantic_jailbreak_score=score,
            attack_categories=categories,
            matches=matches[: self._top_matches],
            max_similarity=round(min(1.0, max(-1.0, max_similarity)), 4),
            hard_negative_suppressed=suppressed,
            explanation=explanation,
        )


# ----------------------------------------------------------------------------
# Construction / process-wide singleton
# ----------------------------------------------------------------------------


def build_semantic_detector(settings: Settings, embedder: Embedder | None = None) -> SemanticJailbreakDetector:
    reference_path = Path(settings.semantic_reference_path) if settings.semantic_reference_path else DEFAULT_REFERENCE_PATH
    negative_path = (
        Path(settings.semantic_hard_negative_path) if settings.semantic_hard_negative_path else DEFAULT_HARD_NEGATIVE_PATH
    )
    references = load_reference_examples(reference_path, allowed_categories=KNOWN_ATTACK_CATEGORIES)
    hard_negatives = load_reference_examples(negative_path, allowed_categories=frozenset({HARD_NEGATIVE_CATEGORY}))
    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            settings.semantic_model_name,
            device=settings.semantic_device,
            local_files_only=settings.semantic_local_files_only,
            text_prefix=settings.semantic_text_prefix,
        )
    return SemanticJailbreakDetector(
        embedder,
        references,
        hard_negatives,
        threshold=settings.semantic_threshold,
        hard_negative_margin=settings.semantic_hard_negative_margin,
        max_segments=settings.semantic_max_segments,
        top_matches=settings.semantic_top_matches,
    )


@lru_cache(maxsize=1)
def get_semantic_detector() -> SemanticJailbreakDetector:
    """One detector (and therefore one loaded model) per process. Construction is
    cheap; the model itself loads lazily on the first detect() call. Tests can
    reset it with `get_semantic_detector.cache_clear()`."""
    return build_semantic_detector(get_settings())
