"""
Phase 5 — Baseline comparison utility.

Runs the rule-based detector (Phase 4), the semantic detector (Phase 5), and
the combined score against datasets/semantic_dev_set.json, and prints a
table of: prompt, expected label, rule result, semantic result, combined
result. This is a DEVELOPMENT UTILITY for eyeballing behavior and for
calibrating semantic_threshold/hard_negative_margin -- it is NOT an
accuracy benchmark. The dev set has ~50 examples; any accuracy/precision/
recall number computed from it would be noise dressed up as a metric. Real
evaluation with a properly sized, held-out dataset is Phase 13's job.

Usage:
    python scripts/compare_detectors.py
    python scripts/compare_detectors.py --threshold 0.75 --margin 0.02
    python scripts/compare_detectors.py --csv out.csv

Requires the semantic model to be available (downloaded, or reachable over
the network) -- see README "Phase 5 -- model download". If the model
cannot be loaded, this script reports that clearly and exits non-zero
rather than printing fabricated semantic results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.detectors.jailbreak_detector import detect as detect_rule  # noqa: E402
from app.detectors.normalizer import normalize  # noqa: E402
from app.detectors.semantic_jailbreak_detector import (  # noqa: E402
    SemanticModelUnavailableError,
    build_semantic_detector,
)
from app.core.risk_engine import combine_jailbreak_scores  # noqa: E402

DEV_SET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "semantic_dev_set.json"


def label_for(rule_score: float, semantic_score: float | None, combined_score: float, threshold: float) -> str:
    del rule_score, semantic_score
    return "flagged" if combined_score >= threshold else "not_flagged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None, help="Override semantic_threshold for this run.")
    parser.add_argument("--margin", type=float, default=None, help="Override semantic_hard_negative_margin.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write results as CSV.")
    args = parser.parse_args()

    settings = get_settings()
    try:
        detector = build_semantic_detector(settings)
        # Touch the model once up front so a failure is reported before we
        # print a half-finished table.
        detector._ensure_reference_embeddings()  # noqa: SLF001 (dev script, intentional)
    except SemanticModelUnavailableError as exc:
        print(f"Semantic model unavailable: {exc}", file=sys.stderr)
        print(
            "Cannot run a rule-vs-semantic comparison without the model. "
            "Download it first (see README 'Phase 5 -- model download') or "
            "run on a machine with network access to huggingface.co.",
            file=sys.stderr,
        )
        return 1

    threshold = args.threshold if args.threshold is not None else settings.semantic_threshold
    margin = args.margin if args.margin is not None else settings.semantic_hard_negative_margin

    dev_set = json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))
    rows = []
    for item in dev_set:
        text = item["text"]
        norm = normalize(text)
        rule_result = detect_rule(text, norm.normalized_text)
        semantic_result = detector.detect(text, norm.normalized_text, threshold=threshold)
        # margin override requires rebuilding; for a single dev-set run we
        # just report the configured margin used, not an override per-row.
        combined = combine_jailbreak_scores(
            rule_result.jailbreak_score, semantic_result.semantic_jailbreak_score, settings
        )
        rows.append(
            {
                "id": item["id"],
                "expected_label": item["expected_label"],
                "language": item.get("language", ""),
                "category": item.get("category", ""),
                "text": text,
                "rule_score": round(rule_result.jailbreak_score, 3),
                "rule_categories": ";".join(rule_result.attack_categories),
                "semantic_score": round(semantic_result.semantic_jailbreak_score, 3),
                "semantic_categories": ";".join(semantic_result.attack_categories),
                "combined_score": round(combined, 3),
                "combined_label": label_for(
                    rule_result.jailbreak_score, semantic_result.semantic_jailbreak_score, combined, threshold
                ),
            }
        )

    header = [
        "id", "expected_label", "language", "combined_label", "combined_score",
        "rule_score", "rule_categories", "semantic_score", "semantic_categories", "text",
    ]
    col_widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in header}
    print(" | ".join(h.ljust(col_widths[h]) for h in header))
    print("-+-".join("-" * col_widths[h] for h in header))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(col_widths[h])[: col_widths[h]] for h in header))

    print(
        f"\n{len(rows)} dev-set examples. threshold={threshold} margin={margin} "
        f"model={settings.semantic_model_name}"
    )
    print(
        "NOTE: this is a ~50-example development dev set, not a validated "
        "benchmark. Do not report accuracy/precision/recall from it as if it "
        "were one -- see Phase 13 for real evaluation methodology."
    )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
