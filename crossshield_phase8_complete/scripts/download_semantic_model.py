"""
Phase 5 — one-time helper to download and cache the semantic model.

Downloads app/core/config.py's configured `semantic_model_name` (default:
sentence-transformers/LaBSE, ~1.8 GB) into the local huggingface cache
(usually ~/.cache/huggingface) so that later runs -- including with
CROSSSHIELD_SEMANTIC_LOCAL_FILES_ONLY=true -- do not need network access.

Usage:
    python scripts/download_semantic_model.py

Requires network access to huggingface.co (and its CDN). If that is
blocked in your environment (e.g. a restricted sandbox), this will fail
with a clear connection error -- that is expected, not a bug in this
script; see README "Phase 5 -- model download" for what to do instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    model_name = settings.semantic_model_name
    print(f"Downloading semantic model: {model_name}")
    print("This can take a while on first run (LaBSE is roughly 1.8 GB).")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        vector = model.encode(["connectivity check"])
        print(f"Downloaded and loaded OK. Embedding dimension: {vector.shape[-1]}")
        return 0
    except Exception as exc:  # broad on purpose: this is a diagnostic script
        print(f"Download/load FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check your network connection and that huggingface.co is reachable "
            "from this machine. If you are behind a restrictive proxy/firewall, "
            "you may need to allowlist huggingface.co and its CDN "
            "(cdn-lfs.huggingface.co), or set HF_HOME to a directory where you "
            "can manually place a pre-downloaded model.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
