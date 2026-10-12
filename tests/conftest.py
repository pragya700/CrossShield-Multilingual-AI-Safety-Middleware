"""
Shared pytest configuration.

The semantic detector needs an embedding model that may have to be
downloaded, so app/core/config.py defaults CROSSSHIELD_SEMANTIC_ENABLED to
false for a fresh checkout. This file makes that explicit and unconditional
for the whole test process (set BEFORE the app's settings are imported), so
Phase 1-4 tests never depend on the model or on network access regardless of
what a developer's local .env happens to set. Tests that exercise the
semantic path opt in explicitly, either with a test-double embedder
(tests/test_semantic_detector.py) or, when a real model is available and
CROSSSHIELD_RUN_MODEL_TESTS=1, with the real one (tests/test_semantic_model.py).
"""

from __future__ import annotations

import os

os.environ.setdefault("CROSSSHIELD_SEMANTIC_ENABLED", "false")
