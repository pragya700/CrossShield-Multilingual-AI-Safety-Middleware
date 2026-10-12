"""Phase 11 tests for metadata-only SQLite security audit persistence."""

from __future__ import annotations

import json
import sqlite3

from app.audit.security_event import SecurityAuditEvent
from app.audit.sqlite_store import SQLiteAuditStore


def _event() -> SecurityAuditEvent:
    return SecurityAuditEvent(
        event_id="event-123",
        request_id="request-123",
        occurred_at="2026-10-04T00:00:00+00:00",
        endpoint="/generate",
        language="en",
        input_risk_level="LOW",
        input_decision="ALLOW",
        overall_risk=0.1,
        risk_analysis_status="complete",
        risk_breakdown={"analysis_status": "complete", "signals": []},
        attack_categories=["instruction_override"],
        matched_signals=["rule-1"],
        llm_called=True,
        llm_provider="mock",
        llm_model="mock-model",
        output_safety_checked=True,
        output_safety_decision="ALLOW",
        output_safety_score=0.0,
        output_categories=[],
        output_matched_signals=[],
        blocked_reason=None,
        request_latency_ms=12.5,
        outcome="completed",
    )


def test_database_initialization_creates_append_only_event_table(tmp_path):
    db_path = tmp_path / "audit.db"
    SQLiteAuditStore(str(db_path)).initialize()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(security_audit_events)")
        }

    assert {"event_id", "request_id", "outcome", "request_latency_ms"} <= columns
    assert "prompt" not in columns
    assert "llm_response" not in columns


def test_complete_metadata_event_is_inserted_without_raw_content(tmp_path):
    db_path = tmp_path / "audit.db"
    store = SQLiteAuditStore(str(db_path))
    store.write_event(_event())

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT request_id, llm_called, llm_provider, output_safety_decision,
                   risk_breakdown_json, attack_categories_json, outcome
            FROM security_audit_events
            """
        ).fetchone()

    assert row[:4] == ("request-123", 1, "mock", "ALLOW")
    assert json.loads(row[4]) == {"analysis_status": "complete", "signals": []}
    assert json.loads(row[5]) == ["instruction_override"]
    assert row[6] == "completed"
    assert b"raw prompt" not in db_path.read_bytes()
    assert b"raw LLM response" not in db_path.read_bytes()
