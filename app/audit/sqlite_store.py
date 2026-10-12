"""Append-only SQLite storage for metadata-only security audit events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.audit.security_event import SecurityAuditEvent


CREATE_SECURITY_AUDIT_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS security_audit_events (
    event_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    language TEXT NOT NULL,
    input_risk_level TEXT NOT NULL,
    input_decision TEXT NOT NULL,
    overall_risk REAL NOT NULL,
    risk_analysis_status TEXT NOT NULL,
    risk_breakdown_json TEXT NOT NULL,
    attack_categories_json TEXT NOT NULL,
    matched_signals_json TEXT NOT NULL,
    llm_called INTEGER NOT NULL,
    llm_provider TEXT,
    llm_model TEXT,
    output_safety_checked INTEGER NOT NULL,
    output_safety_decision TEXT,
    output_safety_score REAL,
    output_categories_json TEXT NOT NULL,
    output_matched_signals_json TEXT NOT NULL,
    blocked_reason TEXT,
    request_latency_ms REAL NOT NULL,
    outcome TEXT NOT NULL
)
"""

CREATE_REQUEST_ID_INDEX = """
CREATE INDEX IF NOT EXISTS idx_security_audit_events_request_id
ON security_audit_events(request_id)
"""


class SQLiteAuditStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(CREATE_SECURITY_AUDIT_EVENTS_TABLE)
            connection.execute(CREATE_REQUEST_ID_INDEX)

    def write_event(self, event: SecurityAuditEvent) -> None:
        self.initialize()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO security_audit_events (
                    event_id, request_id, occurred_at, endpoint, language,
                    input_risk_level, input_decision, overall_risk,
                    risk_analysis_status, risk_breakdown_json,
                    attack_categories_json, matched_signals_json, llm_called,
                    llm_provider, llm_model, output_safety_checked,
                    output_safety_decision, output_safety_score,
                    output_categories_json, output_matched_signals_json,
                    blocked_reason, request_latency_ms, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.request_id,
                    event.occurred_at,
                    event.endpoint,
                    event.language,
                    event.input_risk_level,
                    event.input_decision,
                    event.overall_risk,
                    event.risk_analysis_status,
                    json.dumps(event.risk_breakdown, separators=(",", ":")),
                    json.dumps(event.attack_categories, separators=(",", ":")),
                    json.dumps(event.matched_signals, separators=(",", ":")),
                    int(event.llm_called),
                    event.llm_provider,
                    event.llm_model,
                    int(event.output_safety_checked),
                    event.output_safety_decision,
                    event.output_safety_score,
                    json.dumps(event.output_categories, separators=(",", ":")),
                    json.dumps(event.output_matched_signals, separators=(",", ":")),
                    event.blocked_reason,
                    event.request_latency_ms,
                    event.outcome,
                ),
            )
