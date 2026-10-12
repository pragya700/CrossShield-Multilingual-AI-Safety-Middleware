import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


class DashboardDatabaseError(Exception):
    """Raised when the dashboard cannot safely read the audit database."""


class AuditReader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        path = Path(self.db_path)

        if not path.exists():
            raise DashboardDatabaseError("Audit database is unavailable.")

        uri = f"file:{path.resolve().as_posix()}?mode=ro"

        try:
            return sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            raise DashboardDatabaseError(
                "Audit database is unavailable."
            ) from exc

    def _fetchall(self, query: str, params=()):
        connection = self._connect()

        try:
            cursor = connection.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as exc:
            raise DashboardDatabaseError(
                "Audit database is unavailable."
            ) from exc
        finally:
            connection.close()

    def summary(self, hours: int) -> dict:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()

        rows = self._fetchall(
            """
            SELECT
                input_risk_level,
                input_decision,
                output_safety_decision,
                language,
                attack_categories_json,
                output_categories_json,
                outcome,
                overall_risk,
                request_latency_ms
            FROM security_audit_events
            WHERE occurred_at >= ?
            """,
            (cutoff,),
        )

        risk_levels = Counter()
        input_decisions = Counter()
        output_decisions = Counter()
        languages = Counter()
        attack_categories = Counter()
        output_categories = Counter()

        total_latency = 0.0
        total_risk = 0.0
        latency_count = 0
        risk_count = 0

        outcomes = Counter()

        for row in rows:
            (
                risk_level,
                input_decision,
                output_decision,
                language,
                attack_json,
                output_json,
                outcome,
                overall_risk,
                latency,
            ) = row

            outcomes[outcome] += 1

            if risk_level:
                risk_levels[risk_level] += 1

            if input_decision:
                input_decisions[input_decision] += 1

            if output_decision:
                output_decisions[output_decision] += 1
            else:
                output_decisions["NOT_CHECKED"] += 1

            if language:
                languages[language] += 1

            for category in self._parse_json_list(attack_json):
                attack_categories[category] += 1

            for category in self._parse_json_list(output_json):
                output_categories[category] += 1

            if overall_risk is not None:
                total_risk += float(overall_risk)
                risk_count += 1

            if latency is not None:
                total_latency += float(latency)
                latency_count += 1

        return {
            "total_requests": len(rows),
            "completed": outcomes.get("completed", 0),
            "input_blocked": outcomes.get("input_blocked", 0),
            "output_blocked": outcomes.get("output_blocked", 0),
            "llm_errors": outcomes.get("llm_error", 0),
            "risk_level_distribution": dict(risk_levels),
            "input_decision_distribution": dict(input_decisions),
            "output_safety_distribution": dict(output_decisions),
            "language_distribution": dict(languages),
            "top_attack_categories": attack_categories.most_common(10),
            "top_output_safety_categories": output_categories.most_common(10),
            "average_latency_ms": (
                total_latency / latency_count
                if latency_count
                else 0.0
            ),
            "average_overall_risk": (
                total_risk / risk_count
                if risk_count
                else 0.0
            ),
        }

    def events(
        self,
        limit: int = 50,
        offset: int = 0,
        outcome: str | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """
        Return recent metadata-only audit events.

        This method is read-only and never returns prompt or LLM content.
        """

        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)

        query = """
            SELECT
                occurred_at,
                request_id,
                outcome,
                language,
                input_risk_level,
                input_decision,
                overall_risk,
                llm_called,
                llm_provider,
                llm_model,
                output_safety_checked,
                output_safety_decision,
                output_safety_score,
                attack_categories_json,
                matched_signals_json,
                output_categories_json,
                output_matched_signals_json,
                blocked_reason,
                request_latency_ms
            FROM security_audit_events
            WHERE 1 = 1
        """

        params = []

        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)

        if language:
            query += " AND language = ?"
            params.append(language)

        query += """
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
        """

        params.extend([limit, offset])

        rows = self._fetchall(query, tuple(params))

        events = []

        for row in rows:
            (
                occurred_at,
                request_id,
                event_outcome,
                event_language,
                input_risk_level,
                input_decision,
                overall_risk,
                llm_called,
                llm_provider,
                llm_model,
                output_safety_checked,
                output_safety_decision,
                output_safety_score,
                attack_categories_json,
                matched_signals_json,
                output_categories_json,
                output_matched_signals_json,
                blocked_reason,
                request_latency_ms,
            ) = row

            events.append(
                {
                    "occurred_at": occurred_at,
                    "request_id": request_id,
                    "outcome": event_outcome,
                    "language": event_language,
                    "input_risk_level": input_risk_level,
                    "input_decision": input_decision,
                    "overall_risk": overall_risk,
                    "llm_called": bool(llm_called),
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "output_safety_checked": bool(output_safety_checked),
                    "output_safety_decision": output_safety_decision,
                    "output_safety_score": output_safety_score,
                    "attack_categories": self._parse_json_list(
                        attack_categories_json
                    ),
                    "matched_signals": self._parse_json_list(
                        matched_signals_json
                    ),
                    "output_categories": self._parse_json_list(
                        output_categories_json
                    ),
                    "output_matched_signals": self._parse_json_list(
                        output_matched_signals_json
                    ),
                    "blocked_reason": blocked_reason,
                    "request_latency_ms": request_latency_ms,
                }
            )

        return events

    def trends(self, hours: int) -> list[dict]:
        """
        Return hourly request trends for the requested time window.

        The dashboard only receives aggregate metadata.
        """

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()

        rows = self._fetchall(
            """
            SELECT
                occurred_at,
                overall_risk,
                outcome
            FROM security_audit_events
            WHERE occurred_at >= ?
            ORDER BY occurred_at ASC
            """,
            (cutoff,),
        )

        buckets = {}

        for occurred_at, overall_risk, outcome in rows:
            try:
                timestamp = datetime.fromisoformat(
                    occurred_at.replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                continue

            timestamp = timestamp.astimezone(timezone.utc)

            bucket_time = timestamp.replace(
                minute=0,
                second=0,
                microsecond=0,
            )

            bucket_key = bucket_time.isoformat()

            if bucket_key not in buckets:
                buckets[bucket_key] = {
                    "request_count": 0,
                    "risk_total": 0.0,
                    "risk_count": 0,
                    "blocked_count": 0,
                }

            bucket = buckets[bucket_key]

            bucket["request_count"] += 1

            if overall_risk is not None:
                bucket["risk_total"] += float(overall_risk)
                bucket["risk_count"] += 1

            if outcome in {"input_blocked", "output_blocked"}:
                bucket["blocked_count"] += 1

        trends = []

        for bucket_key in sorted(buckets):
            bucket = buckets[bucket_key]

            trends.append(
                {
                    "bucket": bucket_key,
                    "request_count": bucket["request_count"],
                    "average_overall_risk": (
                        bucket["risk_total"] / bucket["risk_count"]
                        if bucket["risk_count"]
                        else 0.0
                    ),
                    "blocked_count": bucket["blocked_count"],
                }
            )

        return trends

    @staticmethod
    def _parse_json_list(value):
        if not value:
            return []

        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []

        return parsed if isinstance(parsed, list) else []