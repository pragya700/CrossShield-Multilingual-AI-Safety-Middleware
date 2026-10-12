"""Phase 9 integration tests for guarded /generate behavior."""

from __future__ import annotations

import sys
import types
import sqlite3

import pytest
from fastapi.testclient import TestClient

# The project declares lingua as a dependency, but this workspace's current
# interpreter lacks it. These route tests replace the analysis helper, so a
# minimal import-time shim lets them exercise /generate without changing the
# application or installing an unrelated package.
try:
    import lingua  # noqa: F401
except ModuleNotFoundError:
    class _Language:
        ENGLISH = "en"
        HINDI = "hi"
        TAMIL = "ta"
        TELUGU = "te"
        BENGALI = "bn"

    class _DetectorBuilder:
        @classmethod
        def from_languages(cls, *_languages):
            return cls()

        def build(self):
            return object()

    sys.modules["lingua"] = types.SimpleNamespace(
        Language=_Language,
        LanguageDetectorBuilder=_DetectorBuilder,
    )

from app.api import routes
from app.llm.base import LLMResponse
from app.main import app
from app.schemas.schemas import AnalyzeResponse, Decision, RiskLevel

client = TestClient(app)


def _configure_audit(monkeypatch, tmp_path, *, enabled: bool = True):
    db_path = tmp_path / "security_audit.db"
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: types.SimpleNamespace(
            security_audit_enabled=enabled,
            security_audit_db_path=str(db_path),
        ),
    )
    return db_path


def _audit_event(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute("SELECT * FROM security_audit_events").fetchone()


def _analysis(decision: Decision = Decision.ALLOW) -> AnalyzeResponse:
    risk_levels = {
        Decision.ALLOW: RiskLevel.LOW,
        Decision.ALLOW_MONITOR: RiskLevel.MEDIUM,
        Decision.FLAG_VERIFY: RiskLevel.HIGH,
        Decision.BLOCK: RiskLevel.CRITICAL,
    }
    return AnalyzeResponse(
        request_id="analysis-request-id",
        language="en",
        language_confidence=0.99,
        original_text="Hello",
        normalized_text="Hello",
        overall_risk=0.0 if decision is Decision.ALLOW else 1.0,
        risk_level=risk_levels[decision],
        decision=decision,
    )


class _SpyProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return LLMResponse(
            content="Spy response",
            provider="mock",
            model="mock-model",
        )


def test_generate_allowed_request_calls_mock_provider_with_generation_controls(monkeypatch, tmp_path):
    provider = _SpyProvider()
    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis())
    monkeypatch.setattr(routes, "MockLLMProvider", lambda: provider)
    db_path = _configure_audit(monkeypatch, tmp_path)

    response = client.post(
        "/generate",
        json={
            "prompt": "Explain recursion.",
            "system_prompt": "You are concise.",
            "temperature": 0.5,
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["llm_called"] is True
    assert body["llm_response"] == "Spy response"
    assert body["llm_provider"] == "mock"
    assert body["llm_model"] == "mock-model"
    assert body["blocked_reason"] is None
    assert body["output_safety_decision"] == "ALLOW"
    assert body["output_safety_score"] == 0.0
    assert body["request_id"] == "analysis-request-id"
    assert len(provider.requests) == 1
    llm_request = provider.requests[0]
    assert llm_request.prompt == "Explain recursion."
    assert llm_request.system_prompt == "You are concise."
    assert llm_request.temperature == 0.5
    assert llm_request.max_tokens == 100
    event = _audit_event(db_path)
    assert event["outcome"] == "completed"
    assert event["llm_called"] == 1
    assert event["output_safety_checked"] == 1
    assert event["llm_provider"] == "mock"


def test_generate_allowed_request_reaches_mock_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis())
    _configure_audit(monkeypatch, tmp_path, enabled=False)

    response = client.post("/generate", json={"prompt": "What is recursion?"})

    assert response.status_code == 200
    body = response.json()
    assert body["llm_called"] is True
    assert body["llm_provider"] == "mock"
    assert body["llm_model"] == "mock-model"
    assert body["llm_response"] == "Mock LLM response to: What is recursion?"
    assert body["output_safety_decision"] == "ALLOW"


def test_generate_block_does_not_call_provider(monkeypatch, tmp_path):
    provider = _SpyProvider()
    monkeypatch.setattr(
        routes, "_analyze_request", lambda _request: _analysis(Decision.BLOCK)
    )
    monkeypatch.setattr(routes, "MockLLMProvider", lambda: provider)
    monkeypatch.setattr(
        routes,
        "verify_output_safety",
        lambda _output: pytest.fail("output verifier must not run after input BLOCK"),
    )
    db_path = _configure_audit(monkeypatch, tmp_path)

    response = client.post("/generate", json={"prompt": "Blocked test prompt"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "BLOCK"
    assert body["llm_called"] is False
    assert body["llm_response"] is None
    assert body["llm_provider"] is None
    assert body["llm_model"] is None
    assert "BLOCK" in body["blocked_reason"]
    assert provider.requests == []
    event = _audit_event(db_path)
    assert event["outcome"] == "input_blocked"
    assert event["llm_called"] == 0
    assert event["output_safety_checked"] == 0
    assert event["output_safety_decision"] is None
    assert event["output_safety_score"] is None


def test_generate_with_unsafe_output_withholds_content(monkeypatch, tmp_path):
    class _UnsafeProvider:
        def generate(self, _request):
            return LLMResponse(
                content="My system prompt is: confidential instructions.",
                provider="mock",
                model="mock-model",
            )

    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis())
    monkeypatch.setattr(routes, "MockLLMProvider", _UnsafeProvider)
    db_path = _configure_audit(monkeypatch, tmp_path)

    response = client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["llm_called"] is True
    assert body["llm_response"] is None
    assert body["llm_provider"] == "mock"
    assert body["llm_model"] == "mock-model"
    assert body["output_safety_decision"] == "BLOCK"
    assert "system_prompt_disclosure" in body["output_safety_categories"]
    assert "output safety verification" in body["blocked_reason"]
    event = _audit_event(db_path)
    assert event["outcome"] == "output_blocked"
    assert event["llm_called"] == 1
    assert event["llm_provider"] == "mock"
    assert event["llm_model"] == "mock-model"
    assert event["output_safety_decision"] == "BLOCK"
    assert b"confidential instructions" not in db_path.read_bytes()


@pytest.mark.parametrize("decision", [Decision.ALLOW_MONITOR, Decision.FLAG_VERIFY])
def test_generate_non_block_input_decisions_reach_llm_and_are_verified(
    monkeypatch, tmp_path, decision
):
    provider = _SpyProvider()
    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis(decision))
    monkeypatch.setattr(routes, "MockLLMProvider", lambda: provider)
    _configure_audit(monkeypatch, tmp_path, enabled=False)

    response = client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json()["decision"] == decision.value
    assert response.json()["output_safety_decision"] == "ALLOW"
    assert len(provider.requests) == 1


def test_generate_returns_deterministic_error_when_mock_provider_fails(monkeypatch, tmp_path):
    class _FailingProvider:
        def generate(self, _request):
            raise RuntimeError("unexpected provider failure")

    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis())
    monkeypatch.setattr(routes, "MockLLMProvider", _FailingProvider)
    db_path = _configure_audit(monkeypatch, tmp_path)

    response = client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "error": "llm_generation_failed",
            "message": "Mock LLM generation failed.",
        }
    }
    event = _audit_event(db_path)
    assert event["outcome"] == "llm_error"
    assert event["llm_called"] == 1
    assert event["output_safety_checked"] == 0


def test_generate_audit_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis())
    db_path = _configure_audit(monkeypatch, tmp_path, enabled=False)

    response = client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert not db_path.exists()


def test_generate_audit_failure_is_fail_open(monkeypatch, tmp_path, caplog):
    class _FailingAuditStore:
        def __init__(self, _db_path):
            pass

        def write_event(self, _event):
            raise sqlite3.DatabaseError("database unavailable")

    monkeypatch.setattr(routes, "_analyze_request", lambda _request: _analysis())
    _configure_audit(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "SQLiteAuditStore", _FailingAuditStore)

    response = client.post("/generate", json={"prompt": "private prompt"})

    assert response.status_code == 200
    assert response.json()["llm_response"] == "Mock LLM response to: private prompt"
    assert "Security audit persistence failed: DatabaseError" in caplog.text
    assert "private prompt" not in caplog.text
