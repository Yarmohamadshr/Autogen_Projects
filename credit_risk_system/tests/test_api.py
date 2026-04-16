"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from database.connection import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from database import schema  # noqa: F401


@pytest.fixture(scope="function")
def test_app(tmp_path):
    """Create a test FastAPI app with an in-memory DB (StaticPool keeps tables visible)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    app = create_app()

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from api.dependencies import get_db_session, get_predictor
    app.dependency_overrides[get_db_session] = override_get_db

    # Mock the predictor
    mock_pred = MagicMock()
    mock_pred.model_version = "test-1.0"
    mock_pred._instance = mock_pred
    app.dependency_overrides[get_predictor] = lambda: mock_pred

    return app, TestSession


@pytest.fixture
def client(test_app):
    app, _ = test_app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestAuditReport:
    def test_audit_report_empty_db(self, client):
        resp = client.get("/audit-report?window_days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_decisions"] == 0
        assert data["flagged_bias_categories"] == []

    def test_audit_report_window_validation(self, client):
        resp = client.get("/audit-report?window_days=0")
        assert resp.status_code == 422  # Pydantic validation error


class TestDecisionsEndpoint:
    def test_list_decisions_empty(self, client):
        resp = client.get("/decisions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_decision_not_found(self, client):
        resp = client.get("/decision/nonexistent-id")
        assert resp.status_code == 404

    def test_list_decisions_pagination_params(self, client):
        resp = client.get("/decisions?skip=0&limit=5")
        assert resp.status_code == 200
