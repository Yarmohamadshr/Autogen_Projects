"""Shared pytest fixtures."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.connection import Base
from database import schema  # noqa: F401 — registers ORM models


# ── Database fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_engine():
    """In-memory SQLite engine per test (StaticPool so all sessions share one connection)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Fresh SQLAlchemy session per test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


# ── Sample data fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_application() -> dict:
    return {
        "applicant_name": "Jane Doe",
        "fico_score": 720,
        "dti_ratio": 28.0,
        "annual_income": 75_000.0,
        "loan_amount": 20_000.0,
        "loan_term_months": 36,
        "loan_purpose": "debt_consolidation",
        "employment_length_years": 5.0,
        "home_ownership": "RENT",
        "revolving_util": 0.25,
        "ltv_ratio": 0.75,
        "state": "CA",
        "delinq_2yrs": 0,
        "open_accounts": 6,
        "total_accounts": 14,
        "interest_rate": 0.10,
        "grade": "B",
        "gender": "Female",
        "race": "Hispanic",
        "age": 32,
    }


@pytest.fixture
def deny_application() -> dict:
    """Application that should trigger hard deny (FICO < 620)."""
    return {
        "applicant_name": "John Smith",
        "fico_score": 580,
        "dti_ratio": 50.0,
        "annual_income": 30_000.0,
        "loan_amount": 25_000.0,
        "loan_term_months": 60,
        "loan_purpose": "other",
        "employment_length_years": 0.5,
        "home_ownership": "RENT",
        "revolving_util": 0.90,
        "ltv_ratio": 0.95,
        "state": "NY",
        "delinq_2yrs": 3,
        "open_accounts": 2,
        "total_accounts": 5,
        "interest_rate": 0.24,
        "grade": "F",
        "gender": None,
        "race": None,
        "age": None,
    }


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Small synthetic DataFrame mimicking LendingClub structure."""
    n = 50
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "fico_range_low": rng.integers(600, 800, n),
        "fico_range_high": rng.integers(605, 805, n),
        "dti": rng.uniform(10, 40, n),
        "annual_inc": rng.uniform(30_000, 150_000, n),
        "loan_amnt": rng.uniform(5_000, 40_000, n),
        "term": ["36 months"] * n,
        "installment": rng.uniform(100, 1200, n),
        "grade": rng.choice(["A", "B", "C", "D"], n),
        "sub_grade": ["B1"] * n,
        "emp_length": rng.choice(["1 year", "5 years", "10+ years", "< 1 year"], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
        "delinq_2yrs": rng.integers(0, 3, n),
        "open_acc": rng.integers(2, 15, n),
        "pub_rec": rng.integers(0, 2, n),
        "revol_bal": rng.uniform(1_000, 30_000, n),
        "revol_util": [f"{v:.1f}%" for v in rng.uniform(5, 80, n)],
        "total_acc": rng.integers(5, 30, n),
        "initial_list_status": ["f"] * n,
        "int_rate": [f"{v:.2f}%" for v in rng.uniform(5, 25, n)],
        "loan_status": rng.choice(["Fully Paid", "Charged Off"], n, p=[0.8, 0.2]),
    })


# ── Mock predictor fixture ─────────────────────────────────────────────────────

@pytest.fixture
def mock_predictor():
    """Mock CreditRiskPredictor that returns deterministic outputs."""
    mock = MagicMock()
    mock.model_version = "test-1.0"
    mock.predict.return_value = {
        "probability_of_default": 0.08,
        "risk_tier": "LOW",
        "model_version": "test-1.0",
        "feature_vector": {"fico_mid": 0.5, "dti_clipped": -0.3},
    }
    mock.explain.return_value = {
        "shap_values": {"fico_mid": -0.15, "dti_clipped": 0.05},
        "base_value": 0.12,
        "top_features": [
            {"name": "fico_mid", "scaled_value": 0.5, "shap_value": -0.15, "direction": "decreases_risk"},
            {"name": "dti_clipped", "scaled_value": -0.3, "shap_value": 0.05, "direction": "increases_risk"},
        ],
        "shap_sum": -0.10,
    }
    return mock
