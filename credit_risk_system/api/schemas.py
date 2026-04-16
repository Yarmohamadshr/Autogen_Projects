"""Pydantic request/response models for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

class LoanApplicationRequest(BaseModel):
    applicant_name: Optional[str] = None
    fico_score: int = Field(..., ge=300, le=850, description="FICO credit score")
    dti_ratio: float = Field(..., ge=0.0, le=100.0, description="Debt-to-income ratio (%)")
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_term_months: int = Field(..., ge=12, le=360, description="Loan term in months")
    loan_purpose: str = Field(..., description="Loan purpose (e.g. debt_consolidation)")
    employment_length_years: float = Field(default=3.0, ge=0)
    home_ownership: str = Field(default="RENT", description="OWN | MORTGAGE | RENT")
    revolving_util: float = Field(default=0.30, ge=0.0, le=1.0, description="Revolving utilisation [0–1]")
    ltv_ratio: float = Field(default=0.75, ge=0.0, description="Loan-to-value ratio")
    state: str = Field(default="CA", description="US state code")
    delinq_2yrs: int = Field(default=0, ge=0, description="Delinquencies in past 2 years")
    open_accounts: int = Field(default=5, ge=0)
    total_accounts: int = Field(default=10, ge=0)
    interest_rate: float = Field(default=0.10, ge=0.0, description="Current/offered interest rate")
    grade: str = Field(default="C", description="LendingClub credit grade (A-G)")
    verification_status: str = Field(default="Verified", description="Not Verified | Source Verified | Verified")

    # Protected attributes — stored for fairness auditing, NOT used in model
    gender: Optional[str] = None
    race: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=18, le=120)


# ── Response ───────────────────────────────────────────────────────────────────

class ShapFeature(BaseModel):
    name: str
    scaled_value: float
    shap_value: float
    direction: str


class DecisionResponse(BaseModel):
    decision_id: str
    application_id: str
    policy_decision: str
    probability_of_default: float
    risk_tier: str
    recommended_rate: Optional[float]
    narrative: str
    top_shap_features: list[ShapFeature]
    violations: list[str]
    audit_passed: bool
    bias_flags: list[str]
    consistency_check: bool
    decided_at: datetime


class DecisionSummary(BaseModel):
    decision_id: str
    application_id: str
    policy_decision: str
    probability_of_default: float
    risk_tier: str
    decided_at: datetime


class AuditLogResponse(BaseModel):
    id: str
    decision_id: str
    audited_at: datetime
    audit_passed: bool
    consistency_check: bool
    demographic_parity_delta: Optional[float]
    equalized_odds_delta: Optional[float]
    disparate_impact_ratio: Optional[float]
    bias_flags: list[str]
    shap_top_features: list
    shap_base_value: Optional[float]
    shap_sum: Optional[float]
    audit_notes: Optional[str]


class AuditReportResponse(BaseModel):
    window_days: int
    total_decisions: int
    approval_rate_overall: Optional[float]
    approval_rate_by_gender: dict
    approval_rate_by_race: dict
    demographic_parity_gender: Optional[float]
    demographic_parity_race: Optional[float]
    disparate_impact_gender: Optional[float]
    disparate_impact_race: Optional[float]
    flagged_bias_categories: list[str]
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
