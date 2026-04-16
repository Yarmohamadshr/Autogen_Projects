"""SQLAlchemy ORM table definitions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    applicant_name: Mapped[Optional[str]] = mapped_column(String)

    # Core financial fields
    fico_score: Mapped[int] = mapped_column(Integer, nullable=False)
    dti_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    annual_income: Mapped[float] = mapped_column(Float, nullable=False)
    loan_amount: Mapped[float] = mapped_column(Float, nullable=False)
    loan_term_months: Mapped[int] = mapped_column(Integer, nullable=False)
    loan_purpose: Mapped[str] = mapped_column(String, nullable=False)
    employment_length_years: Mapped[Optional[float]] = mapped_column(Float)
    home_ownership: Mapped[Optional[str]] = mapped_column(String)
    revolving_util: Mapped[Optional[float]] = mapped_column(Float)
    ltv_ratio: Mapped[Optional[float]] = mapped_column(Float)
    state: Mapped[Optional[str]] = mapped_column(String)
    delinq_2yrs: Mapped[Optional[int]] = mapped_column(Integer)
    open_accounts: Mapped[Optional[int]] = mapped_column(Integer)
    total_accounts: Mapped[Optional[int]] = mapped_column(Integer)
    interest_rate: Mapped[Optional[float]] = mapped_column(Float)

    # Protected attributes (stored for fairness auditing, excluded from model)
    gender: Mapped[Optional[str]] = mapped_column(String)
    race: Mapped[Optional[str]] = mapped_column(String)
    age: Mapped[Optional[int]] = mapped_column(Integer)

    raw_json: Mapped[Optional[str]] = mapped_column(Text)  # full JSON of original request

    decision: Mapped[Optional["Decision"]] = relationship("Decision", back_populates="application", uselist=False)


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    application_id: Mapped[str] = mapped_column(String, ForeignKey("loan_applications.id"), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    policy_decision: Mapped[str] = mapped_column(String, nullable=False)  # APPROVE / DENY / REFER_TO_HUMAN
    probability_of_default: Mapped[float] = mapped_column(Float, nullable=False)
    risk_tier: Mapped[str] = mapped_column(String, nullable=False)
    recommended_rate: Mapped[Optional[float]] = mapped_column(Float)
    max_loan_amount: Mapped[Optional[float]] = mapped_column(Float)

    violations: Mapped[Optional[str]] = mapped_column(Text)           # JSON array string
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    decision_letter: Mapped[Optional[str]] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    agent_conversation_log: Mapped[Optional[str]] = mapped_column(Text)  # JSON

    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    human_reviewer_id: Mapped[Optional[str]] = mapped_column(String)

    application: Mapped["LoanApplication"] = relationship("LoanApplication", back_populates="decision")
    audit_log: Mapped[Optional["AuditLog"]] = relationship("AuditLog", back_populates="decision", uselist=False)

    @property
    def violations_list(self) -> list:
        return json.loads(self.violations) if self.violations else []


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    decision_id: Mapped[str] = mapped_column(String, ForeignKey("decisions.id"), nullable=False)
    audited_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    audit_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    consistency_check: Mapped[bool] = mapped_column(Boolean, nullable=False)

    demographic_parity_delta: Mapped[Optional[float]] = mapped_column(Float)
    equalized_odds_delta: Mapped[Optional[float]] = mapped_column(Float)
    disparate_impact_ratio: Mapped[Optional[float]] = mapped_column(Float)
    bias_flags: Mapped[Optional[str]] = mapped_column(Text)        # JSON array
    shap_top_features: Mapped[Optional[str]] = mapped_column(Text) # JSON array [{name, value, shap_value}]
    shap_base_value: Mapped[Optional[float]] = mapped_column(Float)
    shap_sum: Mapped[Optional[float]] = mapped_column(Float)
    audit_notes: Mapped[Optional[str]] = mapped_column(Text)

    decision: Mapped["Decision"] = relationship("Decision", back_populates="audit_log")


class FairnessReport(Base):
    __tablename__ = "fairness_reports"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    total_applications: Mapped[int] = mapped_column(Integer, nullable=False)
    approval_rate_overall: Mapped[Optional[float]] = mapped_column(Float)
    approval_rate_by_gender: Mapped[Optional[str]] = mapped_column(Text)  # JSON dict
    approval_rate_by_race: Mapped[Optional[str]] = mapped_column(Text)    # JSON dict
    approval_rate_by_age_band: Mapped[Optional[str]] = mapped_column(Text)

    demographic_parity_gender: Mapped[Optional[float]] = mapped_column(Float)
    demographic_parity_race: Mapped[Optional[float]] = mapped_column(Float)
    disparate_impact_gender: Mapped[Optional[float]] = mapped_column(Float)
    disparate_impact_race: Mapped[Optional[float]] = mapped_column(Float)
    flagged_bias_categories: Mapped[Optional[str]] = mapped_column(Text)  # JSON array

    report_json: Mapped[Optional[str]] = mapped_column(Text)  # full serialized report
