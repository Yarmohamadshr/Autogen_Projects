from __future__ import annotations

"""Database CRUD helpers. All functions accept a SQLAlchemy Session."""

import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from database.schema import AuditLog, Decision, FairnessReport, LoanApplication


# ── Applications ──────────────────────────────────────────────────────────────

def insert_application(session: Session, app_data: dict) -> str:
    """Insert a loan application record. Returns the generated UUID."""
    app_id = str(uuid.uuid4())
    record = LoanApplication(
        id=app_id,
        submitted_at=datetime.utcnow(),
        applicant_name=app_data.get("applicant_name"),
        fico_score=app_data["fico_score"],
        dti_ratio=app_data["dti_ratio"],
        annual_income=app_data["annual_income"],
        loan_amount=app_data["loan_amount"],
        loan_term_months=app_data["loan_term_months"],
        loan_purpose=app_data["loan_purpose"],
        employment_length_years=app_data.get("employment_length_years"),
        home_ownership=app_data.get("home_ownership"),
        revolving_util=app_data.get("revolving_util"),
        ltv_ratio=app_data.get("ltv_ratio"),
        state=app_data.get("state"),
        delinq_2yrs=app_data.get("delinq_2yrs"),
        open_accounts=app_data.get("open_accounts"),
        total_accounts=app_data.get("total_accounts"),
        interest_rate=app_data.get("interest_rate"),
        gender=app_data.get("gender"),
        race=app_data.get("race"),
        age=app_data.get("age"),
        raw_json=json.dumps(app_data),
    )
    session.add(record)
    session.commit()
    return app_id


def get_application_by_id(session: Session, app_id: str) -> dict | None:
    record = session.get(LoanApplication, app_id)
    if not record:
        return None
    return _application_to_dict(record)


# ── Decisions ─────────────────────────────────────────────────────────────────

def insert_decision(session: Session, decision_data: dict) -> str:
    """Insert a decision record. Returns the generated UUID."""
    decision_id = str(uuid.uuid4())
    record = Decision(
        id=decision_id,
        application_id=decision_data["application_id"],
        decided_at=datetime.utcnow(),
        policy_decision=decision_data["policy_decision"],
        probability_of_default=decision_data["probability_of_default"],
        risk_tier=decision_data["risk_tier"],
        recommended_rate=decision_data.get("recommended_rate"),
        max_loan_amount=decision_data.get("max_loan_amount"),
        violations=json.dumps(decision_data.get("violations", [])),
        narrative=decision_data["narrative"],
        decision_letter=decision_data.get("decision_letter"),
        model_version=decision_data.get("model_version", "unknown"),
        agent_conversation_log=decision_data.get("agent_conversation_log"),
        human_reviewed=False,
    )
    session.add(record)
    session.commit()
    return decision_id


def get_decision_by_id(session: Session, decision_id: str) -> dict | None:
    record = session.get(Decision, decision_id)
    if not record:
        return None
    return _decision_to_dict(record)


def get_recent_decisions(session: Session, n: int = 100) -> list[dict]:
    records = (
        session.query(Decision)
        .order_by(Decision.decided_at.desc())
        .limit(n)
        .all()
    )
    return [_decision_to_dict(r) for r in records]


def get_decisions_paginated(
    session: Session,
    skip: int = 0,
    limit: int = 20,
    decision_filter: str | None = None,
) -> list[dict]:
    q = session.query(Decision)
    if decision_filter:
        q = q.filter(Decision.policy_decision == decision_filter)
    records = q.order_by(Decision.decided_at.desc()).offset(skip).limit(limit).all()
    return [_decision_to_dict(r) for r in records]


def get_decisions_by_demographic(
    session: Session, attribute: str, value: str
) -> list[dict]:
    """Returns decisions for applications matching a protected attribute value."""
    results = (
        session.query(Decision)
        .join(LoanApplication, Decision.application_id == LoanApplication.id)
        .filter(getattr(LoanApplication, attribute) == value)
        .all()
    )
    return [_decision_to_dict(r) for r in results]


def get_decisions_in_window(session: Session, window_days: int = 30) -> list[dict]:
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    records = session.query(Decision).filter(Decision.decided_at >= cutoff).all()
    return [_decision_to_dict(r) for r in records]


# ── Audit Logs ────────────────────────────────────────────────────────────────

def insert_audit_log(session: Session, audit_data: dict) -> str:
    audit_id = str(uuid.uuid4())
    record = AuditLog(
        id=audit_id,
        decision_id=audit_data["decision_id"],
        audited_at=datetime.utcnow(),
        audit_passed=audit_data["audit_passed"],
        consistency_check=audit_data.get("consistency_check", True),
        demographic_parity_delta=audit_data.get("demographic_parity_delta"),
        equalized_odds_delta=audit_data.get("equalized_odds_delta"),
        disparate_impact_ratio=audit_data.get("disparate_impact_ratio"),
        bias_flags=json.dumps(audit_data.get("bias_flags", [])),
        shap_top_features=json.dumps(audit_data.get("shap_top_features", [])),
        shap_base_value=audit_data.get("shap_base_value"),
        shap_sum=audit_data.get("shap_sum"),
        audit_notes=audit_data.get("audit_notes"),
    )
    session.add(record)
    session.commit()
    return audit_id


def get_audit_log_by_decision(session: Session, decision_id: str) -> dict | None:
    record = session.query(AuditLog).filter(AuditLog.decision_id == decision_id).first()
    if not record:
        return None
    return {
        "id": record.id,
        "decision_id": record.decision_id,
        "audited_at": record.audited_at.isoformat(),
        "audit_passed": record.audit_passed,
        "consistency_check": record.consistency_check,
        "demographic_parity_delta": record.demographic_parity_delta,
        "equalized_odds_delta": record.equalized_odds_delta,
        "disparate_impact_ratio": record.disparate_impact_ratio,
        "bias_flags": json.loads(record.bias_flags) if record.bias_flags else [],
        "shap_top_features": json.loads(record.shap_top_features) if record.shap_top_features else [],
        "shap_base_value": record.shap_base_value,
        "shap_sum": record.shap_sum,
        "audit_notes": record.audit_notes,
    }


# ── Fairness Reports ──────────────────────────────────────────────────────────

def insert_fairness_report(session: Session, report_data: dict) -> str:
    report_id = str(uuid.uuid4())
    record = FairnessReport(
        id=report_id,
        generated_at=datetime.utcnow(),
        window_start=report_data["window_start"],
        window_end=report_data["window_end"],
        total_applications=report_data["total_applications"],
        approval_rate_overall=report_data.get("approval_rate_overall"),
        approval_rate_by_gender=json.dumps(report_data.get("approval_rate_by_gender", {})),
        approval_rate_by_race=json.dumps(report_data.get("approval_rate_by_race", {})),
        approval_rate_by_age_band=json.dumps(report_data.get("approval_rate_by_age_band", {})),
        demographic_parity_gender=report_data.get("demographic_parity_gender"),
        demographic_parity_race=report_data.get("demographic_parity_race"),
        disparate_impact_gender=report_data.get("disparate_impact_gender"),
        disparate_impact_race=report_data.get("disparate_impact_race"),
        flagged_bias_categories=json.dumps(report_data.get("flagged_bias_categories", [])),
        report_json=json.dumps(report_data),
    )
    session.add(record)
    session.commit()
    return report_id


def generate_fairness_report(session: Session, window_days: int = 30) -> dict:
    """Compute approval rates by demographic and return a fairness report dict."""
    decisions = get_decisions_in_window(session, window_days)
    if not decisions:
        return {"total_applications": 0, "message": "No decisions in window"}

    total = len(decisions)
    approved = [d for d in decisions if d["policy_decision"] == "APPROVE"]
    overall_rate = len(approved) / total if total else 0.0

    def approval_rate_by_attr(attr: str) -> dict:
        groups: dict[str, list] = {}
        for d in decisions:
            val = d.get(attr) or "unknown"
            groups.setdefault(val, []).append(d["policy_decision"] == "APPROVE")
        return {k: sum(v) / len(v) for k, v in groups.items() if v}

    by_gender = approval_rate_by_attr("gender")
    by_race = approval_rate_by_attr("race")

    rates_gender = list(by_gender.values())
    rates_race = list(by_race.values())

    dp_gender = max(rates_gender) - min(rates_gender) if len(rates_gender) >= 2 else None
    dp_race = max(rates_race) - min(rates_race) if len(rates_race) >= 2 else None
    di_gender = min(rates_gender) / max(rates_gender) if len(rates_gender) >= 2 and max(rates_gender) > 0 else None
    di_race = min(rates_race) / max(rates_race) if len(rates_race) >= 2 and max(rates_race) > 0 else None

    flags = []
    if dp_gender and dp_gender > 0.10:
        flags.append("DEMOGRAPHIC_PARITY_GENDER_VIOLATION")
    if dp_race and dp_race > 0.10:
        flags.append("DEMOGRAPHIC_PARITY_RACE_VIOLATION")
    if di_gender and di_gender < 0.80:
        flags.append("DISPARATE_IMPACT_GENDER_VIOLATION")
    if di_race and di_race < 0.80:
        flags.append("DISPARATE_IMPACT_RACE_VIOLATION")

    window_end = datetime.utcnow()
    window_start = window_end - timedelta(days=window_days)

    return {
        "window_start": window_start,
        "window_end": window_end,
        "total_applications": total,
        "approval_rate_overall": overall_rate,
        "approval_rate_by_gender": by_gender,
        "approval_rate_by_race": by_race,
        "approval_rate_by_age_band": {},
        "demographic_parity_gender": dp_gender,
        "demographic_parity_race": dp_race,
        "disparate_impact_gender": di_gender,
        "disparate_impact_race": di_race,
        "flagged_bias_categories": flags,
    }


# ── Private serializers ────────────────────────────────────────────────────────

def _application_to_dict(r: LoanApplication) -> dict:
    return {
        "id": r.id,
        "submitted_at": r.submitted_at.isoformat(),
        "applicant_name": r.applicant_name,
        "fico_score": r.fico_score,
        "dti_ratio": r.dti_ratio,
        "annual_income": r.annual_income,
        "loan_amount": r.loan_amount,
        "loan_term_months": r.loan_term_months,
        "loan_purpose": r.loan_purpose,
        "employment_length_years": r.employment_length_years,
        "home_ownership": r.home_ownership,
        "revolving_util": r.revolving_util,
        "ltv_ratio": r.ltv_ratio,
        "state": r.state,
        "gender": r.gender,
        "race": r.race,
        "age": r.age,
    }


def _decision_to_dict(r: Decision) -> dict:
    # Join application protected attrs for fairness lookups
    app = r.application
    return {
        "id": r.id,
        "application_id": r.application_id,
        "decided_at": r.decided_at.isoformat(),
        "policy_decision": r.policy_decision,
        "probability_of_default": r.probability_of_default,
        "risk_tier": r.risk_tier,
        "recommended_rate": r.recommended_rate,
        "max_loan_amount": r.max_loan_amount,
        "violations": json.loads(r.violations) if r.violations else [],
        "narrative": r.narrative,
        "model_version": r.model_version,
        "human_reviewed": r.human_reviewed,
        # Protected attrs from joined application (for fairness reporting)
        "gender": app.gender if app else None,
        "race": app.race if app else None,
        "age": app.age if app else None,
    }
