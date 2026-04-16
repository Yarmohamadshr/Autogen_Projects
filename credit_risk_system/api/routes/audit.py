"""GET /audit-report and GET /audit-log/{decision_id}."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from api.schemas import AuditLogResponse, AuditReportResponse
from database import crud

router = APIRouter()


@router.get("/audit-report", response_model=AuditReportResponse)
def get_audit_report(
    window_days: int = Query(default=30, ge=1, le=365, description="Lookback window in days"),
    db: Session = Depends(get_db_session),
):
    """Compute and return a fairness report over the specified time window."""
    report = crud.generate_fairness_report(db, window_days=window_days)
    if report.get("total_applications", 0) == 0:
        return AuditReportResponse(
            window_days=window_days,
            total_decisions=0,
            approval_rate_overall=None,
            approval_rate_by_gender={},
            approval_rate_by_race={},
            demographic_parity_gender=None,
            demographic_parity_race=None,
            disparate_impact_gender=None,
            disparate_impact_race=None,
            flagged_bias_categories=[],
            generated_at=datetime.utcnow(),
        )

    return AuditReportResponse(
        window_days=window_days,
        total_decisions=report["total_applications"],
        approval_rate_overall=report.get("approval_rate_overall"),
        approval_rate_by_gender=report.get("approval_rate_by_gender", {}),
        approval_rate_by_race=report.get("approval_rate_by_race", {}),
        demographic_parity_gender=report.get("demographic_parity_gender"),
        demographic_parity_race=report.get("demographic_parity_race"),
        disparate_impact_gender=report.get("disparate_impact_gender"),
        disparate_impact_race=report.get("disparate_impact_race"),
        flagged_bias_categories=report.get("flagged_bias_categories", []),
        generated_at=report.get("window_end", datetime.utcnow()),
    )


@router.get("/audit-log/{decision_id}", response_model=AuditLogResponse)
def get_audit_log(
    decision_id: str,
    db: Session = Depends(get_db_session),
):
    """Return the raw audit record for a specific decision."""
    log = crud.get_audit_log_by_decision(db, decision_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"Audit log for decision {decision_id!r} not found.")

    return AuditLogResponse(
        id=log["id"],
        decision_id=log["decision_id"],
        audited_at=datetime.fromisoformat(log["audited_at"]),
        audit_passed=log["audit_passed"],
        consistency_check=log["consistency_check"],
        demographic_parity_delta=log.get("demographic_parity_delta"),
        equalized_odds_delta=log.get("equalized_odds_delta"),
        disparate_impact_ratio=log.get("disparate_impact_ratio"),
        bias_flags=log.get("bias_flags", []),
        shap_top_features=log.get("shap_top_features", []),
        shap_base_value=log.get("shap_base_value"),
        shap_sum=log.get("shap_sum"),
        audit_notes=log.get("audit_notes"),
    )
