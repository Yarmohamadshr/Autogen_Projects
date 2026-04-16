"""POST /evaluate-loan — run the full multi-agent pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from api.schemas import DecisionResponse, LoanApplicationRequest, ShapFeature
from database import crud
from agents.orchestrator import run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/evaluate-loan", response_model=DecisionResponse, status_code=201)
async def evaluate_loan(
    request: LoanApplicationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session),
):
    """
    Submit a loan application for multi-agent credit risk evaluation.

    The pipeline runs 4 agents in sequence:
    1. RiskAnalyst → predicts default probability
    2. PolicyAgent → checks lending rules
    3. ExplanationAgent → generates SHAP narrative
    4. AuditorAgent → audits fairness + persists decision
    """
    app_data = request.model_dump()

    # 1. Persist application
    application_id = crud.insert_application(db, app_data)
    app_data["application_id"] = application_id

    # 2. Run multi-agent evaluation
    try:
        result = run_evaluation(application=app_data, db_session=db)
    except Exception as exc:
        logger.error("Agent pipeline failed for application %s: %s", application_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(exc)}")

    if result.get("parse_error"):
        raise HTTPException(status_code=500, detail="Failed to parse agent decision.")

    # 3. Trigger fairness report generation every 50 decisions (background)
    total = len(crud.get_recent_decisions(db, n=1000))
    if total % 50 == 0 and total > 0:
        background_tasks.add_task(_generate_fairness_report, db)

    # 4. Retrieve persisted decision
    decision_id = result.get("decision_id")
    if not decision_id or decision_id == "no-db-session":
        raise HTTPException(status_code=500, detail="Decision was not persisted.")

    decision = crud.get_decision_by_id(db, decision_id)
    audit = crud.get_audit_log_by_decision(db, decision_id)

    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found after persistence.")

    return _build_response(decision, audit, result)


def _build_response(decision: dict, audit: dict | None, agent_result: dict) -> DecisionResponse:
    # shap_top_features may be a list of dicts or a list of strings (agent summarises to names)
    # When strings are returned, look up real values from the module-level cache in explanation_tools
    import tools.explanation_tools as _expl
    cached = [f for f in _expl._last_shap_features if isinstance(f, dict)]

    raw_features = agent_result.get("top_shap_features") or []

    # If all features are proper dicts, use them directly
    if raw_features and all(isinstance(f, dict) for f in raw_features):
        top_features = [ShapFeature(**f) for f in raw_features]
    elif cached:
        # LLM passed strings (humanized names or raw names) — use the cached full dicts
        # which have real shap_value, scaled_value, and direction
        top_features = [ShapFeature(**f) for f in cached]
    else:
        # Nothing available — zero-fill with whatever names we have
        top_features = [
            ShapFeature(name=str(f), scaled_value=0.0, shap_value=0.0, direction="unknown")
            for f in raw_features
        ]
    return DecisionResponse(
        decision_id=decision["id"],
        application_id=decision["application_id"],
        policy_decision=decision["policy_decision"],
        probability_of_default=decision["probability_of_default"],
        risk_tier=decision["risk_tier"],
        recommended_rate=decision.get("recommended_rate"),
        narrative=decision["narrative"],
        top_shap_features=top_features,
        violations=decision.get("violations") or [],
        audit_passed=audit["audit_passed"] if audit else True,
        bias_flags=audit["bias_flags"] if audit else [],
        consistency_check=audit["consistency_check"] if audit else True,
        decided_at=datetime.fromisoformat(decision["decided_at"]),
    )


async def _generate_fairness_report(db: Session) -> None:
    try:
        report = crud.generate_fairness_report(db, window_days=30)
        crud.insert_fairness_report(db, report)
        logger.info("Background fairness report generated.")
    except Exception as exc:
        logger.error("Background fairness report failed: %s", exc)
