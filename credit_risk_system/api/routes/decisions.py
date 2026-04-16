from __future__ import annotations

"""GET /decision/{id} and GET /decisions — retrieve persisted decisions."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from api.schemas import DecisionSummary
from database import crud

router = APIRouter()


@router.get("/decision/{decision_id}", response_model=DecisionSummary)
def get_decision(
    decision_id: str,
    db: Session = Depends(get_db_session),
):
    """Retrieve a specific decision by ID."""
    decision = crud.get_decision_by_id(db, decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id!r} not found.")
    return _to_summary(decision)


@router.get("/decisions", response_model=list[DecisionSummary])
def list_decisions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    policy_decision: Optional[str] = Query(default=None, description="APPROVE | DENY | REFER_TO_HUMAN"),
    db: Session = Depends(get_db_session),
):
    """Return a paginated list of decisions, optionally filtered by outcome."""
    records = crud.get_decisions_paginated(db, skip=skip, limit=limit, decision_filter=policy_decision)
    return [_to_summary(r) for r in records]


def _to_summary(d: dict) -> DecisionSummary:
    return DecisionSummary(
        decision_id=d["id"],
        application_id=d["application_id"],
        policy_decision=d["policy_decision"],
        probability_of_default=d["probability_of_default"],
        risk_tier=d["risk_tier"],
        decided_at=datetime.fromisoformat(d["decided_at"]),
    )
