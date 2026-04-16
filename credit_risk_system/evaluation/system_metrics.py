from __future__ import annotations

"""
Agent system evaluation metrics.

- Decision Agreement Rate: fraction of cases where agent decision matches policy-only baseline
- Invalid Decision Rate: decisions that violate hard policy rules (Equifax-style metric)
"""

import logging
from typing import Optional

import pandas as pd

from tools.policy_tools import check_lending_policy

logger = logging.getLogger(__name__)


def compute_system_metrics(
    decisions_df: pd.DataFrame,
    applications_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute agent-system-level evaluation metrics.

    Args:
        decisions_df: DataFrame of persisted decisions (from DB or test fixture).
            Expected columns: policy_decision, probability_of_default, violations,
                              fico_score, dti_ratio, annual_income, loan_amount, ltv_ratio
        applications_df: optional raw applications for policy-only re-evaluation

    Returns:
        dict with decision_agreement_rate, invalid_decision_rate, refer_rate, deny_rate, approve_rate
    """
    total = len(decisions_df)
    if total == 0:
        return {"total_decisions": 0}

    # Basic distribution
    counts = decisions_df["policy_decision"].value_counts()
    approve_rate = counts.get("APPROVE", 0) / total
    deny_rate = counts.get("DENY", 0) / total
    refer_rate = counts.get("REFER_TO_HUMAN", 0) / total

    # Invalid decision rate: approved when hard rules should have denied
    invalid_count = _count_invalid_decisions(decisions_df)
    invalid_rate = invalid_count / total

    # Decision agreement rate (agent vs pure policy baseline)
    agreement_rate = None
    if applications_df is not None and len(applications_df) == total:
        agreement_rate = _compute_agreement_rate(decisions_df, applications_df)

    report = {
        "total_decisions": total,
        "approve_rate": round(approve_rate, 4),
        "deny_rate": round(deny_rate, 4),
        "refer_rate": round(refer_rate, 4),
        "invalid_decision_rate": round(invalid_rate, 4),
        "invalid_count": invalid_count,
        "decision_agreement_rate": round(agreement_rate, 4) if agreement_rate is not None else None,
    }
    logger.info("System metrics: %s", report)
    return report


def _count_invalid_decisions(df: pd.DataFrame) -> int:
    """Count APPROVE decisions that violate hard policy rules."""
    approved = df[df["policy_decision"] == "APPROVE"]
    invalid = 0

    for _, row in approved.iterrows():
        fico = row.get("fico_score", 700)
        dti = row.get("dti_ratio", 30)
        pd_prob = row.get("probability_of_default", 0)
        loan = row.get("loan_amount", 15000)
        income = row.get("annual_income", 60000)
        ltv = row.get("ltv_ratio", 0.75)

        # Hard deny checks (same logic as policy_tools)
        if fico < 620 or dti > 43 or ltv > 0.97 or pd_prob > 0.35:
            invalid += 1
        elif income > 0 and (loan / income) > 5.0:
            invalid += 1

    return invalid


def _compute_agreement_rate(decisions_df: pd.DataFrame, applications_df: pd.DataFrame) -> float:
    """Compare agent decisions to pure policy-rule baseline."""
    agree_count = 0
    for i, (_, dec_row) in enumerate(decisions_df.iterrows()):
        if i >= len(applications_df):
            break
        app_row = applications_df.iloc[i]
        baseline = check_lending_policy(
            probability_of_default=dec_row.get("probability_of_default", 0),
            fico_score=int(app_row.get("fico_score", 700)),
            dti_ratio=float(app_row.get("dti_ratio", 30)),
            loan_amount=float(app_row.get("loan_amount", 15000)),
            annual_income=float(app_row.get("annual_income", 60000)),
            ltv_ratio=float(app_row.get("ltv_ratio", 0.75)),
            loan_purpose=str(app_row.get("loan_purpose", "other")),
            state=str(app_row.get("state", "CA")),
        )
        if baseline["policy_decision"] == dec_row["policy_decision"]:
            agree_count += 1

    return agree_count / len(decisions_df)
