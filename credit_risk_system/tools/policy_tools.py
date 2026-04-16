"""Tools callable by the Policy Agent."""

import logging

from config.feature_config import POLICY_THRESHOLDS, RISK_TIERS
from config.settings import settings

logger = logging.getLogger(__name__)


def check_lending_policy(
    probability_of_default: float,
    fico_score: int,
    dti_ratio: float,
    loan_amount: float,
    annual_income: float,
    ltv_ratio: float,
    loan_purpose: str,
    state: str = "CA",
) -> dict:
    """
    Evaluate a loan application against lending policy rules.

    Hard deny rules (any violation → DENY):
        - FICO < 620
        - DTI > 43%
        - LTV > 97%
        - PD > 0.35
        - loan_amount / annual_income > 5x

    Soft rules (→ REFER_TO_HUMAN):
        - PD between 0.20–0.35
        - LTV between 80–97% (requires PMI note)

    Returns:
        policy_decision: APPROVE | DENY | REFER_TO_HUMAN
        violations: list of violated rule names
        rule_scores: pass/fail per rule
        recommended_rate: suggested APR if approved (None if denied)
        max_loan_amount: policy cap given income
    """
    thresholds = _get_effective_thresholds()
    violations = []
    rule_scores = {}
    soft_flags = []

    # ── Hard deny rules ────────────────────────────────────────────────────────
    if fico_score < thresholds["FICO_MIN"]:
        violations.append(f"FICO_BELOW_{thresholds['FICO_MIN']}")
        rule_scores["FICO_MIN"] = "FAIL"
    else:
        rule_scores["FICO_MIN"] = "PASS"

    if dti_ratio > thresholds["DTI_MAX_PCT"]:
        violations.append(f"DTI_EXCEEDS_{thresholds['DTI_MAX_PCT']}PCT")
        rule_scores["DTI_MAX"] = "FAIL"
    else:
        rule_scores["DTI_MAX"] = "PASS"

    if ltv_ratio > thresholds["LTV_MAX_HARD"]:
        violations.append(f"LTV_EXCEEDS_{int(thresholds['LTV_MAX_HARD']*100)}PCT")
        rule_scores["LTV_MAX"] = "FAIL"
    else:
        rule_scores["LTV_MAX"] = "PASS"

    if probability_of_default > thresholds["PD_DENY_THRESHOLD"]:
        violations.append(f"PD_EXCEEDS_{thresholds['PD_DENY_THRESHOLD']}")
        rule_scores["PD_DENY"] = "FAIL"
    else:
        rule_scores["PD_DENY"] = "PASS"

    income_ratio = loan_amount / annual_income if annual_income > 0 else float("inf")
    if income_ratio > thresholds["MAX_LOAN_TO_INCOME_RATIO"]:
        violations.append("LOAN_TO_INCOME_EXCEEDS_5X")
        rule_scores["LOAN_TO_INCOME"] = "FAIL"
    else:
        rule_scores["LOAN_TO_INCOME"] = "PASS"

    # ── Soft / refer rules ─────────────────────────────────────────────────────
    if thresholds["PD_REFER_THRESHOLD"] < probability_of_default <= thresholds["PD_DENY_THRESHOLD"]:
        soft_flags.append("PD_IN_REFER_ZONE")
        rule_scores["PD_REFER"] = "SOFT_FAIL"
    else:
        rule_scores["PD_REFER"] = "PASS"

    if thresholds["LTV_MAX_CONVENTIONAL"] < ltv_ratio <= thresholds["LTV_MAX_HARD"]:
        soft_flags.append("LTV_REQUIRES_PMI")
        rule_scores["LTV_PMI"] = "SOFT_FAIL"
    else:
        rule_scores["LTV_PMI"] = "PASS"

    # ── Decision ───────────────────────────────────────────────────────────────
    if violations:
        decision = "DENY"
        recommended_rate = None
    elif soft_flags:
        decision = "REFER_TO_HUMAN"
        recommended_rate = _compute_rate(probability_of_default, fico_score)
    else:
        decision = "APPROVE"
        recommended_rate = _compute_rate(probability_of_default, fico_score)

    max_loan = _compute_max_loan(annual_income, thresholds)

    logger.info(
        "Policy decision: %s | violations=%s | soft=%s",
        decision, violations, soft_flags,
    )

    return {
        "policy_decision": decision,
        "violations": violations + soft_flags,
        "rule_scores": rule_scores,
        "recommended_rate": recommended_rate,
        "max_loan_amount": max_loan,
    }


def get_policy_thresholds() -> dict:
    """Return the current effective policy thresholds (env overrides + defaults)."""
    return _get_effective_thresholds()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_effective_thresholds() -> dict:
    """Merge feature_config defaults with settings.py env overrides."""
    t = dict(POLICY_THRESHOLDS)
    t["DTI_MAX_PCT"] = settings.DTI_MAX_PCT
    t["FICO_MIN"] = settings.FICO_MIN
    t["LTV_MAX_HARD"] = settings.LTV_MAX_HARD
    t["PD_DENY_THRESHOLD"] = settings.PD_DENY_THRESHOLD
    t["PD_REFER_THRESHOLD"] = settings.PD_REFER_THRESHOLD
    return t


def _compute_rate(pd_prob: float, fico_score: int) -> float:
    """Simple risk-based pricing: base rate + PD spread + FICO adjustment."""
    base_rate = 0.05
    pd_spread = pd_prob * 0.30       # up to 30pp for high-risk
    fico_discount = max(0.0, (fico_score - 620) / 230 * 0.03)  # up to 3pp for excellent credit
    rate = base_rate + pd_spread - fico_discount
    return round(min(max(rate, 0.035), 0.36), 4)  # clamp to 3.5%–36%


def _compute_max_loan(annual_income: float, thresholds: dict) -> float:
    caps = thresholds.get("MAX_LOAN_CAPS", {})
    if annual_income < 40_000:
        return caps.get("income_lt_40k", 15_000)
    elif annual_income < 80_000:
        return caps.get("income_lt_80k", 35_000)
    elif annual_income < 150_000:
        return caps.get("income_lt_150k", 75_000)
    else:
        return caps.get("income_gte_150k", 150_000)
