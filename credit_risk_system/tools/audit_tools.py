from __future__ import annotations

"""Tools callable by the Auditor Agent."""

import json
import logging

from config.feature_config import FAIRNESS_THRESHOLDS

logger = logging.getLogger(__name__)

# Module-level DB session reference — injected at agent startup
_db_session = None


def set_db_session(session) -> None:
    """Inject the SQLAlchemy session. Called by the orchestrator before each evaluation."""
    global _db_session
    _db_session = session


def audit_decision_fairness(
    application_id: str,
    policy_decision: str,
    probability_of_default: float,
    protected_attributes: dict | None = None,
    feature_vector: dict | None = None,
    shap_values: dict | None = None,
    recent_decisions_window: int = 100,
) -> dict:
    """
    Check the decision for demographic bias and consistency.

    Queries recent decisions from the database and computes:
        - Demographic parity delta per group
        - Equalized odds delta
        - Disparate impact ratio (80% rule)
        - Consistency check (similar profiles → same decision)

    Returns:
        audit_passed: bool
        demographic_parity_delta: float | None
        equalized_odds_delta: float | None
        disparate_impact_ratio: float | None
        bias_flags: list[str]
        consistency_check: bool
        audit_notes: str
    """
    from database import crud

    bias_flags = []
    dp_delta = None
    eo_delta = None
    di_ratio = None

    if _db_session is not None:
        recent = crud.get_recent_decisions(_db_session, n=recent_decisions_window)
    else:
        recent = []

    # Demographic parity check
    if recent and len(recent) >= FAIRNESS_THRESHOLDS["MIN_DECISIONS_FOR_FAIRNESS_CHECK"]:
        dp_delta, di_ratio, eo_delta = _compute_fairness_metrics(
            recent, policy_decision, probability_of_default, protected_attributes or {}
        )

        if dp_delta is not None and dp_delta > FAIRNESS_THRESHOLDS["DEMOGRAPHIC_PARITY_MAX_DELTA"]:
            bias_flags.append("DEMOGRAPHIC_PARITY_VIOLATION")
        if di_ratio is not None and di_ratio < FAIRNESS_THRESHOLDS["DISPARATE_IMPACT_MIN_RATIO"]:
            bias_flags.append("DISPARATE_IMPACT_VIOLATION")
        if eo_delta is not None and eo_delta > FAIRNESS_THRESHOLDS["EQUALIZED_ODDS_MAX_DELTA"]:
            bias_flags.append("EQUALIZED_ODDS_VIOLATION")

    # Consistency check
    consistency = _check_consistency(recent, probability_of_default, policy_decision)

    if not consistency:
        bias_flags.append("INCONSISTENT_DECISION")

    audit_passed = len(bias_flags) == 0

    notes = (
        f"Audited against {len(recent)} recent decisions. "
        f"Bias flags: {bias_flags if bias_flags else 'None'}. "
        f"Consistency: {'OK' if consistency else 'INCONSISTENT'}."
    )

    logger.info("Audit result: passed=%s | flags=%s", audit_passed, bias_flags)

    return {
        "audit_passed": audit_passed,
        "demographic_parity_delta": dp_delta,
        "equalized_odds_delta": eo_delta,
        "disparate_impact_ratio": di_ratio,
        "bias_flags": bias_flags,
        "consistency_check": consistency,
        "audit_notes": notes,
    }


def validate_decision_consistency(
    policy_decision: str,
    probability_of_default: float,
    feature_vector: dict | None = None,
    k_neighbors: int = 5,
) -> dict:
    """
    Find k most similar past applications and check decision agreement.

    Similarity is based on PD proximity (simple heuristic).
    Returns consistency ratio and any disagreement flags.
    """
    from database import crud

    if _db_session is None:
        return {"consistency_ratio": 1.0, "consistent": True, "note": "No DB session — skipping."}

    recent = crud.get_recent_decisions(_db_session, n=200)
    if len(recent) < k_neighbors:
        return {"consistency_ratio": 1.0, "consistent": True, "note": "Insufficient history."}

    # Find k nearest by |PD - recent_PD|
    recent_sorted = sorted(recent, key=lambda d: abs(d["probability_of_default"] - probability_of_default))
    neighbors = recent_sorted[:k_neighbors]

    matching = sum(1 for n in neighbors if n["policy_decision"] == policy_decision)
    consistency_ratio = matching / len(neighbors)
    consistent = consistency_ratio >= 0.6  # 60% agreement threshold

    return {
        "consistency_ratio": round(consistency_ratio, 3),
        "consistent": consistent,
        "k_neighbors": k_neighbors,
        "agreement_count": matching,
        "note": f"{matching}/{k_neighbors} similar decisions matched.",
    }


def finalize_decision(
    application_id: str,
    policy_decision: str,
    probability_of_default: float,
    narrative: str,
    audit_result: dict,
    shap_top_features: list,
    risk_tier: str = "MEDIUM",
    violations: list | None = None,
    recommended_rate: float | None = None,
    conversation_log: str | None = None,
) -> dict:
    """
    Persist the complete decision record to the database.
    This MUST be the last tool called by the AuditorAgent.

    Returns:
        decision_id: persisted decision UUID
        audit_log_id: persisted audit log UUID
        status: "persisted"
    """
    from database import crud
    from models.predictor import MODEL_VERSION

    if _db_session is None:
        logger.warning("No DB session — decision not persisted.")
        return {
            "decision_id": "no-db-session",
            "audit_log_id": "no-db-session",
            "status": "not_persisted",
            "reason": "DB session not initialised.",
        }

    decision_data = {
        "application_id": application_id,
        "policy_decision": policy_decision,
        "probability_of_default": probability_of_default,
        "risk_tier": risk_tier,
        "recommended_rate": recommended_rate,
        "violations": violations or [],
        "narrative": narrative,
        "model_version": MODEL_VERSION,
        "agent_conversation_log": conversation_log,
    }
    decision_id = crud.insert_decision(_db_session, decision_data)

    audit_data = {
        "decision_id": decision_id,
        "audit_passed": audit_result.get("audit_passed", True),
        "consistency_check": audit_result.get("consistency_check", True),
        "demographic_parity_delta": audit_result.get("demographic_parity_delta"),
        "equalized_odds_delta": audit_result.get("equalized_odds_delta"),
        "disparate_impact_ratio": audit_result.get("disparate_impact_ratio"),
        "bias_flags": audit_result.get("bias_flags", []),
        "shap_top_features": shap_top_features,
        "audit_notes": audit_result.get("audit_notes"),
    }
    audit_log_id = crud.insert_audit_log(_db_session, audit_data)

    logger.info("Decision persisted: id=%s | decision=%s", decision_id, policy_decision)
    return {
        "decision_id": decision_id,
        "audit_log_id": audit_log_id,
        "status": "persisted",
    }


# ── Fairness computation ───────────────────────────────────────────────────────

def _compute_fairness_metrics(
    recent: list[dict],
    current_decision: str,
    current_pd: float,
    protected_attributes: dict,
) -> tuple[float | None, float | None, float | None]:
    """Compute demographic parity delta and disparate impact ratio."""
    attr = None
    for key in ("race", "gender"):
        if protected_attributes.get(key):
            attr = key
            break

    if attr is None:
        return None, None, None

    groups: dict[str, list[bool]] = {}
    for d in recent:
        val = d.get(attr) or "unknown"
        groups.setdefault(val, []).append(d["policy_decision"] == "APPROVE")

    if len(groups) < 2:
        return None, None, None

    rates = {g: sum(v) / len(v) for g, v in groups.items() if len(v) >= 5}
    if len(rates) < 2:
        return None, None, None

    rate_values = list(rates.values())
    dp_delta = max(rate_values) - min(rate_values)
    di_ratio = min(rate_values) / max(rate_values) if max(rate_values) > 0 else 1.0

    return round(dp_delta, 4), round(di_ratio, 4), None  # EO requires labels — future work


def _check_consistency(
    recent: list[dict],
    probability_of_default: float,
    policy_decision: str,
    pd_tolerance: float = 0.05,
    min_neighbors: int = 3,
) -> bool:
    """Check that similar PD profiles received the same decision in recent history."""
    neighbors = [
        d for d in recent
        if abs(d["probability_of_default"] - probability_of_default) <= pd_tolerance
    ]
    if len(neighbors) < min_neighbors:
        return True  # insufficient data — don't flag

    matching = sum(1 for n in neighbors if n["policy_decision"] == policy_decision)
    return (matching / len(neighbors)) >= 0.6
