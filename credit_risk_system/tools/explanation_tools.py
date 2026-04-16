from __future__ import annotations

"""Tools callable by the Explanation Agent."""

import logging

logger = logging.getLogger(__name__)

# Module-level cache: stores the full top_features list from the most recent SHAP explanation
# so the API layer can retrieve real shap_value / direction even when the LLM only passes names.
_last_shap_features: list = []


def generate_shap_explanation(
    probability_of_default: float,
    policy_decision: str,
    feature_vector: dict | None = None,
    top_n_features: int = 5,
) -> dict:
    """
    Compute SHAP values for the application and generate a plain-English narrative.

    Args:
        probability_of_default: PD float from risk analyst
        policy_decision: APPROVE | DENY | REFER_TO_HUMAN
        feature_vector: scaled feature dict from predict_default_probability (optional —
            if omitted, the cached vector from the most recent prediction is used)
        top_n_features: number of top SHAP features to include

    Returns:
        narrative: 2-4 sentence plain-English explanation
        top_features: list of {name, scaled_value, shap_value, direction}
        shap_base_value: model base value (expected log-odds)
        shap_sum: sum of SHAP values (reconstructs log-odds offset)
        counterfactual_hints: actionable suggestions if denied/referred
    """
    from models.predictor import CreditRiskPredictor
    from tools.risk_tools import _last_feature_vector

    if feature_vector is None:
        feature_vector = _last_feature_vector

    predictor = CreditRiskPredictor.get_instance()
    shap_result = predictor.explain(feature_vector, top_n=top_n_features)

    top_features = shap_result.get("top_features", [])
    base_value = shap_result.get("base_value")
    shap_sum = shap_result.get("shap_sum")

    # Cache full feature dicts so API layer can recover real values even when LLM passes only names
    global _last_shap_features
    _last_shap_features = top_features

    narrative = _build_narrative(probability_of_default, policy_decision, top_features)
    hints = _build_counterfactual_hints(policy_decision, feature_vector, top_features)

    return {
        "narrative": narrative,
        "top_features": top_features,
        "shap_base_value": base_value,
        "shap_sum": shap_sum,
        "counterfactual_hints": hints,
    }


def format_decision_letter(
    applicant_name: str,
    policy_decision: str,
    narrative: str,
    violations: list,
    recommended_rate: float | None,
) -> str:
    """Generate a formatted approval/adverse-action letter."""
    name = applicant_name or "Applicant"

    if policy_decision == "APPROVE":
        header = "LOAN APPROVAL NOTICE"
        body = (
            f"Dear {name},\n\n"
            f"We are pleased to inform you that your loan application has been approved.\n\n"
            f"{narrative}\n\n"
        )
        if recommended_rate is not None:
            body += f"Your offered Annual Percentage Rate (APR) is {recommended_rate:.2%}.\n\n"
        body += "Please contact us to proceed with the next steps.\n"

    elif policy_decision == "DENY":
        header = "ADVERSE ACTION NOTICE"
        reason_block = "\n".join(f"  - {v}" for v in violations) if violations else "  - Does not meet current lending criteria"
        body = (
            f"Dear {name},\n\n"
            f"We regret to inform you that your loan application has not been approved "
            f"at this time.\n\n"
            f"{narrative}\n\n"
            f"Reasons for this decision:\n{reason_block}\n\n"
            f"You have the right to request the specific reason for this decision within 60 days.\n"
        )

    else:  # REFER_TO_HUMAN
        header = "LOAN APPLICATION — ADDITIONAL REVIEW REQUIRED"
        body = (
            f"Dear {name},\n\n"
            f"Your loan application requires additional review by our lending team.\n\n"
            f"{narrative}\n\n"
            f"A loan officer will contact you within 3-5 business days.\n"
        )

    return f"{'='*60}\n{header}\n{'='*60}\n\n{body}\n{'='*60}"


# ── Narrative builders ────────────────────────────────────────────────────────

def _build_narrative(pd_prob: float, decision: str, top_features: list) -> str:
    pct = f"{pd_prob:.1%}"
    top_names = [_humanise(f["name"]) for f in top_features[:3]]

    if decision == "APPROVE":
        return (
            f"Your application has been reviewed and meets our current lending criteria. "
            f"The estimated probability of default is {pct}, which falls within our acceptable range. "
            f"The most influential factors in this assessment were: {', '.join(top_names)}."
        )
    elif decision == "DENY":
        risk_drivers = [f for f in top_features if f["direction"] == "increases_risk"]
        driver_names = [_humanise(f["name"]) for f in risk_drivers[:3]]
        return (
            f"After a thorough review, your application does not meet our current lending criteria. "
            f"The estimated probability of default is {pct}. "
            f"The primary factors that contributed to this decision include: {', '.join(driver_names) if driver_names else ', '.join(top_names)}."
        )
    else:
        return (
            f"Your application has been flagged for additional review. "
            f"The estimated probability of default is {pct}, which falls in a range that requires "
            f"manual evaluation. Key factors include: {', '.join(top_names)}."
        )


def _build_counterfactual_hints(decision: str, feature_vector: dict, top_features: list) -> list:
    if decision == "APPROVE":
        return []

    hints = []
    feature_names = {f["name"] for f in top_features if f["direction"] == "increases_risk"}

    if "dti_clipped" in feature_names:
        hints.append("Reducing your debt-to-income ratio below 35% may improve your eligibility.")
    if "fico_mid" in feature_names:
        hints.append("Improving your credit score by at least 20 points could significantly reduce your risk tier.")
    if "revolving_util_pct" in feature_names:
        hints.append("Paying down revolving balances to below 30% utilisation is strongly recommended.")
    if "delinq_2yrs" in feature_names:
        hints.append("Maintaining a clean payment record for 12+ months will improve your profile.")
    if "loan_to_income" in feature_names:
        hints.append("Applying for a smaller loan amount relative to your annual income may help qualify.")

    if not hints:
        hints.append("Improving your credit profile over 6–12 months before re-applying is advised.")

    return hints


def _humanise(feature_name: str) -> str:
    mapping = {
        "fico_mid": "credit score",
        "dti_clipped": "debt-to-income ratio",
        "income_log": "annual income",
        "loan_to_income": "loan-to-income ratio",
        "installment_to_income": "monthly payment burden",
        "emp_length_years": "employment length",
        "revolving_util_pct": "revolving utilisation",
        "term_months": "loan term",
        "delinq_2yrs": "recent delinquencies",
        "open_acc": "open accounts",
        "pub_rec": "public records",
        "total_acc": "total accounts",
        "loan_amnt": "loan amount",
        "int_rate": "interest rate",
        "grade_encoded": "credit grade",
        "home_ownership_encoded": "home ownership status",
        "purpose_encoded": "loan purpose",
        "verification_encoded": "income verification",
    }
    return mapping.get(feature_name, feature_name.replace("_", " "))
