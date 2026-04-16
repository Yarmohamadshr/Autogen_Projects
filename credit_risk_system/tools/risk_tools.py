"""Tools callable by the Risk Analyst Agent."""

import logging

logger = logging.getLogger(__name__)

# Module-level cache: stores the feature_vector from the most recent prediction
# so ExplanationAgent can retrieve it without having to pass it through the LLM.
_last_feature_vector: dict = {}


def predict_default_probability(
    fico_score: int,
    dti_ratio: float,
    annual_income: float,
    loan_amount: float,
    loan_term_months: int,
    employment_length_years: float,
    home_ownership: str,
    loan_purpose: str,
    revolving_util: float,
    delinq_2yrs: int,
    open_accounts: int,
    total_accounts: int,
    ltv_ratio: float = 0.75,
    state: str = "CA",
    interest_rate: float = 0.10,
    grade: str = "C",
    verification_status: str = "Verified",
) -> dict:
    """
    Predict the probability of loan default using the trained XGBoost model.

    Returns:
        probability_of_default: float [0, 1]
        risk_tier: LOW | MEDIUM | HIGH | VERY_HIGH
        model_version: str
        feature_vector: dict of scaled feature values used by the model
    """
    from models.predictor import CreditRiskPredictor

    raw_features = {
        "fico_score": fico_score,
        "dti_ratio": dti_ratio,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "employment_length_years": employment_length_years,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "revolving_util": revolving_util,
        "delinq_2yrs": delinq_2yrs,
        "open_accounts": open_accounts,
        "total_accounts": total_accounts,
        "ltv_ratio": ltv_ratio,
        "state": state,
        "int_rate": interest_rate,
        "grade": grade,
        "verification_status": verification_status,
    }

    try:
        predictor = CreditRiskPredictor.get_instance()
        result = predictor.predict(raw_features)
        # Cache feature_vector so ExplanationAgent can retrieve it without LLM passing it
        global _last_feature_vector
        _last_feature_vector = result.get("feature_vector", {})
        logger.info(
            "Risk prediction: PD=%.4f | Tier=%s",
            result["probability_of_default"], result["risk_tier"],
        )
        return result
    except Exception as exc:
        logger.error("predict_default_probability failed: %s", exc)
        raise


def get_model_metadata() -> dict:
    """Return metadata about the currently loaded credit risk model."""
    from models.predictor import CreditRiskPredictor
    predictor = CreditRiskPredictor.get_instance()
    return {
        "model_version": predictor.model_version,
        "model_type": "XGBoostClassifier",
        "target": "probability_of_default",
        "positive_class": "Default / Charged Off",
    }
