"""
CreditRiskPredictor — singleton that wraps the XGBoost model + SHAP explainer.

The SHAP TreeExplainer is instantiated ONCE at startup (~500ms).
Per-request SHAP computation takes ~10ms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import xgboost as xgb

from config.feature_config import MODEL_FEATURES, RISK_TIERS
from config.settings import settings
from data.preprocessor import LoanPreprocessor

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


class CreditRiskPredictor:
    """Singleton credit risk predictor with integrated SHAP explanations."""

    _instance: ClassVar[CreditRiskPredictor | None] = None

    @classmethod
    def get_instance(cls) -> "CreditRiskPredictor":
        if cls._instance is None:
            artifacts = settings.MODEL_ARTIFACTS_PATH
            model_path = str(Path(artifacts) / "model.json")
            preprocessor_path = str(Path(artifacts) / "preprocessor.pkl")
            cls._instance = cls(model_path, preprocessor_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Force re-load (useful in tests)."""
        cls._instance = None

    def __init__(self, model_path: str, preprocessor_path: str):
        logger.info("Loading model from %s …", model_path)
        self._model = xgb.XGBClassifier()
        self._model.load_model(model_path)

        logger.info("Loading preprocessor from %s …", preprocessor_path)
        self._preprocessor = LoanPreprocessor().load(preprocessor_path)

        self._model_version = MODEL_VERSION
        self._explainer = None
        self._init_shap()

    def _init_shap(self) -> None:
        try:
            import shap
            self._explainer = shap.TreeExplainer(self._model)
            logger.info("SHAP TreeExplainer initialised.")
        except ImportError:
            logger.warning("shap not installed — SHAP explanations will be unavailable.")

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, raw_features: dict) -> dict:
        """
        Predict default probability for a single application.

        Args:
            raw_features: dict with keys matching LoanPreprocessor input columns.

        Returns:
            {probability_of_default, risk_tier, model_version, feature_vector}
        """
        X = self._preprocessor.transform_single(raw_features)
        prob = float(self._model.predict_proba(X)[0, 1])
        tier = _pd_to_tier(prob)

        # feature_vector: scaled values keyed by feature name
        feature_vector = dict(zip(MODEL_FEATURES, X.values[0].tolist()))

        return {
            "probability_of_default": round(prob, 6),
            "risk_tier": tier,
            "model_version": self._model_version,
            "feature_vector": feature_vector,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._preprocessor.transform(df)
        probs = self._model.predict_proba(X)[:, 1]
        df = df.copy()
        df["probability_of_default"] = probs
        df["risk_tier"] = [_pd_to_tier(p) for p in probs]
        return df

    # ── Explanations ───────────────────────────────────────────────────────────

    def explain(self, feature_vector: dict, top_n: int = 5) -> dict:
        """
        Compute SHAP values for a single feature vector (already scaled).

        Args:
            feature_vector: dict {feature_name: scaled_value} from predict()
            top_n: number of top features to return

        Returns:
            {shap_values, base_value, top_features, shap_sum}
        """
        if self._explainer is None:
            return {"shap_values": {}, "base_value": None, "top_features": [], "shap_sum": None}

        X = pd.DataFrame([feature_vector])[MODEL_FEATURES]
        shap_vals = self._explainer.shap_values(X)

        # For binary XGBoost, shap_values shape is (1, n_features)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
        shap_row = shap_vals[0]

        base_value = float(self._explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[-1])

        shap_dict = dict(zip(MODEL_FEATURES, shap_row.tolist()))
        shap_sum = float(sum(shap_row))

        # Top features by absolute SHAP value
        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [
            {
                "name": name,
                "scaled_value": round(feature_vector.get(name, 0), 4),
                "shap_value": round(val, 6),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
            for name, val in sorted_features[:top_n]
        ]

        return {
            "shap_values": {k: round(v, 6) for k, v in shap_dict.items()},
            "base_value": round(base_value, 6),
            "top_features": top_features,
            "shap_sum": round(shap_sum, 6),
        }

    @property
    def model_version(self) -> str:
        return self._model_version


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pd_to_tier(pd_prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= pd_prob < hi:
            return tier
    return "VERY_HIGH"
