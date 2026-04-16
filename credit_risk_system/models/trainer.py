from __future__ import annotations

"""XGBoost training pipeline with cross-validation and model persistence."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostTrainer:
    """
    Trains an XGBoost binary classifier for credit default prediction.

    Usage:
        trainer = XGBoostTrainer()
        model = trainer.train(X_train, y_train, X_val, y_val)
        trainer.save_model(model, "models/artifacts/model.json")
    """

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        # Auto-compute class weight for imbalanced data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        logger.info(
            "Training XGBoost | train=%d | val=%d | scale_pos_weight=%.2f",
            len(X_train), len(X_val), scale_pos_weight,
        )

        model = xgb.XGBClassifier(**self.params, scale_pos_weight=scale_pos_weight)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        logger.info("Validation AUC: %.4f", val_auc)
        return model

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> dict:
        neg, pos = (y == 0).sum(), (y == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {k: v for k, v in self.params.items()
                  if k not in ("early_stopping_rounds",)}
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

        result = {
            "auc_mean": float(np.mean(scores)),
            "auc_std": float(np.std(scores)),
            "auc_folds": scores.tolist(),
            "n_folds": n_folds,
        }
        logger.info("CV AUC: %.4f ± %.4f", result["auc_mean"], result["auc_std"])
        return result

    def save_model(self, model: xgb.XGBClassifier, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(path)
        logger.info("Model saved to %s", path)

    def get_feature_importance(self, model: xgb.XGBClassifier) -> dict:
        scores = model.get_booster().get_fscore()
        total = sum(scores.values()) or 1
        return {k: round(v / total, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
