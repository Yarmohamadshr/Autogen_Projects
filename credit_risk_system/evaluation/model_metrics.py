"""Batch model evaluation runner."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.evaluator import ModelEvaluator
from models.predictor import CreditRiskPredictor

logger = logging.getLogger(__name__)


def run_model_evaluation(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    output_dir: str = "evaluation/outputs/",
) -> dict:
    """
    Run full model evaluation on a labelled dataset.

    Args:
        df: DataFrame with raw features + target column
        target_col: binary target column (1 = default)
        output_dir: directory to save plots

    Returns:
        dict with AUC, KS, Gini, Precision, Recall, F1
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    predictor = CreditRiskPredictor.get_instance()
    evaluator = ModelEvaluator()

    y_true = df[target_col].values.astype(int)
    result = predictor.predict_batch(df.drop(columns=[target_col], errors="ignore"))
    y_prob = result["probability_of_default"].values

    report = evaluator.generate_report(y_true, y_prob)
    evaluator.plot_roc_curve(y_true, y_prob, str(Path(output_dir) / "roc_curve.png"))

    logger.info("Model evaluation complete: %s", report)
    return report
