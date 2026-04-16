"""
SHAP explanation faithfulness evaluation.

Measures how well the ExplanationAgent's top features align with actual SHAP rankings.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_shap_faithfulness(
    shap_top_features_list: list[list[dict]],
    model_shap_top_features_list: list[list[dict]],
    top_n: int = 5,
) -> dict:
    """
    Compute faithfulness score between agent-reported top features and model SHAP values.

    Args:
        shap_top_features_list: list of top_features lists from DB audit_logs
                                [{name, shap_value, direction}, ...]
        model_shap_top_features_list: list of top_features from direct predictor.explain()
        top_n: number of top features to compare

    Returns:
        dict with mean_rank_overlap, mean_sign_agreement, faithfulness_score
    """
    if len(shap_top_features_list) != len(model_shap_top_features_list):
        raise ValueError("Input lists must have equal length.")

    rank_overlaps = []
    sign_agreements = []

    for agent_features, model_features in zip(shap_top_features_list, model_shap_top_features_list):
        agent_names = {f["name"] for f in (agent_features or [])[:top_n]}
        model_names = {f["name"] for f in (model_features or [])[:top_n]}

        if not model_names:
            continue

        overlap = len(agent_names & model_names) / len(model_names)
        rank_overlaps.append(overlap)

        # Sign agreement: directions match for shared features
        agent_dir = {f["name"]: f.get("direction", "") for f in (agent_features or [])}
        model_dir = {f["name"]: f.get("direction", "") for f in (model_features or [])}
        shared = agent_names & model_names
        if shared:
            sign_match = sum(1 for n in shared if agent_dir.get(n) == model_dir.get(n))
            sign_agreements.append(sign_match / len(shared))

    mean_overlap = float(np.mean(rank_overlaps)) if rank_overlaps else 0.0
    mean_sign = float(np.mean(sign_agreements)) if sign_agreements else 0.0
    faithfulness_score = (mean_overlap + mean_sign) / 2.0

    result = {
        "n_evaluated": len(rank_overlaps),
        "mean_rank_overlap": round(mean_overlap, 4),
        "mean_sign_agreement": round(mean_sign, 4),
        "faithfulness_score": round(faithfulness_score, 4),
        "interpretation": _interpret(faithfulness_score),
    }
    logger.info("SHAP faithfulness: %s", result)
    return result


def _interpret(score: float) -> str:
    if score >= 0.90:
        return "Excellent — explanations faithfully reflect model logic."
    elif score >= 0.75:
        return "Good — minor discrepancies, acceptable for production."
    elif score >= 0.60:
        return "Fair — some misalignment, review explanation prompts."
    else:
        return "Poor — explanations diverge significantly from SHAP values."
