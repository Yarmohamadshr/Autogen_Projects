from __future__ import annotations

"""Fairness metrics: demographic parity, equalized odds, disparate impact."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_fairness_metrics(
    decisions_df: pd.DataFrame,
    decision_col: str = "policy_decision",
    label_col: Optional[str] = None,
    protected_attributes: list[str] | None = None,
) -> dict:
    """
    Compute group-level fairness metrics.

    Args:
        decisions_df: DataFrame with decision records (from DB or synthetic).
                      Must contain decision_col and optional protected attribute columns.
        decision_col: column with APPROVE/DENY/REFER_TO_HUMAN
        label_col: ground-truth default label column (if available) for equalized odds
        protected_attributes: list of columns to analyse (default: gender, race)

    Returns:
        dict with per-attribute fairness metrics
    """
    attrs = protected_attributes or ["gender", "race"]
    results = {}

    # Binarise decision: APPROVE = 1, else = 0
    decisions_df = decisions_df.copy()
    decisions_df["approved"] = (decisions_df[decision_col] == "APPROVE").astype(int)

    for attr in attrs:
        if attr not in decisions_df.columns:
            continue
        attr_df = decisions_df.dropna(subset=[attr])
        if len(attr_df) < 10:
            logger.warning("Skipping %s — insufficient data (%d records).", attr, len(attr_df))
            continue

        groups = attr_df.groupby(attr)["approved"]
        approval_rates = groups.mean().to_dict()
        group_counts = groups.count().to_dict()

        rates = list(approval_rates.values())
        dp_delta = max(rates) - min(rates) if len(rates) >= 2 else None
        di_ratio = min(rates) / max(rates) if len(rates) >= 2 and max(rates) > 0 else None

        eo_delta = None
        if label_col and label_col in decisions_df.columns:
            eo_delta = _compute_equalized_odds_delta(attr_df, attr, label_col)

        attr_result = {
            "approval_rates": {k: round(v, 4) for k, v in approval_rates.items()},
            "group_counts": group_counts,
            "demographic_parity_delta": round(dp_delta, 4) if dp_delta is not None else None,
            "disparate_impact_ratio": round(di_ratio, 4) if di_ratio is not None else None,
            "equalized_odds_delta": round(eo_delta, 4) if eo_delta is not None else None,
            "flags": [],
        }

        if dp_delta is not None and dp_delta > 0.10:
            attr_result["flags"].append("DEMOGRAPHIC_PARITY_VIOLATION")
        if di_ratio is not None and di_ratio < 0.80:
            attr_result["flags"].append("DISPARATE_IMPACT_VIOLATION_80PCT_RULE")
        if eo_delta is not None and eo_delta > 0.10:
            attr_result["flags"].append("EQUALIZED_ODDS_VIOLATION")

        results[attr] = attr_result
        logger.info("Fairness [%s]: DP_delta=%.4f | DI_ratio=%s | flags=%s",
                    attr, dp_delta or 0, di_ratio, attr_result["flags"])

    return results


def _compute_equalized_odds_delta(
    df: pd.DataFrame,
    attr_col: str,
    label_col: str,
) -> float:
    """
    Equalized odds: max difference in TPR across groups.
    Requires ground-truth labels.
    """
    tprs = {}
    for group, gdf in df.groupby(attr_col):
        positives = gdf[gdf[label_col] == 1]
        if len(positives) == 0:
            continue
        tpr = positives["approved"].mean()
        tprs[group] = tpr

    if len(tprs) < 2:
        return 0.0
    vals = list(tprs.values())
    return float(max(vals) - min(vals))
