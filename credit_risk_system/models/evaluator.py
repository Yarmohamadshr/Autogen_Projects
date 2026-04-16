"""Model evaluation metrics: AUC, KS statistic, Gini, Precision/Recall."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute and report standard credit-model performance metrics."""

    def compute_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(roc_auc_score(y_true, y_prob))

    def compute_ks_statistic(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """KS = max(TPR - FPR) across all thresholds."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return float(np.max(tpr - fpr))

    def compute_gini(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return 2.0 * self.compute_auc(y_true, y_prob) - 1.0

    def compute_precision_recall(
        self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
    ) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_precision = float(average_precision_score(y_true, y_prob))
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "average_precision": round(avg_precision, 4),
            "threshold": threshold,
        }

    def generate_report(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        auc_score = self.compute_auc(y_true, y_prob)
        ks = self.compute_ks_statistic(y_true, y_prob)
        gini = self.compute_gini(y_true, y_prob)
        pr = self.compute_precision_recall(y_true, y_prob)

        report = {
            "auc": round(auc_score, 4),
            "ks_statistic": round(ks, 4),
            "gini": round(gini, 4),
            **{k: v for k, v in pr.items()},
            "n_samples": int(len(y_true)),
            "n_positives": int(y_true.sum()),
            "positive_rate": round(float(y_true.mean()), 4),
        }
        logger.info(
            "Model metrics — AUC: %.4f | KS: %.4f | Gini: %.4f | Precision: %.4f | Recall: %.4f",
            report["auc"], report["ks_statistic"], report["gini"],
            report["precision"], report["recall"],
        )
        return report

    def plot_roc_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, output_path: str
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping ROC plot.")
            return

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = self.compute_auc(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Credit Default Model")
        ax.legend(loc="lower right")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curve saved to %s", output_path)
