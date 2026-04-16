"""Tests for ModelEvaluator metrics (no trained model needed)."""

import numpy as np
import pytest

from models.evaluator import ModelEvaluator


class TestModelEvaluator:
    def _perfect_data(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        return y_true, y_prob

    def _random_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, n)
        y_prob = rng.uniform(0, 1, n)
        return y_true, y_prob

    def test_auc_perfect_classifier(self):
        evaluator = ModelEvaluator()
        y_true, y_prob = self._perfect_data()
        assert evaluator.compute_auc(y_true, y_prob) == pytest.approx(1.0)

    def test_auc_random(self):
        evaluator = ModelEvaluator()
        y_true, y_prob = self._random_data()
        auc = evaluator.compute_auc(y_true, y_prob)
        assert 0.3 < auc < 0.7  # roughly chance for random data

    def test_ks_statistic_perfect(self):
        evaluator = ModelEvaluator()
        y_true, y_prob = self._perfect_data()
        ks = evaluator.compute_ks_statistic(y_true, y_prob)
        assert ks == pytest.approx(1.0)

    def test_gini_equals_2_auc_minus_1(self):
        evaluator = ModelEvaluator()
        y_true, y_prob = self._random_data()
        auc = evaluator.compute_auc(y_true, y_prob)
        gini = evaluator.compute_gini(y_true, y_prob)
        assert gini == pytest.approx(2 * auc - 1, abs=1e-6)

    def test_precision_recall_returns_keys(self):
        evaluator = ModelEvaluator()
        y_true, y_prob = self._random_data()
        result = evaluator.compute_precision_recall(y_true, y_prob)
        assert {"precision", "recall", "f1", "threshold"}.issubset(result.keys())

    def test_generate_report_has_all_fields(self):
        evaluator = ModelEvaluator()
        y_true, y_prob = self._random_data()
        report = evaluator.generate_report(y_true, y_prob)
        assert "auc" in report
        assert "ks_statistic" in report
        assert "gini" in report
        assert "precision" in report
        assert "recall" in report
        assert report["n_samples"] == 200
