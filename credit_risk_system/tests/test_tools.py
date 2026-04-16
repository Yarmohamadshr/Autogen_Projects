"""Tests for agent tool functions (policy and explanation tools are fully testable without a trained model)."""

from unittest.mock import patch

import pytest

from tools.policy_tools import check_lending_policy, get_policy_thresholds


class TestPolicyTools:
    """Policy tools are pure logic — no ML model required."""

    def _base_args(self, **overrides) -> dict:
        base = {
            "probability_of_default": 0.08,
            "fico_score": 720,
            "dti_ratio": 28.0,
            "loan_amount": 20_000.0,
            "annual_income": 75_000.0,
            "ltv_ratio": 0.75,
            "loan_purpose": "debt_consolidation",
            "state": "CA",
        }
        base.update(overrides)
        return base

    def test_approve_clean_application(self):
        result = check_lending_policy(**self._base_args())
        assert result["policy_decision"] == "APPROVE"
        assert result["violations"] == []
        assert result["recommended_rate"] is not None

    def test_deny_fico_below_620(self):
        result = check_lending_policy(**self._base_args(fico_score=610))
        assert result["policy_decision"] == "DENY"
        assert any("FICO" in v for v in result["violations"])

    def test_deny_dti_exceeds_43(self):
        result = check_lending_policy(**self._base_args(dti_ratio=45.0))
        assert result["policy_decision"] == "DENY"
        assert any("DTI" in v for v in result["violations"])

    def test_deny_pd_exceeds_threshold(self):
        result = check_lending_policy(**self._base_args(probability_of_default=0.40))
        assert result["policy_decision"] == "DENY"
        assert any("PD" in v for v in result["violations"])

    def test_deny_ltv_exceeds_97(self):
        result = check_lending_policy(**self._base_args(ltv_ratio=0.98))
        assert result["policy_decision"] == "DENY"
        assert any("LTV" in v for v in result["violations"])

    def test_deny_loan_to_income_exceeds_5x(self):
        result = check_lending_policy(**self._base_args(loan_amount=400_000, annual_income=70_000))
        assert result["policy_decision"] == "DENY"
        assert any("INCOME" in v for v in result["violations"])

    def test_refer_pd_in_grey_zone(self):
        result = check_lending_policy(**self._base_args(probability_of_default=0.25))
        assert result["policy_decision"] == "REFER_TO_HUMAN"
        assert any("REFER" in v for v in result["violations"])

    def test_fico_620_exact_boundary_passes(self):
        result = check_lending_policy(**self._base_args(fico_score=620))
        assert "FICO" not in " ".join(result["violations"])

    def test_fico_619_exact_boundary_fails(self):
        result = check_lending_policy(**self._base_args(fico_score=619))
        assert any("FICO" in v for v in result["violations"])

    def test_dti_43_exact_boundary_passes(self):
        result = check_lending_policy(**self._base_args(dti_ratio=43.0))
        # 43.0 is not > 43, should pass
        assert result["policy_decision"] in ("APPROVE", "REFER_TO_HUMAN")
        assert not any("DTI" in v for v in result["violations"])

    def test_rate_is_float_when_approved(self):
        result = check_lending_policy(**self._base_args())
        assert isinstance(result["recommended_rate"], float)
        assert 0.035 <= result["recommended_rate"] <= 0.36

    def test_max_loan_amount_scales_with_income(self):
        low_income = check_lending_policy(**self._base_args(annual_income=30_000))
        high_income = check_lending_policy(**self._base_args(annual_income=200_000))
        if low_income["max_loan_amount"] and high_income["max_loan_amount"]:
            assert high_income["max_loan_amount"] > low_income["max_loan_amount"]

    def test_get_policy_thresholds_returns_dict(self):
        thresholds = get_policy_thresholds()
        assert "DTI_MAX_PCT" in thresholds
        assert "FICO_MIN" in thresholds
        assert thresholds["FICO_MIN"] == 620


class TestAuditTools:
    """Audit tools — test without DB session."""

    def test_finalize_without_db_session(self):
        from tools.audit_tools import finalize_decision, set_db_session
        set_db_session(None)
        result = finalize_decision(
            application_id="test-id",
            policy_decision="APPROVE",
            probability_of_default=0.08,
            narrative="Test narrative.",
            audit_result={"audit_passed": True, "bias_flags": [], "consistency_check": True},
            shap_top_features=[],
        )
        assert result["status"] == "not_persisted"

    def test_audit_fairness_without_db(self):
        from tools.audit_tools import audit_decision_fairness, set_db_session
        set_db_session(None)
        result = audit_decision_fairness(
            application_id="test-id",
            policy_decision="APPROVE",
            probability_of_default=0.08,
            protected_attributes={"gender": "Female", "race": "Hispanic"},
            feature_vector={"fico_mid": 0.5},
            shap_values={"fico_mid": -0.1},
        )
        assert "audit_passed" in result
        assert isinstance(result["bias_flags"], list)
