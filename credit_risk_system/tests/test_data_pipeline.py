"""Tests for data loading and preprocessing."""

import pandas as pd
import pytest

from config.feature_config import MODEL_FEATURES
from data.preprocessor import LoanPreprocessor, _parse_emp_length, _parse_pct, _parse_term
from data.synthetic_generator import SyntheticEdgeCaseGenerator


class TestParsers:
    def test_parse_emp_length_10_plus(self):
        assert _parse_emp_length("10+ years") == 10.0

    def test_parse_emp_length_less_than_1(self):
        assert _parse_emp_length("< 1 year") == 0.5

    def test_parse_emp_length_numeric(self):
        assert _parse_emp_length("5 years") == 5.0

    def test_parse_emp_length_nan(self):
        assert _parse_emp_length(None) == 0.0

    def test_parse_pct_with_sign(self):
        assert _parse_pct("54.2%") == pytest.approx(0.542, abs=1e-3)

    def test_parse_pct_decimal(self):
        assert _parse_pct("0.54") == pytest.approx(0.54)

    def test_parse_term(self):
        assert _parse_term("36 months") == 36.0


class TestLoanPreprocessor:
    def test_fit_transform_returns_correct_columns(self, sample_raw_df):
        prep = LoanPreprocessor()
        X = prep.fit_transform(sample_raw_df)
        assert list(X.columns) == MODEL_FEATURES

    def test_transform_shape(self, sample_raw_df):
        prep = LoanPreprocessor()
        X = prep.fit_transform(sample_raw_df)
        assert X.shape[0] == len(sample_raw_df)
        assert X.shape[1] == len(MODEL_FEATURES)

    def test_no_nulls_after_transform(self, sample_raw_df):
        prep = LoanPreprocessor()
        X = prep.fit_transform(sample_raw_df)
        assert X.isnull().sum().sum() == 0

    def test_transform_single_application(self, sample_application):
        prep = LoanPreprocessor()
        # fit on minimal DF first
        mini_df = pd.DataFrame([{
            "fico_range_low": 700, "fico_range_high": 710,
            "dti": 28.0, "annual_inc": 75000, "loan_amnt": 20000,
            "term": "36 months", "installment": 600,
            "grade": "B", "sub_grade": "B1",
            "emp_length": "5 years", "home_ownership": "RENT",
            "verification_status": "Verified",
            "purpose": "debt_consolidation", "delinq_2yrs": 0,
            "open_acc": 6, "pub_rec": 0, "revol_bal": 5000,
            "revol_util": "25.0%", "total_acc": 14,
            "initial_list_status": "f", "int_rate": "10.0%",
            "loan_status": 0,
        }])
        prep.fit(mini_df)
        X = prep.transform_single(sample_application)
        assert X.shape == (1, len(MODEL_FEATURES))

    def test_save_and_load(self, sample_raw_df, tmp_path):
        prep = LoanPreprocessor()
        prep.fit_transform(sample_raw_df)
        path = str(tmp_path / "preprocessor.pkl")
        prep.save(path)

        prep2 = LoanPreprocessor()
        prep2.load(path)
        X2 = prep2.transform(sample_raw_df)
        assert X2.shape[1] == len(MODEL_FEATURES)


class TestSyntheticGenerator:
    def test_borderline_fico_range(self):
        gen = SyntheticEdgeCaseGenerator()
        df = gen.generate_borderline_fico(n=50)
        assert df["fico_score"].between(618, 622).all()
        assert (df["source_type"] == "BORDERLINE_FICO").all()

    def test_high_dti_range(self):
        gen = SyntheticEdgeCaseGenerator()
        df = gen.generate_high_dti_cases(n=50)
        assert df["dti_ratio"].between(40, 46).all()

    def test_protected_pairs_identical_financials(self):
        gen = SyntheticEdgeCaseGenerator()
        df = gen.generate_protected_attribute_pairs(n_per_combo=5)
        assert df["fico_score"].nunique() == 1
        assert df["dti_ratio"].nunique() == 1
        assert df["annual_income"].nunique() == 1
        assert df["gender"].nunique() > 1

    def test_generate_all_has_all_types(self):
        gen = SyntheticEdgeCaseGenerator()
        df = gen.generate_all()
        types = set(df["source_type"].unique())
        assert "BORDERLINE_FICO" in types
        assert "HIGH_DTI" in types
        assert "PROTECTED_PAIR" in types
