"""Feature engineering and preprocessing pipeline for LendingClub data."""

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config.feature_config import MODEL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)

# Ordinal mappings
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
HOME_OWNERSHIP_ORDER = ["OWN", "MORTGAGE", "RENT", "ANY", "OTHER", "NONE"]
PURPOSE_CATEGORIES = [
    "credit_card", "debt_consolidation", "educational", "home_improvement",
    "house", "major_purchase", "medical", "moving", "other", "renewable_energy",
    "small_business", "vacation", "wedding",
]
VERIFICATION_ORDER = ["Not Verified", "Source Verified", "Verified"]


class LoanPreprocessor:
    """
    Fit-transform pipeline that engineers features and scales numerics.

    Steps:
        1. Engineer raw columns → model features
        2. Encode categoricals (ordinal)
        3. Scale numerics (StandardScaler)

    Usage:
        prep = LoanPreprocessor()
        X_train = prep.fit_transform(df_train)
        X_test = prep.transform(df_test)
        prep.save("models/artifacts/preprocessor.pkl")
    """

    def __init__(self):
        self._grade_enc = OrdinalEncoder(
            categories=[GRADE_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self._home_enc = OrdinalEncoder(
            categories=[HOME_OWNERSHIP_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self._purpose_enc = OrdinalEncoder(
            categories=[PURPOSE_CATEGORIES],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self._verification_enc = OrdinalEncoder(
            categories=[VERIFICATION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "LoanPreprocessor":
        X = self._engineer_features(df)
        X = self._encode_categoricals_fit(X)
        self._scaler.fit(X[MODEL_FEATURES].fillna(0))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self._engineer_features(df)
        X = self._encode_categoricals_transform(X)
        X[MODEL_FEATURES] = self._scaler.transform(X[MODEL_FEATURES].fillna(0))
        return X[MODEL_FEATURES]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform_single(self, raw_features: dict) -> pd.DataFrame:
        """Transform a single application dict into a 1-row scaled feature DataFrame."""
        row = pd.DataFrame([raw_features])
        return self.transform(row)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "grade_enc": self._grade_enc,
                "home_enc": self._home_enc,
                "purpose_enc": self._purpose_enc,
                "verification_enc": self._verification_enc,
                "scaler": self._scaler,
                "fitted": self._fitted,
            },
            path,
        )
        logger.info("Preprocessor saved to %s", path)

    def load(self, path: str) -> "LoanPreprocessor":
        state = joblib.load(path)
        self._grade_enc = state["grade_enc"]
        self._home_enc = state["home_enc"]
        self._purpose_enc = state["purpose_enc"]
        self._verification_enc = state["verification_enc"]
        self._scaler = state["scaler"]
        self._fitted = state["fitted"]
        logger.info("Preprocessor loaded from %s", path)
        return self

    # ── Feature engineering ────────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        # FICO midpoint
        if "fico_range_low" in X.columns and "fico_range_high" in X.columns:
            X["fico_mid"] = (X["fico_range_low"] + X["fico_range_high"]) / 2
        elif "fico_score" in X.columns:
            X["fico_mid"] = X["fico_score"]
        else:
            X["fico_mid"] = 680.0

        # DTI — clip extreme outliers
        dti_col = "dti" if "dti" in X.columns else "dti_ratio"
        X["dti_clipped"] = X[dti_col].clip(0, 60) if dti_col in X.columns else 30.0

        # Income log transform
        income_col = "annual_inc" if "annual_inc" in X.columns else "annual_income"
        X["income_log"] = np.log1p(X[income_col]) if income_col in X.columns else np.log1p(60000)

        # Loan-to-income
        loan_col = "loan_amnt" if "loan_amnt" in X.columns else "loan_amount"
        if loan_col in X.columns and income_col in X.columns:
            X["loan_to_income"] = X[loan_col] / (X[income_col].replace(0, np.nan))
        else:
            X["loan_to_income"] = 0.5

        # Installment-to-monthly-income
        if "installment" in X.columns and income_col in X.columns:
            X["installment_to_income"] = X["installment"] / (X[income_col] / 12).replace(0, np.nan)
        else:
            X["installment_to_income"] = 0.1

        # Employment length
        emp_col = "emp_length" if "emp_length" in X.columns else "employment_length_years"
        if "emp_length" in X.columns:
            X["emp_length_years"] = X["emp_length"].apply(_parse_emp_length)
        elif "employment_length_years" in X.columns:
            X["emp_length_years"] = X["employment_length_years"]
        else:
            X["emp_length_years"] = 3.0

        # Revolving utilisation
        if "revol_util" in X.columns:
            X["revolving_util_pct"] = X["revol_util"].apply(_parse_pct)
        elif "revolving_util" in X.columns:
            X["revolving_util_pct"] = X["revolving_util"]
        else:
            X["revolving_util_pct"] = 0.3

        # Term months
        if "term" in X.columns:
            X["term_months"] = X["term"].apply(_parse_term)
        elif "loan_term_months" in X.columns:
            X["term_months"] = X["loan_term_months"]
        else:
            X["term_months"] = 36.0

        # Passthrough numeric columns
        for col, default in [
            ("delinq_2yrs", 0), ("open_acc", 5), ("pub_rec", 0), ("total_acc", 10)
        ]:
            src = col if col in X.columns else col.replace("_acc", "_accounts")
            if src in X.columns:
                X[col] = X[src]
            elif col not in X.columns:
                X[col] = default

        # Loan amount
        X["loan_amnt"] = X[loan_col] if loan_col in X.columns else 15000

        # Interest rate
        if "int_rate" in X.columns:
            X["int_rate"] = X["int_rate"].apply(_parse_pct)
        elif "int_rate" not in X.columns:
            X["int_rate"] = 0.10

        return X

    def _encode_categoricals_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        grade_col = "grade" if "grade" in X.columns else None
        if grade_col:
            self._grade_enc.fit(X[[grade_col]].fillna("G"))
        self._home_enc.fit(X[["home_ownership"]].fillna("OTHER") if "home_ownership" in X.columns else pd.DataFrame({"home_ownership": ["OTHER"]}))
        self._purpose_enc.fit(X[["purpose"]].fillna("other") if "purpose" in X.columns else pd.DataFrame({"purpose": ["other"]}))
        self._verification_enc.fit(X[["verification_status"]].fillna("Not Verified") if "verification_status" in X.columns else pd.DataFrame({"verification_status": ["Not Verified"]}))
        return self._encode_categoricals_transform(X)

    def _encode_categoricals_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "grade" in X.columns:
            X["grade_encoded"] = self._grade_enc.transform(X[["grade"]].fillna("G")).flatten()
        else:
            X["grade_encoded"] = 2.0  # default C

        if "home_ownership" in X.columns:
            X["home_ownership_encoded"] = self._home_enc.transform(X[["home_ownership"]].fillna("RENT")).flatten()
        else:
            X["home_ownership_encoded"] = 2.0

        if "purpose" in X.columns:
            X["purpose_encoded"] = self._purpose_enc.transform(X[["purpose"]].fillna("other")).flatten()
        elif "loan_purpose" in X.columns:
            X["purpose"] = X["loan_purpose"]
            X["purpose_encoded"] = self._purpose_enc.transform(X[["purpose"]].fillna("other")).flatten()
        else:
            X["purpose_encoded"] = 1.0

        if "verification_status" in X.columns:
            X["verification_encoded"] = self._verification_enc.transform(X[["verification_status"]].fillna("Not Verified")).flatten()
        else:
            X["verification_encoded"] = 0.0

        return X


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_emp_length(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if "10+" in s:
        return 10.0
    if "< 1" in s:
        return 0.5
    match = re.search(r"(\d+)", s)
    return float(match.group(1)) if match else 0.0


def _parse_pct(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).replace("%", "").strip()
    try:
        v = float(s)
        return v / 100.0 if v > 1.0 else v
    except ValueError:
        return 0.0


def _parse_term(val) -> float:
    if pd.isna(val):
        return 36.0
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else 36.0
