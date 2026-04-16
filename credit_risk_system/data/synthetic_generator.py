"""Generate synthetic edge-case loan applications for testing and evaluation."""

import numpy as np
import pandas as pd


class SyntheticEdgeCaseGenerator:
    """
    Produces labelled synthetic DataFrames that stress-test the agent system.

    Each record has a `source_type` column indicating its edge-case category.
    All records use the raw feature names expected by LoanPreprocessor.
    """

    def __init__(self, random_seed: int = 42):
        self._rng = np.random.default_rng(random_seed)

    # ── Individual generators ──────────────────────────────────────────────────

    def generate_borderline_fico(self, n: int = 100) -> pd.DataFrame:
        """FICO 618–622: validates hard deny at exactly 619, pass at 620."""
        records = []
        for _ in range(n):
            records.append({
                "fico_score": int(self._rng.integers(618, 623)),
                "dti_ratio": round(float(self._rng.uniform(28, 38)), 1),
                "annual_income": float(self._rng.choice([55_000, 65_000, 75_000])),
                "loan_amount": 15_000.0,
                "loan_term_months": 36,
                "loan_purpose": "debt_consolidation",
                "employment_length_years": 3.0,
                "home_ownership": "RENT",
                "revolving_util": round(float(self._rng.uniform(0.20, 0.50)), 2),
                "ltv_ratio": 0.75,
                "state": "CA",
                "delinq_2yrs": 0,
                "open_accounts": 5,
                "total_accounts": 12,
                "grade": "C",
                "source_type": "BORDERLINE_FICO",
            })
        return pd.DataFrame(records)

    def generate_high_dti_cases(self, n: int = 100) -> pd.DataFrame:
        """DTI 40–46 %: validates hard deny at DTI > 43 %, pass below."""
        records = []
        for _ in range(n):
            records.append({
                "fico_score": int(self._rng.integers(700, 760)),
                "dti_ratio": round(float(self._rng.uniform(40.0, 46.0)), 1),
                "annual_income": 80_000.0,
                "loan_amount": 20_000.0,
                "loan_term_months": 36,
                "loan_purpose": "credit_card",
                "employment_length_years": 5.0,
                "home_ownership": "MORTGAGE",
                "revolving_util": 0.30,
                "ltv_ratio": 0.70,
                "state": "TX",
                "delinq_2yrs": 0,
                "open_accounts": 6,
                "total_accounts": 15,
                "grade": "B",
                "source_type": "HIGH_DTI",
            })
        return pd.DataFrame(records)

    def generate_high_ltv(self, n: int = 50) -> pd.DataFrame:
        """LTV 78–99 %: validates PMI flag and hard deny at LTV > 97 %."""
        records = []
        for _ in range(n):
            records.append({
                "fico_score": 720,
                "dti_ratio": 28.0,
                "annual_income": 90_000.0,
                "loan_amount": 250_000.0,
                "loan_term_months": 360,
                "loan_purpose": "house",
                "employment_length_years": 7.0,
                "home_ownership": "MORTGAGE",
                "revolving_util": 0.20,
                "ltv_ratio": round(float(self._rng.uniform(0.78, 0.99)), 3),
                "state": "FL",
                "delinq_2yrs": 0,
                "open_accounts": 4,
                "total_accounts": 10,
                "grade": "A",
                "source_type": "HIGH_LTV",
            })
        return pd.DataFrame(records)

    def generate_protected_attribute_pairs(self, n_per_combo: int = 20) -> pd.DataFrame:
        """
        Identical financial profiles varying only by gender and race.
        All pairs MUST receive the same decision — used to detect disparate treatment.
        """
        base = {
            "fico_score": 720,
            "dti_ratio": 28.0,
            "annual_income": 75_000.0,
            "loan_amount": 25_000.0,
            "loan_term_months": 36,
            "loan_purpose": "debt_consolidation",
            "employment_length_years": 5.0,
            "home_ownership": "RENT",
            "revolving_util": 0.25,
            "ltv_ratio": 0.75,
            "state": "NY",
            "delinq_2yrs": 0,
            "open_accounts": 6,
            "total_accounts": 14,
            "grade": "B",
        }
        genders = ["Male", "Female", "Non-binary"]
        races = ["White", "Black", "Hispanic", "Asian"]
        records = []
        for gender in genders:
            for race in races:
                for _ in range(n_per_combo):
                    row = base.copy()
                    row["gender"] = gender
                    row["race"] = race
                    row["source_type"] = "PROTECTED_PAIR"
                    records.append(row)
        return pd.DataFrame(records)

    def generate_adversarial(self, n: int = 50) -> pd.DataFrame:
        """Extreme value combinations designed to test model stability."""
        records = [
            # Extremely positive — should APPROVE
            {
                "fico_score": 820, "dti_ratio": 5.0, "annual_income": 999_999.0,
                "loan_amount": 500.0, "loan_term_months": 12,
                "loan_purpose": "credit_card", "employment_length_years": 20.0,
                "home_ownership": "OWN", "revolving_util": 0.02,
                "ltv_ratio": 0.10, "state": "CA", "delinq_2yrs": 0,
                "open_accounts": 10, "total_accounts": 30, "grade": "A",
                "source_type": "ADVERSARIAL_POSITIVE",
            },
            # Income/loan mismatch — should DENY via policy
            {
                "fico_score": 800, "dti_ratio": 10.0, "annual_income": 1.0,
                "loan_amount": 500_000.0, "loan_term_months": 360,
                "loan_purpose": "house", "employment_length_years": 0.0,
                "home_ownership": "RENT", "revolving_util": 0.95,
                "ltv_ratio": 0.99, "state": "NY", "delinq_2yrs": 5,
                "open_accounts": 0, "total_accounts": 1, "grade": "G",
                "source_type": "ADVERSARIAL_NEGATIVE",
            },
        ]
        # Fill remaining with random extremes
        for _ in range(n - 2):
            records.append({
                "fico_score": int(self._rng.integers(300, 850)),
                "dti_ratio": round(float(self._rng.uniform(0.0, 60.0)), 1),
                "annual_income": round(float(self._rng.choice([1, 1_000, 50_000, 500_000])), 0),
                "loan_amount": round(float(self._rng.choice([100, 5_000, 40_000, 200_000])), 0),
                "loan_term_months": int(self._rng.choice([12, 36, 60, 120, 360])),
                "loan_purpose": self._rng.choice(["debt_consolidation", "other", "house"]),
                "employment_length_years": round(float(self._rng.uniform(0, 20)), 0),
                "home_ownership": self._rng.choice(["OWN", "RENT", "MORTGAGE"]),
                "revolving_util": round(float(self._rng.uniform(0.0, 1.0)), 2),
                "ltv_ratio": round(float(self._rng.uniform(0.0, 1.2)), 2),
                "state": "CA",
                "delinq_2yrs": int(self._rng.integers(0, 10)),
                "open_accounts": int(self._rng.integers(0, 30)),
                "total_accounts": int(self._rng.integers(1, 50)),
                "grade": self._rng.choice(["A", "B", "C", "D", "E", "F", "G"]),
                "source_type": "ADVERSARIAL_RANDOM",
            })
        return pd.DataFrame(records)

    def generate_all(self) -> pd.DataFrame:
        """Combine all edge-case sets into one labelled DataFrame."""
        parts = [
            self.generate_borderline_fico(),
            self.generate_high_dti_cases(),
            self.generate_high_ltv(),
            self.generate_protected_attribute_pairs(),
            self.generate_adversarial(),
        ]
        df = pd.concat(parts, ignore_index=True)
        return df
