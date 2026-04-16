"""
Feature lists, engineering config, and policy thresholds.
These are the ground-truth constants — settings.py can override policy thresholds via env vars.
"""

# Raw LendingClub columns used (before engineering)
RAW_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "loan_status",
]

# Final engineered features fed into XGBoost (NO protected attributes)
MODEL_FEATURES = [
    "fico_mid",
    "dti_clipped",
    "income_log",
    "loan_to_income",
    "installment_to_income",
    "emp_length_years",
    "revolving_util_pct",
    "term_months",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "total_acc",
    "loan_amnt",
    "int_rate",
    "grade_encoded",           # ordinal: A=0 … G=6
    "home_ownership_encoded",  # ordinal: OWN=0, MORTGAGE=1, RENT=2, OTHER=3
    "purpose_encoded",         # ordinal label encoding
    "verification_encoded",    # ordinal
]

# Protected attributes — stored in DB, NEVER in MODEL_FEATURES
PROTECTED_ATTRIBUTES = ["gender", "race", "age"]

# Target column and positive-class label
TARGET_COLUMN = "loan_status"
POSITIVE_LABELS = {"Charged Off", "Default", "Late (31-120 days)"}

# ── Policy thresholds ──────────────────────────────────────────────────────────
POLICY_THRESHOLDS = {
    # Hard deny rules
    "DTI_MAX_PCT": 43.0,
    "FICO_MIN": 620,
    "LTV_MAX_HARD": 0.97,
    "PD_DENY_THRESHOLD": 0.35,
    "MAX_LOAN_TO_INCOME_RATIO": 5.0,      # loan_amount / annual_income ≤ 5x
    # Soft / refer rules
    "LTV_MAX_CONVENTIONAL": 0.80,
    "FICO_PREFERRED": 680,
    "PD_REFER_THRESHOLD": 0.20,
    # Rate tiers by risk tier
    "RATE_TIERS": {
        "LOW": 0.055,
        "MEDIUM": 0.099,
        "HIGH": 0.149,
        "VERY_HIGH": None,  # deny
    },
    # Max loan cap per income tier
    "MAX_LOAN_CAPS": {
        "income_lt_40k": 15_000,
        "income_lt_80k": 35_000,
        "income_lt_150k": 75_000,
        "income_gte_150k": 150_000,
    },
}

# ── Risk tier PD bands ────────────────────────────────────────────────────────
RISK_TIERS = {
    "LOW": (0.0, 0.10),
    "MEDIUM": (0.10, 0.20),
    "HIGH": (0.20, 0.35),
    "VERY_HIGH": (0.35, 1.01),
}

# ── Fairness thresholds ───────────────────────────────────────────────────────
FAIRNESS_THRESHOLDS = {
    "DEMOGRAPHIC_PARITY_MAX_DELTA": 0.10,   # max 10pp gap between groups
    "DISPARATE_IMPACT_MIN_RATIO": 0.80,     # 80% rule (EEOC guideline)
    "EQUALIZED_ODDS_MAX_DELTA": 0.10,
    "MIN_DECISIONS_FOR_FAIRNESS_CHECK": 20, # need at least N decisions per group
}

FEATURE_CONFIG = {
    "raw_columns": RAW_COLUMNS,
    "model_features": MODEL_FEATURES,
    "protected_attributes": PROTECTED_ATTRIBUTES,
    "target_column": TARGET_COLUMN,
    "positive_labels": POSITIVE_LABELS,
    "risk_tiers": RISK_TIERS,
    "fairness_thresholds": FAIRNESS_THRESHOLDS,
}
