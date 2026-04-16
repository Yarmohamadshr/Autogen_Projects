# Data Guide — LendingClub Credit Risk Dataset

This document explains the dataset used in the Multi-Agent Credit Risk Decision System: where it comes from, what it contains, how it is cleaned, and what each feature means and why it was chosen.

---

## The Dataset — LendingClub

LendingClub was a US peer-to-peer lending platform. They published real loan data for millions of loans from 2007–2018. It is one of the most realistic public credit datasets available because:

- It contains **real borrower financials** — income, FICO scores, debt ratios, employment history
- It contains **real outcomes** — you can see which loans were fully paid back and which defaulted
- It mirrors what a credit bureau like Equifax actually sees when a bank requests a credit decision

**Download**: [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
**Place at**: `data/raw/loan.csv`

---

## Dataset Size and Filtering

| Stage | Row count |
|---|---|
| Raw CSV (all statuses) | ~2.2 million loans |
| After filtering to closed loans only | ~900k rows |
| Positive class (default) | ~20% of filtered rows |
| Negative class (fully paid) | ~80% of filtered rows |

**Why filter?** Only loans with a definitive outcome can be used for training. Loans that are still "Current" or "In Grace Period" have unknown outcomes — including them would introduce noise.

**Loans kept** (closed with known outcome):
- `Fully Paid` → label **0** (no default)
- `Charged Off` → label **1** (default)
- `Default` → label **1**
- `Late (31-120 days)` → label **1**

**Loans dropped** (unknown outcome):
- `Current`, `In Grace Period`, `Late (16-30 days)`, `Issued`

---

## Class Imbalance

The dataset is imbalanced at roughly **80% / 20%** (paid / defaulted). This is realistic — most borrowers do repay their loans. The XGBoost model handles this via `scale_pos_weight`, which is automatically computed from the training data as:

```
scale_pos_weight = count(negative class) / count(positive class) ≈ 4.0
```

This tells XGBoost to weight each default example ~4× more heavily so it does not simply predict "no default" for everyone.

---

## Two Layers of Features

The project separates features into two distinct groups:

### Layer 1 — Raw Columns (23 columns from the CSV)

These are pulled directly from the LendingClub CSV. They are messy — strings like `"10+ years"`, percentages stored as `"54.2%"`, FICO stored as a low/high range rather than a single number. The preprocessor cleans and transforms all of these.

```python
RAW_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "loan_status",   ← target variable
]
```

### Layer 2 — Engineered Model Features (18 features fed to XGBoost)

Clean numeric features derived from the raw columns. These are what XGBoost trains on. No protected attributes (gender, race, age) are ever included here.

---

## The 18 Model Features — Full Explanation

### Group 1 — Creditworthiness (how risky is this borrower's history?)

---

#### `fico_mid`
**Raw source**: average of `fico_range_low` and `fico_range_high`

FICO is the most widely used credit score in the United States, ranging from 300 to 850. It is calculated by the Fair Isaac Corporation from your credit report — it captures your payment history, amounts owed, length of credit history, new credit, and credit mix.

- **Higher score = lower risk**
- 300–579: Poor — most lenders will decline
- 580–619: Below average — high-risk tier
- 620–679: Fair — minimum for most conventional loans
- 680–739: Good — preferred tier, better rates
- 740–850: Excellent — lowest rates available

**Why engineered**: LendingClub stores FICO as a range (e.g. 715–719) rather than a single value. We take the midpoint (717) as a single numeric input.

**Example**: `fico_range_low=715, fico_range_high=719` → `fico_mid = 717.0`

---

#### `dti_clipped`
**Raw source**: `dti` (debt-to-income ratio), clipped at 60

DTI is the percentage of your monthly gross income that already goes toward paying existing debts (credit cards, car loans, student loans, etc.) — before this new loan. It is the single most important measure of whether someone can afford a new payment.

- **Lower DTI = lower risk**
- Below 20%: Excellent — plenty of income headroom
- 20–35%: Acceptable
- 36–43%: High — approaching the regulatory limit
- Above 43%: Hard deny — the Consumer Financial Protection Bureau (CFPB) uses 43% as the "qualified mortgage" ceiling

**Why clipped at 60**: a small number of records have extreme DTI values (200%+) which are data entry errors. Clipping at 60 prevents these outliers from distorting the model without removing the rows entirely.

**Example**: borrower earns $5,000/month gross, has $1,400/month in existing debt payments → DTI = 28%

---

#### `revolving_util_pct`
**Raw source**: `revol_util` stored as string `"54.2%"` → parsed to `0.542`

Revolving utilisation is how much of your total available revolving credit (primarily credit cards) you are currently using. It is calculated as: `current_balance / credit_limit` across all revolving accounts.

- **Lower utilisation = lower risk**
- Below 10%: Excellent
- 10–30%: Good
- 30–50%: Fair — starting to signal stress
- Above 70%: Serious red flag — borrower is near their credit limits

**Why it matters**: high utilisation suggests the borrower is relying heavily on credit to cover expenses, increasing the likelihood they cannot absorb a new loan payment.

**Example**: $5,400 balance on a card with a $10,000 limit → `revolving_util_pct = 0.54`

---

#### `delinq_2yrs`
**Raw source**: direct passthrough

The number of times the borrower was 30+ days late on any payment in the past 2 years. Even a single delinquency within 2 years raises default probability significantly, because recent behavior is the strongest predictor of future behavior.

- `0` = clean record — no late payments
- `1` = one late payment in 2 years — moderate risk signal
- `2+` = repeated delinquencies — high risk

**Example**: someone who missed a credit card payment last year → `delinq_2yrs = 1`

---

#### `pub_rec`
**Raw source**: direct passthrough

Number of derogatory public records on the borrower's credit file. These include:
- **Bankruptcies** (Chapter 7 or Chapter 13)
- **Tax liens** (government claim for unpaid taxes)
- **Civil court judgments** (court-ordered debt)

These are the most serious negative marks possible on a credit report and stay on the report for 7–10 years.

- `0` = clean — no public records
- `1+` = significant negative history

---

### Group 2 — Loan Structure (what exactly is being requested?)

---

#### `loan_amnt`
**Raw source**: direct passthrough

The dollar amount the borrower is requesting. Larger loans represent larger potential losses if the borrower defaults, so size directly affects risk.

**Range in LendingClub**: $1,000 to $40,000

---

#### `term_months`
**Raw source**: parsed from `"36 months"` or `"60 months"` → `36.0` or `60.0`

The repayment period. LendingClub only offered two terms:
- **36 months (3 years)**: shorter, higher monthly payment, less total interest
- **60 months (5 years)**: lower monthly payment, more total interest, longer exposure to default

Longer-term loans statistically default more — there is more time for the borrower's situation to deteriorate (job loss, medical emergency, etc.)

---

#### `int_rate`
**Raw source**: parsed from `"13.56%"` → `0.1356`

The annual interest rate assigned to this specific loan by LendingClub's own risk model. LendingClub assigned rates based on their internal assessment of the borrower — riskier borrowers received higher rates.

This feature partially encodes LendingClub's own risk judgment, making it one of the strongest predictors. A borrower assigned a 25% rate was assessed by LendingClub as far riskier than one assigned 7%.

**Range**: roughly 5% to 36% in the dataset

---

#### `grade_encoded`
**Raw source**: `grade` column (`A`, `B`, `C`, `D`, `E`, `F`, `G`) → ordinal encoded as `0, 1, 2, 3, 4, 5, 6`

LendingClub's own letter grade for the borrower. A = best credit quality, G = worst. This is LendingClub's composite credit assessment and is strongly correlated with FICO and interest rate.

| Grade | Encoded | Risk level |
|---|---|---|
| A | 0 | Lowest risk — best borrowers |
| B | 1 | Low risk |
| C | 2 | Moderate risk |
| D | 3 | Elevated risk |
| E | 4 | High risk |
| F | 5 | Very high risk |
| G | 6 | Highest risk |

**Encoding note**: OrdinalEncoder is used (not one-hot) because the grades have a meaningful order — A is strictly better than B, which is better than C.

---

### Group 3 — Repayment Capacity (can the borrower actually afford this?)

---

#### `income_log`
**Raw source**: `log1p(annual_inc)` — log transform of annual income

The borrower's self-reported gross annual income in USD. Log transform is applied because income is heavily right-skewed — a few people earn $500k+ while most earn $40k–$80k. The log makes the distribution approximately normal, which helps XGBoost.

- `log1p(40,000) ≈ 10.60`
- `log1p(75,000) ≈ 11.22`
- `log1p(150,000) ≈ 11.92`

**Example**: `annual_income = $75,000` → `income_log = 11.22`

---

#### `loan_to_income`
**Raw source**: `loan_amnt / annual_inc`

How large the loan is relative to the borrower's annual income. This is a direct measure of whether the loan size is reasonable for their financial situation.

- Ratio of 0.2 = borrowing 20% of annual income — very manageable
- Ratio of 1.0 = borrowing a full year's salary — high stress
- Ratio > 5.0 = hard policy deny (see Policy Rules)

**Example**: $20,000 loan on $75,000 income → `loan_to_income = 0.267`

---

#### `installment_to_income`
**Raw source**: `installment / (annual_inc / 12)`

The monthly loan payment (installment) as a fraction of monthly gross income. This is the most direct measure of month-to-month affordability.

- 0.10 = 10% of monthly income goes to this loan — comfortable
- 0.20 = 20% — manageable but notable
- 0.40 = 40% — unsustainable alongside other expenses

**Example**: $600 monthly payment on $4,000 monthly income → `installment_to_income = 0.15`

---

#### `emp_length_years`
**Raw source**: parsed from messy string → numeric years

How long the borrower has been employed at their current job. Longer employment signals stable, predictable income.

| Raw string | Parsed value |
|---|---|
| `"< 1 year"` | `0.5` |
| `"1 year"` | `1.0` |
| `"5 years"` | `5.0` |
| `"10+ years"` | `10.0` |
| `NaN` (missing) | `0.0` |

**Why engineered**: the raw column is a string category, not a number. Parsing it to a float allows the model to treat it as a continuous variable with meaningful order.

---

### Group 4 — Credit History Breadth (how experienced are they with credit?)

---

#### `open_acc`
**Raw source**: direct passthrough

The number of currently open credit lines — credit cards, auto loans, student loans, personal loans, mortgages, etc. This measures the breadth of the borrower's current credit activity.

- Too few (0–2): thin credit history, less data to assess risk
- Normal range (5–15): typical for an established borrower
- Very many (25+): potential over-extension

---

#### `total_acc`
**Raw source**: direct passthrough

The total number of credit accounts ever opened — including both currently open and already closed accounts. A borrower with `total_acc = 25` and `open_acc = 7` has successfully closed 18 accounts in the past, which is a positive signal of responsible credit management over time.

---

### Group 5 — Categorical Features (encoded for the model)

---

#### `home_ownership_encoded`
**Raw source**: `home_ownership` → ordinal encoded

Whether the borrower owns their home outright, has a mortgage, or rents. Housing stability correlates with financial stability.

| Value | Encoded | Interpretation |
|---|---|---|
| `OWN` | 0 | Owns home outright — most stable, has an asset |
| `MORTGAGE` | 1 | Has a mortgage — stable, building equity |
| `RENT` | 2 | Rents — less stable, no property asset |
| `ANY` / `OTHER` / `NONE` | 3 | Unknown / other |

**Encoding note**: ordered from most stable to least stable, which is meaningful for ordinal encoding.

---

#### `purpose_encoded`
**Raw source**: loan purpose string → ordinal integer

Why the borrower says they need the loan. Different purposes carry different default risk profiles.

| Purpose | Risk profile |
|---|---|
| `credit_card` | Low-medium — consolidating high-interest debt |
| `debt_consolidation` | Low-medium — simplifying existing debt |
| `home_improvement` | Low — asset-backed investment |
| `major_purchase` | Medium |
| `medical` | Medium-high — often unplanned, stressful event |
| `small_business` | High — business income is volatile |
| `vacation` | High — discretionary spending, no return |
| `other` | Variable |

---

#### `verification_encoded`
**Raw source**: `verification_status` → ordinal encoded

Whether LendingClub verified the borrower's stated income.

| Value | Encoded | Meaning |
|---|---|---|
| `Not Verified` | 0 | Took the borrower's word for income |
| `Source Verified` | 1 | Verified the income source (employer) but not the amount |
| `Verified` | 2 | Confirmed income amount via pay stubs or tax documents |

**Interesting nuance**: empirically, `Verified` loans sometimes default *more* than `Not Verified`. This is because LendingClub only bothered verifying borderline applicants — so verified status is a marker of "this person's income seemed suspicious enough to check."

---

## What Is NOT in the Model — and Why

| Attribute | Where it goes | Why excluded from model |
|---|---|---|
| `gender` | Stored in `loan_applications` DB table | ECOA violation — using gender in credit decisions is illegal in the US |
| `race` | Stored in `loan_applications` DB table | ECOA violation — racial discrimination in lending is illegal |
| `age` | Stored in `loan_applications` DB table | ECOA violation — age discrimination in credit is illegal |
| `funded_amnt` | Dropped | Redundant with `loan_amnt` — almost identical |
| `sub_grade` | Dropped | Redundant with `grade` — just a finer subdivision |
| `revol_bal` | Dropped | Absolute balance is less informative than utilisation rate |
| `initial_list_status` | Dropped | Low predictive signal |

The three protected attributes are stored in the database **only** for the AuditorAgent's fairness analysis — to check after the fact whether the model's outcomes are equitable across groups without having used them as inputs.

---

## Preprocessing Pipeline Summary

Every raw feature goes through this pipeline before reaching XGBoost:

```
Raw CSV columns
      │
      ▼
1. Feature Engineering
   ├── fico_mid          = (fico_range_low + fico_range_high) / 2
   ├── dti_clipped       = clip(dti, 0, 60)
   ├── income_log        = log1p(annual_inc)
   ├── loan_to_income    = loan_amnt / annual_inc
   ├── installment_to_income = installment / (annual_inc / 12)
   ├── emp_length_years  = parse("5 years" → 5.0)
   ├── revolving_util_pct = parse("54.2%" → 0.542)
   └── term_months       = parse("36 months" → 36.0)
      │
      ▼
2. Categorical Encoding (OrdinalEncoder — ordered)
   ├── grade_encoded          A→0, B→1, C→2, D→3, E→4, F→5, G→6
   ├── home_ownership_encoded OWN→0, MORTGAGE→1, RENT→2, OTHER→3
   ├── purpose_encoded        credit_card→0, debt_consolidation→1, ...
   └── verification_encoded   Not Verified→0, Source Verified→1, Verified→2
      │
      ▼
3. StandardScaler (zero mean, unit variance)
   Applied to all 18 features so no single feature dominates due to scale
      │
      ▼
18-feature matrix → XGBoost
```

---

## Feature Summary Table

| # | Feature | Raw Source | Engineering | What it captures |
|---|---|---|---|---|
| 1 | `fico_mid` | `fico_range_low/high` | Average of range | Overall credit quality (300–850) |
| 2 | `dti_clipped` | `dti` | Clip at 60 | Existing debt burden as % of income |
| 3 | `revolving_util_pct` | `revol_util` | Parse `"54%"` → `0.54` | Credit card stress level |
| 4 | `delinq_2yrs` | raw | Passthrough | Recent payment failures |
| 5 | `pub_rec` | raw | Passthrough | Bankruptcies and court judgments |
| 6 | `loan_amnt` | raw | Passthrough | Dollar size of loan requested |
| 7 | `term_months` | `term` | Parse `"36 months"` → `36` | Repayment period |
| 8 | `int_rate` | `int_rate` | Parse `"13%"` → `0.13` | Platform's own risk signal |
| 9 | `grade_encoded` | `grade` | Ordinal A→0…G→6 | Platform's letter grade |
| 10 | `income_log` | `annual_inc` | `log1p(income)` | Borrower income (log-scaled) |
| 11 | `loan_to_income` | derived | `loan_amnt / annual_inc` | Loan affordability ratio |
| 12 | `installment_to_income` | derived | `installment / (income/12)` | Monthly payment burden |
| 13 | `emp_length_years` | `emp_length` | Parse `"5 years"` → `5.0` | Income stability / job tenure |
| 14 | `open_acc` | raw | Passthrough | Active credit lines count |
| 15 | `total_acc` | raw | Passthrough | Total credit history breadth |
| 16 | `home_ownership_encoded` | `home_ownership` | Ordinal OWN→0…RENT→2 | Housing / asset stability |
| 17 | `purpose_encoded` | `purpose` | Ordinal by risk level | Reason for borrowing |
| 18 | `verification_encoded` | `verification_status` | Ordinal 0→1→2 | Was income independently confirmed? |
