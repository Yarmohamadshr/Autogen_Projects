# Testing and API Guide

This document explains two parts of the project that may be new to you:

1. **The `tests/` folder** — what automated tests are, why we write them, and what each test file does
2. **The `api/` folder** — what FastAPI is, how it works, and how to read the JSON responses

Both topics are explained from first principles, with real examples from this project.

---

## Part 1 — The `tests/` Folder

### What is a test?

A test is a small piece of code that checks whether another piece of code behaves correctly.

Instead of manually running the system and checking the output yourself every time you make a change, you write a test once. Then whenever you change something, you run all tests in one command and they tell you instantly if anything broke.

**The idea is simple:**
```
You write code → Tests check it automatically → You know immediately if something broke
```

Think of it like a checklist that runs itself.

---

### Why are tests important for a portfolio project?

For a data scientist, having tests signals to an interviewer:
- You understand that code can break silently — especially ML code where wrong answers look like right answers
- You know how to verify that your policy rules actually work (FICO 619 → DENY, FICO 620 → APPROVE)
- You know how to separate what the model does from what the system does

Without tests, a hiring manager might ask: "How do you know the FICO cutoff is applied correctly?" With tests, you show them.

---

### How to run the tests

```bash
cd credit_risk_system
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run one file at a time
pytest tests/test_tools.py -v
pytest tests/test_model.py -v
pytest tests/test_data_pipeline.py -v
pytest tests/test_api.py -v
```

The `-v` flag means "verbose" — it prints each test name and whether it passed or failed.

**You do NOT need the trained model or an OpenAI key to run any of these tests.** All tests mock (simulate) the parts they don't need to test.

---

### What does test output look like?

```
tests/test_tools.py::TestPolicyTools::test_approve_clean_application  PASSED
tests/test_tools.py::TestPolicyTools::test_deny_fico_below_620        PASSED
tests/test_tools.py::TestPolicyTools::test_deny_dti_exceeds_43        PASSED
tests/test_tools.py::TestPolicyTools::test_fico_620_exact_boundary_passes PASSED
tests/test_tools.py::TestPolicyTools::test_fico_619_exact_boundary_fails  PASSED
...
43 passed in 4.2s
```

If something fails it looks like:
```
FAILED tests/test_tools.py::TestPolicyTools::test_deny_fico_below_620
  AssertionError: assert 'APPROVE' == 'DENY'
```

This tells you exactly which test failed and what it expected vs what it got.

---

### The test files — one by one

---

#### `tests/conftest.py` — The Setup File

**What it is**: a special pytest file that creates shared building blocks (called **fixtures**) that all other test files can use. It runs automatically before tests — you never call it directly.

**Think of it like**: a kitchen that prepares ingredients before cooking begins.

**What it provides:**

**1. An in-memory database (`db_session`)**
```python
engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
```
Instead of writing to your real `credit_risk.db` file on disk, tests use a temporary database that lives only in RAM and gets deleted after each test. This means:
- Tests don't pollute your real data
- Tests can run in any order without interfering with each other
- No cleanup needed after running tests

**2. Sample loan applications**
```python
sample_application = {
    "fico_score": 720,
    "dti_ratio": 28.0,
    "annual_income": 75000,
    ...
}

deny_application = {
    "fico_score": 580,   ← below the 620 minimum — should always DENY
    "dti_ratio": 50.0,   ← above the 43% maximum — should always DENY
    ...
}
```
These are pre-built loan applications that test files can use without typing out all the fields every time.

**3. A mock predictor (`mock_predictor`)**
```python
mock.predict.return_value = {
    "probability_of_default": 0.08,
    "risk_tier": "LOW",
    ...
}
```
A **mock** is a fake version of something — in this case, a fake version of the XGBoost model. Instead of loading the real model file and computing a real prediction, the mock always returns the same fixed answer.

Why? Because we want to test the API logic (routing, database writes, response format) separately from the model logic. If we used the real model in API tests, a model file change could break API tests — even though the API code didn't change.

---

#### `tests/test_tools.py` — Policy Rule Tests

**What it tests**: the `check_lending_policy()` function in `tools/policy_tools.py` — the pure logic that decides whether to APPROVE, DENY, or REFER based on the 7 policy rules.

**Why this is the most important test file**: policy rules are the legal and business layer of the system. A bug here means the system might approve someone who should be denied, which is a financial and compliance risk.

**What the tests check:**

| Test name | What it verifies |
|---|---|
| `test_approve_clean_application` | A good applicant (FICO 720, DTI 28%) gets APPROVED |
| `test_deny_fico_below_620` | FICO 610 → DENY with FICO violation in the list |
| `test_deny_dti_exceeds_43` | DTI 45% → DENY with DTI violation |
| `test_deny_pd_exceeds_threshold` | PD of 0.40 → DENY (above 0.35 hard limit) |
| `test_deny_ltv_exceeds_97` | LTV 0.98 → DENY (above 97% limit) |
| `test_deny_loan_to_income_exceeds_5x` | $400k loan on $70k income → DENY |
| `test_refer_pd_in_grey_zone` | PD of 0.25 → REFER_TO_HUMAN (in the 0.20–0.35 zone) |
| `test_fico_620_exact_boundary_passes` | FICO exactly 620 → no FICO violation (boundary is exclusive) |
| `test_fico_619_exact_boundary_fails` | FICO exactly 619 → FICO violation triggered |
| `test_dti_43_exact_boundary_passes` | DTI exactly 43.0 → not a violation (rule is DTI > 43, not ≥) |
| `test_rate_is_float_when_approved` | Approved applications always get a recommended rate between 3.5%–36% |
| `test_max_loan_amount_scales_with_income` | Higher income → higher maximum loan amount |

**The boundary tests are the most valuable ones.** In credit risk, the exact cutoff matters. FICO 620 must pass. FICO 619 must fail. These tests prove the boundary logic is correct at the exact threshold.

**Example test:**
```python
def test_fico_619_exact_boundary_fails(self):
    result = check_lending_policy(fico_score=619, ...)
    assert any("FICO" in v for v in result["violations"])
```

This calls the real policy function with FICO=619 and checks that the violations list contains something with "FICO" in it. If someone accidentally changes the rule to `fico_score < 619`, this test catches it immediately.

---

#### `tests/test_model.py` — Model Evaluator Tests

**What it tests**: the `ModelEvaluator` class in `models/evaluator.py` — the math that computes AUC, KS statistic, Gini, Precision, and Recall.

**Why test math?** Because these metrics are computed on arrays of numbers, and small bugs (wrong formula, wrong axis) produce wrong numbers silently. You wouldn't know the KS statistic was wrong unless you verified the formula independently.

**What the tests check:**
- AUC of 1.0 for a perfect predictor (all defaults scored higher than all non-defaults)
- AUC of ~0.5 for a random predictor (no skill)
- Gini = 2 × AUC - 1 (always)
- KS is always between 0 and 1
- Precision and Recall are always between 0 and 1

These tests use only NumPy arrays — no model file, no database, no API.

---

#### `tests/test_data_pipeline.py` — Data Processing Tests

**What it tests**: the preprocessing pipeline — parsing raw LendingClub strings into numbers, engineering features, encoding categoricals.

**Why test data processing?** Because data bugs are the hardest to find. If `"10+ years"` gets parsed as `0.0` instead of `10.0`, the model silently gets worse input but still returns a number. Tests catch this at the source.

**What the tests check:**
- Employment length parsing: `"10+ years"` → 10.0, `"< 1 year"` → 0.5
- Interest rate parsing: `"13.56%"` → 0.1356
- Revolving utilisation parsing: `"54.2%"` → 0.542
- Feature engineering: `fico_mid = (fico_low + fico_high) / 2`
- That the preprocessor outputs exactly 18 features (no more, no less)
- That the synthetic generator produces records in the expected ranges

---

#### `tests/test_api.py` — API Endpoint Tests

**What it tests**: the FastAPI endpoints — that they return the right HTTP status codes and the right JSON structure.

**What is `TestClient`?** FastAPI provides a test client that simulates HTTP requests without actually starting a real server. You call `client.get("/health")` in your test code and it behaves exactly as if a real browser made that request — but no network, no port, no waiting.

```python
def test_health_returns_200(self, client):
    resp = client.get("/health")
    assert resp.status_code == 200          ← HTTP 200 means OK
    assert "model_loaded" in resp.json()    ← response must have this field
```

**What the tests check:**

| Test | What it verifies |
|---|---|
| `test_health_returns_200` | `/health` always returns 200 and has `model_loaded` and `db_connected` fields |
| `test_audit_report_empty_db` | `/audit-report` returns 200 with zero decisions on a fresh DB |
| `test_audit_report_window_validation` | `window_days=0` returns 422 (Pydantic rejects invalid input) |
| `test_list_decisions_empty` | `/decisions` returns an empty list `[]` when no decisions exist |
| `test_get_decision_not_found` | `/decision/nonexistent-id` returns 404 (Not Found) |
| `test_list_decisions_pagination_params` | `?skip=0&limit=5` is accepted without error |

**Why are the API tests mocked?** The `test_app` fixture replaces two things:
1. The real database → in-memory SQLite (so tests don't touch your real `credit_risk.db`)
2. The real predictor → a MagicMock (so tests don't need `models/artifacts/model.json`)

This means the API tests check routing, validation, and response format — independent of the model or real data.

---

### Summary — what each test file does

| File | Tests what | Requires model? | Requires DB? |
|---|---|---|---|
| `conftest.py` | Shared setup (fixtures) | No | No (creates temporary one) |
| `test_tools.py` | Policy rules + audit tools | No | No |
| `test_model.py` | AUC/KS/Gini math | No | No |
| `test_data_pipeline.py` | CSV parsing + feature engineering | No | No |
| `test_api.py` | HTTP endpoints + JSON responses | No (mocked) | No (in-memory) |

**None of these tests require a trained model, an OpenAI key, or a real database.** They run in under 5 seconds on any machine.

---

---

## Part 2 — The `api/` Folder and FastAPI

### What is an API?

API stands for Application Programming Interface. It is a way for two programs to talk to each other.

In this project, the API lets any program (a browser, a Python script, a Streamlit app, a mobile app, another company's system) send a loan application to your system and get a decision back — without knowing anything about how the agents or model work internally.

**The analogy**: when you use a weather app on your phone, the app doesn't do the weather calculations itself. It sends a request to a weather API ("give me the weather for San Francisco") and receives a structured response. Your credit risk API works the same way — someone sends a loan application, your system processes it through 4 agents, and returns a structured decision.

---

### What is FastAPI?

FastAPI is a Python framework for building APIs. It handles all the boring parts:
- Receiving HTTP requests
- Validating that the input has the right fields and types
- Routing requests to the right function
- Formatting responses as JSON
- Generating documentation automatically

You write Python functions. FastAPI handles the rest.

---

### What is HTTP?

HTTP is the language that web browsers and APIs use to communicate. Every request has a **method** (verb) that says what kind of action is happening:

| Method | Meaning | Analogy | Used in this project for |
|---|---|---|---|
| `GET` | Retrieve something | Looking up a record | Getting a past decision, audit report |
| `POST` | Send something and trigger an action | Submitting a form | Submitting a loan application |

The response always includes a **status code** — a 3-digit number that says whether it worked:

| Code | Meaning | When you see it |
|---|---|---|
| `200 OK` | Request succeeded | GET requests that work |
| `201 Created` | Something was created | POST /evaluate-loan after a decision is saved |
| `404 Not Found` | The thing you asked for doesn't exist | GET /decision/wrong-id |
| `422 Unprocessable` | Your input was invalid | Missing a required field, FICO out of range |
| `500 Internal Server Error` | Something crashed on the server | Agent pipeline error |

---

### The `api/` folder structure

```
api/
├── app.py           ← Creates the FastAPI application, wires everything together
├── schemas.py       ← Defines what the JSON request and response must look like
├── dependencies.py  ← Provides shared resources (database session, model) to endpoints
└── routes/
    ├── loan.py      ← POST /evaluate-loan
    ├── decisions.py ← GET /decision/{id}, GET /decisions
    └── audit.py     ← GET /audit-report, GET /audit-log/{id}
```

---

### `api/schemas.py` — The JSON contract

Schemas define exactly what fields are allowed in requests and responses. FastAPI uses these to automatically validate input and format output.

**Request schema** (`LoanApplicationRequest`) — what you send IN:
```python
class LoanApplicationRequest(BaseModel):
    fico_score: int          # must be between 300 and 850
    dti_ratio: float         # must be between 0 and 100
    annual_income: float     # must be greater than 0
    loan_amount: float       # must be greater than 0
    loan_term_months: int    # must be between 12 and 360
    loan_purpose: str        # required
    # ... optional fields have defaults
```

If you send `fico_score: 900` (above 850), FastAPI automatically returns a 422 error before your code even runs:
```json
{
  "detail": [{"loc": ["body", "fico_score"], "msg": "ensure this value is less than or equal to 850"}]
}
```

**Response schema** (`DecisionResponse`) — what comes back OUT:
```python
class DecisionResponse(BaseModel):
    decision_id: str
    application_id: str
    policy_decision: str          # "APPROVE", "DENY", or "REFER_TO_HUMAN"
    probability_of_default: float # 0.0 to 1.0
    risk_tier: str                # "LOW", "MEDIUM", "HIGH", "VERY_HIGH"
    recommended_rate: float       # e.g. 0.0879 = 8.79% APR
    narrative: str                # plain-English explanation
    top_shap_features: list       # 5 features with shap values and directions
    violations: list              # policy rules that were triggered
    audit_passed: bool            # did the fairness check pass?
    bias_flags: list              # any demographic bias flags
    consistency_check: bool       # does this match similar past decisions?
    decided_at: datetime          # timestamp
```

---

### The endpoints — what each one does

---

#### `POST /evaluate-loan`

**What it does**: runs the full 4-agent pipeline on a loan application and returns a decision.

**What you send** (the request body — JSON):
```json
{
    "fico_score": 760,
    "dti_ratio": 14.0,
    "annual_income": 120000,
    "loan_amount": 15000,
    "loan_term_months": 36,
    "loan_purpose": "debt_consolidation",
    "grade": "A",
    "verification_status": "Verified"
}
```

**What happens inside** (you don't see this, but this is the pipeline):
1. Application is saved to the `loan_applications` database table
2. The 4-agent GroupChat runs (~30–90 seconds):
   - RiskAnalyst → XGBoost prediction
   - PolicyAgent → checks 7 lending rules
   - ExplanationAgent → SHAP narrative
   - AuditorAgent → fairness check + saves decision to DB
3. The saved decision is retrieved and returned

**What comes back** (the response — JSON):
```json
{
    "decision_id": "184ca0cb-3636-49af-8d2e-22b8255a224f",
    "application_id": "d5ed29f3-2d92-4796-9b7f-d3d0f5549816",
    "policy_decision": "APPROVE",
    "probability_of_default": 0.187,
    "risk_tier": "MEDIUM",
    "recommended_rate": 0.0879,
    "narrative": "Your application has been reviewed and meets our current lending criteria...",
    "top_shap_features": [
        {"name": "fico_mid", "scaled_value": 0.54, "shap_value": -0.152, "direction": "decreases_risk"},
        {"name": "dti_clipped", "scaled_value": -0.31, "shap_value": 0.048, "direction": "increases_risk"}
    ],
    "violations": [],
    "audit_passed": true,
    "bias_flags": [],
    "consistency_check": true,
    "decided_at": "2026-04-13T11:25:39.320526"
}
```

**Reading the response field by field:**

| Field | What it means | Example |
|---|---|---|
| `decision_id` | Unique ID for this decision in the database. Use this to look it up later. | `"184ca0cb-..."` |
| `application_id` | Unique ID for the loan application record. | `"d5ed29f3-..."` |
| `policy_decision` | The final outcome. Three possible values. | `"APPROVE"` |
| `probability_of_default` | XGBoost's estimate that this borrower will default. 0 = certain to repay, 1 = certain to default. | `0.187` = 18.7% |
| `risk_tier` | Bucketed version of the probability. | `"MEDIUM"` (10%–20%) |
| `recommended_rate` | Suggested annual interest rate using risk-based pricing. `null` if denied. | `0.0879` = 8.79% APR |
| `narrative` | Plain-English explanation of the decision, suitable to share with the applicant. | `"Your application has been reviewed..."` |
| `top_shap_features` | The 5 features that most influenced the prediction. Each has a name, a value, and a direction. | See below |
| `violations` | Which policy rules were triggered. Empty if approved cleanly. | `["DTI_EXCEEDS_43PCT"]` |
| `audit_passed` | Whether the fairness check passed — no demographic bias detected against recent decisions. | `true` |
| `bias_flags` | Specific bias issues found. Empty if audit passed. | `["DEMOGRAPHIC_PARITY_VIOLATION"]` |
| `consistency_check` | Whether this decision agrees with similar past applications. | `true` |
| `decided_at` | When the decision was made (UTC timestamp). | `"2026-04-13T11:25:39"` |

**Reading SHAP features:**
```json
{"name": "fico_mid", "scaled_value": 0.54, "shap_value": -0.152, "direction": "decreases_risk"}
```
- `name`: which feature this is (`fico_mid` = credit score)
- `scaled_value`: the feature's value after standardisation (0 = average, positive = above average, negative = below average)
- `shap_value`: how much this feature moved the predicted probability. Negative = pushed toward repay, positive = pushed toward default.
- `direction`: human-readable summary — `"decreases_risk"` or `"increases_risk"`

---

#### `GET /decisions`

**What it does**: returns a list of all past decisions saved in the database.

**Optional query parameters** (add to the URL):
- `?skip=0&limit=20` — pagination (skip first N, return next M)
- `?policy_decision=APPROVE` — filter to only approved decisions

**Example URL**: `http://localhost:8000/decisions?limit=10&policy_decision=DENY`

**What comes back**:
```json
[
    {
        "decision_id": "184ca0cb-...",
        "application_id": "d5ed29f3-...",
        "policy_decision": "APPROVE",
        "probability_of_default": 0.187,
        "risk_tier": "MEDIUM",
        "decided_at": "2026-04-13T11:25:39"
    },
    {
        "decision_id": "cc36acd7-...",
        "application_id": "1351b021-...",
        "policy_decision": "DENY",
        "probability_of_default": 0.479,
        "risk_tier": "VERY_HIGH",
        "decided_at": "2026-04-13T14:31:02"
    }
]
```

Note: this returns a **summary** (fewer fields than the full decision response) — enough to browse the list.

---

#### `GET /decision/{decision_id}`

**What it does**: returns the complete record for one specific decision.

**Example URL**: `http://localhost:8000/decision/184ca0cb-3636-49af-8d2e-22b8255a224f`

**What comes back**: the same full JSON as the POST response — all fields including narrative, SHAP features, violations, audit result.

---

#### `GET /audit-log/{decision_id}`

**What it does**: returns the raw fairness audit record for a specific decision — more detail than what's in the decision response.

**What comes back**:
```json
{
    "id": "fbaa2e70-...",
    "decision_id": "184ca0cb-...",
    "audited_at": "2026-04-13T11:25:40",
    "audit_passed": true,
    "consistency_check": true,
    "demographic_parity_delta": 0.03,
    "equalized_odds_delta": null,
    "disparate_impact_ratio": 0.94,
    "bias_flags": [],
    "shap_top_features": [...],
    "shap_base_value": 0.12,
    "shap_sum": -0.08,
    "audit_notes": "No recent decisions for comparison — audit passed by default."
}
```

**Reading the fairness fields:**

| Field | What it means | Pass threshold |
|---|---|---|
| `demographic_parity_delta` | Max gap in approval rates between demographic groups | < 10 percentage points |
| `disparate_impact_ratio` | Min group rate ÷ max group rate | ≥ 0.80 (the EEOC 80% rule) |
| `equalized_odds_delta` | TPR gap across groups | < 10 percentage points |

---

#### `GET /audit-report?window_days=30`

**What it does**: generates an aggregate fairness report across all decisions in the last N days.

**What comes back**:
```json
{
    "window_days": 30,
    "total_decisions": 42,
    "approval_rate_overall": 0.67,
    "approval_rate_by_gender": {"Male": 0.69, "Female": 0.65},
    "approval_rate_by_race": {"White": 0.71, "Hispanic": 0.63, "Black": 0.61},
    "demographic_parity_gender": 0.04,
    "demographic_parity_race": 0.10,
    "disparate_impact_gender": 0.94,
    "disparate_impact_race": 0.86,
    "flagged_bias_categories": [],
    "generated_at": "2026-04-13T15:00:00"
}
```

This is the report a compliance team would review regularly to check for systemic bias across many decisions.

---

#### `GET /health`

**What it does**: a simple status check that confirms the server is running, the model is loaded, and the database is connected.

```json
{"status": "ok", "model_loaded": true, "db_connected": true}
```

Used by monitoring systems to alert if any component goes down.

---

### How the Streamlit UI relates to the API

The Streamlit app (`streamlit_app.py`) does not do any ML or agent work itself. It is a visual wrapper around the API:

```
User fills form in Streamlit
        ↓
Streamlit sends POST to http://localhost:8000/evaluate-loan
        ↓
FastAPI runs the 4-agent pipeline
        ↓
FastAPI returns JSON response
        ↓
Streamlit reads the JSON and draws charts and badges
```

This is why both servers must run at the same time. Streamlit handles what you see. FastAPI handles everything that computes.

---

### How to read a JSON response

JSON (JavaScript Object Notation) is the standard format for data exchange between programs. It looks like a Python dictionary.

```json
{
    "policy_decision": "APPROVE",        ← string value (text in quotes)
    "probability_of_default": 0.187,     ← float value (decimal number)
    "audit_passed": true,                ← boolean value (true or false, lowercase)
    "violations": [],                    ← empty list
    "top_shap_features": [               ← list of objects
        {
            "name": "fico_mid",
            "shap_value": -0.152,
            "direction": "decreases_risk"
        }
    ],
    "decided_at": "2026-04-13T11:25:39" ← datetime as a string (ISO 8601 format)
}
```

**In Python**, reading JSON from the API is one line:
```python
import requests
response = requests.post("http://localhost:8000/evaluate-loan", json=my_application)
data = response.json()   # converts JSON to a Python dict

print(data["policy_decision"])          # "APPROVE"
print(data["probability_of_default"])   # 0.187
print(data["top_shap_features"][0])     # {"name": "fico_mid", ...}
```

---

### The `api/dependencies.py` file

Dependencies are shared resources that multiple endpoints need. Instead of creating a new database connection or loading the model inside every endpoint function, you define them once in `dependencies.py` and FastAPI injects them automatically.

```python
def get_db_session():
    """Provides a database session to any endpoint that needs one."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()    # always closes, even if an error occurred
```

The `yield` pattern (called a context manager) guarantees that the database connection is always closed after the request — whether it succeeded or failed.

In tests, these dependencies are **overridden** with test versions:
```python
app.dependency_overrides[get_db_session] = override_get_db    # use in-memory DB
app.dependency_overrides[get_predictor] = lambda: mock_pred   # use fake model
```

This is why the tests can run without real infrastructure.
