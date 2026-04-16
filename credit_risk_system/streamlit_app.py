"""
Streamlit UI for the Multi-Agent Credit Risk Decision System.

Runs on: http://localhost:8501
Requires: FastAPI server running at http://localhost:8000

Start both with:
    Terminal 1: python main.py serve
    Terminal 2: streamlit run streamlit_app.py
"""

import time

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.approve-badge {
    background-color: #d4edda; color: #155724;
    padding: 16px 32px; border-radius: 8px;
    font-size: 28px; font-weight: bold;
    text-align: center; border: 2px solid #c3e6cb;
}
.deny-badge {
    background-color: #f8d7da; color: #721c24;
    padding: 16px 32px; border-radius: 8px;
    font-size: 28px; font-weight: bold;
    text-align: center; border: 2px solid #f5c6cb;
}
.refer-badge {
    background-color: #fff3cd; color: #856404;
    padding: 16px 32px; border-radius: 8px;
    font-size: 28px; font-weight: bold;
    text-align: center; border: 2px solid #ffeeba;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 16px; border-radius: 8px;
    border: 1px solid #dee2e6;
    text-align: center;
}
.narrative-box {
    background-color: #f0f4ff;
    padding: 16px; border-radius: 8px;
    border-left: 4px solid #4a6fa5;
    font-size: 15px; line-height: 1.6;
}
.violation-item {
    background-color: #fff3cd;
    padding: 8px 12px; border-radius: 4px;
    margin: 4px 0; font-size: 13px;
}
.bias-item {
    background-color: #f8d7da;
    padding: 8px 12px; border-radius: 4px;
    margin: 4px 0; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Application Form ────────────────────────────────────────────────
st.sidebar.title("🏦 Loan Application")
st.sidebar.markdown("Fill in the applicant details and click **Evaluate**.")

with st.sidebar:
    st.subheader("Required Fields")
    fico_score = st.slider("FICO Credit Score", min_value=300, max_value=850, value=720, step=1,
                           help="Credit score (300=poor, 850=excellent)")
    dti_ratio = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=60.0, value=28.0, step=0.5,
                          help="Monthly debt payments ÷ monthly gross income × 100. Hard deny above 43%.")
    annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=5000000,
                                    value=75000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000,
                                  value=20000, step=500)
    loan_term_months = st.selectbox("Loan Term", options=[12, 24, 36, 48, 60, 84, 120],
                                    index=2, format_func=lambda x: f"{x} months")
    loan_purpose = st.selectbox("Loan Purpose", options=[
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "medical", "car", "vacation", "moving", "other"
    ])

    st.subheader("Optional Fields")
    with st.expander("Financial Details", expanded=True):
        grade = st.selectbox("Credit Grade", options=["A", "B", "C", "D", "E", "F", "G"], index=2,
                             help="A = best, G = worst")
        verification_status = st.selectbox("Income Verification",
                                           options=["Verified", "Source Verified", "Not Verified"])
        home_ownership = st.selectbox("Home Ownership", options=["MORTGAGE", "RENT", "OWN"])
        revolving_util = st.slider("Revolving Utilisation", min_value=0.0, max_value=1.0,
                                   value=0.30, step=0.01,
                                   help="Credit card balance ÷ credit limit. Lower is better.")
        employment_length_years = st.slider("Employment Length (years)", min_value=0.0,
                                            max_value=40.0, value=3.0, step=0.5)
        interest_rate = st.slider("Current Interest Rate", min_value=0.03, max_value=0.36,
                                  value=0.10, step=0.005, format="%.3f")
        ltv_ratio = st.slider("Loan-to-Value Ratio", min_value=0.0, max_value=1.5,
                              value=0.75, step=0.01,
                              help="Loan amount ÷ asset value. Hard deny above 0.97.")
        delinq_2yrs = st.number_input("Delinquencies (past 2 years)", min_value=0, max_value=20, value=0)
        open_accounts = st.number_input("Open Accounts", min_value=0, max_value=50, value=5)
        total_accounts = st.number_input("Total Accounts", min_value=0, max_value=100, value=10)
        state = st.text_input("US State Code", value="CA", max_chars=2).upper()

    with st.expander("Applicant Info (optional)"):
        applicant_name = st.text_input("Applicant Name", value="")
        st.caption("Protected attributes below are stored for fairness auditing only — never used in the model.")
        gender = st.selectbox("Gender", options=["", "Male", "Female", "Non-binary", "Prefer not to say"])
        race = st.selectbox("Race / Ethnicity", options=[
            "", "White", "Black", "Hispanic", "Asian", "Native American", "Other", "Prefer not to say"
        ])
        age = st.number_input("Age", min_value=18, max_value=120, value=35)

    evaluate_btn = st.button("Evaluate Loan Application", type="primary", use_container_width=True)


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("Credit Risk Decision System")
st.caption("Multi-agent AI pipeline: Risk Analyst → Policy Agent → Explanation Agent → Auditor Agent")

# API health check
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded") and health.get("db_connected"):
        st.success("API server connected — model loaded, database ready")
    else:
        st.warning(f"API server reachable but not fully ready: {health}")
except Exception:
    st.error("Cannot reach API server at http://localhost:8000 — run `python main.py serve` first")
    st.stop()

# ── Run evaluation ─────────────────────────────────────────────────────────────
if evaluate_btn:
    payload = {
        "fico_score": fico_score,
        "dti_ratio": dti_ratio,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose,
        "grade": grade,
        "verification_status": verification_status,
        "home_ownership": home_ownership,
        "revolving_util": revolving_util,
        "employment_length_years": employment_length_years,
        "interest_rate": interest_rate,
        "ltv_ratio": ltv_ratio,
        "delinq_2yrs": delinq_2yrs,
        "open_accounts": open_accounts,
        "total_accounts": total_accounts,
        "state": state,
    }
    if applicant_name:
        payload["applicant_name"] = applicant_name
    if gender:
        payload["gender"] = gender
    if race:
        payload["race"] = race
    if age:
        payload["age"] = age

    with st.spinner("Running 4-agent evaluation pipeline... (this takes 30–90 seconds)"):
        try:
            start = time.time()
            response = requests.post(f"{API_URL}/evaluate-loan", json=payload, timeout=180)
            elapsed = time.time() - start

            if response.status_code == 201:
                result = response.json()
                st.session_state["last_result"] = result
                st.session_state["last_elapsed"] = elapsed
            else:
                st.error(f"API error {response.status_code}: {response.text}")
                st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out after 3 minutes. The agents may still be running — check the server logs.")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

# ── Display result ─────────────────────────────────────────────────────────────
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    elapsed = st.session_state.get("last_elapsed", 0)

    decision = result["policy_decision"]
    pd_val = result["probability_of_default"]
    risk_tier = result["risk_tier"]
    rate = result.get("recommended_rate")
    narrative = result.get("narrative", "")
    violations = result.get("violations") or []
    bias_flags = result.get("bias_flags") or []
    audit_passed = result.get("audit_passed", True)
    shap_features = result.get("top_shap_features") or []

    # ── Decision badge ────────────────────────────────────────────────────────
    st.markdown("---")
    col_badge, col_meta = st.columns([1, 2])

    with col_badge:
        if decision == "APPROVE":
            st.markdown('<div class="approve-badge">✅ APPROVED</div>', unsafe_allow_html=True)
        elif decision == "DENY":
            st.markdown('<div class="deny-badge">❌ DENIED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="refer-badge">⚠️ REFER TO HUMAN</div>', unsafe_allow_html=True)
        st.caption(f"Evaluated in {elapsed:.1f}s  •  Decision ID: {result['decision_id'][:8]}...")

    with col_meta:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Probability of Default", f"{pd_val:.1%}")
        with m2:
            st.metric("Risk Tier", risk_tier)
        with m3:
            if rate:
                st.metric("Recommended Rate", f"{rate:.2%}")
            else:
                st.metric("Recommended Rate", "N/A")

    # ── Gauge chart ──────────────────────────────────────────────────────────
    st.markdown("---")
    col_gauge, col_shap = st.columns([1, 1])

    with col_gauge:
        st.subheader("Default Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#e74c3c" if pd_val > 0.35 else "#f39c12" if pd_val > 0.20 else "#27ae60"},
                "steps": [
                    {"range": [0, 10], "color": "#d5f5e3"},
                    {"range": [10, 20], "color": "#fef9e7"},
                    {"range": [20, 35], "color": "#fde8d8"},
                    {"range": [35, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 35,
                },
            },
            title={"text": "Probability of Default<br><span style='font-size:12px'>Hard deny above 35%</span>"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP bar chart ────────────────────────────────────────────────────────
    with col_shap:
        st.subheader("Top Risk Factors (SHAP)")
        if shap_features:
            names = [f["name"].replace("_", " ").title() for f in shap_features]
            shap_vals = [f.get("shap_value", 0.0) for f in shap_features]
            directions = [f.get("direction", "unknown") for f in shap_features]
            colors = ["#e74c3c" if d == "increases_risk" else "#27ae60" for d in directions]

            fig_shap = go.Figure(go.Bar(
                x=shap_vals,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition="outside",
            ))
            fig_shap.update_layout(
                height=280,
                margin=dict(t=20, b=20, l=10, r=60),
                xaxis_title="SHAP Value (impact on default probability)",
                yaxis={"autorange": "reversed"},
                showlegend=False,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Red = increases default risk   |   Green = decreases default risk")
        else:
            st.info("SHAP features not available for this decision.")

    # ── Narrative ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Decision Narrative")
    st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

    # ── Violations & audit ────────────────────────────────────────────────────
    st.markdown("---")
    col_v, col_a = st.columns(2)

    with col_v:
        st.subheader("Policy Violations")
        if violations:
            for v in violations:
                st.markdown(f'<div class="violation-item">⚠️ {v.replace("_", " ")}</div>', unsafe_allow_html=True)
        else:
            st.success("No policy violations")

    with col_a:
        st.subheader("Fairness Audit")
        if audit_passed:
            st.success("Audit passed — no bias detected")
        else:
            st.error("Audit failed — bias flags raised")
        if bias_flags:
            for b in bias_flags:
                st.markdown(f'<div class="bias-item">🚨 {b.replace("_", " ")}</div>', unsafe_allow_html=True)
        consistency = result.get("consistency_check", True)
        if consistency:
            st.info("Consistency check passed — decision aligns with similar past cases")
        else:
            st.warning("Consistency check failed — decision differs from similar past cases")

    # ── Raw JSON expander ─────────────────────────────────────────────────────
    with st.expander("Raw API Response (JSON)"):
        st.json(result)


# ── Past Decisions Tab ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Past Decisions")

col_refresh, col_filter = st.columns([1, 2])
with col_refresh:
    load_decisions = st.button("Load Past Decisions")
with col_filter:
    filter_decision = st.selectbox("Filter by outcome", options=["All", "APPROVE", "DENY", "REFER_TO_HUMAN"])

if load_decisions:
    try:
        params = {}
        if filter_decision != "All":
            params["policy_decision"] = filter_decision
        resp = requests.get(f"{API_URL}/decisions", params=params, timeout=10)
        decisions = resp.json()

        if not decisions:
            st.info("No decisions found.")
        else:
            rows = []
            for d in decisions:
                rows.append({
                    "Decision ID": d["decision_id"][:8] + "...",
                    "Outcome": d["policy_decision"],
                    "PD": f"{d['probability_of_default']:.1%}",
                    "Risk Tier": d["risk_tier"],
                    "Decided At": d["decided_at"][:19].replace("T", " "),
                })
            st.dataframe(rows, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load decisions: {e}")
