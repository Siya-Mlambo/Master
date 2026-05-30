# ============================================================
# Credit Risk XAI — Streamlit Demo App
# Run: streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk XAI — Sya Mlambo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body, .stApp { background-color: #0a0a0f; color: #e8e8f0; }
.metric-card {
    background: #1e1e30;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.risk-high   { color: #ff6b6b; font-size: 3rem; font-weight: 700; }
.risk-medium { color: #f5a623; font-size: 3rem; font-weight: 700; }
.risk-low    { color: #2dd4a8; font-size: 3rem; font-weight: 700; }
.shap-row    { padding: 4px 0; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("# 🏦 Credit Risk Prediction")
st.markdown("### with SHAP Explainability · by Siyabonga Mlambo")
st.markdown("---")

# ── SIDEBAR INPUTS ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Applicant Profile")
    st.markdown("Adjust inputs to simulate a loan applicant.")

    age              = st.slider("Age", 18, 75, 34)
    income           = st.number_input("Annual Income (R)", 50000, 2000000, 420000, step=10000)
    loan_amount      = st.number_input("Loan Amount (R)", 10000, 1000000, 180000, step=5000)
    credit_score     = st.slider("Credit Score", 300, 850, 680)
    employment_years = st.slider("Employment Years", 0, 40, 5)
    num_accounts     = st.slider("Number of Accounts", 1, 15, 4)
    missed_payments  = st.slider("Missed Payments", 0, 10, 0)
    loan_purpose     = st.selectbox("Loan Purpose",
                        ['home', 'car', 'education', 'personal', 'business'])
    employment_type  = st.selectbox("Employment Type",
                        ['employed', 'self_employed', 'unemployed', 'retired'])

    st.markdown("---")
    predict_btn = st.button("⚡ Run Prediction", use_container_width=True, type="primary")

# ── MODEL TRAINING (cached) ──────────────────────────────────
@st.cache_resource(show_spinner="Training model on credit data...")
def train_model():
    np.random.seed(42)
    n = 8000
    age_     = np.random.randint(18, 75, n)
    income_  = np.random.lognormal(11.5, 0.6, n).astype(int)
    loan_    = np.random.lognormal(11.0, 0.7, n).astype(int)
    cs_      = np.random.normal(650, 80, n).clip(300, 850).astype(int)
    emp_     = np.random.exponential(5, n).clip(0, 40).round(1)
    accts_   = np.random.randint(1, 15, n)
    missed_  = np.random.poisson(0.3, n).clip(0, 10)
    dr_      = (loan_ / income_).clip(0, 5).round(3)
    purpose_ = np.random.choice(['home','car','education','personal','business'], n)
    emp_type_= np.random.choice(['employed','self_employed','unemployed','retired'], n,
                                 p=[0.6, 0.2, 0.1, 0.1])

    log_odds = (
        -2.0
        + 0.015 * (650 - cs_) / 50
        + 1.2 * (dr_ > 0.5)
        + 0.8 * (emp_type_ == 'unemployed')
        + 0.5 * (missed_ > 0)
        - 0.4 * (emp_ > 5)
        - 0.3 * (income_ > 500000)
        + 0.2 * (age_ < 25)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    default = (np.random.random(n) < prob).astype(int)

    le_purpose  = LabelEncoder().fit(purpose_)
    le_emptype  = LabelEncoder().fit(emp_type_)

    df = pd.DataFrame({
        'age': age_, 'income': income_, 'loan_amount': loan_,
        'credit_score': cs_, 'employment_years': emp_, 'num_accounts': accts_,
        'missed_payments': missed_, 'debt_ratio': dr_,
        'loan_purpose': le_purpose.transform(purpose_),
        'employment_type': le_emptype.transform(emp_type_),
        'loan_to_income': loan_ / income_,
        'debt_risk_flag': (dr_ > 0.5).astype(int),
        'missed_flag': (missed_ > 1).astype(int),
        'young_flag': (age_ < 25).astype(int),
    })

    X = df.drop(columns=[])
    y = default
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                         random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)

    mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        use_label_encoder=False, eval_metric='auc',
                        random_state=42, n_jobs=-1)
    mdl.fit(X_sm, y_sm)
    exp = shap.TreeExplainer(mdl)
    return mdl, exp, le_purpose, le_emptype, list(df.columns)

model, explainer, le_p, le_e, feature_names = train_model()

# ── PREDICTION ───────────────────────────────────────────────
def make_input():
    dr = loan_amount / income
    return pd.DataFrame([{
        'age'             : age,
        'income'          : income,
        'loan_amount'     : loan_amount,
        'credit_score'    : credit_score,
        'employment_years': employment_years,
        'num_accounts'    : num_accounts,
        'missed_payments' : missed_payments,
        'debt_ratio'      : dr,
        'loan_purpose'    : le_p.transform([loan_purpose])[0],
        'employment_type' : le_e.transform([employment_type])[0],
        'loan_to_income'  : loan_amount / income,
        'debt_risk_flag'  : int(dr > 0.5),
        'missed_flag'     : int(missed_payments > 1),
        'young_flag'      : int(age < 25),
    }])

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown("### 📋 Applicant Summary")
    summary = {
        "Age": age,
        "Annual Income": f"R {income:,}",
        "Loan Amount": f"R {loan_amount:,}",
        "Credit Score": credit_score,
        "Debt-to-Income": f"{loan_amount/income:.2f}",
        "Employment Years": employment_years,
        "Missed Payments": missed_payments,
        "Loan Purpose": loan_purpose.title(),
        "Employment Type": employment_type.replace('_', ' ').title(),
    }
    for k, v in summary.items():
        st.markdown(f"**{k}:** {v}")

with col2:
    st.markdown("### 🔮 Prediction Result")
    inp          = make_input()
    prob_default = model.predict_proba(inp)[0][1]
    pct          = int(prob_default * 100)

    if pct < 30:
        risk_class = "risk-low"
        verdict    = "✅ LOW RISK — Likely Approved"
        bar_color  = "#2dd4a8"
    elif pct < 60:
        risk_class = "risk-medium"
        verdict    = "⚠️ MEDIUM RISK — Manual Review"
        bar_color  = "#f5a623"
    else:
        risk_class = "risk-high"
        verdict    = "❌ HIGH RISK — Likely Declined"
        bar_color  = "#ff6b6b"

    st.markdown(f'<div class="{risk_class}">{pct}%</div>', unsafe_allow_html=True)
    st.markdown(f"**{verdict}**")
    st.markdown(f"Default Probability: **{prob_default:.1%}**")
    st.progress(prob_default)

st.markdown("---")

# ── SHAP EXPLANATION ─────────────────────────────────────────
st.markdown("### 🔍 SHAP Explanation — Why this prediction?")
st.caption("SHAP values show exactly how much each feature contributed to this specific prediction.")

shap_vals = explainer.shap_values(inp)
shap_df = pd.DataFrame({
    'Feature'   : feature_names,
    'Value'     : inp.values[0],
    'SHAP'      : shap_vals[0],
    'Direction' : ['↑ Increases Risk' if s > 0 else '↓ Reduces Risk' for s in shap_vals[0]]
}).reindex(pd.Series(np.abs(shap_vals[0])).sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#1e1e30')
colors = ['#ff6b6b' if v > 0 else '#2dd4a8' for v in shap_df['SHAP']]
bars = ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors, alpha=0.85, edgecolor='none')
ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
ax.set_xlabel('SHAP Value (impact on prediction)', color='white')
ax.set_title('Feature Contributions to this Prediction', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('rgba(255,255,255,0.1)')
plt.tight_layout()
st.pyplot(fig)

st.markdown("#### Feature Contribution Table")
st.dataframe(
    shap_df[['Feature', 'Value', 'SHAP', 'Direction']].round(4),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.caption("Built by Siyabonga Mlambo · MSc Data Science · Sol Plaatje University · "
           "Model: XGBoost + SHAP · Dataset: Synthetic Credit Risk")
