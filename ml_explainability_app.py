# ============================================================
# PROJECT 6: ML Explainability App
# Author   : Siyabonga Mlambo
# Run      : streamlit run app/ml_explainability_app.py
# Goal     : Upload any CSV → train model → get SHAP explanations
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import io, warnings
warnings.filterwarnings('ignore')

# ── PAGE SETUP ────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Explainability App — Sya Mlambo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0a0a0f; color: #e8e8f0; }
.metric-box {
    background: #1e1e30;
    border: 1px solid rgba(124,111,247,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 4px;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #7c6ff7; }
.metric-lbl { font-size: 12px; color: #9898b0; text-transform: uppercase; letter-spacing: 0.06em; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
st.markdown("# 🔍 ML Explainability App")
st.markdown("**Upload any CSV dataset → Train a model → Get instant SHAP explanations**")
st.markdown("*Built by Siyabonga Mlambo · MSc Data Science · Sol Plaatje University*")
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    uploaded = st.file_uploader(
        "Upload CSV dataset", type=['csv'],
        help="Upload any tabular CSV. The app will auto-detect features.")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ Loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
        target = st.selectbox("Select Target Column", df_raw.columns.tolist())
        drop_cols = st.multiselect("Drop Columns (IDs, etc.)",
                                   [c for c in df_raw.columns if c != target])
    else:
        st.info("No file uploaded. Using sample credit risk data.")
        target = 'default'
        drop_cols = []

    model_name = st.selectbox("Choose Model", [
        'XGBoost', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'])

    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    max_shap  = st.slider("Max SHAP Features to Display", 5, 20, 12)
    run_btn   = st.button("🚀 Train & Explain", use_container_width=True, type="primary")

# ── SAMPLE DATA ───────────────────────────────────────────────
def make_sample_data():
    np.random.seed(42)
    n = 2000
    cs     = np.random.normal(650, 80, n).clip(300, 850).astype(int)
    inc    = np.random.lognormal(11.5, 0.6, n).astype(int)
    loan   = np.random.lognormal(11.0, 0.7, n).astype(int)
    emp    = np.random.exponential(5, n).clip(0, 40).round(1)
    missed = np.random.poisson(0.3, n).clip(0, 10)
    dr     = (loan / inc).clip(0, 5).round(3)
    lo     = -2.0 + 0.015*(650-cs)/50 + 1.2*(dr>0.5) + 0.5*(missed>0) - 0.4*(emp>5)
    prob   = 1/(1+np.exp(-lo))
    default = (np.random.random(n) < prob).astype(int)
    return pd.DataFrame({'credit_score':cs,'income':inc,'loan_amount':loan,
                         'employment_years':emp,'missed_payments':missed,
                         'debt_ratio':dr,'default':default})

# ── PREPROCESSING ─────────────────────────────────────────────
def preprocess(df, target, drop_cols):
    df = df.drop(columns=drop_cols, errors='ignore').copy()
    df = df.dropna()
    for col in df.select_dtypes(include='object').columns:
        if col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    return X, pd.Series(y), features

# ── MODEL FACTORY ─────────────────────────────────────────────
def get_model(name):
    if name == 'XGBoost':
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             use_label_encoder=False, eval_metric='logloss',
                             random_state=42, n_jobs=-1)
    elif name == 'Random Forest':
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=200, random_state=42)
    else:
        return LogisticRegression(max_iter=1000, random_state=42)

# ── MAIN LOGIC ────────────────────────────────────────────────
if run_btn or True:  # Always show something
    if uploaded:
        df = df_raw
    else:
        df = make_sample_data()
        st.info("💡 Demo mode — using synthetic credit risk data. Upload your own CSV to analyse it.")

    with st.spinner(f"Training {model_name} model..."):
        X, y, features = preprocess(df, target, drop_cols)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ── METRICS ──────────────────────────────────────────────
    st.markdown("## 📊 Model Performance")
    auc = roc_auc_score(y_test, y_pred_proba)
    f1  = f1_score(y_test, y_pred, average='weighted')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{auc:.3f}</div><div class="metric-lbl">AUC-ROC</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{f1:.3f}</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
    with c3:
        acc = (y_pred == y_test).mean()
        st.markdown(f'<div class="metric-box"><div class="metric-val">{acc:.3f}</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{X_train.shape[0]:,}</div><div class="metric-lbl">Train Samples</div></div>', unsafe_allow_html=True)

    with st.expander("📋 Full Classification Report"):
        st.text(classification_report(y_test, y_pred))

    st.markdown("---")

    # ── SHAP ─────────────────────────────────────────────────
    st.markdown("## 🔍 SHAP Explainability")

    with st.spinner("Computing SHAP values..."):
        if model_name in ['XGBoost', 'Random Forest', 'Gradient Boosting']:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer   = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary (Beeswarm)", "📈 Global Importance", 
        "🔎 Single Prediction", "📋 SHAP Table"])

    with tab1:
        st.markdown("**How each feature affects predictions across all samples**")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0a0a0f')
        shap.summary_plot(shap_values, X_test, max_display=max_shap,
                          show=False, plot_size=None)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.markdown("**Average absolute SHAP value per feature (global importance)**")
        mean_shap = pd.Series(np.abs(shap_values).mean(axis=0),
                              index=features).nlargest(max_shap)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#1e1e30')
        ax.barh(mean_shap.index, mean_shap.values, color='#7c6ff7', alpha=0.85)
        ax.set_title('Global Feature Importance (Mean |SHAP|)', color='white')
        ax.set_xlabel('Mean |SHAP Value|', color='white')
        ax.tick_params(colors='white')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        idx = st.slider("Select sample index", 0, len(X_test) - 1, 0)
        shap_exp = shap.Explanation(
            values      = shap_values[idx],
            base_values = explainer.expected_value if not isinstance(
                              explainer.expected_value, np.ndarray)
                          else explainer.expected_value[1],
            data        = X_test.iloc[idx].values,
            feature_names = features)

        actual    = y_test.iloc[idx]
        predicted = y_pred_proba[idx]
        st.markdown(f"**Actual:** {actual} | **Predicted probability:** {predicted:.1%}")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0a0a0f')
        shap.plots.waterfall(shap_exp, max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        shap_df = pd.DataFrame({
            'Feature'   : features,
            'Value'     : X_test.iloc[0].values,
            'SHAP'      : shap_values[0].round(4),
            'Direction' : ['↑ Risk' if s > 0 else '↓ Risk' for s in shap_values[0]]
        }).sort_values('SHAP', key=abs, ascending=False)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        csv = shap_df.to_csv(index=False).encode()
        st.download_button("⬇ Download SHAP Table (CSV)", csv, "shap_values.csv", "text/csv")

    st.markdown("---")
    st.caption("Built by Siyabonga Mlambo · MSc Data Science · Sol Plaatje University · "
               "XAI Portfolio Project · github.com/syamlambo")
