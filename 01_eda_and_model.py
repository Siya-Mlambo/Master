# ============================================================
# PROJECT 1: Credit Risk Prediction with SHAP Explainability
# Author   : Siyabonga Mlambo
# Dataset  : UCI / Kaggle Credit Risk Dataset
# Goal     : Predict loan default + explain every prediction
# ============================================================

# ── INSTALL (run once) ──────────────────────────────────────
# pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn optuna imbalanced-learn streamlit plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── PLOTTING STYLE ──────────────────────────────────────────
plt.style.use('dark_background')
PURPLE = '#7c6ff7'
TEAL   = '#2dd4a8'
AMBER  = '#f5a623'
CORAL  = '#ff6b6b'
COLORS = [PURPLE, TEAL, AMBER, CORAL, '#a78bfa', '#34d399']

# ============================================================
# STEP 1 — GENERATE SYNTHETIC CREDIT DATA
# (Replace with: df = pd.read_csv('your_data.csv') for real data)
# ============================================================

np.random.seed(42)
n = 10000

def generate_credit_data(n):
    age             = np.random.randint(18, 75, n)
    income          = np.random.lognormal(11.5, 0.6, n).astype(int)
    loan_amount     = np.random.lognormal(11.0, 0.7, n).astype(int)
    credit_score    = np.random.normal(650, 80, n).clip(300, 850).astype(int)
    employment_years= np.random.exponential(5, n).clip(0, 40).round(1)
    num_accounts    = np.random.randint(1, 15, n)
    debt_ratio      = (loan_amount / income).clip(0, 5).round(3)
    missed_payments = np.random.poisson(0.3, n).clip(0, 10)
    loan_purpose    = np.random.choice(
        ['home', 'car', 'education', 'personal', 'business'], n,
        p=[0.3, 0.25, 0.15, 0.2, 0.1])
    employment_type = np.random.choice(
        ['employed', 'self_employed', 'unemployed', 'retired'], n,
        p=[0.6, 0.2, 0.1, 0.1])

    # Default probability influenced by features
    log_odds = (
        -2.0
        + 0.015 * (650 - credit_score) / 50
        + 1.2  * (debt_ratio > 0.5).astype(int)
        + 0.8  * (employment_type == 'unemployed').astype(int)
        + 0.5  * (missed_payments > 0).astype(int)
        + 0.3  * (missed_payments > 2).astype(int)
        - 0.4  * (employment_years > 5).astype(int)
        - 0.3  * (income > 500000).astype(int)
        + 0.2  * (age < 25).astype(int)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (np.random.random(n) < prob_default).astype(int)

    return pd.DataFrame({
        'age': age, 'income': income, 'loan_amount': loan_amount,
        'credit_score': credit_score, 'employment_years': employment_years,
        'num_accounts': num_accounts, 'debt_ratio': debt_ratio,
        'missed_payments': missed_payments, 'loan_purpose': loan_purpose,
        'employment_type': employment_type, 'default': default
    })

df = generate_credit_data(n)
print("=" * 60)
print("CREDIT RISK DATASET")
print("=" * 60)
print(f"Shape       : {df.shape}")
print(f"Default rate: {df['default'].mean():.1%}")
print(f"\n{df.head()}")
print(f"\n{df.describe().round(2)}")

# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Credit Risk — Exploratory Data Analysis',
             fontsize=16, color='white', fontweight='bold', y=1.01)

# 2a. Default rate
ax = axes[0, 0]
counts = df['default'].value_counts()
bars = ax.bar(['No Default', 'Default'], counts.values,
              color=[TEAL, CORAL], alpha=0.85, edgecolor='none')
ax.set_title('Default Distribution', color='white')
ax.set_ylabel('Count', color='white')
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}\n({count/n:.1%})', ha='center', color='white', fontsize=9)

# 2b. Credit score by default
ax = axes[0, 1]
for label, color in [(0, TEAL), (1, CORAL)]:
    ax.hist(df[df['default'] == label]['credit_score'], bins=40,
            alpha=0.6, color=color, label=['No Default', 'Default'][label], density=True)
ax.set_title('Credit Score by Default', color='white')
ax.set_xlabel('Credit Score', color='white')
ax.legend()

# 2c. Debt ratio distribution
ax = axes[0, 2]
for label, color in [(0, TEAL), (1, CORAL)]:
    data = df[df['default'] == label]['debt_ratio'].clip(0, 2)
    ax.hist(data, bins=40, alpha=0.6, color=color,
            label=['No Default', 'Default'][label], density=True)
ax.set_title('Debt Ratio by Default', color='white')
ax.set_xlabel('Debt Ratio', color='white')
ax.legend()

# 2d. Income distribution (log)
ax = axes[0, 3]
for label, color in [(0, TEAL), (1, CORAL)]:
    ax.hist(np.log(df[df['default'] == label]['income']), bins=40,
            alpha=0.6, color=color, label=['No Default', 'Default'][label], density=True)
ax.set_title('Log Income by Default', color='white')
ax.legend()

# 2e. Default rate by employment type
ax = axes[1, 0]
emp_default = df.groupby('employment_type')['default'].mean().sort_values(ascending=False)
ax.bar(emp_default.index, emp_default.values, color=PURPLE, alpha=0.85, edgecolor='none')
ax.set_title('Default Rate by Employment', color='white')
ax.set_ylabel('Default Rate', color='white')
ax.tick_params(axis='x', rotation=30)
for i, v in enumerate(emp_default.values):
    ax.text(i, v + 0.005, f'{v:.1%}', ha='center', color='white', fontsize=9)

# 2f. Default rate by loan purpose
ax = axes[1, 1]
purpose_default = df.groupby('loan_purpose')['default'].mean().sort_values(ascending=False)
ax.bar(purpose_default.index, purpose_default.values, color=AMBER, alpha=0.85, edgecolor='none')
ax.set_title('Default Rate by Loan Purpose', color='white')
ax.set_ylabel('Default Rate', color='white')
ax.tick_params(axis='x', rotation=30)

# 2g. Missed payments vs default
ax = axes[1, 2]
missed_default = df.groupby('missed_payments')['default'].mean()
ax.bar(missed_default.index, missed_default.values, color=CORAL, alpha=0.85, edgecolor='none')
ax.set_title('Default Rate by Missed Payments', color='white')
ax.set_xlabel('Missed Payments', color='white')
ax.set_ylabel('Default Rate', color='white')

# 2h. Correlation heatmap
ax = axes[1, 3]
num_cols = ['age', 'income', 'loan_amount', 'credit_score',
            'employment_years', 'debt_ratio', 'missed_payments', 'default']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, cmap='RdPu', annot=True,
            fmt='.2f', annot_kws={'size': 7}, linewidths=0.5, cbar=False)
ax.set_title('Correlation Matrix', color='white')

plt.tight_layout()
plt.savefig('reports/eda_overview.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f', edgecolor='none')
plt.show()
print("✅ EDA plots saved to reports/eda_overview.png")

# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    # Ratio features
    df['loan_to_income_ratio']   = df['loan_amount'] / df['income']
    df['income_per_year_employed'] = df['income'] / (df['employment_years'] + 1)
    df['accounts_per_year']      = df['num_accounts'] / (df['age'] - 17)

    # Risk buckets
    df['credit_score_bucket'] = pd.cut(
        df['credit_score'],
        bins=[0, 579, 669, 739, 799, 850],
        labels=['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])

    df['income_bucket'] = pd.cut(
        df['income'],
        bins=[0, 150000, 300000, 600000, 1000000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    df['debt_risk_flag']       = (df['debt_ratio'] > 0.5).astype(int)
    df['multiple_missed_flag'] = (df['missed_payments'] > 1).astype(int)
    df['young_borrower_flag']  = (df['age'] < 25).astype(int)
    df['high_income_flag']     = (df['income'] > 600000).astype(int)
    df['long_employed_flag']   = (df['employment_years'] > 7).astype(int)

    # Encode categoricals
    le = LabelEncoder()
    for col in ['loan_purpose', 'employment_type',
                'credit_score_bucket', 'income_bucket']:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

df_fe = engineer_features(df)
print(f"\n✅ Features after engineering: {df_fe.shape[1]} columns")
print(df_fe.dtypes)

# ============================================================
# STEP 4 — TRAIN / TEST SPLIT + SMOTE
# ============================================================

FEATURES = [c for c in df_fe.columns if c != 'default']
X = df_fe[FEATURES]
y = df_fe['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size : {X_train.shape[0]:,}")
print(f"Test size  : {X_test.shape[0]:,}")
print(f"Train default rate: {y_train.mean():.1%}")

# Apply SMOTE to training set only
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Train size: {X_train_sm.shape[0]:,}")
print(f"After SMOTE — Default rate: {y_train_sm.mean():.1%}")

# ============================================================
# STEP 5 — HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================

def objective(trial):
    params = {
        'n_estimators'      : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'         : trial.suggest_int('max_depth', 3, 9),
        'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'         : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree'  : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight'  : trial.suggest_int('min_child_weight', 1, 10),
        'gamma'             : trial.suggest_float('gamma', 0, 5),
        'reg_alpha'         : trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda'        : trial.suggest_float('reg_lambda', 0, 2),
        'use_label_encoder' : False,
        'eval_metric'       : 'auc',
        'random_state'      : 42,
        'n_jobs'            : -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_sm, y_train_sm,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

print("\n⚡ Running Optuna hyperparameter tuning (50 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=False)
print(f"✅ Best AUC: {study.best_value:.4f}")
print(f"   Best params: {study.best_params}")

# ============================================================
# STEP 6 — TRAIN FINAL MODEL
# ============================================================

best_params = study.best_params
best_params.update({'use_label_encoder': False, 'eval_metric': 'auc',
                    'random_state': 42, 'n_jobs': -1})

model = XGBClassifier(**best_params)
model.fit(X_train_sm, y_train_sm,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred      = model.predict(X_test)
y_pred_proba= model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"AUC-ROC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Avg Prec  : {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

# ============================================================
# STEP 7 — VISUALISE PERFORMANCE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Performance', fontsize=14, color='white', fontweight='bold')

# Confusion matrix
ax = axes[0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            linewidths=1, linecolor='#0a0a0f')
ax.set_title('Confusion Matrix', color='white')
ax.set_ylabel('Actual', color='white')
ax.set_xlabel('Predicted', color='white')

# ROC curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score   = roc_auc_score(y_test, y_pred_proba)
ax.plot(fpr, tpr, color=PURPLE, lw=2.5, label=f'AUC = {auc_score:.4f}')
ax.plot([0,1], [0,1], 'w--', lw=1, alpha=0.4, label='Random')
ax.fill_between(fpr, tpr, alpha=0.1, color=PURPLE)
ax.set_title('ROC Curve', color='white')
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate', color='white')
ax.legend(loc='lower right')

# Feature importance
ax = axes[2]
fi = pd.Series(model.feature_importances_, index=FEATURES).nlargest(10)
ax.barh(fi.index, fi.values, color=TEAL, alpha=0.85, edgecolor='none')
ax.set_title('Top 10 Feature Importance', color='white')
ax.set_xlabel('Importance Score', color='white')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('reports/model_performance.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f', edgecolor='none')
plt.show()
print("✅ Performance plots saved to reports/model_performance.png")

# ============================================================
# STEP 8 — SHAP EXPLAINABILITY
# ============================================================

print("\n⚡ Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 8a. SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type='beeswarm',
                  max_display=15, show=False,
                  plot_size=(12, 8))
plt.title('SHAP Summary — Feature Impact on Default Prediction',
          fontsize=13, color='white', pad=15)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.show()

# 8b. SHAP Waterfall for a single high-risk prediction
high_risk_idx = np.where(y_pred_proba > 0.75)[0]
if len(high_risk_idx) > 0:
    idx = high_risk_idx[0]
    shap_exp = shap.Explanation(
        values      = shap_values[idx],
        base_values = explainer.expected_value,
        data        = X_test.iloc[idx].values,
        feature_names = FEATURES)
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_exp, max_display=12, show=False)
    plt.title(f'SHAP Waterfall — High Risk Applicant\n'
              f'Default Probability: {y_pred_proba[idx]:.1%}',
              fontsize=12, color='white')
    plt.tight_layout()
    plt.savefig('reports/shap_waterfall_high_risk.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0a0f')
    plt.show()
    print(f"✅ SHAP waterfall saved (applicant default prob: {y_pred_proba[idx]:.1%})")

# 8c. SHAP Bar Plot (global)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type='bar',
                  max_display=12, show=False)
plt.title('SHAP Global Feature Importance', fontsize=12, color='white')
plt.tight_layout()
plt.savefig('reports/shap_global_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0f')
plt.show()

print("\n" + "=" * 60)
print("✅ PROJECT 1 COMPLETE — Credit Risk XAI")
print("=" * 60)
print(f"  AUC-ROC       : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"  F1 Score      : {f1_score(y_test, y_pred):.4f}")
print(f"  Reports saved : reports/")
print("  Next step     : Run streamlit_app.py for the live demo")
