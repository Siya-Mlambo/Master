# ============================================================
# PROJECT 2: Customer Churn Prediction
# Author   : Siyabonga Mlambo
# Goal     : Predict telecom customer churn + deploy Streamlit
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
PURPLE = '#7c6ff7'; TEAL = '#2dd4a8'; AMBER = '#f5a623'; CORAL = '#ff6b6b'

# ============================================================
# STEP 1 — GENERATE TELECOM CHURN DATA
# ============================================================

np.random.seed(42)
n = 7000

tenure          = np.random.exponential(30, n).clip(1, 72).round()
monthly_charges = np.random.normal(65, 25, n).clip(18, 120).round(2)
total_charges   = (tenure * monthly_charges * np.random.normal(1, 0.05, n)).clip(0).round(2)
num_services    = np.random.randint(1, 9, n)
contract_type   = np.random.choice(['Month-to-month','One year','Two year'], n,
                                    p=[0.55, 0.25, 0.20])
internet_service= np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20])
payment_method  = np.random.choice(
    ['Electronic check','Mailed check','Bank transfer','Credit card'], n)
senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
support_calls   = np.random.poisson(1.2, n).clip(0, 10)
tech_support    = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
online_backup   = np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65])
gender          = np.random.choice(['Male', 'Female'], n)

log_odds_churn = (
    -1.5
    + 1.5  * (contract_type == 'Month-to-month').astype(int)
    - 0.8  * (contract_type == 'Two year').astype(int)
    + 0.6  * (internet_service == 'Fiber optic').astype(int)
    - 0.04 * tenure
    + 0.02 * monthly_charges
    + 0.4  * (support_calls > 3).astype(int)
    + 0.3  * senior_citizen
    - 0.3  * (tech_support == 'Yes').astype(int)
    + 0.2  * (payment_method == 'Electronic check').astype(int)
)
churn_prob = 1 / (1 + np.exp(-log_odds_churn))
churn      = (np.random.random(n) < churn_prob).astype(int)

df = pd.DataFrame({
    'tenure': tenure, 'monthly_charges': monthly_charges,
    'total_charges': total_charges, 'num_services': num_services,
    'contract_type': contract_type, 'internet_service': internet_service,
    'payment_method': payment_method, 'senior_citizen': senior_citizen,
    'support_calls': support_calls, 'tech_support': tech_support,
    'online_backup': online_backup, 'gender': gender, 'churn': churn
})

print("=" * 60)
print("TELECOM CHURN DATASET")
print("=" * 60)
print(f"Shape      : {df.shape}")
print(f"Churn rate : {df['churn'].mean():.1%}")
print(df.head())

# ============================================================
# STEP 2 — EDA
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Customer Churn — EDA', fontsize=16, color='white', fontweight='bold')

# Churn distribution
ax = axes[0, 0]
counts = df['churn'].value_counts()
ax.bar(['Retained', 'Churned'], counts.values, color=[TEAL, CORAL], alpha=0.85)
ax.set_title('Churn Distribution', color='white')
for i, (v, pct) in enumerate(zip(counts.values, counts.values / n)):
    ax.text(i, v + 30, f'{v:,}\n({pct:.1%})', ha='center', color='white', fontsize=9)

# Tenure by churn
ax = axes[0, 1]
for label, color in [(0, TEAL), (1, CORAL)]:
    ax.hist(df[df['churn'] == label]['tenure'], bins=30,
            alpha=0.6, color=color, label=['Retained', 'Churned'][label], density=True)
ax.set_title('Tenure Distribution by Churn', color='white')
ax.set_xlabel('Months', color='white')
ax.legend()

# Monthly charges by churn
ax = axes[0, 2]
df.boxplot(column='monthly_charges', by='churn', ax=ax,
           boxprops=dict(color=PURPLE), medianprops=dict(color=TEAL),
           whiskerprops=dict(color='white'), capprops=dict(color='white'),
           flierprops=dict(marker='o', color=CORAL, alpha=0.3))
ax.set_title('Monthly Charges by Churn', color='white')
ax.set_xlabel('Churn (0=No, 1=Yes)', color='white')

# Contract type churn rate
ax = axes[1, 0]
ct_churn = df.groupby('contract_type')['churn'].mean().sort_values(ascending=False)
bars = ax.bar(ct_churn.index, ct_churn.values, color=[CORAL, AMBER, TEAL], alpha=0.85)
ax.set_title('Churn Rate by Contract Type', color='white')
ax.set_ylabel('Churn Rate', color='white')
ax.tick_params(axis='x', rotation=15)
for bar, v in zip(bars, ct_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.1%}',
            ha='center', color='white', fontsize=9)

# Support calls vs churn
ax = axes[1, 1]
sc_churn = df.groupby('support_calls')['churn'].mean()
ax.bar(sc_churn.index, sc_churn.values, color=AMBER, alpha=0.85)
ax.set_title('Churn Rate by Support Calls', color='white')
ax.set_xlabel('Number of Support Calls', color='white')
ax.set_ylabel('Churn Rate', color='white')

# Internet service churn rate
ax = axes[1, 2]
is_churn = df.groupby('internet_service')['churn'].mean().sort_values(ascending=False)
ax.bar(is_churn.index, is_churn.values, color=PURPLE, alpha=0.85)
ax.set_title('Churn Rate by Internet Service', color='white')
ax.set_ylabel('Churn Rate', color='white')

plt.tight_layout()
plt.savefig('reports/churn_eda.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.show()
print("✅ EDA saved")

# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

def engineer_churn_features(df):
    df = df.copy()
    le = LabelEncoder()
    for col in ['contract_type', 'internet_service', 'payment_method',
                'tech_support', 'online_backup', 'gender']:
        df[col] = le.fit_transform(df[col])

    df['charges_per_service']   = df['monthly_charges'] / (df['num_services'] + 1)
    df['avg_monthly_total']     = df['total_charges'] / (df['tenure'] + 1)
    df['high_support_calls']    = (df['support_calls'] > 3).astype(int)
    df['new_customer']          = (df['tenure'] < 6).astype(int)
    df['loyal_customer']        = (df['tenure'] > 36).astype(int)
    df['high_charges']          = (df['monthly_charges'] > 80).astype(int)
    return df

df_fe = engineer_churn_features(df)
FEATURES = [c for c in df_fe.columns if c != 'churn']

X = df_fe[FEATURES]
y = df_fe['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ============================================================
# STEP 4 — MODEL COMPARISON
# ============================================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=200, random_state=42),
}

print("\n" + "=" * 60)
print("MODEL COMPARISON (5-fold CV AUC)")
print("=" * 60)

results = {}
for name, mdl in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(mdl, X_train_sm, y_train_sm,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    results[name] = scores
    print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Train best model (Random Forest)
best_model = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced')
best_model.fit(X_train_sm, y_train_sm)

y_pred       = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\n{'='*60}")
print("FINAL MODEL PERFORMANCE — Random Forest")
print(f"{'='*60}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

# Feature importance
fi = pd.Series(best_model.feature_importances_, index=FEATURES).nlargest(10)
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#1e1e30')
ax.barh(fi.index, fi.values, color=PURPLE, alpha=0.85, edgecolor='none')
ax.set_title('Top 10 Feature Importance — Churn Model', color='white', fontsize=12)
ax.set_xlabel('Importance', color='white')
ax.tick_params(colors='white')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('reports/churn_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0f')
plt.show()

print("\n✅ PROJECT 2 COMPLETE — Customer Churn Prediction")
print(f"   AUC: {roc_auc_score(y_test, y_pred_proba):.4f} | F1: {f1_score(y_test, y_pred):.4f}")
print("   Next: run app/streamlit_app.py for live demo")
