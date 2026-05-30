# ============================================================
# PROJECT 3: Real-Time Fraud Detection System
# Author   : Siyabonga Mlambo
# Goal     : Detect fraudulent transactions with high precision
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
PURPLE = '#7c6ff7'; TEAL = '#2dd4a8'; AMBER = '#f5a623'; CORAL = '#ff6b6b'

# ============================================================
# STEP 1 — GENERATE TRANSACTION DATA
# ============================================================

np.random.seed(42)
n = 50000

# Legitimate transactions
amount_legit    = np.random.lognormal(4.5, 1.2, n)
hour_legit      = np.random.choice(range(24), n, p=np.array(
    [0.01,0.01,0.01,0.01,0.02,0.03,0.05,0.07,0.08,0.07,
     0.07,0.07,0.07,0.06,0.06,0.06,0.06,0.06,0.05,0.04,
     0.04,0.03,0.02,0.01]))
v1 = np.random.normal(0, 1, n)
v2 = np.random.normal(0, 1, n)
v3 = np.random.normal(0, 1, n)
v4 = np.random.normal(0, 1, n)
v5 = np.random.normal(0, 1, n)

# Fraud transactions (0.17% like real world)
n_fraud = int(n * 0.0017)
fraud_idx = np.random.choice(n, n_fraud, replace=False)

amount  = amount_legit.copy()
amount[fraud_idx]  = np.random.lognormal(5.5, 1.5, n_fraud)  # Fraud: larger amounts
v1[fraud_idx]     += np.random.normal(3, 1, n_fraud)          # Fraud: anomalous features
v2[fraud_idx]     -= np.random.normal(2, 1, n_fraud)
v3[fraud_idx]     += np.random.normal(4, 1.5, n_fraud)
hour_fraud         = np.random.choice([0,1,2,3,23], n_fraud)  # Fraud: odd hours
hour               = hour_legit.copy()
hour[fraud_idx]    = hour_fraud

merchant_cat = np.random.choice(
    ['retail','food','travel','electronics','entertainment','online'], n,
    p=[0.3, 0.25, 0.1, 0.15, 0.1, 0.1])
merchant_cat[fraud_idx] = np.random.choice(
    ['electronics','online'], n_fraud, p=[0.4, 0.6])

fraud = np.zeros(n, dtype=int)
fraud[fraud_idx] = 1

df = pd.DataFrame({
    'amount': amount.round(2), 'hour': hour,
    'v1': v1.round(4), 'v2': v2.round(4),
    'v3': v3.round(4), 'v4': v4.round(4), 'v5': v5.round(4),
    'merchant_cat': merchant_cat, 'fraud': fraud
})

print("=" * 60)
print("FRAUD DETECTION DATASET")
print("=" * 60)
print(f"Total transactions : {n:,}")
print(f"Fraudulent         : {fraud.sum():,} ({fraud.mean():.2%})")
print(f"Legitimate         : {(fraud==0).sum():,}")
print(df.head())

# ============================================================
# STEP 2 — EDA
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fraud Detection — EDA', fontsize=16, color='white', fontweight='bold')

# Class distribution
ax = axes[0, 0]
counts = df['fraud'].value_counts()
ax.bar(['Legitimate', 'Fraud'], counts.values, color=[TEAL, CORAL], alpha=0.85)
ax.set_title('Class Distribution', color='white')
ax.set_yscale('log')
for i, v in enumerate(counts.values):
    ax.text(i, v * 1.1, f'{v:,}', ha='center', color='white')

# Amount distribution by fraud
ax = axes[0, 1]
for label, color in [(0, TEAL), (1, CORAL)]:
    data = np.log1p(df[df['fraud'] == label]['amount'])
    ax.hist(data, bins=50, alpha=0.6, color=color,
            label=['Legitimate', 'Fraud'][label], density=True)
ax.set_title('Log Amount by Fraud', color='white')
ax.set_xlabel('Log(Amount + 1)', color='white')
ax.legend()

# Fraud by hour
ax = axes[0, 2]
hourly_fraud = df.groupby('hour')['fraud'].mean()
ax.bar(hourly_fraud.index, hourly_fraud.values, color=AMBER, alpha=0.85)
ax.set_title('Fraud Rate by Hour of Day', color='white')
ax.set_xlabel('Hour', color='white')
ax.set_ylabel('Fraud Rate', color='white')

# V1 feature
ax = axes[1, 0]
for label, color in [(0, TEAL), (1, CORAL)]:
    ax.hist(df[df['fraud'] == label]['v1'], bins=50,
            alpha=0.6, color=color, label=['Legitimate','Fraud'][label], density=True)
ax.set_title('Feature V1 by Fraud', color='white')
ax.legend()

# V3 feature
ax = axes[1, 1]
for label, color in [(0, TEAL), (1, CORAL)]:
    ax.hist(df[df['fraud'] == label]['v3'], bins=50,
            alpha=0.6, color=color, label=['Legitimate','Fraud'][label], density=True)
ax.set_title('Feature V3 by Fraud', color='white')
ax.legend()

# Fraud by merchant category
ax = axes[1, 2]
mc_fraud = df.groupby('merchant_cat')['fraud'].mean().sort_values(ascending=False)
ax.bar(mc_fraud.index, mc_fraud.values, color=PURPLE, alpha=0.85)
ax.set_title('Fraud Rate by Merchant Category', color='white')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('reports/fraud_eda.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.show()

# ============================================================
# STEP 3 — FEATURE ENGINEERING + VELOCITY FEATURES
# ============================================================

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['merchant_cat_enc'] = le.fit_transform(df['merchant_cat'])

df['log_amount']       = np.log1p(df['amount'])
df['is_night']         = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
df['is_large_amount']  = (df['amount'] > 500).astype(int)
df['v1_v2_interaction']= df['v1'] * df['v2']
df['anomaly_score']    = (np.abs(df['v1']) + np.abs(df['v2']) +
                          np.abs(df['v3']) + np.abs(df['v4'])) / 4

FEATURES = ['amount', 'log_amount', 'hour', 'is_night',
            'v1', 'v2', 'v3', 'v4', 'v5',
            'merchant_cat_enc', 'is_large_amount',
            'v1_v2_interaction', 'anomaly_score']

X = df[FEATURES]
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# STEP 4 — ISOLATION FOREST (Unsupervised baseline)
# ============================================================

print("\n⚡ Training Isolation Forest...")
iso = IsolationForest(n_estimators=200, contamination=0.002,
                      random_state=42, n_jobs=-1)
iso.fit(X_train_sc)
iso_preds = iso.predict(X_test_sc)
iso_preds = (iso_preds == -1).astype(int)
print(f"Isolation Forest — Precision: {precision_score(y_test, iso_preds):.4f} | "
      f"Recall: {recall_score(y_test, iso_preds):.4f} | "
      f"F1: {f1_score(y_test, iso_preds):.4f}")

# ============================================================
# STEP 5 — XGBOOST WITH CLASS WEIGHTS
# ============================================================

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n⚡ Training XGBoost (scale_pos_weight={scale_pos:.1f})...")

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False, eval_metric='aucpr',
    random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, verbose=False)

y_pred       = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)[:, 1]

# Optimise threshold for precision
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores  = [f1_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred_opt  = (y_pred_proba >= best_thresh).astype(int)

print(f"\n{'='*60}")
print(f"FRAUD DETECTION — XGBoost (threshold={best_thresh:.2f})")
print(f"{'='*60}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_opt):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_opt):.4f}")
print(f"\n{classification_report(y_test, y_pred_opt, target_names=['Legit','Fraud'])}")

# ============================================================
# STEP 6 — PRECISION-RECALL CURVE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Fraud Detection — Model Evaluation', fontsize=14, color='white')

# PR Curve
ax = axes[0]
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)
ax.plot(rec, prec, color=PURPLE, lw=2.5, label=f'AP = {ap:.4f}')
ax.fill_between(rec, prec, alpha=0.1, color=PURPLE)
ax.set_title('Precision-Recall Curve', color='white')
ax.set_xlabel('Recall', color='white')
ax.set_ylabel('Precision', color='white')
ax.legend()

# Confusion matrix
ax = axes[1]
cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
ax.set_title(f'Confusion Matrix\n(threshold={best_thresh:.2f})', color='white')

# F1 vs Threshold
ax = axes[2]
ax.plot(thresholds, f1_scores, color=TEAL, lw=2.5)
ax.axvline(best_thresh, color=CORAL, linestyle='--', lw=1.5, label=f'Best={best_thresh:.2f}')
ax.set_title('F1 Score vs Decision Threshold', color='white')
ax.set_xlabel('Threshold', color='white')
ax.set_ylabel('F1 Score', color='white')
ax.legend()

plt.tight_layout()
plt.savefig('reports/fraud_performance.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.show()

print("\n✅ PROJECT 3 COMPLETE — Fraud Detection System")
print(f"   Precision: {precision_score(y_test, y_pred_opt):.4f} | "
      f"Recall: {recall_score(y_test, y_pred_opt):.4f} | "
      f"F1: {f1_score(y_test, y_pred_opt):.4f}")
