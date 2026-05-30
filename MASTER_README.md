# 🗂️ Siyabonga Mlambo — Data Science Project Portfolio

> *"A model no one understands is a model no one trusts."*

**MSc Candidate · Sol Plaatje University · Data Scientist · XAI Researcher**

---

## 📁 Project Index

| # | Project | Category | Key Skills | Status |
|---|---------|----------|-----------|--------|
| 1 | [Credit Risk XAI](#1-credit-risk-prediction--shap-explainability) | Machine Learning | XGBoost, SHAP, Streamlit | ✅ Complete |
| 2 | [Customer Churn Prediction](#2-customer-churn-prediction) | Machine Learning | Random Forest, SMOTE | ✅ Complete |
| 3 | [Fraud Detection System](#3-fraud-detection-system) | Machine Learning | Isolation Forest, XGBoost | ✅ Complete |
| 4 | [Customer Analytics SQL](#4-customer-analytics-sql-pipeline) | SQL & Analytics | PostgreSQL, RFM, CLV | ✅ Complete |
| 5 | [Sales Forecasting](#5-sales-forecasting-with-prophet) | Time Series | Prophet, SARIMA | ✅ Complete |
| 6 | [ML Explainability App](#6-ml-explainability-streamlit-app) | App / Deployment | Streamlit, SHAP, Any CSV | ✅ Complete |

---

## 1. Credit Risk Prediction + SHAP Explainability

**Problem:** Banks lose billions to undetected loan defaults. Existing models lack transparency.

**Solution:** XGBoost pipeline with full SHAP explainability per prediction.

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.940** |
| Precision | **91.2%** |
| Recall | **88.7%** |
| F1 Score | **0.899** |

**Stack:** Python · XGBoost · SHAP · SMOTE · Optuna · Streamlit · Plotly

**Files:**
```
credit-risk-xai/
├── notebooks/01_eda_and_model.py    ← Full pipeline
├── app/streamlit_app.py             ← Live demo
└── reports/                         ← All charts
```

**Run:**
```bash
pip install -r requirements.txt
python notebooks/01_eda_and_model.py
streamlit run app/streamlit_app.py
```

---

## 2. Customer Churn Prediction

**Problem:** Telecom company losing high-value customers with no early warning system.

**Solution:** Random Forest churn model with SMOTE, deployed as a Streamlit app.

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.930** |
| F1 Score | **0.890** |
| Recall | **88.0%** |
| Precision | **90.0%** |

**Stack:** Python · Scikit-learn · Random Forest · SMOTE · Streamlit

**Files:**
```
churn-prediction/
├── notebooks/churn_model.py         ← Full model pipeline
└── reports/                         ← EDA + performance plots
```

**Key insight:** Contract type and tenure were the strongest predictors. Month-to-month customers churned at 4x the rate of two-year contracts.

---

## 3. Fraud Detection System

**Problem:** Financial institution facing $2M+ monthly in undetected fraudulent transactions.

**Solution:** Ensemble of Isolation Forest (unsupervised) + XGBoost with velocity features.

| Metric | Score |
|--------|-------|
| Precision | **97.0%** |
| Recall | **85.0%** |
| F1 Score | **0.906** |
| False Positive Rate | **0.3%** |

**Stack:** Python · XGBoost · Isolation Forest · Scikit-learn · Optimised Threshold

**Files:**
```
fraud-detection/
├── notebooks/fraud_detection.py     ← Full detection pipeline
└── reports/                         ← PR curve + confusion matrix
```

**Key insight:** Transaction velocity features (activity patterns) were the most discriminative signals.

---

## 4. Customer Analytics SQL Pipeline

**Problem:** Marketing running campaigns on the entire customer base with 1.2% conversion.

**Solution:** Full RFM segmentation + CLV modelling + cohort retention analysis in PostgreSQL.

**Deliverables:**
- 8 RFM customer segments with actionable marketing strategies
- 12-month CLV prediction per customer
- Monthly cohort retention curves
- Product affinity / cross-sell analysis
- YoY revenue growth tracking

**Stack:** PostgreSQL · Advanced SQL (CTEs, Window Functions, NTILE) · dbt

**Files:**
```
customer-analytics-sql/
└── notebooks/customer_analytics.sql  ← Complete SQL pipeline
```

**Key insight:** Top 5% of customers (Champions segment) drove 43% of total revenue.

---

## 5. Sales Forecasting with Prophet

**Problem:** Retail business making inventory decisions on gut feel — 18% overstock + 12% stockouts.

**Solution:** Prophet model with custom holiday effects, 90-day horizon, confidence intervals.

| Metric | Score |
|--------|-------|
| MAPE | **4.2%** |
| Forecast Horizon | **90 days** |
| Product Categories | **5** |
| Stores | **15** |

**Stack:** Python · Prophet · SARIMA · Statsmodels · Matplotlib

**Files:**
```
sales-forecasting/
├── notebooks/sales_forecasting.py   ← Full forecasting pipeline
└── reports/                         ← Forecast + decomposition plots
```

**Key insight:** Prophet outperformed SARIMA on categories with strong weekly seasonality. Black Friday accounted for 31% of Q4 variance.

---

## 6. ML Explainability Streamlit App

**Problem:** Data scientists and analysts struggle to explain ML models to non-technical stakeholders.

**Solution:** Upload any CSV → auto-train 4 models → full SHAP explanation suite.

**Features:**
- Upload any CSV classification dataset
- Choose from 4 ML algorithms
- SHAP beeswarm, bar, and waterfall plots
- Single-prediction explanation with slider
- Download SHAP table as CSV

**Stack:** Streamlit · SHAP · XGBoost · Random Forest · Scikit-learn · Plotly

**Files:**
```
ml-explainability-app/
└── app/ml_explainability_app.py     ← Full Streamlit app
```

**Run:**
```bash
streamlit run ml-explainability-app/app/ml_explainability_app.py
```

---

## 🚀 Quick Start — Run All Projects

```bash
# Clone the repo
git clone https://github.com/syamlambo/data-science-portfolio
cd data-science-portfolio

# Install all dependencies
pip install -r requirements.txt

# Project 1 — Credit Risk (model training)
python projects/credit-risk-xai/notebooks/01_eda_and_model.py

# Project 1 — Credit Risk (Streamlit app)
streamlit run projects/credit-risk-xai/app/streamlit_app.py

# Project 2 — Churn
python projects/churn-prediction/notebooks/churn_model.py

# Project 3 — Fraud Detection
python projects/fraud-detection/notebooks/fraud_detection.py

# Project 5 — Sales Forecasting
python projects/sales-forecasting/notebooks/sales_forecasting.py

# Project 6 — ML Explainability App
streamlit run projects/ml-explainability-app/app/ml_explainability_app.py

# Project 4 — SQL (requires PostgreSQL)
# psql -U your_user -d your_db -f projects/customer-analytics-sql/notebooks/customer_analytics.sql
```

---

## 📫 Contact

| | |
|--|--|
| **Email** | mrfaceoff01@gmail.com |
| **LinkedIn** | linkedin.com/in/syamlambo |
| **Portfolio** | syamlambo.netlify.app |
| **University** | Sol Plaatje University |

---

*Built by Siyabonga Mlambo · MSc Computer & Information Science (Data Science) · 2024*
