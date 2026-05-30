# ============================================================
# PROJECT 5: Sales Forecasting with Prophet & SARIMA
# Author   : Siyabonga Mlambo
# Goal     : 90-day revenue forecast with confidence intervals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
PURPLE = '#7c6ff7'; TEAL = '#2dd4a8'; AMBER = '#f5a623'; CORAL = '#ff6b6b'

# ============================================================
# STEP 1 — GENERATE REALISTIC SALES TIME SERIES
# ============================================================

np.random.seed(42)

dates = pd.date_range(start='2021-01-01', end='2024-06-30', freq='D')
n = len(dates)

# Trend component
trend = np.linspace(50000, 95000, n)

# Yearly seasonality (peaks Dec, dips Jan-Feb)
day_of_year  = np.array([d.timetuple().tm_yday for d in dates])
yearly_seas  = (15000 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
                + 8000 * np.sin(4 * np.pi * day_of_year / 365))

# Weekly seasonality (higher on weekdays)
day_of_week  = np.array([d.weekday() for d in dates])
weekly_seas  = np.where(day_of_week < 5, 5000, -8000)

# Black Friday / Christmas events
holiday_boost = np.zeros(n)
for i, d in enumerate(dates):
    if d.month == 11 and d.day >= 24 and d.day <= 30:   # Black Friday week
        holiday_boost[i] = 35000
    if d.month == 12 and d.day >= 15 and d.day <= 31:   # Christmas season
        holiday_boost[i] = 25000
    if d.month == 1 and d.day <= 10:                     # January slump
        holiday_boost[i] = -12000

# Noise
noise = np.random.normal(0, 4000, n)

sales = (trend + yearly_seas + weekly_seas + holiday_boost + noise).clip(5000)

df = pd.DataFrame({'ds': dates, 'y': sales.round(2)})

# Add product categories
for cat, base_mult in [('electronics', 0.35), ('clothing', 0.25),
                        ('home', 0.20), ('food', 0.15), ('other', 0.05)]:
    df[cat] = (df['y'] * base_mult * np.random.normal(1, 0.08, n)).round(2).clip(0)

print("=" * 60)
print("SALES FORECASTING DATASET")
print("=" * 60)
print(f"Date range : {df['ds'].min().date()} → {df['ds'].max().date()}")
print(f"Total days : {len(df):,}")
print(f"Avg daily  : R {df['y'].mean():,.0f}")
print(f"Max daily  : R {df['y'].max():,.0f}")
print(df.head())

# ============================================================
# STEP 2 — TIME SERIES DECOMPOSITION + STATIONARITY
# ============================================================

# Resample to monthly for cleaner visualisation
monthly = df.set_index('ds').resample('M')['y'].sum().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Sales Time Series — Decomposition & EDA',
             fontsize=16, color='white', fontweight='bold')

# Monthly revenue
ax = axes[0, 0]
ax.plot(monthly['ds'], monthly['y'] / 1e6, color=PURPLE, lw=2)
ax.fill_between(monthly['ds'], monthly['y'] / 1e6, alpha=0.15, color=PURPLE)
ax.set_title('Monthly Revenue (Millions R)', color='white')
ax.set_ylabel('Revenue (R Millions)', color='white')

# Rolling stats
ax = axes[0, 1]
roll = df.set_index('ds')['y'].rolling(30)
ax.plot(df['ds'], df['y'] / 1000, color='white', alpha=0.3, lw=0.5, label='Daily')
ax.plot(df['ds'], roll.mean() / 1000, color=TEAL, lw=2, label='30-day MA')
ax.plot(df['ds'], roll.std() / 1000, color=AMBER, lw=1.5, linestyle='--', label='30-day Std')
ax.set_title('Rolling Mean & Std (Daily Sales, R000s)', color='white')
ax.legend()

# Day of week pattern
ax = axes[1, 0]
df['dow'] = df['ds'].dt.day_name()
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_avg   = df.groupby('dow')['y'].mean()[dow_order]
bars = ax.bar(range(7), dow_avg.values / 1000, color=[PURPLE]*5 + [CORAL]*2, alpha=0.85)
ax.set_xticks(range(7))
ax.set_xticklabels([d[:3] for d in dow_order], color='white')
ax.set_title('Average Sales by Day of Week (R000s)', color='white')

# Monthly seasonality
ax = axes[1, 1]
df['month'] = df['ds'].dt.month
month_avg = df.groupby('month')['y'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.bar(range(1, 13), month_avg.values / 1000, color=AMBER, alpha=0.85)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, rotation=30, color='white')
ax.set_title('Average Sales by Month (R000s)', color='white')

plt.tight_layout()
plt.savefig('reports/sales_eda.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.show()

# ============================================================
# STEP 3 — ADF STATIONARITY TEST
# ============================================================

adf_result = adfuller(df['y'])
print(f"\nADF Stationarity Test:")
print(f"  ADF Statistic : {adf_result[0]:.4f}")
print(f"  p-value       : {adf_result[1]:.4f}")
print(f"  Stationary    : {'Yes ✅' if adf_result[1] < 0.05 else 'No ❌ (differencing needed)'}")

# ============================================================
# STEP 4 — PROPHET FORECASTING
# ============================================================

# Train/test split (last 90 days = test)
train_df = df[df['ds'] < '2024-04-01'][['ds', 'y']]
test_df  = df[df['ds'] >= '2024-04-01'][['ds', 'y']]

# Define custom holidays
holidays = pd.DataFrame({
    'holiday': [
        'black_friday','black_friday','black_friday',
        'christmas','christmas','christmas',
        'new_year','new_year','new_year'
    ],
    'ds': pd.to_datetime([
        '2021-11-26','2022-11-25','2023-11-24',
        '2021-12-25','2022-12-25','2023-12-25',
        '2022-01-01','2023-01-01','2024-01-01'
    ]),
    'lower_window': [-3, -3, -3, -7, -7, -7, -1, -1, -1],
    'upper_window': [ 3,  3,  3,  3,  3,  3,  1,  1,  1],
})

print("\n⚡ Training Prophet model...")
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    holidays_prior_scale=10,
    interval_width=0.95
)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(train_df)

# Forecast
future   = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

# Evaluate on test set
test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
merged = test_df.merge(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
mape = mean_absolute_percentage_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))

print(f"\n{'='*60}")
print("PROPHET FORECAST PERFORMANCE (90-day horizon)")
print(f"{'='*60}")
print(f"MAPE : {mape:.2%}")
print(f"RMSE : R {rmse:,.0f}")

# ============================================================
# STEP 5 — FORECAST PLOT
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
fig.suptitle('Sales Forecast — Prophet Model', fontsize=16, color='white', fontweight='bold')

# Full forecast
ax = axes[0]
hist = forecast[forecast['ds'] <= train_df['ds'].max()]
fut  = forecast[forecast['ds'] > train_df['ds'].max()]

ax.plot(train_df['ds'], train_df['y'] / 1000, color='white', alpha=0.5, lw=0.8, label='Historical')
ax.plot(test_df['ds'],  test_df['y'] / 1000,  color=AMBER,  lw=1.5, label='Actual (test)')
ax.plot(fut['ds'],       fut['yhat'] / 1000,   color=PURPLE, lw=2.5, label='Forecast')
ax.fill_between(fut['ds'],
                fut['yhat_lower'] / 1000,
                fut['yhat_upper'] / 1000,
                alpha=0.2, color=PURPLE, label='95% Confidence Interval')
ax.axvline(train_df['ds'].max(), color=CORAL, linestyle='--', lw=1.5, alpha=0.7)
ax.set_title('Revenue Forecast (R000s)', color='white')
ax.set_ylabel('Daily Revenue (R000s)', color='white')
ax.legend(loc='upper left')

# Components
ax = axes[1]
trend_comp = forecast[['ds', 'trend']].set_index('ds')
ax.plot(trend_comp.index, trend_comp['trend'] / 1000, color=TEAL, lw=2.5)
ax.fill_between(trend_comp.index, trend_comp['trend'] / 1000, alpha=0.1, color=TEAL)
ax.set_title('Trend Component (R000s)', color='white')
ax.set_ylabel('Trend (R000s)', color='white')

plt.tight_layout()
plt.savefig('reports/sales_forecast.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.show()

print(f"\n✅ PROJECT 5 COMPLETE — Sales Forecasting")
print(f"   MAPE: {mape:.2%} | RMSE: R {rmse:,.0f}")
print(f"   Horizon: 90 days | Model: Prophet + custom holidays")
print(f"   Plots saved: reports/")
