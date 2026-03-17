"""
Benchmark: insurance-demand ConversionModel vs naive price sensitivity estimate.

The problem: a naive logistic regression of P(convert) on price produces a biased
elasticity estimate because price is set by the underwriting model — high-risk
customers receive higher prices AND may have different baseline conversion rates.
This conflates risk composition with price sensitivity.

This benchmark evaluates ConversionModel against two baselines:
1. Naive logistic on log(quoted_price) — wrong treatment, biased coefficient
2. Logistic on log(price/tech_prem) — correct normalisation, manual implementation
3. ConversionModel — same normalisation, insurance-demand implementation

Metrics: Brier score (calibration), log-loss (discrimination), and price
semi-elasticity estimate vs known true value.

Setup
-----
- 30,000 synthetic UK motor PCW quotes, train/test 70/30 split
- True DGP: logit(P(convert)) = 2.5 - 2.0*log(price/tech_prem) + covariate effects
  True price semi-elasticity on log(price/tech_prem) = -2.0
- Confounding: technical_premium correlated with vehicle_group (risk composition)

Run
---
    python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 30_000

age = RNG.integers(18, 75, N).astype(float)
vehicle_group = RNG.integers(1, 5, N)
ncd_years = RNG.integers(0, 9, N).astype(float)
area = RNG.integers(1, 6, N)

base_premium = (
    300
    + 60 * (vehicle_group - 1)
    - 10 * ncd_years
    + 20 * np.where(age < 25, 1, 0)
    + 15 * (area - 1)
)
tech_prem = np.maximum(base_premium + RNG.normal(0, 30, N), 100.0)

loading_noise = RNG.normal(0, 0.08, N)
price_ratio = 1.0 + 0.05 + 0.05 * (vehicle_group - 2.5) / 2.0 + loading_noise
price_ratio = np.clip(price_ratio, 0.7, 1.5)
quoted_price = tech_prem * price_ratio
rank_position = RNG.integers(1, 6, N).astype(float)

TRUE_ELASTICITY = -2.0

log_odds = (
    2.5
    + TRUE_ELASTICITY * np.log(price_ratio)
    - 0.3 * np.log(rank_position)
    + 0.01 * ncd_years
    - 0.005 * (vehicle_group - 1)
    + RNG.normal(0, 0.05, N)
)
prob_convert = 1.0 / (1.0 + np.exp(-log_odds))
converted = RNG.binomial(1, prob_convert, N)

print("=" * 65)
print("insurance-demand benchmark")
print("ConversionModel vs naive logistic regression")
print("=" * 65)
print(f"\nDGP: {N:,} quotes, true price semi-elasticity = {TRUE_ELASTICITY}")
print(f"Overall conversion rate: {converted.mean():.1%}")
print(f"Mean quoted price: £{quoted_price.mean():.0f}")
print(f"Mean price ratio (loading): {price_ratio.mean():.3f}")
print()

split = int(0.7 * N)
y_tr = converted[:split]
y_te = converted[split:]

# ---------------------------------------------------------------------------
# 2. Naive logistic on log(quoted_price)
# ---------------------------------------------------------------------------

print("Estimator 1: Naive logistic on log(quoted_price)")
print("-" * 50)

t0 = time.perf_counter()
X_naive_tr = np.log(quoted_price[:split]).reshape(-1, 1)
X_naive_te = np.log(quoted_price[split:]).reshape(-1, 1)

pipe_naive = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
pipe_naive.fit(X_naive_tr, y_tr)
t_naive = time.perf_counter() - t0

prob_naive_te = pipe_naive.predict_proba(X_naive_te)[:, 1]
brier_naive = brier_score_loss(y_te, prob_naive_te)
logloss_naive = log_loss(y_te, prob_naive_te)
naive_price_coef = pipe_naive.named_steps["lr"].coef_[0, 0] / pipe_naive.named_steps["sc"].scale_[0]

print(f"  Brier score:          {brier_naive:.5f}")
print(f"  Log-loss:             {logloss_naive:.5f}")
print(f"  Price coef (log):     {naive_price_coef:.4f}")
print(f"  Fit time:             {t_naive:.3f}s")

# ---------------------------------------------------------------------------
# 3. Logistic on log(price/tech_prem) + covariates
# ---------------------------------------------------------------------------

print()
print("Estimator 2: Logistic on log(price/tech_prem) + covariates")
print("-" * 50)

t0 = time.perf_counter()
log_ratio = np.log(price_ratio)
X_ratio_tr = np.column_stack([
    log_ratio[:split],
    np.log(rank_position[:split]),
    ncd_years[:split],
    (vehicle_group[:split] - 1).astype(float),
])
X_ratio_te = np.column_stack([
    log_ratio[split:],
    np.log(rank_position[split:]),
    ncd_years[split:],
    (vehicle_group[split:] - 1).astype(float),
])

pipe_ratio = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
pipe_ratio.fit(X_ratio_tr, y_tr)
t_ratio = time.perf_counter() - t0

prob_ratio_te = pipe_ratio.predict_proba(X_ratio_te)[:, 1]
brier_ratio = brier_score_loss(y_te, prob_ratio_te)
logloss_ratio = log_loss(y_te, prob_ratio_te)
ratio_price_coef = pipe_ratio.named_steps["lr"].coef_[0, 0] / pipe_ratio.named_steps["sc"].scale_[0]

print(f"  Brier score:          {brier_ratio:.5f}")
print(f"  Log-loss:             {logloss_ratio:.5f}")
print(f"  Price coef (log):     {ratio_price_coef:.4f}")
print(f"  True elasticity:      {TRUE_ELASTICITY:.4f}")
print(f"  Bias from true:       {abs(ratio_price_coef - TRUE_ELASTICITY):.4f}")
print(f"  Fit time:             {t_ratio:.3f}s")

# ---------------------------------------------------------------------------
# 4. insurance-demand ConversionModel
# ---------------------------------------------------------------------------

print()
print("Estimator 3: insurance-demand ConversionModel (logistic backend)")
print("-" * 50)

try:
    from insurance_demand import ConversionModel

    df_tr = pl.DataFrame({
        "quoted_price":       quoted_price[:split],
        "technical_premium":  tech_prem[:split],
        "converted":          converted[:split].astype(int),
        "ncd_years":          ncd_years[:split],
        "vehicle_group":      vehicle_group[:split],
        "rank_position":      rank_position[:split],
    })
    df_te = pl.DataFrame({
        "quoted_price":       quoted_price[split:],
        "technical_premium":  tech_prem[split:],
        "converted":          converted[split:].astype(int),
        "ncd_years":          ncd_years[split:],
        "vehicle_group":      vehicle_group[split:],
        "rank_position":      rank_position[split:],
    })

    t0 = time.perf_counter()
    cm = ConversionModel(
        base_estimator="logistic",
        feature_cols=["ncd_years", "vehicle_group"],
        rank_position_col="rank_position",
    )
    cm.fit(df_tr)
    t_cm = time.perf_counter() - t0

    prob_cm_te = cm.predict_proba(df_te).values
    brier_cm = brier_score_loss(y_te, prob_cm_te)
    logloss_cm = log_loss(y_te, prob_cm_te)

    # Extract price coefficient from the pipeline
    # The first feature is 'log_price_ratio' (see _build_features)
    pipe_cm = cm._model  # sklearn Pipeline
    sc_cm = pipe_cm.named_steps["scaler"]
    lr_cm = pipe_cm.named_steps["logit"]

    # Feature names to find log_price_ratio index
    encoded_cols = cm._feature_names_in
    try:
        price_idx = encoded_cols.index("log_price_ratio")
    except ValueError:
        price_idx = 0  # fallback: first feature

    cm_price_coef = lr_cm.coef_[0, price_idx] / sc_cm.scale_[price_idx]
    cm_bias = abs(cm_price_coef - TRUE_ELASTICITY)

    print(f"  Brier score:          {brier_cm:.5f}")
    print(f"  Log-loss:             {logloss_cm:.5f}")
    print(f"  Price coef (log):     {cm_price_coef:.4f}")
    print(f"  True elasticity:      {TRUE_ELASTICITY:.4f}")
    print(f"  Bias from true:       {cm_bias:.4f}")
    print(f"  Fit time:             {t_cm:.3f}s")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    brier_cm = float('nan')
    logloss_cm = float('nan')
    cm_price_coef = float('nan')
    cm_bias = float('nan')
    t_cm = float('nan')

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  {'Metric':<35} {'Naive':>10} {'Price ratio':>12} {'Conv.Model':>12}")
print("  " + "-" * 70)
print(f"  {'Brier score (lower=better)':<35} {brier_naive:>10.5f} {brier_ratio:>12.5f} {brier_cm:>12.5f}")
print(f"  {'Log-loss (lower=better)':<35} {logloss_naive:>10.5f} {logloss_ratio:>12.5f} {logloss_cm:>12.5f}")
print(f"  {'Price coef (true=-2.0)':<35} {naive_price_coef:>10.4f} {ratio_price_coef:>12.4f} {cm_price_coef:>12.4f}")
bias_naive = abs(naive_price_coef - TRUE_ELASTICITY)
bias_ratio = abs(ratio_price_coef - TRUE_ELASTICITY)
print(f"  {'Elasticity bias (abs)':<35} {bias_naive:>10.4f} {bias_ratio:>12.4f} {cm_bias:>12.4f}")
print()
print("Interpretation:")
print("  ConversionModel normalises price by technical premium, removing the")
print("  risk-composition confounding that makes naive price coefficients")
print("  unreliable for elasticity interpretation. The naive log(quoted_price)")
print("  coefficient is dominated by the risk level, not commercial loading.")
print("  On predictive metrics (Brier/log-loss) the correctly specified")
print("  approaches are similar; the key difference is the coefficient")
print("  interpretability for demand curve and price optimisation inputs.")
