# insurance-demand
[![Tests](https://github.com/burning-cost/insurance-demand/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-demand/actions/workflows/tests.yml)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-demand)

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-demand/discussions). Found it useful? A ⭐ helps others find it.

Conversion, retention, and price elasticity modelling for UK personal lines insurance.

---

## The Problem

UK personal lines insurers price using risk models— pure premium equals frequency times severity. That's the right starting point, but the market outcome depends on something the risk model doesn't capture: whether the customer accepts the quoted price.

You can have a perfectly calibrated GLM and still be pricing wrong commercially. Quote too high and conversion rate falls; quote too low and you've left margin on the table. The demand model is what connects the technical premium to the commercial outcome.

This has two components that the industry treats as separate problems:

1. **Static demand**: What is P(buy | current price, risk features)? This is conversion modelling for new business, or renewal probability for existing customers. It tells you expected volume at current prices.

2. **Dynamic demand (elasticity)**: How does P(buy) respond to price changes? This is the question for optimisation. A 5% price increase on inelastic customers costs you almost nothing in volume but recovers substantial margin. On elastic customers, the same increase can wipe out the book.

The industry has the right framework— Akur8, Earnix, and Radar all implement it. None of them expose a Python API, and none of them show you their methodology. When the FCA asks how you arrived at your renewal pricing, "the vendor's algorithm" is not a satisfying answer.

This library covers the full pipeline, open source, with documented methodology at each step.

---

## Regulatory Context

FCA PS21/11 (effective January 2022) banned renewal price-walking: charging renewing customers more than new customers for equivalent risk. The FCA's evaluation in July 2025 (EP25/2) found the ban largely effective.

What PS21/11 does not ban:
- Conversion optimisation for new business
- Retention modelling to understand lapse risk
- Offering targeted retention discounts to high-lapse-risk customers
- Any demand modelling that runs in the space *below* the ENBP ceiling

What it does ban:
- Setting a renewal price above the equivalent new business price (ENBP)
- Using estimated "inertia" (propensity to renew) to justify a higher renewal price

This library includes an `ENBPChecker` that audits a renewal portfolio for ENBP compliance and a `price_walking_report` that detects systematic tenure-based price patterns— the diagnostic the FCA uses in multi-firm reviews.

---

## Installation

```bash
uv add insurance-demand
```

With optional extras:

```bash
uv add insurance-demand[catboost]   # CatBoost backend for models
uv add insurance-demand[dml]        # DML elasticity (doubleml + catboost)
uv add insurance-demand[causal]     # Heterogeneous effects (econml)
uv add insurance-demand[survival]   # Cox/Weibull retention models (lifelines)
uv add insurance-demand[plot]       # DemandCurve.plot()
uv add insurance-demand[all]        # Everything
```

---

## Quick Start

The full quick-start uses `ElasticityEstimator`, which requires the `dml` extra:

```bash
uv add insurance-demand[dml]
```

```python
from insurance_demand import ConversionModel, RetentionModel, ElasticityEstimator
from insurance_demand import DemandCurve, OptimalPrice
from insurance_demand.compliance import ENBPChecker
from insurance_demand.datasets import generate_conversion_data, generate_retention_data

# --- 1. Generate synthetic data (or use your own) ---
df_quotes = generate_conversion_data(n_quotes=150_000)
df_renewals = generate_retention_data(n_policies=80_000)

# --- 2. Fit a conversion model ---
conv_model = ConversionModel(
    base_estimator="logistic",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
)
conv_model.fit(df_quotes)
probs = conv_model.predict_proba(df_quotes)

# --- 3. Fit a retention model ---
retention_model = RetentionModel(
    model_type="logistic",
    feature_cols=["tenure_years", "ncd_years", "payment_method", "claim_last_3yr"],
)
retention_model.fit(df_renewals)
lapse_probs = retention_model.predict_proba(df_renewals)

# --- 4. Estimate causal price elasticity with DML ---
# outcome_transform='identity' gives a linear probability model elasticity:
# d[P(convert)] / d[log_price_ratio]. Use 'logit' for a log-odds specification
# (d[log-odds] / d[log_price_ratio]) -- see ElasticityEstimator docstring.
est = ElasticityEstimator(
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
    outcome_transform="identity",
)
est.fit(df_quotes)
# est.summary() returns a DataFrame with columns:
#   parameter | estimate | std_error | ci_lower_95 | ci_upper_95 | treatment | outcome | n_folds
print(est.summary())

# --- 5. Build a demand curve ---
curve = DemandCurve(
    elasticity=est.elasticity_,
    base_price=500.0,
    base_prob=0.12,
    functional_form="semi_log",
)
prices, probs = curve.evaluate(price_range=(300, 900), n_points=60)

# --- 6. Find the optimal price ---
opt = OptimalPrice(
    demand_curve=curve,
    expected_loss=380.0,
    expense_ratio=0.15,
    enbp=650.0,  # PS21/11 ceiling for renewals
)
result = opt.optimise()
print(f"Optimal price: £{result.optimal_price:.2f}")
print(f"Conversion prob: {result.conversion_prob:.4f}")
print(f"Expected profit per quote: £{result.expected_profit:.2f}")
print(f"Constraints active: {result.constraints_active}")

# --- 7. Audit ENBP compliance ---
checker = ENBPChecker()
report = checker.check(df_renewals)
print(report)
```

**Expected output (n=150,000 quotes, n=80,000 renewals, synthetic data):**

```
# ElasticityEstimator fit: 26s
          parameter  estimate  std_error  ci_lower_95  ci_upper_95        treatment    outcome  n_folds
0  price_elasticity -0.866761   0.027732    -0.921115    -0.812408  log_price_ratio  converted        5

Optimal price: £650.00
Conversion prob: 0.0953
Expected profit per quote: £16.44
Constraints active: ['ENBP']

ENBPBreachReport(
  n_policies=80,000,
  n_breaches=0 (0.00%),
  mean_breach=£0.00,
  max_breach=£0.00,
  total_overpayment=£0.00
)
```

The ENBP constraint is binding here: with `expected_loss=380` and `expense_ratio=0.15`, the unconstrained profit-maximising price lands above the £650 ENBP ceiling, so `OptimalPrice` returns exactly £650. This is the correct behaviour for a renewal segment where the PS21/11 ceiling is the binding constraint — it tells you the optimal price is the maximum permissible one.

---

## Modules

### ConversionModel

Static new business demand model. Predicts P(buy | price, features) for quote-level data.

Two backends:
- `logistic`: sklearn LogisticRegression. Interpretable coefficients, analytical marginal effects. Start here.
- `catboost`: CatBoost classifier. Handles non-linear interactions between price, channel, and risk class. Better predictive accuracy on real data.

Price treatment: uses `log(quoted_price / technical_premium)` by default— the commercial loading rather than the absolute price. This follows industry practice (Guven & McPhail 2013, CAS) and makes the coefficient interpretable across risk segments.

PCW rank position is included as a separate feature because rank has a discrete demand effect that price ratio alone doesn't capture. Being cheapest versus second cheapest on a PCW is worth more than the price gap would suggest.

```python
# Naive marginal effect (biased— confounding not removed)
me = conv_model.marginal_effect(df)

# One-way: observed vs. fitted conversion by factor level
conv_model.oneway(df, "channel")

# Export for insurance-optimise integration
demand_fn = conv_model.as_demand_callable()
```

### RetentionModel

Renewal demand model. Predicts P(lapse | features, price_change) for renewal portfolios.

Four backends:
- `logistic`: standard logistic GLM. Industry default.
- `catboost`: GBM for non-linear effects.
- `cox`: Cox proportional hazards via lifelines. Better for CLV models where you need survival curves across multiple future renewals.
- `weibull`: Weibull AFT model. More flexible hazard shape than Cox.

The treatment variable is `log(renewal_price / prior_year_price)`— the price change, not the absolute price. This is what the customer responds to: a renewal feels expensive relative to what they paid last year, not relative to the actuarial technical premium they've never seen.

```python
# Lapse probability
lapse = retention_model.predict_proba(df)

# Renewal probability (complement)
renewal = retention_model.predict_renewal_proba(df)

# Full survival curve at t=1,2,3,5 years (survival models only)
surv = retention_model.predict_survival(df, times=[1, 2, 3, 5])

# Price sensitivity: dP(lapse)/d(log_price_change)
retention_model.price_sensitivity(df)
```

### ElasticityEstimator

DML-based causal price elasticity estimation. The core methodology.

**Why DML**: In insurance observational data, price is set by the underwriting system based on risk features. High-risk customers get higher prices AND may have different price sensitivity. Naive regression of conversion on price conflates these two effects. The DML estimator removes the confounding by residualising both outcome and treatment on all observed confounders before estimating the price coefficient.

The estimator wraps `doubleml` (for global elasticity) or `econml` (for per-customer heterogeneous elasticity). CatBoost nuisance models handle the categorical structure of insurance data without preprocessing.

```python
est = ElasticityEstimator(
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
    heterogeneous=False,  # True -> per-customer CATE via econml
)
est.fit(df_quotes)

# Global estimate with SE and CI
# Returns a DataFrame: parameter | estimate | std_error | ci_lower_95 | ci_upper_95 | ...
est.summary()

# Sensitivity: how large does unobserved confounding need to be to overturn the result?
est.sensitivity_analysis()

# Per-customer elasticity (heterogeneous=True only)
cates = est.effect(df_quotes)
```

**Data requirements**: At minimum 20,000 observations with genuine within-segment price variation. Rate review periods (where the portfolio loading changes uniformly for a quarter) are the primary source of identification. You need technical premium stored at quote time— retroactively recalculated tech premiums will introduce errors.

### DemandCurve

Converts an elasticity estimate (or fitted model) into a price-to-probability curve.

Two parametric forms:
- `semi_log`: `logit(p) = α + β × log(price)`. Elasticity varies with the probability level. Appropriate for binary outcomes.
- `log_linear`: `log(p) = α + ε × log(price)`. Constant elasticity. Simpler interpretation.

Or use a fitted model directly (`functional_form='model'`)— the curve evaluates the model at each price point, averaged over a reference portfolio.

```python
curve = DemandCurve(
    elasticity=-2.0,
    base_price=500.0,
    base_prob=0.12,
    functional_form="semi_log",
)

# Evaluate
prices, probs = curve.evaluate(price_range=(300, 900), n_points=100)

# Plot
curve.plot(price_range=(300, 900))

# Export for insurance-optimise
fn = curve.as_demand_callable()
```

### OptimalPrice

Finds the profit-maximising price for a single segment subject to constraints.

Objective: `max P(buy | price) × (price - expected_loss - expenses)`

Constraints:
- `min_price`, `max_price`: hard bounds
- `enbp`: PS21/11 ceiling for renewal pricing (max_price is min(max_price, enbp))
- `min_conversion_rate`: volume floor as a conversion rate threshold
- `min_margin_rate`: minimum required margin fraction

This is single-segment pricing. For portfolio-level factor optimisation across many segments simultaneously, use the `insurance-optimise` library with demand callables from this library as inputs.

### ENBPChecker and compliance utilities

```python
from insurance_demand.compliance import ENBPChecker, price_walking_report

# Full portfolio ENBP audit
checker = ENBPChecker(tolerance=0.0)
report = checker.check(df_renewals)
# report.n_breaches, report.by_channel, report.breach_detail

# Tenure-based price pattern diagnostic
walking = price_walking_report(df_renewals, nb_price_col="nb_equivalent_price")
# Rising price-to-ENBP ratio with tenure = potential PS21/11 concern
```

---

## Key Design Decisions

**Why separate ConversionModel from ElasticityEstimator**: These answer different questions. The conversion model gives you expected volume at current prices— useful for forecasting. The elasticity estimator gives you the causal response to price changes— required for optimisation. Conflating them produces the bias that DML fixes.

**Why log(price/technical_premium) as the treatment**: The absolute quoted price is dominated by risk composition. A £600 policy on a high-risk driver and a £600 policy on a low-risk driver mean different things. The price-to-technical-premium ratio isolates commercial decision-making from risk. Variation in this ratio (driven by quarterly rate reviews, not individual risk assessment) is approximately exogenous.

**Why CatBoost for nuisance models**: Insurance features are predominantly categorical (area, vehicle group, NCD band, channel). CatBoost handles ordered and unordered categoricals natively without one-hot encoding. This matters for DML: poorly specified nuisance models leak confounding into the elasticity estimate.

**Why not implement the optimiser here**: The portfolio-level factor optimisation problem (adjusting multiplicative rating factors across a multi-dimensional tariff) requires a different architecture than single-segment pricing. That's what `insurance-optimise` does. The demand library's job is to supply the demand curve; the optimiser's job is to find the factor adjustments that maximise the objective given that curve.

---

## Data Schema

**Conversion data (new business quotes)**:
```
quote_id          str      unique identifier
quote_date        date
channel           str      'pcw_confused', 'pcw_msm', 'pcw_ctm', 'pcw_go', 'direct'
quoted_price      float    our quoted premium
technical_premium float    risk model output at quote time (must be at quote time)
converted         int      1 = policy bound, 0 = quoted only
age               int
vehicle_group     int      1–4 (or your own risk classification)
ncd_years         int      0–9
area              str
[rank_position    int]     1 = cheapest on PCW (optional but recommended)
[competitor_price_min float] cheapest competitor at quote time (optional)
```

**Renewal data**:
```
policy_id         str
renewal_date      date
renewal_price     float    price offered at renewal
prior_year_price  float    what they paid last year
nb_equivalent_price float  ENBP: NB price for same risk/channel
lapsed            int      1 = lapsed, 0 = renewed
tenure_years      float
ncd_years         int
payment_method    str      'dd', 'card', 'cheque'
channel           str
```

---


## Performance

Benchmarked against **naive logistic on log(quoted_price)** on synthetic UK motor PCW data — 30,000 quotes, known true price semi-elasticity = −2.0 on log(price/technical_premium). Full script: `benchmarks/run_benchmark.py`.

The DGP: high-risk vehicles (vehicle_group 4) receive higher technical premiums and higher commercial loadings. A naive model that regresses conversion on log(quoted_price) conflates the absolute price level with the commercial loading the customer actually responds to.

| Metric | Naive (log price) | Logistic (price ratio) | ConversionModel |
|--------|------------------|------------------------|-----------------|
| Brier score | 0.09229 | 0.09186 | 0.09186 |
| Log-loss | 0.33132 | 0.32887 | 0.32887 |
| Price coefficient (true = −2.0) | −0.40 | −2.09 | −2.09 |
| Elasticity bias (abs) | 1.60 | 0.09 | 0.09 |
| Fit time | 0.04s | 0.09s | 0.12s |

On predictive metrics (Brier, log-loss) the naive and correctly specified models are nearly identical — both fit the data. The difference is entirely in the price coefficient. The naive coefficient of −0.40 is dominated by the correlation between absolute price and risk class, not commercial sensitivity. ConversionModel and the correctly specified logistic both recover the true elasticity of −2.0 within 4.5%.

This is the key point: you can have a well-calibrated conversion model with a completely wrong elasticity. If you use the naive coefficient as input to a demand curve or pricing optimiser, your optimised prices will be wrong by a factor of 5.

The DML-based `ElasticityEstimator` (requires the `dml` extra) takes this further: it handles residual confounding that even the price/tech_prem normalisation cannot fully remove. Run `notebooks/benchmark.py` on Databricks for the full DML comparison (3–8 minutes due to cross-fitting).


## References

- Chernozhukov et al. (2018). Double/Debiased Machine Learning. *Econometrics Journal*, 21(1).
- Guven & McPhail (2013). Beyond the Cost Model. *CAS Forum*.
- FCA PS21/11. General Insurance Pricing Practices Amendments.
- FCA EP25/2 (July 2025). Evaluation of GIPP Remedies.
- Spedicato, Dutang, Petrini (2021). Machine Learning Methods for Pricing Optimisation. *CAS e-Forum*.

---


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_demand_demo.py).

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID-based causal evaluation of rate changes — did your pricing action actually work? |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All Burning Cost libraries →](https://burning-cost.github.io)

---

## Read more

[Demand Modelling for Insurance Pricing](https://burning-cost.github.io/blog/demand-modelling-for-insurance-pricing) — the full methodology behind conversion, retention, and causal elasticity estimation.

[Your Demand Model Is Confounded](https://burning-cost.github.io/blog/your-demand-model-is-confounded) — why naive regression of conversion on price gives you the wrong elasticity, and how DML fixes it.

---


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | DML-based causal price elasticity estimation for renewal pricing optimisation |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation — consumes demand curves from this library to find profit-maximising factor adjustments |
| [insurance-survival](https://github.com/burning-cost/insurance-survival) | Customer lifetime value and churn survival models — complements retention modelling with multi-period CLV projections |

MIT License. Built by [Burning Cost](https://burning-cost.github.io).
