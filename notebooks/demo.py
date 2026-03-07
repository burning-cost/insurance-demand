# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Full Pipeline Demo
# MAGIC
# MAGIC This notebook demonstrates the complete `insurance-demand` workflow on
# MAGIC synthetic UK motor insurance data.
# MAGIC
# MAGIC **The problem**: A mid-tier UK motor insurer writes ~150,000 PCW new business
# MAGIC quotes per year and has ~80,000 policies renewing annually. They need to:
# MAGIC
# MAGIC 1. Model conversion probability (P(buy | price, features)) for new business
# MAGIC 2. Model renewal probability (P(renew | features, price_change)) for the book
# MAGIC 3. Estimate the *causal* price elasticity using DML — not the biased naive estimate
# MAGIC 4. Build demand curves to understand how volume responds to price changes
# MAGIC 5. Find the optimal price for a segment subject to ENBP and margin constraints
# MAGIC
# MAGIC **The regulatory context**: FCA PS21/11 bans charging renewing customers more than
# MAGIC the equivalent new business price (ENBP). Demand modelling is permitted and
# MAGIC encouraged — it's how you decide who to offer targeted retention discounts to.
# MAGIC What's banned is using inertia to justify *surcharging*. The ENBP check at the
# MAGIC end of this notebook demonstrates how to audit your renewal portfolio for compliance.

# COMMAND ----------

# MAGIC %pip install insurance-demand[dml,survival,plot] --quiet

# COMMAND ----------

# MAGIC %pip install catboost doubleml --quiet

# COMMAND ----------

import numpy as np
import polars as pl
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Data
# MAGIC
# MAGIC The data generating process embeds **known confounding**: high-risk customers
# MAGIC (high vehicle group, young age) receive higher technical premiums AND are less
# MAGIC price-sensitive (fewer alternatives). This means naive logistic regression will
# MAGIC underestimate the true price elasticity for standard risk. DML corrects this.
# MAGIC
# MAGIC True population-average elasticity: **-2.0** (1% price rise → 2% conversion drop).

# COMMAND ----------

from insurance_demand.datasets import generate_conversion_data, generate_retention_data

print("Generating quote data...")
df_quotes = generate_conversion_data(n_quotes=150_000, true_price_elasticity=-2.0, seed=42)
print(f"Quote data: {df_quotes.shape}")
print(f"Conversion rate: {df_quotes['converted'].mean():.1%}")
print(f"Price ratio range: {df_quotes['price_ratio'].min():.2f} – {df_quotes['price_ratio'].max():.2f}")

# COMMAND ----------

print("Generating renewal data...")
df_renewals = generate_retention_data(n_policies=80_000, true_price_change_elasticity=-3.5, seed=42)
print(f"Renewal data: {df_renewals.shape}")
print(f"Lapse rate: {df_renewals['lapsed'].mean():.1%}")
print(f"Mean price change: {df_renewals['price_change_pct'].mean():.1%}")
print(f"ENBP compliance: {df_renewals['enbp_compliant'].mean():.1%} of renewals compliant")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Conversion Model (New Business)
# MAGIC
# MAGIC We fit a logistic GLM on new business quotes. The model uses:
# MAGIC - Log(price / technical_premium): the commercial loading, main price treatment
# MAGIC - Rank position on PCW: discrete demand jump at rank 1
# MAGIC - Price to market ratio: how our price compares to cheapest competitor
# MAGIC - Risk features: age, vehicle_group, NCD, area, channel

# COMMAND ----------

from insurance_demand import ConversionModel

CONV_FEATURES = ["age", "vehicle_group", "ncd_years", "area", "channel"]

conv_model = ConversionModel(
    base_estimator="logistic",
    feature_cols=CONV_FEATURES,
    rank_position_col="rank_position",
    price_to_market_col="price_to_market",
)

conv_model.fit(df_quotes)
print("Conversion model fitted.")

# COMMAND ----------

# Model summary
summary = conv_model.summary()
print("Top 10 features by absolute coefficient:")
print(summary.head(10).to_string(index=False))

# COMMAND ----------

# Conversion rate predictions
conv_probs = conv_model.predict_proba(df_quotes)
print(f"\nPredicted conversion rate: {conv_probs.mean():.1%}")
print(f"Observed conversion rate:  {df_quotes['converted'].mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-Way Plots: Observed vs. Fitted by Factor
# MAGIC
# MAGIC The one-way diagnostic is the standard pricing check: does the model track
# MAGIC the observed conversion rate by each rating factor level?

# COMMAND ----------

# Channel one-way
channel_ow = conv_model.oneway(df_quotes, "channel")
print("Conversion by channel (observed vs. fitted):")
print(channel_ow.to_string(index=False))

# COMMAND ----------

# Vehicle group one-way
vg_ow = conv_model.oneway(df_quotes, "vehicle_group")
print("\nConversion by vehicle group:")
print(vg_ow.to_string(index=False))

# COMMAND ----------

# NCD one-way
ncd_ow = conv_model.oneway(df_quotes, "ncd_years")
print("\nConversion by NCD years:")
print(ncd_ow.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marginal Price Effect
# MAGIC
# MAGIC The naive marginal effect: dP(conversion)/dPrice. This is the slope of the
# MAGIC logistic curve. It's BIASED because it conflates the price effect with the
# MAGIC risk composition effect. Section 4 (DML) gives the debiased version.

# COMMAND ----------

me = conv_model.marginal_effect(df_quotes)
naive_elasticity = conv_model.price_elasticity(df_quotes)
print(f"Naive price elasticity (biased): {naive_elasticity.mean():.3f}")
print(f"True elasticity: -2.000 (embedded in DGP)")
print(f"Bias: {naive_elasticity.mean() - (-2.0):+.3f}")
print()
print("The naive logistic regression underestimates elasticity because high-risk")
print("customers get higher prices AND have lower price sensitivity — confounding")
print("that DML removes via residualisation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Retention Model (Renewals)
# MAGIC
# MAGIC We fit a logistic model on the renewal portfolio. The key treatment variable is
# MAGIC log(renewal_price / prior_year_price) — the price change from the prior year.
# MAGIC
# MAGIC Post-PS21/11: high predicted lapse probability can inform targeted retention
# MAGIC discounts. It cannot inform surcharges.

# COMMAND ----------

from insurance_demand import RetentionModel

RETENTION_FEATURES = ["tenure_years", "ncd_years", "payment_method", "claim_last_3yr", "channel", "age"]

retention_model = RetentionModel(
    model_type="logistic",
    feature_cols=RETENTION_FEATURES,
)

retention_model.fit(df_renewals)
print("Retention model fitted.")

# COMMAND ----------

summary_ret = retention_model.summary()
print("Retention model coefficients:")
print(summary_ret.head(10).to_string(index=False))

# COMMAND ----------

lapse_probs = retention_model.predict_proba(df_renewals)
print(f"\nPredicted lapse rate: {lapse_probs.mean():.1%}")
print(f"Observed lapse rate:   {df_renewals['lapsed'].mean():.1%}")

# COMMAND ----------

# Payment method one-way
pm_ow = retention_model.oneway(df_renewals, "payment_method")
print("Lapse rate by payment method:")
print(pm_ow.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Price Sensitivity in the Retention Model

# COMMAND ----------

price_sens = retention_model.price_sensitivity(df_renewals)
print(f"Mean dP(lapse)/d(log_price_change): {price_sens.mean():.4f}")
print(f"(Positive: higher price increase → more lapses, as expected)")

# COMMAND ----------

# Demonstrate: what happens to lapse rate if we increase all prices by 10%?
df_renewals_pd = df_renewals.to_pandas()
df_10pct_increase = df_renewals_pd.copy()
df_10pct_increase["log_price_change"] = df_renewals_pd["log_price_change"] + np.log(1.10)

lapse_base = retention_model.predict_proba(df_renewals_pd).mean()
lapse_increased = retention_model.predict_proba(df_10pct_increase).mean()
print(f"\nPortfolio lapse rate at current prices:     {lapse_base:.1%}")
print(f"Portfolio lapse rate with +10% price change: {lapse_increased:.1%}")
print(f"Incremental lapse: +{(lapse_increased - lapse_base):.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. DML Elasticity Estimation (Causal, Debiased)
# MAGIC
# MAGIC This is the methodological heart of the library. We use Double Machine Learning
# MAGIC (Chernozhukov et al. 2018) to estimate the causal price elasticity, removing the
# MAGIC confounding from risk composition.
# MAGIC
# MAGIC **The DML algorithm**:
# MAGIC 1. Regress Y (logit of conversion) on confounders X → get residuals Ỹ
# MAGIC 2. Regress D (log price ratio) on X → get residuals D̃
# MAGIC 3. OLS of Ỹ on D̃ → θ = causal elasticity
# MAGIC
# MAGIC Cross-fitting ensures nuisance model overfitting doesn't bias θ.

# COMMAND ----------

from insurance_demand import ElasticityEstimator

DML_FEATURES = [
    "age", "vehicle_group", "ncd_years", "area", "channel",
    "annual_mileage",
]

est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=DML_FEATURES,
    n_folds=5,
    outcome_model="catboost",
    treatment_model="catboost",
    outcome_transform="logit",
)

print("Fitting DML elasticity estimator (5-fold cross-fitting)...")
print("This takes 2-4 minutes on a medium cluster — CatBoost nuisance models for 150k obs.")
est.fit(df_quotes)

# COMMAND ----------

summary_dml = est.summary()
print("DML Price Elasticity Results:")
print(summary_dml.to_string(index=False))
print()
print(f"Estimated elasticity: {est.elasticity_:.3f}")
print(f"Standard error:       {est.elasticity_se_:.3f}")
print(f"95% CI:               [{est.elasticity_ci_[0]:.3f}, {est.elasticity_ci_[1]:.3f}]")
print(f"True elasticity (DGP):-2.000")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretation
# MAGIC
# MAGIC The DML estimate should be close to -2.0 (the true DGP value). The naive
# MAGIC logistic estimate was biased because it didn't remove the confounding structure.
# MAGIC
# MAGIC The confidence interval tells you the range of plausible elasticity values
# MAGIC given the data. A CI that excludes 0 means the price effect is statistically
# MAGIC distinguishable from zero.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Demand Curves
# MAGIC
# MAGIC DemandCurve turns the elasticity estimate into a curve: price → P(buy).
# MAGIC Two uses:
# MAGIC 1. Visual: understand how demand varies across a price range
# MAGIC 2. Input to OptimalPrice: find the profit-maximising price

# COMMAND ----------

from insurance_demand import DemandCurve

# Use the DML elasticity with a base point from the data
base_price = float(df_quotes["quoted_price"].median())
base_prob = float(df_quotes["converted"].mean())
dml_elasticity = est.elasticity_

print(f"Base price: £{base_price:.0f}")
print(f"Base conversion rate: {base_prob:.1%}")
print(f"DML elasticity: {dml_elasticity:.3f}")

demand_curve = DemandCurve(
    elasticity=dml_elasticity,
    base_price=base_price,
    base_prob=base_prob,
    functional_form="semi_log",
)

# COMMAND ----------

# Evaluate across a price range
prices, probs = demand_curve.evaluate(price_range=(300, 900), n_points=60)
print("Demand curve (price → P(buy)):")
print(f"  At £300: {probs[0]:.1%}")
print(f"  At £{base_price:.0f} (base): {demand_curve._evaluate_parametric(np.array([base_price]))[0]:.1%}")
print(f"  At £900: {probs[-1]:.1%}")

# COMMAND ----------

# You can also build a demand curve from a fitted model directly
model_curve = DemandCurve(
    model=conv_model,
    functional_form="model",
    price_col="quoted_price",
)

# Evaluate the model-based curve on the portfolio
prices_mc, probs_mc = model_curve.evaluate(
    price_range=(300, 900),
    n_points=20,
    context=df_quotes.head(5000),  # use 5k policies as reference portfolio
)
print("\nModel-based demand curve (averaged over 5k policies):")
for p, prob in zip(prices_mc[::5], probs_mc[::5]):
    print(f"  £{p:.0f}: {prob:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Optimal Pricing for a Segment
# MAGIC
# MAGIC Given the demand curve and a cost structure, find the profit-maximising price.
# MAGIC
# MAGIC Objective: max E[profit] = P(buy | price) × (price - expected_loss - expenses)
# MAGIC
# MAGIC Constraints (GIPP-aware):
# MAGIC - price >= min_price (regulatory minimum / anti-competitive floor)
# MAGIC - price <= max_price (or ENBP for renewals)
# MAGIC - P(buy | price) >= min_volume_rate (volume floor for portfolio planning)
# MAGIC - margin >= min_margin_rate (solvency / loss ratio target)

# COMMAND ----------

from insurance_demand import OptimalPrice

# Segment: London, vehicle group 2, standard risk
# Expected loss from risk model: £380 (technical premium)
segment_loss = 380.0
segment_expenses = 0.15   # 15% expense ratio (commission, admin)

opt = OptimalPrice(
    demand_curve=demand_curve,
    expected_loss=segment_loss,
    expense_ratio=segment_expenses,
    min_price=200.0,
    max_price=900.0,
    min_margin_rate=0.05,  # must cover 5% minimum margin
)

result = opt.optimise()

print("=== Price Optimisation Result ===")
print(f"Optimal price:         £{result.optimal_price:.2f}")
print(f"P(conversion):         {result.conversion_prob:.1%}")
print(f"Margin per policy:     £{result.expected_margin:.2f}")
print(f"Expected profit/quote: £{result.expected_profit:.2f}")
print(f"Active constraints:    {result.constraints_active if result.constraints_active else 'none'}")
print(f"Converged:             {result.converged}")

# COMMAND ----------

# Profit curve: see how profit varies across the price range
profit_df = opt.profit_curve(n_points=50)
print("\nProfit curve sample:")
print(profit_df.iloc[::10][["price", "conversion_prob", "margin", "expected_profit"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Renewal Segment with ENBP Constraint
# MAGIC
# MAGIC For renewal pricing, the optimal price is capped at the ENBP. Here we show
# MAGIC the same optimisation run with a tight ENBP constraint.

# COMMAND ----------

# Same segment, but this is a renewal — ENBP = current new business price
enbp = 650.0  # Our NB price for this risk through this channel

opt_renewal = OptimalPrice(
    demand_curve=demand_curve,
    expected_loss=segment_loss,
    expense_ratio=segment_expenses,
    min_price=200.0,
    max_price=900.0,
    enbp=enbp,  # PS21/11 constraint: cannot exceed NB price
)

result_renewal = opt_renewal.optimise()

print("=== Renewal Pricing (ENBP-Constrained) ===")
print(f"ENBP ceiling:          £{enbp:.2f}")
print(f"Optimal renewal price: £{result_renewal.optimal_price:.2f}")
print(f"P(renewal):            {result_renewal.conversion_prob:.1%}")
print(f"Expected profit/quote: £{result_renewal.expected_profit:.2f}")
print(f"Active constraints:    {result_renewal.constraints_active}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. ENBP Compliance Audit (PS21/11)
# MAGIC
# MAGIC FCA PS21/11 requires that renewal prices do not exceed the equivalent new
# MAGIC business price (ENBP) for the same risk through the same channel.
# MAGIC
# MAGIC The synthetic data was generated with ENBP compliance enforced, so we should
# MAGIC see zero breaches. We then demonstrate what breach detection looks like.

# COMMAND ----------

from insurance_demand.compliance import ENBPChecker, price_walking_report

checker = ENBPChecker(
    renewal_price_col="renewal_price",
    nb_price_col="nb_equivalent_price",
    channel_col="channel",
    policy_id_col="policy_id",
    tolerance=0.0,
)

report = checker.check(df_renewals)
print(report)
print("\nBreaches by channel:")
print(report.by_channel.to_string(index=False))

# COMMAND ----------

# Now demonstrate breach detection with artificially non-compliant data
df_bad = df_renewals.to_pandas().copy()
# Simulate a pricing error: 500 policies renewed at 12% above ENBP
df_bad.loc[:499, "renewal_price"] = df_bad.loc[:499, "nb_equivalent_price"] * 1.12

report_bad = checker.check(df_bad)
print(f"\n=== Simulated ENBP Breach Report ===")
print(report_bad)
print("\nBreaches by channel:")
print(report_bad.by_channel.to_string(index=False))
print(f"\nSample breaching policies:")
print(report_bad.breach_detail.head(5).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Price Walking Diagnostic
# MAGIC
# MAGIC The FCA's PS21/11 evaluation (EP25/2, July 2025) specifically looks for
# MAGIC systematic tenure-based price patterns. This report shows whether renewal
# MAGIC prices correlate with tenure — a red flag for residual price walking.

# COMMAND ----------

walking = price_walking_report(
    df_renewals,
    renewal_price_col="renewal_price",
    tenure_col="tenure_years",
    channel_col="channel",
    nb_price_col="nb_equivalent_price",
    n_tenure_bins=5,
)
print("Price by tenure band and channel:")
print(walking.to_string(index=False))
print()
print("Interpretation: If mean_price_to_enbp increases with tenure band, that's a")
print("potential PS21/11 concern — longer-tenured customers paying more relative to")
print("their ENBP. A flat or declining pattern is the target post-GIPP.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Integration with rate-optimiser
# MAGIC
# MAGIC The `insurance-demand` library is designed to feed the `rate-optimiser` library.
# MAGIC Here we show the handoff: converting a ConversionModel or DemandCurve into a
# MAGIC callable that rate-optimiser's DemandModel wrapper can consume.

# COMMAND ----------

# From a fitted conversion model
demand_callable = conv_model.as_demand_callable()

# Test it: give it price ratios (price/tech_premium) and feature data
import polars as pl
test_features = df_quotes.head(100)
price_ratios = np.linspace(0.9, 1.2, 100)

probs = demand_callable(price_ratios, test_features)
print("Demand callable output (first 5 predictions):")
for ratio, prob in zip(price_ratios[:5], probs[:5]):
    print(f"  price_ratio={ratio:.2f}: P(buy)={prob:.3f}")

# COMMAND ----------

# From a demand curve (parametric)
curve_callable = demand_curve.as_demand_callable()
probs_curve = curve_callable(price_ratios)
print("\nParametric curve callable output (first 5):")
for ratio, prob in zip(price_ratios[:5], probs_curve[:5]):
    print(f"  price_ratio={ratio:.2f}: P(buy)={prob:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This demo showed the full `insurance-demand` workflow:
# MAGIC
# MAGIC | Step | What we did | Key result |
# MAGIC |------|-------------|------------|
# MAGIC | Data | Generated 150k quotes + 80k renewals with known DGP | True elasticity: -2.0 |
# MAGIC | ConversionModel | Logistic GLM on quote data | Observed vs. fitted by factor |
# MAGIC | Naive elasticity | From logistic coefficients | Biased (confounding present) |
# MAGIC | DML elasticity | ElasticityEstimator with CatBoost nuisance | Unbiased, close to -2.0 |
# MAGIC | RetentionModel | Logistic on renewal data | DD payers stick, price change matters |
# MAGIC | DemandCurve | Price → P(buy) using DML elasticity | Evaluated over £300-900 range |
# MAGIC | OptimalPrice | Constrained profit maximisation | Optimal price with ENBP cap |
# MAGIC | ENBPChecker | PS21/11 compliance audit | Zero breaches in compliant data |
# MAGIC | Price walking | Tenure-price diagnostic | Flat pattern = GIPP-compliant |
# MAGIC
# MAGIC **Next steps**:
# MAGIC - Replace synthetic data with your actual quote/bind records
# MAGIC - Supply competitor price data to improve DML identification (use as instrument)
# MAGIC - Pass demand callables to `rate-optimiser` for full portfolio optimisation
# MAGIC - Set up quarterly DML re-estimation as pricing conditions change
