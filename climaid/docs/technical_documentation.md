# ClimAID Technical Documentation

## AI-integrated Calibration-Enhanced Stacked Disease Modelling Framework for Climate-Driven Multi-Disease Risk Projection

---

## 1. Overview

ClimAID is a reproducible, climate-driven infectious disease modelling framework designed to quantify and project disease risk under historical and future climate conditions. The framework integrates epidemiological data, climate covariates, and CMIP6 climate projections within a calibration-enhanced stacked modelling architecture.

Originally developed for dengue modelling, ClimAID has been generalized into a multi-disease framework applicable to climate-sensitive infectious diseases, particularly vector-borne diseases such as dengue and malaria.

---

## 2. Core Modelling Architecture

### 2.1 Calibration-Enhanced Stacked Framework

ClimAID employs a three-stage modelling pipeline:

1. **Base Model (Primary Climate–Disease Signal Extraction)**
   Captures nonlinear relationships between climate drivers and disease incidence.

2. **Residual Model (Error Correction Layer)**
   Learns unexplained variance from residuals of the base model:
   Residual = Observed Cases − Base Predictions

3. **Calibration Layer (Isotonic / Regression Correction)**
   Applies monotonic calibration to improve epidemiological realism and prediction reliability.

Final Prediction:

```
Final = Calibration(Base + Residual)
```

This architecture reduces bias, improves calibration, and enhances robustness under non-stationary climate conditions.

---

## 3. Climate–Disease Data Integration

### 3.1 Disease Data

* District-level disease incidence (Excel/CSV)
* Target variable: disease case counts
* Supports multi-disease inputs (e.g., dengue, malaria)

### 3.2 Climate Covariates

The framework supports lagged environmental predictors including:

* Mean temperature
* Rainfall
* Specific humidity
* Additional climate anomalies (if available)

Lagged variables (lag1–lag3) are automatically generated to capture delayed epidemiological responses.

Two smoothed covariates are also built, both strictly backward-looking over the time-sorted
climate series:

* `YA_*`: trailing 12-month mean (up to and including the current month)
* `MA_*`: trailing 120-month (10-year) mean

(Before v0.4.0, `YA_*` was the whole-calendar-year mean, so a January value included that
year's February–December climate. See `CHANGELOG.md`.) The CMIP6 projection pipeline computes these
features with exactly the same definitions as training.

---

## 4. Automated Lag Optimization

### 4.1 Optimization Strategy

Lag structures and model configurations are optimized using Optuna with a seeded TPESampler for reproducibility.

All candidate configurations are screened, tuned and ranked on a chronological
**selection-validation split** carved from the end of the training period (by default the last
20% of training months). The test set is never used during search (see 8.1).

Search runs in two stages. Stage 1 screens every lag combination × base model; with the
default `pruning_strategy="percentile", percentile=90` the best 10% of configurations are kept,
capped at `top_k`. Stage 2 tunes the retained configurations (base, residual and correction
models) with Optuna, each scored on the selection-validation split.

Search space includes:

* Feature subsets (lagged climate variables)
* Base model type
* Residual model type
* Calibration model type
* Hyperparameters

### 4.2 Deterministic Parallel Evaluation

Lag optimization is executed using joblib parallelization with configuration-aware seeding:

* Unique seed per configuration
* Seeded Optuna sampler
* `random_state` passed to every estimator that accepts it (detected from the constructor
  signature or `get_params()`; before v0.4.0 this check never matched and seeds were dropped)

This ensures reproducible model selection across runs.

---

## 5. Model Registry and Supported Algorithms

### 5.1 Base and Residual Models

Supported models include:

* Random Forest (rf)
* XGBoost (xgb)
* LightGBM (lgbm)
* CatBoost
* Gradient Boosting
* Extra Trees
* ElasticNet, Ridge, Lasso
* Poisson Regression (epidemiologically interpretable)
* Neural Networks (MLP)

### 5.2 Calibration Models

* Isotonic Regression (default; monotonic calibration)
* Linear correction models
* Optional no-calibration baseline

---

## 6. CMIP6 Climate Projection Integration

### 6.1 Projection Workflow

```
CMIP6 Climate Data → Feature Engineering → Lag Reconstruction
→ Model Inference → Multi-GCM Projections → Ensemble Analysis
```

### 6.2 Supported Features

* Multi-GCM projections (e.g., ACCESS-ESM1-5, GFDL-ESM4)
* Multi-SSP scenarios (SSP126, SSP245, SSP370, SSP585)
* Ensemble mean projections
* District-level future disease risk estimation
* ClimAID v2 hybrid climate-scenario outlook with probabilistic ranges (see section 13)

CMIP6 integration enables scenario-based epidemiological forecasting under climate change pathways.

---

## 7. Reproducibility and Determinism

ClimAID is designed as a fully reproducible modelling framework.

### 7.1 Seed Control Mechanisms

* Global seed initialization in model constructor
* Seeded Optuna TPESampler
* Model-level random_state injection
* Deterministic train/test split reuse
* Parallel seed offsets for configuration search

### 7.2 Deterministic Pipeline Design

Given fixed:

* Dataset
* Random seed
* Model configuration

The pipeline produces numerically stable and reproducible results across runs.

---

## 8. Training and Evaluation Protocol

### 8.1 Data Splitting

Three disjoint chronological partitions are used:

* **Selection-training** and **selection-validation** (both inside the training period): used
  for every screening, tuning and accept/reject decision during lag optimization, including
  whether to keep the correction model.
* **Test** (after the training period): used exactly once, to report the final configuration's
  performance.

Before v0.4.0 the test set was used to rank all candidate configurations and then reported as
held-out performance, which biases reported R²/RMSE optimistically. By default 2020 is excluded
from training (`drop_2020=True`, COVID-19 reporting disruption).

### 8.2 Performance Metrics

Primary metrics:

* R² (coefficient of determination)
* RMSE (root mean squared error)

Lag optimization performance (validation-like) and final test performance are reported separately to ensure methodological transparency.

---

## 9. Visualization and Scientific Diagnostics

ClimAID provides built-in scientific visualization tools:

* Projection heatmaps (GCM × SSP)
* Multi-model projection grids
* Distribution and KDE diagnostics
* Historical vs predicted trend plots

These outputs support interpretability for both research and policy analysis.

---

## 10. Computational Design

* Parallel lag search using joblib
* Efficient pandas/xarray data handling
* Modular architecture for extensibility
* Offline-first execution (no cloud dependency)

---

## 11. Intended Research Applications

* Climate-sensitive infectious disease modelling
* Early warning systems
* Climate change health impact assessments
* Multi-disease comparative climate risk studies
* Policy-oriented climate-health forecasting

## 12. AI Integration and LLM-Based Interpretation Layer

### 12.1 Multi-Layer AI Integration in the Modelling Pipeline

ClimAID integrates artificial intelligence components across multiple stages of the climate–disease modelling workflow. Rather than functioning as a single predictive model, the framework operates as an AI-enhanced analytical system combining machine learning, automated optimization, calibration, and language model–based interpretation.

The AI integration operates across four layers:

1. Nonlinear machine learning models for climate–disease prediction
2. Automated model and lag discovery (AutoML)
3. Calibration-enhanced stacked learning
4. LLM-based automated scientific reporting

---

### 12.2 Machine Learning Core (Predictive AI Layer)

The primary predictive engine consists of supervised machine learning algorithms capable of modelling complex and nonlinear relationships between environmental drivers and disease incidence. Supported algorithms include tree-based ensembles, gradient boosting methods, neural networks, and generalized linear models.

This layer enables flexible learning of climate-sensitive transmission dynamics under non-stationary environmental conditions.

---

### 12.3 AutoML-Based Lag and Configuration Optimization

ClimAID incorporates a deterministic AutoML component using Optuna with a seeded TPESampler. This system automatically identifies:

* Optimal climate lag structures
* Feature subsets
* Model architectures
* Hyperparameter configurations

Configuration search is executed using parallel deterministic evaluation with configuration-specific seeds, ensuring reproducibility while maintaining exploration efficiency. This automated optimization reduces manual bias in lag selection, which is a critical challenge in climate–epidemiological modelling.

---

### 12.4 Calibration-Enhanced AI Predictions

Following base and residual model training, a calibration layer (e.g., isotonic regression) is applied to stacked predictions. This post-hoc learning step improves prediction reliability, reduces systematic bias, and enhances epidemiological interpretability of AI-generated disease projections.

---

### 12.5 Anomalous signal detection

To enhance early warning capability, a dual-baseline outbreak risk flagging module is implemented for both historical and projected disease trajectories. Projected high-risk periods are identified using 

(i) a historical percentile-based baseline derived from observed case distributions
(ii) a dynamic scenario-specific baseline computed within each GCM–SSP projection ensemble.

### 12.6 LLM-Based Automated Scientific Reporting (Interpretability AI Layer)

ClimAID includes an optional offline Large Language Model (LLM) integration module designed for automated interpretation of modelling outputs. The LLM component generates structured scientific and policy-oriented narratives from model artifacts, including:

* Projection summaries
* Model diagnostics
* Feature importance
* Climate scenario comparisons

The LLM client supports fully offline deployment via local models (e.g., Phi-3, Mistral through Ollama), ensuring data privacy and suitability for secure research or government environments.

---

### 12.7 Deterministic Fallback Reporting

To maintain reproducibility and reliability, the reporting system includes a deterministic fallback mechanism. If the LLM is unavailable, ClimAID automatically generates rule-based scientific reports using model artifacts. This design ensures that report generation remains stable, transparent, and reproducible across computational environments.

---

### 12.8 Role of AI in Climate–Health Decision Support

The LLM layer does not replace epidemiological modelling but functions as an interpretability and communication interface. By translating quantitative model outputs into structured narratives, the framework enhances accessibility of climate–disease insights for interdisciplinary stakeholders, including researchers, policymakers, and public health planners.

---

## 13. ClimAID v2: Probabilistic Forecasting and Climate-Scenario Outlook

### 13.1 Probabilistic forecasting (`DiseaseModel.forecast_v2`)

* Models (21): seasonal naïve benchmark, climate renewal model, and 19 statistical/ML learners:
  linear, ridge, lasso, elastic net, Poisson, Tweedie, spline Poisson (GAM-style), Bayesian ridge,
  Huber, random forest, extra trees, gradient boosting, Poisson histogram gradient boosting,
  XGBoost, LightGBM, CatBoost, MLP, SVR and k-nearest neighbours. Each ML learner is paired with a
  residual random forest fitted on temporal out-of-fold errors. Default set:
  `seasonal_naive, renewal, poisson, random_forest, extra_trees, gradient_boosting`, combined by a
  per-quantile median ensemble. Linear-type models are standardised before fitting.
* `v1_stack`: the full v1 pipeline (lag search, base → residual → correction) run as a v2 model at
  each origin, using only data up to that origin (its internal test year is the last complete
  training year, 2020 excluded). Future months are built with v1's projection feature builder.
  Starting ranges are point ± z × v1 test RMSE, then calibrated from hindcasts like every model.
  Search effort follows the tuning preset. Slow; not selected by default.
* Features: climate lags 0–3 and monthly anomalies, ENSO lags 0–3, seasonal harmonics, case lags
  1–3, and ENSO interactions (each climate variable at each lag, centred, × ENSO's mean over lags
  1–3; `add_enso_interactions=True`).
* Tuning (compulsory; `tuning="fast"|"balanced"|"deep"` = 10/30/80 Optuna trials per model, or an
  integer): candidate settings are scored by expanding-window, time-ordered cross-validation inside
  the training period (one-month-ahead MAE). Default settings are always scored too and kept if no
  candidate beats them. Hindcasts re-tune at every origin.
* Validation: held-out months after the forecast origin, plus rolling-origin hindcasts that use
  only data available before the forecast origin. Months replaced under `drop_2020` are never
  used as scoring targets.
* Interval calibration (`calibrate_intervals=True`): split-conformal rescaling of each model's
  intervals per lead-time band (1–3, 4–6, 7–12, 13–24 months), estimated from the hindcasts.
  A level is estimated within a band only with enough points (≈2/(1−level)), otherwise the
  pooled factor is used, and band factors are shrunk toward the pooled factor.
* Renewal stability cap: expected cases capped at 5× the training maximum (or the population);
  a warning is reported when the cap binds.
* Reporting: C-DSI v2 deterministic interpretation (best-scoring model, sample-size-aware
  calibration check, trend, model disagreement).

### 13.2 Hybrid climate-scenario outlook (`DiseaseModel.project_v2`)

1. Each GCM × SSP series is bias-corrected by monthly mean shift against the observed baseline
   (multiplicative for rainfall).
2. Months 1–12: the v2 ensemble forecast driven by each GCM's corrected climate; months 13–24:
   linear hand-over.
3. Long term: Poisson GLM with negative-binomial dispersion and year-block bootstrap.
   `response="seasonal"` (default) learns climate effects from the seasonal cycle;
   `response="anomaly"` only from year-to-year deviations and is always reported as a
   sensitivity check.
4. Lag structure (`lag_selection`): `"ensemble"` (default) averages every one-lag-per-variable
   structure within 2% of the best blocked-cross-validation score. Climate variables that share
   a seasonal cycle usually cannot be separated from one district's data, and the structures
   imply different warming responses; averaging carries that uncertainty into the ranges.
   `"v1"` reuses v1's selected lags.
5. Optional: `extra_districts` learns shared climate coefficients across districts (own
   intercepts); `temperature_curve` constrains the temperature response to a thermal-suitability
   curve; `population_projection` adds population-scaled results.
6. Tree comparison (`comparison_models`, default random forest and gradient boosting): tuned tree
   models fitted to the same history and run through the same corrected climate, reported next to
   the main projection with the share of projected months outside the training climate range
   (where tree models cannot extrapolate). Comparison only.
7. Backtest: the long-term model is refitted without the last `backtest_years` and checked
   against their observed cases using observed climate.
8. Ranges pool count noise, parameter uncertainty (bootstrap), structural uncertainty and
   climate-model spread.

Validation on synthetic data with known truth (see the
[Status & validation](https://sam-as.github.io/ClimAID/guide/status/) page):
single-district projections of a temperature-driven series were biased low (e.g. +27% vs a true
+60% change) with the truth inside the 10–90% range; pooling across six districts recovered the
true change within 3 percentage points across three seeds.

Assumptions: the climate–disease relationship is stationary; population (unless scaled),
immunity, vector control and reporting are held at historical levels; CMIP6 ENSO variability is
poorly represented.
