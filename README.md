# ClimAID 0.4.0 â€” v2

ClimAID v2 is an additive upgrade to the original ClimAID climateâ€“disease modelling toolkit. **Climate remains mandatory for every v2 forecasting run**, while disease history and epidemic dynamics become first-class components.

## What remains from ClimAID v1

The complete v1 package is retained: all registered ML models, lag optimisation, residual/correction workflow, CMIP6/SSP scenario projection, dual-baseline risk functions, interactive scientific visualisation, deterministic C-DSI reporting, optional local LLM reporting, terminal wizard, and browser/FastAPI interface.

## What v2 adds

- Climate-informed stochastic renewal forecasting.
- Gamma generation-interval formulation.
- Susceptible depletion when population-at-risk information is available.
- Relative-incidence renewal mode when population is unavailable.
- Seasonal-naive benchmark.
- All installed legacy ML model families as v2 forecasting engines.
- Temporal out-of-fold residual learning.
- Probabilistic quantile prediction.
- WIS, RMSE, MAE and interval-coverage metrics.
- Rolling hindcasts using explicit forecast origins.
- Forecast-origin climate-source contract.
- Additive v2 browser and terminal controls.
- v2 probabilistic HTML reporting alongside the existing C-DSI report.

## Install

```bash
pip install -e .
```

## Browser

```bash
climaid browse
```

## Terminal

```bash
climaid wizard
```

## Programmatic v2 forecasting

```python
from climaid.forecasting_v2 import ClimaidV2Forecaster

model = ClimaidV2Forecaster(
    models=["seasonal_naive", "renewal", "random_forest", "xgboost"],
    population_at_risk=None,
)
model.fit(disease_data, climate_data, cutoff="2023-12-31")
forecast = model.predict(future_climate, horizon=12, n_simulations=2000)
```

See `MIGRATION_v2.md` and `V2_REVIEW_RESPONSE.md` for implementation and reviewer-issue mapping.

## What's new in 0.4.0
- Leakage fixes in the v1 pipeline (annual-average climate feature, test-set reuse during lag
  optimisation, projection features that did not match training) and in v2 hindcasts.
  **Reported v1 metrics from earlier versions were optimistic; rerun before citing them.**
- Calibrated v2 forecast intervals (split-conformal, per lead time, from hindcasts).
- `DiseaseModel.project_v2()`: hybrid near-term forecast + CMIP6 scenario outlook with
  bias correction, structural-uncertainty averaging, optional multi-district pooling,
  thermal-suitability curve, population scaling and a long-term backtest.
- Dashboard: separate v2 (default) and v1 pages, info tips on every control, `drop_2020`.

See `CHANGELOG.md` for details and `REVIEW_FINDINGS_2026-09.md` for the evidence behind each change.




