# Migrating to ClimAID v2

ClimAID v2 ships with ClimAID **0.4.0** (upgrading from 0.1.x). It is additive: it does not require users to
abandon the existing API.

> **v1 results change in 0.4.0.** Leakage bugs in the v1 pipeline were fixed, so v1 metrics produced with 0.1.x
> were optimistic. Rerun v1 analyses before comparing or citing them. Every change that alters results is listed
> under *Changed behaviour* in [CHANGELOG.md](CHANGELOG.md).

## Existing v1 usage remains

```python
from climaid.climaid_model import DiseaseModel

model = DiseaseModel(...)
model.optimize_lags(...)
model.train_final_model()
```

The original model registry, CMIP6/SSP projection methods, `DiseaseProjection`, `DiseaseVisualizer`, C-DSI deterministic reporting, and optional LLM reporting remain available.

## New v2 usage

```python
from climaid.forecasting_v2 import ClimaidV2Forecaster

forecaster = ClimaidV2Forecaster(
    models=["seasonal_naive", "renewal", "random_forest"],
    population_at_risk=None,
)
forecaster.fit(disease_data, climate_data, cutoff="2023-12-31")
forecast = forecaster.predict(future_climate, horizon=12, n_simulations=2000)
```

Climate is mandatory for v2.

## Through `DiseaseModel`

```python
result = model.forecast_v2(
    forecast_origin="2023-12-31",
    horizon=12,
    models=("seasonal_naive", "renewal", "random_forest"),
    population_at_risk=None,
    forecast_climate_source="projection",
    run_hindcasts=True,
    tuning="balanced",          # fast | balanced | deep | an integer number of trials (tuning is always on)
    exclude_period="2020",      # COVID-19 period: "2020" (default), "YYYY-MM:YYYY-MM" or "none"
)
```

Omit `models` to use the six defaults (`seasonal_naive, renewal, poisson, random_forest, extra_trees,
gradient_boosting`). For climate-scenario outlooks use `model.project_v2(...)`; see the
[documentation](https://sam-as.github.io/ClimAID/guide/scenarios/).

## Population

A population value is **not** fabricated when unavailable. With no population column/value, renewal uses a relative-incidence formulation and reports that susceptible depletion could not be applied.

## Forecast climate source

- `observed`: use observed climate after the cutoff; only valid when the full horizon is available.
- `projection`: use future projected/scenario climate.
- `auto`: use observed climate for historical hindcasts when a full post-origin horizon exists; otherwise use projection climate.

Use the explicit source options for operational workflows so the information set is unambiguous.

## Further reading

- User guide: [v2 forecasting](https://sam-as.github.io/ClimAID/guide/v2_forecasting/)
- API reference: [ClimAID v2 API](https://sam-as.github.io/ClimAID/api/v2/)
- Reviewer concerns and responses: [V2_FEEDBACK_RESPONSE.md](V2_FEEDBACK_RESPONSE.md)
