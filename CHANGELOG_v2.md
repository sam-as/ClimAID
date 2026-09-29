# ClimAID 0.2.0 — v2 Additive Changelog

## Core principle
ClimAID v2 is an **additive upgrade** to the original ClimAID package. The legacy disease model, complete model registry, CMIP6/SSP projection engine, visualisation functions, deterministic C-DSI reporting, optional local-LLM reporting, terminal wizard, browser/FastAPI interface, and dataset utilities are retained.

## Added
- Climate-mandatory probabilistic forecasting engine.
- Climate-informed stochastic renewal forecasting.
- Discretised gamma generation interval.
- Susceptibility depletion when a valid population-at-risk quantity is supplied.
- Relative-incidence renewal mode when population is unavailable; no population is invented.
- Seasonal-naive probabilistic baseline.
- v2 wrappers around all installed legacy forecasting model families.
- Temporal out-of-fold residual learning.
- Probabilistic quantile forecasts and empirical uncertainty intervals.
- WIS, RMSE, MAE and empirical 50/80/95% coverage.
- Rolling historical hindcast evaluation.
- Explicit forecast-origin climate-source handling.
- Browser controls for v1 and v2 in the same interface.
- v2 forecast report with validation contract and methodological warnings.
- Optional inclusion of the preserved deterministic C-DSI report in the v2 report.

## Hardened
- Future disease observations are never supplied as predictors by the v2 forecasting engine.
- Forecast-origin climate statistics are fitted only on information available at the origin.
- Legacy browser uploads use unique temporary filenames.
- Browser launcher waits for the API server before opening the UI.
- Reports are explicitly served from `/reports/`.
- Legacy and v2 APIs can be run separately or together.

## Preserved
The original package files and public functionality remain available. The following core legacy files were preserved without source-code changes: `model_registry.py`, `model_parameters.py`, `climaid_projections.py`, `climate_data.py`, `districts.py`, `llm_client.py`, `projection_plots.py`, `utils.py`, and `cli.py`.
