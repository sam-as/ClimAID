# Changelog

## 0.3.0

### Changed behaviour (results will differ from 0.2.0)
- v1 `YA_*` climate feature is a trailing 12-month mean (was the whole-calendar-year mean,
  which included future months).
- v1 lag optimisation selects configurations on a selection-validation split inside the
  training period; the test set is used once, for the final report. Lag-search result columns
  renamed `rmse`/`r2` → `val_rmse`/`val_r2`.
- v1 base-model Optuna tuning scores on the validation split (was in-sample).
- v1 Stage 1 pruning keeps the best `(100 - percentile)`% capped at `top_k` (kept ~90% before).
- v1 `predict()` applies the fitted correction model (a typo meant it never did).
- `random_state` is passed to every estimator that accepts it.
- CMIP6 projection features (`MA_*`, `YA_*`) now match the training definitions.
- v1 `drop_2020` is configurable, and `False` now actually keeps 2020.
- v2 hindcasts use only data at or before the forecast origin and never score replaced months.
- v2 intervals are calibrated from hindcasts by default (`calibrate_intervals=True`).
- v2 default models: `seasonal_naive, renewal, poisson, random_forest, extra_trees, gradient_boosting`.
- v2 `drop_2020=True` by default (2020 replaced by monthly medians in the training history).
- Renewal model: expected cases capped at 5x the training maximum, with a warning.
- C-DSI v2: interprets the best-scoring model; sample-size-aware calibration tolerance;
  whole-trajectory trend sentence; model-disagreement flag; recalibration note.
- WIS uses the canonical (K + 1/2) normalisation; seasonal-naive uses weekly lags on weekly data.

### Added
- Documentation bundled in the package (`climaid/documentation`): served by the dashboard at
  `/documentation/`, opened offline with `climaid docs`. "How many trials do I need?" links next to the
  v2 tuning and v1 preset/trials settings open the new *Tuning & optimisation trials* page.
- Trials study (`benchmarks/trials_study.py`, `summarise_trials.py`) with results in the docs.
- Distributed lag effects (`forecasting_v2/distributed_lag.py`): smooth lag curves over 0–6 months
  (ENSO 0–12). On by default for regression-type v2 models; optional in the scenario outlook
  (`lag_selection="distributed"`), which reports the fitted lag curves.
- `v1_mode` for `v1_stack` and `lag_selection="v1"`; "Use v1 lags" now runs v1's lag search on data
  before the forecast start (it previously never worked from the dashboard).
- COVID-19 disruption period choice (`exclude_period`): **2020** (default), a **custom period**
  (`"YYYY-MM:YYYY-MM"`) or **none**, applied the same way in v1, v2, `v1_stack` and the scenario outlook.
  Replaces the 2020-only switch in the dashboard and wizard; `drop_2020=True/False` still works.
- NDMC (National Disease Modelling Consortium) logo in the browser-interface header, the v2 forecast and
  scenario report headers (embedded, so reports stay self-contained) and the documentation.
- `benchmarks/`: all synthetic generators (incl. a realistic dataset with reporting changes, outbreak
  years, gaps and data-entry errors) and a rolling-origin benchmark runner with saved results.
- `v1_stack` v2 model: v1's lag-optimised stacked pipeline run inside v2 (same held-out and
  hindcast tests, calibrated ranges, joins the ensemble).
- Scenario outlook: tree-model comparison (random forest, gradient boosting) with the share of
  projected months outside the training climate; dashboard toggle.
- Compulsory v2 hyperparameter tuning with effort presets (Fast / Balanced / Deep / custom trials),
  leakage-safe (time-ordered CV inside training; re-tuned per hindcast origin; defaults kept if
  not beaten). Dashboard "Model tuning" menu, wizard prompt, `tuning=` in `forecast_v2`/`project_v2`.
- Seven v2 models (scikit-learn only): Tweedie, spline Poisson (GAM-style), Bayesian ridge, Huber,
  Poisson histogram gradient boosting, SVR, k-nearest neighbours (21 v2 models in total).
- v2 ENSO interaction features (on by default).
- Plain-language report layer (`climaid.reporting_plain`) for both v2 reports: "The short version",
  month-by-month table (expected cases, likely range, compared with a typical year, what actually
  happened), a Good / Moderate / Low trust rating with its reasons, plain caveats and a glossary.
  All technical content moved into a collapsible "Technical details" section. Every sentence is
  rule-based from the run's own numbers (no LLM). `generate_forecast_report` gains `history`,
  `observed` and `exclude_times`; the charting library is embedded once per report.
- `DiseaseModel.project_v2()` and `climaid.forecasting_v2.scenario`: hybrid near-term +
  CMIP6 scenario outlook (bias correction, seasonal/anomaly response, lag-structure ensemble,
  multi-district pooling, thermal-suitability curve, population scaling, long-term backtest),
  with its own report (`climaid.reporting_scenario`).
- `climaid.forecasting_v2.calibration` (split-conformal interval calibration).
- Disease-data validation (bad dates, non-numeric or negative counts, duplicates,
  sub-monthly data summed to monthly).
- Dashboard: separate v2/v1 pages (v2 default), info tips, `drop_2020`, scenario outlook card.
- Terminal wizard: pipeline choice and `drop_2020` prompt.
- CI workflow; `slow` test marker.

### Fixed
- `v1_stack` ran a lighter search than ClimAID v1 (3–15 trials, narrower lags); it now uses v1's own
  Fast / Balanced / Deep modes exactly. v1 modes are defined once (`model_parameters.V1_MODES`) and used
  by the dashboard, wizard and v2.
- Wizard Deep preset named a non-existent model (`elastic_net`) and its Fast preset differed from the
  dashboard's; both now use the shared definitions.
- Leftover debug prints on import of the dataset manager; incorrect docstrings for
  `train_final_model` and `load_cmip6`.
- v2 ignored ClimAID's model defaults (name mismatch: `random_forest` vs `rf`), so e.g. random
  forests had 100 trees instead of 400; linear-type v2 models were fitted unscaled; Poisson's
  `max_iter=300` did not converge.
- LLM-fallback crash in `DiseaseReporter`; legacy report modes mislabelled without an LLM;
  hindcast failures silently swallowed; several wizard `NameError`/`UnboundLocalError` paths;
  `project_multi_model_ssp` `KeyError` without RMSE.

### Known limitations
- Single-district scenario projections can understate warming effects when temperature,
  rainfall and humidity share a seasonal cycle; use `extra_districts`, `lag_selection="v1"`
  or a thermal curve where justified.
- The `aedes_aegypti_mordecai2017` curve values (17.8 / 29.1 / 34.6 °C) must be verified
  against the source before publication.
- `generate_report(style="policy")` does not route to `policy_brief()` (open decision).
