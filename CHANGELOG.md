# Changelog

All notable changes to ClimAID are recorded here. This file is the single source for the
changelog: the documentation page (`site_docs/changelog.md`) includes it unchanged.

ClimAID is in the 0.x series and **under active testing**: behaviour and APIs may still change between
minor versions, and changes that alter results are always listed under *Changed behaviour*.

## 0.4.1 — 2026-10-02

New features since 0.4.0: the built-in assistant (`climaid ai`, terminal and chat page) and SARIMAX as an
optional v2 model. No change to the results of existing models.

### Added
- **SARIMAX** (`sarimax`, `climaid.forecasting_v2.sarimax`): seasonal ARIMA with climate as external
  regressors, the classic benchmark in climate-and-disease forecasting. Fitted on log(1 + cases) with
  standardised temperature, rainfall, humidity and ENSO at one lag. Tuning, as for every v2 model, is
  compulsory and leakage-safe: the orders (p, d, q)(P, D, Q)₁₂ and the climate lag (0–3 months) are chosen by
  time-ordered cross-validation inside the training period, keeping the default (1, 0, 0)(1, 0, 0)₁₂ unless a
  candidate beats it. Its ranges are calibrated on the backtests like every model. Monthly data only; if it
  cannot be fitted it is left out of the run with a warning. Optional (not a default model); 22 v2 models in
  total. `statsmodels` is now a required dependency. Synthetic benchmark (WIS relative to the baseline):
  0.63 seasonal, 0.58 non-seasonal (best of all models), 0.84 realistic (worse than Poisson's 0.63, with
  under-covering ranges); details on the *Benchmarks* page.
- **`climaid ai`: built-in assistant** (`climaid.assistant`). A conversational guide in the terminal that
  works offline and uses no AI model: it recognises common requests and settings in plain language, asks for
  anything missing, checks the data file, shows a plan and waits for confirmation, runs `forecast_v2()` or
  `project_v2()`, and explains the results using the reports' own rule-based text (trust rating, month-by-month
  ranges, caveats). Questions are answered from curated summaries and a search of the bundled documentation,
  with links. Every number it shows comes from ClimAID's results. Documented in *ClimAID assistant*.
- **Assistant chat page** in the browser interface (`/assistant.html`, *Assistant* in the dashboard menu, or
  `climaid ai --browser`): the same assistant with file uploads, suggestion buttons and background runs with
  progress (`climaid/browser_ui/assistant_api.py`, one conversation per browser session).
- **Methods explanations in the assistant** (`climaid.assistant.methods`): 24 topics, from data checks, lags,
  the models (including SARIMAX), tuning, backtests, calibration and WIS to bias correction, pooling, the thermal curve, leakage
  safeguards, v1 and limitations, each in plain language with a technical version on request ("more detail");
  "how was this forecast made?" describes a run from its own metadata.
- `benchmarks/run_benchmarks.py --models a,b,...` chooses the models to benchmark.

### Documentation
- The *Aedes aegypti* thermal-curve preset is documented as taken from Mordecai et al. (2017); the release
  instructions in `README_DOCS.md` describe the tag-triggered PyPI workflow.
- New *ClimAID assistant* page; SARIMAX added to the model list and the *Benchmarks* page.

## 0.4.0 — 2026-09-29

First public release of **ClimAID v2**. The previous public release was 0.1.2 (v1 only).

The version numbers 0.2.0 and 0.3.0 were used for development builds of v2 and were not released;
everything they contained is included in 0.4.0 and described below. Where an entry says a behaviour
changed, the comparison is with those builds or with 0.1.x.

> **Rerun v1 analyses.** Several leakage bugs in the v1 pipeline were fixed (see *Changed behaviour*).
> Metrics reported with 0.1.x were optimistic; rerun analyses before comparing or citing them.

### ClimAID v2 (new, additive)
ClimAID v2 is an additive upgrade: the v1 disease model, full model registry, CMIP6/SSP projection
engine, visualisation, deterministic C-DSI reporting, optional local-LLM reporting, terminal wizard,
browser interface and dataset utilities remain available and import-compatible.
- Climate-mandatory probabilistic forecasting engine (`climaid.forecasting_v2`,
  `DiseaseModel.forecast_v2()`).
- Climate-informed stochastic renewal model with a discretised gamma generation interval;
  susceptible depletion when a valid population at risk is supplied, otherwise a relative-incidence
  formulation (no population is invented).
- Seasonal-naive probabilistic baseline, used as the benchmark in every report.
- v2 wrappers around all installed v1 model families, with temporal out-of-fold residual learning.
- Probabilistic quantile forecasts; WIS, RMSE, MAE and 50/80/95% interval coverage.
- Rolling historical hindcasts from explicit forecast origins.
- Explicit forecast-origin climate source (`forecast_climate_source`: `observed`, `projection`, `auto`).
- v2 forecast report with a validation contract and methodological warnings, optionally including the
  v1 C-DSI report.
- Hardening: future disease observations are never used as predictors; climate feature statistics
  are fitted only on data available at the origin; browser uploads use unique temporary filenames;
  the browser launcher waits for the API server; reports are served from `/reports/`.

### Changed behaviour (results will differ from earlier versions)
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
  Poisson histogram gradient boosting, SVR, k-nearest neighbours.
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
- **Reproducibility on multi-core machines.** Tree ensembles and boosting libraries were built with
  `n_jobs=-1`. Multi-threaded Random Forest adds up its trees' predictions in whatever order the threads
  finish, so the same data and `random_state` gave slightly different v1 lag-search results from run to run
  (seen in CI as `val_rmse` 3.33332 vs 3.33371); v2's tree models were built the same way. Every model ClimAID
  builds now runs single-threaded (`model_registry.single_threaded`); parallelism stays at the configuration
  level (`optimize_lags(n_jobs=...)`), where it does not change results. Repeated runs now give identical
  results.
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

### Documentation and packaging
- **Packaging fix:** the built-in South Asia climate data (`climaid/data/*.csv`) is included in the package
  again. It had been dropped from the package-data list after 0.1.2, so an installed copy could not load the
  built-in climate data.
- The package version is defined once, in `climaid/__init__.py`; `pyproject.toml`, the dashboard and the
  documentation banner read it from there. A test checks that every version mention in the documentation
  matches it.
- One changelog (this file); the former `CHANGELOG_v2.md` is merged into the 0.4.0 entry above.
- Documentation rebuilt and re-bundled for 0.4.0 (fixes garbled characters and stale 0.3.0 references in the
  bundled pages); `site_docs/` is tracked in git again (it had been listed in `.gitignore`).
- `pyproject.toml`: project links (documentation, changelog, paper), classifiers and licence file for PyPI.
- `climaid --version` prints the installed version. The PyPI publish workflow runs for version tags
  (`vX.Y.Z`) and published GitHub releases, in the `release` environment.
- The model-registry test checks XGBoost, LightGBM and CatBoost only when they are installed, so the fast CI
  job (installed without `[ml]`) passes.

### Known limitations
- Single-district scenario projections can understate warming effects when temperature,
  rainfall and humidity share a seasonal cycle; use `extra_districts`, `lag_selection="v1"`
  or a thermal curve where justified.
- The `aedes_aegypti_mordecai2017` curve values (17.8 / 29.1 / 34.6 °C) are taken from Mordecai et al. (2017);
  the curve is a simple suitability shape, not a re-implementation of their R0(T) model.
- `generate_report(style="policy")` does not route to `policy_brief()` (open decision).

## 0.1.2

Last v1-only release: stacked (base → residual → correction) climate–disease model with lag
optimisation, CMIP6/SSP projections, C-DSI and optional local-LLM reports, terminal wizard and
browser interface.
