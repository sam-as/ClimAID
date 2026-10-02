# ClimAID v2 forecasting

!!! warning "Under testing"
    The v2 forecasting engine is under active testing. It has been checked on synthetic data with a known
    answer (see [Status & validation](status.md)); accuracy on real surveillance data has not yet been
    established. Always compare the forecast with the simple "same as recent years" baseline in the report.

`DiseaseModel.forecast_v2()` forecasts monthly cases for the coming months, with a **likely range** around
every value, and tests itself on the past before you rely on it.

---

## What happens when you run a forecast

1. **Data checks.** Unreadable dates, non-numeric or negative counts and duplicate rows are removed or
   rejected, and every change is reported. Weekly or daily data are summed to monthly totals for v1.
2. **COVID-19 period** (`exclude_period`, default `"2020"`). Case counts during COVID-19 were distorted by
   less testing, reporting and care-seeking. Each month in the chosen period is replaced in the training
   history with that month's typical value from other years, and is never used as a scoring target.
   Choose `"2020"` (recommended for South Asia), a custom period such as `"2020-03:2020-12"`, or `"none"`.
3. **Model fitting.** Each selected model learns from data **up to the forecast start only**.
4. **Tuning** (always on). Each machine-learning model's settings are tuned before use (see below).
5. **Backtests ("hindcasts").** The whole process is repeated from several earlier start dates, using only
   data available at each, and checked against what actually happened.
6. **Calibrated likely ranges.** The ranges are widened or narrowed, per lead time, so they would have
   contained the real number as often as they claim in those backtests.
7. **Report.** A plain-language summary with a trust rating, followed by full technical details.

---

## Models (22)

| Group | Models |
|---|---|
| Baselines | **Seasonal naïve** ("same as recent years", the benchmark to beat) |
| Mechanistic | **Climate renewal** transmission model (optional susceptible depletion if population is given) |
| Time series | **SARIMAX** (`sarimax`): seasonal ARIMA with climate as external inputs, on log(1 + cases); tuning chooses the orders (p, d, q)(P, D, Q)₁₂ and the climate lag (0–3 months). Monthly data only. Not a default: best on the non-seasonal benchmark but weaker on the realistic one (see [Benchmarks](benchmarks.md#sarimax)); slower than the regression models |
| Regression | Linear, Ridge, Lasso, Elastic net, **Poisson**, Tweedie, Smooth-curve Poisson (GAM-style), Bayesian ridge, Huber (spike-resistant) |
| Tree ensembles | **Random forest**, **Extra trees**, **Gradient boosting**, Hist gradient boosting (Poisson), XGBoost*, LightGBM*, CatBoost* |
| Other | Neural network (MLP), Support vector regression, Nearest neighbours |
| v1 inside v2 | **ClimAID v1 stacked model** (`v1_stack`): the full v1 pipeline in your chosen v1 mode (Fast 50 / Balanced 200 / Deep 500 trials, v1's models and full lag ranges), re-run at every backtest start date and tested exactly like the others. Slow. |

Default selection in **bold** (excluding `sarimax` and `v1_stack`). \*Needs `pip install "climaid[ml]"`.
Each machine-learning model is paired with a residual model trained on its time-ordered out-of-fold errors,
and all selected models are also combined into an **ensemble** (the median of their forecasts).

**Inputs used by the ML models:** temperature, rainfall and humidity at lags 0–3 months and their deviation
from the usual for the month; ENSO at lags 0–3; ENSO interactions (each climate variable × recent ENSO);
seasonal terms; and cases in the previous three months.

**Distributed lag effects.** Regression-type models (linear family, Poisson, Tweedie, smooth-curve Poisson,
Bayesian ridge, Huber) also receive distributed-lag features: each variable's effect is spread over a smooth
curve across lags 0–6 months (ENSO 0–12), so long ENSO lags are available with only a few extra features.
Tree-based and other flexible models combine raw lags themselves and do not use them by default
(`ClimateMLForecaster(distributed_lags=True/False)` overrides this).

!!! note "Evidence (synthetic benchmark, 4 start dates × 3 datasets)"
    Poisson regression improved on all three datasets with distributed lags (score relative to the baseline
    0.99 → 0.95, 0.73 → 0.67, 0.97 → 0.91). Tree models were mixed, so they are left without them.

---

## Tuning (compulsory)

| Preset | Trials per model | Use for |
|---|---|---|
| `fast` | 10 | A quick first look |
| `balanced` (default) | 30 | Most runs |
| `deep` | 80 | Final runs; slowest |
| integer, e.g. `50` | that number | Custom |

Settings are tested with time-ordered cross-validation **inside the training period only**, and backtests
re-tune at every start date, so no later data can leak in. If tuning does not beat the default settings,
the defaults are kept.

See [Tuning & optimisation trials](tuning.md) for how accuracy and runtime change with the number of trials.

!!! note "Tuning helps on average, not always"
    On synthetic data, tuning improved the probabilistic score in 7 of 10 model–dataset pairs (large gains for
    Poisson and gradient boosting, small losses for some tree models). Tuning optimises one-month-ahead error,
    which usually, but not always, carries over to 12-month forecasts.

---

## Main options

| Option | Default | Meaning |
|---|---|---|
| `forecast_origin` | last observation | Use data up to this date |
| `horizon` | 12 | Months to forecast |
| `models` | 6 defaults | Any of the 22 models |
| `tuning` | `"balanced"` | Tuning effort (cannot be switched off) |
| `run_hindcasts` / `hindcast_origins` | True / 4 | Backtests used for checking and calibration |
| `calibrate_intervals` | True | Calibrate likely ranges from the backtests |
| `exclude_period` | `"2020"` | COVID-19 period: `"2020"`, `"YYYY-MM:YYYY-MM"` or `"none"` (the older `drop_2020=True/False` still works) |
| `population_at_risk` | None | Enables susceptible depletion in the renewal model |

!!! tip "Keep forecasts short"
    Forecasts use recent case counts, which stop being informative after about a year. For multi-year
    questions use the [climate scenario outlook](scenarios.md).

---

## Changes in 0.4.0 that alter results

Hindcasts now stop at the forecast origin; the COVID-19 period (2020) is excluded by default; ranges are calibrated;
models are tuned; defaults were corrected (for example random forests now use ClimAID's 400 trees rather
than 100). Results will differ from earlier versions and development builds. See the
[Changelog](../changelog.md).
