# Status & validation

!!! warning "ClimAID 0.4.0 is under active testing"
    The methods have been checked on **synthetic data with a known answer**. They have **not yet been
    validated on real surveillance data or on real CMIP6 projections**. Until that is done, treat all outputs as
    research estimates: check them against local knowledge and surveillance experience, and do not use them as
    the sole basis for public-health decisions.

This page is updated as testing progresses. Last update: version 0.4.0.

---

## What has been tested

| Area | How | Status |
|---|---|---|
| Code correctness | 100+ automated tests, including leakage checks, run on every change | Passing |
| Leakage | Tests that models never see data after the forecast start (v1 selection, v2 hindcasts, climate features, CMIP6 features) | Fixed and tested in 0.4.0 |
| v2 forecast accuracy | Rolling benchmark on synthetic data (below) | Tested (synthetic only) |
| Likely-range calibration | Coverage checked on months the calibration never saw | Tested (synthetic only) |
| Scenario outlook | Projected change compared with the known true change | Tested (synthetic only) |
| Real surveillance data | — | **Not yet done** |
| Real CMIP6 projections | — | **Not yet done** (could not be accessed in development) |

---

## Forecast benchmark (synthetic)

12-month forecasts from four start dates (2018, 2019, 2020 and 2022 forecast years) per dataset, through the
full `forecast_v2()` path with Fast tuning, backtests and calibrated ranges. The score is the probabilistic
error (WIS) **relative to the "same as recent years" baseline**: below 1.0 is better than the baseline.
See [Benchmarks & synthetic data](benchmarks.md) for how the datasets are built.

| Dataset | Ensemble | Random forest | Gradient boosting | Poisson | Likely range (80%) contained the truth |
|---|---|---|---|---|---|
| Seasonal (clean) | **0.63** | **0.63** | 0.74 | 0.94 | 85–90% of months |
| Non-seasonal (clean) | 0.88 | 1.23 | 1.09 | **0.85** | 58–100% of months |
| Realistic (hard) | **0.63** | 0.64 | 0.75 | 0.65 | 79–87% of months |

**What this shows**

* On seasonal, climate-driven data the ensemble was about **35% better** than the baseline.
* Without a seasonal cycle, the tree models were **worse** than the baseline and their ranges were too
  narrow (58–69% instead of 80%); Poisson regression and the ensemble were modestly better. The
  report's trust rating and model comparison are there to catch this.
* On the realistic dataset the models were again about a third better than the baseline, but absolute errors
  were large, driven by events climate cannot predict:

    | Forecast year | Ensemble RMSE | Baseline RMSE | What happened |
    |---|---|---|---|
    | 2018 | 10 | 17 | Ordinary year |
    | 2019 | 82 | 82 | Serotype-driven outbreak: missed by every method |
    | 2020 | 32 | 99 | Reporting collapse: the baseline copied the 2019 outbreak forward |
    | 2022 | 31 | 42 | After the disruption |

* The 95% ranges contained the truth in almost every month, so they are probably somewhat **wider than
  necessary**.

---

## Number of optimisation trials (synthetic)

More trials were **not** reliably more accurate: v2 accuracy stopped improving at about 10–30 trials per model
and was sometimes worse at 80; v1 improved from 5 to 20 trials but not from 20 to 50. Runtime grew in
proportion to the number of trials. Details: [Tuning & optimisation trials](tuning.md).

---

## Scenario outlook (synthetic)

* **Single district:** projections moved in the right direction and ranked scenarios correctly, but
  **understated strong warming effects by about half** (+27% vs a true +60% by the 2050s). The truth was inside
  the stated range in every case tested.
* **Six districts pooled** (`extra_districts`): within 3 percentage points of the truth in three of three
  trials, with narrower ranges.
* **Distributed lags** (optional): accurate in one case (+58% vs +60%), unstable or overestimating in others;
  see [Choosing the lag structure](scenarios.md#choosing-the-lag-structure).
* **Backtest on past years:** about as accurate as the historical average; too few years with little climate
  change to confirm skill either way.

---

## Known limitations

* Accuracy on real data is **unknown** until validation is done.
* Synthetic data flatter the models; even the "realistic" dataset has clean climate inputs.
* Climate cannot predict serotype shifts, reporting changes, testing campaigns or control interventions.
* Single-district climate projections can understate warming effects (see [scenario limitations](scenarios.md#known-limitations-read-before-using)).
* The *Aedes aegypti* thermal-curve preset values must be verified against the source before publication.
* v1 results produced with versions before 0.4.0 (0.1.x) were optimistic because of leakage bugs; rerun them.

---

## Help us test

If you run ClimAID on real data, we would like to hear how it performed, especially the trust rating and the
"what actually happened" column. Email [avik.sam@iitb.ac.in](mailto:avik.sam@iitb.ac.in), [avik.sam@nus.edu.sg](mailto:avik.sam@nus.edu.sg) or open an issue on
[GitHub](https://github.com/sam-as/ClimAID/issues).

