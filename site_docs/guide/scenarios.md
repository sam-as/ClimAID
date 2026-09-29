# Climate scenario outlook

!!! warning "Under testing"
    The scenario outlook is under active testing. It has been checked only on **synthetic** climate-model
    data with a known answer. It could not yet be run on the real CMIP6 files in development. Projections are
    "what if" estimates, not predictions of actual future cases.

`DiseaseModel.project_v2()` estimates how cases could change under future climate, for each CMIP6 climate
model and emissions pathway (SSP), from next season out to 2050 (or 2100).

---

## How the outlook is built

1. **Bias correction.** Each climate model is shifted so its baseline-period monthly averages match observed
   climate (rainfall scaled, other variables shifted). Its projected *change* is kept.
2. **Next 12 months:** the v2 forecast, driven by each climate model's corrected climate.
3. **Hand-over (months 13–24):** a gradual switch as recent cases stop being informative.
4. **Long term:** a model that links cases to climate alone, refitted many times on resampled years so the
   ranges include statistical uncertainty.
5. **Pooling:** results from all climate models are combined, so ranges also include climate-model spread.
6. **Checks reported alongside:** a sensitivity run using a different way of learning climate effects, a
   backtest on held-out past years, and a comparison with tree-based models.

---

## Main options

| Option | Default | Meaning |
|---|---|---|
| `end_year` | 2050 | Last year of the outlook |
| `ssps` | all available | e.g. `["ssp126", "ssp245", "ssp585"]` |
| `response` | `"seasonal"` | Learn climate effects from the seasonal cycle (`"anomaly"`: only from year-to-year deviations) |
| `lag_selection` | `"ensemble"` | `"ensemble"`: average all near-equally-good climate-lag structures. `"v1"`: run ClimAID v1's lag search (in `v1_mode`) on data before the forecast start. `"distributed"`: smooth lag curves over 0–6 months (ENSO 0–12) |
| `v1_mode` | from `tuning` | v1 mode for `lag_selection="v1"`: `"fast"`, `"balanced"`, `"deep"` |
| `extra_districts` | None | Other districts' data: learn one shared climate response (recommended if available) |
| `temperature_curve` | None | e.g. `"aedes_aegypti_mordecai2017"`: constrain the temperature response to a thermal-suitability curve |
| `population_projection` | None | Table of projected population; adds population-scaled results |
| `comparison_models` | random forest, gradient boosting | Tree models shown as a comparison |
| `tuning` | `"balanced"` | Tuning of the near-term and comparison models |
| `exclude_period` | `"2020"` | COVID-19 period left out of the long-term fit and replaced for the near-term forecast |

---

## Choosing the lag structure

Tested on synthetic cases with a known warming effect (change by the 2050s):

| Case (truth) | Averaged (default) | v1 lags | Distributed lags |
|---|---|---|---|
| Seasonal benchmark, SSP5-8.5 (+60%) | +27% | +48% | **+58%**, but range up to +200% |
| Six-district world, single district (+27%, 3 seeds) | +13 / +16 / +30% | — | +14 / +41 / +54%, wide ranges |
| Six-district world, pooled (+27%, 3 seeds) | **+25 / +24 / +30%** | — | +41 / +33 / +34% (overestimates) |

No single choice was best everywhere. The averaged default was the most reliable when districts are pooled;
distributed lags estimated the total effect of warming well in one case but were unstable in others. The
fitted distributed-lag curves recover the **total** effect better than the **timing**, so do not read the
curve shape as "when the effect happens".

## Known limitations (read before using)

!!! danger "Single-district projections can understate warming effects"
    Temperature, rainfall and humidity often share one seasonal cycle, so a single district's data cannot fully
    separate their effects. On synthetic data where warming truly raised cases by 60% by the 2050s, the default
    single-district projection gave **+27%** (the truth was inside its stated range). Pooling six districts
    with `extra_districts` recovered the truth within 3 percentage points. Use `extra_districts`,
    `lag_selection="v1"` or a justified `temperature_curve` where you can.

* The climate–disease relationship is assumed to stay the same in future.
* Population (unless scaled), immunity, vector control, testing and reporting are held at historical levels.
* Climate models represent El Niño poorly; its effects mainly add noise far ahead.
* Where future climate goes beyond the range seen in training, the response is extrapolated (the report
  says how often this happens).
* **The thermal-curve preset values (17.8 / 29.1 / 34.6 °C) were entered from the literature without
  independent verification; check them against Mordecai et al. (2017) before publication.**
