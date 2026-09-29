# ClimAID benchmarks (synthetic data)

ClimAID 0.4.0 is under active testing. These benchmarks check the methods against synthetic data with a
**known truth**. Good results here are necessary but not sufficient: validation on real surveillance data
and real CMIP6 projections is still to come.

- `generators.py` — every synthetic dataset used in testing (fixed seeds, reproducible):
  `seasonal`, `nonseasonal`, `enso_interaction`, `realistic_hard`, `cmip6`, `multi_district`, `thermal_optimum`.
- `run_benchmarks.py` — rolling-origin accuracy benchmark (12-month forecasts from four start dates through
  the full `forecast_v2()` path) on the seasonal, non-seasonal and realistic datasets.
- `results/` — output of the last run (`summary.csv`, `summary.md`, `per_origin_metrics.csv`).

The clean datasets are built the way the models assume the world works, so they flatter the models.
`realistic_hard()` adds a threshold rainfall effect, a saturating temperature effect, a reporting change,
serotype-driven outbreak years, a 2020 reporting collapse, missing months and data-entry errors.

Run: `python benchmarks/run_benchmarks.py --tuning fast` (about 20 minutes on one CPU core).
