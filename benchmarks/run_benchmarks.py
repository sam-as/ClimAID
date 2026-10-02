"""Rolling-origin accuracy benchmark for ClimAID v2 on the synthetic datasets.

For each dataset, 12-month forecasts are made from several start dates with the
full forecast_v2() path (compulsory tuning, hindcasts, interval calibration,
drop_2020) and scored on the following 12 observed months. The key comparison is
each model's probabilistic score (WIS, lower is better) relative to the
seasonal-naive "same as recent years" baseline: below 1.0 means better.

Usage:  python benchmarks/run_benchmarks.py [--tuning fast] [--models a,b,...] [--out benchmarks/results]
"""
from __future__ import annotations
import argparse, contextlib, io, sys, time, warnings
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climaid.climaid_model import DiseaseModel           # noqa: E402
import generators as G                                     # noqa: E402

ORIGINS = ("2017-12-31", "2018-12-31", "2019-12-31", "2021-12-31")
MODELS = ("seasonal_naive", "poisson", "random_forest", "gradient_boosting")


def run(dataset, disease, climate, tuning, models=MODELS):
    rows = []
    for origin in ORIGINS:
        dm = object.__new__(DiseaseModel)
        dm.target_col = "Count"; dm.random_state = 42; dm.district = "Synthetic"; dm.disease_name = dataset
        dm.df_disease = disease.rename(columns={"cases": "Count"}); dm.df_climate_hist = climate; dm.df_climate_proj = None
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = dm.forecast_v2(forecast_origin=origin, horizon=12, n_simulations=500, models=models,
                               run_hindcasts=True, hindcast_origins=3, tuning=tuning, save_report=False)
        m = r["metrics"].assign(dataset=dataset, origin=origin[:4])
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tuning", default="fast"); ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated v2 model names")
    a = ap.parse_args()
    models = tuple(m.strip() for m in a.models.split(",") if m.strip())
    out = Path(a.out) if a.out else Path(__file__).parent / "results"; out.mkdir(parents=True, exist_ok=True)
    sets = {"seasonal (clean)": G.seasonal()[:2], "non-seasonal (clean)": G.nonseasonal()[:2],
            "realistic (hard)": G.realistic_hard()[:2]}
    t0 = time.time(); allm = []
    for name, (d, c) in sets.items():
        allm.append(run(name, d, c, a.tuning, models)); print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    m = pd.concat(allm, ignore_index=True)
    m.to_csv(out / "per_origin_metrics.csv", index=False)
    agg = m.groupby(["dataset", "model"])[["RMSE", "MAE", "WIS", "coverage_80", "coverage_95"]].mean().reset_index()
    base = agg[agg.model == "seasonal_naive"].set_index("dataset").WIS
    agg["WIS_vs_seasonal_naive"] = agg.apply(lambda r: r.WIS / base.get(r.dataset), axis=1)
    agg.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(agg.round(2).to_markdown(index=False) if hasattr(agg, "to_markdown") else agg.round(2).to_string(index=False))
    print(agg.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
