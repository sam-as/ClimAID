"""Effect of the number of optimisation trials on accuracy and runtime (synthetic benchmark).

v2: every ML model is tuned with N Optuna trials (N = 5, 10, 30, 80; 3 time-ordered folds) and
    forecasts 12 months from three start dates (two for N = 80, for runtime) on three datasets.
    Scored on the following 12 observed months. The seasonal-naive baseline does not depend on N.
v1: the v1 stacked pipeline (xgb / rf / isotonic, as in v1 Fast and Balanced) with N trials
    (5, 20, 50) on a reduced lag grid (temperature/rainfall 0-2, humidity 0-1, ENSO 0-3, top 10
    configurations) so runs finish on one CPU core; the grid is fixed, so only the trial count varies.
    Runtimes for 200 and 500 trials are extrapolated, not measured.

Results are appended to benchmarks/results/trials_v2.csv and trials_v1.csv as each run finishes.
Usage: python benchmarks/trials_study.py [v2|v1|both]
"""
import contextlib, io, sys, time, warnings
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import generators as G                                           # noqa: E402
OUT = Path(__file__).parent / "results"
SETS = {"seasonal": G.seasonal, "non-seasonal": G.nonseasonal, "realistic": G.realistic_hard}
ORIGINS = ("2017-12-31", "2019-12-31", "2021-12-31")


def append(path, rows):
    df = pd.DataFrame(rows); df.to_csv(path, mode="a", header=not path.exists(), index=False)


def v2():
    from climaid.forecasting_v2 import ClimaidV2Forecaster
    path = OUT / "trials_v2.csv"
    done = set()
    if path.exists():                      # resume: skip runs already recorded
        prev = pd.read_csv(path)
        done = set(zip(prev.trials, prev.dataset, prev.origin.astype(str), prev.model))
    for trials in (5, 10, 30, 80):
        for name, gen in SETS.items():
            d, c = gen()[:2]
            for origin in (ORIGINS if trials < 80 else ("2017-12-31", "2021-12-31")):
                models = ("poisson", "random_forest", "gradient_boosting") + (("seasonal_naive",) if trials == 5 else ())
                for model in models:
                    if (trials, name, origin[:4], model) in done:
                        continue
                    t0 = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        e = ClimaidV2Forecaster(models=(model,), tuning=trials, include_ensemble=False).fit(d, c, origin)
                        m = e.evaluate(d[d.time > origin].head(12), e.predict(c[c.time > origin].head(12), 12, n_simulations=200))
                    r = m.iloc[0]
                    append(path, [{"trials": trials, "dataset": name, "origin": origin[:4], "model": model,
                                   "WIS": r.WIS, "RMSE": r.RMSE, "seconds": time.time() - t0}])
        print("v2 trials", trials, "done", flush=True)


def v1():
    from climaid.forecasting_v2.v1_stack import V1StackForecaster
    import climaid.forecasting_v2.v1_stack as V
    path = OUT / "trials_v1.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.dataset, prev.trials))
    grid = dict(temp_range=range(0, 3), rain_range=range(0, 3), sh_range=range(0, 2), elnino_range=range(0, 4), top_k=10)
    plan = [(n, t) for t in (5, 20, 50) for n in ("seasonal", "realistic")]
    for name, trials in plan:
        if (name, trials) in done:
            continue
        d, c = SETS[name]()[:2]
        origin = "2019-12-31"
        cfg = {"base_models": ("xgb",), "residual_models": ("rf",), "correction_models": ("isotonic",), "n_trials": trials, **grid}
        V._effort = lambda tuning, v1_mode=None, cfg=cfg: (cfg, f"v1 {cfg['n_trials']} trials (reduced grid)")
        t0 = time.time()
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            f = V1StackForecaster(tuning="fast").fit(d, c, origin)
            p = f.predict(c[c.time > origin].head(12), 12)
        obs = d[d.time > origin].head(12).merge(p[["time", "q500"]], on="time")
        rmse = float(((obs.cases - obs.q500) ** 2).mean() ** 0.5)
        info = f.tuning_info_
        append(path, [{"trials": trials, "dataset": name, "origin": origin[:4], "RMSE": rmse,
                       "v1_validation_rmse": info["v1_validation_rmse"], "selected_lags": str(info["selected_lags"]),
                       "seconds": time.time() - t0}])
        print("v1", name, trials, "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("v2", "both"): v2()
    if what in ("v1", "both"): v1()
