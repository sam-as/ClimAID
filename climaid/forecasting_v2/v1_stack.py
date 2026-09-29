"""ClimAID v1's lag-optimised, tuned, stacked model as a ClimAID v2 model.

Why
---
v1 and v2 were evaluated differently (v1: one test year and point accuracy;
v2: held-out months, rolling hindcasts and calibrated likely ranges). Adding
v1's pipeline as a v2 model (`"v1_stack"`) puts both through exactly the
same tests, so they can be compared fairly and v1 can join the ensemble.

What it does at each forecast origin
------------------------------------
1. Builds a v1 DiseaseModel in memory from the disease and climate data *up
   to the origin only* (v2 never passes later data to a model).
2. Runs v1's lag search and stacked training (base -> residual -> correction).
   v1 keeps its own last full training year as its internal test year and
   leaves the excluded period (default 2020) out, exactly as the v1 pipeline does.
3. Predicts the forecast months from climate alone (v1 has no case-history
   inputs), building features with v1's projection feature builder, whose
   definitions match training.
4. v1 gives point forecasts only. Starting likely ranges are point +/- z x
   v1's test-year RMSE; v2's hindcast interval calibration then rescales them
   like every other model's.

Search effort uses ClimAID v1's own modes exactly (Fast 50 / Balanced 200 / Deep 500
trials, v1's models and full lag ranges), chosen by v2's tuning preset or `v1_mode`. v1's
lag search is slow and is repeated at every hindcast origin, so `v1_stack` is not selected
by default.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from .schema import canonicalize_climate, canonicalize_disease
from .renewal import DEFAULT_PROBS

V1_NAMES = {"temperature": "mean_temperature", "rainfall": "mean_Rain", "humidity": "mean_SH", "enso": "Nino_anomaly"}
Z = {p: float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(p)) for p in DEFAULT_PROBS}


def _effort(tuning, v1_mode=None):
    """optimize_lags() settings for v1_stack.

    Uses ClimAID v1's own modes exactly (same models, trials and lag ranges as the v1 page).
    Without an explicit `v1_mode`, v2's tuning preset picks the matching v1 mode:
    fast -> Fast (50 trials), balanced -> Balanced (200), deep -> Deep (500); a custom number
    of v2 trials maps to the nearest mode. "quick" is a light mode for tests and quick checks.
    """
    from ..model_parameters import v1_mode_config
    from .tuning import resolve_tuning
    if v1_mode is None:
        n, _, _ = resolve_tuning(tuning)
        v1_mode = "fast" if n <= 10 else ("balanced" if n <= 40 else "deep")
    cfg = v1_mode_config(v1_mode)
    label = {"fast": "v1 Fast (50 trials)", "balanced": "v1 Balanced (200 trials)",
             "deep": "v1 Deep (500 trials)", "quick": "v1 quick check (3 trials)"}[v1_mode]
    return cfg, label


class V1StackForecaster:
    def __init__(self, date_col="time", case_col="cases", random_state=42, tuning=None, n_jobs=-1,
                 exclude_period="2020", v1_mode=None):
        self.exclude_period = exclude_period
        self.v1_mode = v1_mode
        self.date_col, self.case_col = date_col, case_col
        self.random_state, self.tuning, self.n_jobs = random_state, tuning, n_jobs

    def fit(self, disease, climate, cutoff=None):
        from ..climaid_model import DiseaseModel
        from ..climaid_projections import DiseaseProjection
        d = canonicalize_disease(disease, date_col=self.date_col, case_col=self.case_col)
        d["time"] = d["time"].dt.to_period("M").dt.to_timestamp()
        self.cutoff_ = pd.Timestamp(cutoff) if cutoff is not None else d["time"].max()
        d = d[d["time"] <= self.cutoff_]
        c = canonicalize_climate(climate, date_col=self.date_col, require_all=True)
        c["time"] = c["time"].dt.to_period("M").dt.to_timestamp()
        self.climate_hist_ = c[c["time"] <= self.cutoff_].copy()
        clim_v1 = self._to_v1(self.climate_hist_)

        dis_v1 = d.rename(columns={self.case_col: "Count"})[["time", "Count"]].copy()
        dis_v1["Year"], dis_v1["Month"] = dis_v1["time"].dt.year, dis_v1["time"].dt.month
        from ..exclusion import resolve_exclusion, in_period
        period = resolve_exclusion(self.exclude_period, drop_2020=False) if not isinstance(self.exclude_period, tuple) \
            else self.exclude_period
        kept = dis_v1[~in_period(dis_v1["time"], period)]
        years = sorted(y for y in kept["Year"].unique() if (kept["Year"] == y).sum() == 12)
        if len(years) < 4:
            raise ValueError("v1_stack needs at least 4 complete years (excluding 2020) before the forecast origin")

        dm = object.__new__(DiseaseModel)
        dm.random_state = self.random_state; dm.district = "v2"; dm.target_col = "Count"; dm.disease_name = "v1_stack"
        dm.df_disease = dis_v1; dm.df_climate_hist = clim_v1; dm.df_climate_proj = None
        dm.train_df = dm.test_df = dm.lag_search_results = dm.best_config = None; dm.final_models = {}
        dm.runtime = {k: None for k in ("lag_optimization_seconds", "training_seconds", "prediction_seconds",
                                        "report_generation_seconds", "total_pipeline_seconds")}
        dm._pipeline_start_time = None
        dm.drop_2020 = period is not None
        dm.exclude_period = period if period is not None else "none"
        dm.test_year = years[-1]            # v1's own internal test year, inside v2's training period
        dm.train_year = years[-1] - 1
        effort, label = _effort(self.tuning, self.v1_mode)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm.df_merged = dm._merge_data()
            dm.optimize_lags(n_jobs=self.n_jobs, **effort)
            final = dm.train_final_model()
        self.dm_ = dm
        dm.df_climate_proj = clim_v1                          # DiseaseProjection requires a projection frame
        self.proj_ = DiseaseProjection(dm)
        self.rmse_ = float(final.get("test_rmse", dm.rmse))
        self.tuning_info_ = {
            "preset": label, "status": "v1 lag search + stacked training",
            "selected_lags": {k: list(v) for k, v in dm.feature_metadata["lags"].items()},
            "base_model": str(dm.best_config["base_model"]), "residual_model": str(dm.best_config["residual_model"]),
            "correction_model": str(dm.best_config["correction_model"]),
            "v1_validation_rmse": float(dm.best_config["val_rmse"]), "v1_test_year": int(dm.test_year),
            "v1_test_rmse": self.rmse_,
        }
        return self

    @staticmethod
    def _to_v1(c):
        out = c.rename(columns=V1_NAMES)[["time", *[v for v in V1_NAMES.values() if v in c.rename(columns=V1_NAMES).columns]]].copy()
        out["Dist_States"] = "v2"
        return out

    def predict(self, future_climate, horizon, probs=None):
        c = canonicalize_climate(future_climate, date_col=self.date_col, require_all=True)
        c["time"] = c["time"].dt.to_period("M").dt.to_timestamp()
        allc = pd.concat([self.climate_hist_, c], ignore_index=True).drop_duplicates("time", keep="last").sort_values("time")
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = self.proj_.prepare_features(self._to_v1(allc).drop(columns=["Dist_States"]))
            dates = sorted(t for t in feats["time"].unique() if t > self.cutoff_)[: int(horizon)]
            feats = feats[feats["time"].isin(dates)].sort_values("time")
            if len(feats) < int(horizon):
                raise ValueError("v1_stack: future climate does not cover the forecast horizon")
            point = np.maximum(np.asarray(self.dm_.predict(feats), dtype=float), 0.0)
        out = pd.DataFrame({"time": pd.to_datetime(feats["time"]).to_numpy()})
        sd = max(self.rmse_, 1e-6)
        for p in DEFAULT_PROBS:
            out[f"q{int(round(p * 1000)):03d}"] = np.maximum(point + Z[p] * sd, 0.0)
        return out
