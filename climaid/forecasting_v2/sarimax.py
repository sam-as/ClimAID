"""SARIMAX: seasonal ARIMA with climate as external regressors, for ClimAID v2.

A standard benchmark in climate-and-disease forecasting. Cases are modelled on the
log(1 + cases) scale, where SARIMAX's Gaussian errors are a reasonable approximation for
counts, and converted back; quantiles transform exactly under this monotone map.

Tuning, like every v2 model, is compulsory and leakage-safe: candidate orders
(p, d, q)(P, D, Q)_12 and the climate lag are scored by expanding-window, time-ordered
cross-validation inside the training period (mean absolute error of one-month-ahead
predictions on the case scale). The default configuration is scored on the same folds and
kept unless a candidate beats it. Climate standardisation uses training data only.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .renewal import DEFAULT_PROBS
from .schema import canonicalize_climate, canonicalize_disease
from .tuning import resolve_tuning

CLIMATE_VARS = ("temperature", "rainfall", "humidity", "enso")
DEFAULT_CONFIG = {"order": (1, 0, 0), "seasonal_order": (1, 0, 0), "climate_lag": 1}
GRID = {"p": (0, 1, 2), "d": (0, 1), "q": (0, 1, 2), "P": (0, 1), "D": (0, 1), "Q": (0, 1), "climate_lag": (0, 1, 2, 3)}
MIN_TRAIN_MONTHS = 24
FOLD_MONTHS = 6


def _fit(endog, exog, order, seasonal_order, period=12):
    """Fit one SARIMAX; returns the results object (warnings silenced, failures raised)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=(*seasonal_order, period),
                        trend="c" if order[1] + seasonal_order[1] == 0 else "n",
                        enforce_stationarity=True, enforce_invertibility=True)
        res = model.fit(disp=False, maxiter=200)
    if not np.all(np.isfinite(res.params)):
        raise RuntimeError("non-finite SARIMAX parameters")
    return res


class SarimaxForecaster:
    """v2 model interface: fit(disease, climate, cutoff) / predict(future_climate, horizon)."""

    def __init__(self, date_col="time", case_col="cases", random_state=42, tuning=None):
        self.date_col, self.case_col = date_col, case_col
        self.random_state, self.tuning = random_state, tuning

    # ------------------------------------------------------------------ features
    def _exog(self, climate: pd.DataFrame, dates: pd.DatetimeIndex, lag: int) -> np.ndarray:
        """Standardised climate at `lag` months before each date (training statistics only)."""
        c = climate.set_index("time")[list(self.vars_)].sort_index()
        c = c[~c.index.duplicated(keep="last")]
        c = c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="MS"))
        x = ((c.shift(lag).reindex(dates) - self.mean_) / self.sd_)
        past = x.index <= self.cutoff_
        if x[~past].isna().any().any():
            raise ValueError("SARIMAX: future climate does not cover the forecast months (including its lag)")
        # before the climate record starts (first `lag` months) use the training mean (0 after standardising)
        return x.fillna(0.0).to_numpy(float)

    def _score(self, y, dates, climate, cfg, n_folds):
        """Expanding-window one-month-ahead MAE (case scale) over the last n_folds blocks."""
        n = len(y)
        errors = []
        try:
            X = self._exog(climate, dates, cfg["climate_lag"])
        except Exception:
            return np.inf
        for k in range(n_folds, 0, -1):
            end_train = n - k * FOLD_MONTHS
            if end_train < MIN_TRAIN_MONTHS:
                continue
            val = slice(end_train, end_train + FOLD_MONTHS)
            try:
                res = _fit(y[:end_train], X[:end_train], cfg["order"], cfg["seasonal_order"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ext = res.append(y[val], exog=X[val], refit=False)
                    pred = ext.get_prediction(start=end_train, end=end_train + FOLD_MONTHS - 1).predicted_mean
                errors.append(np.abs(np.expm1(np.asarray(pred)) - np.expm1(y[val])))
            except Exception:
                return np.inf
        return float(np.mean(np.concatenate(errors))) if errors else np.inf

    # ------------------------------------------------------------------ fit / predict
    def fit(self, disease, climate, cutoff=None):
        d = canonicalize_disease(disease, date_col=self.date_col, case_col=self.case_col)
        d = d.sort_values("time").drop_duplicates("time", keep="last")
        self.cutoff_ = pd.Timestamp(cutoff) if cutoff is not None else d["time"].max()
        d = d[d["time"] <= self.cutoff_]
        step = d["time"].diff().dt.days.median()
        if not 27 <= step <= 32:
            raise ValueError("SARIMAX in ClimAID needs monthly data")
        dates = pd.DatetimeIndex(d["time"])
        if len(dates) < MIN_TRAIN_MONTHS + FOLD_MONTHS:
            raise ValueError(f"SARIMAX needs at least {MIN_TRAIN_MONTHS + FOLD_MONTHS} months before the forecast origin")
        c = canonicalize_climate(climate, date_col=self.date_col, require_all=True)
        c["time"] = pd.to_datetime(c["time"]).dt.to_period("M").dt.to_timestamp()
        dates = dates.to_period("M").to_timestamp()
        self.vars_ = [v for v in CLIMATE_VARS if v in c.columns]
        hist = c[c["time"] <= self.cutoff_]
        self.mean_ = hist[self.vars_].mean()
        self.sd_ = hist[self.vars_].std(ddof=0).replace(0, 1.0)
        self.climate_ = c
        y = np.log1p(np.maximum(pd.to_numeric(d[self.case_col], errors="coerce").to_numpy(float), 0.0))
        if not np.all(np.isfinite(y)):
            raise ValueError("SARIMAX: case counts contain missing values")

        n_trials, n_folds, label = resolve_tuning(self.tuning)
        grid = [dict(order=(p, dd, q), seasonal_order=(P, D, Q), climate_lag=L)
                for p, dd, q, P, D, Q, L in itertools.product(*GRID.values())
                if dd + D <= 1 and p + q + P + Q >= 1]
        rng = np.random.default_rng(self.random_state)
        candidates = [grid[i] for i in rng.choice(len(grid), size=min(n_trials, len(grid)), replace=False)]

        default_score = self._score(y, dates, c, DEFAULT_CONFIG, n_folds)
        best_cfg, best_score = DEFAULT_CONFIG, default_score
        for cfg in candidates:
            s = self._score(y, dates, c, cfg, n_folds)
            if s < best_score:
                best_cfg, best_score = cfg, s

        try:
            self.result_ = _fit(y, self._exog(c, dates, best_cfg["climate_lag"]), best_cfg["order"],
                                best_cfg["seasonal_order"])
        except Exception:
            if best_cfg is DEFAULT_CONFIG:
                raise
            best_cfg = DEFAULT_CONFIG          # fall back to the default if the chosen one fails on all data
            self.result_ = _fit(y, self._exog(c, dates, 1), DEFAULT_CONFIG["order"], DEFAULT_CONFIG["seasonal_order"])
        self.config_ = best_cfg
        self.dates_ = dates
        self.tuning_info_ = {
            "preset": label, "n_trials": len(candidates), "n_folds": n_folds,
            "status": "tuned" if best_cfg is not DEFAULT_CONFIG else "defaults kept",
            "default_cv_mae": None if not np.isfinite(default_score) else default_score,
            "best_cv_mae": None if not np.isfinite(best_score) else best_score,
            "order": list(best_cfg["order"]), "seasonal_order": [*best_cfg["seasonal_order"], 12],
            "climate_lag_months": int(best_cfg["climate_lag"]), "climate_variables": list(self.vars_),
            "scale": "log(1 + cases)",
        }
        return self

    def predict(self, future_climate, horizon, probs=None):
        probs = DEFAULT_PROBS if probs is None else np.asarray(probs, float)
        c = canonicalize_climate(future_climate, date_col=self.date_col, require_all=True)
        c["time"] = pd.to_datetime(c["time"]).dt.to_period("M").dt.to_timestamp()
        allc = pd.concat([self.climate_, c], ignore_index=True).drop_duplicates("time", keep="last")
        future = pd.DatetimeIndex(sorted(t for t in allc["time"].unique() if t > self.cutoff_)[: int(horizon)])
        if len(future) < int(horizon):
            raise ValueError("SARIMAX: future climate does not cover the forecast horizon")
        X = self._exog(allc, future, self.config_["climate_lag"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.result_.get_forecast(steps=int(horizon), exog=X)
        mean = np.asarray(fc.predicted_mean, float)
        se = np.sqrt(np.maximum(np.asarray(fc.var_pred_mean, float), 0.0))
        out = pd.DataFrame({"time": future})
        for p in probs:
            out[f"q{int(round(p * 1000)):03d}"] = np.maximum(np.expm1(mean + norm.ppf(p) * se), 0.0)
        return out
