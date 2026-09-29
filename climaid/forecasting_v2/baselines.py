"""Prespecified forecasting baselines for ClimAID v2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae, rmse


class SeasonalNaive:
    """Seasonal-naive point baseline with empirical uncertainty."""

    def __init__(self, date_col: str = "time", case_col: str = "cases", period: int = 12) -> None:
        self.date_col = date_col
        self.case_col = case_col
        self.period = int(period)
        self.history_: pd.DataFrame | None = None
        self.residuals_: np.ndarray | None = None
        self.lag_offset_unit_: str = "months"

    def fit(self, disease: pd.DataFrame) -> "SeasonalNaive":
        d = disease.copy()
        d[self.date_col] = pd.to_datetime(d[self.date_col], errors="raise")
        d[self.case_col] = pd.to_numeric(d[self.case_col], errors="raise")
        d = d.sort_values(self.date_col).drop_duplicates(self.date_col, keep="last")
        if len(d) <= self.period:
            raise ValueError("Not enough observations for seasonal naive baseline")
        d["_pred"] = d[self.case_col].shift(self.period)
        valid = d["_pred"].notna()
        self.residuals_ = (d.loc[valid, self.case_col] - d.loc[valid, "_pred"]).to_numpy(float)
        self.history_ = d[[self.date_col, self.case_col]].copy()
        # The exact-lag lookup in predict() must offset by the same time unit
        # that `period` steps actually represent in this data (e.g. a
        # period=52 seasonal cycle on weekly data is 52 *weeks*, not 52
        # months). Infer the cadence from the median spacing between
        # observations rather than assuming a fixed unit.
        median_days = float(d[self.date_col].diff().dropna().dt.days.median())
        self.lag_offset_unit_ = "weeks" if median_days <= 8 else "months"
        return self

    def predict(self, future_dates: pd.DatetimeIndex, n_history_years: int = 10) -> pd.DataFrame:
        if self.history_ is None or self.residuals_ is None:
            raise RuntimeError("SeasonalNaive has not been fitted")
        dates = pd.DatetimeIndex(future_dates)
        hist = self.history_.set_index(self.date_col)[self.case_col]
        preds = []
        for date in dates:
            if self.lag_offset_unit_ == "weeks":
                lag_date = date - pd.DateOffset(weeks=self.period)
            else:
                lag_date = date - pd.DateOffset(months=self.period)
            if lag_date in hist.index:
                preds.append(float(hist.loc[lag_date]))
            else:
                month_matches = hist[hist.index.month == date.month]
                preds.append(float(month_matches.tail(n_history_years).median()) if not month_matches.empty else float(hist.median()))
        point = np.maximum(preds, 0.0)
        residual_q = np.quantile(self.residuals_, [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
        qs = np.maximum(0.0, np.asarray(point)[:, None] + residual_q[None, :])
        qs = np.sort(qs, axis=1)
        out = pd.DataFrame({self.date_col: dates})
        for i, p in enumerate([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]):
            out[f"q{int(round(p * 1000)):03d}"] = qs[:, i]
        return out

    def score(self, observed: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
        m = observed[[self.date_col, self.case_col]].merge(forecast, on=self.date_col, how="inner")
        if m.empty:
            raise ValueError("No common dates")
        return {
            "RMSE": rmse(m[self.case_col], m["q500"]),
            "MAE": mae(m[self.case_col], m["q500"]),
        }
