"""Simple, conservative probabilistic ensembles for ClimAID v2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import weighted_interval_score


class QuantileMedianEnsemble:
    """Per-quantile median ensemble.

    The median is intentionally preferred as a robust default when the number
    of historical validation seasons is small. It cannot be optimized against
    the test period and requires no fragile ensemble-weight fitting.
    """

    def __init__(self, date_col: str = "time") -> None:
        self.date_col = date_col
        self.probability_columns_: list[str] | None = None
        self.model_names_: list[str] | None = None

    def fit(self, forecasts: dict[str, pd.DataFrame]) -> "QuantileMedianEnsemble":
        if not forecasts:
            raise ValueError("forecasts cannot be empty")
        names = list(forecasts)
        first = forecasts[names[0]].copy()
        qcols = [c for c in first.columns if c.startswith("q")]
        if not qcols:
            raise ValueError("Forecasts must contain qxxx quantile columns")
        for name, frame in forecasts.items():
            if self.date_col not in frame.columns:
                raise ValueError(f"{name} lacks date column '{self.date_col}'")
            if set(qcols) - set(frame.columns):
                raise ValueError(f"{name} does not contain all quantile columns")
        self.probability_columns_ = qcols
        self.model_names_ = names
        return self

    def predict(self, forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_fitted()
        names = self.model_names_
        qcols = self.probability_columns_
        reference = forecasts[names[0]]
        out = reference[[self.date_col]].copy()
        for q in qcols:
            values = np.column_stack([forecasts[name][q].to_numpy(float) for name in names])
            out[q] = np.median(values, axis=1)
        # Quantile crossing can occur when models disagree; enforce monotonicity.
        out[qcols] = np.sort(out[qcols].to_numpy(float), axis=1)
        return out

    def evaluate(
        self,
        forecast: pd.DataFrame,
        observed: pd.DataFrame,
        case_col: str = "cases",
    ) -> dict[str, float]:
        self._check_fitted()
        merged = observed[[self.date_col, case_col]].merge(
            forecast, on=self.date_col, how="inner", validate="one_to_one"
        )
        if merged.empty:
            raise ValueError("No overlapping dates for evaluation")
        qcols = self.probability_columns_
        probs = np.array([int(c[1:]) / 1000.0 for c in qcols])
        quantiles = merged[qcols].to_numpy(float)
        y = merged[case_col].to_numpy(float)
        median_idx = int(np.argmin(np.abs(probs - 0.5)))
        return {
            "WIS": weighted_interval_score(y, quantiles, probs),
            "MAE_median": float(np.mean(np.abs(y - quantiles[:, median_idx]))),
            "coverage_50": _coverage_from_columns(merged, qcols, 0.25, 0.75),
            "coverage_80": _coverage_from_columns(merged, qcols, 0.10, 0.90),
            "coverage_95": _coverage_from_columns(merged, qcols, 0.025, 0.975),
        }

    def _check_fitted(self) -> None:
        if self.probability_columns_ is None or self.model_names_ is None:
            raise RuntimeError("Ensemble has not been fitted")


def _coverage_from_columns(
    frame: pd.DataFrame, qcols: list[str], lower: float, upper: float
) -> float:
    probs = np.array([int(c[1:]) / 1000.0 for c in qcols])
    lo_idx = int(np.argmin(np.abs(probs - lower)))
    hi_idx = int(np.argmin(np.abs(probs - upper)))
    y = frame.iloc[:, 1].to_numpy(float)
    low = frame[qcols[lo_idx]].to_numpy(float)
    high = frame[qcols[hi_idx]].to_numpy(float)
    return float(np.mean((y >= low) & (y <= high)))
