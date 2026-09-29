"""Leakage-safe temporal validation utilities for ClimAID v2.

All transformations and learned models must be fitted inside each training fold.
The splitter never shuffles observations and only creates chronological folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastOrigin:
    """A single historical forecast origin.

    Parameters
    ----------
    cutoff : pandas.Timestamp
        Last date permitted in the training information set.
    horizon : int
        Number of periods to forecast after ``cutoff``.
    frequency : str
        Pandas frequency used by the data (e.g. ``"MS"`` for monthly or
        ``"W-MON"`` for weekly data).
    """

    cutoff: pd.Timestamp
    horizon: int
    frequency: str = "MS"

    def future_dates(self) -> pd.DatetimeIndex:
        """Return the dates that are genuinely in the forecast horizon."""
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        return pd.date_range(
            self.cutoff,
            periods=self.horizon + 1,
            freq=self.frequency,
        )[1:]


class RollingOriginSplitter:
    """Expanding-window chronological cross-validation.

    Unlike ``KFold``/random CV, this splitter preserves the forecasting
    direction. Each validation block occurs strictly after its training block.
    """

    def __init__(
        self,
        n_splits: int = 4,
        test_size: int = 3,
        step: int | None = None,
        min_train_size: int = 12,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        if min_train_size < 2:
            raise ValueError("min_train_size must be >= 2")
        self.n_splits = int(n_splits)
        self.test_size = int(test_size)
        self.step = int(step if step is not None else test_size)
        self.min_train_size = int(min_train_size)

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if n_samples < self.min_train_size + self.test_size:
            raise ValueError(
                "Not enough observations for the requested temporal split: "
                f"n_samples={n_samples}, min_train_size={self.min_train_size}, "
                f"test_size={self.test_size}."
            )

        first_train_end = self.min_train_size
        for i in range(self.n_splits):
            train_end = first_train_end + i * self.step
            val_end = train_end + self.test_size
            if val_end > n_samples:
                break
            train_idx = np.arange(0, train_end, dtype=int)
            val_idx = np.arange(train_end, val_end, dtype=int)
            yield train_idx, val_idx

    def get_n_splits(self, n_samples: int | None = None) -> int:
        if n_samples is None:
            return self.n_splits
        return sum(1 for _ in self.split(n_samples))


class LeakageSafePreprocessor:
    """Train-only numeric preprocessing for time-series models.

    The transformer supports median imputation and standard scaling. All
    statistics are learned from the supplied training slice and can then be
    applied unchanged to validation/test/future data.
    """

    def __init__(self, scale: bool = True) -> None:
        self.scale = bool(scale)
        self.columns_: list[str] | None = None
        self.medians_: pd.Series | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "LeakageSafePreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        numeric = X.select_dtypes(include=[np.number]).copy()
        if numeric.shape[1] == 0:
            raise ValueError("X contains no numeric columns")
        self.columns_ = list(numeric.columns)
        self.medians_ = numeric.median(numeric_only=True)
        filled = numeric.fillna(self.medians_)
        self.mean_ = filled.mean()
        self.std_ = filled.std(ddof=0).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        missing = [c for c in self.columns_ or [] if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns during transform: {missing}")
        out = X.loc[:, self.columns_].copy()
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out.fillna(self.medians_)
        if self.scale:
            out = (out - self.mean_) / self.std_
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if self.columns_ is None or self.medians_ is None:
            raise RuntimeError("Preprocessor has not been fitted")
