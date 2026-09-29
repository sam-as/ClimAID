"""Climate-informed ML forecasting for ClimAID v2.

All registered legacy ClimAID models remain available. The v2 wrapper adds
strict temporal out-of-fold residual learning and empirical predictive bands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, PoissonRegressor, Ridge
from sklearn.neural_network import MLPRegressor

from ..model_registry import MODEL_REGISTRY, is_model_available
from ..model_parameters import DEFAULT_PARAMS
from .schema import canonicalize_disease, canonicalize_climate
from .features import ClimateFeatureBuilder, ClimateFeatureConfig
from .stacking import TemporalResidualStack
from .renewal import DEFAULT_PROBS


# Regression-type models that receive distributed-lag features by default.
DL_MODELS = {"linear", "ridge", "lasso", "elasticnet", "poisson", "tweedie", "spline_poisson",
             "bayesian_ridge", "huber"}

ALIASES = {
    "rf": "random_forest", "extratrees": "extra_trees", "gbr": "gradient_boosting",
    "xgb": "xgboost", "lgbm": "lightgbm", "nn": "mlp", "neural_net": "mlp",
}


def _estimator(model_name: str, random_state: int = 42, params=None):
    """Construct a v2 model with ClimAID defaults (optionally overridden).

    FIX: this used to look defaults up under the v2 name (e.g. "random_forest")
    while the defaults table uses v1 names ("rf"), so v2 silently fell back to
    scikit-learn's generic defaults. Construction now goes through
    tuning.build_estimator, which maps names, scales linear-type models and
    provides the additional v2 models.
    """
    from .tuning import build_estimator, EXTRA_MODELS
    name = ALIASES.get(model_name, model_name)
    if name not in EXTRA_MODELS and not is_model_available(name):
        raise ValueError(f"Model '{model_name}' is unavailable in this installation.")
    return build_estimator(name, params, random_state)


class ClimateMLForecaster:
    """Climate + autoregressive disease-history model with temporal OOF stacking."""

    def __init__(
        self,
        model_name="random_forest",
        date_col="time",
        case_col="cases",
        climate_config=None,
        random_state=42,
        residual_model="random_forest",
        tuning=None,
        distributed_lags=None,
    ):
        self.tuning = tuning            # compulsory: None -> default preset
        self.distributed_lags = distributed_lags   # None = automatic by model type
        self.model_name = model_name
        self.date_col = date_col
        self.case_col = case_col
        self.climate_config = climate_config or ClimateFeatureConfig(date_col=date_col)
        self.random_state = random_state
        self.residual_model_name = residual_model

    def fit(self, disease, climate, cutoff=None):
        d = canonicalize_disease(disease, date_col=self.date_col, case_col=self.case_col)
        c = canonicalize_climate(climate, date_col=self.date_col, require_all=True)
        self.cutoff_ = pd.Timestamp(cutoff) if cutoff is not None else d.time.max()
        d = d[d.time <= self.cutoff_].copy()
        if d.empty:
            raise ValueError("No disease observations at or before forecast cutoff")

        self.builder_ = ClimateFeatureBuilder(self.climate_config).fit(c, self.cutoff_)
        cf = self.builder_.transform(c)
        m = d.merge(cf, on="time", how="inner", validate="one_to_one").sort_values("time")
        if len(m) < 30:
            raise ValueError("At least 30 aligned historical observations are recommended for ML forecasting")

        for lag in (1, 2, 3):
            m[f"case_lag{lag}"] = m[self.case_col].shift(lag)
        m = m.dropna(subset=["case_lag1", "case_lag2", "case_lag3"]).reset_index(drop=True)

        climate_features = [c for c in cf.columns if c != "time"]
        # Distributed-lag features are used by regression-type models, where they helped in the
        # synthetic benchmark (Poisson better on all three datasets); tree-based and other flexible
        # models combine raw lags themselves and were mixed, so they skip them unless asked.
        use_dl = self.distributed_lags if self.distributed_lags is not None else (
            ALIASES.get(self.model_name, self.model_name) in DL_MODELS)
        if not use_dl:
            climate_features = [c for c in climate_features if "_dl" not in c]
        self.uses_distributed_lags_ = bool(use_dl and any("_dl" in c for c in climate_features))
        self.feature_names_ = climate_features + ["case_lag1", "case_lag2", "case_lag3"]
        X = m[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        y = m[self.case_col].to_numpy(float)
        X = X.replace([np.inf, -np.inf], np.nan)
        self.x_median_ = X.median()
        X = X.fillna(self.x_median_)

        # Compulsory, leakage-safe tuning on the training rows only.
        from .tuning import tune
        canonical = ALIASES.get(self.model_name, self.model_name)
        self.tuned_params_, self.tuning_info_ = tune(canonical, X, y, self.tuning, self.random_state)
        base = _estimator(self.model_name, self.random_state, self.tuned_params_)
        residual = _estimator(self.residual_model_name, self.random_state + 17)
        self.stack_ = TemporalResidualStack(
            base,
            residual,
            outer_splits=4,
            inner_splits=3,
            test_size=3,
            min_train_size=max(18, min(36, len(X) // 2)),
        ).fit(X, y)

        # Conformal-style empirical residual distribution from temporal OOF stack.
        self.residuals_ = np.asarray(self.stack_.oof_residuals_, dtype=float)
        self.history_ = d.copy()
        self.climate_full_ = c.copy()
        self.cutoff_ = self.cutoff_
        return self

    def predict(self, future_climate, horizon, probs=None):
        if not hasattr(self, "stack_"):
            raise RuntimeError("ClimateMLForecaster has not been fitted")
        probs = DEFAULT_PROBS if probs is None else np.asarray(probs, float)
        c = canonicalize_climate(future_climate, date_col=self.date_col, require_all=True)
        allc = pd.concat([self.climate_full_, c], ignore_index=True)
        allc = allc.drop_duplicates("time", keep="last").sort_values("time")
        cf = self.builder_.transform(allc)
        dates = pd.DatetimeIndex(cf.loc[cf.time > self.cutoff_, "time"].drop_duplicates().sort_values().head(int(horizon)))
        if len(dates) < int(horizon):
            raise ValueError("Future climate does not cover requested horizon")

        history = self.history_[self.case_col].to_numpy(float).tolist()
        rows = []
        residual_quantiles = np.quantile(self.residuals_, probs) if len(self.residuals_) else np.zeros(len(probs))
        # Center residual quantiles to maintain the model point forecast at q=.5.
        median_resid = float(np.quantile(self.residuals_, 0.5)) if len(self.residuals_) else 0.0
        residual_quantiles = residual_quantiles - median_resid

        for date in dates:
            row = cf.loc[cf.time == date].iloc[0].to_dict()
            row.update({"case_lag1": history[-1], "case_lag2": history[-2], "case_lag3": history[-3]})
            Xrow = pd.DataFrame([row])[self.feature_names_].apply(pd.to_numeric, errors="coerce")
            Xrow = Xrow.replace([np.inf, -np.inf], np.nan).fillna(self.x_median_)
            point = float(self.stack_.predict(Xrow)[0])
            q = np.maximum(0.0, point + residual_quantiles)
            rows.append((date, point, q))
            # Recursive ML state uses the central forecast, never future observed cases.
            history.append(point)

        out = pd.DataFrame({"time": [x[0] for x in rows]})
        arr = np.vstack([x[2] for x in rows])
        for i, p in enumerate(probs):
            out[f"q{int(round(p * 1000)):03d}"] = arr[:, i]
        qcols = [f"q{int(round(p * 1000)):03d}" for p in probs]
        out[qcols] = np.sort(out[qcols].to_numpy(float), axis=1)
        return out


class ClimateMLModelFactory:
    """Expose all installed ClimAID registry models to the v2 forecaster."""

    @staticmethod
    def available_models():
        from .tuning import EXTRA_MODELS
        names = [
            "linear", "ridge", "lasso", "elasticnet", "poisson", "tweedie", "spline_poisson",
            "bayesian_ridge", "huber", "random_forest", "extra_trees", "gradient_boosting",
            "hist_gradient_boosting", "xgboost", "lightgbm", "catboost", "mlp", "svr", "knn",
        ]
        return [n for n in names if n in EXTRA_MODELS or is_model_available(n)]


def available_ml_estimators():
    """Backward-compatible helper returning installed canonical ML names/classes."""
    names = ClimateMLModelFactory.available_models()
    return {name: _estimator(name) for name in names}
