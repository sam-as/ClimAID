"""Public ClimAID v2 forecasting orchestrator.

This module is additive to the original ClimAID pipeline. It keeps the legacy
model registry and CMIP6 projection workflow intact while providing a new,
climate-mandatory probabilistic forecasting layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd

from .schema import canonicalize_disease, canonicalize_climate
from .features import ClimateFeatureConfig
from .renewal import ClimateRenewalModel, DEFAULT_PROBS

# Default v2 model set used by forecast_v2(), the browser dashboard and the
# terminal wizard. Previously only seasonal_naive + renewal + random_forest,
# which made the ensemble a median of just three models, one of them the only
# ML learner. This set mixes a GLM (poisson), two bagged tree ensembles and a
# boosted ensemble, so the per-quantile median ensemble is less dominated by
# any single model. All are scikit-learn, so no optional installs needed.
DEFAULT_V2_MODELS = ("seasonal_naive", "renewal", "poisson",
                     "random_forest", "extra_trees", "gradient_boosting")
from .baselines import SeasonalNaive
from .ml import ClimateMLForecaster, ClimateMLModelFactory
from .ensemble import QuantileMedianEnsemble
from .metrics import rmse, mae, weighted_interval_score, coverage


@dataclass
class ForecastBundle:
    forecasts: dict[str, pd.DataFrame]
    metadata: dict = field(default_factory=dict)


class ClimaidV2Forecaster:
    """Climate-mandatory probabilistic disease forecaster.

    Parameters
    ----------
    models : iterable[str]
        Any combination of ``seasonal_naive``, ``renewal``, ``sarimax``, ``v1_stack`` and all model names
        available in the original ClimAID registry.
    population_at_risk : float, optional
        Used by the renewal component only when the disease data do not contain
        a population column. If omitted, renewal falls back to relative-incidence
        mode and records that susceptible depletion was unavailable.
    """

    def __init__(
        self,
        date_col="time",
        case_col="cases",
        population_col="population",
        climate_config=None,
        models=("seasonal_naive", "renewal", "random_forest"),
        renewal_kwargs=None,
        population_at_risk=None,
        random_state=42,
        include_ensemble=True,
        tuning=None,
        v1_mode=None,
    ):
        self.tuning = tuning
        self.v1_mode = v1_mode
        self.date_col = date_col
        self.case_col = case_col
        self.population_col = population_col
        self.climate_config = climate_config or ClimateFeatureConfig(date_col=date_col)
        self.models = tuple(dict.fromkeys(models))
        self.renewal_kwargs = renewal_kwargs or {}
        self.population_at_risk = population_at_risk
        self.random_state = random_state
        self.include_ensemble = bool(include_ensemble)

    @staticmethod
    def available_models():
        return ["seasonal_naive", "renewal", "sarimax"] + ClimateMLModelFactory.available_models() + ["v1_stack"]

    @staticmethod
    def _seasonal_period(disease: pd.DataFrame, date_col="time") -> int:
        d = disease.sort_values(date_col)
        diffs = d[date_col].diff().dropna().dt.days
        if diffs.empty:
            return 12
        median_days = float(diffs.median())
        return 52 if median_days <= 8 else 12

    def fit(self, disease, climate, cutoff=None):
        d = canonicalize_disease(
            disease,
            date_col=self.date_col,
            case_col=self.case_col,
            population_col=self.population_col,
        )
        c = canonicalize_climate(
            climate,
            date_col=self.date_col,
            require_all=True,
        )
        self.cutoff_ = pd.Timestamp(cutoff) if cutoff is not None else d.time.max()
        self.disease_ = d[d.time <= self.cutoff_].copy()
        if self.disease_.empty:
            raise ValueError("No disease observations at or before the forecast origin")
        if c[c.time <= self.cutoff_].empty:
            raise ValueError("Climate data do not cover the forecast origin")
        self.climate_ = c.copy()

        self.model_warnings_ = []
        self.fitted_models_ = []

        if "seasonal_naive" in self.models:
            period = self._seasonal_period(self.disease_, self.date_col)
            self.snaive_ = SeasonalNaive(self.date_col, self.case_col, period).fit(self.disease_)
            self.fitted_models_.append("seasonal_naive")

        if "renewal" in self.models:
            kwargs = {
                "date_col": self.date_col,
                "case_col": self.case_col,
                "population_col": self.population_col,
                "climate_config": self.climate_config,
                "random_state": self.random_state,
            }
            kwargs.update(self.renewal_kwargs)
            self.renewal_ = ClimateRenewalModel(**kwargs).fit(
                self.disease_,
                c,
                self.cutoff_,
                population_at_risk=self.population_at_risk,
            )
            self.fitted_models_.append("renewal")
            if self.renewal_.population_mode_ == "relative_incidence":
                self.model_warnings_.append(
                    "Renewal model ran in relative-incidence mode because no population-at-risk value was available; susceptible depletion was not applied."
                )

        self.ml_ = {}
        for model_name in self.models:
            if model_name in ("seasonal_naive", "renewal"):
                continue
            if model_name == "sarimax":
                from .sarimax import SarimaxForecaster
                try:
                    self.ml_[model_name] = SarimaxForecaster(date_col=self.date_col, case_col=self.case_col,
                                                             random_state=self.random_state, tuning=self.tuning,
                                                             ).fit(self.disease_, c, self.cutoff_)
                    self.fitted_models_.append(model_name)
                except Exception as exc:          # convergence or data problems: skip SARIMAX, keep the run
                    self.model_warnings_.append(f"SARIMAX could not be fitted and was left out: {exc}")
                continue
            if model_name == "v1_stack":
                from .v1_stack import V1StackForecaster
                self.ml_[model_name] = V1StackForecaster(date_col=self.date_col, case_col=self.case_col,
                                                         random_state=self.random_state, tuning=self.tuning,
                                                         exclude_period=getattr(self, "exclude_period_", "2020"),
                                                         v1_mode=self.v1_mode,
                                                         ).fit(self.disease_, c, self.cutoff_)
                continue
            self.ml_[model_name] = ClimateMLForecaster(
                model_name=model_name,
                date_col=self.date_col,
                case_col=self.case_col,
                climate_config=self.climate_config,
                random_state=self.random_state,
                tuning=self.tuning,
            ).fit(self.disease_, c, self.cutoff_)
            self.fitted_models_.append(model_name)

        self.metadata_ = {
            "forecast_origin": str(self.cutoff_),
            "models": list(self.fitted_models_),
            "climate_mandatory": True,
            "n_training_observations": int(len(self.disease_)),
            "climate_variables": [x for x in ("temperature", "rainfall", "humidity", "enso") if x in c.columns],
            "population_available": "population" in self.disease_.columns or self.population_at_risk is not None,
            "seasonal_period": int(self._seasonal_period(self.disease_, self.date_col)),
            "warnings": list(self.model_warnings_),
        }
        return self

    def predict(self, future_climate, horizon, n_simulations=2000):
        if not hasattr(self, "metadata_"):
            raise RuntimeError("ClimaidV2Forecaster has not been fitted")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")

        c = canonicalize_climate(future_climate, date_col=self.date_col, require_all=True)
        dates = pd.DatetimeIndex(
            c.loc[c.time > self.cutoff_, "time"].drop_duplicates().sort_values().head(int(horizon))
        )
        if len(dates) < int(horizon):
            raise ValueError("Future climate does not cover the requested forecast horizon")

        fc = {}
        if hasattr(self, "snaive_"):
            fc["seasonal_naive"] = self.snaive_.predict(dates)
        if hasattr(self, "renewal_"):
            fc["renewal"] = self.renewal_.predict(c, int(horizon), n_simulations=n_simulations)
        for name, model in getattr(self, "ml_", {}).items():
            fc[name] = model.predict(c, int(horizon))

        # Every model must share the same forecast calendar. A previous v2
        # report exposed a mixed calendar (e.g. one model on 2021 dates and
        # others on 2026 dates). Fail fast rather than generating a misleading
        # ensemble/report.
        for name, frame in list(fc.items()):
            frame = frame.copy()
            if len(frame) != len(dates):
                raise ValueError(f"Model '{name}' returned {len(frame)} forecast rows; expected {len(dates)}")
            if self.date_col not in frame.columns:
                raise ValueError(f"Model '{name}' did not return a '{self.date_col}' column")
            actual_dates = pd.DatetimeIndex(pd.to_datetime(frame[self.date_col], errors='raise'))
            if not actual_dates.equals(dates):
                raise ValueError(
                    f"Model '{name}' returned forecast dates inconsistent with the common horizon: "
                    f"expected {dates.min().date()} to {dates.max().date()}, "
                    f"got {actual_dates.min().date()} to {actual_dates.max().date()}"
                )
            fc[name] = frame

        if self.include_ensemble and len(fc) > 1:
            fc["ensemble"] = QuantileMedianEnsemble(self.date_col).fit(fc).predict(fc)

        meta = dict(self.metadata_)
        meta["tuning"] = {k: getattr(v, "tuning_info_", {}) for k, v in getattr(self, "ml_", {}).items()}
        meta["warnings"] = list(meta.get("warnings", []))
        capped = getattr(getattr(self, "renewal_", None), "capped_fraction_", 0.0) if "renewal" in fc else 0.0
        if capped and capped > 0.01:
            meta["warnings"].append(
                f"Renewal model hit its stability cap in {capped:.0%} of simulated months "
                "(expected cases capped at 5x the training maximum, or the population). "
                "Its fitted reproduction rate implies unbounded growth; treat the renewal "
                "forecast with caution."
            )
        meta.update({
            "horizon": int(horizon),
            "n_simulations": int(n_simulations),
            "quantiles": [float(x) for x in DEFAULT_PROBS],
            "forecast_start": str(dates.min()),
            "forecast_end": str(dates.max()),
        })
        return ForecastBundle(fc, meta)

    def evaluate(self, observed, bundle: ForecastBundle):
        o = canonicalize_disease(observed, date_col=self.date_col, case_col=self.case_col)
        qcols = [f"q{int(round(p * 1000)):03d}" for p in DEFAULT_PROBS]
        rows = []
        for name, frame in bundle.forecasts.items():
            m = o[["time", "cases"]].merge(frame, on="time", how="inner", validate="one_to_one")
            if m.empty or "q500" not in m.columns:
                continue
            q = m[qcols].to_numpy(float)
            y = m.cases.to_numpy(float)
            rows.append({
                "model": name,
                "n": len(m),
                "RMSE": rmse(y, m.q500),
                "MAE": mae(y, m.q500),
                "WIS": weighted_interval_score(y, q, DEFAULT_PROBS),
                "coverage_50": coverage(y, m.q250, m.q750),
                "coverage_80": coverage(y, m.q100, m.q900),
                "coverage_95": coverage(y, m.q025, m.q975),
            })
        return pd.DataFrame(rows)
