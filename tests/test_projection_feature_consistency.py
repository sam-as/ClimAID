"""Regression test for a train/inference consistency bug found in
DiseaseProjection.prepare_features() (climaid/climaid_projections.py), the
feature builder used for CMIP6 projections.

It used to compute the "MA_*"/"YA_*" climate-average features completely
differently from how DiseaseModel._merge_data() computes the identically
named features the model was actually trained on:
  - YA_* here was a whole-calendar-year pooled mean
    (df.groupby("Year")[var].transform("mean")), vs. a strictly
    backward-looking trailing 12-month rolling mean at train time.
  - MA_* here was a same-calendar-month rolling mean across the last 10
    occurrences of that month, vs. a trailing 120-row (10 years, all
    months) rolling mean at train time.

Same feature name, two different formulas -- every CMIP6 projection fed
the trained model a value it had never seen associated with that feature
name during training, silently degrading projection quality without any
error being raised. Fixed so both use the exact same trailing-window
definitions.
"""
import numpy as np
import pandas as pd
import pytest

from climaid.climaid_model import DiseaseModel
from climaid.climaid_projections import DiseaseProjection


def _make_projection(features):
    dp = object.__new__(DiseaseProjection)
    dp.target_col = "Count"
    dp.rmse = 1.0
    dp.scaler = None
    dp.best_config = {"features": features}
    dp.feature_metadata = {
        "lags": {"mean_temperature": [0]},
        "interactions": [],
        "scaled_columns": [],
    }
    return dp


def _synthetic_climate(n=200):
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    idx = np.arange(n, dtype=float)
    return pd.DataFrame({
        "time": dates, "mean_temperature": idx, "mean_Rain": idx, "mean_SH": idx,
        "Nino_anomaly": idx,
    })


def test_prepare_features_matches_merge_data_exactly_for_ma_and_ya():
    climate = _synthetic_climate()

    disease = pd.DataFrame({"time": climate["time"]})
    disease["Year"] = disease.time.dt.year
    disease["Month"] = disease.time.dt.month
    disease["Count"] = 1

    dm = object.__new__(DiseaseModel)
    dm.df_disease = disease
    dm.df_climate_hist = climate.assign(Dist_States="IN_Test_State")
    merged = dm._merge_data()

    dp = _make_projection(["mean_temperature_lag0", "YA_mean_temperature", "MA_mean_temperature"])
    proj_climate = climate.copy()
    proj_climate["model"] = "GCM1"
    proj_climate["ssp"] = "ssp245"
    out = dp.prepare_features(proj_climate)

    check_date = pd.Timestamp("2020-06-01")
    m_row = merged[merged.time == check_date].iloc[0]
    p_row = out[out.time == check_date].iloc[0]

    assert p_row["YA_mean_temperature"] == pytest.approx(m_row["YA_mean_temperature"])
    assert p_row["MA_mean_temperature"] == pytest.approx(m_row["MA_mean_temperature"])


def test_prepare_features_drops_rows_with_incomplete_rolling_windows():
    """Matching training's min_periods (12, 120) means the first 11/119
    rows of a series now correctly produce NaN MA_/YA_ values -- these must
    be dropped rather than silently reaching self.model.predict() as NaN."""
    climate = _synthetic_climate(n=15)
    climate["model"] = "GCM1"
    climate["ssp"] = "ssp245"

    dp = _make_projection(["mean_temperature_lag0", "YA_mean_temperature", "MA_mean_temperature"])
    out = dp.prepare_features(climate)

    assert out[["YA_mean_temperature", "MA_mean_temperature"]].isna().sum().sum() == 0
    # 15 rows in, needs 12 for YA_ to be valid -> at most 4 valid rows.
    assert len(out) <= 4


def test_prepare_features_respects_multi_scenario_grouping():
    """MA_/YA_ must be computed per (model, ssp) group, not pooled across
    different climate scenarios sharing the same calendar year/month."""
    dates = pd.date_range("2015-01-01", periods=24, freq="MS")
    warm = pd.DataFrame({
        "time": dates, "mean_temperature": np.full(24, 30.0), "mean_Rain": 5.0,
        "mean_SH": 70.0, "Nino_anomaly": 0.0, "model": "GCM_WARM", "ssp": "ssp585",
    })
    cool = pd.DataFrame({
        "time": dates, "mean_temperature": np.full(24, 10.0), "mean_Rain": 5.0,
        "mean_SH": 70.0, "Nino_anomaly": 0.0, "model": "GCM_COOL", "ssp": "ssp126",
    })
    combined = pd.concat([warm, cool], ignore_index=True)

    dp = _make_projection(["mean_temperature_lag0", "YA_mean_temperature", "MA_mean_temperature"])
    out = dp.prepare_features(combined)

    warm_out = out[out["model"] == "GCM_WARM"]
    cool_out = out[out["model"] == "GCM_COOL"]
    assert not warm_out.empty and not cool_out.empty
    # If scenarios were pooled together, both would show a blended ~20.0;
    # kept separate, each should reflect only its own constant value.
    assert (warm_out["YA_mean_temperature"] == 30.0).all()
    assert (cool_out["YA_mean_temperature"] == 10.0).all()
