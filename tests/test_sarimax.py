"""SARIMAX in ClimAID v2: interface, leakage safety, tuning record and graceful failure."""
import numpy as np
import pandas as pd
import pytest

from climaid.forecasting_v2 import ClimaidV2Forecaster
from climaid.forecasting_v2.sarimax import SarimaxForecaster
from test_assistant import _synthetic

ORIGIN = pd.Timestamp("2023-12-01")


@pytest.fixture(scope="module")
def data():
    disease, climate = _synthetic()
    return disease.rename(columns={"date": "time"}), climate


@pytest.fixture(scope="module")
def fitted(data):
    disease, climate = data
    return SarimaxForecaster(case_col="cases", tuning=3).fit(disease, climate, ORIGIN)


def test_forecast_has_ordered_non_negative_quantiles(fitted, data):
    _, climate = data
    f = fitted.predict(climate, 6)
    assert list(f["time"]) == list(pd.date_range("2024-01-01", periods=6, freq="MS"))
    q = f.filter(like="q").to_numpy()
    assert (q >= 0).all() and (np.diff(q, axis=1) >= -1e-9).all()


def test_tuning_is_recorded(fitted):
    info = fitted.tuning_info_
    assert info["status"] in ("tuned", "defaults kept") and info["n_trials"] == 3
    assert len(info["order"]) == 3 and info["seasonal_order"][-1] == 12
    assert 0 <= info["climate_lag_months"] <= 3 and info["scale"] == "log(1 + cases)"


def test_cases_after_the_origin_are_never_used(data, fitted):
    disease, climate = data
    corrupted = disease.copy()
    corrupted.loc[corrupted["time"] > ORIGIN, "cases"] = 10 ** 6
    other = SarimaxForecaster(case_col="cases", tuning=3).fit(corrupted, climate, ORIGIN)
    pd.testing.assert_frame_equal(other.predict(climate, 6), fitted.predict(climate, 6))


def test_climate_statistics_use_training_data_only(data, fitted):
    disease, climate = data
    shifted = climate.copy()
    later = shifted["time"] > ORIGIN + pd.DateOffset(months=6)    # beyond the forecast window
    shifted.loc[later, ["temperature", "rainfall", "humidity"]] += 50
    other = SarimaxForecaster(case_col="cases", tuning=3).fit(disease, shifted, ORIGIN)
    pd.testing.assert_frame_equal(other.predict(shifted, 6), fitted.predict(climate, 6))


def test_rejects_weekly_and_too_short_data(data):
    disease, climate = data
    weekly = pd.DataFrame({"time": pd.date_range("2015-01-05", periods=300, freq="W-MON"), "cases": 5})
    with pytest.raises(ValueError, match="monthly"):
        SarimaxForecaster(case_col="cases", tuning=1).fit(weekly, climate)
    with pytest.raises(ValueError, match="at least"):
        SarimaxForecaster(case_col="cases", tuning=1).fit(disease.head(20), climate)


def test_forecaster_offers_sarimax_and_skips_it_gracefully(data):
    disease, climate = data
    assert "sarimax" in ClimaidV2Forecaster.available_models()
    short = disease[disease["time"] <= "2013-06-01"]       # too short for SARIMAX, enough for the baseline
    f = ClimaidV2Forecaster(case_col="cases", models=("seasonal_naive", "sarimax"), tuning=1,
                            include_ensemble=False).fit(short, climate)
    assert f.fitted_models_ == ["seasonal_naive"]
    assert any("SARIMAX could not be fitted" in w for w in f.metadata_["warnings"])
