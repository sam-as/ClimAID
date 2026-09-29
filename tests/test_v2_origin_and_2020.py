"""forecast_v2: hindcasts stay before the forecast origin; drop_2020 option;
renewal stability cap; C-DSI v2 calibration tolerance and trend wording."""
import numpy as np
import pandas as pd
import pytest

from climaid.climaid_model import DiseaseModel
from climaid.forecasting_v2 import ClimateRenewalModel, DEFAULT_V2_MODELS
from climaid.reporting_v2 import _calibration_label, _trend_sentence, _disagreement_sentence


def _dm(years=range(2010, 2024)):
    rng = np.random.default_rng(0)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")
    m = dates.month.to_numpy()
    cases = np.maximum(0, np.round(30 + 12 * np.sin(2 * np.pi * (m - 5) / 12) + rng.normal(0, 3, len(dates))))
    cases[dates.year == 2020] = 1          # an anomalous "COVID" year
    dm = object.__new__(DiseaseModel)
    dm.target_col = "Count"; dm.random_state = 42; dm.district = "X"; dm.disease_name = "Test"
    dm.df_disease = pd.DataFrame({"time": dates, "Count": cases})
    dm.df_climate_hist = pd.DataFrame({
        "time": dates, "mean_temperature": 27 + 2 * np.sin(2 * np.pi * m / 12),
        "mean_Rain": 5 + 3 * np.sin(2 * np.pi * (m - 2) / 12), "mean_SH": 14 + np.sin(2 * np.pi * m / 12),
        "Nino_anomaly": rng.normal(0, .4, len(dates))})
    dm.df_climate_proj = None
    return dm


def _run(dm, **kw):
    return dm.forecast_v2(forecast_origin="2020-12-31", horizon=12, n_simulations=200,
                          models=("seasonal_naive",), run_hindcasts=True, hindcast_origins=4, **kw)


def test_hindcasts_never_extend_past_the_forecast_origin():
    out = _run(_dm())
    hm = out["hindcast_metrics"]
    assert not hm.empty and "hindcast_error" not in set(hm["model"])
    last_month_scored = pd.to_datetime(hm["origin"]).max() + pd.DateOffset(months=12)
    assert last_month_scored <= pd.Timestamp("2020-12-31")


def test_drop_2020_default_replaces_2020_with_monthly_medians_and_says_so():
    out = _run(_dm())
    meta = out["bundle"].metadata if "bundle" in out else out["metadata"]
    assert meta["drop_2020"] is True
    assert meta["excluded_period"] == "2020"
    assert any("Disrupted period excluded (2020)" in w for w in meta["warnings"])
    # Seasonal naive for 2021 would copy 2020 (all 1s) without the replacement.
    sn = (out["bundle"].forecasts if "bundle" in out else out["forecasts"])["seasonal_naive"]
    assert sn["q500"].mean() > 10


def test_drop_2020_false_keeps_recorded_2020_counts():
    out = _run(_dm(), drop_2020=False)
    meta = out["bundle"].metadata if "bundle" in out else out["metadata"]
    assert meta["drop_2020"] is False
    assert meta["excluded_period"] is None
    assert not any("Disrupted period excluded" in w for w in meta["warnings"])
    sn = (out["bundle"].forecasts if "bundle" in out else out["forecasts"])["seasonal_naive"]
    assert sn["q500"].mean() < 5


def test_default_v2_models_include_several_ml_learners():
    ml = set(DEFAULT_V2_MODELS) - {"seasonal_naive", "renewal"}
    assert len(ml) >= 3 and "random_forest" in ml


def test_renewal_is_capped_when_growth_would_explode():
    dm = _dm(range(2012, 2016))
    disease = dm.df_disease.rename(columns={"Count": "cases"})
    clim = dm.df_climate_hist
    model = ClimateRenewalModel().fit(disease, clim, cutoff="2014-12-31")
    model.coef_ = np.array(model.coef_, dtype=float); model.coef_[0] += 4.0   # force R >> 1
    fut = clim[clim.time > "2014-12-31"].head(12)
    pred = model.predict(fut, 12, n_simulations=200)
    cap = 5 * disease[disease.time <= "2014-12-31"]["cases"].max()
    assert model.capped_fraction_ > 0
    assert pred["q500"].max() <= cap * 3     # NB noise around a capped mean, not ~10^4


@pytest.mark.parametrize("emp,nominal,n,expected", [
    (0.81, 0.95, 48, "under-covered"),     # was wrongly "consistent" with a fixed 15-point band
    (0.33, 0.50, 48, "under-covered"),
    (0.90, 0.95, 12, "consistent"),        # small n -> wider tolerance
    (0.95, 0.95, 48, "consistent"),
])
def test_calibration_tolerance_scales_with_sample_size(emp, nominal, n, expected):
    assert _calibration_label(emp, nominal, n).startswith(expected)


def test_trend_sentence_describes_dip_and_recovery():
    t = pd.date_range("2021-01-01", periods=12, freq="MS")
    y = [24, 10, 3, 3, 0, 6, 3, 6, 7, 11, 11, 10]
    txt = _trend_sentence(pd.DataFrame({"time": t, "q500": y, "q025": 0, "q975": 80}))
    assert "lowest point of 0.0 around May 2021" in txt and "rises again" in txt
    assert "95% interval spans 0–80" in txt


def test_disagreement_is_flagged():
    t = pd.date_range("2021-01-01", periods=3, freq="MS")
    fc = {"a": pd.DataFrame({"time": t, "q500": [1, 1, 1]}), "b": pd.DataFrame({"time": t, "q500": [30, 30, 30]})}
    assert "Models disagree substantially" in _disagreement_sentence(fc)


def test_custom_exclusion_period_replaces_only_those_months():
    dm = _dm()
    out = _run(dm, exclude_period="2020-03:2020-06")
    meta = out["metadata"]
    assert meta["excluded_period"] == "March 2020 to June 2020" and meta["excluded_months"] == 4
    sn = out["forecasts"]["seasonal_naive"].set_index("time")["q500"]
    # Jan-Feb and Jul-Dec 2020 keep their recorded value (1); Mar-Jun 2021 copy replaced (typical) values
    assert sn.loc["2021-03-01":"2021-06-01"].min() > 10 and sn.loc["2021-01-01"] < 5 and sn.loc["2021-09-01"] < 5


def test_exclusion_option_parsing():
    import pandas as pd
    from climaid.exclusion import resolve_exclusion, describe
    assert describe(resolve_exclusion()) == "2020"
    assert resolve_exclusion("none") is None and resolve_exclusion(None, drop_2020=False) is None
    assert resolve_exclusion("2020-04:2020-09") == (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-09-01"))
    assert resolve_exclusion(("2020-04-15", "2020-05-20")) == (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01"))
    for bad in ("2021", "2020-09:2020-03", "sometime"):
        with pytest.raises(ValueError):
            resolve_exclusion(bad)
