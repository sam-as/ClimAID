"""Interval calibration, multi-district pooling, thermal curve, long-term
backtest and population scaling (ClimAID v2.1)."""
import numpy as np
import pandas as pd
import pytest

from climaid.forecasting_v2.calibration import fit_interval_calibration, apply_interval_calibration, QCOLS
from climaid.forecasting_v2 import HybridScenarioProjector
from climaid.forecasting_v2.scenario import thermal_suitability, TEMPERATURE_CURVES
from tests.test_scenario_outlook import _world


# ---------------- interval calibration
def _hindcast_frame(width, n=60, seed=0):
    """Forecasts whose intervals are `width` x too narrow for N(0, 10) errors."""
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2015-12-31")
    t = pd.date_range("2016-01-01", periods=n, freq="MS")
    y = 50 + rng.normal(0, 10, n)
    z = {"q025": -1.96, "q050": -1.645, "q100": -1.2816, "q250": -.6745, "q500": 0,
         "q750": .6745, "q900": 1.2816, "q950": 1.645, "q975": 1.96}
    f = pd.DataFrame({"time": t, "origin": origin, "model": "m", **{k: 50 + v * 10 / width for k, v in z.items()}})
    return f, pd.DataFrame({"time": t, "cases": y}), origin


def test_calibration_widens_too_narrow_intervals_to_nominal():
    f, obs, origin = _hindcast_frame(width=2.5)
    cal = fit_interval_calibration(f, obs)["m"]
    assert 1.8 < cal["pooled"][0.95] < 3.5
    g = apply_interval_calibration(f, cal, origin)
    cov95 = ((obs.cases >= g.q025) & (obs.cases <= g.q975)).mean()
    assert cov95 >= 0.9
    assert (np.diff(g[QCOLS].to_numpy(), axis=1) >= -1e-9).all()      # quantiles stay ordered


def test_calibration_leaves_well_calibrated_intervals_roughly_alone():
    f, obs, _ = _hindcast_frame(width=1.0, n=200)
    cal = fit_interval_calibration(f, obs)["m"]
    assert 0.8 < cal["pooled"][0.95] < 1.3 and 0.7 < cal["pooled"][0.5] < 1.4


def test_small_bands_fall_back_to_pooled_factor_for_extreme_levels():
    f, obs, _ = _hindcast_frame(width=2.0, n=30)
    cal = fit_interval_calibration(f, obs)["m"]
    for band, ks in cal["bands"].items():
        if cal["n"][band] < 40:
            assert ks[0.95] == cal["pooled"][0.95]


def test_forecast_v2_calibrates_and_keeps_uncalibrated_metrics():
    from tests.test_v2_origin_and_2020 import _dm
    dm = _dm()
    out = dm.forecast_v2(forecast_origin="2018-12-31", horizon=12, n_simulations=200, models=("seasonal_naive", "poisson"),
                         run_hindcasts=True, hindcast_origins=4, drop_2020=False)
    assert out["interval_calibration"] and "interval_calibration" in out["metadata"]
    assert not out["metrics_uncalibrated"].empty and not out["metrics"].empty


# ---------------- thermal curve
def test_thermal_curve_shape():
    tmin, topt, tmax = TEMPERATURE_CURVES["aedes_aegypti_mordecai2017"]
    s = thermal_suitability([tmin - 1, tmin, topt, tmax, tmax + 1], "aedes_aegypti_mordecai2017")
    assert list(s) == [0, 0, 1, 0, 0]
    assert thermal_suitability(topt - 2, (tmin, topt, tmax)) > thermal_suitability(topt - 6, (tmin, topt, tmax))


def test_projection_runs_with_thermal_curve():
    dis, clim, proj = _world()
    o = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10,
                                temperature_curve="aedes_aegypti_mordecai2017").project(dis, clim, proj, "2020-12-31", end_year=2040)
    assert o.metadata["temperature_curve"]["tmin_topt_tmax"] == [17.8, 29.1, 34.6]
    assert not any("temperature_lvl_sq" in k for k in o.metadata["climate_coefficients"])


# ---------------- pooling
def test_pooled_fit_uses_district_intercepts_and_target_reference():
    dis, clim, proj = _world(0)
    others = [(_world(s)[0].assign(cases=lambda d, k=s: (d.cases * (1 + k)).astype(int)), _world(s)[1]) for s in (1, 2)]
    o = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10).project(
        dis, clim, proj, "2020-12-31", end_year=2040, extra_districts=others)
    assert o.metadata["pooled_districts"] == 2
    assert {"district_1", "district_2"} <= set(o.metadata["climate_coefficients"])
    # target-district baseline, not a pooled average: matches its own observed mean closely
    assert abs(o.baseline["model_annual_mean"] / o.baseline["observed_annual_mean"] - 1) < 0.1


def test_pooling_rejected_in_anomaly_mode():
    dis, clim, proj = _world()
    with pytest.raises(ValueError, match="requires response='seasonal'"):
        HybridScenarioProjector(response="anomaly", near_term_months=0, blend_months=0).project(
            dis, clim, proj, "2020-12-31", end_year=2030, extra_districts=[(dis, clim)])


# ---------------- backtest and population
def test_backtest_returns_per_year_table_and_summary():
    dis, clim, _ = _world()
    pr = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10)
    tab, sm = pr.backtest(dis, clim, test_years=4)
    assert len(tab) == 4 and set(tab.columns) >= {"observed", "projected_median", "p10", "p90", "inside_80"}
    assert sm["test_years"] == "2017-2020" and 0 <= sm["share_inside_80"] <= 1


def test_population_scaling_multiplies_projections():
    dis, clim, proj = _world()
    years = np.arange(2000, 2051)
    pop = pd.DataFrame({"year": years, "population": np.where(years <= 2020, 1e6, 2e6)})
    o = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10, random_state=3).project(
        dis, clim, proj, "2020-12-31", end_year=2040, population_projection=pop)
    a = o.decades.set_index(["ssp", "decade"]).annual_median
    b = o.decades_with_population.set_index(["ssp", "decade"]).annual_median
    assert np.allclose((b / a).loc[:, "2030s"], 2.0, rtol=0.02)
