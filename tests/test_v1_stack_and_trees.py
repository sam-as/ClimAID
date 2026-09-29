"""v1's stacked pipeline as a v2 model, and the tree-model comparison in scenarios."""
import numpy as np
import pandas as pd
import pytest

from climaid.forecasting_v2 import ClimaidV2Forecaster, HybridScenarioProjector
from tests.test_scenario_outlook import _world


def _v1_data():
    rng = np.random.default_rng(4)
    t = pd.date_range("2008-01-01", "2020-12-01", freq="MS"); m = t.month.to_numpy()
    T = 27 + 3 * np.sin(2 * np.pi * (m - 4) / 12) + rng.normal(0, .4, len(t))
    clim = pd.DataFrame({"time": t, "temperature": T, "rainfall": 5 + 3 * np.sin(2 * np.pi * (m - 7) / 12) + rng.normal(0, .5, len(t)),
                         "humidity": 14 + 2 * np.sin(2 * np.pi * m / 12), "enso": rng.normal(0, .4, len(t))})
    zT = (T - T.mean()) / T.std()
    dis = pd.DataFrame({"time": t, "cases": rng.poisson(np.exp(3 + 0.5 * np.r_[0, zT[:-1]]))})
    return dis[t >= "2009-01-01"], clim


def test_v1_stack_is_listed():
    assert "v1_stack" in ClimaidV2Forecaster.available_models()


@pytest.mark.slow
def test_v1_stack_runs_as_a_v2_model():
    dis, clim = _v1_data()
    e = ClimaidV2Forecaster(models=("seasonal_naive", "v1_stack"), tuning="fast", v1_mode="quick").fit(dis, clim, "2019-12-31")
    b = e.predict(clim[clim.time > "2019-12-31"].head(12), 12, n_simulations=50)
    f = b.forecasts["v1_stack"]
    q = [c for c in f.columns if c.startswith("q")]
    assert len(f) == 12 and (np.diff(f[q].to_numpy(), axis=1) >= -1e-9).all() and (f[q].to_numpy() >= 0).all()
    info = b.metadata["tuning"]["v1_stack"]
    assert info["v1_test_year"] == 2019 and set(info["selected_lags"]) >= {"mean_temperature", "mean_Rain"}
    assert "ensemble" in b.forecasts                           # joins the ensemble like any model


def test_tree_comparison_reports_changes_and_out_of_range_share():
    dis, clim, proj = _world()
    o = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10, tuning=3).project(
        dis, clim, proj, "2020-12-31", end_year=2045)
    t = o.tree_comparison
    assert set(t.model) == {"random_forest", "gradient_boosting"}
    assert t.share_months_outside_training_range.between(0, 1).all()
    assert (t.change_min_gcm <= t.change_median_across_gcms).all() and (t.change_median_across_gcms <= t.change_max_gcm).all()


def test_tree_comparison_can_be_switched_off():
    dis, clim, proj = _world()
    o = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10, comparison_models=()).project(
        dis, clim, proj, "2020-12-31", end_year=2035)
    assert o.tree_comparison.empty
