"""v1 modes shared across interfaces, and distributed lag effects."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climaid.model_parameters import V1_MODES, v1_mode_config
from climaid.model_registry import MODEL_REGISTRY
from climaid.forecasting_v2.distributed_lag import lag_basis, cross_basis, lag_curve
from climaid.forecasting_v2 import ClimaidV2Forecaster, HybridScenarioProjector
from climaid.forecasting_v2.v1_stack import _effort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import generators as G  # noqa: E402


def test_v1_modes_match_the_documented_presets_and_use_registered_models():
    assert V1_MODES["fast"]["n_trials"] == 50 and V1_MODES["balanced"]["n_trials"] == 200 and V1_MODES["deep"]["n_trials"] == 500
    for mode in ("fast", "balanced", "deep", "quick"):
        cfg = v1_mode_config(mode)
        for k in ("base_models", "residual_models", "correction_models"):
            assert all(m in MODEL_REGISTRY for m in cfg[k]), (mode, k, cfg[k])


def test_v1_stack_follows_v1_modes():
    assert _effort("fast")[0]["n_trials"] == 50
    assert _effort("balanced")[0]["n_trials"] == 200
    assert _effort("deep")[0]["n_trials"] == 500
    assert "temp_range" not in _effort("deep")[0]          # v1 default lag ranges (ENSO 0-12)
    assert _effort("fast", v1_mode="quick")[0]["n_trials"] == 3


def test_dashboard_and_wizard_share_the_v1_modes():
    from climaid.browser_ui.api import _legacy_model_config, WizardConfig
    for preset in ("fast", "balanced", "deep"):
        cfg = WizardConfig(mode="southasia", country="IN", district="X", state="Y", disease_name="D", preset=preset)
        b, r, c, n = _legacy_model_config(cfg)
        ref = v1_mode_config(preset)
        assert (b, r, c, n) == (ref["base_models"], ref["residual_models"], ref["correction_models"], ref["n_trials"])
    src = (Path(__file__).resolve().parents[1] / "climaid" / "wizard.py").read_text()
    assert "v1_mode_config(mode)" in src and "elastic_net" not in src


def test_lag_basis_and_cumulative_effect():
    B = lag_basis(6, 2)
    assert B.shape == (7, 3) and np.allclose(B[:, 0], 1)
    cb = cross_basis(np.ones(20), 6, 2)
    assert np.isnan(cb[:6]).all() and np.allclose(cb[6:, 0], 7)
    # a constant lag curve of 0.1 per lag gives a cumulative effect of 0.7
    assert lag_curve([0.1, 0, 0], 6)["cumulative"] == pytest.approx(0.7)


def test_distributed_lags_used_by_regression_models_only():
    d, c = G.seasonal()
    e = ClimaidV2Forecaster(models=("poisson", "random_forest"), tuning=2, include_ensemble=False).fit(d, c, "2019-12-31")
    assert e.ml_["poisson"].uses_distributed_lags_ and not e.ml_["random_forest"].uses_distributed_lags_
    e2 = ClimaidV2Forecaster(models=("poisson",), tuning=2, include_ensemble=False)
    e2.fit(d, c, "2019-12-31")
    assert any("enso_dl" in f for f in e2.ml_["poisson"].feature_names_)   # ENSO lags up to 12 months


def test_scenario_distributed_lag_option_reports_lag_curves():
    from tests.test_scenario_outlook import _world
    dis, clim, proj = _world()
    o = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=10, lag_selection="distributed",
                                comparison_models=()).project(dis, clim, proj, "2020-12-31", end_year=2040)
    curves = o.metadata["distributed_lag_curves"]
    assert set(curves) >= {"temperature", "rainfall", "enso"}
    assert len(curves["enso"]["by_lag"]) == 13 and len(curves["temperature"]["by_lag"]) == 7
    assert "distributed lag" in o.metadata["lag_selection"]


@pytest.mark.slow
def test_project_v2_v1_lags_runs_v1_search_on_pre_origin_data():
    from climaid.climaid_model import DiseaseModel
    from tests.test_scenario_outlook import _world
    dis, clim, proj = _world()
    dm = object.__new__(DiseaseModel)
    dm.target_col = "Count"; dm.random_state = 1; dm.district = "X"; dm.disease_name = "T"
    dm.df_disease = dis.rename(columns={"cases": "Count"}); dm.df_climate_hist = clim; dm.df_climate_proj = proj
    r = dm.project_v2(forecast_origin="2020-12-31", end_year=2035, lag_selection="v1", v1_mode="quick", n_bootstrap=10,
                      near_term_models=("seasonal_naive",), sensitivity=False, run_backtest=False, comparison_models=(),
                      save_report=False)
    notes = " ".join(r["metadata"]["notes"])
    assert "lags selected by ClimAID v1 (v1 quick check" in notes and "December 2020" in notes
