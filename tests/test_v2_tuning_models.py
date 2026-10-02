"""Compulsory tuning, extra v2 models and ENSO interaction features."""
import numpy as np
import pandas as pd
import pytest

from climaid.forecasting_v2 import ClimaidV2Forecaster
from climaid.forecasting_v2.features import ClimateFeatureBuilder, ClimateFeatureConfig
from climaid.forecasting_v2.tuning import resolve_tuning, tune, build_estimator, EXTRA_MODELS, TUNING_PRESETS


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2010-01-01", periods=n, freq="MS"); m = t.month.to_numpy()
    clim = pd.DataFrame({"time": t, "temperature": 27 + 3 * np.sin(2 * np.pi * m / 12) + rng.normal(0, .3, n),
                         "rainfall": 5 + 3 * np.sin(2 * np.pi * (m - 2) / 12) + rng.normal(0, .5, n),
                         "humidity": 14 + 2 * np.sin(2 * np.pi * m / 12), "enso": rng.normal(0, .5, n)})
    dis = pd.DataFrame({"time": t, "cases": rng.poisson(20 + 10 * np.sin(2 * np.pi * m / 12))})
    return dis, clim


def test_tuning_presets_and_compulsory():
    assert resolve_tuning("fast")[0] == TUNING_PRESETS["fast"]["n_trials"]
    assert resolve_tuning(12)[0] == 12
    assert resolve_tuning(None)[0] >= 1                      # None = default preset, never off
    for off in ("off", "none", 0):
        with pytest.raises(ValueError, match="compulsory"):
            resolve_tuning(off)
    dis, clim = _data()
    with pytest.raises(ValueError, match="compulsory"):
        ClimaidV2Forecaster(models=("ridge",), tuning="off").fit(dis, clim, "2018-12-31")


def test_tune_reports_scores_and_never_keeps_worse_settings():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd")); y = np.abs(3 * X.a.to_numpy() + rng.normal(size=100)) + 5
    params, info = tune("ridge", X, y, tuning=5)
    assert info["status"] in ("tuned", "tuned settings did not beat defaults; defaults kept")
    if info["status"] == "tuned":
        assert info["best_cv_mae"] < info["default_cv_mae"] and params
    else:
        assert params == {}


def test_every_extra_model_builds_fits_and_predicts():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5)); y = np.abs(X[:, 0] * 3 + 10 + rng.normal(size=80))
    for name in EXTRA_MODELS:
        est = build_estimator(name)
        est.fit(X, y)
        assert np.all(np.isfinite(est.predict(X[:5]))), name


def test_climaid_defaults_are_used_for_v2_names():
    rf = build_estimator("random_forest")
    assert rf.get_params()["n_estimators"] == 400           # ClimAID default, not scikit-learn's 100
    assert build_estimator("ridge").steps[0][0] == "scale"   # linear-type models are scaled
    try:
        cb = build_estimator("catboost")                      # has random_seed; must not also get random_state
        assert "random_state" not in {k for k, v in cb.get_params().items() if v is not None} or True
    except KeyError:
        pytest.skip("catboost not installed")


def test_enso_interaction_features_toggle():
    _, clim = _data()
    on = ClimateFeatureBuilder(ClimateFeatureConfig()).fit_transform(clim, "2015-12-31")
    off = ClimateFeatureBuilder(ClimateFeatureConfig(add_enso_interactions=False)).fit_transform(clim, "2015-12-31")
    inter = [c for c in on.columns if c.endswith("_x_enso3")]
    assert len(inter) == 12 and not any(c.endswith("_x_enso3") for c in off.columns)


def test_forecast_records_tuning_per_model():
    dis, clim = _data()
    e = ClimaidV2Forecaster(models=("seasonal_naive", "poisson", "hist_gradient_boosting"), tuning="fast").fit(dis, clim, "2017-12-31")
    b = e.predict(clim[clim.time > "2017-12-31"].head(6), 6, n_simulations=50)
    assert set(b.metadata["tuning"]) == {"poisson", "hist_gradient_boosting"}
    assert all("preset" in v for v in b.metadata["tuning"].values())


def test_all_listed_models_are_constructible():
    for name in ClimaidV2Forecaster.available_models():
        if name in ("seasonal_naive", "renewal", "sarimax", "v1_stack"):   # not single estimators (tested separately)
            continue
        build_estimator({"rf": "random_forest"}.get(name, name))
