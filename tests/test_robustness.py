"""Tests for the sixth-pass robustness improvements."""
import warnings

import numpy as np
import pandas as pd
import pytest

from climaid.climaid_model import _accepts_random_state, _validate_disease_frame
from climaid.model_registry import MODEL_REGISTRY
from climaid.forecasting_v2 import HindcastEvaluator
from climaid.reporting_v2 import generate_c_dsi_v2, _rank_models_by_wis


# ---------------------------------------------------------------------------
# random_state detection
# ---------------------------------------------------------------------------
def test_accepts_random_state_detects_seedable_models():
    # The old check ("random_state" in str(cls)) returned False for all of these.
    for name in ("rf", "gbr", "extra_trees", "mlp", "ridge"):
        assert _accepts_random_state(MODEL_REGISTRY[name]), name
    for name in ("linear", "isotonic"):
        assert not _accepts_random_state(MODEL_REGISTRY[name]), name


def test_accepts_random_state_handles_kwargs_only_wrappers():
    if "xgb" not in MODEL_REGISTRY:
        pytest.skip("xgboost not installed")
    assert _accepts_random_state(MODEL_REGISTRY["xgb"])


# ---------------------------------------------------------------------------
# Disease input validation
# ---------------------------------------------------------------------------
def _monthly(n=24):
    return pd.DataFrame({"time": pd.date_range("2015-01-01", periods=n, freq="MS"),
                         "Count": np.arange(n)})


def test_validation_drops_bad_dates_and_non_numeric_counts():
    df = _monthly(6).astype({"time": str, "Count": object})  # as read from a CSV
    df.loc[1, "time"] = "not a date"
    df.loc[2, "Count"] = "<5"
    df.loc[3, "Count"] = "7"  # numeric string must be kept and converted
    out = _validate_disease_frame(df, verbose=False)
    assert len(out) == 4
    assert out["Count"].dtype.kind in "if"
    assert 7 in out["Count"].tolist()
    assert out["Year"].notna().all() and out["Month"].notna().all()


def test_validation_rejects_negative_counts():
    df = _monthly(6)
    df.loc[2, "Count"] = -3
    with pytest.raises(ValueError, match="negative"):
        _validate_disease_frame(df, verbose=False)


def test_validation_drops_exact_duplicates():
    df = pd.concat([_monthly(6), _monthly(6).iloc[:2]], ignore_index=True)
    out = _validate_disease_frame(df, verbose=False)
    assert len(out) == 6
    assert out["time"].is_unique


def test_validation_aggregates_weekly_data_to_monthly_totals():
    weeks = pd.date_range("2020-01-06", periods=9, freq="W-MON")  # Jan-Mar 2020
    df = pd.DataFrame({"time": weeks, "Count": 1})
    out = _validate_disease_frame(df, verbose=False)
    assert out["time"].is_unique
    assert len(out) == 3
    assert out["Count"].sum() == 9  # totals preserved
    assert (out["time"].dt.day == 1).all()


def test_validation_leaves_clean_monthly_data_unchanged():
    df = _monthly(12)
    out = _validate_disease_frame(df, verbose=False)
    pd.testing.assert_series_equal(out["Count"], df["Count"], check_dtype=False, check_names=False)


# ---------------------------------------------------------------------------
# Hindcast failure reporting
# ---------------------------------------------------------------------------
def _synthetic(n_months=120, climate_months=None):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    mo = dates.month.to_numpy()
    disease = pd.DataFrame({"time": dates, "cases": np.maximum(0, np.round(
        30 + 10 * np.sin(2 * np.pi * mo / 12) + rng.normal(0, 3, n_months)))})
    cdates = dates[: climate_months or n_months]
    cm = cdates.month.to_numpy()
    climate = pd.DataFrame({
        "time": cdates,
        "temperature": 27 + 2 * np.sin(2 * np.pi * cm / 12),
        "rainfall": 5 + 2 * np.sin(2 * np.pi * cm / 12),
        "humidity": 70 + 5 * np.sin(2 * np.pi * cm / 12),
        "enso": rng.normal(0, .3, len(cdates)),
    })
    return disease, climate


def test_hindcast_raises_with_reasons_when_every_origin_fails():
    # Climate stops well before any hindcast origin's horizon ends.
    disease, climate = _synthetic(n_months=120, climate_months=40)
    hc = HindcastEvaluator(horizon=12, n_origins=3, min_train_size=48)
    with pytest.raises(RuntimeError, match="All 3 hindcast origins failed"):
        hc.evaluate(disease, climate, models=("seasonal_naive",), n_simulations=50)
    assert len(hc.skipped_origins_) == 3
    assert hc.skipped_origins_["reason"].str.contains("climate months").all()


def test_hindcast_warns_and_records_partial_failures():
    # Enough climate for early origins but not the last one.
    disease, climate = _synthetic(n_months=120, climate_months=110)
    hc = HindcastEvaluator(horizon=12, n_origins=3, min_train_size=48)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, metrics = hc.evaluate(disease, climate, models=("seasonal_naive",), n_simulations=50)
    assert not metrics.empty
    assert len(hc.skipped_origins_) >= 1
    assert any("origins skipped" in str(x.message) for x in w)


def test_hindcast_clean_run_records_no_skips():
    disease, climate = _synthetic()
    hc = HindcastEvaluator(horizon=12, n_origins=2, min_train_size=48)
    _, metrics = hc.evaluate(disease, climate, models=("seasonal_naive",), n_simulations=50)
    assert not metrics.empty
    assert hc.skipped_origins_.empty


# ---------------------------------------------------------------------------
# C-DSI v2 handling of failed hindcasts
# ---------------------------------------------------------------------------
def _error_row():
    return pd.DataFrame([{"origin": "N/A", "model": "hindcast_error", "n": 0, "RMSE": np.nan,
                          "MAE": np.nan, "WIS": np.nan, "coverage_50": np.nan,
                          "coverage_80": np.nan, "coverage_95": np.nan,
                          "error": "RuntimeError: All 3 hindcast origins failed"}])


def test_error_row_is_never_ranked_as_a_model():
    assert _rank_models_by_wis(_error_row()) == []


def test_c_dsi_v2_reports_hindcast_error_and_falls_back_to_held_out():
    dates = pd.date_range("2021-01-01", periods=3, freq="MS")
    frame = pd.DataFrame({"time": dates, "q500": [10.0, 12.0, 11.0]})
    held_out = pd.DataFrame([{"model": "seasonal_naive", "WIS": 1.0,
                              "coverage_50": 0.5, "coverage_80": 0.8, "coverage_95": 0.95}])
    out = generate_c_dsi_v2(
        disease_name="Dengue", district="X", metadata={"models": ["seasonal_naive"]},
        forecasts={"seasonal_naive": frame}, metrics=held_out, hindcast_metrics=_error_row(),
    )
    assert "Rolling hindcasts were requested but failed" in out
    assert "All 3 hindcast origins failed" in out
    assert "hindcast_error" not in out.split("3. Model comparison")[0]
    assert "the held-out evaluation window" in out


# ---------------------------------------------------------------------------
# Stage 1 pruning keeps the best configurations, not ~90% of them
# ---------------------------------------------------------------------------
def test_percentile_pruning_keeps_top_fraction_and_respects_top_k():
    from tests.test_climaid_model_leakage import _make_real_model, _tiny_search_kwargs
    dm = _make_real_model()
    dm.optimize_lags(**_tiny_search_kwargs())          # 16 lag combos x 1 x 1 x 1
    n_default = len(dm.lag_search_results)
    # Old behaviour kept every config <= 90th percentile (~15 of 16).
    assert n_default <= 3, n_default

    dm2 = _make_real_model()
    dm2.optimize_lags(**{**_tiny_search_kwargs(), "percentile": 0, "top_k": 4})
    assert len(dm2.lag_search_results) == 4           # keep-all percentile, capped by top_k
