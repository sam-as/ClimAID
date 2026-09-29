"""Tests for the C-DSI v2 deterministic interpreter added to reporting_v2.py.

C-DSI v2 mirrors the original (v1) C-DSI's design principle -- a
template-based narrative built only from precomputed numeric artifacts, with
no LLM and no invented content -- but interprets v2's own probabilistic
forecast, held-out evaluation and rolling-hindcast outputs rather than the
v1 point-forecast/CMIP6 pipeline.
"""
import numpy as np
import pandas as pd

from climaid.forecasting_v2 import ClimaidV2Forecaster, HindcastEvaluator
from climaid.reporting_v2 import generate_c_dsi_v2, generate_forecast_report


def _synthetic():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2010-01-01", "2022-12-01", freq="MS")
    mo = dates.month.to_numpy()
    cases = np.maximum(
        0, np.round(30 + 10 * np.sin(2 * np.pi * mo / 12) + rng.normal(0, 4, len(dates)))
    )
    disease = pd.DataFrame({"time": dates, "cases": cases})
    climate = pd.DataFrame({
        "time": dates,
        "temperature": 27 + 2 * np.sin(2 * np.pi * mo / 12),
        "rainfall": 5 + 2 * np.sin(2 * np.pi * (mo - 2) / 12),
        "humidity": 70 + 5 * np.sin(2 * np.pi * mo / 12),
        "enso": rng.normal(0, .3, len(dates)),
    })
    return disease, climate


def test_c_dsi_v2_report_mode_and_sections_present():
    disease, climate = _synthetic()
    engine = ClimaidV2Forecaster(models=("seasonal_naive", "renewal", "random_forest")).fit(
        disease, climate, cutoff="2020-12-31"
    )
    future = climate[climate.time > "2020-12-31"].head(12)
    bundle = engine.predict(future, 12, n_simulations=300)
    observed = disease[disease.time > "2020-12-31"].head(12)
    metrics = engine.evaluate(observed, bundle)

    out = generate_c_dsi_v2(
        disease_name="Dengue", district="IN_Test_State", metadata=bundle.metadata,
        forecasts=bundle.forecasts, metrics=metrics, hindcast_metrics=None,
    )
    assert "ClimAID v2 Deterministic Scientific Interpreter (C-DSI v2)" in out
    assert "1. Forecast summary" in out
    assert "2. Interval calibration" in out
    assert "3. Model comparison" in out
    assert "4. Structural caveats" in out
    assert "5. Methodological warnings" in out
    # No stray unmatched parentheses in the trend sentence.
    assert out.count("(") == out.count(")")


def test_c_dsi_v2_handles_no_metrics_gracefully():
    disease, climate = _synthetic()
    engine = ClimaidV2Forecaster(models=("seasonal_naive",)).fit(disease, climate, cutoff="2020-12-31")
    future = climate[climate.time > "2020-12-31"].head(12)
    bundle = engine.predict(future, 12, n_simulations=100)

    out = generate_c_dsi_v2(
        disease_name="Dengue", district="IN_Test_State", metadata=bundle.metadata,
        forecasts=bundle.forecasts, metrics=None, hindcast_metrics=None,
    )
    assert "not assessable" in out or "No held-out or hindcast observations" in out
    assert "No rolling hindcast scores were available" in out


def test_c_dsi_v2_picks_best_hindcast_model_deterministically():
    disease, climate = _synthetic()
    hc = HindcastEvaluator(horizon=12, n_origins=2, min_train_size=60)
    _, hindcast_metrics = hc.evaluate(
        disease, climate, models=("seasonal_naive", "random_forest"), n_simulations=100
    )
    engine = ClimaidV2Forecaster(models=("seasonal_naive", "random_forest")).fit(
        disease, climate, cutoff="2020-12-31"
    )
    future = climate[climate.time > "2020-12-31"].head(12)
    bundle = engine.predict(future, 12, n_simulations=100)

    out = generate_c_dsi_v2(
        disease_name="Dengue", district="IN_Test_State", metadata=bundle.metadata,
        forecasts=bundle.forecasts, metrics=None, hindcast_metrics=hindcast_metrics,
    )
    best_model = hindcast_metrics.groupby("model")["WIS"].mean().idxmin()
    assert best_model in out
    assert "lowest mean WIS" in out


def test_c_dsi_v2_is_embedded_in_full_report_and_labelled():
    disease, climate = _synthetic()
    engine = ClimaidV2Forecaster(models=("seasonal_naive", "renewal")).fit(
        disease, climate, cutoff="2020-12-31"
    )
    future = climate[climate.time > "2020-12-31"].head(12)
    bundle = engine.predict(future, 12, n_simulations=200)

    report = generate_forecast_report(
        disease_name="Dengue", district="IN_Test_State", bundle=bundle,
        legacy_report_text="# Legacy C-DSI\n\nSome v1 narrative.",
    )
    assert "C-DSI v2 — Deterministic Forecast Interpretation" in report
    assert "cdsi-v2" in report
    # v1 legacy content is still present but clearly marked as an appendix,
    # not conflated with the v2 interpretation above it.
    assert "Appendix — ClimAID v1 (legacy) C-DSI report" in report
    assert report.index("C-DSI v2") < report.index("Appendix — ClimAID v1")
