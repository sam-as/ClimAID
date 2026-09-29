"""Regression tests for two correctness bugs found during v2 review.

1. weighted_interval_score used a normalising constant that depended on the
   chosen alpha levels instead of the canonical (K + 1/2), and weighted the
   median term incorrectly. WIS should equal 2x the mean pinball loss across
   the 2K+1 quantile levels, regardless of which probs are used.

2. SeasonalNaive.predict always looked up the seasonal lag using a *monthly*
   date offset, even when the model had been fit on weekly data with
   period=52 (the value ClimaidV2Forecaster derives automatically for
   weekly-cadence disease data). That sent the lookup ~4 years off instead
   of ~1 year and silently fell back to a coarse monthly-median baseline.
"""
import numpy as np
import pandas as pd

from climaid.forecasting_v2.metrics import weighted_interval_score
from climaid.forecasting_v2.baselines import SeasonalNaive


def _pinball(y, q, tau):
    return np.where(y >= q, tau * (y - q), (1 - tau) * (q - y))


def test_wis_matches_pinball_loss_identity():
    """WIS must equal 2x mean pinball loss over all quantile levels used,
    for *any* choice of probs -- not just a specific, symmetric grid."""
    rng = np.random.default_rng(1)
    for probs in (
        np.array([.025, .05, .10, .25, .50, .75, .90, .95, .975]),
        np.array([.1, .3, .5, .7, .9]),
        np.array([.2, .5, .8]),
    ):
        n = 40
        y = rng.normal(50, 15, n)
        q = np.sort(rng.normal(50, 15, (n, len(probs))), axis=1)
        got = weighted_interval_score(y, q, probs)
        ref = 2 * np.mean([
            np.mean(_pinball(y, q[:, j], p)) for j, p in enumerate(probs)
        ])
        assert np.isclose(got, ref, rtol=1e-8), (probs, got, ref)


def test_seasonal_naive_uses_weekly_lag_on_weekly_data():
    dates = pd.date_range("2015-01-05", periods=300, freq="W-MON")
    rng = np.random.default_rng(0)
    cases = 100 + 40 * np.sin(2 * np.pi * dates.dayofyear / 365) + rng.normal(0, 1, len(dates))
    df = pd.DataFrame({"time": dates, "cases": cases})

    model = SeasonalNaive("time", "cases", period=52).fit(df)
    assert model.lag_offset_unit_ == "weeks"

    future = pd.date_range(dates[-1] + pd.Timedelta(weeks=1), periods=8, freq="W-MON")
    pred = model.predict(future)

    # An exact-lag seasonal-naive forecast on clean weekly data should vary
    # week to week (tracking the underlying seasonal curve), not collapse to
    # one flat value per calendar month as the monthly-offset bug produced.
    assert pred["q500"].nunique() == len(pred)


def test_seasonal_naive_keeps_monthly_lag_on_monthly_data():
    dates = pd.date_range("2010-01-01", periods=180, freq="MS")
    rng = np.random.default_rng(0)
    cases = 50 + 10 * np.sin(2 * np.pi * dates.month / 12) + rng.normal(0, 1, len(dates))
    df = pd.DataFrame({"time": dates, "cases": cases})

    model = SeasonalNaive("time", "cases", period=12).fit(df)
    assert model.lag_offset_unit_ == "months"
