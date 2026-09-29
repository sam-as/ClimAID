"""Regression tests for two leakage bugs found in DiseaseModel._merge_data()
(climaid/climaid_model.py), the v1/legacy climate-disease feature builder.

1. YA_mean_* ("annual average") used to be computed as
   climate.groupby("Year")[vars].mean() and merged onto every row sharing
   that Year -- so a January row's annual-average feature included that
   same year's February-through-December climate, i.e. up to eleven months
   of *future* information relative to that row. Fixed to a strictly
   backward-looking (trailing) 12-month rolling average.

2. Both YA_mean_* and MA_mean_* (the pre-existing 10-year trailing average)
   are rolling-window features computed over `climate` in row order. Without
   sorting by time first, an unsorted or non-chronological input file would
   make the "trailing" window meaningless (and potentially leak
   future-dated rows into it). Fixed by sorting by time before any rolling
   computation.

Both features feed directly into the default feature grid used by
optimize_lags()/train_final_model(), so this is not a cosmetic issue.
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

from climaid.climaid_model import DiseaseModel


def _make_model(disease_df: pd.DataFrame, climate_df: pd.DataFrame) -> DiseaseModel:
    """Build a DiseaseModel instance without touching disk/network: bypass
    __init__ (which loads files and built-in datasets) and set only the
    attributes _merge_data() actually needs."""
    dm = object.__new__(DiseaseModel)
    dm.df_disease = disease_df
    dm.df_climate_hist = climate_df
    return dm


def _synthetic_climate(years, seed=0):
    """Monthly climate where each variable is a clean, deterministic,
    strictly-increasing function of the (sorted) row's chronological
    position. This makes it trivial to tell whether a "trailing" feature
    accidentally used a later (larger) value."""
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")
    idx = np.arange(len(dates), dtype=float)
    return pd.DataFrame({
        "time": dates,
        "mean_Rain": idx,
        "mean_temperature": idx,
        "mean_SH": idx,
        "Nino_anomaly": idx,
        "Dist_States": "IN_Test_State",
    })


def test_annual_average_does_not_use_future_months_within_the_same_year():
    climate = _synthetic_climate([2014, 2015, 2016, 2017])
    disease = pd.DataFrame({
        "time": pd.date_range("2015-01-01", "2016-12-01", freq="MS"),
    })
    disease["Year"] = disease["time"].dt.year
    disease["Month"] = disease["time"].dt.month
    disease["Count"] = 1

    dm = _make_model(disease, climate)
    merged = dm._merge_data()

    jan_2016 = merged[(merged.Year == 2016) & (merged.Month == 1)]
    assert not jan_2016.empty
    jan_value_row_index = climate.index[
        (climate.time.dt.year == 2016) & (climate.time.dt.month == 1)
    ][0]

    # The trailing 12-month annual average ending at January 2016 must only
    # use rows up to and including that January -- its value must be <= the
    # raw climate value *at* January 2016 (since the series is strictly
    # increasing and the average of the past includes values <= that point).
    # Before the fix, it would instead average across the whole 2016
    # calendar year, pulling in November/December 2016 values that are far
    # larger than anything available as of January.
    raw_jan_value = float(climate.loc[jan_value_row_index, "mean_temperature"])
    ya_value = float(jan_2016["YA_mean_temperature"].iloc[0])
    assert ya_value <= raw_jan_value + 1e-9, (
        "YA_mean_temperature for January used same-year future months"
    )


def test_annual_average_matches_manual_trailing_12_month_mean():
    climate = _synthetic_climate([2014, 2015, 2016])
    disease = pd.DataFrame({"time": pd.date_range("2015-01-01", "2016-12-01", freq="MS")})
    disease["Year"] = disease["time"].dt.year
    disease["Month"] = disease["time"].dt.month
    disease["Count"] = 1

    dm = _make_model(disease, climate)
    merged = dm._merge_data()

    target = pd.Timestamp("2016-06-01")
    row = merged[merged.time == target]
    assert not row.empty

    climate_sorted = climate.sort_values("time").reset_index(drop=True)
    pos = climate_sorted.index[climate_sorted.time == target][0]
    expected = climate_sorted["mean_temperature"].iloc[pos - 11: pos + 1].mean()
    assert row["YA_mean_temperature"].iloc[0] == pytest.approx(expected)


def test_merge_data_sorts_climate_before_rolling_even_if_input_is_shuffled():
    climate = _synthetic_climate([2014, 2015, 2016])
    shuffled = climate.sample(frac=1.0, random_state=7).reset_index(drop=True)
    disease = pd.DataFrame({"time": pd.date_range("2015-01-01", "2016-12-01", freq="MS")})
    disease["Year"] = disease["time"].dt.year
    disease["Month"] = disease["time"].dt.month
    disease["Count"] = 1

    dm_sorted = _make_model(disease, climate.copy())
    dm_shuffled = _make_model(disease, shuffled)

    merged_sorted = dm_sorted._merge_data().sort_values("time").reset_index(drop=True)
    merged_shuffled = dm_shuffled._merge_data().sort_values("time").reset_index(drop=True)

    pd.testing.assert_series_equal(
        merged_sorted["YA_mean_temperature"], merged_shuffled["YA_mean_temperature"]
    )
    pd.testing.assert_series_equal(
        merged_sorted["MA_mean_temperature"], merged_shuffled["MA_mean_temperature"]
    )


# ---------------------------------------------------------------------------
# Test-set-reuse leakage in optimize_lags()/train_final_model()
# ---------------------------------------------------------------------------
# Previously, optimize_lags() screened and ranked every candidate lag/
# feature/model configuration (potentially hundreds to thousands of them)
# directly against self.test_df, and train_final_model() then reported that
# same test set's RMSE/R2 as the "held-out" performance metric shown in the
# report. Reusing a test set to choose among many candidates optimistically
# biases the resulting score -- this is a textbook "test-set reuse"/
# "double-dipping" leak, distinct from the temporal (same-row) leaks above.
#
# Fixed by carving a chronological selection-validation split out of
# train_df alone (DiseaseModel._chronological_holdout /
# self.sel_train_df / self.sel_val_df), using it for all screening/ranking/
# accept-reject decisions, and reserving test_df for a single, final,
# honest evaluation of whichever configuration that process settles on.

def _make_real_model(n_years=18, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=12 * n_years, freq="MS")
    mo = dates.month.to_numpy()
    climate = pd.DataFrame({
        "time": dates,
        "mean_Rain": 5 + 2 * np.sin(2 * np.pi * (mo - 2) / 12) + rng.normal(0, .3, len(dates)),
        "mean_temperature": 27 + 2 * np.sin(2 * np.pi * mo / 12) + rng.normal(0, .2, len(dates)),
        "mean_SH": 70 + 5 * np.sin(2 * np.pi * mo / 12) + rng.normal(0, 1, len(dates)),
        "Nino_anomaly": rng.normal(0, .5, len(dates)),
        "Dist_States": "IN_Test_State",
    })
    disease = pd.DataFrame({"time": dates[12:]})
    disease["Year"] = disease.time.dt.year
    disease["Month"] = disease.time.dt.month
    disease["Count"] = np.maximum(
        0, np.round(20 + 8 * np.sin(2 * np.pi * disease.Month / 12) + rng.normal(0, 3, len(disease)))
    ).astype(int)

    dm = object.__new__(DiseaseModel)
    dm.random_state = 42
    dm.district = "IN_Test_State"
    dm.target_col = "Count"
    dm.disease_name = "TestDisease"
    dm.df_disease = disease
    dm.df_climate_hist = climate
    dm.df_merged = dm._merge_data()
    dm.train_df = None
    dm.test_df = None
    dm.lag_search_results = None
    dm.best_config = None
    dm.final_models = {}
    dm.runtime = {
        "lag_optimization_seconds": None, "training_seconds": None,
        "prediction_seconds": None, "report_generation_seconds": None,
        "total_pipeline_seconds": None,
    }
    dm._pipeline_start_time = None
    return dm


def _tiny_search_kwargs():
    return dict(
        base_models=("rf",), residual_models=("rf",), correction_models=("isotonic",),
        n_trials=2, n_jobs=1,
        sh_range=range(0, 2), temp_range=range(0, 2), rain_range=range(0, 2), elnino_range=range(0, 2),
    )


def test_optimize_lags_carves_a_selection_validation_split_disjoint_from_test():
    dm = _make_real_model()
    dm.optimize_lags(**_tiny_search_kwargs())

    assert hasattr(dm, "sel_train_df") and hasattr(dm, "sel_val_df")
    assert len(dm.sel_val_df) > 0
    # The selection split is carved from train_df alone, so it must never
    # overlap with test_df.
    sel_times = set(dm.sel_train_df["time"]).union(dm.sel_val_df["time"])
    test_times = set(dm.test_df["time"])
    assert sel_times.isdisjoint(test_times)
    # And it must be entirely contained within train_df's own dates.
    train_times = set(dm.train_df["time"])
    assert sel_times.issubset(train_times)


def test_lag_search_results_use_validation_metrics_not_test_metrics():
    dm = _make_real_model()
    dm.optimize_lags(**_tiny_search_kwargs())

    # Renamed to make the distinction explicit: these come from sel_val_df,
    # not test_df. The old "rmse"/"r2" names (which used to mean test-set
    # metrics) should no longer appear.
    assert "val_rmse" in dm.lag_search_results.columns
    assert "val_r2" in dm.lag_search_results.columns
    assert "rmse" not in dm.lag_search_results.columns
    assert "r2" not in dm.lag_search_results.columns
    assert dm.best_config["val_rmse"] == dm.lag_search_results["val_rmse"].min()


def test_config_selection_is_unaffected_by_corrupting_the_test_set():
    """If optimize_lags() genuinely never uses test_df for screening/ranking,
    corrupting test_df's target values after the train/test split (but
    before running optimize_lags) must not change which configuration gets
    selected, and must not prevent the search from completing cleanly."""
    dm_clean = _make_real_model()
    dm_clean._train_test_split(train_year=None, test_year=None)
    clean_test_df = dm_clean.test_df.copy()

    dm_corrupt = _make_real_model()
    dm_corrupt._train_test_split(train_year=None, test_year=None)
    # Wildly out-of-distribution / NaN-laced target values in the test set.
    dm_corrupt.test_df = dm_corrupt.test_df.copy()
    dm_corrupt.test_df["Count"] = np.nan

    kwargs = _tiny_search_kwargs()
    # Force both runs through the exact same split so optimize_lags() reuses
    # the train/test split already set above rather than recomputing it.
    dm_clean.train_year = dm_clean.train_df["Year"].max()
    dm_clean.test_year = clean_test_df["Year"].min()
    dm_corrupt.train_year = dm_corrupt.train_df["Year"].max()
    dm_corrupt.test_year = dm_corrupt.test_df["Year"].min()

    dm_clean.optimize_lags(**kwargs)
    dm_corrupt.optimize_lags(**kwargs)

    # A NaN-corrupted test set must not poison the search (it would, if any
    # screening/ranking step evaluated a fit against test_df).
    assert np.isfinite(dm_corrupt.best_config["val_rmse"])
    assert dm_corrupt.best_config["val_rmse"] == pytest.approx(
        dm_clean.best_config["val_rmse"]
    )


def test_train_final_model_trusts_search_decision_without_reconsulting_test_set():
    dm = _make_real_model()
    dm.optimize_lags(**_tiny_search_kwargs())
    decided_correction = dm.best_config["correction_model"]

    result = dm.train_final_model()

    # train_final_model() must apply the already-decided correction model
    # unconditionally (it no longer re-checks against test_df), so the
    # fitted correction stage matches what optimize_lags() selected.
    if decided_correction in (None, "none", "base_only"):
        assert dm.corr is None
    else:
        assert dm.corr is not None
    assert np.isfinite(result["test_rmse"])
    assert np.isfinite(dm.uncorrected_test_rmse)


# ---------------------------------------------------------------------------
# predict() silently skipping the fitted correction stage (hasattr typo)
# ---------------------------------------------------------------------------
# predict() checked hasattr(self, "cor") -- a typo for "corr", the actual
# attribute train_final_model() sets. hasattr(self, "cor") is always False,
# so the fitted correction/calibration model was silently never applied by
# predict(), even though the reported test_rmse/test_r2 reflected the
# *corrected* model. Fixing the typo also exposed a second, previously
# unreachable bug: predict() passed a 1D array to self.corr.predict(), but
# any non-isotonic correction model was fit on a (-1, 1)-reshaped array in
# train_final_model() and needs the same shape at predict time.
#
# Both scenarios (isotonic and non-isotonic correction) are tested by
# directly assigning dm.corr after a single, fast optimize_lags()/
# train_final_model() run, rather than depending on the stochastic search
# happening to select a correction model -- that keeps the test both fast
# and deterministic.

@pytest.fixture(scope="module")
def trained_dm():
    dm = _make_real_model()
    dm.optimize_lags(**_tiny_search_kwargs())
    dm.train_final_model()
    return dm


def test_predict_applies_an_isotonic_correction_model(trained_dm):
    dm = trained_dm
    from sklearn.isotonic import IsotonicRegression

    X = dm.test_df[dm.best_config["features"]]
    base_pred = dm.base.predict(X)
    resid_pred = dm.res.predict(X) if dm.res is not None else np.zeros_like(base_pred)
    combined = base_pred + resid_pred

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(combined, dm.test_df[dm.target_col].to_numpy() + 5)  # deliberate offset
    dm.corr = iso
    try:
        actual = dm.predict(dm.test_df)
        expected = iso.predict(combined)
        np.testing.assert_allclose(actual, expected)
        assert not np.allclose(actual, combined), "correction stage was not applied"
    finally:
        dm.corr = None


def test_predict_applies_a_non_isotonic_correction_model_with_correct_input_shape(trained_dm):
    dm = trained_dm
    from sklearn.linear_model import Ridge

    X = dm.test_df[dm.best_config["features"]]
    base_pred = dm.base.predict(X)
    resid_pred = dm.res.predict(X) if dm.res is not None else np.zeros_like(base_pred)
    combined = base_pred + resid_pred

    ridge = Ridge().fit(
        np.asarray(combined).reshape(-1, 1), dm.test_df[dm.target_col].to_numpy() + 5
    )
    dm.corr = ridge
    try:
        # Must not raise a shape-mismatch error (this branch was previously
        # unreachable because of the hasattr("cor") typo).
        actual = dm.predict(dm.test_df)
        expected = ridge.predict(np.asarray(combined).reshape(-1, 1))
        np.testing.assert_allclose(actual, expected)
    finally:
        dm.corr = None



