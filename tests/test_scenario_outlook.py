"""Hybrid near-term + CMIP6 scenario outlook (forecasting_v2/scenario.py)."""
import numpy as np
import pandas as pd
import pytest

from climaid.climaid_model import DiseaseModel
from climaid.forecasting_v2 import HybridScenarioProjector, bias_correct
from climaid.forecasting_v2.scenario import _canon_projection, monthly_climatology, near_best_structures, select_lags_cv
from climaid.forecasting_v2.schema import canonicalize_climate
from climaid.reporting_scenario import generate_scenario_report


def _world(seed=0):
    """Temperature-driven dengue-like history plus 2 GCMs x 2 SSPs; SSP5-8.5 warms twice as fast."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2004-01-01", "2020-12-01", freq="MS"); m = t.month.to_numpy()
    temp = 27 + 3 * np.sin(2 * np.pi * (m - 4) / 12) + rng.normal(0, .4, len(t))
    rain = np.maximum(0, 2 + 8 * np.exp(-((m - 8) ** 2) / 3) + rng.normal(0, 1, len(t)))
    sh = 14 + 4 * np.sin(2 * np.pi * (m - 6) / 12) + rng.normal(0, .5, len(t))
    enso = np.cumsum(rng.normal(0, .2, len(t))) * .3
    clim = pd.DataFrame({"time": t, "mean_temperature": temp, "mean_Rain": rain, "mean_SH": sh, "Nino_anomaly": enso})
    zt = (temp - temp.mean()) / temp.std()
    mu = np.exp(3.0 + 0.5 * np.r_[np.nan, zt[:-1]])
    cases = rng.poisson(np.nan_to_num(mu, nan=20.0))
    dis = pd.DataFrame({"time": t, "cases": cases})[t >= "2005-01-01"]
    ft = pd.date_range("2015-01-01", "2050-12-01", freq="MS"); fm = ft.month.to_numpy(); yrs = (ft.year - 2015).to_numpy()
    rows = []
    for g, off in (("G1", 1.5), ("G2", -1.0)):
        for s, rate in (("ssp245", .02), ("ssp585", .04)):
            rows.append(pd.DataFrame({"time": ft, "model": g, "ssp": s,
                "mean_temperature": 27 + off + 3 * np.sin(2 * np.pi * (fm - 4) / 12) + rate * yrs + rng.normal(0, .4, len(ft)),
                "mean_Rain": np.maximum(0, 2 + 8 * np.exp(-((fm - 8) ** 2) / 3) + rng.normal(0, 1, len(ft))),
                "mean_SH": 14 + 4 * np.sin(2 * np.pi * (fm - 6) / 12) + rng.normal(0, .5, len(ft)),
                "Nino_anomaly": rng.normal(0, .3, len(ft))}))
    return dis, clim, pd.concat(rows, ignore_index=True)


def test_projection_series_are_not_averaged_together():
    _, _, proj = _world()
    c = _canon_projection(proj)
    assert set(zip(c.model, c.ssp)) == {("G1", "ssp245"), ("G1", "ssp585"), ("G2", "ssp245"), ("G2", "ssp585")}
    assert len(c) == len(proj)


def test_bias_correction_removes_gcm_offset_in_baseline():
    _, clim, proj = _world()
    obs = canonicalize_climate(clim); obs["time"] = obs.time.dt.to_period("M").dt.to_timestamp()
    oc = monthly_climatology(obs, range(2005, 2021))
    corr, notes = bias_correct(_canon_projection(proj), oc, range(2005, 2021))
    for g in ("G1", "G2"):
        b = corr[(corr.model == g) & (corr.ssp == "ssp245") & corr.time.dt.year.between(2015, 2020)]
        gm = b.groupby(b.time.dt.month).temperature.mean()
        assert np.allclose(gm.values, oc.temperature.values, atol=1e-6)
    assert notes == []


def test_blend_weights():
    w = HybridScenarioProjector(near_term_months=12, blend_months=12)._weights(40)
    assert (w[:12] == 1).all() and (np.diff(w[11:25]) < 0).all() and (w[24:] == 0).all()
    assert (HybridScenarioProjector(near_term_months=0, blend_months=0)._weights(10) == 0).all()


def test_structure_ensemble_includes_best_and_drops_nothing_silently():
    dis, clim, _ = _world()
    obs = canonicalize_climate(clim); obs["time"] = obs.time.dt.to_period("M").dt.to_timestamp()
    oc = monthly_climatology(obs, range(2005, 2021))
    best, table = select_lags_cv(dis, obs, oc)
    structs = near_best_structures(table, ["temperature", "rainfall", "humidity", "enso"])
    assert structs[0] == best and 1 <= len(structs) <= 8


@pytest.fixture(scope="module")
def outlook():
    dis, clim, proj = _world()
    # tuning set explicitly: module-scoped fixtures run before the conftest's fast-tuning patch
    return HybridScenarioProjector(n_bootstrap=20, samples_per_series=200, near_term_simulations=200,
                                   near_term_models=("seasonal_naive", "poisson"), tuning="fast").project(
        dis, clim, proj, origin="2020-12-31", end_year=2050)


def test_warmer_scenario_projects_more_cases_and_uncertainty_grows(outlook):
    d = outlook.decades.set_index(["ssp", "decade"])
    assert d.loc[("ssp585", "2040s"), "change_vs_baseline_median"] > d.loc[("ssp245", "2040s"), "change_vs_baseline_median"]
    assert d.loc[("ssp585", "2040s"), "change_vs_baseline_median"] > 0
    w20 = d.loc[("ssp585", "2020s"), "change_p90"] - d.loc[("ssp585", "2020s"), "change_p10"]
    w40 = d.loc[("ssp585", "2040s"), "change_p90"] - d.loc[("ssp585", "2040s"), "change_p10"]
    assert w40 > w20


def test_outlook_structure_and_hand_over(outlook):
    s = outlook.series
    one = s[(s.model == "G1") & (s.ssp == "ssp245")].sort_values("time")
    assert one.time.min() == pd.Timestamp("2021-01-01") and one.time.max() == pd.Timestamp("2050-12-01")
    assert one.weight_near_term.iloc[0] == 1 and one.weight_near_term.iloc[30] == 0
    assert {"annual_median", "change_p10", "change_p90", "share_gcm_samples_increase"} <= set(outlook.decades.columns)
    assert not outlook.annual.empty and outlook.metadata["gcms"] == ["G1", "G2"]


def test_report_and_disease_model_integration(tmp_path):
    dis, clim, proj = _world()
    dm = object.__new__(DiseaseModel)
    dm.target_col = "Count"; dm.random_state = 1; dm.district = "X"; dm.disease_name = "Test"
    dm.df_disease = dis.rename(columns={"cases": "Count"}); dm.df_climate_hist = clim; dm.df_climate_proj = proj
    res = dm.project_v2(forecast_origin="2020-12-31", end_year=2045, n_bootstrap=20, output_dir=str(tmp_path),
                        near_term_models=("seasonal_naive", "poisson"))
    html = open(res["report_path"]).read()
    for needle in ("C-DSI v2 — Scenario interpretation", "Decade summary", "Sensitivity to how climate effects",
                   "Disrupted period excluded (2020)", "Projections from January 2021", "The short version"):
        assert needle in html, needle
    assert res["sensitivity"] is not None
    assert res["sensitivity"].metadata["response_mode"] == "anomaly"


def test_project_v2_requires_projection_data():
    dis, clim, _ = _world()
    dm = object.__new__(DiseaseModel)
    dm.target_col = "Count"; dm.df_disease = dis.rename(columns={"cases": "Count"}); dm.df_climate_hist = clim
    dm.df_climate_proj = None
    with pytest.raises(ValueError, match="No CMIP6 projection data"):
        dm.project_v2(forecast_origin="2020-12-31")
