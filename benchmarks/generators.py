"""Synthetic datasets used to test ClimAID (all reproducible from fixed seeds).

Every dataset has a known "truth", so model output can be checked against it.
The clean datasets are built the way the models assume the world works, so
results on them are a best case. `realistic_hard()` adds problems real
surveillance data has, and is the more informative test.

Functions
---------
climate()                 shared synthetic monthly climate, 2004-2022
seasonal(), nonseasonal() main benchmark disease series (2005-2022)
enso_interaction()        seasonal series + El Nino x temperature interaction
realistic_hard()          seasonal series + reporting change, outbreaks, gaps,
                          threshold rainfall effect and data-entry errors
cmip6(obs_climate)        3 imaginary climate models x 2 SSPs, 2015-2060
multi_district(seed)      6 districts sharing one climate response (+ target projections)
thermal_optimum(seed)     hot district whose transmission follows a thermal-suitability curve
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATES = pd.date_range("2004-01-01", "2022-12-01", freq="MS")


def _lag(x, k):
    return np.r_[np.full(k, np.nan), x[:-k]] if k else x


def _z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def _ar1(n, phi, sd, rng):
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, sd)
    return e


def _negbin(mu, rng, k=12):
    """Counts scattered more than Poisson (variance mu + mu^2/k)."""
    return rng.poisson(rng.gamma(k, mu / k))


def climate(seed=2026):
    rng = np.random.default_rng(seed)
    m = DATES.month.to_numpy(); n = len(DATES)
    yrs = (DATES.year - 2004).to_numpy() + (m - 1) / 12
    temp = 27 + 3 * np.sin(2 * np.pi * (m - 4) / 12) + 0.02 * yrs + rng.normal(0, 0.4, n)
    rain = np.maximum(0, 1.5 + 9 * np.exp(-((m - 7.5) ** 2) / 3) + rng.gamma(2, 0.6, n) - 1.2)
    sh = 14 + 4 * np.sin(2 * np.pi * (m - 5) / 12) + rng.normal(0, 0.5, n)
    enso = _ar1(n, 0.92, 0.3, rng)
    return pd.DataFrame({"time": DATES, "mean_temperature": temp.round(3), "mean_Rain": rain.round(3),
                         "mean_SH": sh.round(3), "Nino_anomaly": enso.round(3)})


def _series(eta, rng, start="2005-01-01"):
    keep = DATES >= start
    return pd.DataFrame({"time": DATES[keep], "cases": _negbin(np.exp(np.nan_to_num(eta[keep], nan=3.2)), rng)})


def seasonal(seed=2026):
    """log mu = 3.2 + 0.30 z(temp, lag 1) + 0.40 z(rain, lag 2) + 0.20 ENSO(lag 3) + AR(1) noise."""
    c = climate(seed); rng = np.random.default_rng(seed + 1)
    T, R, E = c.mean_temperature.to_numpy(), c.mean_Rain.to_numpy(), c.Nino_anomaly.to_numpy()
    eta = 3.2 + 0.30 * _lag(_z(T), 1) + 0.40 * _lag(_z(R), 2) + 0.20 * _lag(E, 3) + _ar1(len(c), 0.5, 0.12, rng)
    return _series(eta, rng), c


def nonseasonal(seed=2026):
    """log mu = 3.2 + 0.50 ENSO(lag 4) + slow random-walk drift + AR(1) noise (no annual cycle)."""
    c = climate(seed); rng = np.random.default_rng(seed + 2)
    E = c.Nino_anomaly.to_numpy()
    drift = np.cumsum(rng.normal(0, 0.03, len(c)))
    eta = 3.2 + 0.50 * _lag(E, 4) + drift + _ar1(len(c), 0.5, 0.12, rng)
    return _series(eta, rng), c


def enso_interaction(seed=2026, strength=0.60):
    """Seasonal truth plus strength x z(temp, lag 1) x mean ENSO over lags 1-3."""
    c = climate(seed); rng = np.random.default_rng(seed + 3)
    T, R, E = c.mean_temperature.to_numpy(), c.mean_Rain.to_numpy(), c.Nino_anomaly.to_numpy()
    e3 = (_lag(E, 1) + _lag(E, 2) + _lag(E, 3)) / 3
    eta = (3.2 + 0.30 * _lag(_z(T), 1) + 0.40 * _lag(_z(R), 2) + 0.20 * _lag(E, 3)
           + strength * _lag(_z(T), 1) * e3 + _ar1(len(c), 0.5, 0.12, rng))
    return _series(eta, rng), c


def realistic_hard(seed=2026):
    """A seasonal series with problems real surveillance data has.

    - Threshold rainfall effect: rain only matters above 6 mm (e.g. breeding sites form).
    - Temperature effect saturates above 29 C.
    - Reporting change: from 2016 a surveillance expansion records 60% more of the same cases.
    - Two serotype-driven outbreak years (2012, 2019): Aug-Nov cases up to 3x, unrelated to climate.
    - 2020 COVID-like disruption: reporting falls to 40%.
    - Missing data: ~5% of months missing at random, plus a 3-month gap (Jun-Aug 2014).
    - Data-entry errors: two months recorded 10x too high.
    Returns (disease, climate, truth) where truth holds the expected cases before reporting effects.
    """
    c = climate(seed); rng = np.random.default_rng(seed + 4)
    T, R, E = c.mean_temperature.to_numpy(), c.mean_Rain.to_numpy(), c.Nino_anomaly.to_numpy()
    Tl = np.minimum(_lag(T, 1), 29.0)                      # saturating temperature
    Rl = np.maximum(_lag(R, 2) - 6.0, 0.0)                 # threshold rainfall
    eta = 2.9 + 0.18 * (Tl - 27) + 0.25 * Rl + 0.20 * _lag(E, 3) + _ar1(len(c), 0.5, 0.12, rng)
    mu = np.exp(np.nan_to_num(eta, nan=2.9))
    yr, mo = DATES.year.to_numpy(), DATES.month.to_numpy()
    outbreak = np.ones(len(c))
    for y in (2012, 2019):
        sel = (yr == y) & (mo >= 8) & (mo <= 11)
        outbreak[sel] = np.array([1.8, 3.0, 2.6, 1.6])[: sel.sum()]
    reporting = np.where(yr >= 2016, 1.6, 1.0) * np.where(yr == 2020, 0.4, 1.0)
    true_mu = mu * outbreak
    keep = DATES >= "2005-01-01"
    cases = _negbin(true_mu * reporting, rng)[keep].astype(float)
    t = DATES[keep]
    miss = rng.random(len(t)) < 0.05
    miss |= (t >= "2014-06-01") & (t <= "2014-08-01")
    errs = rng.choice(np.flatnonzero(~miss), 2, replace=False)
    cases[errs] *= 10
    d = pd.DataFrame({"time": t, "cases": cases})[~miss].reset_index(drop=True)
    truth = pd.DataFrame({"time": t, "expected_cases": true_mu[keep], "reporting_factor": reporting[keep]})
    return d, c, truth


def cmip6(obs_climate=None, seed=7):
    """3 imaginary climate models (own offsets and warming sensitivities) x SSP2-4.5 / SSP5-8.5, 2015-2060."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2060-12-01", freq="MS"); m = dates.month.to_numpy()
    yrs = (dates.year - 2015).to_numpy() + (m - 1) / 12
    rows = []
    for gcm, tb, rb, sens in [("GCM-A", 1.5, 1.3, 1.0), ("GCM-B", -0.8, 0.8, 0.7), ("GCM-C", 0.4, 1.1, 1.4)]:
        for ssp, rate in [("ssp245", 0.025), ("ssp585", 0.050)]:
            temp = 27 + tb + 3 * np.sin(2 * np.pi * (m - 4) / 12) + sens * rate * yrs + rng.normal(0, .5, len(dates))
            rain = rb * np.maximum(0, 1.5 + 9 * np.exp(-((m - 7.5) ** 2) / 3) + rng.gamma(2, .6, len(dates)) - 1.2) \
                * (1 + 0.004 * sens * yrs * (rate / 0.025))
            sh = 14 + 4 * np.sin(2 * np.pi * (m - 5) / 12) + 0.3 * sens * rate * yrs + rng.normal(0, .5, len(dates))
            rows.append(pd.DataFrame({"Dist_States": "Synthetic_District", "time": dates, "model": gcm, "ssp": ssp,
                                      "mean_SH": sh, "mean_Rain": rain, "mean_temperature": temp,
                                      "Nino_anomaly": _ar1(len(dates), .9, .3, rng)}))
    return pd.concat(rows, ignore_index=True)


def multi_district(seed=1):
    """Six districts sharing one climate response (+12%/C at lag 1, +10%/mm rain at lag 2, ENSO lag 3),
    with different seasonal climates. Returns (target, others, target_projection, true_2050s_change)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2004-01-01", "2020-12-01", freq="MS"); m = t.month.to_numpy()

    def district(t_peak, r_peak, t_mean, amp, a):
        T = t_mean + amp * np.cos(2 * np.pi * (m - t_peak) / 12) + rng.normal(0, .4, len(t))
        R = np.maximum(0, 1.5 + 9 * np.exp(-(((m - r_peak + 6) % 12 - 6) ** 2) / 3) + rng.gamma(2, .6, len(t)) - 1.2)
        H = 14 + 0.8 * (T - t_mean) + rng.normal(0, .5, len(t))
        E = _ar1(len(t), .92, .3, rng)
        eta = a + 0.12 * (_lag(T, 1) - 27) + 0.10 * _lag(R, 2) + 0.2 * _lag(E, 3)
        y = _negbin(np.exp(np.nan_to_num(eta, nan=a)), rng)
        c = pd.DataFrame({"time": t, "mean_temperature": T, "mean_Rain": R, "mean_SH": H, "Nino_anomaly": E})
        return pd.DataFrame({"time": t, "cases": y})[t >= "2005-01-01"], c
    target = district(7, 7.5, 27, 3, 3.0)
    others = [district(*p) for p in [(5, 8, 25, 4, 2.6), (6, 1, 28, 2, 2.8), (4, 10, 26, 5, 2.4),
                                     (8, 3, 29, 2.5, 3.2), (3, 7, 24, 6, 2.5)]]
    _, c0 = target
    ft = pd.date_range("2015-01-01", "2060-12-01", freq="MS"); fm = ft.month.to_numpy()
    yrs = (ft.year - 2015).to_numpy() + (fm - 1) / 12
    cm = c0.groupby(c0.time.dt.month)[["mean_temperature", "mean_Rain", "mean_SH"]].mean()
    proj = pd.concat([pd.DataFrame({"time": ft, "model": g, "ssp": "ssp585",
        "mean_temperature": cm.mean_temperature.loc[fm].to_numpy() + 0.05 * yrs + rng.normal(0, .4, len(ft)),
        "mean_Rain": np.maximum(0, cm.mean_Rain.loc[fm].to_numpy() + rng.normal(0, 1, len(ft))),
        "mean_SH": cm.mean_SH.loc[fm].to_numpy() + 0.04 * yrs + rng.normal(0, .5, len(ft)),
        "Nino_anomaly": rng.normal(0, .3, len(ft))}) for g in ("G1", "G2")])
    return target, others, proj, float(np.exp(0.12 * 0.05 * (2055 - 2015)) - 1)


def thermal_optimum(seed=1, curve=(17.8, 29.1, 34.6)):
    """Hot district (about 26-32 C) where transmission follows a thermal-suitability curve;
    warming pushes the hottest months past the optimum. Returns (disease, climate, projection, true_2050s_change)."""
    from climaid.forecasting_v2.scenario import thermal_suitability as S
    rng = np.random.default_rng(seed)
    t = pd.date_range("2004-01-01", "2020-12-01", freq="MS"); m = t.month.to_numpy()
    T = 29 + 3 * np.cos(2 * np.pi * (m - 5) / 12) + rng.normal(0, .5, len(t))
    R = np.maximum(0, 1.5 + 9 * np.exp(-(((m - 8 + 6) % 12 - 6) ** 2) / 3) + rng.gamma(2, .6, len(t)) - 1.2)
    H = 14 + .8 * (T - 29) + rng.normal(0, .5, len(t)); E = rng.normal(0, .3, len(t))
    eta = 1.2 + 2.0 * S(np.r_[T[0], T[:-1]], curve) + 0.08 * np.r_[R[:2], R[:-2]]
    y = _negbin(np.exp(eta), rng)
    clim = pd.DataFrame({"time": t, "mean_temperature": T, "mean_Rain": R, "mean_SH": H, "Nino_anomaly": E})
    ft = pd.date_range("2015-01-01", "2060-12-01", freq="MS"); fm = ft.month.to_numpy(); yrs = (ft.year - 2015) + (fm - 1) / 12
    cm = clim.groupby(clim.time.dt.month)[["mean_temperature", "mean_Rain", "mean_SH"]].mean()
    proj = pd.concat([pd.DataFrame({"time": ft, "model": g, "ssp": "ssp585",
        "mean_temperature": cm.mean_temperature.loc[fm].to_numpy() + 0.06 * yrs + rng.normal(0, .5, len(ft)),
        "mean_Rain": np.maximum(0, cm.mean_Rain.loc[fm].to_numpy() + rng.normal(0, 1, len(ft))),
        "mean_SH": cm.mean_SH.loc[fm].to_numpy() + .8 * .06 * yrs, "Nino_anomaly": rng.normal(0, .3, len(ft))}) for g in ("G1", "G2")])
    f = lambda Tm, Rm: np.exp(1.2 + 2.0 * S(Tm, curve) + 0.08 * Rm)
    base = sum(f(cm.mean_temperature.loc[k], cm.mean_Rain.loc[k]) for k in range(1, 13))
    fut = sum(f(cm.mean_temperature.loc[k] + 0.06 * 40, cm.mean_Rain.loc[k]) for k in range(1, 13))
    return pd.DataFrame({"time": t, "cases": y})[t >= "2005-01-01"], clim, proj, float(fut / base - 1)
