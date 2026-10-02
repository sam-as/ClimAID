"""Hybrid ClimAID v2 outlook: near-term probabilistic forecast blended into
CMIP6 scenario projections.

Why hybrid
----------
The v2 forecaster uses recent case counts (autoregressive lags, renewal
dynamics). That information is valuable for the next few months but fades
within about a year, and recursive forecasts become unreliable and too
narrow over multi-year horizons. Long-run questions ("what might dengue look
like in the 2040s under SSP5-8.5?") need a model driven by climate alone.

This module therefore produces one continuous outlook per climate model
(GCM) and SSP:

1. Near term (months 1..near_term_months): the v2 ensemble forecast, driven
   by that GCM's bias-corrected climate.
2. Long term: a climate-response count model (month-of-year seasonality +
   lagged climate anomalies relative to observed climatology), refitted on
   block-bootstrap resamples of whole years so parameter uncertainty is
   carried into the projections.
3. Hand-over (the next blend_months): a linear mixture of the two predictive
   distributions, sample by sample.

Samples are then pooled across GCMs within each SSP, so the reported ranges
include (a) count noise, (b) statistical-model parameter uncertainty and (c)
the spread between climate models. Uncertainty therefore grows with the
horizon as the GCMs diverge.

Bias correction
---------------
Each GCM series is shifted so its baseline-period monthly means match the
observed monthly climatology (additive for temperature, humidity and ENSO;
multiplicative for rainfall). This keeps each GCM's projected *change*
while removing its systematic offset from the observed record.

Key assumptions (stated in the report)
--------------------------------------
- The climate-disease relationship estimated from history stays the same.
- Population, immunity, vector control, diagnostics and reporting do not
  change. Projections isolate the climate signal; they are not predictions
  of future case counts.
- CMIP6 models represent ENSO variability poorly; ENSO terms mainly add noise
  far ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from .schema import canonicalize_climate, canonicalize_disease
from .renewal import DEFAULT_PROBS

CLIM_VARS = ("temperature", "rainfall", "humidity", "enso")
MULTIPLICATIVE = {"rainfall"}
QCOLS = [f"q{int(round(p * 1000)):03d}" for p in DEFAULT_PROBS]


# Thermal-suitability presets: (T_min, T_opt, T_max) in degrees C.
# aedes_aegypti_mordecai2017: thermal limits and optimum of the mechanistic
# R0(T) for dengue transmission by Aedes aegypti, as reported by Mordecai et
# al. (2017, PLoS Neglected Tropical Diseases, doi:10.1371/journal.pntd.0005568).
TEMPERATURE_CURVES = {"aedes_aegypti_mordecai2017": (17.8, 29.1, 34.6)}


def thermal_suitability(temp, curve):
    """Unimodal suitability in [0, 1]: 0 at/below T_min and at/above T_max,
    1 at T_opt, asymmetric parabola on each side.

    This constrains the *shape* of the temperature response (an optimum and a
    decline beyond it); the model still estimates how strongly suitability
    affects cases. It is not a re-implementation of any published R0(T)
    model, and monthly mean temperatures smooth out daily extremes.
    """
    tmin, topt, tmax = (TEMPERATURE_CURVES[curve] if isinstance(curve, str) else curve)
    t = np.asarray(temp, dtype=float)
    below = 1.0 - ((t - topt) / (topt - tmin)) ** 2
    above = 1.0 - ((t - topt) / (tmax - topt)) ** 2
    s_ = np.where(t <= topt, below, above)
    s_ = np.where((t <= tmin) | (t >= tmax), 0.0, s_)
    return np.clip(s_, 0.0, 1.0)


def _transform(v, x, curve=None):
    """Variable transform used before standardising (seasonal mode)."""
    if v == "temperature" and curve is not None:
        return np.log(thermal_suitability(x, curve) + 0.05)
    return np.log1p(x) if v in MULTIPLICATIVE else x


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _canon_projection(proj: pd.DataFrame, model_col="model", ssp_col="ssp") -> pd.DataFrame:
    """Canonicalise a long CMIP6 table *per (model, ssp)* series.

    canonicalize_climate() averages rows that share a timestamp, which would
    silently blend different GCMs/SSPs together, so each series is
    canonicalised separately.
    """
    for c in (model_col, ssp_col):
        if c not in proj.columns:
            raise ValueError(f"Projection data must contain a '{c}' column (found {list(proj.columns)})")
    parts = []
    for (m, s), g in proj.groupby([model_col, ssp_col], sort=True):
        c = canonicalize_climate(g.drop(columns=[model_col, ssp_col]), require_all=True)
        c["time"] = c["time"].dt.to_period("M").dt.to_timestamp()
        parts.append(c.assign(model=str(m), ssp=str(s).lower()))
    if not parts:
        raise ValueError("Projection data is empty")
    return pd.concat(parts, ignore_index=True)


def monthly_climatology(climate: pd.DataFrame, years) -> pd.DataFrame:
    c = climate[climate["time"].dt.year.isin(list(years))]
    if c.empty:
        raise ValueError("No climate observations in the requested baseline years")
    return c.groupby(c["time"].dt.month)[[v for v in CLIM_VARS if v in c.columns]].mean()


def bias_correct(proj: pd.DataFrame, obs_clim: pd.DataFrame, baseline_years, min_years=5):
    """Monthly mean-shift (delta) bias correction, per GCM x SSP series.

    Returns (corrected_frame, notes). If a series has fewer than `min_years`
    of overlap with the baseline, its first 10 years are used as its
    reference instead and a note is returned: its projected change is then
    measured relative to those years rather than the observed baseline, so
    long-run change is understated.
    """
    out, notes = [], []
    base = set(baseline_years)
    for (m, s), g in proj.groupby(["model", "ssp"], sort=True):
        g = g.sort_values("time").copy()
        yrs = sorted(set(g["time"].dt.year) & base)
        if len(yrs) < min_years:
            first = sorted(g["time"].dt.year.unique())[:10]
            ref_years = first
            notes.append(f"{m}/{s}: only {len(yrs)} baseline year(s) overlap the observations; "
                         f"bias-corrected against its own {first[0]}-{first[-1]} instead, "
                         "so its projected change is understated.")
        else:
            ref_years = yrs
        ref = g[g["time"].dt.year.isin(ref_years)]
        gcm_clim = ref.groupby(ref["time"].dt.month)[[v for v in CLIM_VARS if v in g.columns]].mean()
        mon = g["time"].dt.month
        for v in CLIM_VARS:
            if v not in g.columns or v not in obs_clim.columns:
                continue
            o = mon.map(obs_clim[v]); r = mon.map(gcm_clim[v])
            if v in MULTIPLICATIVE:
                g[v] = np.where(r > 1e-6, g[v] * (o / r.where(r > 1e-6, 1.0)), o)
                g[v] = np.maximum(g[v], 0.0)
            else:
                g[v] = g[v] + (o - r)
        out.append(g)
    return pd.concat(out, ignore_index=True), notes


def _anomaly_features(clim: pd.DataFrame, obs_clim: pd.DataFrame, scale: dict, lags=(0, 1, 2),
                      response="anomaly", centre=None, quadratic_temperature=True, lag_map=None,
                      temperature_curve=None, distributed=None):
    """Design matrix for the long-term climate-response model.

    response="anomaly": month-of-year dummies + one window-mean anomaly per
        variable. Seasonality is absorbed by the dummies, so climate effects
        are learned only from year-to-year deviations within a month.
        Conservative, but weakly identified from short records.
    response="seasonal": no month dummies; standardised climate values at
        each lag (0..3) plus a squared temperature term (thermal optimum).
        The seasonal climate cycle identifies the response, which is much
        better determined, but assumes seasonality is climate-driven.
    """
    c = clim.sort_values("time").reset_index(drop=True)
    mon = c["time"].dt.month
    X = pd.DataFrame({"time": c["time"]})
    if response == "seasonal":
        centre = centre or {}
        for v in CLIM_VARS:
            if v not in c.columns:
                continue
            x = pd.Series(_transform(v, c[v].to_numpy(float), temperature_curve), index=c.index)
            z = (x - centre.get(v, (0.0, 1.0))[0]) / centre.get(v, (0.0, 1.0))[1]
            if distributed:
                from .distributed_lag import cross_basis
                L = int(distributed["max_lag"].get(v, 6))
                cb = cross_basis(z.to_numpy(float), L, distributed.get("degree", 2))
                for k in range(cb.shape[1]):
                    X[f"{v}_dl{k}"] = cb[:, k]
                if v == "temperature" and quadratic_temperature and temperature_curve is None:
                    X["temperature_lvl_sq"] = ((z + z.shift(1) + z.shift(2)) / 3.0) ** 2
                continue
            use = list((lag_map or {}).get(v, range(0, 4)))
            for L in use:
                X[f"{v}_lvl_l{L}"] = z.shift(L)
            if v == "temperature" and quadratic_temperature and use and temperature_curve is None:
                X["temperature_lvl_sq"] = (sum(z.shift(L) for L in use) / len(use)) ** 2
        return X
    for m in range(2, 13):
        X[f"m{m}"] = (mon == m).astype(float)
    for v in CLIM_VARS:
        if v not in c.columns or v not in obs_clim.columns:
            continue
        ref = mon.map(obs_clim[v])
        a = (np.log1p(c[v]) - np.log1p(ref)) if v in MULTIPLICATIVE else (c[v] - ref)
        a = a / scale.get(v, 1.0)
        # One feature per variable: the mean anomaly over the lag window.
        # Separate per-lag terms are poorly identified from short records
        # (within-month anomalies are mostly short-lived noise, so lag
        # coefficients can cancel), yet under sustained warming every lag
        # shifts together and only their sum matters. The window mean's
        # coefficient is exactly that sustained-shift response.
        X[f"{v}_anom_mean{min(lags)}_{max(lags)}"] = sum(a.shift(L) for L in lags) / len(lags)
    return X


def _samples_from_quantiles(qrow: np.ndarray, n: int, rng) -> np.ndarray:
    """Inverse-CDF sampling from the 9 forecast quantiles (tails clamped)."""
    u = rng.uniform(DEFAULT_PROBS[0], DEFAULT_PROBS[-1], n)
    return np.interp(u, DEFAULT_PROBS, np.maximum.accumulate(qrow))


# ---------------------------------------------------------------------------
# long-term climate-response model
# ---------------------------------------------------------------------------
class ClimateResponseModel:
    """Poisson GLM on month dummies + lagged climate anomalies, with an NB2
    dispersion estimate and year-block bootstrap for parameter uncertainty."""

    def __init__(self, n_bootstrap=60, alpha=None, lags=(0, 1, 2), response="seasonal",
                 quadratic_temperature=True, lag_map=None, temperature_curve=None, distributed=None,
                 random_state=42):
        self.quadratic_temperature = bool(quadratic_temperature); self.lag_map = lag_map
        self.distributed = distributed
        self.temperature_curve = temperature_curve
        if response not in ("seasonal", "anomaly"):
            raise ValueError("response must be 'seasonal' or 'anomaly'")
        self.response = response
        self.alpha = alpha if alpha is not None else (1e-2 if response == "seasonal" else 1e-3)
        self.n_bootstrap = int(n_bootstrap); self.lags = tuple(lags)
        self.rng = np.random.default_rng(random_state)

    def fit(self, disease: pd.DataFrame, climate: pd.DataFrame, obs_clim: pd.DataFrame,
            exclude_times=(), extra=None):
        """Fit on the target district, optionally pooling `extra` districts.

        extra : list of (disease, climate) frames for other districts. The
            climate coefficients are shared; each extra district gets its
            own intercept (dummy), so the fitted model's intercept belongs to
            the target district. Climate is standardised on the pooled scale
            so coefficients mean the same thing everywhere. Only supported
            with response="seasonal".
        """
        extra = list(extra or [])
        if extra and self.response != "seasonal":
            raise ValueError("Pooling across districts requires response='seasonal'")
        c = climate.sort_values("time").reset_index(drop=True)
        mon = c["time"].dt.month
        self.scale_ = {}
        for v in CLIM_VARS:
            if v in c.columns and v in obs_clim.columns:
                ref = mon.map(obs_clim[v])
                a = (np.log1p(c[v]) - np.log1p(ref)) if v in MULTIPLICATIVE else (c[v] - ref)
                sd = float(np.nanstd(a)); self.scale_[v] = sd if sd > 1e-9 else 1.0
        self.obs_clim_ = obs_clim
        self.centre_ = _pooled_centre([c] + [ec for _, ec in extra], self.temperature_curve)
        self.n_extra_ = len(extra)
        parts = []
        for k, (dk, ck) in enumerate([(disease, c)] + extra):
            Xk = self._features(ck.sort_values("time").reset_index(drop=True))
            for j in range(1, len(extra) + 1):
                Xk[f"district_{j}"] = float(k == j)
            dd = Xk.merge(dk[["time", "cases"]], on="time", how="inner")
            if len(exclude_times):
                dd = dd[~dd["time"].isin(pd.to_datetime(list(exclude_times)))]
            parts.append(dd.dropna())
        d = pd.concat(parts, ignore_index=True)
        if len(d) < 36:
            raise ValueError(f"Climate-response model needs >= 36 complete months; got {len(d)}")
        self.features_ = [col for col in d.columns if col not in ("time", "cases")]
        Xm, y = d[self.features_].to_numpy(float), d["cases"].to_numpy(float)
        years = d["time"].dt.year.to_numpy()
        self.models_ = [PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(Xm, y)]
        uy = np.unique(years)
        for _ in range(self.n_bootstrap):
            pick = self.rng.choice(uy, size=len(uy), replace=True)
            idx = np.concatenate([np.flatnonzero(years == yy) for yy in pick])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models_.append(PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(Xm[idx], y[idx]))
        mu = self.models_[0].predict(Xm)
        self.deviance_explained_ = float(self.models_[0].score(Xm, y))
        # NB2 dispersion by moments: Var = mu + k mu^2
        k = np.sum(((y - mu) ** 2 - mu)) / max(np.sum(mu ** 2), 1e-9)
        self.dispersion_ = float(max(k, 1e-6))
        self.n_train_ = int(len(d))
        return self

    def _features(self, clim):
        return _anomaly_features(clim, self.obs_clim_, self.scale_, self.lags, self.response, self.centre_,
                                 self.quadratic_temperature, self.lag_map, self.temperature_curve, self.distributed)

    def mean_paths(self, climate: pd.DataFrame, history_climate: pd.DataFrame):
        """Expected monthly cases for each bootstrap fit; rows=fits, cols=months.

        `history_climate` supplies the lagged anomalies for the first months.
        """
        allc = pd.concat([history_climate, climate], ignore_index=True)
        allc = allc.drop_duplicates("time", keep="last").sort_values("time")
        X = self._features(allc)
        X = X[X["time"].isin(set(climate["time"]))]
        for col in self.features_:
            if col.startswith("district_") and col not in X.columns:
                X[col] = 0.0                                   # target district = reference
        Xm = X[self.features_].fillna(0.0).to_numpy(float)
        return X["time"].to_numpy(), np.vstack([m.predict(Xm) for m in self.models_])

    def sample(self, mu_paths: np.ndarray, n_per_fit: int, rng) -> np.ndarray:
        mu = np.repeat(mu_paths, n_per_fit, axis=0)
        k = self.dispersion_
        lam = rng.gamma(1.0 / k, k * mu)
        return rng.poisson(lam).astype(float)


def _pooled_centre(climates, temperature_curve=None):
    allc = pd.concat(climates, ignore_index=True)
    centre = {}
    for v in CLIM_VARS:
        if v in allc.columns:
            x = _transform(v, allc[v].to_numpy(float), temperature_curve)
            centre[v] = (float(np.nanmean(x)), float(np.nanstd(x)) or 1.0)
    return centre


def _dl_curves(model, dist_cfg):
    """Per-variable lag curves (coefficient per lag, cumulative effect, peak lag) from a fitted model."""
    if not dist_cfg:
        return None
    from .distributed_lag import lag_curve
    coef = dict(zip(model.features_, model.models_[0].coef_))
    out = {}
    for v, L in dist_cfg["max_lag"].items():
        g = [coef.get(f"{v}_dl{k}") for k in range(dist_cfg.get("degree", 2) + 1)]
        if all(x is not None for x in g):
            out[v] = lag_curve(g, L, dist_cfg.get("degree", 2))
    return out


def select_lags_cv(disease, climate, obs_clim, exclude_times=(), max_lag=3, n_blocks=4,
                   quadratic_temperature=True, alpha=1e-2, extra=None, temperature_curve=None):
    """Choose ONE lag (0..max_lag) per climate variable by blocked
    cross-validation over contiguous groups of years (Poisson deviance).

    Why: with every lag of every variable in the model, climate variables
    that share a seasonal cycle can stand in for one another, so effects get
    credited to the wrong variable. In a synthetic test the flexible model
    fitted history slightly better yet put ~zero weight on temperature and
    halved the projected warming response; one lag per variable recovered it.
    Returns (lag_map, cv_table).
    """
    from itertools import product
    from sklearn.metrics import mean_poisson_deviance
    extra = list(extra or [])
    c = climate.sort_values("time").reset_index(drop=True)
    vars_ = [v for v in CLIM_VARS if v in c.columns]
    centre = _pooled_centre([c] + [ec for _, ec in extra], temperature_curve)
    if temperature_curve is not None:
        quadratic_temperature = False
    parts = []
    for k, (dk, ck) in enumerate([(disease, c)] + extra):
        full = _anomaly_features(ck.sort_values("time").reset_index(drop=True), obs_clim, {}, response="seasonal",
                                 centre=centre, quadratic_temperature=False,
                                 lag_map={v: range(0, max_lag + 1) for v in vars_},
                                 temperature_curve=temperature_curve)
        for j in range(1, len(extra) + 1):
            full[f"district_{j}"] = float(k == j)
        dd = full.merge(dk[["time", "cases"]], on="time").dropna()
        if len(exclude_times):
            dd = dd[~dd["time"].isin(pd.to_datetime(list(exclude_times)))]
        parts.append(dd)
    d = pd.concat(parts, ignore_index=True)
    dcols = [f"district_{j}" for j in range(1, len(extra) + 1)]
    years = d["time"].dt.year.to_numpy(); uy = np.unique(years)
    blocks = np.array_split(uy, min(n_blocks, len(uy)))
    y = d["cases"].to_numpy(float)
    rows = []
    for combo in product(range(-1, max_lag + 1), repeat=len(vars_)):      # -1 = variable left out
        if all(L < 0 for L in combo):
            continue
        cols = [f"{v}_lvl_l{L}" for v, L in zip(vars_, combo) if L >= 0] + dcols
        X = d[cols].to_numpy(float)
        if quadratic_temperature and "temperature" in vars_ and combo[vars_.index("temperature")] >= 0:
            X = np.column_stack([X, d[f"temperature_lvl_l{combo[vars_.index('temperature')]}"].to_numpy(float) ** 2])
        dev = []
        for b in blocks:
            te = np.isin(years, b)
            if te.sum() == 0 or (~te).sum() < 24:
                continue
            m = PoissonRegressor(alpha=alpha, max_iter=500).fit(X[~te], y[~te])
            dev.append(mean_poisson_deviance(y[te], np.maximum(m.predict(X[te]), 1e-9)) * te.sum())
        rows.append({**{v: L for v, L in zip(vars_, combo)}, "cv_deviance": float(np.sum(dev) / len(y))})
    table = pd.DataFrame(rows).sort_values("cv_deviance").reset_index(drop=True)
    best = table.iloc[0]
    return {v: [int(best[v])] for v in vars_ if best[v] >= 0}, table


def near_best_structures(table, vars_, tolerance=0.02, max_structures=8):
    """All lag structures whose CV deviance is within `tolerance` of the best.

    The data usually cannot distinguish these (climate variables share a
    seasonal cycle), yet they can imply very different responses to warming.
    Averaging over them carries that structural uncertainty into the ranges
    instead of betting on one arbitrary winner.
    """
    best = table["cv_deviance"].iloc[0]
    keep = table[table["cv_deviance"] <= best * (1 + tolerance)].head(max_structures)
    return [{v: [int(r[v])] for v in vars_ if r[v] >= 0} for _, r in keep.iterrows()]


# ---------------------------------------------------------------------------
# hybrid projector
# ---------------------------------------------------------------------------
@dataclass
class ScenarioOutlook:
    series: pd.DataFrame      # time, ssp, model, weight_near_term, quantiles  (per GCM)
    pooled: pd.DataFrame      # time, ssp, quantiles (samples pooled across GCMs)
    decades: pd.DataFrame     # decade summary vs baseline, per SSP
    annual: pd.DataFrame      # ssp, year, quantiles of annual total (pooled across GCMs)
    baseline: dict
    metadata: dict = field(default_factory=dict)


class HybridScenarioProjector:
    def __init__(self, near_term_models=("seasonal_naive", "poisson", "random_forest",
                                         "extra_trees", "gradient_boosting"),
                 near_term_months=12, blend_months=12, baseline_years=None,
                 n_bootstrap=60, samples_per_series=400, near_term_simulations=500,
                 population_at_risk=None, response="seasonal", quadratic_temperature=True,
                 alpha=None, lag_selection="ensemble", structure_tolerance=0.02, max_structures=8,
                 temperature_curve=None, tuning=None, comparison_models=("random_forest", "gradient_boosting"),
                 random_state=42):
        self.temperature_curve = temperature_curve; self.tuning = tuning
        self.comparison_models = tuple(comparison_models or ())
        self.lag_selection = lag_selection; self.structure_tolerance = structure_tolerance
        self.max_structures = int(max_structures)
        self.response = response; self.quadratic_temperature = quadratic_temperature; self.alpha = alpha
        self.near_term_models = tuple(near_term_models)
        self.near_term_months = int(near_term_months); self.blend_months = int(blend_months)
        self.baseline_years = baseline_years; self.n_bootstrap = n_bootstrap
        self.S = int(samples_per_series); self.near_sims = int(near_term_simulations)
        self.population_at_risk = population_at_risk; self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def _weights(self, n):
        """Share of each month's samples taken from the near-term forecast."""
        k = np.arange(1, n + 1, dtype=float)
        w = np.zeros(n)
        w[k <= self.near_term_months] = 1.0
        if self.blend_months > 0:
            ramp = (k > self.near_term_months) & (k <= self.near_term_months + self.blend_months)
            w[ramp] = 1.0 - (k[ramp] - self.near_term_months) / float(self.blend_months + 1)
        return w

    @staticmethod
    def _population_factors(pop, baseline_population, base_years):
        """Return f(ssp, year) = population / baseline population, or None.

        pop : DataFrame with columns year, population and optionally ssp
            (e.g. SSP population projections for the district). Years are
            linearly interpolated; outside the table the nearest value is used.
        baseline_population : float, optional. Defaults to the table's mean
            population over the baseline years (which must then be covered).
        """
        if pop is None:
            return None
        p = pop.copy()
        if not {"year", "population"} <= set(p.columns):
            raise ValueError("population_projection needs columns 'year' and 'population' (and optionally 'ssp')")
        p["ssp"] = p["ssp"].str.lower() if "ssp" in p.columns else "all"
        if baseline_population is None:
            b = p[p["year"].isin(list(base_years))]
            if b.empty:
                raise ValueError("population_projection does not cover the baseline years; pass baseline_population")
            baseline_population = float(b["population"].mean())
        curves = {k: g.sort_values("year") for k, g in p.groupby("ssp")}
        def f(ssp, year):
            g = curves.get(ssp) if ssp in curves else curves.get("all")
            if g is None:
                return 1.0
            return float(np.interp(year, g["year"], g["population"])) / baseline_population
        return f

    def _tree_comparison(self, d, obs, obs_clim, proj, origin, excl, centre, base_years, peak_months):
        """Tree-based models as a comparison for the long-term projections.

        Tree models (random forest, gradient boosting) cannot extrapolate: when
        projected climate goes beyond the training range, their predictions
        stay at the level of the most extreme training months. They are
        therefore reported next to the main projection, never instead of it,
        together with how often projected climate leaves the training range.
        """
        from .tuning import tune, build_estimator
        vars_ = [v for v in CLIM_VARS if v in obs.columns]
        lag_map = {v: range(0, 4) for v in vars_}

        def feats(clim):
            X = _anomaly_features(clim, obs_clim, {}, response="seasonal", centre=centre,
                                  quadratic_temperature=False, lag_map=lag_map)
            mo = X["time"].dt.month.to_numpy()
            X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
            return X
        tr = feats(obs).merge(d[["time", "cases"]], on="time").dropna()
        tr = tr[~tr["time"].isin(excl)].sort_values("time")
        cols = [c for c in tr.columns if c not in ("time", "cases")]
        lvl0 = [f"{v}_lvl_l0" for v in vars_]
        lo, hi = tr[lvl0].min(), tr[lvl0].max()
        rows = []
        for name in self.comparison_models:
            params, _ = tune(name, tr[cols], tr["cases"].to_numpy(float), self.tuning, self.random_state)
            est = build_estimator(name, params, self.random_state).fit(tr[cols].to_numpy(float), tr["cases"].to_numpy(float))
            b = feats(obs); b = b[b["time"].dt.year.isin(list(base_years)) & ~b["time"].isin(excl)].dropna()
            base_annual = float(np.maximum(est.predict(b[cols].to_numpy(float)), 0).sum() / max(b["time"].dt.year.nunique(), 1))
            for s_ in sorted(proj["ssp"].unique()):
                per_gcm = {}
                outside = []
                for m_, g in proj[proj["ssp"] == s_].groupby("model"):
                    X = feats(pd.concat([obs, g[g["time"] > origin]]).drop_duplicates("time", keep="first"))
                    X = X[X["time"] > origin].dropna()
                    pred = np.maximum(est.predict(X[cols].to_numpy(float)), 0)
                    outside.append(float(((X[lvl0] < lo) | (X[lvl0] > hi)).any(axis=1).mean()))
                    t = X["time"]
                    for dd in sorted(set((t.dt.year // 10) * 10)):
                        sel = ((t.dt.year // 10) * 10 == dd).to_numpy()
                        yrs = t[sel].dt.year.unique()
                        if len(yrs) >= 5:
                            per_gcm.setdefault(dd, []).append(pred[sel].sum() / len(yrs) / max(base_annual, 1e-9) - 1)
                for dd, ch in per_gcm.items():
                    rows.append({"model": name, "ssp": s_, "decade": f"{dd}s",
                                 "change_median_across_gcms": float(np.median(ch)),
                                 "change_min_gcm": float(np.min(ch)), "change_max_gcm": float(np.max(ch)),
                                 "share_months_outside_training_range": float(np.mean(outside))})
        return pd.DataFrame(rows)

    def backtest(self, disease, climate_hist, test_years=5, exclude_times=(), extra_districts=None):
        """Check the long-term climate response on years it has not seen.

        Refits everything (lag selection, bootstrap) on data up to
        `test_years` before the end of the record, then "projects" the
        held-out years using their observed climate, exactly as future
        scenarios are projected. Returns a per-year table (observed vs
        projected annual cases with 80% range) and a summary including a
        comparison with simply repeating the training-period mean.
        """
        d = canonicalize_disease(disease); d["time"] = d["time"].dt.to_period("M").dt.to_timestamp()
        c = canonicalize_climate(climate_hist, require_all=True); c["time"] = c["time"].dt.to_period("M").dt.to_timestamp()
        excl = set(pd.to_datetime(list(exclude_times)))
        last = int(d["time"].dt.year.max())
        origin = pd.Timestamp(f"{last - int(test_years)}-12-31")
        pseudo = c.rename(columns={"temperature": "mean_temperature", "rainfall": "mean_Rain",
                                   "humidity": "mean_SH", "enso": "Nino_anomaly"}).assign(model="observed", ssp="backtest")
        bt = HybridScenarioProjector(near_term_months=0, blend_months=0, n_bootstrap=max(20, self.n_bootstrap // 2),
                                     samples_per_series=self.S, response=self.response,
                                     quadratic_temperature=self.quadratic_temperature, alpha=self.alpha,
                                     lag_selection=self.lag_selection, temperature_curve=self.temperature_curve,
                                     comparison_models=(), random_state=self.random_state)
        o = bt.project(d, c, pseudo, origin, end_year=last, exclude_times=excl, extra_districts=extra_districts)
        obs = d[(d["time"] > origin) & ~d["time"].isin(excl)]
        full = obs.groupby(obs["time"].dt.year).agg(cases=("cases", "sum"), months=("cases", "size"))
        full = full[full["months"] == 12]["cases"]
        a = o.annual.set_index("year")
        rows = []
        train = d[(d["time"] <= origin) & ~d["time"].isin(excl)]
        clim_mean = float(train.groupby(train["time"].dt.year)["cases"].sum().mean())
        for yy, y in full.items():
            if yy not in a.index:
                continue
            r = a.loc[yy]
            rows.append({"year": int(yy), "observed": float(y), "projected_median": float(r.p50),
                         "p10": float(r.p10), "p90": float(r.p90),
                         "inside_80": bool(r.p10 <= y <= r.p90), "training_mean": clim_mean})
        tab = pd.DataFrame(rows)
        if tab.empty:
            return tab, {"note": "no complete held-out years to compare"}
        ape = (tab.projected_median - tab.observed).abs() / tab.observed.clip(lower=1)
        ape0 = (tab.training_mean - tab.observed).abs() / tab.observed.clip(lower=1)
        summary = {"test_years": f"{origin.year + 1}-{last}", "n_years": int(len(tab)),
                   "share_inside_80": float(tab.inside_80.mean()),
                   "median_abs_pct_error": float(ape.median()),
                   "median_abs_pct_error_training_mean": float(ape0.median()),
                   "lag_structures": o.metadata.get("lag_structures")}
        return tab, summary

    def project(self, disease, climate_hist, projection, origin, end_year=2050,
                ssps=None, gcms=None, exclude_times=(), peak_months=None, extra_districts=None,
                population_projection=None, baseline_population=None):
        """extra_districts : optional list/dict of (disease, climate) for other
        districts. When given, the long-term climate response is learned
        jointly across districts (shared coefficients, own intercepts), which
        separates climate variables whose seasonal cycles coincide in one
        place but not in others."""
        from .forecaster import ClimaidV2Forecaster
        origin = pd.Timestamp(origin)
        d = canonicalize_disease(disease)
        d["time"] = d["time"].dt.to_period("M").dt.to_timestamp()
        d = d[d["time"] <= origin]
        obs = canonicalize_climate(climate_hist, require_all=True)
        obs["time"] = obs["time"].dt.to_period("M").dt.to_timestamp()
        obs = obs[obs["time"] <= origin]
        excl = set(pd.to_datetime(list(exclude_times)))
        extra = []
        if extra_districts:
            items = extra_districts.values() if isinstance(extra_districts, dict) else extra_districts
            for dk, ck in items:
                dk = canonicalize_disease(dk); dk["time"] = dk["time"].dt.to_period("M").dt.to_timestamp()
                ck = canonicalize_climate(ck, require_all=True); ck["time"] = ck["time"].dt.to_period("M").dt.to_timestamp()
                extra.append((dk[dk["time"] <= origin], ck[ck["time"] <= origin]))
            if self.response != "seasonal":
                raise ValueError("extra_districts requires response='seasonal'")

        yrs = sorted(set(d["time"].dt.year) - {t.year for t in excl})
        base_years = list(self.baseline_years or yrs)
        obs_clim = monthly_climatology(obs, base_years)

        proj = _canon_projection(projection)
        if ssps: proj = proj[proj["ssp"].isin([s.lower() for s in ssps])]
        if gcms: proj = proj[proj["model"].isin(gcms)]
        proj = proj[proj["time"].dt.year <= int(end_year)]
        if proj.empty:
            raise ValueError("No projection data for the requested SSPs / models / years")
        proj, notes = bias_correct(proj, obs_clim, base_years)

        # lag structure(s) for the long-term model (seasonal mode only)
        structures, lag_info = [None], "anomaly mode: 3-month mean anomaly per variable"
        if self.response == "seasonal":
            if isinstance(self.lag_selection, dict):
                structures = [{k: [int(x) for x in (v if isinstance(v, (list, tuple, range)) else [v])]
                               for k, v in self.lag_selection.items()}]
                lag_info = "user/v1-supplied lags"
            elif self.lag_selection in ("auto", "ensemble"):
                best_map, cvt = select_lags_cv(d, obs, obs_clim, exclude_times=excl,
                                               quadratic_temperature=self.quadratic_temperature, extra=extra,
                                               temperature_curve=self.temperature_curve)
                self.lag_cv_table_ = cvt
                vars_ = [v for v in CLIM_VARS if v in cvt.columns]
                if self.lag_selection == "ensemble":
                    structures = near_best_structures(cvt, vars_, self.structure_tolerance, self.max_structures)
                    lag_info = (f"{len(structures)} lag structure(s) within {self.structure_tolerance:.0%} of the best "
                                "blocked-cross-validation score, averaged (structural uncertainty)")
                else:
                    structures = [best_map]
                    lag_info = "single best lag structure by blocked cross-validation over years"
            elif self.lag_selection == "distributed":
                from .distributed_lag import DEFAULT_MAX_LAG, DEFAULT_DEGREE
                self.distributed_cfg_ = {"max_lag": dict(DEFAULT_MAX_LAG), "degree": DEFAULT_DEGREE}
                lag_info = ("distributed lag: a smooth curve over lags 0-6 months (ENSO 0-12) for every variable, "
                            "so the effect of a sustained change is summed over all lags")
            else:
                lag_info = "all lags 0-3 for every variable"
        per_struct_boot = max(10, self.n_bootstrap // len(structures))
        dist_cfg = getattr(self, "distributed_cfg_", None) if self.lag_selection == "distributed" else None
        lts = [ClimateResponseModel(lag_map=lm, distributed=dist_cfg, n_bootstrap=per_struct_boot, response=self.response, alpha=self.alpha,
                                    quadratic_temperature=self.quadratic_temperature,
                                    temperature_curve=self.temperature_curve,
                                    random_state=self.random_state + i).fit(d, obs, obs_clim, exclude_times=excl,
                                                                            extra=extra)
               for i, lm in enumerate(structures)]
        lt = lts[0]
        lag_map = structures[0]

        # near-term engine (fitted once, driven by each GCM's corrected climate)
        nt = None
        if self.near_term_months > 0:
            try:
                nt = ClimaidV2Forecaster(models=self.near_term_models, tuning=self.tuning,
                                         population_at_risk=self.population_at_risk).fit(d, obs, cutoff=origin)
            except Exception as exc:
                notes.append(f"Near-term forecaster could not be fitted ({exc}); outlook uses the "
                             "climate-response model from the first month.")

        # baseline: model-implied under observed historical climate (apples to apples)
        b_obs = obs[obs["time"].dt.year.isin(base_years)]
        hist_before = obs[obs["time"] < b_obs["time"].min()]
        _b = [m_.mean_paths(b_obs, hist_before) for m_ in lts]
        bt = _b[0][0]; bmu = np.vstack([np.atleast_2d(x[1][0]) for x in _b])   # central fit of each structure
        bmu = bmu.mean(axis=0, keepdims=True)
        hist_means = d[d["time"].dt.year.isin(base_years) & ~d["time"].isin(excl)]
        if peak_months is None:
            clim_cases = hist_means.groupby(hist_means["time"].dt.month)["cases"].mean()
            peak_months = sorted(clim_cases.sort_values(ascending=False).index[:3].tolist())
        bt = pd.to_datetime(bt)
        n_years = len(set(bt.year))
        baseline = {
            "years": f"{min(base_years)}-{max(base_years)}",
            "peak_months": [int(m) for m in peak_months],
            "observed_annual_mean": float(hist_means.groupby(hist_means["time"].dt.year)["cases"].sum().mean()),
            "model_annual_mean": float(bmu[0].sum() / max(n_years, 1)),
            "model_peak_season_mean": float(bmu[0][np.isin(bt.month, peak_months)].sum() / max(n_years, 1)),
        }

        series_rows, pooled_samples = [], {}
        for (m, s), g in proj.groupby(["model", "ssp"], sort=True):
            fut = g[g["time"] > origin].sort_values("time")
            if fut.empty:
                continue
            hist_g = pd.concat([obs, g[g["time"] <= origin]]).drop_duplicates("time", keep="first")
            draws = []
            for m_ in lts:
                times, mu = m_.mean_paths(fut, hist_g)
                per_fit = max(1, self.S // (mu.shape[0] * len(lts)))
                draws.append(m_.sample(mu, per_fit, self.rng))
            times = pd.to_datetime(times)
            lt_s = np.vstack(draws)                                      # (S', T)
            S = lt_s.shape[0]
            w = self._weights(len(times))
            if nt is not None and w.max() > 0:
                h = int(np.max(np.flatnonzero(w > 0)) + 1)
                try:
                    b = nt.predict(fut.head(h)[["time", *[v for v in CLIM_VARS if v in fut.columns]]],
                                   h, n_simulations=self.near_sims)
                    ens = b.forecasts["ensemble"] if "ensemble" in b.forecasts else next(iter(b.forecasts.values()))
                    q = ens[QCOLS].to_numpy(float)
                    for j in range(h):
                        n_near = int(round(w[j] * S))
                        if n_near:
                            idx = self.rng.choice(S, n_near, replace=False)
                            lt_s[idx, j] = _samples_from_quantiles(q[j], n_near, self.rng)
                except Exception as exc:
                    notes.append(f"{m}/{s}: near-term forecast failed ({exc}); used climate-response model only.")
                    w = np.zeros_like(w)
            qs = np.quantile(lt_s, DEFAULT_PROBS, axis=0).T
            frame = pd.DataFrame(qs, columns=QCOLS)
            frame.insert(0, "time", times); frame.insert(1, "ssp", s); frame.insert(2, "model", m)
            frame["weight_near_term"] = w
            series_rows.append(frame)
            pooled_samples.setdefault(s, []).append(pd.DataFrame(lt_s.T, index=times))

        # Extrapolation check: how often do projected anomalies exceed the
        # range seen in training? Beyond it the response is extrapolated.
        tr = lt._features(obs)
        anom_cols = [c for c in tr.columns if ("_anom_" in c or "_lvl_l0" in c)]
        tr_max = tr[anom_cols].abs().max()
        extrap = {}
        for s_ in sorted(proj["ssp"].unique()):
            g = proj[proj["ssp"] == s_]
            shares = []
            for m_, gm in g.groupby("model"):
                f = lt._features(pd.concat([obs, gm[gm["time"] > origin]]))
                f = f[f["time"] > origin]
                shares.append((f[anom_cols].abs() > tr_max).mean())
            extrap[s_] = pd.concat(shares, axis=1).mean(axis=1).round(3).to_dict()
        for s_, dct in extrap.items():
            for col, share in dct.items():
                if share > 0.25:
                    notes.append(f"{s_}: {share:.0%} of projected months have {col.split('_anom')[0].split('_lvl')[0]} values "
                                 "beyond anything in the training record, so the response there is extrapolated.")

        if not series_rows:
            raise ValueError("No projection months after the forecast origin")
        series = pd.concat(series_rows, ignore_index=True)

        def _summaries(A, t, s, base_annual, base_peak):
            ann, dec_ = [], []
            for yy in sorted(set(t.year)):
                sel_y = t.year == yy
                if sel_y.sum() < 12:
                    continue
                tot = A[sel_y].sum(axis=0)
                ann.append({"ssp": s, "year": int(yy), **{f"p{int(q*100):02d}": float(np.quantile(tot, q))
                                                          for q in (.05, .10, .25, .50, .75, .90, .95)}})
            dec = (t.year // 10) * 10
            for dd in sorted(set(dec)):
                sel = dec == dd
                yrs_d = sorted(set(t[sel].year))
                if len(yrs_d) < 5:
                    continue
                annual = np.vstack([A[sel & (t.year == yy)].sum(axis=0) for yy in yrs_d]).mean(axis=0)
                pk = np.vstack([A[sel & (t.year == yy) & np.isin(t.month, peak_months)].sum(axis=0)
                                for yy in yrs_d]).mean(axis=0)
                ch = annual / max(base_annual, 1e-9) - 1.0
                dec_.append({
                    "ssp": s, "decade": f"{dd}s", "years": len(yrs_d),
                    "annual_median": float(np.median(annual)),
                    "annual_p10": float(np.quantile(annual, .1)), "annual_p90": float(np.quantile(annual, .9)),
                    "peak_season_median": float(np.median(pk)),
                    "change_vs_baseline_median": float(np.median(ch)),
                    "change_p10": float(np.quantile(ch, .1)), "change_p90": float(np.quantile(ch, .9)),
                    "share_gcm_samples_increase": float(np.mean(ch > 0)),
                })
            return ann, dec_

        pop_factor = self._population_factors(population_projection, baseline_population, base_years)
        pooled_rows, dec_rows, ann_rows, dec_pop, ann_pop = [], [], [], [], []
        for s, frames in pooled_samples.items():
            common = sorted(set.intersection(*[set(f.index) for f in frames]))
            A = np.hstack([f.loc[common].to_numpy() for f in frames])     # (T, S*n_gcm)
            qs = np.quantile(A, DEFAULT_PROBS, axis=1).T
            pf = pd.DataFrame(qs, columns=QCOLS); pf.insert(0, "time", pd.to_datetime(common)); pf.insert(1, "ssp", s)
            pf["n_gcms"] = len(frames); pooled_rows.append(pf)
            t = pd.to_datetime(common)
            a_, d_ = _summaries(A, t, s, baseline["model_annual_mean"], baseline["model_peak_season_mean"])
            ann_rows += a_; dec_rows += d_
            if pop_factor is not None:
                f_ = np.array([pop_factor(s, yy) for yy in t.year])[:, None]
                a_, d_ = _summaries(A * f_, t, s, baseline["model_annual_mean"], baseline["model_peak_season_mean"])
                ann_pop += a_; dec_pop += d_
        pooled = pd.concat(pooled_rows, ignore_index=True)
        decades = pd.DataFrame(dec_rows)
        coefs = dict(zip(lt.features_, lt.models_[0].coef_))
        meta = {
            "forecast_origin": str(origin), "end_year": int(end_year),
            "ssps": sorted(proj["ssp"].unique()), "gcms": sorted(proj["model"].unique()),
            "near_term_months": self.near_term_months, "blend_months": self.blend_months,
            "near_term_models": list(self.near_term_models) if nt is not None else [],
            "baseline_years": baseline["years"], "bias_correction": "monthly mean shift (additive; multiplicative for rainfall)",
            "long_term_model": f"Poisson GLM ({('climate levels, ' + (lag_info if len(structures) > 1 else ('at ' + ', '.join(f'{k} lag {v[0]}' for k, v in lag_map.items()) if lag_map else 'lags 0-3')) + (' + temperature squared' if self.quadratic_temperature else '')) if self.response == 'seasonal' else 'month-of-year + 3-month mean climate anomalies'}), {self.n_bootstrap} year-block bootstrap refits, NB dispersion {lt.dispersion_:.3f}",
            "n_training_months_long_term": lt.n_train_, "excluded_months": len(excl),
            "response_mode": self.response,
            "pooled_districts": len(extra),
            "temperature_curve": (None if self.temperature_curve is None else
                                  {"name": self.temperature_curve if isinstance(self.temperature_curve, str) else "custom",
                                   "tmin_topt_tmax": list(TEMPERATURE_CURVES.get(self.temperature_curve, self.temperature_curve)
                                                          if isinstance(self.temperature_curve, str) else self.temperature_curve)}),
            "lag_selection": lag_info, "lags_used": lag_map, "lag_structures": structures,
            "distributed_lag_curves": _dl_curves(lt, dist_cfg),
            "deviance_explained_long_term": round(lt.deviance_explained_, 3),
            "climate_coefficients": {k: float(v) for k, v in coefs.items() if not k.startswith("m") or "_" in k},
            "extrapolation_share": extrap,
            "notes": notes,
        }
        out = ScenarioOutlook(series, pooled, decades, pd.DataFrame(ann_rows), baseline, meta)
        out.tree_comparison = pd.DataFrame()
        if self.comparison_models:
            try:
                out.tree_comparison = self._tree_comparison(d, obs, obs_clim, proj, origin, excl, lt.centre_,
                                                            base_years, peak_months)
            except Exception as exc:
                meta["notes"].append(f"Tree-model comparison could not be run: {exc}")
        if pop_factor is not None:
            out.decades_with_population = pd.DataFrame(dec_pop)
            out.annual_with_population = pd.DataFrame(ann_pop)
            meta["population_scaling"] = ("expected cases scaled by projected population / baseline population "
                                          "(constant incidence per person); climate-only results are kept alongside")
        return out
