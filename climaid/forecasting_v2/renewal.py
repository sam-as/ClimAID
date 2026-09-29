"""Climate-informed stochastic renewal epidemic model for ClimAID v2."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma
from .features import ClimateFeatureBuilder, ClimateFeatureConfig
from .schema import canonicalize_disease, canonicalize_climate

DEFAULT_PROBS = np.array([.025, .05, .10, .25, .50, .75, .90, .95, .975])


def generation_weights(mean=2.5, sd=1.0, max_lag=8):
    """Discretised gamma generation-interval weights for lags 1..max_lag."""
    if mean <= 0 or sd <= 0 or max_lag < 2:
        raise ValueError("mean and sd must be positive and max_lag >= 2")
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    # Interval probabilities over [0,1],...,[max_lag-1,max_lag].
    edges = np.arange(max_lag + 1, dtype=float)
    w = np.diff(gamma.cdf(edges, a=shape, scale=scale))
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        raise ValueError("Unable to construct generation-interval weights")
    return w / w.sum()


def renewal_force(cases, weights):
    """Past incidence weighted by a discretised generation interval."""
    cases = np.asarray(cases, float)
    weights = np.asarray(weights, float)
    out = np.full(len(cases), np.nan)
    for t in range(len(cases)):
        vals = cases[max(0, t - len(weights)):t][::-1]
        out[t] = float(np.sum(vals * weights[: len(vals)])) if len(vals) else np.nan
    return out


def _nb_nll(y, mu, alpha):
    """Negative-binomial negative log likelihood under NB2 parameterisation."""
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    alpha = max(float(alpha), 1e-8)
    r = 1.0 / alpha
    p = r / (r + mu)
    ll = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + r * np.log(np.clip(p, 1e-12, 1.0))
        + y * np.log(np.clip(1.0 - p, 1e-12, 1.0))
    )
    return float(-np.mean(ll))


@dataclass
class RenewalFitResult:
    success: bool
    message: str
    n_obs: int
    generation_mean: float
    generation_sd: float
    dispersion: float
    depletion_rate: float
    population_mode: str
    coefficients: dict
    objective: float


class ClimateRenewalModel:
    """Renewal model with climate-dependent transmission and optional depletion.

    Population is not required to *run* the renewal model. When population is
    unavailable, v2 estimates a relative epidemic state (S=1 throughout) and
    explicitly records that depletion was not applied. This is preferable to
    silently inventing a population-at-risk value.
    """

    def __init__(
        self,
        date_col="time",
        case_col="cases",
        population_col="population",
        climate_config=None,
        generation_mean=2.5,
        generation_sd=1.0,
        generation_window=8,
        depletion_rate=1.0,
        min_susceptible=.05,
        ridge=.10,
        random_state=42,
    ):
        self.date_col = date_col
        self.case_col = case_col
        self.population_col = population_col
        self.climate_config = climate_config or ClimateFeatureConfig(date_col=date_col)
        self.generation_mean = float(generation_mean)
        self.generation_sd = float(generation_sd)
        self.generation_window = int(generation_window)
        self.depletion_rate = float(depletion_rate)
        self.min_susceptible = float(min_susceptible)
        self.ridge = float(ridge)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.fit_result_ = None

    def fit(self, disease, climate, cutoff=None, population_at_risk=None, feature_names=None):
        d = canonicalize_disease(
            disease, date_col=self.date_col, case_col=self.case_col,
            population_col=self.population_col
        )
        c = canonicalize_climate(climate, date_col=self.date_col, require_all=True)
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else d.time.max()
        d = d[d.time <= cutoff].copy()
        ch = c[c.time <= cutoff].copy()
        if d.empty or ch.empty:
            raise ValueError("No aligned disease/climate history at cutoff")

        pop, pop_mode = self._population(d, population_at_risk)
        builder = ClimateFeatureBuilder(self.climate_config).fit(ch, cutoff)
        cf = builder.transform(ch)
        m = d.merge(cf, on="time", how="inner", validate="one_to_one").sort_values("time").reset_index(drop=True)
        if len(m) < self.generation_window + 10:
            raise ValueError("Too few aligned observations for renewal model")

        default_names = [
            "temperature_lag0", "rainfall_lag0", "humidity_lag0", "enso_lag0",
            "temperature_anomaly", "rainfall_anomaly", "humidity_anomaly",
            "month_sin", "month_cos",
        ]
        names = [x for x in (feature_names or default_names) if x in m.columns]
        if not names:
            raise ValueError("No climate features available for renewal model")

        xr = m[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        med = np.nanmedian(xr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        filled = np.where(np.isfinite(xr), xr, med)
        scale = np.nanstd(filled, axis=0)
        scale[(~np.isfinite(scale)) | (scale <= 1e-12)] = 1.0
        X = (filled - med) / scale
        y = m[self.case_col].to_numpy(float)

        w = generation_weights(self.generation_mean, self.generation_sd, self.generation_window)
        force = renewal_force(y, w)

        if pop_mode == "observed_population":
            # Proper susceptible depletion on a population-at-risk scale.
            s = 1.0
            susceptible = np.empty(len(y))
            for i, (cases, n) in enumerate(zip(y, pop)):
                susceptible[i] = s
                s = max(self.min_susceptible, s - self.depletion_rate * cases / max(float(n), 1.0))
        elif pop_mode == "fixed_population":
            s = 1.0
            susceptible = np.empty(len(y))
            for i, cases in enumerate(y):
                susceptible[i] = s
                s = max(self.min_susceptible, s - self.depletion_rate * cases / max(float(pop[0]), 1.0))
        else:
            susceptible = np.ones(len(y), dtype=float)
            if self.depletion_rate > 0:
                warnings.warn(
                    "Population-at-risk unavailable: renewal forecast will run in relative-incidence mode and susceptible depletion is disabled.",
                    RuntimeWarning,
                )

        valid = (np.arange(len(y)) >= self.generation_window) & np.isfinite(force) & (force > 0)
        if valid.sum() < 12:
            raise ValueError("Too few valid renewal observations after generation-interval warm-up")

        xv, yv = X[valid], y[valid]
        sv, fv = susceptible[valid], force[valid]
        p = xv.shape[1]

        ratio = yv / np.maximum(fv * sv, 1e-6)
        initial_intercept = float(np.log(np.median(np.maximum(ratio, 1e-6))))
        initial_intercept = float(np.clip(initial_intercept, -3.0, 3.0))
        init = np.r_[initial_intercept, np.zeros(p), np.log(.25)]

        def obj(theta):
            eta = theta[0] + xv @ theta[1:1+p]
            rt = np.exp(np.clip(eta, -8, 8))
            mu = rt * sv * fv
            alpha = np.exp(np.clip(theta[-1], -8, 5))
            return _nb_nll(yv, mu, alpha) + self.ridge * float(np.sum(theta[1:1+p] ** 2))

        result = minimize(
            obj, init, method="L-BFGS-B",
            bounds=[(-8, 8)] + [(-3, 3)] * p + [(-8, 5)],
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Renewal optimisation failed: {result.message}")

        self.builder_ = builder
        self.feature_names_ = names
        self.x_median_ = med
        self.x_scale_ = scale
        self.coef_ = result.x[:-1]
        self.dispersion_ = float(np.exp(result.x[-1]))
        self.cutoff_ = cutoff
        self.history_cases_ = y
        self.history_population_ = pop
        self.history_climate_ = ch
        self.initial_susceptible_ = float(susceptible[-1])
        self.population_mode_ = pop_mode
        self.fit_result_ = RenewalFitResult(
            bool(result.success), str(result.message), int(valid.sum()),
            self.generation_mean, self.generation_sd, self.dispersion_,
            self.depletion_rate if pop_mode != "relative_incidence" else 0.0,
            pop_mode,
            {n: float(v) for n, v in zip(["intercept"] + names, self.coef_)},
            float(result.fun),
        )
        return self

    def predict(self, future_climate, horizon=None, n_simulations=2000, probs=None):
        if not hasattr(self, "coef_"):
            raise RuntimeError("ClimateRenewalModel has not been fitted")
        if n_simulations < 100:
            raise ValueError("n_simulations must be >= 100")
        probs = DEFAULT_PROBS if probs is None else np.asarray(probs, float)
        c = canonicalize_climate(future_climate, date_col=self.date_col, require_all=True)
        allc = pd.concat([self.history_climate_, c], ignore_index=True)
        allc = allc.drop_duplicates("time", keep="last").sort_values("time")
        cf = self.builder_.transform(allc)
        ff = cf[cf.time > self.cutoff_].head(int(horizon) if horizon else len(c)).copy()
        if horizon and len(ff) < int(horizon):
            raise ValueError("Future climate does not cover requested horizon")
        if ff.empty:
            raise ValueError("No future climate observations after forecast cutoff")

        xr = ff[self.feature_names_].to_numpy(float)
        X = (np.where(np.isfinite(xr), xr, self.x_median_) - self.x_median_) / self.x_scale_
        weights = generation_weights(self.generation_mean, self.generation_sd, self.generation_window)
        sims = np.zeros((int(n_simulations), len(ff)), dtype=float)
        alpha = max(float(self.dispersion_), 1e-8)
        pop = float(self.history_population_[-1]) if len(self.history_population_) else 1.0

        # STABILITY GUARD: each month is drawn from the model's own previous
        # draws, so a fitted reproduction number above 1 compounds without
        # limit (e.g. an early hindcast with ~3 years of training produced an
        # RMSE of ~19,000 on a series whose monthly counts were under 100).
        # Cap the expected count at `max_growth_factor` x the largest count
        # seen in training, and at the population when one is known. The
        # share of simulated months that hit the cap is exposed as
        # `self.capped_fraction_` so the forecaster can warn about it.
        hist_max = float(np.nanmax(self.history_cases_)) if len(self.history_cases_) else 1.0
        mu_cap = max(getattr(self, "max_growth_factor", 5.0) * max(hist_max, 1.0), 10.0)
        if self.population_mode_ != "relative_incidence" and pop > 1.0:
            mu_cap = min(mu_cap, pop)
        n_capped = 0

        for i in range(int(n_simulations)):
            seq = self.history_cases_.tolist()
            susceptible = self.initial_susceptible_ if self.population_mode_ != "relative_incidence" else 1.0
            for j in range(len(ff)):
                recent = np.asarray(seq[-len(weights):][::-1], dtype=float)
                force = float(np.sum(recent * weights[:len(recent)])) if len(recent) else 0.0
                rt = float(np.exp(np.clip(self.coef_[0] + X[j] @ self.coef_[1:], -8, 8)))
                mu = max(rt * susceptible * max(force, 1e-6), 1e-6)
                if mu > mu_cap:
                    mu = mu_cap
                    n_capped += 1
                # Gamma-Poisson mixture => NB2 with Var(mu + alpha*mu^2).
                draw = float(self.rng.poisson(self.rng.gamma(1.0 / alpha, alpha * mu)))
                sims[i, j] = draw
                seq.append(draw)
                if self.population_mode_ != "relative_incidence":
                    susceptible = max(self.min_susceptible, susceptible - self.depletion_rate * draw / max(pop, 1.0))

        self.capped_fraction_ = n_capped / float(sims.size)
        q = np.quantile(sims, probs, axis=0).T
        out = pd.DataFrame({"time": ff.time.to_numpy()})
        for i, p in enumerate(probs):
            out[f"q{int(round(p * 1000)):03d}"] = q[:, i]
        qcols = [f"q{int(round(p * 1000)):03d}" for p in probs]
        out[qcols] = np.sort(np.maximum(0, out[qcols].to_numpy(float)), axis=1)
        return out

    def _population(self, d, pop):
        if "population" in d.columns and d["population"].notna().any():
            p = pd.to_numeric(d["population"], errors="coerce").ffill().bfill()
            if p.notna().all() and np.all(p.to_numpy(float) > 0):
                return p.to_numpy(float), "observed_population"
        if pop is not None and float(pop) > 0:
            return np.full(len(d), float(pop)), "fixed_population"
        return np.ones(len(d), dtype=float), "relative_incidence"
