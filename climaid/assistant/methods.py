"""How ClimAID's methods work, in plain language and in technical detail.

Each topic has a short plain explanation and a technical one, written from ClimAID's code and
documentation (values such as lag ranges, trial counts and window lengths are the package
defaults). The assistant shows the plain version first and the technical one on request.

`explain_run` describes how a particular forecast was made, from the run's own metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    patterns: tuple          # any match selects the topic
    plain: str
    detail: str
    page: str                # documentation page (relative to the docs root)


TOPICS = [
    Topic("overview", "How ClimAID works (overview)",
          (r"\bhow (does|do) climaid work\b", r"\boverview\b", r"\b(whole|full|climaid) pipeline\b", r"\bwhole (method|process)\b"),
          "ClimAID links monthly disease cases to climate (temperature, rainfall, humidity and El Niño). For the "
          "coming months it makes probabilistic forecasts: expected cases with likely ranges, from several models "
          "that are tuned, tested on the past and compared with a simple 'same as last year' baseline. For the "
          "long term it estimates how cases could change under future climate from CMIP6 climate models. Every "
          "run ends with a plain-language report and a trust rating.",
          "Two pipelines. v2 (recommended): climate-mandatory probabilistic forecasting (22 models, compulsory "
          "leakage-safe tuning, rolling-origin hindcasts, split-conformal interval calibration, WIS scoring "
          "against a seasonal-naive baseline) plus a hybrid near-term + CMIP6 scenario outlook. v1 (legacy): "
          "lag-optimised stacked point model (base -> residual -> correction) with CMIP6 projections and C-DSI "
          "reports. Both use only information available at the forecast start.", ""),
    Topic("data", "Data checks and preparation",
          (r"\bdata (check|cleaning|preparation|validation)\b", r"\bclean(ing)? (the )?data\b", r"\bmissing (months|data|values)\b",
           r"\bweekly\b", r"\bduplicate"),
          "Before modelling, ClimAID checks your case data: unreadable dates, non-numeric or negative counts and "
          "duplicate rows are removed or rejected, and every change is reported. Weekly or daily data are added "
          "up to monthly totals. Built-in climate data for South Asian districts are matched to your district.",
          "Disease columns are recognised by name (date/time/datetime; cases/case/count, with close-match "
          "fallback). Counts are coerced to numeric; invalid rows are dropped with a report; exact duplicate "
          "(time, count) rows are removed; sub-monthly data are summed to months. Climate data provide "
          "temperature, rainfall, specific humidity and an ENSO (Niño) anomaly per district and month.",
          "guide/v2_forecasting/#what-happens-when-you-run-a-forecast"),
    Topic("covid", "The COVID-19 period",
          (r"\bcovid\b", r"\b2020\b", r"\bexclude[ds]? period\b", r"\bdisrupt"),
          "Case counts during COVID-19 were distorted by less testing, reporting and care-seeking. By default "
          "ClimAID replaces each month of 2020 in the training history with that month's typical value from other "
          "years, and never uses those months to score the models. You can keep 2020 or choose another period.",
          "exclude_period: '2020' (default), 'YYYY-MM:YYYY-MM' or 'none'. v2 replaces excluded months at or before "
          "the forecast origin with the same-calendar-month median of the other training years (rows are replaced, "
          "not deleted, because seasonal-naive, renewal and lagged ML models need an unbroken monthly series) and "
          "excludes them from hindcast scoring; months after the origin are never changed. v1 leaves the period "
          "out of training. The scenario outlook excludes it from the long-term fit.",
          "guide/v2_forecasting/#what-happens-when-you-run-a-forecast"),
    Topic("features", "Climate inputs and lags",
          (r"\b(input|feature|predictor)s?\b", r"\blags?\b", r"\bdelay", r"\banomal(y|ies)\b", r"\binteraction"),
          "Climate affects cases with a delay: rain now can mean more mosquitoes, and more dengue, one or two months "
          "later. So the models see climate from the same month and up to three months earlier, how unusual each "
          "month's climate is for the time of year, El Niño, and the cases of the last three months.",
          "v2 ML inputs: temperature, rainfall and humidity at lags 0-3 months and their anomalies from the "
          "monthly climatology (10-year window, fitted only on data before the origin); ENSO at lags 0-3; ENSO "
          "interactions (each climate variable x recent ENSO); seasonal terms; cases at lags 1-3. Regression-type "
          "models also get distributed-lag features (see 'distributed lags'). Feature statistics are fitted only on "
          "data available at the forecast origin.",
          "guide/v2_forecasting/#models-22"),
    Topic("distributed_lags", "Distributed lag effects",
          (r"\bdistributed[ -]lags?\b", r"\blag curves?\b", r"\bsmooth lag"),
          "Instead of one fixed delay, a variable's effect can be spread smoothly over several months. This lets "
          "the regression models use long delays (for El Niño up to a year) with only a few extra inputs.",
          "Each variable's lags are summarised by a degree-2 polynomial basis over lags 0-6 months (ENSO 0-12), "
          "giving a smooth lag curve. On by default for regression-type v2 models (linear family, Poisson, Tweedie, "
          "spline Poisson, Bayesian ridge, Huber); tree and other flexible models combine raw lags themselves. On the "
          "synthetic benchmark Poisson improved on all three datasets with them. In the scenario outlook they are "
          "optional (lag_selection='distributed'); the fitted curves recover the total effect better than its timing.",
          "guide/v2_forecasting/#models-22"),
    Topic("models", "The models",
          (r"\bwhich models\b", r"\bwhat models\b", r"\bmodels? (are|is) (used|there|available)\b", r"\blist (of )?models\b",
           r"\bmachine learning\b"),
          "ClimAID v2 compares several kinds of model: the 'same as last year' baseline every model must beat, a "
          "transmission model in which new cases come from recent cases and climate, the classic SARIMAX time-series "
          "model, regression models, and "
          "machine-learning models such as random forests. Their forecasts are also combined into an ensemble.",
          "22 v2 models. Baseline: seasonal naive. Mechanistic: climate renewal model. Time series: SARIMAX. Regression: linear, ridge, "
          "lasso, elastic net, Poisson, Tweedie, spline Poisson (GAM-style), Bayesian ridge, Huber. Tree ensembles: "
          "random forest, extra trees, gradient boosting, histogram gradient boosting (Poisson loss), XGBoost, "
          "LightGBM, CatBoost (last three need climaid[ml]). Other: MLP neural network, SVR, k-nearest neighbours. "
          "v1 inside v2: v1_stack. Defaults: seasonal_naive, renewal, poisson, random_forest, extra_trees, "
          "gradient_boosting, plus the ensemble.",
          "guide/v2_forecasting/#models-22"),
    Topic("baseline", "The 'same as last year' baseline",
          (r"\bseasonal[ _-]?naive\b", r"\bsame as (last|recent) years?\b", r"\bthe baseline\b", r"\bbenchmark to beat\b"),
          "The simplest possible forecast: each month gets last year's value for the same month. If a model can't "
          "beat this, it isn't adding anything, so every report compares the models with it.",
          "Seasonal naive: point forecast y(t) = y(t - 12) (period in the data's own time unit, so weekly data use "
          "weekly lags), with uncertainty from the empirical distribution of its past seasonal errors. Scores are "
          "reported relative to it (WIS ratio below 1.0 = better than the baseline).",
          "guide/v2_forecasting/#models-22"),
    Topic("renewal", "The climate transmission (renewal) model",
          (r"\brenewal\b", r"\btransmission model\b", r"\bmechanistic\b", r"\bgeneration interval\b", r"\bsusceptible"),
          "A model of how infection spreads: new cases come from recent cases, multiplied by how favourable the "
          "climate is for transmission. If you give the population at risk, it also allows for people becoming "
          "immune; otherwise it works with relative numbers and the report says so.",
          "Expected cases mu(t) = R(t) x S(t) x Lambda(t). Lambda(t) is past incidence weighted by a discretised "
          "gamma generation interval (mean 2.5, SD 1.0 time steps, over 8 steps). log R(t) = intercept + climate "
          "features x coefficients (ridge-penalised). S(t) is the susceptible fraction, depleted by cases / "
          "population when a population at risk is available, else fixed at 1 (relative-incidence mode). Fitted by "
          "negative-binomial maximum likelihood; forecasts are simulated. Expected cases are capped at 5x the "
          "training maximum, with a warning.",
          "guide/v2_forecasting/#models-22"),
    Topic("sarimax", "SARIMAX (seasonal ARIMA with climate)",
          (r"\bsarimax\b", r"\bs?arima\b", r"\bbox[- ]jenkins\b", r"\btime[- ]series model\b"),
          "The classic time-series model used in many climate-and-disease studies. It forecasts from the series' "
          "own recent months and its usual yearly pattern, adjusted for climate a few months earlier. ClimAID "
          "includes it as a familiar benchmark, tested and given likely ranges exactly like the other models. "
          "It is optional, not a default.",
          "SARIMAX(p, d, q)(P, D, Q)_12 on log(1 + cases) with standardised temperature, rainfall, humidity and ENSO "
          "at a single lag L as regressors (training-period statistics only; a constant when undifferenced). "
          "Tuning samples n_trials configurations from p, q in {0, 1, 2}, d, D in {0, 1} (at most one difference), "
          "P, Q in {0, 1} and L in {0, ..., 3}, scored by expanding-window CV inside the training period "
          "(one-month-ahead MAE on the case scale, 6-month folds); the default (1, 0, 0)(1, 0, 0)_12, L = 1 is kept "
          "unless beaten. Quantiles come from the Gaussian forecast distribution on the log scale, transformed "
          "back (exact for quantiles), then calibrated on the hindcasts like every model. Monthly data only, at "
          "least 30 months; if it cannot be fitted it is left out with a warning. Synthetic benchmark (WIS vs "
          "baseline): 0.63 seasonal, 0.58 non-seasonal (best model), 0.84 realistic (worse than Poisson, 0.63).",
          "guide/v2_forecasting/#models-22"),
    Topic("residuals", "Residual learning and the ensemble",
          (r"\bresidual", r"\bensemble\b", r"\bstack(ing|ed)?\b", r"\bcombined forecast\b"),
          "Each machine-learning model has a partner that learns from its mistakes in the past and corrects them. "
          "All the selected models are then combined into an ensemble by taking their middle value, which is often "
          "more reliable than any single model.",
          "Each ML model is paired with a residual model trained on its time-ordered out-of-fold errors (temporal "
          "residual stacking; no future data in any fold), and its intervals come from the out-of-fold error "
          "distribution. The ensemble takes the per-quantile median of all selected models' forecasts: robust, and "
          "with no weights to fit, so it cannot be tuned to the test period.",
          "guide/v2_forecasting/#models-22"),
    Topic("tuning", "Tuning",
          (r"\btun(e|ing)\b", r"\boptuna\b", r"\bhyperparameter", r"\btrials?\b", r"\boptimi[sz]ation trials\b"),
          "Before use, each machine-learning model's settings are adjusted automatically, using only data from "
          "before the forecast start. You choose how much effort: Fast, Balanced or Deep. More effort is not "
          "reliably more accurate; Balanced is a good default.",
          "Compulsory Optuna tuning per ML model: Fast 10 trials / 2 folds, Balanced 30 / 3, Deep 80 / 4, or a "
          "custom trial count. Candidates are scored by expanding-window, time-ordered cross-validation inside the "
          "training period (mean absolute error of one-month-ahead predictions). The defaults are scored on the same "
          "folds and kept if no candidate beats them. Hindcasts re-tune at every origin. On the synthetic benchmark "
          "v2 accuracy stopped improving at about 10-30 trials and was sometimes worse at 80.",
          "guide/tuning/"),
    Topic("hindcasts", "Backtests (hindcasts)",
          (r"\bhindcasts?\b", r"\bbacktests?\b", r"\bpast tests?\b", r"\brolling origin"),
          "ClimAID pretends to be at earlier dates, makes forecasts using only what was known then, and checks them "
          "against what actually happened. This shows how well each model would have done, and is used to set the "
          "likely ranges and the trust rating.",
          "Rolling-origin hindcasts (default 4 origins before the forecast start): at each origin the full process "
          "(feature fitting, tuning, model fitting) is re-run on data up to that origin, then scored on the months "
          "after it. Excluded (COVID-19) months are never scoring targets. Failed origins are recorded and reported, "
          "not silently skipped.",
          "guide/v2_forecasting/#what-happens-when-you-run-a-forecast"),
    Topic("calibration", "Calibrated likely ranges",
          (r"\bcalibrat", r"\blikely range\b", r"\bintervals?\b", r"\bcoverage\b", r"\bconformal\b", r"\buncertainty\b"),
          "A likely range should contain the real number 8 times out of 10. ClimAID checks this in the backtests "
          "and widens or narrows each model's ranges so they would have been right as often as they claim, "
          "separately for near and far months.",
          "Split-conformal calibration per lead-time band: for each model and central interval (50/80/90/95%), each "
          "hindcast month gets a normalised miss score r = (y - median) / (upper - median) above the median, or "
          "(median - y) / (median - lower) below it; the factor k is the conformal quantile of r at the nominal "
          "level, and quantiles are rescaled about the median, q' = median + k (q - median). Bands with too few "
          "hindcast months use the pooled factor; lead times beyond the hindcasts use the longest band, with a "
          "warning.", "api/v2/#interval-calibration"),
    Topic("scoring", "How forecasts are scored (WIS, RMSE, coverage)",
          (r"\bwis\b", r"\bweighted interval score\b", r"\brmse\b", r"\bmae\b", r"\bscor(e|ing)\b", r"\baccuracy measure"),
          "The main score, WIS (weighted interval score), rewards likely ranges that are narrow but still contain the real number; lower is "
          "better. Reports show it relative to the baseline: below 1.0 means better than 'same as last year'. They "
          "also show ordinary errors and how often the ranges contained the truth.",
          "Weighted interval score with the canonical (K + 1/2) normalisation over the median and the central "
          "50/80/90/95% intervals; RMSE and MAE of the median; empirical 50/80/95% interval coverage. Model "
          "comparison and the primary model use mean hindcast WIS (else held-out WIS).",
          "api/v2/"),
    Topic("trust", "The trust rating",
          (r"\btrust rating\b", r"\bgood.*moderate.*low\b", r"\bhow is the rating\b", r"\brating (works|calculated|worked out)\b"),
          "Good, Moderate or Low: one point each for the likely range containing the real number about as often as "
          "it should, the model beating the baseline in past tests, the models broadly agreeing, and the forecast "
          "looking no more than 12 months ahead. 4 points is Good, 2-3 Moderate, 0-1 Low. It describes past "
          "performance on your data, not a guarantee.",
          "Forecast: coverage of the 80% range within 65-95% of checked months; primary model's hindcast WIS below "
          "the seasonal-naive WIS; model disagreement below a threshold; horizon <= 12. Scenario: backtest not worse "
          "than repeating the historical mean; >= 80% of climate-model outcomes agreeing on the direction of change "
          "in the last decade; limited sensitivity to the response mode; clear identification of climate effects. "
          "'Not tested' when no checks exist.", "guide/reports/#the-trust-rating"),
    Topic("leakage", "Avoiding leakage (no peeking at the future)",
          (r"\bleak(age)?\b", r"\bpeek", r"\bfuture data\b", r"\btest set\b", r"\boverfit"),
          "A forecast must never use information from after its start date, or it will look better than it really "
          "is. ClimAID enforces this everywhere: in the features, the tuning, the backtests and the model choice.",
          "v2: future case observations are never predictors; climate feature statistics are fitted at or before "
          "the origin; tuning uses time-ordered CV inside training; hindcasts re-fit and re-tune per origin; "
          "the ensemble has no fitted weights. v1 (fixed in 0.4.0): trailing (not calendar-year) annual climate "
          "averages, lag and model selection on a validation split inside the training period with the test set "
          "used once, CMIP6 projection features matching the training definitions. Automated tests check each of "
          "these.", "guide/status/"),
    Topic("scenarios", "Climate scenario outlook",
          (r"\bscenario outlook\b", r"\bclimate (change )?outlook\b", r"\bprojections?\b", r"\bcmip6\b", r"\blong[ -]term\b",
           r"\bhow (are|is) (the )?(outlook|projection)"),
          "To ask 'what if the climate changes', ClimAID uses many climate models and emissions pathways. Each "
          "climate model is first adjusted to match your district's observed climate. The next year comes from the "
          "normal forecast; after that, a model linking cases to climate alone takes over. Results from all climate "
          "models are pooled, so the ranges include their disagreement.",
          "project_v2: (1) bias correction per climate model to the baseline-period monthly climatology (rainfall "
          "scaled, other variables shifted; projected change kept); (2) months 1-12 from the v2 forecast driven by "
          "each model's corrected climate; (3) months 13-24 linear hand-over; (4) long-term climate-only response "
          "model refitted on resampled years (60 bootstraps) for statistical uncertainty; (5) pooling across climate "
          "models; (6) sensitivity run with the other response mode, a backtest on held-out years (default 5), and a "
          "tree-model comparison reporting the share of months outside the training climate.",
          "guide/scenarios/#how-the-outlook-is-built"),
    Topic("bias_correction", "Bias correction of climate models",
          (r"\bbias[ -]correct", r"\bclimate models? (are|is) (adjusted|corrected)\b", r"\bdelta method\b"),
          "Climate models get local climate somewhat wrong, for example a bit too warm. ClimAID shifts each one so "
          "that its past matches your district's observed climate month by month, and keeps the change it projects.",
          "Per climate model and variable: the baseline-period monthly mean is matched to the observed monthly "
          "climatology, additively for temperature, humidity and ENSO and multiplicatively for rainfall; the "
          "model's projected change relative to its own baseline is preserved.",
          "guide/scenarios/#how-the-outlook-is-built"),
    Topic("response", "Seasonal vs anomaly climate response",
          (r"\bresponse mode\b", r"\bseasonal response\b", r"\banomaly response\b", r"\bresponse=", r"\bsensitivity run\b"),
          "The long-term model can learn climate effects from the seasonal cycle (assuming the yearly rise and fall "
          "of cases is largely climate-driven), or only from year-to-year differences (more cautious). The report "
          "runs the other choice too, so you can see how much the results depend on it.",
          "response='seasonal' (default) identifies climate coefficients from the full seasonal and interannual "
          "variation; response='anomaly' only from deviations from the monthly climatology, with seasonality "
          "absorbed by month effects. sensitivity=True runs the alternative (long-term only) and reports the "
          "difference.", "guide/scenarios/#main-options"),
    Topic("lag_selection", "Lag structure in the outlook",
          (r"\blag[ _]selection\b", r"\blag structure\b", r"\bv1 lags\b", r"\bwhich lags\b"),
          "Temperature, rain and humidity often rise and fall together, so a single district's data can't always "
          "tell which one matters, or with what delay. By default ClimAID averages all lag choices that fit about "
          "equally well, rather than betting on one.",
          "lag_selection: 'ensemble' (default) averages all lag structures cross-validating within 2% of the best; "
          "'v1' runs v1's lag search (in v1_mode) on data before the origin; 'distributed' uses smooth lag curves; "
          "'auto' keeps the single best; 'all' uses lags 0-3 for every variable. On synthetic data no single "
          "choice was best everywhere; the averaged default was most reliable when districts are pooled.",
          "guide/scenarios/#choosing-the-lag-structure"),
    Topic("pooling", "Pooling several districts",
          (r"\bpool(ing|ed)?\b", r"\bextra[ _]districts\b", r"\bseveral districts\b", r"\bmultiple districts\b"),
          "Learning the climate response from several districts at once helps separate the effects of variables "
          "that move together in one place but not in another. On synthetic data this recovered the true warming "
          "effect much better than a single district.",
          "extra_districts in project_v2: a joint long-term model with shared climate coefficients and "
          "district-specific intercepts. Synthetic six-district test: within 3 percentage points of the true change "
          "in three of three trials, versus about half the true effect for single districts. Available from Python "
          "and the dashboard; this assistant runs one district.",
          "guide/scenarios/#known-limitations-read-before-using"),
    Topic("thermal", "Thermal-suitability curve",
          (r"\bthermal\b", r"\btemperature curve\b", r"\bmordecai\b", r"\bsuitability\b", r"\boptimum temperature\b"),
          "Mosquito-borne transmission rises with temperature up to an optimum and then falls. You can ask ClimAID "
          "to make its temperature effect follow such a curve, so extreme future heat is not extrapolated as ever "
          "more transmission.",
          "temperature_curve: a unimodal suitability in [0, 1], zero at or below T_min and at or above T_max, one at "
          "T_opt, asymmetric parabola on each side; the model still estimates how strongly suitability affects "
          "cases. Preset 'aedes_aegypti_mordecai2017' uses 17.8 / 29.1 / 34.6 °C, from Mordecai et al. (2017). It is not a re-implementation of a "
          "published R0(T) model, and monthly means smooth out daily extremes.", "api/v2/#scenario-outlook"),
    Topic("v1", "ClimAID v1 (legacy pipeline)",
          (r"\bv1\b", r"\blegacy\b", r"\bstacked model\b", r"\bc-dsi\b", r"\blag optimi[sz]"),
          "The original ClimAID: it searches for the best climate delays, then trains a three-stage model (a main "
          "model, a second that corrects its errors, and a final adjustment) and gives single-number predictions "
          "and CMIP6 projections. v2 is recommended; v1 results from 0.1.x were optimistic and should be rerun.",
          "v1: climate features at lags (humidity, temperature and rainfall 0-3, ENSO 0-12) plus trailing 12-month "
          "(YA_*) and 10-year (MA_*) means. Stage 1 screens every lag configuration with a base model and keeps the "
          "best (100 - percentile)% capped at top_k; Stage 2 tunes base, residual and correction models with Optuna. "
          "Selection uses the last 20% of the training period; the test set is used once. Presets Fast / Balanced / "
          "Deep = 50 / 200 / 500 trials. Reports: deterministic C-DSI or a local LLM.", "api/model/"),
    Topic("limitations", "Limitations",
          (r"\blimitations?\b", r"\bweakness", r"\bcaveats?\b", r"\bwhat (can't|cannot) (it|climaid)\b",
           r"\bdownsides?\b"),
          "ClimAID has been checked only on synthetic data so far. Climate cannot predict new virus strains, "
          "reporting changes, testing campaigns or mosquito control. Single-district climate projections can "
          "understate warming effects, and projections assume the climate-disease relationship and everything else "
          "stay as in the past.",
          "Not yet validated on real surveillance data or real CMIP6 projections; synthetic data flatter the models; "
          "non-seasonal series are harder (tree models were worse than the baseline there); single-district outlooks "
          "understated a strong warming effect by about half; distributed-lag outlooks can be unstable; extrapolation "
          "beyond the training climate is reported.", "guide/status/#known-limitations"),
]

_BY_KEY = {t.key: t for t in TOPICS}
MENU_WORDS = re.compile(r"^(methods?|methodology|all methods|explain (the |all )?methods|how (does|do) (it|this|the method) work|"
                        r"what methods|list (of )?topics|topics)\b")
MORE_WORDS = re.compile(r"^(more|more detail[s]?|tell me more|technical( detail[s]?| version)?|in detail|go deeper|details?)\W*$")
RUN_WORDS = re.compile(r"\b(how was (this|my|the) (forecast|result|outlook|run) (made|done|produced|calculated)|"
                       r"(method|methods) (of|for|used in|behind) (this|my|the) (run|forecast|result|outlook)|"
                       r"explain (this|my|the) (run|forecast|method)|what did you (do|run)|how did you get)")


def find_topic(text: str) -> Topic | None:
    """The topic whose pattern matches the most specific (longest) part of the text."""
    q = text.lower()
    best, best_len = None, 0
    for t in TOPICS:
        for p in t.patterns:
            m = re.search(p, q)
            if m and len(m.group(0)) > best_len:
                best, best_len = t, len(m.group(0))
    return best


def topic(key: str) -> Topic:
    return _BY_KEY[key]


def menu() -> str:
    lines = ["Here are the methods I can explain (type a number, or ask about any of them):"]
    lines += [f"  {i}. {t.title}" for i, t in enumerate(TOPICS, 1)]
    return "\n".join(lines)


def describe(t: Topic, detailed: bool = False, base: str = "https://sam-as.github.io/ClimAID/") -> str:
    body = t.detail if detailed else t.plain
    head = f"{t.title}{' (technical detail)' if detailed else ''}:"
    tail = "" if detailed else "\n\nSay \"more detail\" for the technical version."
    link = f"\n\nDocumentation: {base}{t.page}" if t.page else ""
    return f"{head}\n{body}{tail}{link}"


# --------------------------------------------------------------------------- this run

def explain_run(meta: dict, hindcast_metrics=None, primary: str | None = None, source: str | None = None) -> str:
    """How a particular v2 forecast was made, from its metadata."""
    from climaid.reporting_plain import MODEL_PLAIN
    name = lambda m: MODEL_PLAIN.get(m, m)
    lines = ["How this forecast was made:"]
    o = pd.Timestamp(meta["forecast_origin"]) if meta.get("forecast_origin") is not None else None
    if o is not None:
        lines.append(f"  - Data used: {meta.get('n_training_observations', '?')} months of cases up to {o:%B %Y}, with "
                     f"climate ({', '.join(meta.get('climate_variables', []))}).")
    if meta.get("forecast_start") is not None:
        lines.append(f"  - Forecast: {meta.get('horizon')} months, {pd.Timestamp(meta['forecast_start']):%B %Y} to "
                     f"{pd.Timestamp(meta['forecast_end']):%B %Y}.")
    models = [m for m in meta.get("models", [])]
    if models:
        lines.append("  - Models compared: " + ", ".join(name(m) for m in models) + ", and their combined forecast.")
    tuning = meta.get("tuning") or {}
    if tuning:
        tuned = [m for m, v in tuning.items() if isinstance(v, dict) and v.get("status") == "tuned"]
        kept = [m for m, v in tuning.items() if isinstance(v, dict) and v.get("status") not in (None, "tuned")]
        preset = next((v.get("preset") for v in tuning.values() if isinstance(v, dict) and v.get("preset")), None)
        txt = f"  - Tuning{f' ({preset})' if preset else ''}: "
        parts = []
        if tuned:
            parts.append("better settings found for " + ", ".join(name(m) for m in tuned))
        if kept:
            parts.append("default settings kept for " + ", ".join(name(m) for m in kept) + " (tuning did not beat them)")
        lines.append(txt + ("; ".join(parts) if parts else "done") + ".")
    if hindcast_metrics is not None and len(hindcast_metrics) and "origin" in hindcast_metrics:
        origins = sorted(pd.to_datetime(hindcast_metrics["origin"]).unique())
        lines.append(f"  - Backtests from {len(origins)} earlier start date(s) ("
                     + ", ".join(f"{pd.Timestamp(x):%b %Y}" for x in origins) + ").")
    cal = meta.get("interval_calibration")
    if cal:
        lines.append(f"  - Likely ranges calibrated from the backtests ({cal.get('method', 'split-conformal')}).")
    if meta.get("excluded_months"):
        lines.append(f"  - COVID-19 period: {meta.get('excluded_period')} ({meta['excluded_months']} months) replaced by "
                     "typical values in the training history and not used for scoring.")
    if meta.get("population_available") is False and "renewal" in models:
        lines.append("  - The transmission model ran in relative-incidence mode (no population at risk was given).")
    if primary:
        why = " because it did best in the backtests" if source == "past tests" else (
            f" because it did best on {source}" if source else "")
        lines.append(f"  - Shown in the summary: the {name(primary)}{why}.")
    lines.append("\nAsk about any step, e.g. \"explain tuning\", \"explain calibration\", or type \"methods\" for all topics.")
    return "\n".join(lines)
