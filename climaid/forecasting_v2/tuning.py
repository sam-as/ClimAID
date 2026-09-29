"""Model catalogue and compulsory, leakage-safe hyperparameter tuning for
ClimAID v2 machine-learning forecasters.

Tuning
------
Every v2 ML model is tuned with Optuna before it is fitted; users choose how
much effort to spend (`TUNING_PRESETS`) but cannot switch tuning off.

Leakage safety: candidate settings are scored by expanding-window,
time-ordered cross-validation *inside the training period only* (the last
`n_folds` blocks of training months are predicted from the months before
them). Hindcasts re-tune at every origin, so settings are never chosen with
data from after the point being forecast.

Guard: the default settings are always scored on the same folds, and if no
tuned candidate beats them the defaults are kept, so tuning cannot make a
model worse on this criterion.

Score: mean absolute error of one-month-ahead predictions (actual recent
cases as inputs). It is robust to outbreak spikes and valid for every model.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor, TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVR

TUNING_PRESETS = {
    "fast": {"n_trials": 10, "n_folds": 2, "label": "Fast (10 trials per model)"},
    "balanced": {"n_trials": 30, "n_folds": 3, "label": "Balanced (30 trials per model)"},
    "deep": {"n_trials": 80, "n_folds": 4, "label": "Deep (80 trials per model)"},
}
DEFAULT_TUNING = "balanced"          # tests set this to "fast" via conftest


def resolve_tuning(tuning):
    """Map a preset name or an integer trial count to (n_trials, n_folds, label).
    Tuning is compulsory: None means the default preset, never 'off'."""
    if tuning is None:
        tuning = DEFAULT_TUNING
    if isinstance(tuning, (int, np.integer)):
        n = int(tuning)
        if n < 1:
            raise ValueError("Tuning is compulsory in ClimAID v2: use at least 1 trial or a preset")
        return n, 3, f"Custom ({n} trials per model)"
    key = str(tuning).lower()
    if key in ("off", "none", "false", "0", "no"):
        raise ValueError("Tuning is compulsory in ClimAID v2; choose 'fast', 'balanced', 'deep' or a number of trials")
    if key not in TUNING_PRESETS:
        raise ValueError(f"Unknown tuning preset '{tuning}'; choose from {list(TUNING_PRESETS)} or an integer")
    p = TUNING_PRESETS[key]
    return p["n_trials"], p["n_folds"], p["label"]


# ---------------------------------------------------------------------------
# additional v2 models (scikit-learn only; no extra installs)
# ---------------------------------------------------------------------------
def _scaled(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


EXTRA_MODELS = {
    # Poisson-loss histogram boosting: fast, handles count targets natively.
    "hist_gradient_boosting": lambda p, rs: HistGradientBoostingRegressor(
        loss="poisson", random_state=rs, **{"max_iter": 300, "learning_rate": 0.05, **p}),
    # Tweedie GLM (power 1-2): overdispersed counts, log link.
    "tweedie": lambda p, rs: _scaled(TweedieRegressor(link="log", max_iter=2000, **{"power": 1.5, "alpha": 0.1, **p})),
    # GAM-style: cubic splines of each input + penalised Poisson GLM, so climate effects can curve.
    "spline_poisson": lambda p, rs: Pipeline([
        ("scale", StandardScaler()),
        ("spline", SplineTransformer(n_knots=int(p.get("n_knots", 5)), degree=3, extrapolation="linear")),
        ("model", __import__("sklearn.linear_model", fromlist=["PoissonRegressor"]).PoissonRegressor(
            alpha=p.get("alpha", 1.0), max_iter=2000))]),
    "bayesian_ridge": lambda p, rs: _scaled(BayesianRidge(**p)),
    # Robust to outbreak spikes in the training data.
    "huber": lambda p, rs: _scaled(HuberRegressor(max_iter=3000, **{"epsilon": 1.35, "alpha": 1e-3, **p})),
    "svr": lambda p, rs: _scaled(SVR(**{"C": 10.0, "epsilon": 1.0, **p})),
    "knn": lambda p, rs: _scaled(KNeighborsRegressor(**{"n_neighbors": 8, "weights": "distance", **p})),
}

# linear-type registry models are scale-sensitive; v2 wraps them with a scaler
SCALE_SENSITIVE = {"linear", "ridge", "lasso", "elasticnet", "poisson", "mlp"}

# v2 name -> key used in model_parameters.DEFAULT_PARAMS (v1 naming)
V1_PARAM_KEYS = {"random_forest": "rf", "extra_trees": "extratrees", "xgboost": "xgb", "lightgbm": "lgbm"}


def _ll(trial, name, lo, hi):
    return trial.suggest_float(name, lo, hi, log=True)


SEARCH_SPACES = {
    "ridge": lambda t: {"alpha": _ll(t, "alpha", 1e-3, 1e2)},
    "lasso": lambda t: {"alpha": _ll(t, "alpha", 1e-4, 1.0)},
    "elasticnet": lambda t: {"alpha": _ll(t, "alpha", 1e-4, 1.0), "l1_ratio": t.suggest_float("l1_ratio", .05, .95)},
    "poisson": lambda t: {"alpha": _ll(t, "alpha", 1e-4, 10.0)},
    "random_forest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
                                "max_depth": t.suggest_categorical("max_depth", [4, 6, 8, 12, None]),
                                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 8),
                                "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.6])},
    "extra_trees": lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
                              "max_depth": t.suggest_categorical("max_depth", [None, 6, 10, 20]),
                              "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 6),
                              "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5])},
    "gradient_boosting": lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
                                    "learning_rate": _ll(t, "learning_rate", .01, .2),
                                    "max_depth": t.suggest_int("max_depth", 2, 5),
                                    "subsample": t.suggest_float("subsample", .6, 1.0),
                                    "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10)},
    "hist_gradient_boosting": lambda t: {"learning_rate": _ll(t, "learning_rate", .01, .3),
                                         "max_iter": t.suggest_int("max_iter", 100, 500, step=50),
                                         "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 7, 63),
                                         "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 40),
                                         "l2_regularization": t.suggest_float("l2_regularization", 0.0, 1.0)},
    "xgboost": lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
                          "learning_rate": _ll(t, "learning_rate", .01, .2), "max_depth": t.suggest_int("max_depth", 2, 7),
                          "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
                          "subsample": t.suggest_float("subsample", .6, 1.0),
                          "colsample_bytree": t.suggest_float("colsample_bytree", .6, 1.0),
                          "reg_lambda": _ll(t, "reg_lambda", .5, 5.0)},
    "lightgbm": lambda t: {"num_leaves": t.suggest_int("num_leaves", 7, 63),
                           "learning_rate": _ll(t, "learning_rate", .01, .2),
                           "n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
                           "min_child_samples": t.suggest_int("min_child_samples", 5, 40),
                           "subsample": t.suggest_float("subsample", .6, 1.0),
                           "colsample_bytree": t.suggest_float("colsample_bytree", .6, 1.0)},
    "catboost": lambda t: {"depth": t.suggest_int("depth", 3, 8), "learning_rate": _ll(t, "learning_rate", .01, .2),
                           "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0),
                           "iterations": t.suggest_int("iterations", 200, 800, step=100)},
    "mlp": lambda t: {"hidden_layer_sizes": t.suggest_categorical("hidden_layer_sizes", [(32,), (64,), (64, 32)]),
                      "alpha": _ll(t, "alpha", 1e-5, 1e-2), "learning_rate_init": _ll(t, "learning_rate_init", 5e-4, 5e-3)},
    "tweedie": lambda t: {"power": t.suggest_float("power", 1.1, 1.9), "alpha": _ll(t, "alpha", 1e-4, 10.0)},
    "spline_poisson": lambda t: {"n_knots": t.suggest_int("n_knots", 3, 8), "alpha": _ll(t, "alpha", 1e-3, 10.0)},
    "bayesian_ridge": lambda t: {"alpha_1": _ll(t, "alpha_1", 1e-7, 1e-3), "lambda_1": _ll(t, "lambda_1", 1e-7, 1e-3)},
    "huber": lambda t: {"epsilon": t.suggest_float("epsilon", 1.1, 2.0), "alpha": _ll(t, "alpha", 1e-5, 1.0)},
    "svr": lambda t: {"C": _ll(t, "C", .1, 100.0), "epsilon": _ll(t, "epsilon", .1, 10.0),
                      "gamma": t.suggest_categorical("gamma", ["scale", "auto"])},
    "knn": lambda t: {"n_neighbors": t.suggest_int("n_neighbors", 3, 20),
                      "weights": t.suggest_categorical("weights", ["uniform", "distance"])},
}


def build_estimator(name, params=None, random_state=42):
    """Construct a v2 model: ClimAID defaults (v1 table) overridden by `params`."""
    from ..model_registry import MODEL_REGISTRY
    from ..model_parameters import DEFAULT_PARAMS
    params = dict(params or {})
    if name in EXTRA_MODELS:
        return EXTRA_MODELS[name](params, random_state)
    cls = MODEL_REGISTRY[name]
    base = dict(DEFAULT_PARAMS.get(V1_PARAM_KEYS.get(name, name), {}))
    base.update(params)
    try:
        import inspect
        if "random_seed" not in base and (
                "random_state" in inspect.signature(cls).parameters or "random_state" in cls().get_params()):
            base.setdefault("random_state", random_state)
    except Exception:
        pass
    if name in ("lightgbm",):
        base.setdefault("verbose", -1)
    if name in ("catboost",):
        base.setdefault("verbose", 0)
    if name in ("mlp",):
        base.setdefault("max_iter", 2000)
    if name in ("poisson",):
        base["max_iter"] = max(int(base.get("max_iter", 0)), 2000)   # v1 default 300 does not converge
    from ..model_registry import single_threaded
    base = single_threaded(cls, base)     # reproducible on multi-core machines (see its docstring)
    try:
        est = cls(**base)
    except TypeError:
        base.pop("random_state", None)
        est = cls(**base)
    return _scaled(est) if name in SCALE_SENSITIVE else est


def _folds(n, n_folds):
    val = max(6, min(12, n // (n_folds + 3)))
    out = []
    for k in range(n_folds):
        end = n - k * val
        start = end - val
        if start >= 24:
            out.append((start, end))
    return out[::-1]


def cv_mae(name, params, X, y, n_folds, random_state=42):
    Xv = X.to_numpy(float) if hasattr(X, "to_numpy") else np.asarray(X, float)
    errs = []
    for start, end in _folds(len(y), n_folds):
        est = build_estimator(name, params, random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xv[:start], y[:start])
            pred = np.maximum(est.predict(Xv[start:end]), 0.0)
        errs.append(np.mean(np.abs(pred - y[start:end])))
    return float(np.mean(errs)) if errs else float("nan")


def tune(name, X, y, tuning=None, random_state=42):
    """Tune `name` on (X, y) (time-ordered training data). Returns (params, info)."""
    n_trials, n_folds, label = resolve_tuning(tuning)
    info = {"preset": label, "n_trials": n_trials, "n_folds": n_folds}
    if name not in SEARCH_SPACES:
        info.update(status="no tunable settings", default_cv_mae=cv_mae(name, {}, X, y, n_folds, random_state))
        return {}, info
    if not _folds(len(y), n_folds):
        info.update(status="too little training data to tune; defaults used")
        return {}, info
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    default = cv_mae(name, {}, X, y, n_folds, random_state)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(lambda t: cv_mae(name, SEARCH_SPACES[name](t), X, y, n_folds, random_state),
                   n_trials=n_trials, show_progress_bar=False)
    best = study.best_value
    if not (np.isfinite(best) and best < default):
        info.update(status="tuned settings did not beat defaults; defaults kept",
                    default_cv_mae=default, best_cv_mae=best)
        return {}, info
    params = SEARCH_SPACES[name](optuna.trial.FixedTrial(study.best_params))
    info.update(status="tuned", default_cv_mae=default, best_cv_mae=best, best_params=params,
                improvement=float(1 - best / default) if default > 0 else float("nan"))
    return params, info
