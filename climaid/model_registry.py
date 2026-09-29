"""
model_registry.py
-----------------
Central registry for all supported ML, ensemble, and neural models
used in ClimAID disease modelling.

This file defines:
- A unified MODEL_REGISTRY mapping
- Safe optional imports for advanced models
- Consistent user-facing model names

Author: Avik Kumar Sam
"""

# ======================================================
# Core sklearn models
# ======================================================

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    PoissonRegressor,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from sklearn.neural_network import MLPRegressor

from sklearn.isotonic import IsotonicRegression

# ======================================================
# Optional third-party models (safe imports)
# ======================================================

# XGBoost
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    _XGB_AVAILABLE = False

# LightGBM
try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:
    LGBMRegressor = None
    _LGBM_AVAILABLE = False

# CatBoost
try:
    from catboost import CatBoostRegressor
    _CAT_AVAILABLE = True
except ImportError:
    CatBoostRegressor = None
    _CAT_AVAILABLE = False


# ======================================================
# MODEL REGISTRY
# ======================================================
# Keys = user-facing model names
# Values = model classes
# ======================================================

MODEL_REGISTRY = {

    # -----------------------------
    # Classical / GLM models
    # -----------------------------
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "poisson": PoissonRegressor,

    # -----------------------------
    # Tree-based ensembles
    # -----------------------------
    "rf": RandomForestRegressor,
    "random_forest": RandomForestRegressor,

    "extratrees": ExtraTreesRegressor,
    "extra_trees": ExtraTreesRegressor,

    "gbr": GradientBoostingRegressor,
    "gradient_boosting": GradientBoostingRegressor,

    # -----------------------------
    # Gradient boosting (advanced)
    # -----------------------------
    **(
        {"xgb": XGBRegressor, "xgboost": XGBRegressor}
        if _XGB_AVAILABLE else {}
    ),

    **(
        {"lgbm": LGBMRegressor, "lightgbm": LGBMRegressor}
        if _LGBM_AVAILABLE else {}
    ),

    **(
        {"catboost": CatBoostRegressor}
        if _CAT_AVAILABLE else {}
    ),

    # -----------------------------
    # Neural networks (tabular)
    # -----------------------------
    "mlp": MLPRegressor,
    "neural_net": MLPRegressor,
    "nn": MLPRegressor,

    # -----------------------------
    # Isotonic calibrator
    # -----------------------------
    "isotonic" : IsotonicRegression
}


# ======================================================
# Helper utilities
# ======================================================

def list_available_models():
    """
    Return a sorted list of available model names.
    """
    return sorted(MODEL_REGISTRY.keys())


def is_model_available(model_name: str) -> bool:
    """
    Check whether a given model name is available
    (including optional dependencies).
    """
    return model_name in MODEL_REGISTRY


def single_threaded(model_cls, params: dict) -> dict:
    """Return a copy of `params` that makes `model_cls` use one thread.

    BUG FIX (reproducibility): DEFAULT_PARAMS sets n_jobs=-1 for the tree
    ensembles and boosting libraries. With more than one core, a
    RandomForestRegressor sums its trees' predictions in whatever order the
    threads finish, so predictions differ in the last few bits between runs.
    Those differences feed the residual/correction stages and the Optuna
    comparisons, so the same data and random_state gave slightly different
    lag-search results (e.g. val_rmse 3.33332 vs 3.33371) and v2 forecasts --
    on multi-core machines only, which is why it showed up in CI and not on
    one core. Parallelism is kept at the configuration level (joblib, the
    `n_jobs` argument of optimize_lags), where it does not affect results.
    Used for every model that ClimAID v1 and v2 construct. The previous guard,
    `"n_jobs" in str(model_cls)`, was always False for the same reason as
    the random_state check in climaid_model._accepts_random_state.
    """
    import inspect
    params = dict(params)
    try:
        names = set(inspect.signature(model_cls).parameters)
    except (TypeError, ValueError):
        names = set()
    if not names or "kwargs" in names:
        try:
            names |= set(model_cls().get_params())
        except Exception:
            pass
    if "thread_count" in names:          # CatBoost
        params.pop("n_jobs", None)
        params["thread_count"] = 1
    elif "n_jobs" in names:               # scikit-learn, XGBoost, LightGBM
        params["n_jobs"] = 1
    return params

