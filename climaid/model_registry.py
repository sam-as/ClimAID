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
