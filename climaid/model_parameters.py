"""
model_parameters.py
-------------------
Expanded hyperparameter spaces for all supported ML & AI models.
"""

# ======================================================
# DEFAULT PARAMETERS (safe, fast, stable)
# ======================================================

DEFAULT_PARAMS = {

    # ---------- Linear / GLM ----------
    "linear": {},

    "ridge": {
        "alpha": 1.0,
        "solver": "auto",

    },

    "lasso": {
        "alpha": 0.1,
        "max_iter": 2000,

    },

    "elasticnet": {
        "alpha": 0.1,
        "l1_ratio": 0.5,
        "max_iter": 2000,

    },

    "poisson": {
        "alpha": 1e-4,
        "fit_intercept": True,
        "max_iter": 300,
        "tol": 1e-6
    },

    # ---------- Tree Ensembles ----------
    "rf": {
        "n_estimators": 400,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,

    },

    "extratrees": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "n_jobs": -1,

    },

    # ---------- Gradient Boosting ----------
    "xgb": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "n_jobs": -1
    },

    "lgbm": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,

    },

    "catboost": {
        "iterations": 600,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "bootstrap_type": "Bayesian",
        "random_seed": 42,
        "verbose": False
    },

    # ---------- Neural Networks ----------
    "mlp": {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "alpha": 1e-4,
        "batch_size": 64,
        "max_iter": 600,
        "early_stopping": True,

    },
}


# ======================================================
# HYPERPARAMETER SEARCH SPACES
# ======================================================

SEARCH_SPACES = {

    # ---------- GLM ----------
    "ridge": {
        "alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]
    },

    "lasso": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1]
    },

    "elasticnet": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    },

    # ---------- Random Forest ----------
    "rf": {
        "n_estimators": [200, 400, 800, 1200],
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3, 0.6]
    },

    # ---------- ExtraTrees ----------
    "extratrees": {
        "n_estimators": [400, 800, 1200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    },

    # ---------- XGBoost ----------
    "xgb": {
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.005, 0.01, 0.03, 0.1],
        "max_depth": [3, 5, 7, 10],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0]
    },

    # ---------- LightGBM ----------
    "lgbm": {
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.005, 0.01, 0.03, 0.1],
        "min_child_samples": [5, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },

    # ---------- CatBoost ----------
    "catboost": {
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10]
    },

    # ---------- Neural Networks ----------
    "mlp": {
        "hidden_layer_sizes": [
            (64,), (128,),
            (128, 64), (256, 128),
            (256, 128, 64)
        ],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.0005, 0.001, 0.005],
        "alpha": [1e-5, 1e-4, 1e-3],
        "batch_size": [32, 64, 128]
    }
}

# ======================================================
# HYPERPARAMETER SEARCH SPACES
# ======================================================

SEARCH_SPACES = {

    # ---------- GLM ----------
    "ridge": {
        "alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]
    },

    "lasso": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1]
    },

    "elasticnet": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    },

    # ---------- Random Forest ----------
    "rf": {
        "n_estimators": [200, 400, 800, 1200],
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3, 0.6]
    },

    # ---------- ExtraTrees ----------
    "extratrees": {
        "n_estimators": [400, 800, 1200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    },

    # ---------- XGBoost ----------
    "xgb": {
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.005, 0.01, 0.03, 0.1],
        "max_depth": [3, 5, 7, 10],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0]
    },

    # ---------- LightGBM ----------
    "lgbm": {
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.005, 0.01, 0.03, 0.1],
        "min_child_samples": [5, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },

    # ---------- CatBoost ----------
    "catboost": {
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10]
    },

    # ---------- Neural Networks ----------
    "mlp": {
        "hidden_layer_sizes": [
            (64,), (128,),
            (128, 64), (256, 128),
            (256, 128, 64)
        ],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.0005, 0.001, 0.005],
        "alpha": [1e-5, 1e-4, 1e-3],
        "batch_size": [32, 64, 128]
    }
}

