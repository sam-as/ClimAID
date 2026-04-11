import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.isotonic import IsotonicRegression

from .climate_data import ClimateData
from .model_registry import MODEL_REGISTRY, is_model_available
from .model_parameters import DEFAULT_PARAMS, SEARCH_SPACES

# ======================================================
# Bayesian optimisation objective
# ======================================================
def _optuna_objective(
    trial,
    model_name,
    X_train,
    y_train,
    random_state=None
):
    model_cls = MODEL_REGISTRY[model_name]
    search_space = SEARCH_SPACES.get(model_name, {})

    params = {}
    for k, v in search_space.items():
        v_clean = [x for x in v if x is not None]
        if not v_clean:
            continue

        if all(isinstance(x, int) for x in v_clean):
            params[k] = trial.suggest_int(k, min(v_clean), max(v_clean))
        elif all(isinstance(x, (float, int)) for x in v_clean):
            params[k] = trial.suggest_float(k, min(v_clean), max(v_clean))
        else:
            params[k] = trial.suggest_categorical(k, v_clean)

    if random_state is not None and "random_state" in str(model_cls):
        params["random_state"] = random_state

    model = model_cls(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_train)

    # training error (not test!)
    return root_mean_squared_error(y_train, preds)

# ======================================================
# Merge Params
# =====================================================
def merge_params(defaults, best_params):
    """
    Combines the default model configuration with parameters 
    discovered during the Optuna search.
    """
    final_params = defaults.copy()
    if best_params:
        for k, v in best_params.items():
            # If Optuna finds a value that is None but 
            # the default has a value, we keep the default.
            if v is not None:
                final_params[k] = v
    return final_params

# ======================================================
# Time
# =====================================================
import time
from functools import wraps

def track_time(runtime_key):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            end = time.perf_counter()

            if hasattr(self, "runtime"):
                self.runtime[runtime_key] = end - start

            self._update_total_runtime()

            return result
        return wrapper
    return decorator

# ======================================================
# Helper function for getting the best split year during residual model training
# =====================================================
def find_best_split_year(
    train_df,
    features,
    target_col,
    model_name,
    max_val_ratio=0.25,   
    min_train_size=30
):
    from sklearn.metrics import root_mean_squared_error

    df = train_df.sort_values("time").reset_index(drop=True)
    years = sorted(df["Year"].unique())

    best_year = None
    best_score = float("inf")

    total_n = len(df)

    for split_year in years:

        train_part = df[df["Year"] <= split_year]
        val_part   = df[df["Year"] > split_year]

        n_train = len(train_part)
        n_val   = len(val_part)

        # -------------------------------
        # CONSTRAINTS
        # -------------------------------
        if n_train < min_train_size:
            continue
        if n_val == 0:
            continue
        if (n_val / total_n) > max_val_ratio:
            continue

        # -------------------------------
        # Train + evaluate
        # -------------------------------
        X_tr = train_part[features]
        y_tr = train_part[target_col]

        X_val = val_part[features]
        y_val = val_part[target_col]

        try:
            model = MODEL_REGISTRY[model_name](**DEFAULT_PARAMS.get(model_name, {}))
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, preds)

            if rmse < best_score:
                best_score = rmse
                best_year = split_year

        except Exception:
            continue

    # -------------------------------
    # FALLBACK (IMPORTANT)
    # -------------------------------
    if best_year is None:
        best_year = years[int(len(years) * 0.8)]

    return best_year

# ======================================================
# Parallel evaluation unit (lags + triple stacking)
# ======================================================
def _evaluate_configuration(
    features,
    base_model,
    residual_model,
    correction_model,
    train_df,
    test_df,
    target_col,
    n_trials,
    random_state=None,
):
    import optuna
    import random
    import numpy as np
    from sklearn.metrics import r2_score, root_mean_squared_error

    # Worker-level seeding
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    # -------------------------------
    # YEAR-BASED SPLIT
    # -------------------------------
    years = train_df["Year"].values

    # choose split year (IMPORTANT)
    train_df = train_df.sort_values("time").reset_index(drop=True)

    split_year = find_best_split_year(
        train_df,
        features,
        target_col,
        base_model,
        max_val_ratio=0.2
    )

    train_mask = years <= split_year
    val_mask = years > split_year

    X_tr = X_train[train_mask]
    X_val = X_train[val_mask]

    y_tr = y_train[train_mask]
    y_val = y_train[val_mask]

    # ======================================================
    # ---------- Base model (Tuning + Fit) ----------
    # ======================================================
    base_defaults = DEFAULT_PARAMS.get(base_model, {})
    best_base_params = base_defaults.copy()

    if base_model in SEARCH_SPACES:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state) if random_state else None
        )
        study.optimize(
            lambda t: _optuna_objective(t, base_model, X_train, y_train, random_state),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_base_params = merge_params(base_defaults, study.best_params)

    base_params = best_base_params.copy()
    if random_state is not None and "random_state" in str(MODEL_REGISTRY[base_model]):
        base_params["random_state"] = random_state

    base = MODEL_REGISTRY[base_model](**base_params)
    base.fit(X_train, y_train)
    y_base_train = base.predict(X_train)
    y_base_test = base.predict(X_test)

    # ======================================================
    # ---------- Residual model ----------
    # ======================================================
    resid_train = y_train - y_base_train
    res_defaults = DEFAULT_PARAMS.get(residual_model, {})
    best_res_params = res_defaults.copy()

    if residual_model in SEARCH_SPACES:
        res_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state) if random_state else None
        )

        def _res_objective(t):
            model_cls = MODEL_REGISTRY[residual_model]
            search_space = SEARCH_SPACES.get(residual_model, {})
            params = {}
            # FIXED: Corrected dictionary lookup logic
            for k, v in search_space.items():
                v_clean = [x for x in v if x is not None]
                if all(isinstance(x, int) for x in v_clean):
                    params[k] = t.suggest_int(k, min(v_clean), max(v_clean))
                elif all(isinstance(x, (float, int)) for x in v_clean):
                    params[k] = t.suggest_float(k, min(v_clean), max(v_clean))
                else:
                    params[k] = t.suggest_categorical(k, v_clean)

            if random_state is not None and "random_state" in str(model_cls):
                params["random_state"] = random_state

            model = model_cls(**params)

            y_base_tr = y_base_train[train_mask]
            y_base_val = y_base_train[val_mask]

            resid_tr = y_tr - y_base_tr
            model.fit(X_tr, resid_tr)
            res_preds = model.predict(X_val)

            return root_mean_squared_error(y_val, y_base_val + res_preds)

        res_study.optimize(_res_objective, n_trials=n_trials, show_progress_bar=False)
        best_res_params = merge_params(res_defaults, res_study.best_params)

    res_params = best_res_params.copy()
    if random_state is not None and "random_state" in str(MODEL_REGISTRY[residual_model]):
        res_params["random_state"] = random_state

    res = MODEL_REGISTRY[residual_model](**res_params)
    res.fit(X_train, resid_train)
    y_res_train = res.predict(X_train)
    y_res_test = res.predict(X_test)

    # ======================================================
    # ---------- Correction model ----------
    # ======================================================
    y_train_comb = y_base_train + y_res_train
    y_test_comb = y_base_test + y_res_test
    X_corr_train = np.asarray(y_train_comb).reshape(-1, 1)
    X_corr_test = np.asarray(y_test_comb).reshape(-1, 1)

    # Calculate baseline R2 before correction to prevent degradation
    baseline_rmse = root_mean_squared_error(y_test, y_test_comb)

    if correction_model in [None, "none", "base_only"]:
        final_test = y_test_comb
        corr_final_params = {}
    
    elif correction_model == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        corr_model = IsotonicRegression(out_of_bounds="clip")
        corr_model.fit(y_train_comb, y_train) 
        final_test = corr_model.predict(y_test_comb)
        corr_final_params = {}

    else:
        corr_defaults = DEFAULT_PARAMS.get(correction_model, {})
        best_corr_params = corr_defaults.copy()

        if correction_model in SEARCH_SPACES:
            corr_study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state) if random_state else None
            )

            def _corr_objective(t):
                model_cls = MODEL_REGISTRY[correction_model]
                search_space = SEARCH_SPACES.get(correction_model, {})
                params = {}
                # FIXED: Corrected dictionary lookup logic
                for k, v in search_space.items():
                    v_clean = [x for x in v if x is not None]
                    if all(isinstance(x, int) for x in v_clean):
                        params[k] = t.suggest_int(k, min(v_clean), max(v_clean))
                    elif all(isinstance(x, (float, int)) for x in v_clean):
                        params[k] = t.suggest_float(k, min(v_clean), max(v_clean))
                    else:
                        params[k] = t.suggest_categorical(k, v_clean)

                if random_state is not None and "random_state" in str(model_cls):
                    params["random_state"] = random_state

                model = model_cls(**params)

                X_corr_tr = X_corr_train[train_mask]
                X_corr_val = X_corr_train[val_mask]

                y_tr_corr = y_train[train_mask]
                y_val_corr = y_train[val_mask]

                model.fit(X_corr_tr, y_tr_corr)
                preds = model.predict(X_corr_val)

                return root_mean_squared_error(y_val_corr, preds)

            corr_study.optimize(_corr_objective, n_trials=n_trials, show_progress_bar=False)
            best_corr_params = merge_params(corr_defaults, corr_study.best_params)

        corr_params = best_corr_params.copy()
        if random_state is not None and "random_state" in str(MODEL_REGISTRY[correction_model]):
            corr_params["random_state"] = random_state

        corr_model = MODEL_REGISTRY[correction_model](**corr_params)
        corr_model.fit(X_corr_train, y_train)
        final_test = corr_model.predict(X_corr_test)
        corr_final_params = best_corr_params

    # Final Metric Calculation with Safety Check
    actual_rmse = root_mean_squared_error(y_test, final_test)
    actual_r2 = r2_score(y_test, final_test)  # keep for reporting

    # SAFETY CHECK: If the correction stage made the model worse, 
    # revert to the stacked Base+Residual results.
    if actual_rmse > baseline_rmse:
        rmse = baseline_rmse
        r2 = r2_score(y_test, y_test_comb)
        final_correction_name = "none"
        final_corr_params = {}
    else:
        rmse = actual_rmse
        r2 = actual_r2
        final_correction_name = correction_model
        final_corr_params = corr_final_params

    return {
        "base_model": base_model,
        "residual_model": residual_model,
        "correction_model": final_correction_name,
        "features": features,
        "r2": r2,
        "rmse": rmse,
        "base_params": best_base_params,
        "residual_params": res_params,
        "correction_params": final_corr_params,
    }

# ======================================================
# Main DiseaseModel class
# ======================================================
class DiseaseModel:
    """
    Climate-informed disease prediction and projection model.

    This class implements an end-to-end pipeline for modeling climate-sensitive
    diseases (e.g., dengue, malaria) using historical climate data, lagged features,
    and hybrid machine learning approaches.

    The pipeline includes:
        1. Data ingestion (disease + climate)
        2. Climate–disease data merging
        3. Lagged feature optimisation (Bayesian search)
        4. Model training (stacked learning framework)
        5. Prediction and evaluation
        6. Climate-based future projections (via DiseaseProjection)

    Parameters
    ----------

    district : str
        District identifier (must match supported naming convention, e.g.,
        'IND_Pune_MAHARASHTRA').

    disease_file : str or Path :
        Path to the disease dataset file. Supported formats include:
        CSV (.csv), Excel (.xlsx), and Parquet (.parquet).

    target_col : str, default="Count" : 
        Column name representing disease case counts.

    random_state : int, default=42 : 
        Random seed for reproducibility across the pipeline.

    disease_name : str, optional :
        Name of the disease (e.g., "Dengue", "Malaria").
        Used for reporting and visualization.

    Attributes
    ----------

    df_disease : pandas.DataFrame :
        Raw disease dataset.

    df_climate_hist : pandas.DataFrame :
        Historical climate data.

    df_climate_proj : pandas.DataFrame :
        Climate projection data (CMIP6).

    df_merged : pandas.DataFrame :
        Combined dataset with aligned disease and climate variables.

    train_df : pandas.DataFrame :
        Training subset after domain-aware split.

    test_df : pandas.DataFrame :
        Testing subset.

    lag_search_results : dict
        Results from lag optimisation process.

    best_config : dict
        Best-performing lag configuration.

    final_models : dict
        Trained models (base, residual, calibration).

    runtime : dict
        Execution time tracking for each pipeline stage.

    Notes
    -----

    - The modeling framework uses
        - Base model: captures main signal (e.g., XGBoost)
        - Residual model: models unexplained variation (e.g., Random Forest)
        - Calibration model: adjusts predictions (e.g., isotonic regression)
    
    """

    def __init__(
        self,
        district: str,
        disease_file: str,
        target_col: str = "Count",
        random_state: int = 42,
        disease_name: str | None = None,
        weather_file: str | None = None,        
        projection_file: str | None = None,
    ):
        """
        Initialize the DiseaseModel pipeline.

        This automatically:
        1. Loads disease data
        2. Fetches climate data (historical + projections)
        3. Merges datasets for modeling

        Parameters
        ----------

        district : str
            District identifier.

        disease_file : str
            Path to disease dataset.

        target_col : str, optional
            Target variable column.

        random_state : int, optional
            Seed for reproducibility.

        disease_name : str, optional
            Name of disease.

        weather_file : path, optional
            Used in ClimAID Global Mode.

        projection_file : path, optional
            Used in ClimAID Global Mode.
        """

        # -----------------------------
        # Core configuration
        # -----------------------------
        self.random_state = random_state
        self._set_global_seed()
        self.district = district
        self.disease_file = Path(disease_file)
        self.target_col = target_col
        self.disease_name = disease_name
        self.weather_file = weather_file
        self.projection_file = projection_file

        # -----------------------------
        # Load disease data
        # -----------------------------
        self.df_disease = self._load_disease_data()

        # -----------------------------
        # Load climate data
        # -----------------------------
        self.weather_file = weather_file
        self.projection_file = projection_file
        
        clim = ClimateData(
            weather_file=self.weather_file,
            projection_file=self.projection_file
        )

        self.df_climate_hist = clim.get_historical(district=district)
        self.df_climate_proj = clim.get_projection(district=district)

        # -----------------------------
        # Merge datasets
        # -----------------------------
        self.df_merged = self._merge_data()

        # -----------------------------
        # Model placeholders
        # -----------------------------
        self.train_df = None
        self.test_df = None
        self.lag_search_results = None
        self.best_config = None
        self.final_models = {}

        # -----------------------------
        # Runtime tracking
        # -----------------------------
        self.runtime = {
            "lag_optimization_seconds": None,
            "training_seconds": None,
            "prediction_seconds": None,
            "report_generation_seconds": None,
            "total_pipeline_seconds": None,
        }

        self._pipeline_start_time = None

    # --------------------------------------------------
    # Global Random Seed
    # --------------------------------------------------
    def _set_global_seed(self):
        """Set global random seeds for full pipeline reproducibility."""
        if self.random_state is None:
            return  # Do not override global RNG if user did not specify seed

        np.random.seed(self.random_state)
        random.seed(self.random_state)
        os.environ["PYTHONHASHSEED"] = str(self.random_state)

    # --------------------------------------------------
    # Runtime correcter
    # --------------------------------------------------
    def _update_total_runtime(self):
        """Compute corrected total runtime excluding idle time."""

        components = [
            self.runtime.get("lag_optimization_seconds"),
            self.runtime.get("training_seconds"),
            self.runtime.get("prediction_seconds"),
            self.runtime.get("report_generation_seconds"),
        ]

        # Remove None values
        valid_times = [t for t in components if t is not None]

        self.runtime["Corrected_total_pipeline_seconds"] = sum(valid_times)

    # --------------------------------------------------
    # Load disease data
    # --------------------------------------------------
    def _load_disease_data(self):

        from difflib import get_close_matches

        suffix = self.disease_file.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(self.disease_file)

        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(self.disease_file)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        df = df.copy()
        original_cols = df.columns.tolist()

        # standardize for matching
        df.columns = df.columns.str.strip().str.lower()

        # =====================================================
        # TARGETS
        # =====================================================
        targets = {
            "time": ["date", "time", "datetime"],
            "Count": ["cases", "case", "count"]
        }

        rename_dict = {}
        mapping_log = {}

        # =====================================================
        # MATCHING
        # =====================================================
        for col in df.columns:
            for target, aliases in targets.items():

                # exact match
                if col in aliases:
                    rename_dict[col] = target
                    mapping_log[col] = (target, "exact")
                    break

                # fuzzy match
                match = get_close_matches(col, aliases, n=1, cutoff=0.9)
                if match:
                    rename_dict[col] = target
                    mapping_log[col] = (target, "fuzzy")
                    break

        df = df.rename(columns=rename_dict)

        # =====================================================
        # VALIDATION REPORT
        # =====================================================
        print("\n--- COLUMN STANDARDIZATION REPORT ---")
        print("Original columns:", original_cols)

        print("\nMappings:")
        for k, v in mapping_log.items():
            print(f"{k} -> {v[0]} ({v[1]})")

        # =====================================================
        # REQUIRED CHECK
        # =====================================================
        required = ["time", "Count"]

        print("\n--- REQUIRED COLUMN CHECK ---")
        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column: {col}\n"
                    f"Supported date columns: date, time, datetime\n"
                    f"Supported case columns: cases, case, count"
                )
            else:
                print(f"{col}: OK")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["Year"] = df["time"].dt.year
        df["Month"] = df["time"].dt.month

        # dropping nan values based on cases
        df = df.dropna(subset=['Count']).reset_index(drop = True)

        if df.empty:
            raise ValueError(
                "The disease data loaded is empty. "
                "The file may be corrupted"
            )

        return df

    # --------------------------------------------------
    # Merge climate + disease
    # --------------------------------------------------
    def _merge_data(self):
        from difflib import get_close_matches
        from climaid.utils import _map_columns
        climate = self.df_climate_hist.copy()

        climate = _map_columns(climate)

        # Removed (from DiseaseModel)
        # # =====================================================
        # # STANDARDIZE COLUMN NAMES
        # # =====================================================
        # original_cols = climate.columns.tolist()
        # climate.columns = climate.columns.str.strip().str.lower()

        # # canonical names
        # standard_vars = {
        #     "mean_Rain": ["rain", "rainfall", "precipitation", "pr"],
        #     "mean_temperature": ["temp", "temperature", "tas"],
        #     "mean_SH": ["humidity", "specific_humidity", "huss"],
        #     "Nino_anomaly": ["nino", "enso", "nino34"],
        #     "time": ["date", "datetime"],
        #     "Dist_States" : ['diststates', 'dist_states', 'dist_state', 'DistStates']
        # }

        # rename_dict = {}
        # unmatched_cols = []
        # mapping_log = []

        # for col in climate.columns:
        #     matched = False

        #     for standard, aliases in standard_vars.items():
        #         # exact match
        #         if col == standard.lower() or col in aliases:
        #             rename_dict[col] = standard
        #             mapping_log.append((col, standard, "exact"))
        #             matched = True
        #             break

        #         # fuzzy match
        #         matches = get_close_matches(col, aliases, n=1, cutoff=0.8)
        #         if matches:
        #             rename_dict[col] = standard
        #             mapping_log.append((col, standard, "fuzzy"))
        #             matched = True
        #             break

        #     if not matched:
        #         unmatched_cols.append(col)

        # climate = climate.rename(columns=rename_dict)

        # # =====================================================
        # # VALIDATION REPORT
        # # =====================================================
        # print("\n--- COLUMN MAPPING REPORT ---")
        # print("Original columns:", original_cols)

        # print("\nMapped columns:")
        # for old, new, mtype in mapping_log:
        #     print(f"{old} -> {new} ({mtype})")

        # if unmatched_cols:
        #     print("\nUnmapped columns:")
        #     for col in unmatched_cols:
        #         print(f"- {col}")

        # =====================================================
        # CHECK REQUIRED COLUMNS
        # =====================================================
        required_cols = ["time"]

        print("\n--- REQUIRED COLUMN CHECK ---")
        for col in required_cols:
            if col not in climate.columns:
                raise ValueError(f"Missing required column: {col}")
            else:
                print(f"{col}: OK")

        # =====================================================
        # TIME HANDLING
        # =====================================================
        climate["time"] = pd.to_datetime(climate["time"], errors="coerce")

        invalid_time = climate["time"].isna().sum()
        print(f"\nInvalid time entries: {invalid_time}")

        climate["Year"] = climate["time"].dt.year.astype("Int16")
        climate["Month"] = climate["time"].dt.month.astype("Int8")

        # =====================================================
        # NUMERIC VARIABLES
        # =====================================================
        climate_vars = ["mean_Rain", "mean_temperature", "mean_SH", "Nino_anomaly"]

        # print("\n--- NUMERIC CONVERSION REPORT ---")

        # available_vars = []

        for col in climate_vars:
            climate[col] = pd.to_numeric(climate[col], errors="coerce")

        # =====================================================
        # CLEAN VALUES
        # =====================================================
        if "mean_Rain" in climate.columns:
            climate["mean_Rain"] = climate["mean_Rain"].clip(lower=0)

        if "mean_SH" in climate.columns:
            climate["mean_SH"] = climate["mean_SH"].clip(lower=0)

        # # =====================================================
        # # SUMMARY REPORT
        # # =====================================================
        # print("\n--- FINAL DATA SUMMARY ---")
        # print("Shape:", climate.shape)

        # for col in available_vars:
        #     missing_pct = climate[col].isna().mean() * 100
        #     print(f"{col}: missing {missing_pct:.2f}%")

        # =====================================================
        # get the annual averages
        # =====================================================
        vars = ["mean_Rain", "mean_temperature", "mean_SH"]
        annual = (
            climate.groupby("Year")[vars]
            .mean()
            .reset_index()
        )

        for col in vars:
            annual[f"YA_{col}"] = annual[col]
            annual.drop(columns=col, inplace=True)

        climate = climate.merge(annual, on="Year", how="left")

        # =====================================================
        # get the rolling monthly averages
        # =====================================================
        for col in vars:
            climate[f"MA_{col}"] = climate[col].rolling(window=120, min_periods=12).mean()

        # =====================================================
        # The climate data will have one additional year i.e., previous to the year from when the disease data has been recorded
        # =====================================================
        year_den = self.df_disease['Year'].min()
        year_clim_desired_min = year_den - 1
        year_clim_desired = self.df_disease['Year'].max() 

        climate = climate[
                        climate["Year"].between(year_clim_desired_min, year_clim_desired)
                    ].copy()

        print('The rolling monthly and yearly averages are calculated. Now, merging the disease and the climate data... ')

        climate = climate.drop_duplicates(subset=["Year", "Month"]).reset_index(drop = True)

        # =====================================================
        # Merging + Returning
        # =====================================================

        merged = pd.merge(
            self.df_disease,
            climate[
                [
                    "Year", "Month",
                    "mean_temperature", "mean_Rain",
                    "mean_SH", "Nino_anomaly", 
                    "MA_mean_temperature" , "MA_mean_Rain", "MA_mean_SH",
                    "YA_mean_temperature" , "YA_mean_Rain", "YA_mean_SH",
                ]
            ],
            on=["Year", "Month"],
            how="right"
        )

        merged["time"] = pd.to_datetime(
            dict(year=merged["Year"], month=merged["Month"], day=1)
        )

        # merged = merged.dropna(subset=['Count']).reset_index(drop=True)

        if merged.empty:
            raise ValueError(
                "Merged DataFrame is empty. This likely indicates a column/value mismatch "
                "between disease and climate data (e.g., Year/Month alignment or missing columns). \n"
                "Please verify that both datasets share overlapping Year and Month values"
            )
        else:
            print('The climate-disease data has been merged, now performing train-test split....')

        return merged

    # --------------------------------------------------
    # train/test split splitting
    # --------------------------------------------------
    def _train_test_split(
        self,
        train_year: int | None = None,
        test_year: int | None = None,
        drop_2020: bool = True,
    ):
        """
        Perform a time-aware train/test split for disease modeling.

        This method splits the merged dataset into training and testing subsets
        based on temporal ordering, ensuring no data leakage from future observations.

        Parameters
        ----------

        train_year : int or None, optional
            Last year to include in the training dataset.
            If None, the split is determined automatically.

        test_year : int or None, optional
            First year to include in the test dataset.
            If None, the split is determined automatically.

        drop_2020 : bool, default=True
            Whether to exclude data from the year 2020.
            This is useful to remove anomalies due to COVID-19 disruptions
            in disease reporting and transmission patterns.

        Returns
        -------

        None:
            Updates internal attributes:

            - self.train_df : pandas.DataFrame
                Training dataset

            - self.test_df : pandas.DataFrame
                Testing dataset

        Notes
        -----

        - The split preserves temporal ordering (no random shuffling).
        - Designed for time-series epidemiological data.
        - Avoids leakage by ensuring test data strictly follows training data.
        - If both `train_year` and `test_year` are provided, they must be consistent.

        Warning
        -------

        This is an internal method and is not intended for direct use.

        """

        if hasattr(self, "df") and self.df is not None:
            df = self.df.copy()
        else:
            df = self.df_merged.copy()

        if drop_2020:
            df = df[df["Year"] != 2020]

        # -------------------------------
        # SPLIT LOGIC
        # -------------------------------
        if train_year is None and test_year is None:
            train_df = df[df["Year"] < 2020].copy()
            test_df  = df[df["Year"] > 2020].copy()

        elif train_year is None:
            test_year = int(test_year)
            train_df = df[df["Year"] < test_year].copy()
            test_df  = df[df["Year"] >= test_year].copy()

        elif test_year is None:
            train_year = int(train_year)
            train_df = df[df["Year"] <= train_year].copy()
            test_df  = df[df["Year"] > train_year].copy()

        else:
            train_year = int(train_year)
            test_year  = int(test_year)

            if train_year >= test_year:
                raise ValueError("train_year must be earlier than test_year")

            train_df = df[df["Year"] <= train_year].copy()
            test_df  = df[df["Year"] >= test_year].copy()

        # Dropping NaN values from both train_df and test_df
        train_df = train_df.dropna(subset = ['Count']).reset_index(drop = True)
        
        test_df = test_df.dropna(subset = ['Count']).reset_index(drop = True)

        # -------------------------------
        # STORE INTERNALLY (KEY FIX)
        # -------------------------------
        self.train_df = train_df
        self.test_df  = test_df
        self.train_year = train_year
        self.test_year = test_year

        # # Optional debug
        # print("Split applied:")
        # print("Train years:", sorted(train_df["Year"].unique()))
        # print("Test years :", sorted(test_df["Year"].unique()))

        return train_df, test_df
        
    # --------------------------------------------------
    # Detect outbreaks
    # --------------------------------------------------
    def detect_historical_outbreaks(
        self,
        method: str = "zscore",
        window: int = 12,
        threshold: float = 2.0,
        date_col: str = "time",
    ):
        """
        Detect historical outbreak periods using anomaly-based thresholds.

        This method identifies unusually high disease incidence relative to
        a historical baseline using statistical anomaly detection techniques.

        Parameters
        ----------
        
        method : str, default="zscore" :
            Method used for anomaly detection. Supported options:

            - "zscore" :
                Computes standardized anomalies relative to rolling mean and
                standard deviation.

            - "percentile" :
                Flags observations exceeding a specified percentile threshold.

        window : int, default=12 : 
            Rolling window size (in time steps, typically months) used to compute
            baseline statistics such as mean and standard deviation.

        threshold : float, default=2.0 :
            Threshold for outbreak detection:

            - For "zscore":
                Number of standard deviations above the rolling mean.

            - For "percentile":
                Percentile cutoff (e.g., 0.9 for 90th percentile).

        date_col : str, default="time" : 
            Column name representing temporal ordering of the data.

        Returns
        -------

        df : pandas.DataFrame 
            DataFrame with additional columns :

            - anomaly_score : float
                - Computed anomaly metric (z-score or percentile-based)
            - outbreak_flag : int
                - Binary indicator (1 = outbreak, 0 = normal)

        Notes
        -----

        - The method assumes time-series structure in the dataset.
        - Rolling statistics are computed using past observations only
        (no future leakage).
        - Suitable for identifying epidemic spikes in climate-sensitive diseases.

        """

        df = self.df_merged.copy()
        df = df.sort_values(date_col)

        y = df[self.target_col]

        rolling_mean = y.rolling(window=window, min_periods=3).mean()
        rolling_std = y.rolling(window=window, min_periods=3).std()

        zscore = (y - rolling_mean) / (rolling_std + 1e-8)

        df["historical_outbreak_flag"] = zscore > threshold
        df["zscore"] = zscore

        self.df_outbreaks_hist = df

        return df[df["historical_outbreak_flag"]]
    
    @track_time("lag_optimization_seconds")
    def optimize_lags(
        self,
        base_models=("xgb", "rf"),
        residual_models=("rf", "extratrees"),
        correction_models=("ridge", "gbr", "isotonic"),
        n_trials=30,
        debug=False,
        n_jobs=-1,
        pruning_strategy="percentile",
        top_k=50,
        sh_range=range(0, 4),
        temp_range=range(0, 4),
        rain_range=range(0, 4),
        elnino_range=range(0, 13),
        percentile=90,
        scalar=None,
    ):
        """
        Optimize climate lag structures and model configurations using Bayesian search.

        This method identifies the optimal combination of lagged climate variables
        (e.g., temperature, rainfall, humidity, ENSO indices) along with the best
        model architecture using a hybrid AutoML framework.

        The optimization jointly searches over:
            - Climate lag configurations
            - Base models (signal learning)
            - Residual models (error correction)
            - Calibration models (prediction adjustment)

        Parameters
        ----------
        
        base_models : tuple of str, default=("xgb", "rf") :
            Models used to capture the primary disease–climate relationship.

            - Supported options include 
                - "xgb" : XGBoost
                - "rf" : Random Forest

        residual_models : tuple of str, default=("rf", "extratrees")
            Models used to capture residual patterns not explained by base models.

        correction_models : tuple of str, default=("ridge", "gbr", "isotonic")
            Models used for calibration or bias correction of predictions.

        n_trials : int, default=30
            Number of optimization iterations (Bayesian search trials).

        debug : bool, default=False
            If True, enables verbose logging and debugging output.

        n_jobs : int, default=-1
            Number of parallel jobs:
            - -1 uses all available CPU cores

        pruning_strategy : str, default="percentile"
            Strategy for pruning poor-performing configurations during search.

            - Options:
                - "percentile" : keeps top-performing configurations
                - "threshold" : uses fixed cutoff

        top_k : int, default=50
            Number of top configurations retained after pruning.

        sh_range : iterable, default=range(0, 4)
            Lag range for specific humidity (months).

        temp_range : iterable, default=range(0, 4)
            Lag range for temperature.

        rain_range : iterable, default=range(0, 4)
            Lag range for rainfall.

        elnino_range : iterable, default=range(0, 13)
            Lag range for ENSO (El Niño) index.

        percentile : int, default=90
            Percentile threshold used for pruning or model selection.

        scalar : object, optional
            Optional feature scaling object (e.g., StandardScaler).

        Returns
        -------

        tuple :
            (feature_metadata, lag_search_results, best_config)

            - feature_metadata : dict
                Information about generated lagged features

            - lag_search_results : pandas.DataFrame
                Performance of all evaluated configurations

            - best_config : dict
                Best-performing configuration (lags + model combination)

        Notes
        -----
        - Uses Bayesian optimization for efficient hyperparameter search.
        - Supports parallel evaluation of configurations.
        - Designed for climate-sensitive disease systems with delayed effects.
        - Lag ranges should reflect domain knowledge (e.g., incubation periods).

        """

        import numpy as np
        import pandas as pd
        import time
        from joblib import Parallel, delayed
        from sklearn.metrics import r2_score, root_mean_squared_error

        from sklearn.preprocessing import (
            StandardScaler,
            MinMaxScaler,
            MaxAbsScaler,
            RobustScaler,
            QuantileTransformer,
            PowerTransformer,
            Normalizer,
        )

        # --------------------------------------------------
        # Scaler registry
        # --------------------------------------------------
        scaler_dict = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "maxabs": MaxAbsScaler(),
            "robust": RobustScaler(),
            "quantile": QuantileTransformer(output_distribution="normal"),
            "power": PowerTransformer(),
            "normalize": Normalizer(),
            None: None,
        }

        if scalar not in scaler_dict:
            raise ValueError(
                f"Invalid scaler '{scalar}'. Choose from {list(scaler_dict.keys())}"
            )

        scaler_obj = scaler_dict[scalar]

        # --------------------------------------------------
        # Load dataset
        # --------------------------------------------------
        if hasattr(self, "df_merged") and self.df_merged is not None:
            df = self.df_merged.copy()
        else:
            raise RuntimeError("Merged dataset not available.")

        # --------------------------------------------------
        # Lag definitions
        # --------------------------------------------------
        lag_defs = {
            "mean_SH": sh_range,
            "mean_temperature": temp_range,
            "mean_Rain": rain_range,
            "Nino_anomaly": elnino_range,
        }

        # --------------------------------------------------
        # Generate lag features
        # --------------------------------------------------
        for var, lags in lag_defs.items():
            for lag in lags:
                df[f"{var}_lag{lag}"] = df[var].shift(lag)

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        # --------------------------------------------------
        # ENSO interaction terms
        # --------------------------------------------------
        climate_vars = ["mean_SH", "mean_temperature", "mean_Rain"]

        for clim in climate_vars:
            for lag_c in lag_defs[clim]:
                for lag_n in lag_defs["Nino_anomaly"]:

                    clim_col = f"{clim}_lag{lag_c}"
                    nino_col = f"Nino_anomaly_lag{lag_n}"

                    inter_col = f"{clim}_lag{lag_c}_x_Nino_anomaly_lag{lag_n}"
                    df[inter_col] = df[clim_col] * df[nino_col]

        # --------------------------------------------------
        # Final dataframe with engineered features
        # --------------------------------------------------
        self.df = df.dropna().reset_index(drop=True)

        # --------------------------------------------------
        # Train test split
        # --------------------------------------------------
        self._train_test_split(
            train_year=getattr(self, "train_year", None),
            test_year=getattr(self, "test_year", None)
        )

        print('Initial Train Test Split completed...')
        print('Now performing lag optimisation....')
        
        # --------------------------------------------------
        # Fit scaler
        # --------------------------------------------------
        if scaler_obj is not None:

            numeric_cols = [
                c for c in self.train_df.select_dtypes(include=[np.number]).columns
                if c != "Year"
            ]

            if len(numeric_cols) > 0:

                scaler_obj.fit(self.train_df[numeric_cols])

                self.train_df.loc[:, numeric_cols] = scaler_obj.transform(
                    self.train_df[numeric_cols]
                )

                self.test_df.loc[:, numeric_cols] = scaler_obj.transform(
                    self.test_df[numeric_cols]
                )

        # --------------------------------------------------
        # Feature grid
        # --------------------------------------------------
        if debug:
            all_features = [[
                "mean_SH_lag1",
                "mean_temperature_lag2",
                "mean_Rain_lag2",
                "Nino_anomaly_lag12",
                "Year",
                "MA_mean_temperature","MA_mean_Rain","MA_mean_SH",
                "YA_mean_temperature","YA_mean_Rain","YA_mean_SH",
            ]]
        else:

            all_features = [
                [
                    f"mean_SH_lag{rh}",
                    f"mean_temperature_lag{t}",
                    f"mean_Rain_lag{r}",
                    f"Nino_anomaly_lag{n}",

                    f"mean_SH_lag{rh}_x_Nino_anomaly_lag{n}",
                    f"mean_temperature_lag{t}_x_Nino_anomaly_lag{n}",
                    f"mean_Rain_lag{r}_x_Nino_anomaly_lag{n}",

                    "Year",
                    "MA_mean_temperature","MA_mean_Rain","MA_mean_SH",
                    "YA_mean_temperature","YA_mean_Rain","YA_mean_SH",
                ]

                for rh in lag_defs["mean_SH"]
                for t in lag_defs["mean_temperature"]
                for r in lag_defs["mean_Rain"]
                for n in lag_defs["Nino_anomaly"]
            ]

        # --------------------------------------------------
        # Config grid
        # --------------------------------------------------
        configs = [
            (feats, base, res, corr)
            for feats in all_features
            for base in base_models if is_model_available(base)
            for res in residual_models if is_model_available(res)
            for corr in correction_models if is_model_available(corr)
        ]

        if len(configs) == 0:
            raise RuntimeError("No valid model configurations available.")

        # --------------------------------------------------
        # Precompute feature matrices
        # --------------------------------------------------
        feature_pool = sorted({f for feats,_,_,_ in configs for f in feats})

        X_train_full = self.train_df[feature_pool].to_numpy()
        X_test_full = self.test_df[feature_pool].to_numpy()

        y_train = self.train_df[self.target_col].to_numpy()
        y_test = self.test_df[self.target_col].to_numpy()

        feature_index = {f: i for i, f in enumerate(feature_pool)}

        # --------------------------------------------------
        # Stage 1: Base screening
        # --------------------------------------------------
        def _evaluate_base_screen(idx, feats, base_model):

            try:

                seed = (
                    self.random_state + idx
                    if self.random_state is not None else None
                )

                if seed is not None:
                    np.random.seed(seed)

                cols = [feature_index[f] for f in feats]

                X_train = X_train_full[:, cols]
                X_test = X_test_full[:, cols]

                base_defaults = DEFAULT_PARAMS.get(base_model, {}).copy()

                if seed is not None and "random_state" in str(MODEL_REGISTRY[base_model]):
                    base_defaults["random_state"] = seed

                if "n_jobs" in str(MODEL_REGISTRY[base_model]):
                    base_defaults["n_jobs"] = 1

                model = MODEL_REGISTRY[base_model](**base_defaults)

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                base_rmse = root_mean_squared_error(y_test, preds)

                return {"idx": idx, "base_rmse": base_rmse}

            except Exception:

                return {"idx": idx, "base_rmse": np.inf}

        base_scores = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_base_screen)(i, feats, base)
            for i,(feats,base,_,_) in enumerate(configs)
        )

        base_df = pd.DataFrame(base_scores)

        # --------------------------------------------------
        # Hierarchical pruning
        # --------------------------------------------------
        if pruning_strategy is None:
            selected_indices = base_df["idx"].tolist()

        elif pruning_strategy == "top_k":

            k = min(top_k, len(base_df))
            selected_indices = base_df.nsmallest(k, "base_rmse")["idx"].tolist()

        elif pruning_strategy == "percentile":

            valid_scores = base_df["base_rmse"].replace(np.inf, np.nan).dropna()

            if len(valid_scores)==0:
                selected_indices = base_df["idx"].tolist()
            else:
                threshold = np.percentile(valid_scores, percentile)

                selected_indices = base_df[
                    base_df["base_rmse"] <= threshold
                ]["idx"].tolist()
        else:
            raise ValueError("pruning_strategy must be {'top_k','percentile',None}")

        if len(selected_indices)==0:
            selected_indices = base_df.nsmallest(10, "base_rmse")["idx"].tolist()

        pruned_configs = [configs[i] for i in selected_indices]

        # --------------------------------------------------
        # Stage 2: Full stacked optimization
        # --------------------------------------------------
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_configuration)(
                feats,
                base,
                res,
                corr,
                self.train_df,
                self.test_df,
                self.target_col,
                n_trials,
                (
                    self.random_state + idx
                    if self.random_state is not None else None
                ),
            )
            for idx,(feats,base,res,corr) in enumerate(pruned_configs)
        )

        self.lag_search_results = pd.DataFrame(results)

        self.best_config = (
            self.lag_search_results.sort_values("rmse", ascending=True).iloc[0]
        )

        self.scaler = scaler_obj

        self.scaled_columns = [
            c for c in self.train_df.select_dtypes(include=[np.number]).columns
            if c != "Year"
        ]

        if self._pipeline_start_time is None:
            self._pipeline_start_time = time.perf_counter()

        # --------------------------------------------------
        # Store feature metadata
        # --------------------------------------------------
        lag_map = {}
        interaction_map = []

        for feat in self.best_config["features"]:

            if "_lag" in feat and "_x_" not in feat:

                base, lag = feat.split("_lag")
                lag = int(lag)

                lag_map.setdefault(base, set()).add(lag)

            if "_x_" in feat:

                left, right = feat.split("_x_")

                # fix shorthand naming if it occurs
                if right.startswith("nino_"):
                    right = right.replace("nino_", "Nino_anomaly_")

                interaction_map.append((left, right))

        self.feature_metadata = {
            "lags": {k: sorted(v) for k, v in lag_map.items()},
            "interactions": interaction_map,
            "scaled_columns": self.scaled_columns
        }

        return self.feature_metadata, self.lag_search_results, self.best_config

    # --------------------------------------------------
    # Train final triple-stacked model
    # -------------------------------------------------
    @track_time("training_seconds")
    def train_final_model(self):
        """
        Train the final disease prediction model using the optimized feature set.

        This method fits the final model after preprocessing, feature engineering,
        and lag optimization steps have been completed. It typically uses the
        best-performing configuration identified during model selection (e.g.,
        stacked learning framework or tuned estimator).

        The trained model is stored internally and used for subsequent prediction
        and projection tasks.

        Parameters
        ----------

        None

        Returns
        -------

        self : DiseaseModel
            Returns the instance with the trained model stored internally.

        Notes
        -----

        - Assumes that data ingestion and preprocessing have already been performed.
        - Requires optimized lagged features to be available.
        - May internally split data into training and validation sets.
        - Supports integration with ensemble or stacked models if configured.
        """

        if self.best_config is None:
            raise RuntimeError("Run optimize_lags() first.")

        import numpy as np
        import pandas as pd
        from sklearn.metrics import r2_score, root_mean_squared_error

        feats = self.best_config["features"]
        X_train = self.train_df[feats]
        y_train = self.train_df[self.target_col]
        X_test = self.test_df[feats]
        y_test = self.test_df[self.target_col]

        # ======================================================
        # BASE MODEL
        # ======================================================
        base_model_name = self.best_config["base_model"]
        base_params = self.best_config["base_params"].copy()

        if self.random_state is not None and "random_state" in str(MODEL_REGISTRY[base_model_name]):
            base_params["random_state"] = self.random_state

        self.base = MODEL_REGISTRY[base_model_name](**base_params)
        self.base.fit(X_train, y_train)

        y_base_train = self.base.predict(X_train)
        y_base_test = self.base.predict(X_test)

        # ======================================================
        # RESIDUAL STAGE
        # ======================================================
        res_model_name = self.best_config["residual_model"]

        if res_model_name in [None, "none", "base_only"]:
            self.res = None
            y_res_train = np.zeros_like(y_base_train)
            y_res_test = np.zeros_like(y_base_test)
        else:
            res_params = self.best_config["residual_params"].copy()

            if self.random_state is not None and "random_state" in str(MODEL_REGISTRY[res_model_name]):
                res_params["random_state"] = self.random_state

            self.res = MODEL_REGISTRY[res_model_name](**res_params)
            self.res.fit(X_train, y_train - y_base_train)

            y_res_train = self.res.predict(X_train)
            y_res_test = self.res.predict(X_test)

        # ======================================================
        # CORRECTION STAGE
        # ======================================================
        corr_model_name = self.best_config["correction_model"]

        raw_train_preds = y_base_train + y_res_train
        raw_test_preds = y_base_test + y_res_test

        # RMSE baseline (instead of R2)
        baseline_rmse = root_mean_squared_error(y_test, raw_test_preds)

        if corr_model_name in [None, "none", "base_only"]:
            self.corr = None
            final_test = raw_test_preds

        elif corr_model_name == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_train_preds, y_train)
            iso_test = iso.predict(raw_test_preds)

            iso_rmse = root_mean_squared_error(y_test, iso_test)

            # RMSE-based guard
            if iso_rmse < baseline_rmse:
                self.corr = iso
                final_test = iso_test
            else:
                self.corr = None
                final_test = raw_test_preds

        else:
            corr_params = self.best_config["correction_params"].copy()

            if self.random_state is not None and "random_state" in str(MODEL_REGISTRY[corr_model_name]):
                corr_params["random_state"] = self.random_state

            self.corr = MODEL_REGISTRY[corr_model_name](**corr_params)

            X_corr_train = np.asarray(raw_train_preds).reshape(-1, 1)
            X_corr_test = np.asarray(raw_test_preds).reshape(-1, 1)

            self.corr.fit(X_corr_train, y_train)
            corr_test = self.corr.predict(X_corr_test)

            corr_rmse = root_mean_squared_error(y_test, corr_test)

            # RMSE-based safety check
            if corr_rmse < baseline_rmse:
                final_test = corr_test
            else:
                self.corr = None
                final_test = raw_test_preds

        # ======================================================
        # METRICS (FINAL)
        # ======================================================
        self.rmse = root_mean_squared_error(y_test, final_test)
        self.r2 = r2_score(y_test, final_test)

        # ======================================================
        # Prediction DataFrame
        # ======================================================
        self.dpred = pd.DataFrame({
            "time": self.test_df["time"],
            "Actual": y_test,
            "Predicted": final_test,
        }).reset_index(drop=True)

        # ======================================================
        # Uncertainty (Gaussian approx)
        # ======================================================
        self.dpred["Predicted_lower"] = (
            self.dpred["Predicted"] - 1.96 * self.rmse
        ).clip(lower=0)

        self.dpred["Predicted_upper"] = (
            self.dpred["Predicted"] + 1.96 * self.rmse
        )

        # ======================================================
        # Return
        # ======================================================
        return {
            "test_r2": self.r2,
            "test_rmse": self.rmse,
            "predictions": self.dpred,
            "model_info": {
                "base_model": self.best_config["base_model"],
                "residual_model": self.best_config["residual_model"],
                "correction_model": self.best_config["correction_model"],
                "features": self.best_config["features"],
            },
            "data_summary": {
                "train_size": len(self.train_df),
                "test_size": len(self.test_df),
                "train_period": str(self.train_df["time"].min()),
                "test_period": str(self.test_df["time"].max()),
            }
        }
    
    # --------------------------------------------------
    # AUTO BUILD REPORT ARTIFACTS 
    # --------------------------------------------------
    @track_time("report_generation_seconds")
    def build_report_artifacts(self, projection_summary, tidy_df):
        """
        Construct standardized reporting artifacts from model outputs.

        This method generates a structured collection of artifacts required for
        downstream reporting and visualization. It transforms model outputs,
        including projections and evaluation summaries, into a consistent format
        that can be used by the reporting layer (e.g., dashboards, plots, or exports).

        Users are not expected to manually assemble these artifacts. This function
        ensures that all required components are created automatically and in a
        reproducible format.

        Parameters
        ----------

        projection_summary : dict or pandas.DataFrame
            Summary of model projections, typically produced by the projection
            pipeline (e.g., `DiseaseProjection`). This may include predicted values,
            confidence intervals, and temporal aggregation.

        Returns
        -------

        artifacts : ReportArtifacts
            - A structured object containing all elements required for reporting,
            such as:
                - Processed projection data
                - Evaluation metrics (if available)
                - Metadata (e.g., model configuration, time range)
                - Visualization-ready datasets

        Notes
        -----

        - This method is part of the reporting pipeline and is typically called
        after model training and projection steps.
        - Ensures consistency between modeling outputs and reporting interfaces.
        - Designed to support reproducible research workflows.
        """

        from climaid.reporting import ReportArtifacts

        if self.best_config is None:
            raise RuntimeError("Run optimize_lags() first.")

        if not hasattr(self, "r2"):
            raise RuntimeError("Run train_final_model() first.")

        # ---------------------------
        # Metrics
        # ---------------------------
        metrics = {
            "test_r2": np.round(self.r2,2), 
            "test_rmse": np.round(self.rmse, 2),
        }

        # ---------------------------
        # Selected lags (auto-extract)
        # ---------------------------
        selected_lags = {}
        interaction_lags = []

        for f in self.best_config['features']:

            # --------------------------
            # Interaction terms
            # --------------------------
            if "_x_" in f:

                left, right = f.split("_x_")

                if "_lag" in left and "_lag" in right:

                    var1, lag1 = left.split("_lag")
                    var2, lag2 = right.split("_lag")

                    interaction_lags.append({
                        "var1": var1,
                        "lag1": int(lag1),
                        "var2": var2,
                        "lag2": int(lag2)
                    })

            # --------------------------
            # Single lag terms
            # --------------------------
            elif "_lag" in f:

                var, lag = f.split("_lag")
                
                selected_lags[var] = int(lag)

        # ---------------------------
        # Feature importance 
        # ---------------------------
        importance = {}
        if hasattr(self.base, "feature_importances_"):
            importance = dict(
                zip(self.best_config["features"], self.base.feature_importances_)
            )

        # ---------------------------
        # Model metadata
        # ---------------------------
        model_info = {
            "base_model": self.best_config["base_model"],
            "residual_model": self.best_config["residual_model"],
            "correction_model": self.best_config["correction_model"],
            "stacking_pipeline": "Base → Residual → Correction",
            "n_features": len(self.best_config["features"]),
        }

        # ---------------------------
        # Data summary
        # ---------------------------
        # Ensure datetime safety (prevents malformed tokens in reports)
        train_time = pd.to_datetime(self.train_df["time"], errors="coerce")
        test_time = pd.to_datetime(self.test_df["time"], errors="coerce")

        # Drop invalid timestamps if any
        train_time = train_time.dropna()
        test_time = test_time.dropna()

        # Format clean ISO dates for reporting (LLM-friendly)
        if not train_time.empty:
            train_start = train_time.min().strftime("%d-%m-%Y")
            train_end = train_time.max().strftime("%d-%m-%Y")
        else:
            train_start, train_end = "Unknown", "Unknown"

        if not test_time.empty:
            test_start = test_time.min().strftime("%d-%m-%Y")
            test_end = test_time.max().strftime("%d-%m-%Y")
        else:
            test_start, test_end = "Unknown", "Unknown"

        data_summary = {
            "train_size": int(len(self.train_df)),
            "test_size": int(len(self.test_df)),
            "train_period": f"{train_start} to {train_end}",
            "test_period": f"{test_start} to {test_end}",
        }

        # ---------------------------
        # Create artifacts (AUTO)
        # ---------------------------
        artifacts = ReportArtifacts(
            district=self.district,
            disease_name=self.disease_name,
            date_range=f"{train_start} to {test_end}",
            metrics=metrics,
            selected_lags=selected_lags,
            interaction_lags = interaction_lags,
            features=self.best_config["features"],
            importance=importance,
            projection_summary=projection_summary,
            runtime=self.runtime,
            model_info=model_info,
            data_summary=data_summary,
            download_data=tidy_df,
        )

        return artifacts

    @track_time("prediction_seconds")
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate disease case predictions using the trained stacked model.

        This method applies the full prediction pipeline consisting of:
        
        1. **Base model** – Generates initial predictions from input features.
        2. **Residual model** – Learns and predicts residual errors from the base model.
        3. **Correction step** – Combines base predictions and residual corrections
        to produce final refined predictions.

        The method assumes that the model has already been trained using
        `train_final_model`.

        Parameters
        ----------

        X : pandas.DataFrame :
            Input feature matrix containing the same variables used during training,
            including climate variables and any engineered lagged features.

        Returns
        -------

        predictions : numpy.ndarray
            Array of predicted disease case counts (or risk scores), aligned with
            the input samples in `X`.

        Notes
        -----

        - Input features must match the training schema (column names and preprocessing).
        - If lagged features were used during training, they must be present in `X`.
        - This method uses the internally stored trained models and does not retrain.
        """

        # -----------------------
        # Safety checks
        # -----------------------
        if not hasattr(self, "base"):
            raise RuntimeError(
                "diseaseModel has not been trained. "
                "Call train_final_model() before predict()."
            )

        # Ensure correct feature order
        X = X[self.best_config["features"]]

        # -----------------------
        # Base model prediction
        # -----------------------
        base_pred = self.base.predict(X)

        # -----------------------
        # Residual correction
        # -----------------------
        if hasattr(self, "res") and self.res is not None:
            resid_pred = self.res.predict(X)
            combined_pred = base_pred + resid_pred
        else:
            combined_pred = base_pred

        # -----------------------
        # Calibration correction
        # -----------------------
        if hasattr(self, "cor") and self.corr is not None:
            final_pred = self.corr.predict(combined_pred)
            # upperbound = final_pred + 1.96 * self.rmse
            # lowerbound = final_pred - 1.96 * self.rmse
        else:
            final_pred = combined_pred
            # upperbound = final_pred + 1.96 * self.rmse
            # lowerbound = final_pred - 1.96 * self.rmse

        return final_pred     
    
    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------
    @track_time("report_generation_seconds")
    def generate_report(self, 
                        projection_summary = None, 
                        llm_client=None, 
                        tidy_df=None, 
                        style="detailed", 
                        open_browser = False, 
                        save_copy=False, 
                        output_dir = None):
        """
        Generate a comprehensive disease projection report.

        This method provides a fully automated reporting pipeline that transforms
        model outputs into a structured, human-readable report. It integrates
        projections, optional language model summarization, and formatted outputs
        for visualization or sharing.

        Users are not required to manually construct intermediate artifacts—this
        method handles all necessary steps internally, including artifact creation,
        formatting, and optional rendering.

        Parameters
        ----------

        projection_summary : dict or pandas.DataFrame, optional
            Precomputed projection results. If not provided, the method may internally
            generate projections using the trained model.

        llm_client : object, optional
            Language model client used to generate narrative summaries or insights.
            If provided, the report may include AI-generated interpretations of trends.

        tidy_df : pandas.DataFrame
            DataFrame for exporting to an excel file. 

        style : {"detailed", "summary", "policy"}, default="detailed"
            - Level of detail in the generated report:
                - "detailed": Includes full analysis, metrics, and explanations.
                - "summary": Provides a concise overview of key findings.
                - "policy": Provides a comprehensive policy report. 

        open_browser : bool, default=False
            If True, automatically opens the generated report in a web browser.

        save_copy : bool, default=False
            If True, saves a copy of the report to disk.

        output_dir : str, optional
            Directory where the report will be saved if `save_copy=True`.
            If not specified, a default output location is used.

        Returns
        -------

        report : str or pathlib.Path
            Path to the generated report file or rendered output, depending on configuration.

        Notes
        -----
        
        - This method combines multiple pipeline stages:
            1. Projection (if not provided)
            2. Artifact construction
            3. Report formatting and rendering
        - Designed for end-to-end usability with minimal user intervention.
        - Supports integration with LLMs for enhanced interpretability.
        """

        import time
        from climaid.reporting import DiseaseReporter, open_report_in_browser

        if projection_summary is None:
            projection_summary = {
                "mode": "historical_only",
                "note": "No climate projection summary provided. Report based on historical model performance only."
            }

        # Auto-build artifacts internally
        artifacts = self.build_report_artifacts(projection_summary, tidy_df)

        # Create reporter
        reporter = DiseaseReporter(llm_client=llm_client)
        report = reporter.generate(artifacts, style=style)

        if self._pipeline_start_time is not None:
            self.runtime["total_pipeline_seconds"] = (
                time.perf_counter() - self._pipeline_start_time
            )

        if open_browser:

            from climaid.reporting import open_report_in_browser

            title = f"ClimAID {self.disease_name} Risk Intelligence Report"

            report_path = open_report_in_browser(
                report_text=report,
                artifacts=artifacts,
                title=title,
                save_copy=save_copy,
                output_dir="climaid_outputs/reports"
            )

        # Generate report
        return report

    # --------------------------------------------------
    # PLOTTING FUNCTION
    # --------------------------------------------------
    def plot_historical_predictions(
            self,
            actual_color='#006d77',
            prediction_color='#e29578',
            hatch_color = '#ffddd2',
            figsize=(12, 5),
            hatch = '.',
            save=False,
            alpha = 0.4, 
            path='path',
            theme='ticks',
            dpi=500,
        ):
            """
            Plot observed vs predicted disease cases over time.

            This method visualizes the model's performance by comparing historical
            observed values with predicted values. It helps assess model fit,
            identify temporal patterns, and detect systematic deviations.

            The plot may optionally include shaded or hatched regions to highlight
            differences between observed and predicted values.

            Parameters
            ----------

            actual_color : str, default='#006d77'
                Color used for plotting observed (true) disease cases.

            prediction_color : str, default='#e29578'
                Color used for plotting model predictions.

            hatch_color : str, default='#ffddd2'
                Color used for shaded or hatched regions representing prediction error
                or uncertainty.

            figsize : tuple of int, default=(12, 5)
                Size of the figure in inches (width, height).

            hatch : str, default='.'
                Matplotlib hatch pattern used to highlight differences between
                observed and predicted values.

            save : bool, default=False
                If True, saves the plot to disk.

            alpha : float, default=0.4
                Transparency level for shaded or hatched regions.

            path : str, default='path'
                File path where the plot will be saved if `save=True`.

            theme : str, default='ticks'
                Seaborn or matplotlib style theme applied to the plot.

            dpi : int, default=500
                Resolution of the saved figure in dots per inch.

            Returns
            -------

            - None
            - Displays the plot and optionally saves it to disk.

            Notes
            -----
            
            - Requires the model to be trained and predictions to be available.
            - Assumes temporal alignment between observed and predicted values.
            - Useful for diagnostic evaluation of model performance.
            """

            import matplotlib.pyplot as plt
            import seaborn as sns
            import matplotlib.dates as mdates

            if "Actual" not in self.dpred.columns:
                raise KeyError("Dataframe must contain 'Actual' column.")

            sns.set_theme(style=theme, font_scale=1.2)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor('#f8f9fa')

            ax.plot(self.dpred['time'], self.dpred['Actual'], label='Actual cases',
                    marker='8', color=actual_color, linewidth=3)

            ax.plot(self.dpred['time'], self.dpred['Predicted'], label='Predicted cases',
                    marker='o', color=prediction_color, linewidth=3)

            ax.fill_between(
                self.dpred['time'],
                self.dpred['Predicted_lower'],
                self.dpred['Predicted_upper'],
                hatch=hatch,
                linewidth=0.0,
                color=hatch_color,
                alpha=alpha
            )

            ax.legend(loc='upper left', ncol=2, frameon=False)
            ax.set_ylabel('Cases', weight='demi')
            ax.grid(alpha=0.2, linestyle='--', color='#6c757d')

            ax.set_xlabel("Month-Year", weight='demi')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            plt.xticks(fontsize=10)

            plt.text(
                0.715, 0.9675,
                f"$R^2$ = {self.r2:.3f}; RMSE = {self.rmse:.2f}",
                transform=plt.gca().transAxes,
                fontsize=14,
                verticalalignment='top'
            )

            fig.autofmt_xdate(rotation=90)
            plt.tight_layout()

            if save:
                plt.savefig(path, dpi=dpi)

            plt.show()

    # --------------------------------------------------
    # Time Summary
    # --------------------------------------------------
    def print_runtime_summary(self):
        """
        Prints a clean runtime summary for the full pipeline.
        """
        if not hasattr(self, "runtime") or not self.runtime:
            print("Runtime information not available.")
            return

        print("\n==============================")
        print("⏱ ClimAID Pipeline Runtime Summary")
        print("==============================")

        total = 0.0
        for k, v in self.runtime.items():
            if isinstance(v, (int, float)):
                print(f"{k.replace('_', ' ').title()}: {v:.2f} seconds")
                total += v

        if total > 0:
            print("------------------------------")
            print(f"Total Computation Time: {total:.2f} seconds")
            print("==============================\n")
