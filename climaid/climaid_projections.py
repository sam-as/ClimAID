"""
climaid_projections.py
----------------
Created Date: Wednesday; November 19th, 2025
Author: Avik Kumar Sam
Webpage: https://sites.google.com/view/aviksam

This module gives predictions projected for the SSPs and the years
"""

# projection.py

from __future__ import annotations
import pandas as pd

from .climate_data import ClimateData
from .climaid_model import DiseaseModel


class DiseaseProjection:
    """
    Generate climate-driven disease projections using CMIP6 data.

    This class provides a high-level interface for producing future disease
    projections under different climate scenarios. It integrates trained
    disease models with CMIP6 climate projections to simulate potential
    disease dynamics across time, regions, and emission pathways.

    The projection pipeline supports flexible configurations, including
    multiple climate models (GCMs), Shared Socioeconomic Pathways (SSPs),
    and ensemble-based approaches.

    Features
    --------
    - Single scenario projection (one GCM + SSP)
    - Multi-GCM projections for model comparison
    - Multi-SSP projections for scenario analysis
    - Ensemble mean projections across multiple simulations
    - Visualization of projected disease trends
    - Export of results to CSV with prediction intervals

    Notes
    -----
    
    - Requires a trained `DiseaseModel` instance.
    - Assumes CMIP6 climate inputs are preprocessed and aligned with the
      model's feature requirements.
    - Designed for scenario-based forecasting and climate impact assessment.

    """

    def __init__(self, disease_model: DiseaseModel):
        """
        Initialize the DiseaseProjection interface using a trained DiseaseModel.

        This constructor extracts the necessary components from a trained
        `DiseaseModel` instance to enable future climate-based projections.
        It validates the availability of projected climate data and prepares
        all required attributes for downstream projection tasks.

        Parameters
        ----------

        disease_model : DiseaseModel
            A trained DiseaseModel instance containing fitted models, optimized
            features, preprocessing objects, and projected climate data
            (`df_climate_proj`).

        Raises
        ------

        ValueError : 
            If the provided DiseaseModel does not contain projected climate data.

        Notes
        -----

        - The DiseaseModel must be fully trained prior to initialization.
        - Requires `df_climate_proj` to be available for generating projections.
        - Internally extracts:
            - Target variable name
            - Selected feature set (from best configuration)
            - Preprocessing objects (e.g., scaler)
            - Model performance metrics (e.g., RMSE)
            - Feature metadata for consistent transformation
        """


        if disease_model.df_climate_proj is None:
            raise ValueError("DiseaseModel must include projected climate data (df_climate_proj).")
        self.model = disease_model
        self.target_col = disease_model.target_col
        self.climate = disease_model.df_climate_proj
        self.feats = disease_model.best_config["features"]
        self.rmse = disease_model.rmse
        self.df_merged = disease_model.df_merged
        self.scaler = disease_model.scaler
        self.best_config = disease_model.best_config
        self.feature_metadata = disease_model.feature_metadata

    # ------------------------------------------------------
    # Helper: Create lagged features required by the model
    # ------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and transform input features for projection or prediction.

        This method ensures that input data is aligned with the feature
        configuration used during model training. It applies necessary
        preprocessing steps such as feature selection, ordering, and scaling
        to maintain consistency with the trained model.

        Parameters
        ----------

        df : pandas.DataFrame : 
            Input data containing climate variables and any required predictors.
            Must include all features expected by the trained model.

        Returns
        -------

        df_out : pandas.DataFrame : 
            Transformed feature matrix ready for model inference, with the
            correct feature set, ordering, and preprocessing applied.

        Notes
        -----

        - Ensures consistency between training and projection pipelines.
        - Applies the same feature selection defined in `best_config`.
        - Uses the stored scaler or preprocessing objects from the trained model.
        - Missing or misaligned features may result in errors.

        """
        df = df.copy()

        meta = self.feature_metadata
        lag_map = meta["lags"]
        interactions = meta["interactions"]

        # -------------------------------------------------
        # Getting the year and month
        # -------------------------------------------------

        df["time"] = pd.to_datetime(df["time"])
        df["Year"] = df["time"].dt.year
        df["Month"] = df["time"].dt.month

        # -------------------------------------------------
        # Monthly averages
        # -------------------------------------------------

        climate_vars = ["mean_temperature", "mean_Rain", "mean_SH"]

        for var in climate_vars:

            if var in df.columns:

                df[f"MA_{var}"] = (
                    df.groupby("Month")[var]
                    .transform(
                        lambda x: x.rolling(window=10, min_periods=1).mean()
                    )
                )

        # -------------------------------------------------
        # Yearly averages
        # -------------------------------------------------

        for var in climate_vars:

            if var in df.columns:

                df[f"YA_{var}"] = (
                    df.groupby("Year")[var]
                    .transform("mean")
                )

        # ----------------------------------
        # Create ENSO base interaction features
        # ----------------------------------
        climate_vars = ["mean_temperature", "mean_Rain", "mean_SH"]

        for var in climate_vars:
            enso_col = f"ENSO_{var}"

            if enso_col not in df.columns:
                if var in df.columns and "Nino_anomaly" in df.columns:
                    df[enso_col] = df[var] * df["Nino_anomaly"]

        # ----------------------------------
        # Sorting for lag generation
        # ----------------------------------
        if "time" in df.columns:
            sort_cols = ["time"]
        elif {"Year", "Month"}.issubset(df.columns):
            sort_cols = ["Year", "Month"]
        else:
            raise ValueError(
                "Input dataframe must contain 'time' or ['Year','Month']."
            )

        # Determine grouping (for climate projections)
        group_cols = [c for c in ["model", "ssp", "member"] if c in df.columns]

        if group_cols:
            df = df.sort_values(group_cols + sort_cols)
            grouped = df.groupby(group_cols)
        else:
            df = df.sort_values(sort_cols)
            grouped = None

        # ----------------------------------
        # Generate lag features
        # ----------------------------------
        lag_cols = []

        for var, lags in lag_map.items():

            if var not in df.columns:
                raise ValueError(
                    f"Missing required base variable in projection data: {var}"
                )

            for lag in lags:
                colname = f"{var}_lag{lag}"
                if grouped is not None:
                    df[colname] = grouped[var].shift(lag)
                else:
                    df[colname] = df[var].shift(lag)

                lag_cols.append(colname)

        # ----------------------------------
        # Recreate lagged ENSO interactions
        # ----------------------------------
        for left, right in interactions:

            if left not in df.columns or right not in df.columns:
                raise ValueError(
                    f"Missing required interaction inputs: {left}, {right}"
                )

            df[f"{left}_x_{right}"] = df[left] * df[right]

        # ----------------------------------
        # Ensure Month column exists
        # ----------------------------------
        if "Month" not in df.columns:

            if "time" in df.columns:
                df["Month"] = pd.to_datetime(df["time"]).dt.month
            elif {"Year", "Month"}.issubset(df.columns):
                pass

        # ----------------------------------
        # Remove incomplete lag rows
        # ----------------------------------
        if lag_cols:
            df = df.dropna(subset=lag_cols).reset_index(drop=True)

        # ----------------------------------
        # Select final feature space
        # ----------------------------------
        feats = self.best_config["features"]

        base_cols = ["time"]
        optional_cols = [c for c in ["model", "ssp", "member"] if c in df.columns]

        # Safety check
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required features: {missing}")

        df_out = df[base_cols + optional_cols + feats]
        if "Month" not in df_out.columns:
            df_out["Month"] = pd.to_datetime(df_out["time"]).dt.month

        # ----------------------------------
        # Apply scaler
        # ----------------------------------
        if self.scaler is not None:
            cols = [c for c in meta["scaled_columns"] if c in df_out.columns]
            if cols:
                df_out.loc[:, cols] = self.scaler.transform(df_out[cols])

        return df_out
    
    # ------------------------------------------------------
    #  Single CMIP6 model prediction (core function)
    # ------------------------------------------------------
    def project(self, model_name: str, ssp: str, data=None) -> pd.DataFrame:
        """
        Generate disease projections for a single CMIP6 model and SSP scenario.

        This method produces future disease predictions using climate projections
        from a specified General Circulation Model (GCM) under a given Shared
        Socioeconomic Pathway (SSP). It forms the core projection step by combining
        climate inputs with the trained disease model.

        The method internally prepares features, applies the trained model, and
        returns time-resolved projections.

        Parameters
        ----------

        model_name : str
            Name of the CMIP6 climate model (GCM) to use (e.g., "MPI-ESM1-2-HR").

        ssp : str
            Shared Socioeconomic Pathway scenario (e.g., "ssp245", "ssp585").

        data : pandas.DataFrame, optional : 
            Optional pre-filtered climate dataset. If not provided, the method
            uses internally stored CMIP6 projection data.

        Returns
        -------

        proj_features : pandas.DataFrame or None : 
            DataFrame containing projected disease values over time for the
            specified model and scenario. Returns None if the requested data
            is not available.
        """

        import pandas as pd

        # --------------------------------------------------
        # Load CMIP6 climate data
        # --------------------------------------------------
        if data is not None:
            proj = data[(data['ssp'] == ssp) & (data['model'] == model_name)].copy()
        else:
            proj = self.climate[
                (self.climate["model"] == model_name) &
                (self.climate["ssp"] == ssp)
            ].copy()

        # ---- Check 1: climate data exists ----
        if proj is None or len(proj) == 0:
            print(f"[INFO] No climate data for {model_name} - {ssp}")
            return None

        # --------------------------------------------------
        # Prepare features
        # --------------------------------------------------
        try:
            if data is not None:
                proj_features = proj.copy()
            else:
                proj_features = self.prepare_features(proj)
        except Exception as e:
            print(f"[ERROR] Feature preparation failed for {model_name} - {ssp}: {e}")
            return None

        # ---- Check 2: features not empty (lags may drop rows) ----
        if proj_features is None or len(proj_features) == 0:
            print(f"[INFO] No usable features after preprocessing for {model_name} - {ssp}")
            return None

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------
        try:
            preds = self.model.predict(proj_features)
        except Exception as e:
            print(f"[ERROR] Prediction failed for {model_name} - {ssp}: {e}")
            return None

        # --------------------------------------------------
        # Add outputs
        # --------------------------------------------------
        proj_features["disease_projection"] = preds
        proj_features["GCM"] = model_name
        proj_features["SSP"] = ssp

        # --------------------------------------------------
        # Add RMSE-based uncertainty
        # --------------------------------------------------
        if hasattr(self, "rmse") and self.rmse is not None:
            proj_features['lower_bound'] = (
                proj_features["disease_projection"] - 1.96 * self.rmse
            )
            proj_features['upper_bound'] = (
                proj_features["disease_projection"] + 1.96 * self.rmse
            )

            proj_features['lower_bound'] = proj_features['lower_bound'].clip(lower=0)
            proj_features['upper_bound'] = proj_features['upper_bound'].clip(lower=0)

        else:
            print("[WARNING] RMSE not found — uncertainty not added")

        return proj_features

    # ------------------------------------------------------
    #  Multiple GCM projections
    # ------------------------------------------------------
    def project_model_list(self, model_list: list[str], ssp: str):
        """
        Generate disease projections for multiple CMIP6 models under a single SSP.

        This method iterates over a list of General Circulation Models (GCMs)
        and computes disease projections for each model using the specified
        Shared Socioeconomic Pathway (SSP). It acts as a wrapper around the
        core `project` method, enabling multi-model comparison.

        Models with missing or unavailable data are skipped automatically.

        Parameters
        ----------

        model_list : list of str
            List of CMIP6 model names (GCMs) to evaluate.

        ssp : str
            Shared Socioeconomic Pathway scenario (e.g., "ssp245", "ssp585").

        Returns
        -------

        outputs : dict[str, pandas.DataFrame]
            Dictionary mapping each valid GCM name to its corresponding
            projection DataFrame.


        Notes
        -----

        - Internally calls `project` for each model.
        - Skips models with missing or empty projection data.
        - Prints informational messages when models are skipped.
        - Returns an empty dictionary if no valid projections are generated.

        """

        outputs = {}

        for gcm in model_list:
            df = self.project(gcm, ssp)

            if df is None or len(df) == 0:
                print(f"[INFO] Skipping {gcm} for {ssp}")
                continue

            outputs[gcm] = df

        if len(outputs) == 0:
            print("[WARNING] No projections generated.")

        return outputs

    # ------------------------------------------------------
    #  Multiple SSP + Model projections
    # ------------------------------------------------------
    def project_multi_model_ssp(
        self,
        model_list: list[str],
        ssp_list: list[str],
    ):
        """
        Generate disease projections across multiple CMIP6 models and SSP scenarios.

        This method performs a full projection sweep over all combinations of
        General Circulation Models (GCMs) and Shared Socioeconomic Pathways (SSPs).
        It aggregates results into a single unified DataFrame for comparative
        analysis and downstream processing.

        Each projection is computed using the core `project` method, with results
        annotated by model and scenario identifiers.

        Parameters
        ----------

        model_list : list of str
            List of CMIP6 model names (GCMs) to evaluate.

        ssp_list : list of str
            List of SSP scenarios (e.g., ["ssp245", "ssp585"]).

        Returns
        -------

        master_df : pandas.DataFrame :
            Combined DataFrame containing projections for all valid model–scenario
            combinations. Includes:

            - Predicted disease values (`disease_projection`)
            - Prediction intervals (e.g., `lower_bound`)
            - Model identifier (`GCM`)
            - Scenario identifier (`SSP`)
            - Temporal columns (e.g., `Year`, `Month`)

        Raises
        ------

        ValueError : 
            If no valid projections are generated across all combinations.

        Notes
        -----

        - Iterates over all combinations of `model_list × ssp_list`.
        - Skips combinations with missing or invalid data.
        - Catches and logs errors for individual model–scenario pairs without
        interrupting the full pipeline.
        - Clips negative projection values to zero for physical consistency.
        - Sorts output by GCM, SSP, and time (if columns are available).

        """

        import pandas as pd

        all_outputs = []

        for model_name in model_list:
            for ssp in ssp_list:

                try:
                    df = self.project(model_name, ssp)

                    # Skip empty outputs
                    if df is None or len(df) == 0:
                        print(f"Skipping {model_name} {ssp} (no projection data)")
                        continue

                    df = df.copy()
                    df["GCM"] = model_name
                    df["SSP"] = ssp.lower()

                    all_outputs.append(df)

                except Exception as e:
                    print(f"Skipping {model_name} {ssp} due to error:", e)
                    continue

        if len(all_outputs) == 0:
            raise ValueError("No projections were generated.")

        master_df = pd.concat(all_outputs, ignore_index=True)

        # Final check 
        master_df['disease_projection'] = master_df['disease_projection'].clip(lower = 0)
        master_df['lower_bound'] = master_df['lower_bound'].clip(lower = 0)

        # Sort for consistency
        if {"GCM","SSP","Year","Month"}.issubset(master_df.columns):
            master_df = master_df.sort_values(["GCM","SSP","Year","Month"])

        return master_df
    
    # ------------------------------------------------------
    #  Projected outbreaks
    # ------------------------------------------------------    
    def flag_outbreak_risk(
        self,
        df_proj,
        method: str = "both",  # "historical", "dynamic", "gam" or "both"
        percentile: float = 0.9,
        projection_col: str = "disease_projection",
        year_col: str = "Year",
    ):
        
        """
        Flag outbreak risk based on projected disease levels.

        This method classifies projected disease values into outbreak risk
        categories using percentile-based thresholds. It supports both
        fixed (historical) and adaptive (scenario-specific) baselines.

        Two approaches are available:

        A) Historical baseline:
        Uses a fixed percentile threshold derived from historical data
        (e.g., 90th percentile of observed cases).

        B) Dynamic baseline:
        Computes thresholds separately for each GCM–SSP combination,
        allowing risk levels to adapt to scenario-specific distributions.

        Parameters
        ----------

        df_proj : pandas.DataFrame :
            DataFrame containing projection results. Must include the column
            specified by `projection_col`, and optionally `GCM` and `SSP`
            for dynamic thresholding.

        method : {"historical", "dynamic", "both"}, default="both" :
            Method used to define outbreak thresholds:
            - "historical": fixed threshold from historical data
            - "dynamic": scenario-specific thresholds
            - 'gam' : Generalised-Additive Model using Q_{0.9}​(t)≈ μ(t)+ zσ
            - "both": compute and return both risk indicators

        percentile : float, default=0.9 : 
            Percentile used to define the outbreak threshold (e.g., 0.9 = 90th percentile).

        projection_col : str, default="disease_projection" :
            Column name containing projected disease values.

        year_col: str, default = "Year",
            Column name for the year which comes from the model itself. 

        Returns
        -------

        df : pandas.DataFrame :
            Input DataFrame with additional columns indicating outbreak risk.
            Depending on `method`, includes:

            - `risk_historical` : binary or categorical flag based on historical threshold
            - `risk_dynamic` : binary or categorical flag based on adaptive thresholds

        Notes
        -----

        - Historical thresholds are typically computed from training or observed data.
        - Dynamic thresholds are computed within each (GCM, SSP) group.
        - Useful for identifying high-risk periods under future climate scenarios.
        - The method does not modify original projection values.
        """

        import pandas as pd

        df = df_proj.copy()

        # -------------------------
        # A) Historical Baseline
        # -------------------------
        if method in ("historical", "both"):
            hist_cases = self.df_merged[self.target_col].dropna()
            hist_threshold = hist_cases.quantile(percentile)

            df["risk_flag_historical_baseline"] = (
                df[projection_col] > hist_threshold
            )
            df["historical_threshold"] = hist_threshold

        # -------------------------
        # B) Dynamic Time-Window Baseline
        # -------------------------
        if method in ("dynamic", "both"):

            bins = [0, 2030, 2050, 2070, float("inf")]
            labels = ["upto_2030", "2030_2050", "2050_2070", "2070_plus"]

            df["time_window"] = pd.cut(df[year_col], bins=bins, labels=labels)

            df["risk_flag_dynamic_baseline"] = False
            df["dynamic_threshold"] = None

            grouping_cols = ["GCM", "SSP", "time_window"]

            for _, group_idx in df.groupby(grouping_cols).groups.items():
                group = df.loc[group_idx]

                dyn_threshold = group[projection_col].quantile(percentile)

                df.loc[group_idx, "dynamic_threshold"] = dyn_threshold
                df.loc[group_idx, "risk_flag_dynamic_baseline"] = (
                    group[projection_col] > dyn_threshold
                )

        # -------------------------
        # Combined Flag (NO GAM)
        # -------------------------
        if method == "both":
            df["risk_flag_combined"] = (
                df.get("risk_flag_historical_baseline", False)
                | df.get("risk_flag_dynamic_baseline", False)
            )

        self.df_projection_risk = df
        return df

    # ------------------------------------------------------
    #  Ensemble mean projection
    # ------------------------------------------------------
    def project_ensemble_mean(self, model_list: list[str], ssp: str):
        """
        Compute ensemble-mean disease projections across multiple CMIP6 models.

        This method aggregates projections from multiple General Circulation Models
        (GCMs) under a single Shared Socioeconomic Pathway (SSP) to produce a
        consolidated ensemble estimate. It summarizes central tendency and
        uncertainty across models.

        The ensemble provides a more robust estimate of future disease dynamics by
        reducing reliance on any single climate model.

        Parameters
        ----------

        model_list : list of str
            List of CMIP6 model names (GCMs) to include in the ensemble.

        ssp : str
            Shared Socioeconomic Pathway scenario (e.g., "ssp245", "ssp585").

        Returns
        -------

        ensemble : pandas.DataFrame
            DataFrame containing aggregated projections with the following columns:

            - `Year`, `Month` : Time indices
            - `mean` : Ensemble mean projection
            - `min`, `max` : Range across models
            - `p05`, `p95` : 5th and 95th percentile bounds
            - `GCM` : Set to "ENSEMBLE_MEAN"
            - `SSP` : Scenario identifier

        Notes
        -----

        - Internally calls `project_model_list` to generate individual projections.
        - Assumes consistent temporal alignment across models.
        - Percentile bounds (`p05`, `p95`) provide a measure of inter-model uncertainty.
        - Useful for summarizing projections in reports and decision-making contexts.

        """

        all_models = self.project_model_list(model_list, ssp)

        combined = pd.concat(all_models.values(), axis=0)

        # Mean disease projection per month
        ensemble = (
            combined
            .groupby(["Year", "Month"])["disease_projection"]
            .agg(
                mean="mean",
                min="min",
                max="max", 
                p05=lambda x: x.quantile(0.05),
                p95=lambda x: x.quantile(0.95)
            )
            .reset_index()
        )

        ensemble["GCM"] = "ENSEMBLE_MEAN"
        ensemble["SSP"] = ssp

        return ensemble
    
    def build_projection_summary(self, df: pd.DataFrame) -> dict:
        """
        Construct a structured summary of CMIP6-based disease projections.

        This method transforms a multi-model, multi-scenario projection DataFrame
        into a standardized summary dictionary suitable for reporting, visualization,
        and downstream analysis. It aggregates projections across GCMs and SSPs,
        computes ensemble statistics, and organizes outputs into interpretable blocks.

        Parameters
        ----------

        df : pandas.DataFrame :
            Projection DataFrame generated from multi-model and multi-SSP workflows
            (e.g., `project_multi_model_ssp`). Expected to include:

            - `time` or (`Year`, `Month`)
            - `GCM` (climate model identifier)
            - `SSP` (scenario identifier)
            - `disease_projection`
            - `lower_bound`
            - `upper_bound`

        Returns
        -------

        projection_summary : dict :
            Structured dictionary containing:

            - `mode` : Type of analysis (e.g., future climate projection)
            - `method` : Description of projection methodology
            - `projection_period` : Time span of projections
            - `climate_models` : List of GCMs used
            - `ssp_scenarios` : List of SSPs evaluated

            - `ensemble_mean` : Aggregated statistics across all models
                - mean_projection
                - max_projection
                - min_projection
                - trend
                - peak_transmission_months

            - `ssp_ensemble` : Scenario-specific aggregated summaries
            - `ensemble_timeseries` : Time series of ensemble mean projections
            - `ssp_timeseries` : Time series per SSP
            - `gcm_summary` : Model-wise summary statistics
            - `risk_matrix` : Risk classification across models and scenarios

            - `uncertainty` :
                - mean_uncertainty_range
                - definition of uncertainty metric

        Notes
        -----

        - Designed as the primary interface between projection outputs and
        reporting/visualization layers.
        - Aggregates both central tendency and uncertainty metrics.
        - Assumes input data is preprocessed and validated.
        - Supports downstream use in `generate_report`.
        """

        import numpy as np

        df = df.copy()

        # Ensure datetime
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # -----------------------------
        # Basic metadata
        # -----------------------------
        climate_models = sorted(df["GCM"].dropna().unique().tolist())
        ssp_scenarios = sorted(df["SSP"].dropna().unique().tolist())

        time_clean = df["time"].dropna()

        if not time_clean.empty:
            proj_start = time_clean.min().strftime("%Y-%m-%d")
            proj_end = time_clean.max().strftime("%Y-%m-%d")
        else:
            proj_start, proj_end = "Unknown", "Unknown"

        projection_period = {
            "start": proj_start,
            "end": proj_end,
            "n_timesteps": int(len(time_clean)),
        }

        # -----------------------------
        # GLOBAL ENSEMBLE MEAN (All GCM + All SSP)
        # -----------------------------
        ensemble_mean_projection = float(df["disease_projection"].mean())
        ensemble_max_projection = float(df["disease_projection"].max())
        ensemble_min_projection = float(df["disease_projection"].min())

        ensemble_uncertainty_mean = float((df["upper_bound"] - df["lower_bound"]).mean())

        # Simple trend (linear slope over time)
        df_sorted = df.sort_values("time")
        x = np.arange(len(df_sorted))
        y = df_sorted["disease_projection"].values

        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0:
                trend = "increasing"
            elif slope < 0:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"

        # Peak months (seasonality)
        if "Month" in df.columns:
            peak_months = (
                df.groupby("Month")["disease_projection"]
                .mean()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
        else:
            peak_months = []

        # -----------------------------
        # SSP-WISE ENSEMBLE SUMMARY
        # -----------------------------
        ssp_summary = {}

        for ssp in ssp_scenarios:
            df_ssp = df[df["SSP"] == ssp]

            if df_ssp.empty:
                continue

            # Ensemble across GCMs for this SSP
            mean_proj = float(df_ssp["disease_projection"].mean())
            max_proj = float(df_ssp["disease_projection"].max())
            min_proj = float(df_ssp["disease_projection"].min())

            mean_uncertainty = float(
                (df_ssp["upper_bound"] - df_ssp["lower_bound"]).mean()
            )

            # Trend per SSP
            df_ssp_sorted = df_ssp.sort_values("time")
            x_ssp = np.arange(len(df_ssp_sorted))
            y_ssp = df_ssp_sorted["disease_projection"].values

            if len(y_ssp) > 1:
                slope_ssp = np.polyfit(x_ssp, y_ssp, 1)[0]
                if slope_ssp > 0:
                    ssp_trend = "increasing"
                elif slope_ssp < 0:
                    ssp_trend = "decreasing"
                else:
                    ssp_trend = "stable"
            else:
                ssp_trend = "insufficient data"

            ssp_summary[ssp] = {
                "mean_projection": mean_proj,
                "max_projection": max_proj,
                "min_projection": min_proj,
                "trend": ssp_trend,
                "mean_uncertainty_range": mean_uncertainty,
            }

        # -----------------------------
        # GCM-WISE SUMMARY
        # -----------------------------
        gcm_summary = {}

        for gcm in climate_models:
            df_gcm = df[df["GCM"] == gcm]
            if df_gcm.empty:
                continue

            gcm_summary[gcm] = {
                "mean_projection": float(df_gcm["disease_projection"].mean()),
                "max_projection": float(df_gcm["disease_projection"].max()),
                "min_projection": float(df_gcm["disease_projection"].min()),
            }

        # -----------------------------
        # TIME-SERIES ENSEMBLE (FOR PLOTTING)
        # -----------------------------
        ensemble_timeseries = []

        df_ts = (
            df.groupby("time")
            .agg({
                "disease_projection": "mean",
                "lower_bound": "mean",
                "upper_bound": "mean",
            })
            .reset_index()
            .sort_values("time")
        )

        for _, row in df_ts.iterrows():
            ensemble_timeseries.append({
                "time": row["time"].strftime("%Y-%m-%d"),
                "mean": float(row["disease_projection"]),
                "lower_bound": float(row["lower_bound"]),
                "upper_bound": float(row["upper_bound"]),
            })

        # -----------------------------
        # SSP-WISE TIME-SERIES (FOR GRID)
        # -----------------------------
        ssp_timeseries = {}

        for ssp in ssp_scenarios:
            df_ssp = df[df["SSP"] == ssp]

            df_ssp_ts = (
                df_ssp.groupby("time")
                .agg({
                    "disease_projection": "mean",
                    "lower_bound": "mean",
                    "upper_bound": "mean",
                })
                .reset_index()
                .sort_values("time")
            )

            ssp_timeseries[ssp] = [
                {
                    "time": row["time"].strftime("%Y-%m-%d"),
                    "mean": float(row["disease_projection"]),
                    "lower_bound": float(row["lower_bound"]),
                    "upper_bound": float(row["upper_bound"]),
                }
                for _, row in df_ssp_ts.iterrows()
            ]

        # -----------------------------
        # DUAL-BASELINE RISK MATRIX (PROBABILISTIC)
        # -----------------------------
        risk_matrix = []

        df_risk = self.flag_outbreak_risk(
            df.copy(),
            method="both",
            percentile=0.9
        )

        if "risk_flag_combined" in df_risk.columns:

            # Convert boolean → int
            df_risk["risk_numeric"] = df_risk["risk_flag_combined"].astype(int)

            # KEY STEP: probability across GCMs
            prob_df = (
                df_risk
                .groupby(["time", "SSP"])["risk_numeric"]
                .mean()
                .reset_index()
            )

            # Map probability → risk categories
            def classify_risk(p):
                if p >= 0.66:
                    return 2   # High
                elif p >= 0.33:
                    return 1   # Elevated
                else:
                    return 0   # Normal

            prob_df["risk"] = prob_df["risk_numeric"].apply(classify_risk)

            risk_matrix = [
                {
                    "time": row["time"].strftime("%Y-%m-%d"),
                    "SSP": row["SSP"],
                    "risk": int(row["risk"]),
                    "probability": float(row["risk_numeric"])  # 🔥 NEW
                }
                for _, row in prob_df.iterrows()
            ]

        # -----------------------------
        # FINAL STRUCTURED SUMMARY
        # -----------------------------
        projection_summary = {
            "mode": "future_climate_projection",
            "method": "CMIP6 climate-driven disease projection",
            "projection_period": projection_period,
            "climate_models": climate_models,
            "ssp_scenarios": ssp_scenarios,
            "ensemble_mean": {
                "mean_projection": ensemble_mean_projection,
                "max_projection": ensemble_max_projection,
                "min_projection": ensemble_min_projection,
                "trend": trend,
                "peak_transmission_months": peak_months,
            },
            "ssp_ensemble": ssp_summary,
            "ensemble_timeseries": ensemble_timeseries,
            "ssp_timeseries": ssp_timeseries,
            "gcm_summary": gcm_summary,
            "risk_matrix": risk_matrix,
            "uncertainty": {
                "mean_uncertainty_range": ensemble_uncertainty_mean,
                "definition": "Average (upper_bound - lower_bound)",
            },
        }

        return projection_summary


    # ------------------------------------------------------
    #  Export results to csv/excel with prediction intervals
    # ------------------------------------------------------
    def export_tidy_projections(
    self,
    df: pd.DataFrame,
    projection_summary: dict,
    path: str,
    saveformat: str = "csv",
    ):
        """
        Export projections + probabilistic risk in tidy (long) format.

        This creates a unified dataset combining:
        - Raw projections (per GCM)
        - Ensemble probability risk (per time, SSP)

        Suitable for dashboards, Power BI, Tableau, and downstream analytics.

        Parameters
        ----------
        df : pandas.DataFrame
            Output of project_multi_model_ssp()

        projection_summary : dict
            Output of build_projection_summary()

        path : str
            Output file path

        saveformat : {"csv", "excel"}
        """

        import pandas as pd

        # -----------------------------
        # Validate input
        # -----------------------------
        required_cols = [
            "GCM", "SSP", "Year", "time",
            "disease_projection", "lower_bound", "upper_bound"
        ]

        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")

        df = df.copy()

        # -----------------------------
        # Prepare projection data (long format)
        # -----------------------------
        proj_long = df.copy()

        proj_long["variable"] = "projection"
        proj_long["value"] = proj_long["disease_projection"]

        proj_long = proj_long[
            ["time", "Year", "SSP", "GCM", "variable", "value",
            "lower_bound", "upper_bound"]
        ]

        # -----------------------------
        # Prepare probability data
        # -----------------------------
        risk_data = projection_summary.get("risk_matrix", [])

        if not risk_data:
            raise ValueError("No risk_matrix found in projection_summary")

        risk_df = pd.DataFrame(risk_data)

        risk_df["time"] = pd.to_datetime(risk_df["time"])
        risk_df["Year"] = risk_df["time"].dt.year

        risk_df["variable"] = "outbreak_probability"
        risk_df["value"] = risk_df["probability"]

        # No GCM (aggregated)
        risk_df["GCM"] = "ensemble"

        risk_long = risk_df[
            ["time", "Year", "SSP", "GCM", "variable", "value"]
        ]

        # -----------------------------
        # Combine datasets
        # -----------------------------
        tidy_df = pd.concat([proj_long, risk_long], ignore_index=True)

        # -----------------------------
        # Sort for readability
        # -----------------------------
        tidy_df = tidy_df.sort_values(["SSP", "GCM", "time"])

        # -----------------------------
        # Export
        # -----------------------------
        if path:
            if saveformat == "csv":
                tidy_df.to_csv(path, index=False)
            elif saveformat == "excel":
                tidy_df.to_excel(path, index=False)

            print(f"Saved tidy projections → {path}")

        return tidy_df