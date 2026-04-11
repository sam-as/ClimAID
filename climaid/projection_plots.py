"""
ClimAID Projection Plotting Module
-----------------------------------

Seaborn-based visualisation tools for disease climate projections.

Designed for output of:
    dp.project_multi_model_ssp()

Expected columns:
    model | ssp | Year | time |
    disease_projection | lower_bound | upper_bound
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class DiseaseVisualizer:
    def __init__(self, df):
        """
        Initialize with the dataframe and apply global beautification settings.
        """
        self.df = df.copy()
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.df["Month"] = self.df["time"].dt.month
        
        # Consistent styling parameters
        self.blue = "#08306b"
        self.red = "#67000d"
        self.palette = ["#4292c6", "#ef3b2c", "#6a51a3", "#238b45"]
        
        # Set the global theme
        sns.set_theme(style="white", font_scale=1.2)

    def _filter(self, models=None, ssps=None):
        """Filter data and report unavailable model/SSP combinations."""

        df_filtered = self.df.copy()

        # -------------------------
        # AVAILABLE VALUES
        # -------------------------
        available_models = set(self.df["model"].unique())
        available_ssps = set(self.df["ssp"].unique())

        # -------------------------
        # CHECK INVALID INPUTS
        # -------------------------
        if models:
            invalid_models = set(models) - available_models
        else:
            invalid_models = set()

        if ssps:
            invalid_ssps = set(ssps) - available_ssps
        else:
            invalid_ssps = set()

        # -------------------------
        # PRINT INVALID ENTRIES
        # -------------------------
        if invalid_models:
            print(f"Invalid model(s): {sorted(invalid_models)}")

        if invalid_ssps:
            print(f"Invalid SSP(s): {sorted(invalid_ssps)}")

        # -------------------------
        # FILTER VALID ONES ONLY
        # -------------------------
        if models:
            valid_models = list(set(models) & available_models)
            df_filtered = df_filtered[df_filtered["model"].isin(valid_models)]

        if ssps:
            valid_ssps = list(set(ssps) & available_ssps)
            df_filtered = df_filtered[df_filtered["ssp"].isin(valid_ssps)]

        # -------------------------
        # CHECK COMBINATIONS
        # -------------------------
        if models and ssps:
            available_combos = set(
                zip(self.df["model"], self.df["ssp"])
            )

            requested_combos = set(
                (m, s) for m in models for s in ssps
            )

            invalid_combos = requested_combos - available_combos

            if invalid_combos:
                print("Invalid model–SSP combinations:")
                for combo in sorted(invalid_combos):
                    print(f"   - {combo[0]} + {combo[1]}")

        # -------------------------
        # FINAL CHECK
        # -------------------------
        if df_filtered.empty:
            raise ValueError(
                "No valid data available for the requested filters.\n"
                "Please check model and SSP combinations."
            )

        return df_filtered
        
    def plot_projection_grid(self, models=None, ssps=None, linecolor=None, shadecolor=None):
        """Grid of SSP x Model (only valid combinations plotted)."""

        df = self._filter(models, ssps)

        if df.empty:
            print("No data available for the selected filters.")
            return

        if linecolor is None or shadecolor is None:
            linecolor = "#4281a4"
            shadecolor = "#f2e8cf"

        # Detect correct SSP column
        ssp_col = "scenario" if "scenario" in df.columns else "ssp"

        # -------------------------
        # KEEP ONLY VALID COMBINATIONS
        # -------------------------
        valid_combos = df[[ssp_col, "model"]].drop_duplicates()

        # Inner merge ensures only valid (model, ssp) remain
        df = df.merge(valid_combos, on=[ssp_col, "model"], how="inner")

        # -------------------------
        # CREATE GRID (ONLY VALID DATA)
        # -------------------------
        g = sns.FacetGrid(
            df,
            row=ssp_col,
            col="model",
            height=4,
            aspect=1.4,
            margin_titles=True,
            dropna=True
        )

        # -------------------------
        # LINE PLOT
        # -------------------------
        g.map_dataframe(
            sns.lineplot,
            x="time",
            y="disease_projection",
            color=linecolor,
            linewidth=1
        )

        # -------------------------
        # UNCERTAINTY SHADING
        # -------------------------
        for (row_val, col_val), ax in g.axes_dict.items():

            subdf = df[
                (df[ssp_col] == row_val) &
                (df["model"] == col_val)
            ].sort_values("time")

            if subdf.empty:
                continue

            if {"lower_bound", "upper_bound"}.issubset(subdf.columns):
                ax.fill_between(
                    subdf["time"],
                    subdf["lower_bound"],
                    subdf["upper_bound"],
                    color=shadecolor,
                    alpha=0.5
                )

            ax.tick_params(left=True, bottom=True)
            sns.despine(ax=ax)

        # -------------------------
        # LABELS
        # -------------------------
        g.set_axis_labels("Year", "Disease Cases", fontweight='bold')

        # Rotate x labels
        for ax in g.axes.flatten():
            if ax is not None:
                ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, model, ssp, cmap=None):
        """Heatmap with intelligent auto-selection based on data availability."""

        # Detect correct SSP column
        ssp_col = "scenario" if "scenario" in self.df.columns else "ssp"

        df = self.df.copy()

        # -------------------------
        # REMOVE EMPTY COMBINATIONS
        # -------------------------
        combo_counts = (
            df.groupby(["model", ssp_col])["disease_projection"]
            .count()
            .reset_index(name="n")
        )

        # Keep only combinations with actual data
        valid_combos = combo_counts[combo_counts["n"] > 0]

        # -------------------------
        # CHECK REQUESTED COMBO
        # -------------------------
        requested_mask = (
            (valid_combos["model"] == model) &
            (valid_combos[ssp_col] == ssp)
        )

        if requested_mask.any():
            selected_model = model
            selected_ssp = ssp

        else:
            print(f"Requested combination not available: {model} | {ssp}")

            # -------------------------
            # PRIORITY 1: SAME MODEL
            # -------------------------
            same_model = valid_combos[valid_combos["model"] == model]

            if not same_model.empty:
                # choose SSP with max data
                best = same_model.sort_values("n", ascending=False).iloc[0]
                selected_model = best["model"]
                selected_ssp = best[ssp_col]

                print(f"Auto-selected (same model): {selected_model} | {selected_ssp}")

            # -------------------------
            # PRIORITY 2: SAME SSP
            # -------------------------
            else:
                same_ssp = valid_combos[valid_combos[ssp_col] == ssp]

                if not same_ssp.empty:
                    best = same_ssp.sort_values("n", ascending=False).iloc[0]
                    selected_model = best["model"]
                    selected_ssp = best[ssp_col]

                    print(f"Auto-selected (same SSP): {selected_model} | {selected_ssp}")

                # -------------------------
                # PRIORITY 3: BEST OVERALL
                # -------------------------
                else:
                    best = valid_combos.sort_values("n", ascending=False).iloc[0]
                    selected_model = best["model"]
                    selected_ssp = best[ssp_col]

                    print(f"Auto-selected (best available): {selected_model} | {selected_ssp}")

        # -------------------------
        # FILTER FINAL DATA
        # -------------------------
        df = df[
            (df["model"] == selected_model) &
            (df[ssp_col] == selected_ssp)
        ]

        if df.empty:
            print("No data available after auto-selection.")
            return

        # -------------------------
        # PIVOT
        # -------------------------
        heat = df.pivot(index="Year", columns="Month", values="disease_projection")

        if heat.empty:
            print("No plottable data after auto-selection.")
            return

        heat.index = heat.index.astype(int)

        sns.set_theme(style='ticks', font_scale=1.2)
        plt.figure(figsize=(7, 14))

        # -------------------------
        # DEFAULT COLORMAP
        # -------------------------
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list(
                "custom_cpal",
                ['#' + i for i in [
                    "ffdcc2","ffd1ad","ffcea1","ffc599","fac398","eda268",
                    "ff750a","da7e37","c06722","a85311","8f3e00","713200",
                    "522500","291200"
                ]]
            )

        # -------------------------
        # HEATMAP
        # -------------------------
        ax = sns.heatmap(
            heat,
            cmap=cmap,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Projected Cases', "shrink": 0.5},
        )

        # -------------------------
        # STYLE
        # -------------------------
        plt.axhline(y=len(heat.index), linestyle='-.', linewidth=2, color='#522500')
        plt.axvline(x=0, linestyle='-.', linewidth=2, color='#522500')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.xticks(color='#522500')
        plt.yticks(color='#522500')

        # -------------------------
        # LABELS
        # -------------------------
        plt.title(f"{selected_model} | {selected_ssp}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Month", fontweight='bold')
        plt.ylabel("Year", fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_distribution_with_kde(self, var):
        """Uses the ax.hist() method for thick adjacent bars."""
        data = self.df[var].dropna()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Histogram with adjacent bars
        ax.hist(data, bins=20, density=True, color='royalblue', 
                edgecolor='black', linewidth=1.2, rwidth=1.0)
        
        # KDE on twin axis to prevent color interference
        ax_kde = ax.twinx()
        sns.kdeplot(data, ax=ax_kde, color='black', linewidth=2)
        
        # Beautify: Remove right-side 'twin' labels but keep left axis
        ax_kde.set_yticks([])
        ax_kde.set_yticklabels([])
        ax_kde.set_ylabel('')
        
        ax.set_title(f"Distribution of {var}", fontweight='bold')
        sns.despine(right=True)
        plt.tight_layout()
        plt.show()
