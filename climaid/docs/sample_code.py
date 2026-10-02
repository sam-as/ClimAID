"""
ClimAID Example Workflow
========================

Author: Avik Sam
Created: January 28, 2026
Website: https://sites.google.com/view/aviksam

This example demonstrates the complete ClimAID v1 pipeline (ClimAID 0.4.1). For the v2
probabilistic forecast and scenario outlook, see DiseaseModel.forecast_v2() and
DiseaseModel.project_v2() in the documentation: https://sam-as.github.io/ClimAID/

1. Load disease and climate data
2. Detect historical outbreaks
3. Optimize lag structure
4. Train the final prediction model
5. Generate climate-driven disease projections
6. Visualize projections
7. Generate an automated climate risk report
"""

# -------------------------------------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------------------------------------

from climaid.climaid_model import DiseaseModel
from climaid.climaid_projections import DiseaseProjection
from climaid.projection_plots import DiseaseVisualizer
from climaid.reporting import DiseaseReporter
from climaid.llm_client import LocalOllamaLLM


# -------------------------------------------------------------
# 2. INITIALIZE DISEASE MODEL
# -------------------------------------------------------------

dm = DiseaseModel(
    district="IND_Pune_MAHARASHTRA",
    disease_file="dengue_data.xlsx",   # path to your disease data (Excel or CSV)
    disease_name="Dengue",
    random_state=42,
)

# Inspect merged disease–climate dataset
print(dm.df_merged.head())


# -------------------------------------------------------------
# 3. DETECT HISTORICAL OUTBREAKS
# -------------------------------------------------------------

outbreaks = dm.detect_historical_outbreaks()
print(outbreaks)


# -------------------------------------------------------------
# 4. OPTIMIZE LAG STRUCTURE
# -------------------------------------------------------------

feature_metadata, lag_search_result, best_config = dm.optimize_lags(
    base_models=("xgb", ),
    residual_models=("rf", ),
    correction_models=("isotonic",),
    debug=True,
    n_jobs=-1,  
    n_trials=5,
)

print(lag_search_result)
print(best_config)


# -------------------------------------------------------------
# 5. TRAIN FINAL MODEL
# -------------------------------------------------------------

final_output = dm.train_final_model()

print("Final Test R²:", final_output["test_r2"])
print("Final Test RMSE:", final_output["test_rmse"])

predictions = final_output["predictions"]
predictions = predictions.rename({"Date": "time"}, axis=1)


# -------------------------------------------------------------
# 6. HISTORICAL PREDICTION PLOTS
# -------------------------------------------------------------

dm.plot_historical_predictions()


# -------------------------------------------------------------
# 7. LOAD CLIMATE PROJECTIONS
# -------------------------------------------------------------

future_climate = dm.df_climate_proj

dp = DiseaseProjection(dm)

future_features = dp.prepare_features(future_climate)

print("Available GCMs:", future_features["model"].unique().tolist())
print("Available SSPs:", future_features["ssp"].unique().tolist())


# -------------------------------------------------------------
# 8. GENERATE DISEASE PROJECTIONS
# -------------------------------------------------------------

# Single model projection
proj_single = dp.project(
    model_name="ACCESS-ESM1-5",
    ssp="ssp126",
    data=future_features
)

# Multiple model projection
proj_models = dp.project_model_list(
    model_list=["ACCESS-ESM1-5", "CNRM-CM6-1", "GFDL-ESM4", "MRI-ESM2-0"],
    ssp="ssp126"
)

# Ensemble projection
proj_ensemble = dp.project_ensemble_mean(
    model_list=["ACCESS-ESM1-5", "CNRM-CM6-1", "GFDL-ESM4", "MRI-ESM2-0"],
    ssp="ssp126"
)

# Full multi-model / multi-SSP projections
proj_all = dp.project_multi_model_ssp(
    model_list=["ACCESS-ESM1-5", "CNRM-CM6-1", "GFDL-ESM4", "MRI-ESM2-0"],
    ssp_list=["ssp126", "ssp245", "ssp370", "ssp585"]
)


# -------------------------------------------------------------
# 9. OUTBREAK RISK FLAGGING
# -------------------------------------------------------------

risk_df = dp.flag_outbreak_risk(
    proj_all,
    method="both",
    percentile=0.9,
)

historical_risk = risk_df["risk_flag_historical_baseline"].sum()
dynamic_risk = risk_df["risk_flag_dynamic_baseline"].sum()


# -------------------------------------------------------------
# 10. BUILD PROJECTION SUMMARY
# -------------------------------------------------------------

projection_summary = dp.build_projection_summary(proj_all)

artifacts = dm.build_report_artifacts(projection_summary)

print(artifacts.keys())


# -------------------------------------------------------------
# 11. VISUALIZE PROJECTIONS
# -------------------------------------------------------------

viz = DiseaseVisualizer(proj_all)

# Heatmap
viz.plot_heatmap("ACCESS-ESM1-5", "ssp585")

# Projection grid
viz.plot_projection_grid(
    ["ACCESS-ESM1-5", "CNRM-CM6-1", "GFDL-ESM4", "MRI-ESM2-0"],
    ["ssp126", "ssp245", "ssp370", "ssp585"],
    linecolor="#3a5a40",
    shadecolor="#dad7cd",
)


# -------------------------------------------------------------
# 12. GENERATE CLIMATE RISK REPORT
# -------------------------------------------------------------

llm = LocalOllamaLLM(model="phi3")

reporter = DiseaseReporter(llm_client=llm)

response = llm.generate("Explain disease climate risk in three sentences.")
print(response)


# Generate automated policy report
report = dm.generate_report(
    projection_summary=projection_summary,
    llm_client=llm,
    style="policy",
    open_browser=True,
)

print(report)


# -------------------------------------------------------------
# 13. PRINT RUNTIME SUMMARY
# -------------------------------------------------------------

dm.print_runtime_summary()