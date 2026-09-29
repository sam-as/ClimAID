# ClimAID: An AI-integrated Reproducible Climate-Driven Modelling Framework for Multi-Disease Risk Projection under CMIP6 Climate Scenarios

**Version 0.4.0 (beta, under active testing).** Full, current documentation:
[https://sam-as.github.io/ClimAID/](https://sam-as.github.io/ClimAID/), or offline with `climaid docs`.
Methods have been checked on synthetic data only; results on real data are not yet validated.

ClimAID (Climate change impact using AI on Diseases) is an offline-first, scientific Python library for climate-driven disease modelling, forecasting, and future risk projection using CMIP6 climate scenarios.

Originally developed for dengue modelling in India, ClimAID is now a generalized framework for climate-sensitive infectious diseases including malaria, dengue, and other vector-borne diseases.

Designed for:

    * Epidemiologists
    * Climate scientists
    * Public health researchers
    * Policy analysts
    * Government agencies

---

## Key Features

### Climate–Disease Modelling (Research-Grade)

    * Stacked machine learning architecture (Base → Residual → Correction)
    * Lag optimization using Optuna
    * Support for nonlinear climate–disease relationships
    * Epidemiology-aware modelling (Poisson, ElasticNet, etc.)

### CMIP6 Climate Projections

    * Multi-GCM support (ACCESS, GFDL, CNRM, MRI, etc.)
    * Multi-SSP scenarios (SSP126, SSP245, SSP370, SSP585)
    * Ensemble mean projections
    * District-level future disease risk estimation

### Fully Automated Scientific Reporting

    * One-command report generation
    * Publication-quality HTML dashboards
    * Climate + epidemiological interpretation
    * Deterministic fallback if LLM unavailable
    * Offline local LLM support (Ollama)

### Offline-First Design

    * No cloud API dependency
    * Works in restricted environments (HPC, government systems)
    * Local LLM integration (Phi-3, Mistral via Ollama)

### Interactive CLI Wizard

Run the full pipeline with guided prompts:

    * Lag optimization
    * Model training
    * Climate projections
    * Visualization
    * Automated report generation

---

## Installation

```bash
pip install climaid
```

Optional gradient-boosting libraries (XGBoost, LightGBM, CatBoost):

```bash
pip install "climaid[ml]"
```

Optional local LLM reports: install [Ollama](https://ollama.com) and run `ollama serve` (no extra Python
package is needed).

---

## Quick Start (Interactive Wizard)

Run the complete climate-disease modelling pipeline:

```bash
climaid wizard     # terminal wizard (choose v1, v2 or both)
climaid browse     # browser dashboard
```

The wizard will:

    1. Validate district selection
    2. Optimize climate lags
    3. Train Hybrid disease model
    4. Generate CMIP6 projections
    5. Create scientific visualizations
    6. Produce automated policy and research reports

---

## Programmatic Usage (Advanced Users)

```python
from climaid.climaid_model import DiseaseModel

dm = DiseaseModel(
    district="IND_Pune_MAHARASHTRA",
    disease_file="dengue_data.xlsx",
    disease_name="Dengue",
    random_state=42,
)

# v2: probabilistic forecast with calibrated likely ranges
result = dm.forecast_v2(forecast_origin="2023-12-31", horizon=12, save_report=True)

# v2: climate-scenario outlook
outlook = dm.project_v2(end_year=2050, ssps=["ssp245", "ssp585"])

# v1: optimise lag structure and train the stacked model
feature_metadata, lag_search_result, best_config = dm.optimize_lags()
final_out = dm.train_final_model()

# v1: report (projection_summary from DiseaseProjection.build_projection_summary; see sample_code.py)
report = dm.generate_report(
    projection_summary=projection_summary,
    style="policy",
    open_browser=True,
)
```

---

## Supported Modelling Capabilities

    * Random Forest (rf)
    * XGBoost (xgb)
    * ExtraTrees 
    * LightGBM (lgbm)
    * CatBoost
    * ElasticNet / Lasso / Ridge
    * Poisson Regression (epidemiology-friendly)
    * Gradient Boosting
    * Neural Networks (mlp/nn)
    * Isotonic calibration layer

ClimAID v2 adds a climate renewal (transmission) model, a seasonal-naive benchmark, Tweedie,
spline Poisson, Bayesian ridge, Huber, histogram gradient boosting, SVR and nearest neighbours
(21 v2 models in total), each tuned automatically.

---

## Scientific Use Cases

    * Climate change impact on dengue transmission
    * Malaria early warning systems
    * District-level climate risk forecasting for infectious diseases
    * Public health adaptation planning
    * Climate-health vulnerability assessment

---

## Output Artifacts

ClimAID automatically generates:

    * Model diagnostics (R², RMSE)
    * Selected climate lags
    * Feature importance
    * Projection summaries (CMIP6)
    * Interactive HTML dashboard
    * Policy-ready reports

---

## Project Structure

```
climaid/
    ├── climaid_model.py 
    ├── climaid_projections.py
    ├── climate_data.py
    ├── reporting.py
    ├── llm_client.py
    ├── projection_plots.py
    ├── wizard.py
    ├── districts.py
    ├── model_parameters.py
    ├── model_registry.py
    ├── utils.py
    ├── cli.py                  (climaid browse | wizard | docs)
    ├── exclusion.py            (COVID-19 period handling)
    ├── reporting_v2.py, reporting_scenario.py, reporting_plain.py
    ├── forecasting_v2/         (ClimAID v2 engine)
    ├── browser_ui/             (dashboard)
    ├── documentation/          (bundled documentation site)
    └── data/
```

---

## AI-Integrated Climate–Disease Modelling and Reporting

ClimAID is an AI-integrated framework that combines machine learning, automated optimization, and offline large language models (LLMs) to support climate-driven infectious disease analysis and interpretation.

### Multi-Layer AI Architecture

ClimAID integrates AI across four core layers:

1. Machine learning models for nonlinear climate–disease prediction
2. Automated lag and model optimization (AutoML via Optuna)
3. Calibration-enhanced stacked learning for epidemiological realism
4. Offline LLM-based automated scientific and policy reporting

This design enables both high-performance modelling and interpretable outputs suitable for research and policy use.

---

## Offline LLM-Based Automated Report Generation

ClimAID includes a built-in AI reporting engine that automatically converts model outputs and climate projections into structured scientific and policy narratives.

Key capabilities:

* Climate risk interpretation from CMIP6 projections
* District-level disease risk summaries
* Policy-style outputs
* Publication-ready HTML dashboards
* Deterministic fallback reports (no LLM dependency)

The LLM module operates fully offline using local models (e.g., Phi-3, Mistral via Ollama), ensuring:

* Data privacy
* Secure institutional deployment
* Reproducible report generation
* No external API requirements

Example usage:

```python
from climaid.llm_client import LocalOllamaLLM

llm = LocalOllamaLLM(model="phi3")

report = dm.generate_report(
    projection_summary=projection_summary,
    llm_client=llm,
    style="policy",
    open_browser=True
)
```

If the LLM is unavailable, ClimAID automatically generates deterministic scientific reports using model artifacts, ensuring reproducibility across environments.

## Citation (Forthcoming Manuscript)

If you use ClimAID in academic work, please cite:

> Sam, A.K., Phuleria, H.C. (2026). ClimAID: An AI-integrated Global Hybrid Climate-Disease Modelling Framework. Preprint: [https://doi.org/10.21203/rs.3.rs-9394047/v1](https://doi.org/10.21203/rs.3.rs-9394047/v1)

---

## License

MIT License

---

## Author

Avik Kumar Sam
Indian Institute of Technology Bombay
