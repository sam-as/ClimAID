# ClimAID: An AI-integrated Reproducible Climate-Driven Modelling Framework for Multi-Disease Risk Projection under CMIP6 Climate Scenarios 

ClimAID (Climate Change Impact on Infectious Diseases Toolkit for India) is an offline-first, scientific Python library for climate-driven disease modelling, forecasting, and future risk projection using CMIP6 climate scenarios.

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

Optional (for full features including local LLM):

```bash
pip install climaid[full]
```

---

## Quick Start (Interactive Wizard)

Run the complete climate-disease modelling pipeline:

```bash
    climaid
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
        district="Pune_MAHARASHTRA",
        disease_file="dengue_data.xlsx",
        disease_name="Dengue",
        random_state=42
    )

    # Optimize lag structure
    lag_result, best_config = dm.optimize_lags()

    # Train final stacked model
    final_out = dm.train_final_model()

    # Generate projections and report
    report = dm.generate_report(
        projection_summary=projection_summary,
        style="policy_brief",
        open_browser=True
    )
```

---

## Supported Modelling Capabilities

    * Random Forest (rf)
    * XGBoost (xgb)
    * ExtraTrees 
    * LightGBM (lgbm)
    * Gradient Boosting Regreesion
    * CatBoost
    * ElasticNet / Lasso / Ridge
    * Poisson Regression (epidemiology-friendly)
    * Gradient Boosting
    * Neural Networks (mlp/nn)
    * Isotonic calibration layer

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
    ├── browse.py
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
* Policy brief–style outputs
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
    style="policy_brief",
    open_browser=True
)
```

If the LLM is unavailable, ClimAID automatically generates deterministic scientific reports using model artifacts, ensuring reproducibility across environments.

## Citation (Forthcoming Manuscript)

If you use ClimAID in academic work, please cite:

> Sam, A.K., Pathak, M., Phuleria, H.C. (2026). ClimAID: A Climate-Driven Disease Modelling Framework using CMIP6 Projections.

---

## License

MIT License

---

## Author

Avik Kumar Sam
Indian Institute of Technology Bombay
