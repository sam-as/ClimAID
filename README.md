# ClimAID - **Climate Change impact using AI on Diseases**

ClimAID is an integrated toolkit for modeling, forecasting, and projecting climate-sensitive diseases such as dengue and malaria using machine learning and climate model ensembles.

* ClimAID has inbuilt climate data for South Asian countries, namely India, Nepal, Bhutan, Sri Lanka, Myanmar, Afghanistan, Pakistan and Bangladesh. 

* ClimAID support data from other countries through the global mode on the browser interface. 

---

## What you can do

* Analyze historical disease patterns
* Integrate climate variables (temperature, rainfall, humidity)
* Automatically detect optimal lag effects using AutoML
* Train hybrid ML models
* Generate CMIP6-based future projections
* Identify outbreak risk under climate change scenarios
* Generate automated policy reports using the integrated C-DSI or local LLM models. 

---

## Quick Example using Codes

```python
from climaid.climaid_model import DiseaseModel

dm = DiseaseModel(
    district='IND_Mumbai_MAHARASHTRA',
    disease_file="dengue_data.xlsx",
    disease_name="Dengue"
)

dm.optimize_lags()
dm.train_final_model()
```

---

## Workflow

ClimAID has two interfaces, 

* **ClimAID Browser Interface** (For both South Asian + Global countries)
    
    For initialisation through terminal, use
    ```text
        climaid browse
    ```

* **ClimAID Wizard Interface** (For South Asian countries)
    
    For initialisation through terminal, use
    ```text
        climaid wizard
    ```
---

## Documentation

The full documentation is available here: [https://sam-as.github.io/ClimAID/](https://sam-as.github.io/ClimAID/) 

---

## Designed for

* Epidemiologists
* Climate scientists
* Public health analysts
* Data scientists

---

## Dependencies & License

### Dependencies

* Core Requirements

    ```bash
    pandas  
    numpy  
    geopandas  
    matplotlib  
    scikit-learn  
    xarray  
    regionmask  
    plotly  
    xgboost  
    optuna
    ```  

* Additional Utilities

    These packages support extended functionality and will be 
    auto-installed. 

    ```bash
    requests
    joblib
    fastapi
    uvicorn
    typer
    markdown
    fastparquet
    python-multipart
    seaborn
    openpyxl
    ```

* Optional (LLM Support)

    To enable local LLM-based report generation:

    ```bash
    pip install climaid[full]
    ```

    Includes:
    ```bash
    ollama 
    ``` 

---

### License

Designed by **Avik Kumar Sam** & **Harish C. Phuleria** as an open-access software. 

* MIT License Summary
    - Free to use, modify, and distribute  
    - Suitable for research and commercial use  
    - No warranty is provided  
    - Attribution is required  

* Full License Text
    - See the complete license here: [https://github.com/sam-as/ClimAID/blob/main/LICENSE](https://github.com/sam-as/ClimAID/blob/main/LICENSE)

---
