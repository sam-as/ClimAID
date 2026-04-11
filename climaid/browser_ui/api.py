"""
api.py
------------

Provides helper functions to call the FastAPI backend
for the ClimAID browser wizard.

Author: Avik Kumar Sam
Created: March 2026
Updated: 2026
"""

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from io import BytesIO
import tempfile
import uuid
from climaid.climaid_model import DiseaseModel
from climaid.climaid_projections import DiseaseProjection
from climaid.districts import get_available_districts
from climaid.model_registry import list_available_models
from .state import wizard_state


# ======================================================
# API Router
# ======================================================
router = APIRouter()

# ======================================================
# State → District catalog
# ======================================================
@router.get("/district_catalog")
def district_catalog():
    """
    Returns country → state → district mapping
    for populating dropdowns in the UI.
    """

    districts = get_available_districts()

    data = []

    for d in districts:

        parts = d.split("_")

        country = parts[0] if len(parts) >= 1 else "Unknown"
        district = parts[1] if len(parts) >= 2 else "Unknown"
        state = parts[2] if len(parts) >= 3 else "Unknown"

        data.append({
            "country": country,
            "state": state,
            "district": district
        })

    df = pd.DataFrame(data)

    result = {}

    for (country, state), group in df.groupby(["country", "state"]):
        result.setdefault(country, {})[state] = sorted(group["district"].tolist())

    return result

# ======================================================
# Disease Dataset Upload (CSV / Excel)
# ======================================================
@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile):

    # --------------------------------------------------
    # Read uploaded file contents into memory
    # --------------------------------------------------

    contents = await file.read()

    try:

        # --------------------------------------------------
        # Handle CSV files
        # --------------------------------------------------
        if file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        # --------------------------------------------------
        # Handle Excel files (.xlsx / .xls)
        # --------------------------------------------------
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(contents))

        else:
            return {"error": "Unsupported file format"}

        # --------------------------------------------------
        # Store dataset and filename in wizard state
        # --------------------------------------------------

        wizard_state["dataset"] = df
        wizard_state["filename"] = file.filename

        return {
            "status": "uploaded",
            "rows": len(df),
            "columns": list(df.columns)
        }

    except Exception as e:
        return {"error": str(e)}

# ======================================================
# Weather Dataset Upload in Global Mode
# ======================================================
@router.post("/upload_weather")
async def upload_weather(file: UploadFile):

    contents = await file.read()

    ext = Path(file.filename).suffix
    temp_file = Path(tempfile.gettempdir()) / f"weather_{uuid.uuid4().hex}{ext}"

    with open(temp_file, "wb") as f:
        f.write(contents)

    wizard_state["weather_file"] = str(temp_file)

    return {"status": "weather uploaded"}

# ======================================================
# Projection Dataset Upload in Global Mode
# ======================================================
@router.post("/upload_projection")
async def upload_projection(file: UploadFile):

    contents = await file.read()

    ext = Path(file.filename).suffix
    temp_file = Path(tempfile.gettempdir()) / f"projection_{uuid.uuid4().hex}{ext}"

    with open(temp_file, "wb") as f:
        f.write(contents)

    wizard_state["projection_file"] = str(temp_file)

    return {"status": "projection uploaded"}

# ======================================================
# Wizard Configuration Schema
# ======================================================
class WizardConfig(BaseModel):
    mode: str
    country: str
    district: str
    state: str
    disease_name: str
    preset: str
    test_year: int | None = None

    base_models: list[str] | None = None
    residual_models: list[str] | None = None
    correction_models: list[str] | None = None
    n_trials: int | None = None

# ======================================================
# Available Model Registry
# ======================================================
@router.get("/available_models")
def available_models():
    """
    Returns available models registered in ClimAID.
    """

    return {
        "models": list_available_models()
    }

# ======================================================
# Run ClimAID Pipeline
# ======================================================
@router.post("/run")
def run_pipeline(cfg: WizardConfig):

    # --------------------------------------------------
    # Retrieve uploaded dataset from wizard state
    # --------------------------------------------------

    df = wizard_state.get("dataset")
    filename = wizard_state.get("filename")

    if df is None:
        return {"error": "No dataset uploaded"}

    # --------------------------------------------------
    # Determine original file extension
    # This ensures compatibility with CSV and Excel
    # --------------------------------------------------

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        ext = ".xlsx"
    else:
        ext = ".csv"

    # --------------------------------------------------
    # Create a unique temporary file
    # Prevents overwriting when multiple runs occur
    # --------------------------------------------------

    temp_file = Path(tempfile.gettempdir()) / f"climaid_{uuid.uuid4().hex}{ext}"

    # --------------------------------------------------
    # Save dataframe to temporary file
    # Using correct pandas writer
    # --------------------------------------------------

    if ext == ".csv":
        df.to_csv(temp_file, index=False)

    else:
        df.to_excel(temp_file, index=False)

    # --------------------------------------------------
    # Checking the type of mode
    # Defining the weather and projections file
    # --------------------------------------------------

    mode = cfg.mode.lower()

    weather_file = wizard_state.get("weather_file")
    projection_file = wizard_state.get("projection_file")

    if mode == "global":
        if not weather_file:
            return {"error": "Global mode requires weather dataset"}
    else:
        # South Asia mode → ignore uploaded files
        weather_file = None
        projection_file = None

    # --------------------------------------------------
    # Reconstruct ClimAID district identifier
    # expected format: district_state
    # --------------------------------------------------
    district_key = f"{cfg.country.upper()}_{cfg.district.title()}_{cfg.state.upper()}"

    # --------------------------------------------------
    # Initialize DiseaseModel
    # DiseaseModel requires a file path
    # --------------------------------------------------

    dm = DiseaseModel(
        district=district_key,
        disease_file=str(temp_file),
        disease_name=cfg.disease_name,
        random_state=42,
        weather_file=weather_file,
        projection_file=projection_file, 
    )

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------

    if cfg.test_year:
        dm._train_test_split(test_year=cfg.test_year)
    else:
        dm._train_test_split()

    # --------------------------------------------------
    # Configure optimization preset
    # --------------------------------------------------
    if cfg.preset == "fast":

        base = ("xgb",)
        residual = ("rf",)
        corr = ("isotonic",)
        trials = 50


    elif cfg.preset == "deep":

        base = ("rf", "xgb")
        residual = ("xgb", "rf", "extra_trees")
        corr = ("isotonic", "poisson", "elasticnet")
        trials = 500
        

    elif cfg.preset == "custom":

        # User-defined configuration from browser wizard

        base = tuple(cfg.base_models or ["xgb"])
        residual = tuple(cfg.residual_models or ["rf"])
        corr = tuple(cfg.correction_models or ["isotonic"])
        trials = cfg.n_trials if cfg.n_trials is not None else 200


    else:

        # Balanced (default)

        base = ("xgb",)
        residual = ("rf",)
        corr = ("isotonic",)
        trials = 200

    # --------------------------------------------------
    # Run lag optimization
    # --------------------------------------------------

    dm.optimize_lags(
        base_models=base,
        residual_models=residual,
        correction_models=corr,
        n_trials=trials,
        n_jobs=-1,
    )

    # --------------------------------------------------
    # Train final stacked model
    # --------------------------------------------------

    dm.train_final_model()

    # --------------------------------------------------
    # Climate projections using CMIP6 models
    # --------------------------------------------------

    dp = DiseaseProjection(dm)

    dpro = dp.project_multi_model_ssp(
        model_list=['ACCESS-ESM1-5', 'CESM2', 
                    'CNRM-CM6-1', 'GFDL-ESM4',
                    'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 
                    'MPI-ESM1-2-HR', 'MRI-ESM2-0',
                    'NorESM2-LM', 'UKESM1-0-LL'],
        ssp_list=[
            "ssp126",
            "ssp245",
            "ssp370",
            "ssp585",
        ],
    )

    # --------------------------------------------------
    # Build projection summary
    # --------------------------------------------------

    projection_summary = dp.build_projection_summary(dpro)

    # --------------------------------------------------
    # Generate DataFrame for exporting. 
    # --------------------------------------------------

    tidy_df = dp.export_tidy_projections(
        df=dpro,
        projection_summary=projection_summary,
        path=None 
    )

    # --------------------------------------------------
    # Generate ClimAID scientific dashboard
    # --------------------------------------------------

    dm.generate_report(
        projection_summary=projection_summary,
        style="policy_brief",
        open_browser=True,
        save_copy = True,
        tidy_df=tidy_df,
    )

    return {
        "status": "completed",
        "preset": cfg.preset
    }