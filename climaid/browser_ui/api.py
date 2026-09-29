from pathlib import Path
import uuid
import tempfile
import shutil
from io import BytesIO

from fastapi import APIRouter, UploadFile, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd

from climaid.climaid_model import DiseaseModel
from climaid.climaid_projections import DiseaseProjection
from climaid.districts import get_available_districts
from climaid.model_registry import list_available_models
from .state import get_wizard_state, clear_wizard_state

router = APIRouter()
REPORT_DIR = Path("climaid_outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _state_for_session(session_id: str | None):
    sid, state = get_wizard_state(session_id)
    return sid, state


def _session_header(value: str | None) -> str:
    return _state_for_session(value)[0]


# ======================================================
# Original catalog/upload APIs — retained
# ======================================================
@router.get("/district_catalog")
def district_catalog():
    districts = get_available_districts()
    data = []
    for d in districts:
        parts = d.split("_")
        data.append({
            "country": parts[0] if len(parts) >= 1 else "Unknown",
            "state": parts[2] if len(parts) >= 3 else "Unknown",
            "district": parts[1] if len(parts) >= 2 else "Unknown",
        })
    df = pd.DataFrame(data)
    result = {}
    if df.empty:
        return result
    for (country, state), group in df.groupby(["country", "state"]):
        result.setdefault(country, {})[state] = sorted(group["district"].tolist())
    return result


@router.post("/upload_dataset")
async def upload_dataset(
    file: UploadFile,
    x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session"),
):
    sid, state = _state_for_session(x_climaid_session)
    contents = await file.read()
    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(contents))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, Excel or Parquet.")
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded disease dataset is empty.")
        state["dataset"] = df
        state["filename"] = file.filename
        state["dataset_id"] = uuid.uuid4().hex
        return {"status": "uploaded", "session_id": sid, "filename": file.filename, "rows": len(df), "columns": list(df.columns)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read disease dataset: {e}")


def _save_uploaded_file(file: UploadFile, prefix: str) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls", ".parquet"}:
        raise ValueError("Unsupported file format. Use CSV, Excel or Parquet.")
    temp_file = Path(tempfile.gettempdir()) / f"{prefix}_{uuid.uuid4().hex}{ext}"
    with temp_file.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return str(temp_file)


@router.post("/upload_weather")
async def upload_weather(
    file: UploadFile,
    x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session"),
):
    sid, state = _state_for_session(x_climaid_session)
    try:
        path = _save_uploaded_file(file, "weather")
        state["weather_file"] = path
        state["weather_filename"] = file.filename
        return {"status": "weather uploaded", "session_id": sid, "filename": file.filename, "path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload_projection")
async def upload_projection(
    file: UploadFile,
    x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session"),
):
    sid, state = _state_for_session(x_climaid_session)
    try:
        path = _save_uploaded_file(file, "projection")
        state["projection_file"] = path
        state["projection_filename"] = file.filename
        return {"status": "projection uploaded", "session_id": sid, "filename": file.filename, "path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ======================================================
# Configuration — legacy fields preserved + additive v2
# ======================================================
class WizardConfig(BaseModel):
    mode: str
    country: str
    district: str
    state: str
    disease_name: str
    preset: str = "balanced"
    test_year: int | None = None

    # Original v1 names are retained for compatibility.
    base_models: list[str] | None = None
    residual_models: list[str] | None = None
    correction_models: list[str] | None = None
    n_trials: int | None = None

    # Additive run controls.
    run_legacy: bool = True
    legacy_report_mode: str = "deterministic"
    run_cmip6: bool = True

    # v2 probabilistic controls.
    run_v2: bool = False
    forecast_origin_year: int | None = None  # first forecast year; cutoff = prior Dec
    forecast_horizon: int = Field(default=12, ge=1, le=240)
    simulations: int = Field(default=2000, ge=100, le=100000)
    v2_models: list[str] | None = None
    population_at_risk: float | None = Field(default=None, gt=0)
    population_column: str = "population"
    forecast_climate_source: str = "auto"  # auto | observed | projection
    run_hindcasts: bool = True
    hindcast_origins: int = Field(default=4, ge=1, le=20)
    hindcast_horizon: int | None = Field(default=None, ge=1, le=120)
    save_html: bool = True
    # Replace 2020 (COVID-19 reporting disruption) in the training history.
    drop_2020: bool = True            # legacy; used only when exclude_period is not given
    # COVID-19 / disrupted period: "2020" (default), "none" or "YYYY-MM:YYYY-MM".
    exclude_period: str | None = "2020"
    # Compulsory v2 model tuning: effort preset.
    v2_tuning: str = "balanced"
    # Hybrid CMIP6 climate-scenario outlook (v2 page).
    run_scenarios: bool = True
    scenario_end_year: int = Field(default=2050, ge=2025, le=2100)
    scenario_ssps: list[str] | None = None
    scenario_response: str = "seasonal"          # seasonal | anomaly
    scenario_lag_selection: str = "ensemble"     # ensemble | auto | v1 | all
    scenario_temperature_curve: str | None = None
    scenario_compare_trees: bool = True


@router.get("/available_models")
def available_models():
    return {"models": list_available_models()}


@router.get("/available_v2_models")
def available_v2_models():
    from climaid.forecasting_v2 import ClimaidV2Forecaster, DEFAULT_V2_MODELS
    return {"models": ClimaidV2Forecaster.available_models(), "defaults": list(DEFAULT_V2_MODELS)}


@router.get("/reports/{filename}")
def get_report(filename: str):
    safe = Path(filename).name
    path = REPORT_DIR / safe
    if not path.exists():
        return {"error": "Report not found"}
    return FileResponse(path)


@router.post("/reset")
def reset(x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session")):
    sid = clear_wizard_state(x_climaid_session)
    return {"status": "reset", "session_id": sid}


def _district_key(cfg: WizardConfig) -> str:
    return f"{cfg.country.upper()}_{cfg.district.title()}_{cfg.state.upper()}"


def _write_disease_temp(df: pd.DataFrame, filename: str) -> str:
    ext = ".xlsx" if (filename or "").lower().endswith((".xlsx", ".xls")) else ".csv"
    temp_file = Path(tempfile.gettempdir()) / f"climaid_{uuid.uuid4().hex}{ext}"
    if ext == ".csv":
        df.to_csv(temp_file, index=False)
    else:
        df.to_excel(temp_file, index=False)
    return str(temp_file)


def _legacy_model_config(cfg: WizardConfig):
    from climaid.model_parameters import v1_mode_config
    if cfg.preset == "custom":
        return (
            tuple(cfg.base_models or ["xgb"]),
            tuple(cfg.residual_models or ["rf"]),
            tuple(cfg.correction_models or ["isotonic"]),
            int(cfg.n_trials or 200),
        )
    c = v1_mode_config(cfg.preset if cfg.preset in ("fast", "balanced", "deep") else "balanced")
    return c["base_models"], c["residual_models"], c["correction_models"], c["n_trials"]


def _run_legacy(dm: DiseaseModel, cfg: WizardConfig):
    """Original ClimAID v1 pipeline retained as a callable branch."""
    dm.exclude_period = cfg.exclude_period          # same COVID-19 period as v2
    base, residual, correction, trials = _legacy_model_config(cfg)
    dm.optimize_lags(
        base_models=base,
        residual_models=residual,
        correction_models=correction,
        n_trials=trials,
        n_jobs=-1,
    )
    final = dm.train_final_model()

    projection_summary = None
    tidy_df = None
    report_text = None
    report_path = None

    if cfg.run_cmip6:
        dp = DiseaseProjection(dm)
        dpro = dp.project_multi_model_ssp(
            model_list=[
                "ACCESS-ESM1-5", "CESM2", "CNRM-CM6-1", "GFDL-ESM4",
                "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MPI-ESM1-2-HR",
                "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL"
            ],
            ssp_list=["ssp126", "ssp245", "ssp370", "ssp585"],
        )
        projection_summary = dp.build_projection_summary(dpro)
        tidy_df = dp.export_tidy_projections(df=dpro, projection_summary=projection_summary, path=None)
        try:
            dp.flag_outbreak_risk(dpro, method="both", percentile=0.9)
            projection_summary["risk_semantics"] = (
                "Legacy dual-baseline flags represent threshold exceedance/ensemble agreement; "
                "they are not calibrated outbreak probabilities."
            )
        except Exception:
            pass

    # Preserve the original reporting system. Deterministic C-DSI is now explicit/default.
    # -------------------------------------------------------------------
    # BUG FIX: "detailed"/"summary"/"policy" here used to be forwarded as
    # `style` to DiseaseReporter.generate() while `llm_client` was hardcoded
    # to None a few lines below. DiseaseReporter.generate() only varies its
    # output by `style` when an LLM client is provided -- with none, it
    # always returns the exact same deterministic C-DSI report regardless
    # of `style`. So selecting "Detailed", "Summary" or "Policy" in the
    # browser UI silently produced byte-identical output to "C-DSI
    # deterministic", while the result was labelled (both in `style` used
    # for the report title, and in the "type" field returned below) as if
    # that requested narrative style had actually been generated -- a
    # misleading result, not just an unused option. No LLM client is wired
    # up anywhere in the browser UI (WizardConfig has no such field), so
    # there is currently no way for these styles to do anything different
    # here. Fixed by being explicit: if a narrative style was requested
    # without an LLM available, fall back to the deterministic engine and
    # say so, rather than silently mislabelling the output.
    llm_unavailable_note = None
    if cfg.legacy_report_mode == "deterministic":
        style = "_deterministic_engine"
    else:
        style = "_deterministic_engine"
        llm_unavailable_note = (
            f"Legacy report mode '{cfg.legacy_report_mode}' requires an LLM client, "
            "which is not configured in this interface. Showing the deterministic "
            "C-DSI report instead."
        )
    report_text = dm.generate_report(
        projection_summary=projection_summary,
        llm_client=None,
        style=style,
        open_browser=False,
        save_copy=False,
        tidy_df=tidy_df,
    )

    # Render the legacy report to a persistent HTML file for browser access.
    try:
        from climaid.reporting import open_report_in_browser
        artifacts = dm.build_report_artifacts(projection_summary, tidy_df)
        report_path = open_report_in_browser(
            report_text=report_text,
            artifacts=artifacts,
            title=f"ClimAID {cfg.disease_name} — C-DSI deterministic",
            save_copy=True,
            output_dir=str(REPORT_DIR),
        )
    except Exception:
        report_path = None

    return {
        "final": final,
        "projection_summary": projection_summary,
        "tidy_df": tidy_df,
        "report_text": report_text,
        "report_path": report_path,
        "note": llm_unavailable_note,
    }


def _resolve_v2_climate(dm: DiseaseModel, origin: pd.Timestamp, horizon: int, source: str):
    """Resolve future climate without silently leaking unavailable future observations."""
    source = (source or "auto").lower()
    hist = dm.df_climate_hist.copy()
    if "time" not in hist.columns and {"Year", "Month"}.issubset(hist.columns):
        hist["time"] = pd.to_datetime(dict(year=hist["Year"], month=hist["Month"], day=1))
    hist["time"] = pd.to_datetime(hist["time"])
    future_hist = hist[hist["time"] > origin].sort_values("time")

    if source == "observed":
        if len(future_hist) < horizon:
            raise ValueError("Observed climate does not cover the full requested forecast horizon")
        return future_hist.head(horizon), "observed_historical_climate"

    if source == "projection":
        proj = dm.df_climate_proj.copy()
        if "time" not in proj.columns and {"Year", "Month"}.issubset(proj.columns):
            proj["time"] = pd.to_datetime(dict(year=proj["Year"], month=proj["Month"], day=1))
        proj["time"] = pd.to_datetime(proj["time"])
        future_proj = proj[proj["time"] > origin].sort_values("time")
        if len(future_proj) < horizon:
            raise ValueError("Projection climate does not cover the full requested forecast horizon")
        return future_proj.head(horizon), "projection_climate"

    # auto: observed for historical hindcast if available, otherwise projection.
    if len(future_hist) >= horizon:
        return future_hist.head(horizon), "observed_historical_climate"

    proj = dm.df_climate_proj.copy()
    if "time" not in proj.columns and {"Year", "Month"}.issubset(proj.columns):
        proj["time"] = pd.to_datetime(dict(year=proj["Year"], month=proj["Month"], day=1))
    proj["time"] = pd.to_datetime(proj["time"])
    future_proj = proj[proj["time"] > origin].sort_values("time")
    if len(future_proj) < horizon:
        raise ValueError("No climate data cover the requested forecast horizon; provide projection data or shorten the horizon")
    return future_proj.head(horizon), "projection_climate"


@router.post("/run")
def run_pipeline(
    cfg: WizardConfig,
    x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session"),
):
    sid, wizard_state = _state_for_session(x_climaid_session)
    df = wizard_state.get("dataset")
    filename = wizard_state.get("filename", "upload.csv")
    if df is None:
        return {"error": "No dataset uploaded"}

    mode = cfg.mode.lower()
    weather_file = wizard_state.get("weather_file") if mode == "global" else None
    projection_file = wizard_state.get("projection_file") if mode == "global" else None
    if mode == "global" and (cfg.run_v2 or cfg.run_legacy) and not weather_file:
        return {"error": "Global mode requires weather dataset"}

    temp_file = _write_disease_temp(df, filename)
    district_key = _district_key(cfg)

    def make_model():
        return DiseaseModel(
            district=district_key,
            disease_file=str(temp_file),
            disease_name=cfg.disease_name,
            random_state=42,
            weather_file=weather_file,
            projection_file=projection_file,
        )

    result = {"status": "completed", "session_id": sid, "district": district_key, "reports": []}

    if cfg.run_legacy:
        try:
            legacy = _run_legacy(make_model(), cfg)
            result["legacy"] = {
                "status": "completed",
                "report_path": legacy.get("report_path"),
            }
            if legacy.get("note"):
                # Set when a narrative report style ("detailed"/"summary"/
                # "policy") was requested without an LLM client available --
                # see the note in _run_legacy(). Surfaced here rather than
                # silently mislabelling the report as that requested style.
                result["legacy"]["note"] = legacy["note"]
            if legacy.get("report_path"):
                # Always "legacy_c_dsi": every legacy_report_mode currently
                # produces the deterministic C-DSI report in this interface
                # (see the note in _run_legacy() -- no LLM client is wired
                # up here), so labelling it any other way would misrepresent
                # what was actually generated.
                result["reports"].append({
                    "type": "legacy_c_dsi",
                    "url": f"/reports/{Path(legacy['report_path']).name}",
                })
        except Exception as exc:
            result["legacy_error"] = str(exc)

    if cfg.run_v2:
        try:
            dm = make_model()
            # UI says "first forecast year": 2021 => cutoff is 2020-12-31.
            first_year = cfg.forecast_origin_year
            if first_year is None:
                # Retain a sensible backward-compatible fallback.
                origin = pd.to_datetime(dm.df_disease["time"]).max()
            else:
                origin = pd.Timestamp(year=int(first_year) - 1, month=12, day=31)

            selected_climate, climate_source = _resolve_v2_climate(
                dm, origin, cfg.forecast_horizon, cfg.forecast_climate_source
            )
            from climaid.forecasting_v2 import DEFAULT_V2_MODELS
            selected_models = tuple(cfg.v2_models or DEFAULT_V2_MODELS)
            legacy_text_for_v2 = None
            # When both branches are requested, embed the preserved deterministic
            # C-DSI narrative into the v2 report as a legacy appendix.
            if cfg.run_legacy:
                try:
                    legacy_text_for_v2 = legacy.get("report_text") if isinstance(legacy, dict) else None
                except Exception:
                    legacy_text_for_v2 = None

            v2 = dm.forecast_v2(
                forecast_origin=origin,
                horizon=cfg.forecast_horizon,
                n_simulations=cfg.simulations,
                models=selected_models,
                population_at_risk=cfg.population_at_risk,
                population_col=cfg.population_column,
                forecast_climate=selected_climate,
                forecast_climate_source=cfg.forecast_climate_source,
                run_hindcasts=cfg.run_hindcasts,
                hindcast_origins=cfg.hindcast_origins,
                hindcast_horizon=cfg.hindcast_horizon,
                legacy_report_text=legacy_text_for_v2,
                save_report=cfg.save_html,
                output_dir=str(REPORT_DIR),
                drop_2020=cfg.drop_2020, exclude_period=cfg.exclude_period,
                tuning=cfg.v2_tuning,
            )
            # Rename the report deterministically to allow repeated browser runs.
            if v2.get("report_path"):
                src = Path(v2["report_path"])
                run_name = f"climaid_v2_{cfg.disease_name}_{uuid.uuid4().hex[:8]}.html"
                dst = REPORT_DIR / run_name
                src.replace(dst)
                v2["report_path"] = str(dst)
                result["reports"].append({"type": "v2_probabilistic", "url": f"/reports/{dst.name}"})
            v2["metadata"]["forecast_climate_source"] = climate_source
            if cfg.run_scenarios:
                try:
                    sc = dm.project_v2(
                        forecast_origin=origin, end_year=cfg.scenario_end_year,
                        ssps=cfg.scenario_ssps or None, response=cfg.scenario_response,
                        lag_selection=cfg.scenario_lag_selection, drop_2020=cfg.drop_2020,
                        exclude_period=cfg.exclude_period,
                        temperature_curve=cfg.scenario_temperature_curve or None,
                        tuning=cfg.v2_tuning,
                        comparison_models=("random_forest", "gradient_boosting") if cfg.scenario_compare_trees else (),
                        population_at_risk=cfg.population_at_risk, output_dir=str(REPORT_DIR),
                    )
                    if sc.get("report_path"):
                        src = Path(sc["report_path"])
                        dst = REPORT_DIR / f"climaid_v2_scenarios_{cfg.disease_name}_{uuid.uuid4().hex[:8]}.html"
                        src.replace(dst)
                        result["reports"].append({"type": "v2_scenarios", "url": f"/reports/{dst.name}"})
                except Exception as exc:
                    result["scenario_error"] = str(exc)
            result["v2"] = {
                "metadata": v2["metadata"],
                "metrics": v2["metrics"].to_dict(orient="records") if isinstance(v2["metrics"], pd.DataFrame) else v2["metrics"],
                "hindcast_metrics": v2["hindcast_metrics"].to_dict(orient="records") if isinstance(v2["hindcast_metrics"], pd.DataFrame) else v2["hindcast_metrics"],
            }
        except Exception as exc:
            result["v2_error"] = str(exc)

    if not cfg.run_legacy and not cfg.run_v2:
        result["error"] = "Select at least one pipeline: v1 or v2"
        result["status"] = "not_run"

    return result
