import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from climaid.model_registry import list_available_models
from climaid.reporting import DiseaseReporter, ReportArtifacts
from climaid.reporting_v2 import generate_forecast_report
from climaid.climaid_model import DiseaseModel
from climaid.forecasting_v2 import ClimaidV2Forecaster


def _synthetic():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2012-01-01", "2024-12-01", freq="MS")
    mo = dates.month.to_numpy()
    temp = 28 + 2*np.sin(2*np.pi*mo/12) + rng.normal(0, .3, len(dates))
    rain = 5 + 4*np.maximum(0, np.sin(2*np.pi*(mo-2)/12)) + rng.normal(0, .4, len(dates))
    hum = 70 + 8*np.sin(2*np.pi*(mo-1)/12) + rng.normal(0, 1, len(dates))
    enso = rng.normal(0, .5, len(dates))
    cases = np.maximum(0, np.round(18 + 7*np.sin(2*np.pi*mo/12) + 2*enso + rng.normal(0,3,len(dates)))).astype(int)
    disease = pd.DataFrame({"Date": dates, "Case": cases})
    climate = pd.DataFrame({"time": dates, "temperature": temp, "rainfall": rain, "humidity": hum, "enso": enso})
    return disease, climate


def test_original_model_registry_preserved():
    names = set(list_available_models())
    expected = {"rf", "random_forest", "xgb", "xgboost", "lgbm", "lightgbm", "catboost", "poisson", "ridge", "lasso", "elasticnet", "linear", "mlp", "neural_net", "nn", "extra_trees", "extratrees", "gbr", "gradient_boosting", "isotonic"}
    assert expected.issubset(names)


def test_original_deterministic_report_preserved():
    artifacts = ReportArtifacts(
        district="IND_Pune_MAHARASHTRA", disease_name="Dengue", date_range="2019-2023",
        metrics={"test_r2": 0.4, "test_rmse": 10}, selected_lags={}, interaction_lags=[],
        features=[], importance={}, projection_summary={"mode": "historical_only"},
        model_info={}, data_summary={}, runtime={}
    )
    report = DiseaseReporter(llm_client=None).generate(artifacts, style="_deterministic_engine")
    assert "ClimAID Deterministic Scientific Interpreter (C-DSI)" in report


def test_v2_supports_no_population_and_climate_mandatory():
    disease, climate = _synthetic()
    model = ClimaidV2Forecaster(models=["seasonal_naive", "renewal", "random_forest"], population_at_risk=None)
    model.fit(disease, climate, cutoff="2021-12-31")
    future = climate[climate.time > "2021-12-31"].head(12)
    bundle = model.predict(future, 12, n_simulations=300)
    assert set(bundle.forecasts) == {"seasonal_naive", "renewal", "random_forest", "ensemble"}
    assert len(bundle.forecasts["renewal"]) == 12
    assert bundle.metadata["climate_mandatory"] is True
    assert "relative-incidence mode" in " ".join(bundle.metadata["warnings"])


def test_disease_model_v2_method_is_attached():
    d, c = _synthetic()
    dm = object.__new__(DiseaseModel)
    dm.target_col = "Count"
    dm.disease_name = "Dengue"
    dm.district = "IND_Pune_MAHARASHTRA"
    dm.random_state = 42
    dm.df_disease = d.rename(columns={"Case": "Count"})
    dm.df_climate_hist = c.copy()
    dm.df_climate_proj = c.copy()
    out = dm.forecast_v2(
        forecast_origin="2021-12-31", horizon=12, n_simulations=300,
        models=("seasonal_naive", "renewal"), population_at_risk=None,
        forecast_climate=c[c.time > "2021-12-31"].head(12), run_hindcasts=False,
        save_report=False,
    )
    assert "renewal" in out["forecasts"]


def test_v2_report_wrapper_contains_validation_contract():
    d, c = _synthetic()
    model = ClimaidV2Forecaster(models=["seasonal_naive", "renewal"], population_at_risk=None).fit(d, c, cutoff="2021-12-31")
    bundle = model.predict(c[c.time > "2021-12-31"].head(12), 12, n_simulations=200)
    html = generate_forecast_report(disease_name="Dengue", district="IND_Pune_MAHARASHTRA", bundle=bundle)
    assert "Validation contract" in html
    assert "climate_mandatory" in html
