"""ClimAID v2: climate-mandatory, leakage-safe, probabilistic forecasting."""
from .forecaster import DEFAULT_V2_MODELS, ClimaidV2Forecaster, ForecastBundle
from .renewal import ClimateRenewalModel, RenewalFitResult, DEFAULT_PROBS, generation_weights, renewal_force
from .features import ClimateFeatureBuilder, ClimateFeatureConfig
from .validation import ForecastOrigin, RollingOriginSplitter, LeakageSafePreprocessor
from .stacking import TemporalResidualStack, QuantileResidualCalibrator, OOFIsotonicCalibrator
from .baselines import SeasonalNaive
from .ensemble import QuantileMedianEnsemble
from .hindcast import HindcastEvaluator, HindcastOrigin
from .scenario import HybridScenarioProjector, ClimateResponseModel, ScenarioOutlook, bias_correct
from .metrics import weighted_interval_score, coverage, rmse, mae, pinball_loss
from .schema import canonicalize_disease, canonicalize_climate, DataProvenance
from .ml import ClimateMLForecaster, available_ml_estimators
__all__=["HybridScenarioProjector","ClimateResponseModel","ScenarioOutlook","bias_correct","DEFAULT_V2_MODELS","ClimaidV2Forecaster","ForecastBundle","ClimateRenewalModel","RenewalFitResult","DEFAULT_PROBS","generation_weights","renewal_force","ClimateFeatureBuilder","ClimateFeatureConfig","ForecastOrigin","RollingOriginSplitter","LeakageSafePreprocessor","TemporalResidualStack","QuantileResidualCalibrator","OOFIsotonicCalibrator","SeasonalNaive","QuantileMedianEnsemble","HindcastEvaluator","HindcastOrigin","weighted_interval_score","coverage","rmse","mae","pinball_loss","canonicalize_disease","canonicalize_climate","DataProvenance","ClimateMLForecaster","available_ml_estimators"]
