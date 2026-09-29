import numpy as np
import pandas as pd
from climaid.forecasting_v2 import ClimaidV2Forecaster

def test_all_forecasts_share_common_calendar():
    dates=pd.date_range("2015-01-01","2022-12-01",freq="MS")
    m=dates.month.to_numpy()
    rng=np.random.default_rng(0)
    disease=pd.DataFrame({"Date":dates,"Case":np.maximum(0,np.round(20+5*np.sin(2*np.pi*m/12)+rng.normal(0,2,len(dates))))})
    climate=pd.DataFrame({"time":dates,"temperature":27+2*np.sin(2*np.pi*m/12),"rainfall":5+2*np.sin(2*np.pi*(m-2)/12),"humidity":70+5*np.sin(2*np.pi*m/12),"enso":rng.normal(0,.3,len(dates))})
    f=ClimaidV2Forecaster(models=("seasonal_naive","renewal","random_forest"),population_at_risk=None).fit(disease,climate,cutoff="2020-12-31")
    future=climate[climate.time>"2020-12-31"].head(12)
    b=f.predict(future,12,n_simulations=120)
    expected=pd.DatetimeIndex(future.time)
    for frame in b.forecasts.values():
        assert pd.DatetimeIndex(frame.time).equals(expected)
