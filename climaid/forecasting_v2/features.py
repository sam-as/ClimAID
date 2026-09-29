"""Leakage-safe climate feature engineering for ClimAID v2."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from .schema import canonicalize_climate

@dataclass
class ClimateFeatureConfig:
    date_col: str = 'time'
    climate_columns: tuple[str,...] = ('temperature','rainfall','humidity')
    lags: tuple[int,...] = (0,1,2,3)
    add_anomalies: bool = True
    add_seasonality: bool = True
    anomaly_window_years: int = 10
    add_enso: bool = True
    add_enso_interactions: bool = True
    # Distributed lag effects: smooth lag curves (see distributed_lag.py). max lags per variable.
    add_distributed_lags: bool = True
    distributed_max_lag: tuple = (('temperature', 6), ('rainfall', 6), ('humidity', 6), ('enso', 12))
    distributed_degree: int = 2

class ClimateFeatureBuilder:
    def __init__(self, config=None):
        self.config=config or ClimateFeatureConfig(); self.training_cutoff_=None; self.month_climatology_=None; self.columns_used_=[]
    def fit(self, climate, cutoff):
        c=canonicalize_climate(climate,date_col=self.config.date_col,require_all=True); cutoff=pd.Timestamp(cutoff); h=c[c.time<=cutoff].copy()
        if h.empty: raise ValueError('No climate observations at or before cutoff')
        self.columns_used_=[x for x in self.config.climate_columns if x in h.columns]
        if self.config.add_anomalies: self.month_climatology_=h.assign(_month=h.time.dt.month).groupby('_month')[self.columns_used_].mean()
        self.train_means_=h[self.columns_used_].mean().to_dict()
        dl_cols=self.columns_used_+(['enso'] if 'enso' in h.columns else [])
        self.dl_scale_={col:(float(h[col].mean()), float(h[col].std()) or 1.0) for col in dl_cols}
        self.training_cutoff_=cutoff; return self
    def transform(self, climate):
        if self.training_cutoff_ is None: raise RuntimeError('ClimateFeatureBuilder must be fitted first')
        c=canonicalize_climate(climate,date_col=self.config.date_col,require_all=True).sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True); out=pd.DataFrame({'time':c.time})
        for col in self.columns_used_:
            for lag in self.config.lags: out[f'{col}_lag{lag}']=c[col].shift(lag).to_numpy(float)
            if self.config.add_anomalies:
                clim=self.month_climatology_[col]; out[f'{col}_anomaly']=[float(v-clim.get(m,np.nan)) if pd.notna(v) else np.nan for v,m in zip(c[col],c.time.dt.month)]
        if self.config.add_enso and 'enso' in c.columns:
            for lag in self.config.lags: out[f'enso_lag{lag}']=c.enso.shift(lag).to_numpy(float)
        if self.config.add_enso and self.config.add_enso_interactions and 'enso' in c.columns:
            # v1-style ENSO interactions: each climate variable at each lag, centred on its
            # training-period mean, times ENSO's mean over lags 1-3 (El Nino conditions in the
            # preceding season can amplify or damp the effect of heat and rain). Using one ENSO
            # summary keeps this to len(vars) x len(lags) features instead of every lag pair.
            enso3 = (c.enso.shift(1) + c.enso.shift(2) + c.enso.shift(3)) / 3.0
            for col in self.columns_used_:
                mu = float(self.train_means_.get(col, 0.0))
                for lag in self.config.lags:
                    out[f'{col}_lag{lag}_x_enso3'] = ((c[col].shift(lag) - mu) * enso3).to_numpy(float)
        if self.config.add_distributed_lags:
            from .distributed_lag import cross_basis
            for col, L in dict(self.config.distributed_max_lag).items():
                if col not in c.columns or col not in self.dl_scale_:
                    continue
                mu, sd = self.dl_scale_[col]
                cb = cross_basis(((c[col] - mu) / sd).to_numpy(float), int(L), self.config.distributed_degree)
                for k in range(cb.shape[1]):
                    out[f'{col}_dl{k}'] = cb[:, k]
        if self.config.add_seasonality:
            mo=c.time.dt.month.to_numpy(); out['month_sin']=np.sin(2*np.pi*mo/12); out['month_cos']=np.cos(2*np.pi*mo/12)
        return out
    def fit_transform(self,climate,cutoff): return self.fit(climate,cutoff).transform(climate)
