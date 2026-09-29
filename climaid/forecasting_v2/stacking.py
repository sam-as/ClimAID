"""Leakage-safe OOF residual stacking and empirical calibration."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from .validation import RollingOriginSplitter
from .renewal import DEFAULT_PROBS

@dataclass
class OOFStackDiagnostics: n_oof:int; outer_folds:int; inner_folds:int; residual_mean:float; residual_sd:float
class TemporalResidualStack:
    def __init__(self,base_estimator,residual_estimator,outer_splits=4,inner_splits=3,test_size=3,min_train_size=18): self.base_estimator=base_estimator; self.residual_estimator=residual_estimator; self.outer_splits=outer_splits; self.inner_splits=inner_splits; self.test_size=test_size; self.min_train_size=min_train_size
    def fit(self,X,y):
        X=X.reset_index(drop=True); y=np.asarray(y,float); outer=RollingOriginSplitter(self.outer_splits,self.test_size,min_train_size=self.min_train_size); oof=np.full(len(y),np.nan)
        for tr,va in outer.split(len(y)):
            base=clone(self.base_estimator).fit(X.iloc[tr],y[tr]); base_val=base.predict(X.iloc[va]); inner=self._base_oof(X.iloc[tr].reset_index(drop=True),y[tr]); mask=np.isfinite(inner)
            if mask.sum()<8:
                # Early folds may be too short for a second-stage residual learner.
                # Use the base forecast for that fold rather than introducing leakage.
                oof[va]=np.maximum(0,base_val)
            else:
                res=clone(self.residual_estimator).fit(X.iloc[tr].reset_index(drop=True).loc[mask],y[tr][mask]-inner[mask])
                oof[va]=np.maximum(0,base_val+res.predict(X.iloc[va]))
        self.oof_stacked_predictions_=oof; mask=np.isfinite(oof)
        if mask.sum()<12: raise ValueError('Too few temporal OOF predictions for residual stacking')
        inner=self._base_oof(X,y); m=np.isfinite(inner); self.base_model_=clone(self.base_estimator).fit(X,y); self.residual_model_=clone(self.residual_estimator).fit(X.loc[m],y[m]-inner[m]); self.oof_predictions_=inner; r=y[mask]-oof[mask]; self.oof_residuals_=r; self.diagnostics_=OOFStackDiagnostics(int(mask.sum()),self.outer_splits,self.inner_splits,float(r.mean()),float(r.std(ddof=1))); return self
    def _base_oof(self,X,y):
        n=len(y)
        min_train=min(self.min_train_size,max(6,n-self.test_size-1))
        max_splits=max(1,(n-min_train)//self.test_size)
        n_splits=min(self.inner_splits,max_splits)
        oof=np.full(n,np.nan)
        splitter=RollingOriginSplitter(n_splits=n_splits,test_size=self.test_size,min_train_size=min_train)
        for tr,va in splitter.split(n):
            model=clone(self.base_estimator)
            model.fit(X.iloc[tr],y[tr])
            oof[va]=model.predict(X.iloc[va])
        return oof
    def predict(self,X):
        return np.maximum(0,np.asarray(self.base_model_.predict(X),float)+np.asarray(self.residual_model_.predict(X),float))

class QuantileResidualCalibrator:
    def __init__(self,nonnegative=True): self.nonnegative=nonnegative; self.residuals_=None
    def fit(self,y,oof_prediction):
        y,p=np.asarray(y,float),np.asarray(oof_prediction,float); m=np.isfinite(y)&np.isfinite(p)
        if m.sum()<12: raise ValueError('At least 12 OOF predictions are needed for calibration')
        self.residuals_=np.sort(y[m]-p[m]); return self
    def predict_quantiles(self,point_prediction,probs=DEFAULT_PROBS):
        p=np.asarray(probs,float); out=np.asarray(point_prediction,float).reshape(-1,1)+np.quantile(self.residuals_,p).reshape(1,-1); out=np.maximum(0,out) if self.nonnegative else out; return np.sort(out,1)

class OOFIsotonicCalibrator:
    def fit(self,y,oof_prediction):
        y,p=np.asarray(y,float),np.asarray(oof_prediction,float); m=np.isfinite(y)&np.isfinite(p)
        if m.sum()<20: raise ValueError('At least 20 OOF predictions are recommended for isotonic calibration')
        self.model_=IsotonicRegression(out_of_bounds='clip').fit(p[m],y[m]); return self
    def predict(self,p): return np.maximum(0,self.model_.predict(np.asarray(p,float)))
