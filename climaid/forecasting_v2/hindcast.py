"""Rolling hindcast evaluator for reproducible benchmark comparisons."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np,pandas as pd
from .schema import canonicalize_disease,canonicalize_climate
from .baselines import SeasonalNaive
from .forecaster import ClimaidV2Forecaster
from .renewal import DEFAULT_PROBS
from .metrics import rmse,mae,weighted_interval_score,coverage
@dataclass(frozen=True)
class HindcastOrigin: cutoff:pd.Timestamp; horizon:int
class HindcastEvaluator:
    def __init__(self,date_col='time',case_col='cases',population_col='population',horizon=12,n_origins=4,min_train_size=36): self.date_col=date_col; self.case_col=case_col; self.population_col=population_col; self.horizon=horizon; self.n_origins=n_origins; self.min_train_size=min_train_size
    def make_origins(self,disease):
        d=canonicalize_disease(disease,date_col=self.date_col,case_col=self.case_col); dates=pd.DatetimeIndex(d.time.drop_duplicates().sort_values()); start=self.min_train_size-1; end=len(dates)-self.horizon-1
        if end<start: raise ValueError('Not enough observations for requested hindcasts')
        pos=np.linspace(start,end,min(self.n_origins,end-start+1),dtype=int); return [HindcastOrigin(dates[i],self.horizon) for i in pos]
    def evaluate(self,disease,climate,models=('seasonal_naive','renewal','random_forest'),population_at_risk=None,renewal_kwargs=None,n_simulations=1000,score_exclude_times=(),tuning=None):
        """Run rolling-origin hindcasts and score them.

        ROBUSTNESS FIX: this used to wrap each fold in a bare
        ``except Exception: continue`` and silently skip folds whose
        post-origin climate was shorter than the horizon. If every fold
        failed (bad climate columns, a model error, too-short climate), the
        caller got an empty table and the report said "no hindcast scores
        were available", hiding the actual cause. Now every skipped fold is
        recorded in ``self.skipped_origins_`` (origin + reason), partial
        failures raise a ``RuntimeWarning``, and if *no* fold succeeds a
        ``RuntimeError`` listing the reasons is raised, so callers (e.g.
        ``DiseaseModel.forecast_v2``) can surface it.
        """
        import warnings
        # Months whose observed values were replaced (e.g. 2020 under
        # drop_2020) are still usable as model *inputs* but must never be
        # used as *targets* when scoring: that would measure agreement with
        # an imputed average, not forecast accuracy.
        self.score_exclude_times_=set(pd.to_datetime(list(score_exclude_times)))
        d=canonicalize_disease(disease,date_col=self.date_col,case_col=self.case_col,population_col=self.population_col)
        c=canonicalize_climate(climate,date_col=self.date_col,require_all=True).sort_values('time').reset_index(drop=True)
        fc=[]; skipped=[]
        origins=self.make_origins(d)
        # Same model set and forecast origin for each fold.
        use=[m for m in models if m in ('seasonal_naive','renewal') or m in ClimaidV2Forecaster.available_models()]
        dropped=[m for m in models if m not in use]
        if dropped:
            warnings.warn(f"Hindcast: model(s) {dropped} unavailable in this environment and were skipped.",RuntimeWarning,stacklevel=2)
        if not use:
            raise ValueError(f"None of the requested hindcast models are available: {list(models)}")
        for o in origins:
            train=d[d.time<=o.cutoff]; future=c[c.time>o.cutoff].head(o.horizon)
            if len(future)<o.horizon:
                skipped.append({'origin':o.cutoff,'reason':f'only {len(future)} of {o.horizon} post-origin climate months available'}); continue
            try:
                eng=ClimaidV2Forecaster(date_col=self.date_col,case_col=self.case_col,population_col=self.population_col,models=tuple(use),renewal_kwargs=renewal_kwargs,population_at_risk=population_at_risk,tuning=tuning).fit(train,c,o.cutoff)
                bundle=eng.predict(future,o.horizon,n_simulations)
                for name,f in bundle.forecasts.items(): fc.append(f.assign(origin=o.cutoff,model=name))
            except Exception as exc:
                skipped.append({'origin':o.cutoff,'reason':f'{type(exc).__name__}: {exc}'})
        self.skipped_origins_=pd.DataFrame(skipped,columns=['origin','reason'])
        if not fc:
            detail='; '.join(f"{r['origin']:%Y-%m-%d}: {r['reason']}" for r in skipped) or 'no origins produced'
            raise RuntimeError(f"All {len(origins)} hindcast origins failed -- {detail}")
        if skipped:
            warnings.warn(f"Hindcast: {len(skipped)} of {len(origins)} origins skipped (see HindcastEvaluator.skipped_origins_). First reason: {skipped[0]['reason']}",RuntimeWarning,stacklevel=2)
        forecasts=pd.concat(fc,ignore_index=True); return forecasts,self._score(forecasts,d)
    def _score(self,f,obs):
        if f.empty:return pd.DataFrame()
        q=[f'q{int(round(p*1000)):03d}' for p in DEFAULT_PROBS]; rows=[]
        for (model,origin),g in f.groupby(['model','origin']):
            m=obs[['time','cases']].merge(g[['time']+q],on='time',validate='one_to_one')
            excl=getattr(self,'score_exclude_times_',set())
            if excl: m=m[~m.time.isin(excl)]
            # Require a full fold normally; with excluded months, at least half.
            if len(m)<(self.horizon if not excl else max(3,self.horizon//2)): continue
            y=m.cases.to_numpy(float); rows.append({'origin':origin,'model':model,'n':len(m),'RMSE':rmse(y,m.q500),'MAE':mae(y,m.q500),'WIS':weighted_interval_score(y,m[q].to_numpy(float),DEFAULT_PROBS),'coverage_50':coverage(y,m.q250,m.q750),'coverage_80':coverage(y,m.q100,m.q900),'coverage_95':coverage(y,m.q025,m.q975)})
        return pd.DataFrame(rows)
