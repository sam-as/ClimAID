"""Input schema normalization and provenance for ClimAID v2."""
from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd

CLIMATE_ALIASES = {
    "temperature": ["temperature","temp","mean_temperature","mean temp","tas","t2m"],
    "rainfall": ["rainfall","rain","precipitation","mean_rain","mean_Rain","pr"],
    "humidity": ["humidity","specific_humidity","mean_sh","mean_SH","rh","relative humidity","huss"],
    "enso": ["enso","nino_anomaly","nino","nino34","oni"],
}
TIME_ALIASES=["time","date","datetime","timestamp"]
CASE_ALIASES=["cases","case","count","dengue_cases","incidence","y"]
POP_ALIASES=["population","pop","population_at_risk","pop_est"]

def _norm(x):
    return re.sub(r"[^a-z0-9]+","_",str(x).strip().lower()).strip("_")

def find_column(df, aliases, explicit=None):
    if explicit:
        if explicit in df.columns: return explicit
        ne=_norm(explicit)
        for c in df.columns:
            if _norm(c)==ne: return c
    m={_norm(c):c for c in df.columns}
    for a in aliases:
        if _norm(a) in m: return m[_norm(a)]
    return None

def canonicalize_disease(df, date_col=None, case_col=None, population_col=None):
    if not isinstance(df,pd.DataFrame): raise TypeError("disease must be a pandas DataFrame")
    out=df.copy()
    t=find_column(out,TIME_ALIASES,date_col); y=find_column(out,CASE_ALIASES,case_col)
    if t is None or y is None: raise ValueError(f"Disease data must contain date/time and case count columns. Available: {list(out.columns)}")
    ren={t:"time",y:"cases"}; out=out.rename(columns=ren)
    out["time"]=pd.to_datetime(out["time"],errors="coerce"); out["cases"]=pd.to_numeric(out["cases"],errors="coerce")
    out=out.dropna(subset=["time","cases"]).sort_values("time")
    if (out["cases"]<0).any(): raise ValueError("cases must be non-negative")
    p=find_column(out,POP_ALIASES,population_col)
    if p: out=out.rename(columns={p:"population"}); out["population"]=pd.to_numeric(out["population"],errors="coerce")
    if out["time"].duplicated().any():
        agg={"cases":"sum"};
        if "population" in out.columns: agg["population"]="last"
        out=out.groupby("time",as_index=False).agg(agg)
    return out.reset_index(drop=True)

def canonicalize_climate(df,date_col=None,climate_map=None,require_all=True):
    if not isinstance(df,pd.DataFrame): raise TypeError("climate must be a pandas DataFrame")
    out=df.copy(); t=find_column(out,TIME_ALIASES,date_col)
    if t is None: raise ValueError("Climate data must contain a date/time column")
    out=out.rename(columns={t:"time"}); out["time"]=pd.to_datetime(out["time"],errors="coerce")
    explicit=climate_map or {}; ren={}; missing=[]
    for can,aliases in CLIMATE_ALIASES.items():
        c=find_column(out,aliases,explicit.get(can))
        if c: ren[c]=can
        elif can!="enso" and require_all: missing.append(can)
    out=out.rename(columns=ren)
    if missing: raise ValueError("ClimAID v2 requires temperature, rainfall, and humidity. Missing: "+", ".join(missing))
    for c in ["temperature","rainfall","humidity","enso"]:
        if c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce")
    out=out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if out["time"].duplicated().any():
        keep=[c for c in ["temperature","rainfall","humidity","enso"] if c in out.columns]
        out=out.groupby("time",as_index=False)[keep].mean()
    return out

@dataclass(frozen=True)
class DataProvenance:
    disease_source: str|None
    climate_source: str|None
    disease_start: str|None
    disease_end: str|None
    climate_start: str|None
    climate_end: str|None
    climate_columns: tuple[str,...]
