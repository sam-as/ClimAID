"""Recalibrate forecast intervals from rolling hindcasts.

Problem
-------
v2 intervals were too narrow: in a Pune dengue run the ensemble's 95%
intervals contained only 58% of hindcast months. Intervals built from
in-sample residuals or simulation noise miss model error and grow too
slowly with lead time.

Method (split-conformal, per lead-time band)
--------------------------------------------
For each model and each central interval (50/80/90/95%), every hindcast month
gets a normalised miss score

    r = (y - median) / (q_upper - median)    if y >= median
    r = (median - y) / (median - q_lower)    otherwise

so r <= 1 means the observation fell inside the original interval. The
calibration factor k is the conformal quantile of r at the nominal level,
i.e. the smallest multiple of each half-width that would have contained that
share of hindcast outcomes. Quantiles are then rescaled about the median:
q' = median + k (q - median). k > 1 widens, k < 1 narrows.

Factors are estimated separately for lead-time bands because errors grow
with lead time. A band with fewer than `min_points` hindcast months uses the
factor pooled across all bands; lead times beyond any hindcast use the
longest available band, which is reported as a warning because uncertainty
there is extrapolated.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

PAIRS = [("q025", "q975", 0.95), ("q050", "q950", 0.90), ("q100", "q900", 0.80), ("q250", "q750", 0.50)]
BANDS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 10 ** 6)]
QCOLS = ["q025", "q050", "q100", "q250", "q500", "q750", "q900", "q950", "q975"]


def _lead(times, origin):
    t = pd.to_datetime(times); o = pd.Timestamp(origin)
    return (t.dt.year - o.year) * 12 + (t.dt.month - o.month)


def _band(h):
    for lo, hi in BANDS:
        if lo <= h <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    return f"{BANDS[-1][0]}+"


def _scores(g, lo_col, hi_col):
    med = g["q500"].to_numpy(float); y = g["cases"].to_numpy(float)
    floor = np.maximum(0.5, 0.05 * np.abs(med))
    up = np.maximum(g[hi_col].to_numpy(float) - med, floor)
    dn = np.maximum(med - g[lo_col].to_numpy(float), floor)
    return np.where(y >= med, (y - med) / up, (med - y) / dn)


def _conformal_k(r, level, k_min=0.5, k_max=25.0):
    r = np.sort(r[np.isfinite(r)]); n = len(r)
    if n == 0:
        return 1.0
    idx = min(n - 1, max(0, math.ceil((n + 1) * level) - 1))
    return float(np.clip(r[idx], k_min, k_max))


def fit_interval_calibration(hindcast_forecasts: pd.DataFrame, observed: pd.DataFrame,
                             exclude_times=(), min_points=8, shrink=20) -> dict:
    """Return {model: {"bands": {band: {level: k}}, "pooled": {level: k},
    "n": {band: n}, "max_lead": int}} estimated from hindcast forecasts
    (columns time, origin, model, q025..q975) and observed cases."""
    if hindcast_forecasts is None or hindcast_forecasts.empty:
        return {}
    f = hindcast_forecasts.copy()
    f["time"] = pd.to_datetime(f["time"])
    obs = observed[["time", "cases"]].copy(); obs["time"] = pd.to_datetime(obs["time"])
    f = f.merge(obs, on="time", how="inner")
    if len(exclude_times):
        f = f[~f["time"].isin(pd.to_datetime(list(exclude_times)))]
    f["lead"] = [(_t.year - _o.year) * 12 + (_t.month - _o.month) for _t, _o in zip(f["time"], pd.to_datetime(f["origin"]))]
    f["band"] = f["lead"].map(_band)
    out = {}
    for model, g in f.groupby("model"):
        pooled = {lvl: _conformal_k(_scores(g, lo, hi), lvl) for lo, hi, lvl in PAIRS}
        bands, counts = {}, {}
        for band, gb in g.groupby("band"):
            n = int(len(gb)); counts[band] = n
            ks = {}
            for lo, hi, lvl in PAIRS:
                # A level-p quantile needs roughly 2/(1-p) points to be more
                # than "the largest miss" (95% -> 40, 90% -> 20, 80% -> 10).
                # Below that, use the pooled factor; above it, shrink the band
                # factor toward the pooled one with a prior weight of
                # `shrink` points, to damp noise.
                need = max(min_points, int(math.ceil(2.0 / (1.0 - lvl))))
                if n < need:
                    ks[lvl] = pooled[lvl]
                else:
                    kb = _conformal_k(_scores(gb, lo, hi), lvl)
                    ks[lvl] = float((n * kb + shrink * pooled[lvl]) / (n + shrink))
            bands[band] = ks
        out[model] = {"bands": bands, "pooled": pooled, "n": counts, "max_lead": int(g["lead"].max())}
    return out


def apply_interval_calibration(frame: pd.DataFrame, cal: dict, origin) -> pd.DataFrame:
    """Rescale a forecast frame's quantiles about the median using `cal`
    (one model's entry from fit_interval_calibration)."""
    if not cal:
        return frame
    out = frame.copy()
    lead = _lead(out["time"], origin).to_numpy()
    order = [b for b in cal["bands"]]
    longest = max(order, key=lambda b: int(b.split("-")[0].rstrip("+"))) if order else None
    med = out["q500"].to_numpy(float)
    for i, h in enumerate(lead):
        b = _band(int(h))
        ks = cal["bands"].get(b) or (cal["bands"].get(longest) if longest else None) or cal["pooled"]
        for lo, hi, lvl in PAIRS:
            k = ks.get(lvl, 1.0)
            out.iat[i, out.columns.get_loc(lo)] = med[i] + k * (out.iat[i, out.columns.get_loc(lo)] - med[i])
            out.iat[i, out.columns.get_loc(hi)] = med[i] + k * (out.iat[i, out.columns.get_loc(hi)] - med[i])
    q = np.maximum(out[QCOLS].to_numpy(float), 0.0)
    out[QCOLS] = np.maximum.accumulate(q, axis=1)          # keep quantiles ordered
    return out


def calibration_summary(cal_all: dict) -> pd.DataFrame:
    rows = []
    for model, c in cal_all.items():
        for band, ks in c["bands"].items():
            rows.append({"model": model, "lead_months": band, "n": c["n"].get(band, 0),
                         **{f"k_{int(l * 100)}": round(v, 2) for l, v in ks.items()}})
    return pd.DataFrame(rows)
