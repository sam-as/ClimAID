"""Disrupted-period handling (e.g. COVID-19) shared by v1, v2 and the scenario outlook.

Options offered to users
------------------------
- "2020" (default): January-December 2020.
- a custom period: "YYYY-MM:YYYY-MM" or a (start, end) pair of dates, e.g. to cover a local lockdown or
  a known reporting outage. Whole months, inclusive.
- "none": use the recorded counts.

There is deliberately no "2020-2021" preset (the default use is South Asia, where 2020 is the
disrupted year); a custom period can still be entered where local evidence supports it.

What happens to excluded months
-------------------------------
- v1 leaves them out of training and testing.
- v2 replaces them in the training history with that calendar month's median from the other
  training months (models need an unbroken monthly series), never scores hindcasts on them, and
  the scenario outlook excludes them from the long-term fit. Months after the forecast origin are
  never altered, so they can still be used to score a forecast.
"""
from __future__ import annotations

import pandas as pd

PRESETS = {"2020": ("2020-01", "2020-12")}


def resolve_exclusion(exclude_period=None, drop_2020=True):
    """Return (start, end) month-start Timestamps, or None for no exclusion.

    `exclude_period` takes precedence; when it is None the legacy `drop_2020` flag decides.
    """
    if exclude_period is None:
        return resolve_exclusion("2020") if drop_2020 else None
    if exclude_period is False or (isinstance(exclude_period, str) and exclude_period.strip().lower() in ("none", "", "no", "off")):
        return None
    if isinstance(exclude_period, str):
        key = exclude_period.strip()
        if key in PRESETS:
            a, b = PRESETS[key]
        else:
            for sep in (":", " to ", "/", ".."):
                if sep in key:
                    a, b = [x.strip() for x in key.split(sep, 1)]
                    break
            else:
                raise ValueError(f"exclude_period '{exclude_period}' not understood; use '2020', 'none' or 'YYYY-MM:YYYY-MM'")
    else:
        try:
            a, b = exclude_period
        except Exception as exc:
            raise ValueError("exclude_period must be '2020', 'none', 'YYYY-MM:YYYY-MM' or a (start, end) pair") from exc
    start = pd.Timestamp(a).to_period("M").to_timestamp()
    end = pd.Timestamp(b).to_period("M").to_timestamp()
    if end < start:
        raise ValueError(f"exclude_period ends ({end:%Y-%m}) before it starts ({start:%Y-%m})")
    return start, end


def in_period(times, period):
    t = pd.to_datetime(pd.Series(times)).dt.to_period("M").dt.to_timestamp()
    if period is None:
        return pd.Series(False, index=t.index).to_numpy()
    return ((t >= period[0]) & (t <= period[1])).to_numpy()


def describe(period):
    if period is None:
        return "none"
    a, b = period
    if a.month == 1 and b.month == 12 and a.year == b.year:
        return str(a.year)
    return f"{a:%B %Y} to {b:%B %Y}"


def replace_with_monthly_median(disease, origin, period, case_col="cases"):
    """Replace excluded months at or before `origin` by the calendar-month median of the other months
    at or before `origin`. Returns (new_frame, replaced_times)."""
    if period is None:
        return disease, []
    d = disease.copy()
    t = pd.to_datetime(d["time"])
    hist = (t <= pd.Timestamp(origin)).to_numpy()
    ex = hist & in_period(t, period)
    if not ex.any():
        return d, []
    ref = d[hist & ~ex]
    med = ref.groupby(pd.to_datetime(ref["time"]).dt.month)[case_col].median()
    fill = pd.to_datetime(d.loc[ex, "time"]).dt.month.map(med)
    d.loc[ex, case_col] = fill.fillna(d.loc[ex, case_col]).round().values
    return d, pd.to_datetime(d.loc[ex, "time"]).tolist()
