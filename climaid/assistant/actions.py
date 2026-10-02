"""What the assistant can do: check a data file, run ClimAID, and summarise results.

Every number shown to the user comes from ClimAID's own code: the run results and the
rule-based plain-language report layer (``climaid.reporting_plain``), the same text the HTML
reports use. Nothing is generated.
"""
from __future__ import annotations

import contextlib
import html
import io
import re
from pathlib import Path

import pandas as pd

DISEASE_SUFFIXES = (".csv", ".xlsx", ".xls")
DATE_NAMES = ("date", "time", "datetime")
COUNT_NAMES = ("cases", "case", "count")


def html_to_text(fragment: str) -> str:
    """Plain text from the report's HTML fragments (lists become '- ' lines)."""
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.S | re.I)
    t = re.sub(r"<li[^>]*>", "\n- ", t, flags=re.I)
    t = re.sub(r"</(p|li|ul|div|h\d|tr)>", "\n", t, flags=re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


# --------------------------------------------------------------------------- data check

def inspect_disease_file(path: str) -> dict:
    """Quick look at a disease file, using the same column names DiseaseModel recognises.

    Returns {"ok": bool, "problems": [...], "summary": str, ...}; never raises.
    """
    from difflib import get_close_matches
    p = Path(path).expanduser()
    out = {"ok": False, "path": str(p), "problems": []}
    if not p.exists():
        out["problems"].append(f"I can't find the file {p}. Check the path (put it in quotes if it has spaces).")
        return out
    if p.suffix.lower() not in DISEASE_SUFFIXES:
        out["problems"].append(f"Disease data must be a CSV or Excel file ({', '.join(DISEASE_SUFFIXES)}); "
                               f"this one is {p.suffix or 'without an extension'}.")
        return out
    try:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
    except Exception as exc:          # unreadable file
        out["problems"].append(f"I couldn't read the file: {exc}")
        return out

    cols = {c: str(c).strip().lower() for c in df.columns}
    date_col = next((c for c, low in cols.items() if low in DATE_NAMES or get_close_matches(low, DATE_NAMES, 1, .9)), None)
    count_col = next((c for c, low in cols.items() if low in COUNT_NAMES or get_close_matches(low, COUNT_NAMES, 1, .9)), None)
    if date_col is None:
        out["problems"].append("No date column found. Name it 'date' or 'time' (found: "
                               + ", ".join(map(str, df.columns)) + ").")
    if count_col is None:
        out["problems"].append("No case-count column found. Name it 'cases' or 'count' (found: "
                               + ", ".join(map(str, df.columns)) + ").")
    if out["problems"]:
        return out

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().mean() > 0.2:      # e.g. 31-01-2020: try day-first dates
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    counts = pd.to_numeric(df[count_col], errors="coerce")
    good = dates.notna() & counts.notna()
    if good.sum() < 24:
        out["problems"].append(f"Only {int(good.sum())} usable rows; ClimAID needs at least two years of monthly data.")
        return out
    d = dates[good].sort_values()
    step = d.diff().dt.days.median()
    freq = "daily" if step <= 2 else "weekly" if step <= 10 else "monthly" if step <= 40 else "irregular"
    out.update(ok=True, rows=int(good.sum()), start=d.iloc[0], end=d.iloc[-1], frequency=freq,
               date_col=str(date_col), count_col=str(count_col),
               bad_rows=int((~good).sum()), total_cases=float(counts[good].sum()))
    notes = []
    if out["bad_rows"]:
        notes.append(f"{out['bad_rows']} row(s) have an unreadable date or count and will be dropped")
    if freq in ("daily", "weekly"):
        notes.append(f"the data look {freq}; they are summed to monthly totals")
    if (counts[good] < 0).any():
        notes.append("some counts are negative; ClimAID will report them")
    out["summary"] = (f"{out['rows']} {freq} records from {d.iloc[0]:%B %Y} to {d.iloc[-1]:%B %Y} "
                      f"(columns '{date_col}' and '{count_col}')" + (". Note: " + "; ".join(notes) if notes else ""))
    out["notes"] = notes
    return out


# --------------------------------------------------------------------------- running ClimAID

class Runner:
    """Runs ClimAID. Kept separate so the assistant can be tested without fitting models,
    and so a dashboard can reuse the same assistant with its own runner."""

    def __init__(self, log_path: str | None = "climaid_outputs/assistant_log.txt"):
        self.log_path = log_path

    @contextlib.contextmanager
    def _quiet(self):
        """ClimAID prints a lot while it works; send that to a log file instead of the chat."""
        if not self.log_path:
            yield
            return
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield

    def load_model(self, settings: dict):
        from climaid.climaid_model import DiseaseModel
        with self._quiet():
            return DiseaseModel(district=settings["district"], disease_file=settings["disease_file"],
                                disease_name=settings.get("disease_name") or "Disease",
                                weather_file=settings.get("weather_file"),
                                projection_file=settings.get("projection_file"))

    def forecast(self, dm, settings: dict) -> dict:
        kwargs = dict(horizon=int(settings.get("horizon") or 12), tuning=settings.get("tuning") or "balanced",
                      exclude_period=settings.get("exclude_period") or "2020",
                      save_report=True, output_dir=settings.get("output_dir") or "climaid_outputs/reports")
        if settings.get("forecast_origin"):
            kwargs["forecast_origin"] = settings["forecast_origin"]
        with self._quiet():
            return dm.forecast_v2(**kwargs)

    def project(self, dm, settings: dict) -> dict:
        kwargs = dict(end_year=int(settings.get("end_year") or 2050), tuning=settings.get("tuning") or "balanced",
                      exclude_period=settings.get("exclude_period") or "2020",
                      save_report=True, output_dir=settings.get("output_dir") or "climaid_outputs/reports")
        if settings.get("ssps"):
            kwargs["ssps"] = list(settings["ssps"])
        if settings.get("forecast_origin"):
            kwargs["forecast_origin"] = settings["forecast_origin"]
        with self._quiet():
            return dm.project_v2(**kwargs)


# --------------------------------------------------------------------------- summaries

def _history(dm):
    d = getattr(dm, "df_disease", None)
    if d is None or "time" not in d or getattr(dm, "target_col", None) not in d:
        return None
    return pd.DataFrame({"time": pd.to_datetime(d["time"]), "cases": d[dm.target_col]})


def summarise_forecast(result: dict, dm=None, disease_name="Disease", district="") -> dict:
    """Plain-language summary of a forecast_v2 result, from the report's own rule-based text."""
    from climaid.reporting_plain import plain_forecast_html, forecast_trust, select_primary_model
    forecasts, metrics, hind = result["forecasts"], result.get("metrics"), result.get("hindcast_metrics")
    try:
        history = _history(dm) if dm is not None else None
        if history is not None and len(forecasts):
            start = pd.to_datetime(next(iter(forecasts.values()))["time"]).min()
            history = history[history["time"] < start]
        summary_html, _, _, notes_html, _, primary, _ = plain_forecast_html(
            forecasts=forecasts, metrics=metrics, hindcast_metrics=hind, metadata=result.get("metadata", {}),
            disease_name=disease_name, district=district, history=history)
        summary = html_to_text(summary_html)
    except Exception:            # summarise without the "typical year" comparison
        primary, _ = select_primary_model(forecasts, metrics, hind)
        summary = ""
    _, source = select_primary_model(forecasts, metrics, hind)
    rating, reasons = forecast_trust(forecasts, primary, metrics, hind, len(forecasts[primary]))
    f = forecasts[primary].copy()
    f["time"] = pd.to_datetime(f["time"])
    lo, hi = ("q100", "q900") if {"q100", "q900"} <= set(f.columns) else ("q025", "q975")
    months = [(t.strftime("%b %Y"), float(m), float(a), float(b)) for t, m, a, b in
              zip(f["time"], f["q500"], f[lo], f[hi])]
    return {"kind": "forecast", "summary": summary, "rating": rating, "reasons": list(reasons),
            "primary": primary, "primary_source": source, "months": months,
            "metadata": dict(result.get("metadata", {})), "hindcast_metrics": hind, "warnings": list(result.get("metadata", {}).get("warnings", [])),
            "report_path": result.get("report_path")}


def summarise_projection(result: dict, disease_name="Disease", district="") -> dict:
    from climaid.reporting_plain import plain_scenario_html, scenario_trust
    outlook = result["outlook"]
    summary_html, _, notes_html, _ = plain_scenario_html(outlook, disease_name=disease_name, district=district,
                                                         sensitivity=result.get("sensitivity"),
                                                         backtest=result.get("backtest"))
    rating, reasons = scenario_trust(outlook, result.get("sensitivity"), result.get("backtest"))
    return {"kind": "projection", "summary": html_to_text(summary_html), "rating": rating,
            "reasons": list(reasons), "not_included": html_to_text(notes_html),
            "report_path": result.get("report_path")}


def format_months(months, limit=12) -> str:
    def n(x):
        return f"{x:,.0f}"
    rows = [f"  {m:<9} {n(e):>8}   ({n(a)} to {n(b)})" for m, e, a, b in months[:limit]]
    return "  Month     Expected   Likely range (8 in 10)\n" + "\n".join(rows)
