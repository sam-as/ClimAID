"""ClimAID v2 reporting layer.

This module augments (and does not replace) the original ClimAID reporting.py.
It renders probabilistic forecasts, hindcasts, calibration/coverage diagnostics,
leakage controls and model configuration while allowing the original deterministic
C-DSI report and CMIP6 visual/reporting functions to remain available.
"""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _json(obj):
    return json.dumps(obj, indent=2, default=str)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return html.escape(str(v))


# ======================================================================
# C-DSI v2 — ClimAID v2 Deterministic Scientific Interpreter
# ======================================================================
# Same design principle as the original (v1) C-DSI in reporting.py: a
# template-based, LLM-free narrative built *only* from precomputed
# numeric artifacts, so it carries zero risk of hallucinated content and
# is fully reproducible offline. The v1 C-DSI interprets a point-forecast
# ML pipeline and CMIP6 scenario trends; this interprets the v2
# probabilistic/mechanistic forecast and its own leakage-safe validation
# outputs (held-out evaluation, rolling hindcasts, interval coverage) --
# it does not simply repeat the v1 narrative, because v2 makes different
# claims (probabilistic, climate-mandatory, hindcast-validated) that need
# their own deterministic interpretation.

_COVERAGE_NOMINAL = {"coverage_50": 0.50, "coverage_80": 0.80, "coverage_95": 0.95}


def _calibration_label(empirical: float, nominal: float, n: int | None = None) -> str:
    """Classify empirical interval coverage against its nominal level.

    FIX: this used a fixed +/-15-point tolerance at every level, so 81%
    empirical coverage for a 95% interval was labelled "consistent" even
    though it means missing ~19% of months instead of 5%. The tolerance now
    scales with the binomial standard error at that level and sample size:
    +/- max(2 * sqrt(p(1-p)/n), 3 points). With 48 evaluated months that is
    about +/-6 points at 95%, +/-12 at 80% and +/-14 at 50%. Still fully
    deterministic and reproducible from the numbers.
    """
    if empirical is None or not np.isfinite(empirical):
        return "not assessable (insufficient held-out observations)"
    n = max(int(n or 12), 1)
    tol = max(2.0 * float(np.sqrt(nominal * (1.0 - nominal) / n)), 0.03)
    diff = empirical - nominal
    basis = f" [n = {n} months; tolerance ±{tol*100:.0f} points]"
    if diff < -tol:
        return "under-covered (intervals too narrow -- true values fell outside more often than the nominal rate)" + basis
    if diff > tol:
        return "over-covered (intervals wider than necessary at this nominal level)" + basis
    return "consistent with its nominal level" + basis


def _rank_models_by_wis(metrics_df: pd.DataFrame) -> list[tuple[str, float]]:
    if metrics_df is None or metrics_df.empty or "WIS" not in metrics_df.columns:
        return []
    # Ignore rows without a finite score (e.g. the "hindcast_error"
    # placeholder row forecast_v2 records when hindcasting fails) so a
    # failure is never ranked as if it were a model.
    valid = metrics_df[np.isfinite(pd.to_numeric(metrics_df["WIS"], errors="coerce"))]
    if valid.empty:
        return []
    agg = valid.groupby("model")["WIS"].mean().sort_values()
    return list(agg.items())


def _trend_sentence(frame: pd.DataFrame) -> str:
    """Describe the whole median trajectory, not just first vs last value.

    FIX: the earlier version compared only the first and last month, so a
    forecast that dipped mid-year and rose again was summarised as "falls",
    and the reported peak was simply whichever month was highest.
    """
    if frame is None or frame.empty or "q500" not in frame.columns or len(frame) < 2:
        return "The forecast horizon is too short to characterise a trend."
    y = frame["q500"].to_numpy(dtype=float)
    t = pd.to_datetime(frame["time"]).dt.strftime("%b %Y").tolist()
    i_max, i_min = int(np.argmax(y)), int(np.argmin(y))
    half = len(y) // 2
    first, second = float(np.mean(y[:half])), float(np.mean(y[half:]))
    denom = max(abs(first), 1.0)
    if (second - first) / denom > 0.15:
        drift = "is higher on average in the second half of the horizon than the first"
    elif (second - first) / denom < -0.15:
        drift = "is lower on average in the second half of the horizon than the first"
    else:
        drift = "is similar on average in the first and second halves of the horizon"
    text = (
        f"The median forecast starts at {y[0]:.1f} cases ({t[0]}), reaches its lowest point of "
        f"{y[i_min]:.1f} around {t[i_min]} and its highest of {y[i_max]:.1f} around {t[i_max]}, "
        f"and ends at {y[-1]:.1f} ({t[-1]}). Overall it {drift}."
    )
    if i_min not in (0, len(y) - 1) and y[-1] - y[i_min] > 0.25 * max(y.max() - y.min(), 1e-9):
        text += f" After the {t[i_min]} low it rises again, to {y[-1]:.1f} by {t[-1]}."
    q025, q975 = frame.get("q025"), frame.get("q975")
    if q025 is not None and q975 is not None:
        text += (f" The 95% interval spans {float(q025.min()):.0f}–{float(q975.max()):.0f} cases "
                 f"across the horizon.")
    return text


def _disagreement_sentence(forecasts: dict) -> str:
    """Flag when fitted models forecast very different totals."""
    tot = {k: float(pd.to_numeric(v["q500"], errors="coerce").sum())
           for k, v in forecasts.items() if k != "ensemble" and "q500" in v}
    if len(tot) < 2:
        return ""
    lo_k, hi_k = min(tot, key=tot.get), max(tot, key=tot.get)
    lo, hi = tot[lo_k], tot[hi_k]
    if hi <= 2.0 * max(lo, 1.0):
        return ""
    return (f"<p><strong>Models disagree substantially:</strong> total median cases over the "
            f"horizon range from {lo:.0f} ({html.escape(lo_k)}) to {hi:.0f} "
            f"({html.escape(hi_k)}). The ensemble median sits between them and should not be "
            f"read as a consensus.</p>")


def _calibration_note(metadata: dict, primary) -> str:
    cal = (metadata or {}).get("interval_calibration")
    if not cal:
        return ""
    rows = [r for r in cal.get("factors", []) if r.get("model") == primary]
    if not rows:
        return ""
    k95 = [r.get("k_95") for r in rows if r.get("k_95") is not None]
    k50 = [r.get("k_50") for r in rows if r.get("k_50") is not None]
    rng = lambda v: f"{min(v):.2f}" if min(v) == max(v) else f"{min(v):.2f}–{max(v):.2f}"
    return ("<p><strong>Intervals have been recalibrated.</strong> The coverage figures above describe the "
            "hindcasts <em>before</em> recalibration. Every interval shown in this report was then rescaled, "
            "per lead-time band, so it would have reached its stated coverage in those hindcasts "
            f"(split-conformal). For {html.escape(str(primary))}, 95% half-widths were multiplied by "
            f"{rng(k95)} and 50% half-widths by {rng(k50)} (values above 1 widen). The held-out table is "
            f"scored after recalibration. Calibration uses hindcasts up to "
            f"{cal.get('max_hindcast_lead', '?')} months ahead.</p>")


def generate_c_dsi_v2(
    *, disease_name: str, district: str, metadata: dict, forecasts: dict,
    metrics: pd.DataFrame | None = None, hindcast_metrics: pd.DataFrame | None = None,
) -> str:
    """Deterministic, LLM-free narrative interpretation of a ClimAID v2 forecast.

    Mirrors the v1 C-DSI's design (template-based, reproducible, no
    generative content) but is built from v2's own artifacts: the
    probabilistic forecast bundle, held-out evaluation metrics and rolling
    hindcast metrics. Returns an HTML fragment (not a full document).
    """
    metrics = metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame()
    hindcast_metrics = hindcast_metrics if isinstance(hindcast_metrics, pd.DataFrame) else pd.DataFrame()
    models = list(metadata.get("models", [])) or list(forecasts.keys())

    # --- 1. Primary model selection (deterministic, not a preference) ---
    # Prefer the ensemble when present (it is the most conservative summary
    # of model disagreement); otherwise the model with the lowest mean
    # hindcast WIS; otherwise the lowest held-out WIS; otherwise the first
    # fitted model. Every branch is a plain numeric comparison.
    # FIX: previously the ensemble was always interpreted when present, even
    # when another model scored better (e.g. Pune: poisson best in hindcasts,
    # seasonal_naive best held out). Now the best-scoring model is chosen:
    # lowest mean hindcast WIS, else lowest held-out WIS, else the ensemble,
    # else the first fitted model. The ensemble's rank is stated alongside.
    primary = None
    selection_reason = ""
    ensemble_note = ""
    for table, label in ((hindcast_metrics, "mean hindcast WIS"), (metrics, "held-out WIS")):
        ranked = [(m, w) for m, w in _rank_models_by_wis(table) if m in forecasts]
        if ranked:
            primary, best = ranked[0]
            selection_reason = f"the lowest {label} ({best:.3f}) of the fitted models"
            names = [m for m, _ in ranked]
            if "ensemble" in names and primary != "ensemble":
                r = names.index("ensemble")
                ensemble_note = (f"The ensemble ranked {r + 1} of {len(names)} on the same score "
                                 f"({ranked[r][1]:.3f}).")
            break
    if primary is None:
        if "ensemble" in forecasts:
            primary, selection_reason = "ensemble", "the per-quantile median ensemble (no scores were available to rank models)"
        elif models:
            primary, selection_reason = models[0], "the only fitted model available for interpretation"

    primary_frame = forecasts.get(primary) if primary else None
    trend_text = _trend_sentence(primary_frame) if primary_frame is not None else (
        "No forecast frame was available for the selected model."
    )

    # --- 2. Calibration assessment from hindcasts (preferred) or held-out eval ---
    # Only prefer hindcasts when they actually contain scored rows; an
    # error-only hindcast table must not block the held-out fallback.
    hindcast_usable = bool(_rank_models_by_wis(hindcast_metrics))
    calib_source_df = hindcast_metrics if hindcast_usable else metrics
    calib_source_label = "rolling historical hindcasts" if hindcast_usable else "the held-out evaluation window"
    calib_lines = []
    if not calib_source_df.empty and primary in set(calib_source_df.get("model", [])):
        row = calib_source_df[calib_source_df["model"] == primary]
        n_obs = int(pd.to_numeric(row["n"], errors="coerce").fillna(0).sum()) if "n" in row.columns else 12 * len(row)
        for col, nominal in _COVERAGE_NOMINAL.items():
            if col in row.columns:
                emp = float(row[col].mean())
                calib_lines.append(
                    f"<li>{int(nominal*100)}% interval: empirical coverage {emp*100:.0f}% "
                    f"— {_calibration_label(emp, nominal, n_obs)}.</li>"
                )
    calib_html = "".join(calib_lines) or (
        "<li>No held-out or hindcast observations were available to assess calibration for this run.</li>"
    )

    # --- 3. Model comparison across hindcast origins ---
    ranked_hindcast = _rank_models_by_wis(hindcast_metrics)
    if len(ranked_hindcast) >= 2:
        best_name, best_wis = ranked_hindcast[0]
        second_name, second_wis = ranked_hindcast[1]
        rel_gap = (second_wis - best_wis) / max(best_wis, 1e-9)
        comparison_text = (
            f"Across {hindcast_metrics['origin'].nunique() if 'origin' in hindcast_metrics.columns else 'the evaluated'} "
            f"rolling hindcast origins, <strong>{html.escape(str(best_name))}</strong> had the lowest mean WIS "
            f"({best_wis:.3f}), "
            + (
                f"comparable to <strong>{html.escape(str(second_name))}</strong> ({second_wis:.3f})."
                if rel_gap < 0.10
                else f"ahead of the next-best model, <strong>{html.escape(str(second_name))}</strong> ({second_wis:.3f})."
            )
        )
    elif len(ranked_hindcast) == 1:
        comparison_text = (
            f"Only one model produced usable hindcast scores "
            f"(<strong>{html.escape(str(ranked_hindcast[0][0]))}</strong>, mean WIS {ranked_hindcast[0][1]:.3f}); "
            f"no cross-model comparison is possible from this run."
        )
    elif "error" in hindcast_metrics.columns and hindcast_metrics["error"].notna().any():
        err = str(hindcast_metrics["error"].dropna().iloc[0])
        comparison_text = (
            "Rolling hindcasts were requested but failed, so no cross-model comparison "
            f"is possible for this run. Reported cause: <code>{html.escape(err)}</code>"
        )
    else:
        comparison_text = (
            "No rolling hindcast scores were available, so v2 cannot make a deterministic "
            "cross-model performance comparison for this run. Enable hindcasts "
            "(<code>run_hindcasts=True</code>) to get one."
        )

    # --- 4. Structural caveats restated deterministically from metadata ---
    pop_available = bool(metadata.get("population_available"))
    population_note = (
        "Population-at-risk was available, so the renewal component applied susceptible depletion."
        if pop_available else
        "No population-at-risk value was available, so the renewal component (if used) ran in "
        "relative-incidence mode: it reports relative case dynamics, not an absolute outbreak "
        "probability, and susceptible depletion was not modelled."
    )
    source = metadata.get("forecast_climate_source") or "unspecified"
    source_note = {
        "observed": "Future climate for this forecast came from observed records after the forecast origin.",
        "observed_climate": "Future climate for this forecast came from observed records after the forecast origin.",
        "observed_climate_for_hindcast": "Future climate for this forecast came from observed records after the forecast origin (auto-selected because a full post-origin observed period was available).",
        "projection": "Future climate for this forecast came from CMIP6/SSP scenario projections, not observations.",
        "CMIP6_projection": "Future climate for this forecast came from CMIP6/SSP scenario projections, not observations.",
    }.get(source, f"Future climate source for this forecast was recorded as '{html.escape(str(source))}'.")

    warnings_list = metadata.get("warnings") or []
    warnings_html = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings_list) or "<li>None recorded.</li>"

    return f"""
<div class="cdsi-v2">
<p><strong>Report mode:</strong> ClimAID v2 Deterministic Scientific Interpreter (C-DSI v2) —
this section is generated entirely from the run's own numeric artifacts (forecast quantiles,
held-out evaluation, rolling hindcasts). It contains no LLM-generated or invented content.</p>

<h3>1. Forecast summary</h3>
<p>Primary model for this interpretation: <strong>{html.escape(str(primary or 'n/a'))}</strong>
(selected as {html.escape(selection_reason or 'the only available result')}). {html.escape(ensemble_note)}</p>
<p>{trend_text}</p>
{_disagreement_sentence(forecasts)}

<h3>2. Interval calibration</h3>
<p>Assessed against {html.escape(calib_source_label)}:</p>
<ul>{calib_html}</ul>
{_calibration_note(metadata, primary)}

<h3>3. Model comparison</h3>
<p>{comparison_text}</p>

<h3>4. Structural caveats</h3>
<ul>
<li>{population_note}</li>
<li>{source_note}</li>
<li>Ensemble/threshold agreement across models is not presented as a calibrated outbreak
probability; only the interval-coverage figures above quantify calibration, and only to the
extent the held-out/hindcast sample size supports.</li>
</ul>

<h3>5. Methodological warnings recorded during fitting</h3>
<ul>{warnings_html}</ul>
</div>
""".strip()


def _plain_period(metadata):
    def m(x):
        try:
            return pd.Timestamp(str(x)).strftime("%B %Y")
        except Exception:
            return ""
    a, b, o = m(metadata.get("forecast_start")), m(metadata.get("forecast_end")), m(metadata.get("forecast_origin"))
    return html.escape(f"{a} to {b} · uses data up to {o}" if a and b else "")


def generate_forecast_report(
    *, disease_name: str, district: str, bundle, metrics: pd.DataFrame | None = None,
    hindcast_metrics: pd.DataFrame | None = None, metadata: dict | None = None,
    legacy_report_text: str | None = None, title: str | None = None,
    warnings_list: list[str] | None = None, history: pd.DataFrame | None = None,
    observed: pd.DataFrame | None = None, exclude_times=(),
):
    """Generate a self-contained HTML report for a v2 forecast.

    The report opens with a plain-language layer for non-specialists (short
    version, month-by-month table, trust rating, caveats, glossary; see
    reporting_plain.py). All technical content follows in a collapsible
    "Technical details" section. `history` (cases up to the forecast start)
    enables "compared with a typical year"; `observed` (cases after the start,
    if known) enables "what actually happened".
    """
    metadata = {**getattr(bundle, "metadata", {}), **(metadata or {})}
    forecasts = getattr(bundle, "forecasts", bundle)
    warnings_list = list(warnings_list or [])
    warnings_list.extend(metadata.get("warnings", []) or [])
    title = title or f"ClimAID v2 Forecast Report — {disease_name}"

    # ---- plain-language layer (built first so it can carry the plotly script)
    from climaid.reporting_plain import partner_logo_html, plain_forecast_html, plain_forecast_chart, PLAIN_CSS, RESIZE_JS, _district_name
    try:
        (p_summary, p_table, p_trust, p_notes, p_gloss, p_primary, p_chart_data) = plain_forecast_html(
            forecasts=forecasts, metrics=metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame(),
            hindcast_metrics=hindcast_metrics if isinstance(hindcast_metrics, pd.DataFrame) else pd.DataFrame(),
            metadata=metadata, disease_name=disease_name, district=district,
            history=history, observed=observed, exclude_times=exclude_times)
        p_chart = plain_forecast_chart(*p_chart_data, disease_name)
    except Exception as exc:                       # never let the plain layer break the report
        p_summary = f"<p>The plain-language summary could not be produced ({html.escape(str(exc))}).</p>"
        p_table = p_trust = p_notes = p_gloss = p_chart = ""

    def df_html(frame):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame.to_html(index=False, classes="metrics", border=0, float_format=lambda x: f"{x:.3f}")
        return "<p>No evaluation table was provided.</p>"

    metrics_html = df_html(metrics)
    hindcast_html = df_html(hindcast_metrics)

    c_dsi_v2_html = generate_c_dsi_v2(
        disease_name=disease_name, district=district, metadata=metadata,
        forecasts=forecasts, metrics=metrics, hindcast_metrics=hindcast_metrics,
    )

    model_rows = "".join(
        f"<span class='chip'>{html.escape(str(name))}</span>" for name in forecasts.keys()
    )

    plot_html = "<p>Interactive forecast plot unavailable.</p>"
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for name, frame in forecasts.items():
            if name == "ensemble" or "q500" not in frame.columns:
                continue
            fig.add_trace(go.Scatter(
                x=frame["time"], y=frame["q500"], mode="lines", name=f"{name} median"
            ))
        ens = forecasts.get("ensemble")
        if ens is not None and "q500" in ens.columns:
            if "q025" in ens.columns and "q975" in ens.columns:
                fig.add_trace(go.Scatter(
                    x=ens["time"], y=ens["q975"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=ens["time"], y=ens["q025"], mode="lines",
                    fill="tonexty", line=dict(width=0), name="Ensemble 95% interval"
                ))
            fig.add_trace(go.Scatter(
                x=ens["time"], y=ens["q500"], mode="lines",
                line=dict(width=3), name="Ensemble median"
            ))
        fig.update_layout(
            template="plotly_white", height=520,
            margin=dict(l=60, r=30, t=50, b=60),
            title="Probabilistic disease forecast",
            xaxis_title="Time", yaxis_title="Cases",
        )
        plot_html = fig.to_html(full_html=False, include_plotlyjs=(not p_chart))
    except Exception as exc:
        warnings_list.append(f"Interactive plotting unavailable: {exc}")

    legacy_section = ""
    if legacy_report_text:
        try:
            import markdown
            legacy_html = markdown.markdown(
                legacy_report_text, extensions=["extra", "tables", "sane_lists"]
            )
        except Exception:
            legacy_html = f"<pre>{html.escape(legacy_report_text)}</pre>"
        legacy_section = (
            "<div class='card'><h2>Appendix — ClimAID v1 (legacy) C-DSI report</h2>"
            "<p class='muted'>Unchanged v1 output, included for comparison only. "
            "It describes a different pipeline (point-forecast ML + CMIP6 scenario trends) "
            "than the v2 sections above and should not be read as validating the v2 forecast.</p>"
            f"<div class='legacy'>{legacy_html}</div></div>"
        )

    validation_contract = {
        "climate_mandatory": True,
        "forecast_origin": metadata.get("forecast_origin"),
        "forecast_start": metadata.get("forecast_start"),
        "forecast_end": metadata.get("forecast_end"),
        "horizon": metadata.get("horizon"),
        "training_observations": metadata.get("n_training_observations"),
        "models": list(forecasts.keys()),
        "quantiles": metadata.get("quantiles"),
        "seasonal_period": metadata.get("seasonal_period"),
        "population_available": metadata.get("population_available"),
        "climate_variables": metadata.get("climate_variables"),
        "forecast_climate_source": metadata.get("forecast_climate_source"),
        "leakage_control": [
            "Chronological forecast origins only",
            "Preprocessing/model fitting confined to the historical information set",
            "No future disease observations used as predictors",
            "Residual learning uses temporal out-of-fold predictions",
            "Forecast-origin climate statistics are fitted only on information available at that origin",
        ],
    }

    warn_html = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings_list) or "<li>None recorded.</li>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#f6f7f9;color:#1d2430;margin:0}}
.wrap{{max-width:1200px;margin:0 auto;padding:28px}}
header{{background:#463f3a;color:#fff;border-radius:18px;padding:30px 34px;margin-bottom:22px}}
h1{{margin:0 0 8px;font-size:30px}}h2{{margin-top:0}}.sub{{opacity:.84}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #e6e8eb;border-radius:16px;padding:20px;box-shadow:0 5px 18px rgba(0,0,0,.04);margin-bottom:18px}}
.metric{{font-size:28px;font-weight:700}}.muted{{color:#6b7280;font-size:13px}}
.chip{{display:inline-block;border:1px solid #d7dce2;border-radius:99px;padding:7px 10px;margin:4px;font-size:13px;background:#fafafa}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #edf0f2;text-align:right}}th:first-child,td:first-child{{text-align:left}}
pre{{background:#111827;color:#e5e7eb;padding:16px;border-radius:12px;overflow:auto;font-size:12px}}
li{{margin:7px 0}}.legacy{{line-height:1.65}}.legacy h1{{font-size:25px;color:#222;background:none;padding:0}}
.cdsi-v2 h3{{font-size:15px;margin:18px 0 6px}}.cdsi-v2 p,.cdsi-v2 li{{font-size:14px;line-height:1.6}}
.cdsi-v2>p:first-child{{background:#f0f4f8;border-radius:10px;padding:12px 14px;font-size:13px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr}}.wrap{{padding:14px}}}}
{PLAIN_CSS}</style></head><body><div class='wrap'>
<header><div><h1>{html.escape(disease_name)} forecast for {html.escape(_district_name(district))}</h1><div class='sub'>{_plain_period(metadata)}</div></div>{partner_logo_html()}</header>
<div class='card'><h2>The short version</h2>{p_summary}</div>
<div class='card'><h2>Expected cases, month by month</h2>{p_chart}{p_table}</div>
<div class='card'><h2>How much can you trust this?</h2>{p_trust}</div>
<div class='card'><h2>Things to keep in mind</h2>{p_notes}</div>
<div class='card'><h2>Words used in this report</h2>{p_gloss}</div>
<details class='tech'><summary>Technical details (for specialists)</summary>
<p class='muted'>Everything below is the full technical output: every model, every score, the validation rules and the deterministic C-DSI v2 interpretation.</p>

<div class='grid'>
<div class='card'><div class='muted'>Forecast horizon</div><div class='metric'>{_fmt(metadata.get('horizon'))}</div><div class='muted'>periods</div></div>
<div class='card'><div class='muted'>Training observations</div><div class='metric'>{_fmt(metadata.get('n_training_observations'))}</div><div class='muted'>available at origin</div></div>
<div class='card'><div class='muted'>Simulations</div><div class='metric'>{_fmt(metadata.get('n_simulations'))}</div><div class='muted'>for probabilistic models</div></div>
<div class='card'><div class='muted'>Forecast source</div><div class='metric' style='font-size:18px'>{_fmt(metadata.get('forecast_climate_source'))}</div><div class='muted'>{_fmt(metadata.get('forecast_start'))} → {_fmt(metadata.get('forecast_end'))}</div></div>
</div>
<div class='card'><h2>v2 models fitted</h2>{model_rows}</div>
<div class='card'><h2>v2 probabilistic forecast</h2>{plot_html}</div>
<div class='card'><h2>C-DSI v2 — Deterministic Forecast Interpretation</h2>{c_dsi_v2_html}</div>
<div class='card'><h2>v2 held-out evaluation (numeric)</h2>{metrics_html}</div>
<div class='card'><h2>v2 historical hindcast (numeric)</h2>{hindcast_html}</div>
<div class='card'><h2>v2 Validation contract</h2><pre>{html.escape(_json(validation_contract))}</pre></div>
<div class='card'><h2>v2 methodological warnings</h2><ul>{warn_html}</ul></div>
<div class='card'><h2>v2 model specification</h2>
<p>The v2 framework retains climate as a mandatory information source, while combining disease history, seasonal structure, a climate-informed renewal process, statistical/machine-learning models, probabilistic simulation and leakage-safe temporal validation.</p>
<p>The renewal component uses a discretised gamma generation interval and dynamically updates susceptibility only when a valid population-at-risk quantity is supplied. Without population information it operates in relative-incidence mode and explicitly reports that limitation.</p>
<p>Ensemble agreement across CMIP6 models is not presented as a calibrated outbreak probability. Retrospective hindcast calibration is required before such a probability statement can be made.</p></div>
{legacy_section}

</details>
{RESIZE_JS}
</div></body></html>"""


def save_forecast_report(report_html: str, output_dir="climaid_outputs/reports", filename="climaid_v2_forecast.html"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(report_html, encoding="utf-8")
    return str(path)
