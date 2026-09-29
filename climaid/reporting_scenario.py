"""HTML report for ClimAID v2 hybrid scenario outlooks (near-term forecast
blended into CMIP6 scenario projections), with a deterministic C-DSI-style
interpretation built only from the outlook's own numbers."""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
import pandas as pd

SSP_LABELS = {"ssp119": "SSP1-1.9 (very low emissions)", "ssp126": "SSP1-2.6 (low emissions)",
              "ssp245": "SSP2-4.5 (intermediate)", "ssp370": "SSP3-7.0 (high)",
              "ssp434": "SSP4-3.4", "ssp460": "SSP4-6.0", "ssp585": "SSP5-8.5 (very high emissions)"}
SSP_COLORS = {"ssp119": "#1a9850", "ssp126": "#66bd63", "ssp245": "#fdae61", "ssp370": "#f46d43", "ssp585": "#a50026"}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _lab(s):
    from climaid.reporting_plain import SSP_PLAIN
    code = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0",
            "ssp460": "SSP4-6.0", "ssp585": "SSP5-8.5"}.get(s, s.upper())
    return f"{SSP_PLAIN[s]} ({code})" if s in SSP_PLAIN else code


def _pct(x):
    return f"{x * 100:+.0f}%"


def _direction(share):
    if share >= 0.9:
        return "almost all simulated outcomes show an increase"
    if share >= 0.66:
        return "most simulated outcomes show an increase"
    if share > 0.33:
        return "simulated outcomes are split between increases and decreases, so the direction of change is uncertain"
    if share > 0.1:
        return "most simulated outcomes show a decrease"
    return "almost all simulated outcomes show a decrease"


def scenario_interpretation(outlook, sensitivity=None) -> str:
    dec, base, meta = outlook.decades, outlook.baseline, outlook.metadata
    if dec.empty:
        return "<p>No complete decade (at least 5 years) is covered by the projection, so no decadal interpretation is given.</p>"
    decades = sorted(dec["decade"].unique())
    last = decades[-1]; mid = decades[len(decades) // 2] if len(decades) > 2 else None
    peak = ", ".join(MONTHS[m - 1] for m in base["peak_months"])
    parts = [f"<p>Changes are measured against the model's own simulation of the {base['years']} baseline "
             f"({base['model_annual_mean']:.0f} cases a year; observed mean {base['observed_annual_mean']:.0f}), "
             f"so any bias in the model largely cancels. Ranges are 10th–90th percentiles across all "
             f"{len(meta['gcms'])} climate models, bootstrap refits and count noise.</p><ul>"]
    for s in meta["ssps"]:
        r = dec[(dec.ssp == s) & (dec.decade == last)]
        if r.empty:
            continue
        r = r.iloc[0]
        pk_change = r.peak_season_median / max(base["model_peak_season_mean"], 1e-9) - 1
        txt = (f"<li><strong>{html.escape(_lab(s))}, {last}:</strong> median {r.annual_median:.0f} cases a year, "
               f"{_pct(r.change_vs_baseline_median)} versus baseline (range {_pct(r.change_p10)} to {_pct(r.change_p90)}); "
               f"{_direction(r.share_gcm_samples_increase)} ({r.share_gcm_samples_increase:.0%}). "
               f"Peak season ({peak}): {_pct(pk_change)}.")
        if mid:
            m = dec[(dec.ssp == s) & (dec.decade == mid)]
            if not m.empty:
                txt += f" For the {mid}: {_pct(m.iloc[0].change_vs_baseline_median)}."
        parts.append(txt + "</li>")
    parts.append("</ul>")
    lastrows = dec[dec.decade == last]
    if len(lastrows) >= 2:
        hi, lo = lastrows.loc[lastrows.change_vs_baseline_median.idxmax()], lastrows.loc[lastrows.change_vs_baseline_median.idxmin()]
        parts.append(f"<p><strong>Scenario contrast:</strong> by the {last}, {html.escape(_lab(hi.ssp))} is "
                     f"{(hi.change_vs_baseline_median - lo.change_vs_baseline_median) * 100:.0f} percentage points above "
                     f"{html.escape(_lab(lo.ssp))} at the median. The overlap of their ranges shows how much of that "
                     f"difference is distinguishable from climate-model and statistical uncertainty.</p>")
    if sensitivity is not None and not sensitivity.decades.empty:
        rows = []
        for s in meta["ssps"]:
            a = dec[(dec.ssp == s) & (dec.decade == last)]; b = sensitivity.decades[(sensitivity.decades.ssp == s) & (sensitivity.decades.decade == last)]
            if len(a) and len(b):
                rows.append((s, a.iloc[0].change_vs_baseline_median, b.iloc[0].change_vs_baseline_median))
        if rows:
            gap = max(abs(x - y) for _, x, y in rows)
            items = "; ".join(f"{html.escape(_lab(s))}: {_pct(x)} (main) vs {_pct(y)} (anomaly-only)" for s, x, y in rows)
            verdict = ("The conclusions depend strongly on this assumption." if gap > 0.10
                       else "The conclusions are not sensitive to this assumption.")
            parts.append(f"<p><strong>Sensitivity to how climate effects are identified ({last}):</strong> {items}. "
                         f"The main run learns climate effects from the seasonal cycle (assumes seasonality is largely "
                         f"climate-driven); the anomaly-only run learns them only from year-to-year deviations. {verdict}</p>")
    structs = meta.get("lag_structures") or []
    if len(structs) > 1:
        def fmt(st):
            return ", ".join(f"{k} lag {v[0]}" for k, v in st.items()) or "none"
        with_t = sum(1 for st in structs if "temperature" in st)
        parts.append(
            f"<p><strong>Structural uncertainty:</strong> {len(structs)} different climate-lag structures fit the "
            f"history almost equally well (within 2% on blocked cross-validation), so they are averaged. "
            f"{with_t} of them include temperature. Because temperature, rainfall and humidity share a seasonal "
            f"cycle, the data cannot fully separate their effects, and the structures disagree about how much "
            f"warming matters. If warming is known to drive transmission here, projections that give it little "
            f"weight will understate change; supplying lags from v1 or prior studies narrows this.</p>"
            "<ul class='muted'>" + "".join(f"<li>{html.escape(fmt(st))}</li>" for st in structs) + "</ul>")
    notes = meta.get("notes") or []
    if notes:
        parts.append("<p><strong>Warnings:</strong></p><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>")
    return "\n".join(parts)


def _backtest_html(backtest):
    if not backtest or backtest.get("table") is None or backtest["table"].empty:
        return ""
    t, sm = backtest["table"], backtest["summary"]
    e, e0 = sm["median_abs_pct_error"], sm["median_abs_pct_error_training_mean"]
    if e < 0.8 * e0:
        verdict = ("The climate response clearly beat simply repeating the training-period average, which supports "
                   "using it for projection.")
    elif e <= 1.2 * e0:
        verdict = ("The climate response did about as well as simply repeating the training-period average, so this "
                   "backtest cannot confirm it adds skill. That is expected when the test years' climate differs "
                   "little from the training years; it mainly checks the model is not worse and its ranges are sensible.")
    else:
        verdict = ("The climate response did <strong>worse</strong> than simply repeating the training-period average on "
                   "these years, so its projected changes should be treated with extra caution.")
    tbl = t.assign(**{"inside 80% range": t.inside_80.map({True: "yes", False: "no"})})[
        ["year", "observed", "projected_median", "p10", "p90", "inside 80% range"]].round(0)
    return (f"<div class='card'><h2>Backtest of the long-term climate response</h2>"
            f"<p>Everything (lag selection, fitting, bootstrap) was redone using only data before {sm['test_years'].split('-')[0]}, "
            f"then the held-out years {html.escape(sm['test_years'])} were projected from their <em>observed</em> climate, "
            f"exactly as future scenarios are. Observed annual cases fell inside the 80% range in "
            f"{sm['share_inside_80']:.0%} of {sm['n_years']} years. Median error {sm['median_abs_pct_error']:.0%}, versus "
            f"{sm['median_abs_pct_error_training_mean']:.0%} for repeating the training-period average. {verdict} "
            f"With only {sm['n_years']} test years this is a rough check, not a proof.</p>"
            f"{tbl.to_html(index=False, border=0, classes='metrics')}</div>")


def _tree_html(outlook):
    t = getattr(outlook, "tree_comparison", None)
    if t is None or t.empty:
        return ""
    main = outlook.decades.set_index(["ssp", "decade"]).change_vs_baseline_median
    t = t.copy()
    t["main projection"] = [main.get((s, d), np.nan) for s, d in zip(t.ssp, t.decade)]
    names = {"random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
    tbl = pd.DataFrame({
        "model": t.model.map(lambda m: names.get(m, m)), "scenario": t.ssp.map(_lab), "decade": t.decade,
        "change (median over climate models)": t.change_median_across_gcms.map(_pct),
        "range over climate models": t.change_min_gcm.map(_pct) + " to " + t.change_max_gcm.map(_pct),
        "main projection": t["main projection"].map(lambda x: _pct(x) if np.isfinite(x) else "—"),
        "months outside training climate": t.share_months_outside_training_range.map(lambda x: f"{x:.0%}"),
    })
    out_share = t.share_months_outside_training_range.max()
    note = (f"In up to {out_share:.0%} of projected months at least one climate variable is outside the range seen in "
            "training. There, tree models stay at the level of the most extreme past months, so their change is "
            "expected to be smaller than the main projection's." if out_share > 0.05 else
            "Projected climate stays mostly within the training range, so tree models are a fair comparison here.")
    return ("<div class='card'><h2>Comparison with tree-based models</h2><p>Random forest and gradient boosting, fitted to "
            "the same history and run through the same bias-corrected climate models, as a check on the main "
            "projection. Tree models cannot extend a trend beyond the climate they were trained on. " + html.escape(note) +
            " They are shown for comparison only; the main projection above uses the climate-response model.</p>"
            f"{tbl.to_html(index=False, border=0, classes='metrics')}</div>")


def _dl_html(outlook):
    curves = outlook.metadata.get("distributed_lag_curves")
    if not curves:
        return ""
    names = {"temperature": "Temperature", "rainfall": "Rainfall", "humidity": "Humidity", "enso": "ENSO"}
    rows = "".join(
        f"<tr><td>{names.get(v, v)}</td><td>0–{len(c['by_lag']) - 1}</td><td>{c['cumulative']:+.3f}</td>"
        f"<td>{' '.join(f'{b:+.2f}' for b in c['by_lag'])}</td></tr>" for v, c in curves.items())
    return ("<div class='card'><h2>Distributed lag curves</h2><p>Effect of a one-standard-deviation change at each lag "
            "(log scale; standardised units). The <strong>cumulative</strong> column is the effect of a sustained change and "
            "drives the projections. Treat the curve <em>shape</em> (which lag matters most) with caution: in synthetic tests "
            "the cumulative effect was recovered well but the timing was not.</p>"
            "<table class='metrics'><thead><tr><th>Variable</th><th>Lags (months)</th><th>Cumulative effect</th>"
            f"<th>Effect by lag</th></tr></thead><tbody>{rows}</tbody></table></div>")


def _population_html(outlook):
    dp = getattr(outlook, "decades_with_population", None)
    if dp is None or dp.empty:
        return ""
    d = dp.assign(scenario=dp.ssp.map(_lab),
                  **{"cases/yr (median)": dp.annual_median.round(0).astype(int),
                     "change (median)": dp.change_vs_baseline_median.map(_pct),
                     "change 10–90%": dp.change_p10.map(_pct) + " to " + dp.change_p90.map(_pct)})
    tbl = d[["scenario", "decade", "cases/yr (median)", "change (median)", "change 10–90%"]]
    return ("<div class='card'><h2>With projected population change</h2><p>Same projections, scaled by projected "
            "population relative to the baseline (assumes constant incidence per person). The main tables above "
            "isolate the climate effect; this table shows climate and population together.</p>"
            f"{tbl.to_html(index=False, border=0, classes='metrics')}</div>")


def generate_scenario_report(outlook, *, disease_name="Disease", district="Unknown", sensitivity=None,
                             backtest=None, title=None) -> str:
    import plotly.graph_objects as go
    from climaid.reporting_plain import partner_logo_html, plain_scenario_html, PLAIN_CSS, RESIZE_JS, _district_name
    meta, base, ann, pooled = outlook.metadata, outlook.baseline, outlook.annual, outlook.pooled
    try:
        p_summary, p_trust, p_notes, p_gloss = plain_scenario_html(outlook, disease_name=disease_name, district=district,
                                                                   sensitivity=sensitivity, backtest=backtest)
    except Exception as exc:
        p_summary = f"<p>The plain-language summary could not be produced ({html.escape(str(exc))}).</p>"
        p_trust = p_notes = p_gloss = ""
    title = title or f"ClimAID v2 Climate Scenario Outlook — {disease_name}"

    fig = go.Figure()
    for s in meta["ssps"]:
        a = ann[ann.ssp == s]; col = SSP_COLORS.get(s, "#555")
        fig.add_trace(go.Scatter(x=a.year, y=a.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=a.year, y=a.p10, mode="lines", fill="tonexty", line=dict(width=0), fillcolor=col,
                                 opacity=.18, name=f"{_lab(s)}: likely range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=a.year, y=a.p50, mode="lines", line=dict(color=col, width=2.5), name=f"{_lab(s)}",
                                 hovertemplate="%{x}: about %{y:.0f} cases<extra></extra>"))
    fig.add_hline(y=base["model_annual_mean"], line_dash="dash", line_color="#6b7280",
                  annotation_text=f"baseline {base['years']}", annotation_position="bottom right")
    fig.update_layout(template="plotly_white", height=460, font=dict(size=14), margin=dict(l=50, r=20, t=20, b=40),
                      xaxis_title="Year", yaxis_title="Cases per year", legend=dict(orientation="h", y=-0.2))
    for t in fig.data:
        if t.fill == "tonexty":
            t.update(opacity=None, fillcolor=_rgba(t.fillcolor, .18))
    chart1 = fig.to_html(full_html=False, include_plotlyjs=True)

    origin = pd.Timestamp(meta["forecast_origin"])
    near_end = origin + pd.DateOffset(months=meta["near_term_months"])
    blend_end = near_end + pd.DateOffset(months=meta["blend_months"])
    fig2 = go.Figure()
    first = pooled[pooled.time <= origin + pd.DateOffset(months=36)]
    for s in meta["ssps"]:
        p = first[first.ssp == s]; col = SSP_COLORS.get(s, "#555")
        fig2.add_trace(go.Scatter(x=p.time, y=p.q975, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=p.time, y=p.q025, fill="tonexty", line=dict(width=0), fillcolor=_rgba(col, .15),
                                  name=f"{_lab(s)} 95% interval", hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=p.time, y=p.q500, line=dict(color=col, width=2), name=f"{_lab(s)} median"))
    if meta["near_term_months"]:
        fig2.add_vrect(x0=origin, x1=near_end, fillcolor="#2563eb", opacity=.05, line_width=0,
                       annotation_text="near-term forecast", annotation_position="top left")
    if meta["blend_months"]:
        fig2.add_vrect(x0=near_end, x1=blend_end, fillcolor="#9333ea", opacity=.05, line_width=0,
                       annotation_text="hand-over", annotation_position="top left")
    fig2.update_layout(template="plotly_white", height=420, title="First three years, monthly",
                       xaxis_title="Month", yaxis_title="Cases", legend=dict(orientation="h", y=-0.25))
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    dec = outlook.decades.copy()
    if not dec.empty:
        dec["scenario"] = dec.ssp.map(_lab)
        dec_tbl = dec.assign(
            **{"cases/yr (median)": dec.annual_median.round(0).astype(int),
               "cases/yr 10–90%": dec.annual_p10.round(0).astype(int).astype(str) + "–" + dec.annual_p90.round(0).astype(int).astype(str),
               "change (median)": dec.change_vs_baseline_median.map(_pct),
               "change 10–90%": dec.change_p10.map(_pct) + " to " + dec.change_p90.map(_pct),
               "share of outcomes ↑": dec.share_gcm_samples_increase.map(lambda x: f"{x:.0%}")}
        )[["scenario", "decade", "years", "cases/yr (median)", "cases/yr 10–90%", "change (median)", "change 10–90%", "share of outcomes ↑"]]
        dec_html = dec_tbl.to_html(index=False, border=0, classes="metrics", escape=True)
    else:
        dec_html = "<p>No complete decades.</p>"

    coef = meta.get("climate_coefficients", {})
    coef_html = "".join(f"<li><code>{html.escape(k)}</code>: {v:+.3f}</li>" for k, v in coef.items()) or "<li>—</li>"
    chips = "".join(f"<span class='chip'>{html.escape(x)}</span>" for x in meta["gcms"])
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
.metric{{font-size:24px;font-weight:700}}.muted{{color:#6b7280;font-size:13px}}
.chip{{display:inline-block;border:1px solid #d7dce2;border-radius:99px;padding:6px 10px;margin:3px;font-size:13px;background:#fafafa}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #edf0f2;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.interp p,.interp li{{font-size:14px;line-height:1.6}}.interp>p:first-child{{background:#f0f4f8;border-radius:10px;padding:12px 14px;font-size:13px}}
li{{margin:6px 0}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.grid{{grid-template-columns:1fr}}.wrap{{padding:14px}}}}
{PLAIN_CSS}</style></head><body><div class='wrap'>
<header><div><h1>How climate change could affect {html.escape(disease_name)} in {html.escape(_district_name(district))}</h1><div class='sub'>Projections from {(origin + pd.Timedelta(days=1)):%B %Y} to {meta['end_year']}, compared with {html.escape(base['years'])}</div></div>{partner_logo_html()}</header>
<div class='card'><h2>The short version</h2>{p_summary}</div>
<div class='card'><h2>Cases per year under each future</h2><p class='muted'>Lines show the most likely number of cases each year; shaded bands show the likely range. The dashed line is the {html.escape(base['years'])} average.</p>{chart1}</div>
<div class='card'><h2>How confident are we?</h2>{p_trust}</div>
<div class='card'><h2>What these projections do not include</h2>{p_notes}</div>
<div class='card'><h2>Words used in this report</h2>{p_gloss}</div>
<details class='tech'><summary>Technical details (for specialists)</summary>
<p class='muted'>Full technical output: decade tables, the deterministic C-DSI v2 scenario interpretation, sensitivity and backtest results, methods and model coefficients.</p>

<div class='grid'>
<div class='card'><div class='muted'>Scenarios</div><div class='metric'>{len(meta['ssps'])}</div><div class='muted'>{html.escape(', '.join(s.upper() for s in meta['ssps']))}</div></div>
<div class='card'><div class='muted'>Climate models</div><div class='metric'>{len(meta['gcms'])}</div><div class='muted'>bias-corrected to observations</div></div>
<div class='card'><div class='muted'>Baseline</div><div class='metric'>{base['model_annual_mean']:.0f}</div><div class='muted'>cases/yr, {html.escape(base['years'])}</div></div>
<div class='card'><div class='muted'>Hand-over</div><div class='metric'>{meta['near_term_months']}+{meta['blend_months']} mo</div><div class='muted'>near-term forecast, then blend</div></div>
</div>

<div class='card interp'><h2>C-DSI v2 — Scenario interpretation</h2>
<p><strong>Report mode:</strong> deterministic interpretation generated only from this outlook's own numbers; no LLM-generated content.</p>
{scenario_interpretation(outlook, sensitivity)}</div>
<div class='card'><h2>Decade summary</h2>{dec_html}</div>
{_population_html(outlook)}
{_tree_html(outlook)}
{_dl_html(outlook)}
{_backtest_html(backtest)}
<div class='card'>{chart2}</div>
<div class='card'><h2>How this outlook is built</h2>
<ol>
<li><strong>Bias correction.</strong> Each climate model's series is shifted so its {html.escape(base['years'])} monthly means match observed climate ({html.escape(meta['bias_correction'])}). Its projected change is kept; its offset from reality is removed.</li>
<li><strong>Near term (first {meta['near_term_months']} months).</strong> The ClimAID v2 ensemble forecast ({html.escape(', '.join(meta['near_term_models']) or 'not used')}), driven by each climate model's corrected climate. Recent case counts matter most here.</li>
<li><strong>Hand-over (next {meta['blend_months']} months).</strong> A linear mixture of the near-term and long-term distributions, as recent cases stop being informative.</li>
<li><strong>Long term.</strong> {html.escape(meta['long_term_model'])}. It explains {meta.get('deviance_explained_long_term', float('nan')):.0%} of the deviance in the {meta['n_training_months_long_term']} training months.</li>
{"<li><strong>Multi-district climate response.</strong> The climate coefficients were learned jointly with " + str(meta.get("pooled_districts", 0)) + " other district(s), each with its own baseline, which helps separate climate variables whose seasonal cycles coincide here.</li>" if meta.get("pooled_districts") else ""}
{"<li><strong>Temperature constrained to a thermal-suitability curve</strong> (" + html.escape(str((meta.get("temperature_curve") or {}).get("name"))) + ", T_min/T_opt/T_max = " + html.escape(str((meta.get("temperature_curve") or {}).get("tmin_topt_tmax"))) + " °C): the data estimate how strongly suitability matters, but not its shape. Verify the curve values against the source before publication.</li>" if meta.get("temperature_curve") else ""}
<li><strong>Pooling.</strong> Samples from all climate models are pooled within each scenario, so ranges include count noise, statistical-model uncertainty and climate-model spread.</li>
</ol>
<p class='muted'>Climate coefficients (standardised units):</p><ul class='muted'>{coef_html}</ul>
<p class='muted'>Climate models: {chips}</p></div>
<div class='card'><h2>Assumptions and limits</h2><ul>
<li>The climate–disease relationship estimated from {html.escape(base['years'])} is assumed to hold in the future.</li>
<li>Population, immunity, vector control, diagnostics and reporting are held at their historical levels. These projections isolate the effect of climate; they are not predictions of future case counts.</li>
<li>CMIP6 models represent ENSO variability poorly; ENSO terms mainly add noise far ahead.</li>
<li>Where projected climate goes beyond the training range (see warnings), the response is extrapolated and less reliable.</li>
</ul></div>

</details>
{RESIZE_JS}
</div></body></html>"""


def _rgba(color, a):
    c = str(color)
    if c.startswith("#") and len(c) == 7:
        r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{a})"
    return c


def save_scenario_report(report_html, output_dir="climaid_outputs/reports", filename="climaid_v2_scenarios.html"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / filename; p.write_text(report_html, encoding="utf-8"); return str(p)
