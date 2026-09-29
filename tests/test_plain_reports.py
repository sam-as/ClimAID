"""Plain-language layer of the v2 forecast and scenario reports."""
import re
import numpy as np
import pandas as pd

from climaid.reporting_plain import (forecast_trust, plain_forecast_html, _pathway_comparison,
                                     select_primary_model, _district_name)
from climaid.reporting_v2 import generate_forecast_report
from climaid.forecasting_v2 import ForecastBundle

Q = ["q025", "q050", "q100", "q250", "q500", "q750", "q900", "q950", "q975"]


def _frame(med, width=0.4, start="2021-01-01"):
    t = pd.date_range(start, periods=len(med), freq="MS")
    z = [-1.96, -1.645, -1.28, -.67, 0, .67, 1.28, 1.645, 1.96]
    return pd.DataFrame({"time": t, **{q: np.maximum(0, np.array(med) * (1 + width * k / 1.96)) for q, k in zip(Q, z)}})


def _metrics(rows):
    return pd.DataFrame(rows, columns=["model", "WIS", "coverage_50", "coverage_80", "coverage_95"])


def test_district_names_are_readable():
    assert _district_name("IN_Pune_Maharashtra") == "Pune"
    assert _district_name("Synthetic_District") == "Synthetic District"


def test_trust_rating_good_moderate_low():
    fc = {"a": _frame([10] * 12), "seasonal_naive": _frame([11] * 12)}
    good = _metrics([("a", 2.0, .5, .8, .95), ("seasonal_naive", 3.0, .5, .8, .95)])
    assert forecast_trust(fc, "a", good, pd.DataFrame(), 12)[0] == "Good"
    bad = _metrics([("a", 4.0, .2, .4, .6), ("seasonal_naive", 3.0, .5, .8, .95)])
    fc_dis = {"a": _frame([10] * 12), "seasonal_naive": _frame([40] * 12)}
    assert forecast_trust(fc_dis, "a", bad, pd.DataFrame(), 24)[0] == "Low"
    assert forecast_trust(fc, "a", pd.DataFrame(), pd.DataFrame(), 12)[0] == "Not tested"


def test_month_table_compares_with_typical_year_and_observed():
    hist = pd.DataFrame({"time": pd.date_range("2016-01-01", "2020-12-01", freq="MS"), "cases": 10})
    fc = {"a": _frame([10, 30, 5] + [10] * 9)}
    obs = pd.DataFrame({"time": pd.date_range("2021-01-01", periods=3, freq="MS"), "cases": [11, 100, 5]})
    summary, table, trust, notes, gloss, primary, _ = plain_forecast_html(
        forecasts=fc, metrics=pd.DataFrame(), hindcast_metrics=pd.DataFrame(), metadata={},
        disease_name="Dengue", district="IN_Pune_Maharashtra", history=hist, observed=obs)
    assert "Higher than usual" in table and "Lower than usual" in table and "About usual" in table
    assert "outside range" in table and "inside range" in table and "not yet known" in table
    assert "higher than usual in <strong>February</strong>" in summary
    assert "lower than usual in <strong>March</strong>" in summary
    assert "We can already check 3" in summary
    for jargon in ("WIS", "quantile", "hindcast", "q500", "coverage"):
        assert jargon not in summary + table + trust + notes


def test_primary_model_is_best_scoring():
    fc = {"a": _frame([1] * 3), "b": _frame([1] * 3), "ensemble": _frame([1] * 3)}
    hm = _metrics([("a", 5.0, .5, .8, .9), ("b", 2.0, .5, .8, .9), ("ensemble", 3.0, .5, .8, .9)])
    assert select_primary_model(fc, pd.DataFrame(), hm) == ("b", "past tests")


def test_pathway_sentence_is_derived_from_numbers():
    def dec(rows):
        return pd.DataFrame(rows, columns=["ssp", "decade", "change_vs_baseline_median", "change_p10", "change_p90"])
    similar = dec([("ssp245", "2030s", .12, 0, .2), ("ssp585", "2030s", .18, 0, .3),
                   ("ssp245", "2050s", .21, .05, .39), ("ssp585", "2050s", .27, .01, .85)])
    txt = _pathway_comparison(similar, ["2030s", "2050s"])
    assert txt.startswith("The pathways give similar results") and "6 percentage points" in txt
    grows = dec([("ssp245", "2030s", .10, 0, .2), ("ssp585", "2030s", .12, 0, .3),
                 ("ssp245", "2050s", .15, .05, .25), ("ssp585", "2050s", .45, .30, .70)])
    txt = _pathway_comparison(grows, ["2030s", "2050s"])
    assert txt.startswith("The gap between the pathways grows") and "overlap" not in txt


def test_forecast_report_layout_plain_first_technical_folded_single_plotly():
    fc = {"a": _frame([10] * 12), "seasonal_naive": _frame([11] * 12)}
    b = ForecastBundle(fc, {"forecast_start": "2021-01-01", "forecast_end": "2021-12-01",
                            "forecast_origin": "2020-12-31", "models": ["a", "seasonal_naive"]})
    html_ = generate_forecast_report(disease_name="Dengue", district="IN_Pune_Maharashtra", bundle=b,
                                     metrics=_metrics([("a", 2, .5, .8, .95), ("seasonal_naive", 3, .5, .8, .95)]))
    assert html_.index("The short version") < html_.index("<details class='tech'>") < html_.index("C-DSI v2")
    assert "Dengue forecast for Pune" in html_ and "January 2021 to December 2021" in html_
    assert len(re.findall(r"plotly\.js v\d", html_)) <= 1                    # library embedded once
