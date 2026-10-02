"""Answers from ClimAID's own documentation, offline.

The bundled documentation (``climaid/documentation``) ships a search index with one entry per
section. The assistant ranks those sections with BM25 and quotes the best one, with a link to
the local page and the website, so answers always come from the documentation itself.
"""
from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from climaid import __version__

DOCS_DIR = Path(__file__).resolve().parents[1] / "documentation"
INDEX_FILE = DOCS_DIR / "search" / "search_index.json"
SITE_URL = "https://sam-as.github.io/ClimAID/"

STOP = set("""a an the and or of to in on for with by is are was were be been it its this that these those
what which who how why when where do does did can could should would will i you we they he she my your our
me us them as at from about into than then so if not no yes there here have has had just also more most
some any each other such only own same very s t don use using used please tell show give""".split())

# Short, plain definitions for terms people often ask about. Taken from the report glossaries and
# the documentation; the docs search adds the detailed section.
GLOSSARY = {
    "wis": "WIS (weighted interval score) measures how good a forecast with ranges is: it rewards ranges that "
           "are narrow but still contain the real number. Lower is better. Reports compare it with the "
           "'same as recent years' baseline: below 1.0 means better than the baseline.",
    "likely range": "The range the real number should fall inside 8 times out of 10. Wider ranges mean more "
                    "uncertainty.",
    "hindcast": "A backtest: ClimAID pretends to be at an earlier date, makes a forecast with only the data "
                "available then, and checks it against what really happened.",
    "backtest": "A test on past data: forecasts are made from earlier dates and compared with what happened.",
    "trust rating": "Good / Moderate / Low, one point each for: the likely range contained the real number about "
                    "as often as it should; the method beat the 'same as recent years' baseline in past tests; "
                    "the methods broadly agree; and the forecast looks no more than 12 months ahead. 4 points = "
                    "Good, 2-3 = Moderate, 0-1 = Low.",
    "ssp": "An emissions pathway (Shared Socioeconomic Pathway): a standard scenario for future greenhouse-gas "
           "emissions, e.g. SSP2-4.5 (middle of the road) or SSP5-8.5 (very high emissions).",
    "ensemble": "A combination of all the models, taking their middle value.",
    "seasonal naive": "The 'same as recent years' baseline: each month is forecast from the same month in "
                      "recent years. Every model is compared against it.",
    "baseline": "In forecasts, the 'same as recent years' method every model must beat. In scenario outlooks, "
                "the average number of cases per year in the historical period.",
    "renewal": "ClimAID's climate-informed transmission model: new cases arise from recent cases, scaled by how "
               "favourable the climate is for transmission.",
    "tuning": "Automatic adjustment of each model's settings before it is used, tested only on data before "
              "the forecast start. Fast = 10, Balanced = 30, Deep = 80 trials per model.",
    "cmip6": "The current generation of global climate-model simulations, used for ClimAID's scenario outlooks.",
    "lag": "A delay between climate and cases: for example, rain may affect dengue cases 1-2 months later.",
}
ALIASES = {"weighted interval score": "wis", "interval score": "wis", "range": "likely range",
           "hindcasts": "hindcast", "backtests": "backtest", "rating": "trust rating", "ssps": "ssp",
           "emissions pathway": "ssp", "seasonal-naive": "seasonal naive", "seasonal_naive": "seasonal naive",
           "lags": "lag", "renewal model": "renewal", "optimisation trials": "tuning", "trials": "tuning"}


# Words people use -> words the documentation uses.
EXPAND = {"covid": ["2020", "exclude", "period", "disruption"], "2020": ["covid", "exclude", "period"],
          "pandemic": ["covid", "2020"], "multiple": ["extra", "pooled", "pooling"],
          "several": ["extra", "pooled", "pooling"], "trials": ["tuning", "optimisation"],
          "reliable": ["trust", "rating"], "trustworthy": ["trust", "rating"], "accuracy": ["benchmark", "wis"],
          "validated": ["validation", "status", "tested"], "uncertainty": ["range", "likely", "calibrated"],
          "lags": ["lag", "lag_selection"], "future": ["scenario", "outlook", "projection"]}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


# Curated answers to the most common questions, written from the documentation. Each entry:
# (patterns that must all match, answer, documentation page).
FAQ = [
    ((r"how", r"forecast"), "A v2 forecast: (1) checks your data; (2) leaves out the COVID-19 period (2020 by "
     "default) from the training history; (3) fits each model on data up to the forecast start only, tuning every "
     "machine-learning model automatically; (4) repeats the whole process from several earlier start dates "
     "(backtests) and checks those forecasts against what happened; (5) widens or narrows the likely ranges so they "
     "would have contained the real number as often as they claim; (6) writes a plain-language report with a trust "
     "rating.", "guide/v2_forecasting/#what-happens-when-you-run-a-forecast"),
    ((r"(outlook|scenario|projection|climate change)", r"(how|work|built|made)"),
     "A climate outlook: each climate model is bias-corrected to your district's observed climate; the next 12 "
     "months come from the v2 forecast; months 13-24 hand over gradually to a model that links cases to climate "
     "alone, refitted many times on resampled years; results are pooled across climate models, so the ranges include "
     "climate-model spread. A backtest and a sensitivity run are reported alongside.",
     "guide/scenarios/#how-the-outlook-is-built"),
    ((r"(2020|covid)",), "Case counts during COVID-19 were distorted by less testing, reporting and care-seeking. By "
     "default each 2020 month is replaced in the training history with that month's typical value from other years, "
     "and is never used to score backtests. You can keep it (\"keep 2020\") or choose another period "
     "(\"exclude 2020-03 to 2020-12\").", "guide/v2_forecasting/#what-happens-when-you-run-a-forecast"),
    ((r"(which|what|list).*models|models.*(available|are there|used)",), "ClimAID v2 has 22 models: the seasonal "
     "naive baseline, a climate renewal (transmission) model, SARIMAX (seasonal ARIMA with climate), regression models (linear, ridge, lasso, elastic net, "
     "Poisson, Tweedie, smooth-curve Poisson, Bayesian ridge, Huber), tree ensembles (random forest, extra trees, "
     "gradient boosting, hist gradient boosting, XGBoost, LightGBM, CatBoost), a neural network, SVR, nearest "
     "neighbours, and v1's stacked model. The defaults are seasonal naive, renewal, Poisson, random forest, extra "
     "trees and gradient boosting, plus their ensemble.", "guide/v2_forecasting/#models-22"),
    ((r"(how long|time|slow|fast|minutes|hours)", r"(take|run|forecast|outlook|wait)"),
     "Mostly set by tuning, because every machine-learning model is tuned and the backtests repeat the fitting at 4 "
     "earlier start dates. On one CPU core a forecast with the default models takes a few minutes with Fast tuning "
     "(10 trials per model) and can take well over ten minutes with Balanced (30, the default); Deep (80) is several "
     "times slower again and rarely more accurate. Climate outlooks take longer than forecasts.", "guide/tuning/"),
    ((r"(how many|number of|enough|more).*trials|trials.*(need|enough)",), "More trials are not reliably more "
     "accurate: on the benchmark, v2 stopped improving at about 10-30 trials per model and was sometimes worse at 80. "
     "Balanced (30) is a good default; Fast (10) for a first look.", "guide/tuning/"),
    ((r"(format|columns?|prepare|structure|what).*(data|file)|(data|file).*(format|columns?|need)",),
     "A CSV or Excel file with a date column (named date or time) and a case-count column (named cases or count), "
     "at least two years long. Monthly data are best; weekly or daily data are summed to months. Climate data for "
     "South Asian districts are built in; for other places provide your own climate file.",
     "api/resources/#sample-datasets"),
    ((r"(other|outside|my own|different) (countr|place|region)|not in south asia|(africa|europe|america|global)",),
     "Built-in climate data cover South Asia (India, Nepal, Bhutan, Sri Lanka, Myanmar, Afghanistan, Pakistan, "
     "Bangladesh). For other places, give me your own climate file (\"my climate file is weather.csv\", with a "
     "Dist_States column naming the district) and, for outlooks, a projection file; or use the global mode of the "
     "browser interface (climaid browse).", "api/climate/"),
    ((r"(several|multiple|more|other|extra) districts|pool",), "Pooling several districts (extra_districts in "
     "project_v2) learns one shared climate response and, on synthetic data, recovered the true warming effect within "
     "3 percentage points, where a single district understated it by about half. This assistant runs one district at "
     "a time; pooling is available from Python (DiseaseModel.project_v2(extra_districts=[...])).",
     "guide/scenarios/#known-limitations-read-before-using"),
    ((r"(validated|tested|real data|accurate|accuracy|reliable|trust climaid)",), f"ClimAID {__version__} is under active "
     "testing: methods have been checked on synthetic data with a known answer, but not yet validated on real "
     "surveillance data or real CMIP6 projections. On the synthetic benchmark the v2 ensemble was about a third "
     "better than the 'same as recent years' baseline on seasonal data. Every report includes a trust rating for "
     "that particular run.", "guide/status/"),
    ((r"v1", r"v2"), "v2 (recommended) gives forecasts with likely ranges, 22 tuned models, backtests and calibrated "
     "ranges, plus climate outlooks. v1 (legacy) gives lag-optimised point predictions from a stacked model, "
     "checked on one test year. This assistant runs v2.", ""),
]


def faq(question: str):
    q = question.lower()
    for patterns, text, page in FAQ:
        if all(re.search(pt, q) for pt in patterns):
            return text, page
    return None


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def _clean(fragment: str) -> str:
    text = re.sub(r"<pre>.*?</pre>", " ", fragment, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


@lru_cache(maxsize=1)
def _index():
    try:
        docs = json.loads(INDEX_FILE.read_text(encoding="utf-8"))["docs"]
    except (OSError, ValueError, KeyError):
        return [], {}, 0.0
    sections = []
    for d in docs:
        title, text = _clean(d.get("title", "")), _clean(d.get("text", ""))
        if not text or d["location"].startswith("api/") and len(text) < 40:
            continue
        sections.append({"title": title, "text": text, "location": d["location"],
                         "tokens": _tokens(title) * 3 + _tokens(text)})
    df = Counter(t for s in sections for t in set(s["tokens"]))
    avg = sum(len(s["tokens"]) for s in sections) / max(1, len(sections))
    return sections, df, avg


def search(query: str, k: int = 3) -> list[dict]:
    """Best documentation sections for `query` (BM25)."""
    sections, df, avg = _index()
    q = _tokens(query)
    q += [_stem(w) for t in list(q) for w in EXPAND.get(t, [])]
    if not sections or not q:
        return []
    n, k1, b = len(sections), 1.5, 0.75
    scored = []
    for s in sections:
        tf = Counter(s["tokens"])
        score = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(s["tokens"]) / avg))
        if s["location"].startswith("api/"):
            score *= 0.6          # prefer the user guide over API reference pages
        elif s["location"].startswith("changelog"):
            score *= 0.5          # the changelog lists changes; the guide explains them
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda t: -t[0])
    return [dict(s, score=sc) for sc, s in scored[:k]]


def glossary_lookup(text: str) -> tuple[str, str] | None:
    t = re.sub(r"[^a-z0-9 \-_]", " ", text.lower())
    t = re.sub(r"\s+", " ", t)
    for term in sorted(list(GLOSSARY) + list(ALIASES), key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", t):
            key = ALIASES.get(term, term)
            return key, GLOSSARY[key]
    return None


def links(location: str, local_base: str | None = None) -> tuple[str, str]:
    """(offline link, website URL) for a search-index location such as 'guide/status/#known-limitations'.

    The offline link is a file:// URI to the bundled page, or `local_base` + location when the bundled
    documentation is served over HTTP (the dashboard serves it at /documentation/).
    """
    page, _, anchor = location.partition("#")
    local = DOCS_DIR / page / "index.html" if page else DOCS_DIR / "index.html"
    url = SITE_URL + location
    if not local.exists():
        return "", url
    if local_base is not None:
        return local_base + location, url
    return local.as_uri() + (f"#{anchor}" if anchor else ""), url


def excerpt(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > 200 else cut.rstrip() + "…")


def answer(question: str, local_base: str | None = None) -> str | None:
    """A documentation-based answer, or None if nothing relevant was found."""
    hit = faq(question)
    if hit:
        text, page = hit
        if not page:
            return text
        return text + f"\n\nMore: {(local_base if local_base is not None else SITE_URL) + page}"
    parts = []
    g = glossary_lookup(question)
    hits = search(question, k=3)
    if g:
        parts.append(f"{g[0].upper() if len(g[0]) <= 5 else g[0].capitalize()}: {g[1]}")
        if hits and hits[0]["score"] >= 2.0:
            base = local_base if local_base is not None else SITE_URL
            parts.append(f"More in the documentation: \"{hits[0]['title']}\", {base}{hits[0]['location']}")
        return "\n\n".join(parts)
    if hits and hits[0]["score"] >= 2.0:
        top = hits[0]
        local, url = links(top["location"], local_base)
        if local_base is not None and local:      # served documentation: one link is enough
            parts.append(f"From the documentation, \"{top['title']}\":\n{excerpt(top['text'])}\nRead more: {local}")
        else:
            parts.append(f"From the documentation, \"{top['title']}\":\n{excerpt(top['text'])}\n"
                         f"Read more: {url}" + (f"\n(offline copy: {local})" if local else ""))
        more = [h for h in hits[1:] if h["score"] >= 0.5 * top["score"]]
        if more:
            base = local_base if local_base is not None else SITE_URL
            parts.append("Related: " + "; ".join(f"{h['title']} ({base}{h['location']})" for h in more))
    return "\n\n".join(parts) if parts else None
