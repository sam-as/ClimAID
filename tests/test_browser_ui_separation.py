"""The dashboard shows ClimAID v2 and v1 as separate pages (v2 by default),
each with its own settings and Run button, and every control has an info tip."""
import re
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[1] / "climaid" / "browser_ui" / "static"


def _read(name):
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _section(html, sec_id):
    start = html.index(f'<section id="{sec_id}"')
    return html[start: html.index("</section>", start)]


def test_v2_is_the_default_page_and_v1_is_separate():
    html = _read("index.html")
    v2, v1 = _section(html, "climaid-v2"), _section(html, "legacy-v1-controls")
    assert 'data-page="v2"' in v2 and "hidden" not in v2.split(">", 1)[0]
    assert 'data-page="v1"' in v1 and "hidden" in v1.split(">", 1)[0]
    assert 'id="tab-v2" class="page-tab active"' in html
    # Each page owns its own configuration and Run button.
    assert 'id="model-selection"' in v1 and 'id="model-selection"' not in v2
    assert 'id="test_year"' in v1
    assert "runClimAIDV1Only" in v1 and "runClimAIDV2Only" in v2
    # Redundant cross-pipeline toggles are gone.
    for gone in ('id="run_legacy"', 'id="run_v2"', "Run Both Pipelines"):
        assert gone not in html


def test_covid_period_is_shared_defaults_to_2020_and_has_no_2021_preset():
    html = _read("index.html")
    shared = html[: html.index('<nav class="page-tabs"')]
    sel = shared[shared.index('id="exclude_period_mode"'):]
    sel = sel[: sel.index("</select>")]
    assert '<option value="2020" selected>' in sel
    assert 'value="custom"' in sel and 'value="none"' in sel and "2021" not in sel
    assert 'id="exclude_start"' in shared and 'id="exclude_end"' in shared
    js = _read("browser.js")
    assert "exclude_period:excludePeriodValue()" in js and "function toggleCustomPeriod" in js

def test_every_control_has_an_info_tip():
    html = _read("index.html")
    bare = re.findall(r"<label>[^<]*</label>|<h3>[^<]*</h3>", html)
    assert bare == [], bare
    assert html.count('class="info"') >= 25


def test_browser_js_supports_pages_2020_and_model_tips():
    js = _read("browser.js")
    for needle in ("function showPage", "function runClimAIDV1Only", "function runClimAIDV2Only",
                   "exclude_period:", "function modelTip", "MODEL_TIPS", "data.defaults"):
        assert needle in js, needle


def test_scenario_outlook_controls_live_on_the_v2_page():
    html = _read("index.html")
    v2 = _section(html, "climaid-v2")
    for i in ("run_scenarios", "scenario_end_year", "scenario_ssps", "scenario_response", "scenario_lag_selection"):
        assert f'id="{i}"' in v2, i
    assert "run_scenarios:" in _read("browser.js")
