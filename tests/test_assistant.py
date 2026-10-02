"""The built-in assistant (`climaid ai`): understanding, guidance, docs answers, and that every
number it shows comes from ClimAID's own results."""
import re

import numpy as np
import pandas as pd
import pytest

from climaid.assistant import Assistant, Runner
from climaid.assistant import knowledge
from climaid.assistant.actions import inspect_disease_file
from climaid.assistant.understanding import match_districts, parse
from climaid.climaid_model import DiseaseModel

DISTRICTS = ["IND_Pune_MAHARASHTRA", "IND_Pune_OTHERSTATE", "NPL_Kathmandu_BAGMATI", "IND_Mumbai_MAHARASHTRA"]


def _synthetic():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2012-01-01", "2024-12-01", freq="MS")
    mo = dates.month.to_numpy()
    climate = pd.DataFrame({"time": dates,
                            "temperature": 28 + 2 * np.sin(2 * np.pi * mo / 12) + rng.normal(0, .3, len(dates)),
                            "rainfall": 5 + 4 * np.maximum(0, np.sin(2 * np.pi * (mo - 2) / 12)),
                            "humidity": 70 + 8 * np.sin(2 * np.pi * (mo - 1) / 12),
                            "enso": rng.normal(0, .5, len(dates))})
    cases = np.maximum(0, np.round(18 + 7 * np.sin(2 * np.pi * mo / 12) + rng.normal(0, 3, len(dates)))).astype(int)
    return pd.DataFrame({"date": dates, "cases": cases}), climate


@pytest.fixture
def data_file(tmp_path):
    disease, _ = _synthetic()
    path = tmp_path / "pune_dengue.csv"
    disease.to_csv(path, index=False)
    return str(path)


class RecordingRunner(Runner):
    """Runs a small real v2 forecast on synthetic data, and records what it was asked to do."""

    def __init__(self, tmp_path):
        super().__init__(log_path=str(tmp_path / "log.txt"))
        self.tmp_path, self.calls = tmp_path, []

    def load_model(self, settings):
        self.calls.append(("load", dict(settings)))
        disease, climate = _synthetic()
        dm = object.__new__(DiseaseModel)
        dm.target_col, dm.random_state = "Count", 42
        dm.disease_name, dm.district = settings["disease_name"], settings["district"]
        dm.df_disease = disease.rename(columns={"date": "time", "cases": "Count"})
        dm.df_climate_hist, dm.df_climate_proj = climate, climate
        return dm

    def forecast(self, dm, settings):
        self.calls.append(("forecast", dict(settings)))
        with self._quiet():
            self.raw = dm.forecast_v2(forecast_origin=settings.get("forecast_origin"),
                                      horizon=settings.get("horizon", 12), models=("seasonal_naive", "poisson"),
                                      n_simulations=200, hindcast_origins=2, tuning="fast",
                                      save_report=True, output_dir=str(self.tmp_path))
        return self.raw


# ------------------------------------------------------------------ understanding

def test_parse_extracts_settings_from_a_sentence():
    p = parse('Forecast malaria in Pune for the next 6 months using "my data/pune dengue.xlsx", '
              'data up to December 2023, quick tuning, keep 2020')
    assert p.intent == "forecast"
    assert p.slots == {"disease_file": "my data/pune dengue.xlsx", "disease_name": "Malaria",
                       "forecast_origin": "2023-12-31", "horizon": 6, "tuning": "fast", "exclude_period": "none"}


@pytest.mark.parametrize("text, ssps", [
    ("under very high emissions", ["ssp585"]),
    ("high emissions and very high emissions", ["ssp370", "ssp585"]),
    ("SSP2-4.5 and ssp126", ["ssp126", "ssp245"]),
    ("all scenarios", []),
])
def test_parse_emissions_pathways(text, ssps):
    assert parse("what could happen by 2050 " + text).slots["ssps"] == ssps


@pytest.mark.parametrize("text, slot", [
    ('my climate file is "w.csv"', "weather_file"), ("weather data: w.csv", "weather_file"),
    ("my climate projection file is p.csv", "projection_file"), ("projections data p.parquet", "projection_file"),
    ("cases are in d.xlsx", "disease_file"),
])
def test_parse_which_file(text, slot):
    assert set(parse(text).slots) == {slot}


def test_parse_outlook_year_and_custom_exclusion():
    p = parse("project to 2080 and exclude 2020-03 to 2021-06")
    assert p.intent == "project" and p.slots["end_year"] == 2080
    assert p.slots["exclude_period"] == "2020-03:2021-06"


def test_district_matching():
    assert match_districts("Kathmandu", DISTRICTS) == ["NPL_Kathmandu_BAGMATI"]
    assert match_districts("kath", DISTRICTS) == ["NPL_Kathmandu_BAGMATI"]          # partial
    assert match_districts("Kathmandoo", DISTRICTS) == ["NPL_Kathmandu_BAGMATI"]    # misspelt
    assert set(match_districts("pune", DISTRICTS)) == {"IND_Pune_MAHARASHTRA", "IND_Pune_OTHERSTATE"}
    assert match_districts("pune maharashtra", DISTRICTS) == ["IND_Pune_MAHARASHTRA"]
    assert match_districts("banana", DISTRICTS) == []


# ------------------------------------------------------------------ data checks

def test_inspect_disease_file(data_file, tmp_path):
    info = inspect_disease_file(data_file)
    assert info["ok"] and info["frequency"] == "monthly" and info["rows"] == 156
    assert "January 2012 to December 2024" in info["summary"]
    assert not inspect_disease_file(str(tmp_path / "missing.csv"))["ok"]
    pd.DataFrame({"when": ["2020-01-01"] * 30, "n": range(30)}).to_csv(tmp_path / "bad.csv", index=False)
    bad = inspect_disease_file(str(tmp_path / "bad.csv"))
    assert not bad["ok"] and any("date column" in x for x in bad["problems"])


# ------------------------------------------------------------------ docs

@pytest.mark.parametrize("question, expected", [
    ("what is WIS?", "weighted interval score"),
    ("why is 2020 excluded?", "COVID-19"),
    ("how does the forecast work?", "backtests"),
    ("how many trials do I need?", "not reliably more accurate"),
    ("is ClimAID validated on real data?", "not yet validated"),
    ("what format should my data be in?", "date column"),
])
def test_docs_answers(question, expected):
    assert expected in knowledge.answer(question)


def test_docs_search_uses_the_bundled_documentation():
    hits = knowledge.search("thermal suitability curve temperature")
    assert hits and hits[0]["location"].startswith("guide/")
    assert knowledge.answer("purple elephants dancing") is None


# ------------------------------------------------------------------ the conversation

def test_guided_forecast_asks_for_what_is_missing_then_runs(data_file, tmp_path):
    runner = RecordingRunner(tmp_path)
    a = Assistant(runner=runner, districts=DISTRICTS)
    assert "Which one?" in a.ask("I want a dengue forecast in Pune for the next 6 months")
    reply = a.ask("1")
    assert "IND_Pune_MAHARASHTRA" in reply and "data file" in reply.lower()     # asks for the file next
    reply = a.ask(data_file)
    assert "looks good" in reply and "Shall I run it?" in reply
    assert runner.calls == []                                                   # nothing runs before "yes"
    reply = a.ask("use data up to December 2023, quick tuning")
    assert "December 2023" in reply and "Fast" in reply
    reply = a.ask("yes")

    kinds = [c[0] for c in runner.calls]
    assert kinds == ["load", "forecast"]
    settings = runner.calls[1][1]
    assert settings["district"] == "IND_Pune_MAHARASHTRA" and settings["horizon"] == 6
    assert settings["forecast_origin"] == "2023-12-31" and settings["tuning"] == "fast"
    assert "Trust rating:" in reply and "under active testing" in reply

    # every number in the month-by-month table comes from the forecast itself
    primary = a.results["forecast"]["primary"]
    f = runner.raw["forecasts"][primary]
    for t, q500 in zip(pd.to_datetime(f["time"]), f["q500"]):
        assert f"{t:%b %Y}" in reply and f"{q500:,.0f}" in reply

    assert a.results["forecast"]["rating"] in a.ask("how reliable is this?")
    how = a.ask("how was this forecast made?")          # the run's own method, from its metadata
    assert "How this forecast was made" in how and "December 2023" in how and "Backtests from 2" in how
    assert "file://" in a.ask("open the report")


def test_answers_questions_mid_flow_without_losing_settings(data_file, tmp_path):
    a = Assistant(runner=RecordingRunner(tmp_path), districts=DISTRICTS)
    a.ask(f"forecast dengue in Kathmandu using {data_file}")
    assert "Shall I run it?" in a.ask("12 months")
    assert "weighted interval score" in a.ask("what is WIS?")
    assert a.settings["district"] == "NPL_Kathmandu_BAGMATI" and a.awaiting == "confirm"
    assert "not running it" in a.ask("no").lower()


def test_outlook_plan_and_adding_scenarios(data_file, tmp_path):
    a = Assistant(runner=RecordingRunner(tmp_path), districts=DISTRICTS)
    reply = a.ask(f"What could happen to malaria in Kathmandu by 2050 under very high emissions? Data: {data_file}")
    assert "climate-change outlook for Malaria" in reply and "SSP5-8.5" in reply
    reply = a.ask("can you also include middle emissions")
    assert "SSP2-4.5" in reply and "SSP5-8.5" in reply


def test_run_errors_are_explained(data_file, tmp_path):
    class Offline(RecordingRunner):
        def load_model(self, settings):
            raise ConnectionError("Max retries exceeded")

    a = Assistant(runner=Offline(tmp_path), districts=DISTRICTS)
    a.ask(f"forecast dengue in Kathmandu using {data_file}")
    reply = a.ask("yes")
    assert "internet connection once" in reply


def test_methods_menu_topics_and_more_detail():
    a = Assistant(districts=DISTRICTS)
    menu = a.ask("methods")
    assert "1. How ClimAID works" in menu and a.awaiting == "topic_choice"
    from climaid.assistant.methods import TOPICS
    n = next(i for i, t in enumerate(TOPICS, 1) if t.key == "calibration")
    assert a.ask(str(n)).startswith("Calibrated likely ranges:")
    assert "split-conformal" in a.ask("more detail").lower()
    assert a.ask("explain lag selection").startswith("Lag structure in the outlook")     # most specific topic wins
    assert a.ask("how does the renewal model work?").startswith("The climate transmission (renewal) model")
    assert "generation interval" in a.ask("tell me more")
    assert a.ask("how are climate models bias corrected?").startswith("Bias correction")
    assert "haven't run anything" in a.ask("how was this forecast made?")


METHOD_QUESTIONS = {
    "overview": "how does climaid work?", "data": "how does data cleaning work?", "covid": "why is 2020 excluded?",
    "features": "what inputs do the models use?", "distributed_lags": "what are distributed lags?",
    "models": "which models are used?", "baseline": "what is the seasonal naive baseline?",
    "renewal": "how does the renewal model work?", "sarimax": "what is SARIMAX?", "residuals": "what is the ensemble?",
    "tuning": "how does tuning work?", "hindcasts": "what are hindcasts?", "calibration": "how are the intervals calibrated?",
    "scoring": "what is WIS?", "trust": "how is the trust rating calculated?", "leakage": "how do you avoid leakage?",
    "scenarios": "how is the scenario outlook built?", "bias_correction": "how does bias correction work?",
    "response": "what is the anomaly response?", "lag_selection": "how does lag selection work?",
    "pooling": "what does pooling districts do?", "thermal": "what is the thermal curve?",
    "v1": "what is the legacy v1 pipeline?", "limitations": "what are the limitations?",
}


def test_every_methods_topic_is_reachable_and_linked():
    from climaid.assistant import methods
    from climaid.assistant.knowledge import DOCS_DIR
    assert set(METHOD_QUESTIONS) == {t.key for t in methods.TOPICS}
    for key, question in METHOD_QUESTIONS.items():
        found = methods.find_topic(question)
        assert found is not None and found.key == key, (question, found and found.key)
    for t in methods.TOPICS:
        if t.page:
            assert (DOCS_DIR / t.page.split("#")[0] / "index.html").exists(), t.page


def test_no_future_climate_is_explained(data_file, tmp_path):
    a = Assistant(runner=RecordingRunner(tmp_path), districts=DISTRICTS)
    a.ask(f"forecast dengue in Kathmandu using {data_file}")       # data end when the climate data end
    assert "earlier forecast start" in a.ask("yes")


def test_unknown_input_and_trust_before_any_run():
    a = Assistant(districts=DISTRICTS)
    assert "didn't understand" in a.ask("banana")
    assert "haven't run anything" in a.ask("how reliable is this?")
    assert a.ask("help").startswith("Things you can say")
    a.ask("quit")
    assert a.finished


def test_cli_has_the_ai_command(monkeypatch):
    from typer.testing import CliRunner
    from climaid.cli import app
    import climaid.assistant.core as core
    monkeypatch.setattr(core.Assistant, "districts", property(lambda self: DISTRICTS))
    result = CliRunner().invoke(app, ["ai"], input="what is WIS?\nquit\n")
    assert result.exit_code == 0
    assert "ClimAID assistant" in result.output and "weighted interval score" in result.output
