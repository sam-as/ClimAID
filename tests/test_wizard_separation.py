"""End-to-end test for wizard.py's v1/v2 pipeline separation.

Previously, climaid.wizard.run_interactive_pipeline() interleaved v2's
probabilistic-forecasting prompt in the middle of the v1 flow (train/test
split -> outbreak detection -> v2 block -> lag optimization -> ... -> final
report), with no way to run one pipeline without walking through the other's
prompts. It now asks upfront which pipeline(s) to run and only asks each
pipeline's own questions when that pipeline was selected.

This drives the actual wizard function with scripted input() answers and a
fake DiseaseModel (to avoid touching disk/network or running real ML), and
proves separation two ways:
  1. Selecting "v2 only" never calls any v1-only DiseaseModel method
     (_train_test_split, detect_historical_outbreaks, optimize_lags,
     train_final_model) and never asks any v1-specific question.
  2. Selecting "v1 only" never calls forecast_v2 and never asks the v2
     question at all.
A mismatch between the scripted answers and what the wizard actually asks
raises immediately (StopIteration from the exhausted answer queue), so an
unexpected extra prompt (e.g. a v1 question leaking into a v2-only run)
fails the test loudly rather than silently.
"""
import builtins
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import climaid.wizard as wizard_mod


class _FakeDiseaseModel:
    """Duck-types just enough of DiseaseModel's public surface for the
    wizard to run through, recording which methods were actually called."""

    def __init__(self, district, disease_file, random_state=42, disease_name=None,
                 weather_file=None, projection_file=None):
        self.district = district
        self.disease_name = disease_name
        self.calls = []
        dates = pd.date_range("2010-01-01", periods=48, freq="MS")
        self.df_merged = pd.DataFrame({"time": dates, "Count": range(48), "Year": dates.year})
        self.df_disease = pd.DataFrame({"time": dates, "Count": range(48), "Year": dates.year})
        self.df_climate_hist = pd.DataFrame({"time": dates})
        self.df_climate_proj = pd.DataFrame({"time": dates})
        self.runtime = {}

    def _train_test_split(self, *a, **k):
        self.calls.append("_train_test_split")

    def detect_historical_outbreaks(self, *a, **k):
        self.calls.append("detect_historical_outbreaks")
        return pd.DataFrame()

    def optimize_lags(self, *a, **k):
        self.calls.append("optimize_lags")
        return {}, pd.DataFrame(), {}

    def train_final_model(self):
        self.calls.append("train_final_model")
        return {"test_r2": 0.0, "test_rmse": 0.0}

    def plot_historical_predictions(self, *a, **k):
        self.calls.append("plot_historical_predictions")

    def generate_report(self, *a, **k):
        self.calls.append("generate_report")
        return "<mock report>"

    def print_runtime_summary(self):
        self.calls.append("print_runtime_summary")

    def forecast_v2(self, *a, **k):
        self.calls.append("forecast_v2")
        return {
            "metrics": pd.DataFrame(),
            "hindcast_metrics": pd.DataFrame(),
            "report_path": None,
        }


class _ScriptedInput:
    """Feeds a fixed queue of answers to input(); raises loudly (rather than
    hanging or silently reusing an answer) if the wizard asks for more
    input than was scripted -- e.g. because a prompt from the "wrong"
    pipeline leaked through."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, prompt=""):
        self.asked.append(prompt)
        if not self.answers:
            raise AssertionError(
                f"Wizard asked for more input than scripted (prompt: {prompt!r}). "
                f"Prompts asked so far: {self.asked}"
            )
        return self.answers.pop(0)


@pytest.fixture
def fake_district_tree(monkeypatch):
    monkeypatch.setattr(wizard_mod, "get_available_districts", lambda: ["IN_testdistrict_teststate"])


@pytest.fixture
def disease_csv(tmp_path):
    path = tmp_path / "disease.csv"
    dates = pd.date_range("2010-01-01", periods=48, freq="MS")
    pd.DataFrame({"Date": dates, "Cases": range(48)}).to_csv(path, index=False)
    return str(path)


def _run_wizard(monkeypatch, answers, disease_csv):
    fake_dm_holder = {}

    def fake_disease_model(*args, **kwargs):
        dm = _FakeDiseaseModel(*args, **kwargs)
        fake_dm_holder["dm"] = dm
        return dm

    monkeypatch.setattr(wizard_mod, "DiseaseModel", fake_disease_model)
    monkeypatch.setattr(builtins, "input", _ScriptedInput(answers))

    wizard_mod.run_interactive_pipeline()
    return fake_dm_holder["dm"]


def test_v2_only_never_touches_v1_methods_or_prompts(monkeypatch, fake_district_tree, disease_csv):
    answers = [
        "1",            # country
        "1",            # state
        "1",            # district
        "Dengue",       # disease name
        disease_csv,    # disease file path
        "2",            # pipeline choice: v2 only
        "",             # COVID-19 period (default 1 = 2020)
        "y",            # "Run ClimAID v2 probabilistic forecasting?"
        "",             # model tuning effort (default balanced)
        "",             # v2 models (default)
        "",             # first forecast year (default)
        "",             # forecast horizon (default)
        "",             # simulations (default)
        "",             # population at risk (none)
        "n",            # run historical hindcasts? -- no, avoids needing hind_origins input
    ]
    dm = _run_wizard(monkeypatch, answers, disease_csv)

    assert dm.calls == ["forecast_v2"]
    for v1_only_method in ("_train_test_split", "detect_historical_outbreaks", "optimize_lags", "train_final_model"):
        assert v1_only_method not in dm.calls


def test_v1_only_never_calls_forecast_v2_or_asks_its_question(monkeypatch, fake_district_tree, disease_csv):
    """This also doubles as a regression test for two unrelated bugs this
    exact path used to hit: `plt` and `use_headless_backend` were each only
    imported inside conditional prompts nested under "Run lag optimization
    (Optuna search) using AutoML?", but both were called unconditionally a
    few steps later in the LLM-report section. Declining lag optimization
    (as this test does) used to raise NameError/UnboundLocalError there."""
    answers = [
        "1",            # country
        "1",            # state
        "1",            # district
        "Dengue",       # disease name
        disease_csv,    # disease file path
        "1",            # pipeline choice: v1 only
        "",             # COVID-19 period (default 1 = 2020)
        "n",            # "Configure train/test split?" -> no (use default, no sub-prompts)
        "n",            # "Detect historical outbreak signals?" -> no
        "n",            # "Run lag optimization (Optuna search) using AutoML?" -> no (skips training/plots/CMIP6 entirely)
        "n",            # "Print runtime summary?" -> no
        # (v2's "Run ClimAID v2 probabilistic forecasting?" must NOT be
        # asked at all -- if it were, this scripted queue would be
        # exhausted and _ScriptedInput would raise.)
    ]
    dm = _run_wizard(monkeypatch, answers, disease_csv)

    assert "forecast_v2" not in dm.calls
