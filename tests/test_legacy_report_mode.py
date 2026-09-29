"""Regression test for a bug found in climaid/browser_ui/api.py: selecting
"detailed"/"summary"/"policy" as the legacy report mode silently produced
byte-identical output to "deterministic" (because llm_client is hardcoded to
None and DiseaseReporter.generate() only varies by style when an LLM client
is supplied), while the UI labelled it as if that requested narrative style
had actually been generated. Fixed to fall back explicitly and say so,
rather than mislabelling the result.

Uses a lightweight fake model (duck-typing DiseaseModel's public surface)
instead of driving the real ML pipeline, since _run_legacy()'s only job
here is orchestration/labelling -- exercising the real optimize_lags()
default search space would take minutes and adds nothing to this test.
"""
from climaid.browser_ui.api import WizardConfig, _run_legacy


class _FakeDiseaseModel:
    disease_name = "Dengue"

    def optimize_lags(self, **kwargs):
        return None

    def train_final_model(self):
        return {"predictions": None}

    def generate_report(self, **kwargs):
        # Record the style actually requested of the reporter, so the test
        # can confirm _run_legacy() always asks for the deterministic
        # engine regardless of what the UI's dropdown selected.
        self.last_style_requested = kwargs.get("style")
        return "<mock deterministic report text>"

    def build_report_artifacts(self, *a, **k):
        return object()


def _cfg(**overrides):
    base = dict(
        mode="southasia", country="IN", district="TestDistrict", state="TestState",
        disease_name="Dengue", preset="fast", run_legacy=True, run_cmip6=False,
    )
    base.update(overrides)
    return WizardConfig(**base)


def test_non_deterministic_legacy_style_falls_back_with_an_explicit_note():
    dm = _FakeDiseaseModel()
    cfg = _cfg(legacy_report_mode="policy")

    result = _run_legacy(dm, cfg)

    assert result["note"] is not None
    assert "policy" in result["note"].lower()
    assert "LLM" in result["note"]
    # The reporter itself must always be asked for the deterministic
    # engine here -- never silently told it produced "policy"/"summary"/
    # "detailed" when no LLM client exists to actually do that.
    assert dm.last_style_requested == "_deterministic_engine"


def test_deterministic_legacy_style_has_no_fallback_note():
    dm = _FakeDiseaseModel()
    cfg = _cfg(legacy_report_mode="deterministic")

    result = _run_legacy(dm, cfg)

    assert result["note"] is None
    assert dm.last_style_requested == "_deterministic_engine"


def test_all_non_deterministic_modes_produce_a_note():
    for mode in ("detailed", "summary", "policy"):
        dm = _FakeDiseaseModel()
        result = _run_legacy(dm, _cfg(legacy_report_mode=mode))
        assert result["note"] is not None, f"expected a fallback note for mode={mode}"
