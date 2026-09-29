"""Regression test for a bug found during review: DiseaseReporter._llm_generate
returned a tuple (not a str) whenever the LLM client raised, even though every
caller and every downstream consumer (open_report_in_browser -> html.escape)
requires a plain string. That meant the LLM-unavailable fallback path -- the
one thing it exists to handle safely -- crashed instead.
"""
from climaid.reporting import DiseaseReporter, ReportArtifacts


class _BrokenLLM:
    def generate(self, prompt):
        raise RuntimeError("local LLM offline")


def _artifacts():
    return ReportArtifacts(
        district="IN_TestDistrict_TestState",
        disease_name="Dengue",
        date_range="2010-2020",
        metrics={},
        selected_lags={},
        interaction_lags=[],
        features=[],
        importance={},
        projection_summary={},
    )


def test_llm_fallback_returns_a_string_not_a_tuple():
    reporter = DiseaseReporter(llm_client=_BrokenLLM())
    out = reporter.generate(_artifacts(), style="summary")
    assert isinstance(out, str)
    assert "C-DSI" in out


def test_llm_fallback_is_html_escapable():
    # This is exactly what open_report_in_browser() does to the returned
    # report text; it must not raise.
    import html as html_mod
    reporter = DiseaseReporter(llm_client=_BrokenLLM())
    out = reporter.generate(_artifacts(), style="summary")
    html_mod.escape(out)
