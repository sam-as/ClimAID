"""Keep the suite fast: tuning is compulsory in v2, so tests use the smallest preset
unless a test asks otherwise."""
import pytest


@pytest.fixture(autouse=True)
def _fast_tuning(monkeypatch):
    import climaid.forecasting_v2.tuning as tuning
    monkeypatch.setattr(tuning, "DEFAULT_TUNING", "fast")
