"""The documentation ships inside the package and is reachable from the dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "climaid" / "documentation"


def test_docs_are_bundled_with_the_tuning_page():
    assert (DOCS / "index.html").exists(), "built documentation missing from climaid/documentation"
    assert (DOCS / "guide" / "tuning" / "index.html").exists()
    assert (DOCS / "guide" / "status" / "index.html").exists()


def test_package_data_includes_the_documentation():
    toml = (ROOT / "pyproject.toml").read_text()
    assert '"documentation/*"' in toml and '"documentation/*/*/*/*"' in toml


def test_dashboard_serves_documentation():
    from fastapi.testclient import TestClient
    from climaid.browser_ui.server import app
    c = TestClient(app)
    assert c.get("/documentation/").status_code == 200
    r = c.get("/documentation/guide/tuning/")
    assert r.status_code == 200 and "trials" in r.text.lower()
    assert c.get("/docs").status_code == 200            # FastAPI's own API page still works


def test_trial_settings_link_to_the_tuning_page():
    html = (ROOT / "climaid" / "browser_ui" / "static" / "index.html").read_text()
    links = html.count('href="/documentation/guide/tuning/"')
    assert links >= 3
    for anchor in ('id="v2_tuning"', 'id="preset"', 'id="trials"'):
        i = html.index(anchor)
        assert 'href="/documentation/guide/tuning/"' in html[i: i + 600], anchor


def test_cli_has_docs_command():
    from typer.testing import CliRunner
    from climaid.cli import app
    r = CliRunner().invoke(app, ["--help"])
    assert "docs" in r.output


def test_package_data_includes_the_built_in_climate_data():
    toml = (ROOT / "pyproject.toml").read_text()
    assert '"data/*.csv"' in toml, "built-in South Asia climate data would be missing from the wheel"
    assert (ROOT / "climaid" / "data" / "SouthAsia_weather_data.csv").exists()
