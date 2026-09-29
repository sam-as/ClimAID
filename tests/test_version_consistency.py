"""The version is defined once and every piece of documentation agrees with it.

`climaid/__init__.py` is the single source of the version. These tests fail when a release
leaves any documentation naming another version, when the bundled documentation was not
rebuilt, or when a file was saved with a byte-order mark or garbled (mis-encoded) characters.
See README_DOCS.md, "Releasing a new version".
"""
import re
from pathlib import Path

import pytest

import climaid

ROOT = Path(__file__).resolve().parents[1]
VERSION = climaid.__version__
BUNDLED = ROOT / "climaid" / "documentation"

# Phrases that state which version a document describes. Historical mentions ("before v0.4.0",
# "development builds 0.2.0 and 0.3.0", changelog sections) are deliberately not matched.
CURRENT_VERSION_PATTERNS = [
    r"ClimAID (\d+\.\d+\.\d+) is under active testing",
    r"version (\d+\.\d+\.\d+) \(beta",
    r"Version (\d+\.\d+\.\d+) \(beta",
    r"Last update: version (\d+\.\d+\.\d+)",
    r"what changed in (\d+\.\d+\.\d+)",
    r"What's new in (\d+\.\d+\.\d+)",
    r"^# ClimAID (\d+\.\d+\.\d+)",
]

DOC_FILES = [
    "README.md",
    "MIGRATION_v2.md",
    "V2_FEEDBACK_RESPONSE.md",
    "benchmarks/README.md",
    "climaid/docs/readme.md",
    "site_docs/index.md",
    "site_docs/guide/status.md",
]

# Text files written by people (not generated). Data files keep whatever encoding they came with.
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".html", ".js", ".css", ".txt", ".ini", ".sh"}
SKIP_DIRS = {".git", "site", "documentation", "__pycache__", "dist", "build"}

# UTF-8 text decoded as Windows-1252: an em dash becomes the three characters U+00E2 U+20AC U+201D,
# an e-acute becomes U+00C3 U+00A9, and so on. Written as escapes so this file does not match itself.
MOJIBAKE = re.compile("\u00e2\u20ac|\u00c3[\u0080-\u00bf]|\u00c2[\u00a0-\u00bf]")


def _text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES \
                and not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            yield path


def test_version_is_well_formed():
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), VERSION


def test_pyproject_reads_the_version_from_the_package():
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in toml
    assert 'version = {attr = "climaid.__version__"}' in toml
    assert not re.search(r'^version\s*=\s*"', toml, re.M), "hard-coded version in pyproject.toml"


def test_dashboard_reports_the_package_version():
    from climaid.browser_ui.server import app
    assert app.version == VERSION


def test_cli_reports_the_package_version():
    from typer.testing import CliRunner
    from climaid.cli import app
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0 and VERSION in result.output


def test_changelog_has_a_section_for_this_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    first = re.search(r"^## (\d+\.\d+\.\d+)", changelog, re.M)
    assert first and first.group(1) == VERSION, "newest CHANGELOG.md section must be the current version"


def test_docs_changelog_page_includes_the_repository_changelog():
    page = (ROOT / "site_docs" / "changelog.md").read_text(encoding="utf-8")
    assert '--8<-- "CHANGELOG.md"' in page
    assert not (ROOT / "CHANGELOG_v2.md").exists(), "a second changelog would drift out of date"


@pytest.mark.parametrize("rel", DOC_FILES)
def test_documents_name_the_current_version(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    found = [m.group(1) for p in CURRENT_VERSION_PATTERNS for m in re.finditer(p, text, re.M)]
    wrong = sorted(set(v for v in found if v != VERSION))
    assert not wrong, f"{rel} names version(s) {wrong}; the package is {VERSION}"


def test_bundled_documentation_was_rebuilt_for_this_version():
    index = (BUNDLED / "index.html").read_text(encoding="utf-8")
    assert f"ClimAID {VERSION} is under active testing" in index, \
        "climaid/documentation is stale: run ./build_and_bundle.sh"
    for page in BUNDLED.rglob("*.html"):
        banner = re.findall(r"ClimAID (\d+\.\d+\.\d+) is under active testing", page.read_text(encoding="utf-8"))
        assert set(banner) <= {VERSION}, f"{page.relative_to(ROOT)} shows version {banner}"


def test_readme_links_point_to_existing_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for rel in re.findall(r"github\.com/sam-as/ClimAID/blob/main/([^)\s#]+)", readme):
        assert (ROOT / rel).exists(), f"README links to missing file {rel}"
    for rel in re.findall(r"github\.com/sam-as/ClimAID/tree/main/([^)\s#]+)", readme):
        assert (ROOT / rel).is_dir(), f"README links to missing folder {rel}"


def test_no_byte_order_marks_or_garbled_characters():
    bad = []
    for path in list(_text_files()) + list(BUNDLED.rglob("*.html")):
        data = path.read_bytes()
        rel = path.relative_to(ROOT)
        if data.startswith(b"\xef\xbb\xbf"):
            bad.append(f"{rel}: byte-order mark")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            bad.append(f"{rel}: not UTF-8")
            continue
        if MOJIBAKE.search(text):
            bad.append(f"{rel}: garbled characters (dashes or accents shown as odd character pairs)")
    assert not bad, "\n".join(bad)
