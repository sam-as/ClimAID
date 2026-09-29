"""MkDocs hooks for the ClimAID documentation (registered in mkdocs.yml under ``hooks:``).

Reads the package version from ``climaid/__init__.py`` so that the site-wide banner
(``overrides/main.html``) always shows the version being documented. The file is parsed
rather than imported, so building the docs does not need ClimAID's dependencies.
"""
import re
from pathlib import Path

_INIT = Path(__file__).resolve().parent / "climaid" / "__init__.py"


def climaid_version() -> str:
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']',
                      _INIT.read_text(encoding="utf-8"), re.M)
    if not match:
        raise RuntimeError(f"__version__ not found in {_INIT}")
    return match.group(1)


def on_config(config, **kwargs):
    config.extra["version"] = climaid_version()
    return config
