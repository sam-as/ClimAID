"""
server.py
------------
Provides the FastAPI application for the ClimAID browser wizard.

The original browser endpoints and static interface are retained, with additive
v2 probabilistic forecasting and report-serving routes.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .. import __version__
from .api import router, REPORT_DIR

app = FastAPI(title="ClimAID Wizard", version=__version__)
app.include_router(router)

STATIC_DIR = Path(__file__).parent / "static"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Explicit report route is mounted before the catch-all static directory.
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")

# Bundled documentation (built MkDocs site shipped in climaid/documentation), served at
# /documentation (FastAPI already uses /docs for its API page).
DOCS_DIR = Path(__file__).resolve().parents[1] / "documentation"
if (DOCS_DIR / "index.html").exists():
    app.mount("/documentation", StaticFiles(directory=DOCS_DIR, html=True), name="documentation")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

