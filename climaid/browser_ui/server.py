"""
server.py
------------

Provides helper functions to call the flask api

Author: Avik Kumar Sam
Created: March 2026
Updated: 2026
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .api import router

app = FastAPI(title="ClimAID Wizard")

app.include_router(router)

STATIC_DIR = Path(__file__).parent / "static"

app.mount(
    "/",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="static",
)
