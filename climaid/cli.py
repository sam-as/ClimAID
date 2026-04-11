"""
cli.py
------------

Author: Avik Kumar Sam
Created: March 2026
Updated: 2026
"""

import typer

app = typer.Typer()

@app.command()
def browse():
    """Launch ClimAID browser wizard"""

    from climaid.browser_ui.launcher import launch_browser_ui



    launch_browser_ui()

@app.command()
def wizard():
    """Run terminal wizard"""

    from climaid.wizard import run_interactive_pipeline

    run_interactive_pipeline()

if __name__ == "__main__":
    app()