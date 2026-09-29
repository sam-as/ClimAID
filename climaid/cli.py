"""
cli.py
------------

Author: Avik Kumar Sam
Created: March 2026
Updated: 2026
"""

import typer

app = typer.Typer()


def _print_version(value: bool):
    if value:
        from climaid import __version__
        typer.echo(f"ClimAID {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_print_version, is_eager=True,
                                      help="Show the ClimAID version and exit.")):
    """ClimAID: climate-informed disease modelling, forecasting and CMIP6 scenario analysis."""

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

@app.command()
def docs(port: int = typer.Option(0, help="Port to serve on (0 = any free port)"),
         page: str = typer.Option("", help="Page to open, e.g. guide/tuning/")):
    """Open the ClimAID documentation shipped with this installation (works offline)."""
    import functools, http.server, threading, webbrowser
    from pathlib import Path

    site = Path(__file__).resolve().parent / "documentation"
    if not (site / "index.html").exists():
        typer.echo("The documentation is not bundled with this installation. "
                   "Online version: https://sam-as.github.io/ClimAID/")
        raise typer.Exit(1)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/{page.lstrip('/')}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    typer.echo(f"ClimAID documentation: {url}   (press Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    app()