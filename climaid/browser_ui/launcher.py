from __future__ import annotations

import queue
import socket
import time
import webbrowser
import threading

import uvicorn


def _wait_for_server(host: str, port: int, timeout: float = 12.0) -> bool:
    """Wait until the TCP listener is reachable."""
    started = time.time()
    while time.time() - started < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def launch_browser_ui():
    """Start the ClimAID FastAPI app and open the browser.

    Startup exceptions are captured from the background thread and surfaced to
    the terminal instead of being converted into a misleading timeout error.
    """
    host = "127.0.0.1"
    port = 8765
    url = f"http://{host}:{port}"
    errors: queue.Queue[BaseException] = queue.Queue()

    def _run_server() -> None:
        try:
            uvicorn.run(
                "climaid.browser_ui.server:app",
                host=host,
                port=port,
                log_level="info",
            )
        except BaseException as exc:  # surface import/startup failures
            errors.put(exc)

    thread = threading.Thread(target=_run_server, name="climaid-uvicorn", daemon=False)
    thread.start()

    deadline = time.time() + 12.0
    while time.time() < deadline:
        if not errors.empty():
            exc = errors.get_nowait()
            raise RuntimeError(
                "ClimAID browser server failed to start. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if _wait_for_server(host, port, timeout=0.25):
            webbrowser.open(url)
            # Keep the CLI process alive while the browser server runs.
            # The server thread is intentionally non-daemon so it is not
            # terminated as soon as this function returns.
            thread.join()
            return
        time.sleep(0.1)

    if not errors.empty():
        exc = errors.get_nowait()
        raise RuntimeError(
            "ClimAID browser server failed to start. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    raise RuntimeError(
        "ClimAID browser server did not start within the timeout. "
        f"URL: {url}. Try running `python -m uvicorn climaid.browser_ui.server:app "
        "--host 127.0.0.1 --port 8765` to see the full startup error."
    )
