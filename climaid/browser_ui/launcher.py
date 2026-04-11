import webbrowser
import uvicorn


def launch_browser_ui():

    url = "http://127.0.0.1:8765"

    webbrowser.open(url)

    uvicorn.run(
        "climaid.browser_ui.server:app",
        host="127.0.0.1",
        port=8765,
        log_level="warning",
    )