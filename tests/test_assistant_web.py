"""The assistant's chat page API (climaid/browser_ui/assistant_api.py)."""
import copy
import faulthandler
import shutil
import time

import pytest
from fastapi.testclient import TestClient

import climaid.assistant.core as core
import climaid.browser_ui.assistant_api as aa
from climaid.browser_ui.server import app
from test_assistant import DISTRICTS, RecordingRunner, _synthetic


@pytest.fixture(scope="module")
def precomputed(tmp_path_factory):
    """One real (small) v2 forecast, computed up front in the main thread.

    These tests check the chat page's plumbing (uploads, background runs, polling, report links); the
    forecast itself is tested in test_assistant.py. Computing it once here keeps the background thread
    short and independent of the machine's speed or thread scheduling.
    """
    tmp = tmp_path_factory.mktemp("precomputed")
    rec = RecordingRunner(tmp)
    dm = rec.load_model({"disease_name": "Dengue", "district": "NPL_Kathmandu_BAGMATI"})
    raw = rec.forecast(dm, {"forecast_origin": "2023-12-31", "horizon": 6})
    return raw


@pytest.fixture
def client(monkeypatch, tmp_path, precomputed):
    monkeypatch.setattr(core.Assistant, "districts", property(lambda self: DISTRICTS))
    monkeypatch.setattr(aa, "ASSISTANT_REPORTS", aa.REPORT_DIR / "assistant_test")
    rec = RecordingRunner(tmp_path)
    monkeypatch.setattr(aa.WebRunner, "load_model", lambda self, settings: rec.load_model(settings))

    def forecast(self, dm, settings):
        assert settings["horizon"] == 6 and settings["forecast_origin"] == "2023-12-31"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        result = copy.deepcopy(precomputed)
        report = self.out_dir / "climaid_v2_forecast.html"
        shutil.copyfile(precomputed["report_path"], report)
        result["report_path"] = str(report)
        return self._unique(result, "forecast")
    monkeypatch.setattr(aa.WebRunner, "forecast", forecast)
    aa._sessions.clear()
    return TestClient(app)


def _wait(client, sid, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        data = client.get("/assistant/messages", headers={"X-ClimAID-Session": sid}).json()
        if not data["busy"]:
            return data
        time.sleep(0.2)
    faulthandler.dump_traceback(all_threads=True)      # shows where the background run is stuck
    raise AssertionError(f"assistant still busy after {timeout} s; thread stacks printed above")


def _say(client, sid, text):
    r = client.post("/assistant/message", json={"text": text}, headers={"X-ClimAID-Session": sid})
    assert r.status_code == 200, r.text
    return _wait(client, sid)


def test_new_session_starts_with_a_greeting(client):
    data = client.get("/assistant/messages", headers={"X-ClimAID-Session": "s1"}).json()
    assert data["total"] == 1 and "ClimAID assistant" in data["messages"][0]["text"]
    assert "📎" in data["messages"][0]["text"]                 # the web intro points to the upload button
    assert client.get("/assistant.html").status_code == 200


def test_chat_upload_run_and_report_link(client, tmp_path):
    sid = "s2"
    _say(client, sid, "forecast dengue in Kathmandu for the next 6 months, use data up to December 2023")
    disease, _ = _synthetic()
    csv = tmp_path / "d.csv"
    disease.to_csv(csv, index=False)
    with csv.open("rb") as fh:
        r = client.post("/assistant/upload", files={"file": ("pune.csv", fh, "text/csv")}, data={"kind": "disease"},
                        headers={"X-ClimAID-Session": sid})
    assert r.status_code == 200
    data = _wait(client, sid)
    texts = [m["text"] for m in data["messages"]]
    assert "📎 Uploaded disease data: pune.csv" in texts
    assert "Shall I run it?" in texts[-1]
    data = _say(client, sid, "yes")
    final = "\n".join(m["text"] for m in data["messages"])
    assert "Trust rating:" in final
    link = next(w for w in final.split() if w.startswith("/reports/assistant_test/"))
    assert client.get(link).status_code == 200                # the report link works in the browser
    # messages are returned incrementally
    later = client.get(f"/assistant/messages?after={data['total']}", headers={"X-ClimAID-Session": sid}).json()
    assert later["messages"] == []


def test_docs_links_point_to_the_served_documentation(client):
    data = _say(client, "s3", "how does the scenario backtest work in detail")
    text = data["messages"][-1]["text"]
    assert "/documentation/" in text and "file://" not in text


def test_bad_uploads_and_messages_are_refused(client):
    h = {"X-ClimAID-Session": "s4"}
    assert client.post("/assistant/message", json={"text": "  "}, headers=h).status_code == 400
    r = client.post("/assistant/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")},
                    data={"kind": "disease"}, headers=h)
    assert r.status_code == 400
    assert client.post("/assistant/upload", files={"file": ("x.csv", b"a", "text/csv")},
                       data={"kind": "nonsense"}, headers=h).status_code == 400


def test_reset_starts_a_new_conversation(client):
    h = {"X-ClimAID-Session": "s5"}
    _say(client, "s5", "what is WIS?")
    assert client.post("/assistant/reset", headers=h).status_code == 200
    assert client.get("/assistant/messages", headers=h).json()["total"] == 1
