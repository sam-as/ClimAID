import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from climaid.browser_ui.server import app


def test_uploads_are_session_scoped_and_immediate():
    client = TestClient(app)
    sid1 = "8c3a9d9c-5f8e-4df3-9a10-7f4c8b2e0a11"
    sid2 = "1f2c7d8e-6b9a-4c31-8e12-4a5b9d0c6e22"
    payload1 = b"Date,Case\n2020-01-01,3\n2020-02-01,4\n"
    payload2 = b"Date,Case\n2020-01-01,8\n2020-02-01,9\n"
    r1 = client.post("/upload_dataset", headers={"X-ClimAID-Session": sid1}, files={"file": ("a.csv", io.BytesIO(payload1), "text/csv")})
    r2 = client.post("/upload_dataset", headers={"X-ClimAID-Session": sid2}, files={"file": ("b.csv", io.BytesIO(payload2), "text/csv")})
    assert r1.status_code == 200 and r1.json()["rows"] == 2
    assert r2.status_code == 200 and r2.json()["rows"] == 2
    assert r1.json()["session_id"] != r2.json()["session_id"]


def test_upload_rejects_bad_dataset_with_http_error():
    client = TestClient(app)
    r = client.post("/upload_dataset", headers={"X-ClimAID-Session": "8c3a9d9c-5f8e-4df3-9a10-7f4c8b2e0a11"}, files={"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 400
