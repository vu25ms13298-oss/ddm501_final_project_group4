"""Tests for the Alertmanager webhook receiver (docker/alert-receiver/receiver.py)."""

import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

RECEIVER = (
    Path(__file__).resolve().parents[1] / "docker" / "alert-receiver" / "receiver.py"
)

PAYLOAD = {
    "status": "firing",
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "APIDown", "severity": "critical"},
            "annotations": {"summary": "LPR API is down"},
            "startsAt": "2026-10-04T10:00:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
        },
        {
            "status": "resolved",
            "labels": {"alertname": "InputBrightnessDrift", "severity": "warning"},
            "annotations": {},
        },
    ],
}


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_LOG", str(tmp_path / "alerts.jsonl"))
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    spec = importlib.util.spec_from_file_location("alert_receiver", RECEIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server(receiver):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), receiver.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body: bytes):
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=5)


def test_summarize_flattens_alerts(receiver):
    records = receiver.summarize(PAYLOAD)
    assert [r["alertname"] for r in records] == ["APIDown", "InputBrightnessDrift"]
    assert records[1]["status"] == "resolved"
    assert records[1]["summary"] == ""


def test_format_line(receiver):
    line = receiver.format_line(receiver.summarize(PAYLOAD)[0])
    assert line == "[FIRING] critical APIDown: LPR API is down"


def test_webhook_logs_and_lists_alerts(receiver, server, tmp_path):
    resp = _post(f"{server}/alerts", json.dumps(PAYLOAD).encode())
    assert json.loads(resp.read()) == {"received": 2}

    stored = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(stored) == 2
    listed = json.loads(urllib.request.urlopen(f"{server}/alerts", timeout=5).read())
    assert listed[0]["alertname"] == "APIDown"


def test_invalid_json_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{server}/alerts", b"not json")
    assert err.value.code == 400


def test_health_and_unknown_path(server):
    health = json.loads(urllib.request.urlopen(f"{server}/health", timeout=5).read())
    assert health == {"status": "ok"}
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{server}/nope", timeout=5)
    assert err.value.code == 404
