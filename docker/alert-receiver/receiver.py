"""Minimal Alertmanager webhook receiver (standard library only).

POST /alerts   Alertmanager webhook payload -> one log line per alert, appended to
               ALERT_LOG (JSON lines) and, if SLACK_WEBHOOK_URL is set, forwarded
               to Slack as a text message.
GET  /alerts   The last 50 received alerts as JSON (for demos and debugging).
GET  /health   Liveness probe.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALERT_LOG = os.environ.get("ALERT_LOG", "/data/alerts.jsonl")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
PORT = int(os.environ.get("PORT", "8080"))
MAX_BODY_BYTES = 1024 * 1024
RECENT_LIMIT = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("alert-receiver")
recent: deque[dict] = deque(maxlen=RECENT_LIMIT)


def summarize(payload: dict) -> list[dict]:
    """Flattens an Alertmanager webhook payload into one record per alert."""
    records = []
    for alert in payload.get("alerts", []):
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        records.append(
            {
                "status": alert.get("status", payload.get("status", "unknown")),
                "alertname": labels.get("alertname", "unknown"),
                "severity": labels.get("severity", "none"),
                "summary": annotations.get("summary", ""),
                "starts_at": alert.get("startsAt"),
                "ends_at": alert.get("endsAt"),
            }
        )
    return records


def format_line(record: dict) -> str:
    """Human-readable one-liner, e.g. '[FIRING] critical APIDown: LPR API is down'."""
    return (
        f"[{record['status'].upper()}] {record['severity']} "
        f"{record['alertname']}: {record['summary']}"
    )


def forward_to_slack(lines: list[str]) -> None:
    if not SLACK_WEBHOOK_URL or not lines:
        return
    body = json.dumps({"text": "\n".join(lines)}).encode()
    request = urllib.request.Request(
        SLACK_WEBHOOK_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except OSError:
        log.exception("Slack forwarding failed")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/alerts":
            self._send(200, list(recent))
        else:
            self._send(404, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if self.path != "/alerts":
            self._send(404, {"detail": "Not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, {"detail": "Missing or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send(400, {"detail": "Body is not valid JSON"})
            return

        records = summarize(payload)
        lines = [format_line(r) for r in records]
        for record, line in zip(records, lines):
            log.info(line)
            recent.append(record)
        try:
            os.makedirs(os.path.dirname(ALERT_LOG) or ".", exist_ok=True)
            with open(ALERT_LOG, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
        except OSError:
            log.exception("Could not append to %s", ALERT_LOG)
        forward_to_slack(lines)
        self._send(200, {"received": len(records)})

    def log_message(self, fmt: str, *args) -> None:
        """Silences per-request access logs; alerts are logged explicitly."""


if __name__ == "__main__":
    log.info("alert-receiver listening on :%d (log: %s)", PORT, ALERT_LOG)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
