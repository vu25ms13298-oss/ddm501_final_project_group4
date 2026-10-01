"""Integration tests for the FastAPI API endpoints."""

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

MODELS_DIR = str(PROJECT_ROOT / "models" / "ocr_hog_svm")
os.environ.setdefault("MODELS_DIR", MODELS_DIR)
os.environ.setdefault("YOLO_MODEL_PATH", "")

from api.main import app

_has_models = os.path.isdir(MODELS_DIR) and os.path.exists(
    os.path.join(MODELS_DIR, "svm_classifier.pkl")
)
requires_model = pytest.mark.skipif(not _has_models, reason="OCR model files not found")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    @requires_model
    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True


class TestRootEndpoint:
    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "docs" in data


class TestModelInfoEndpoint:
    @requires_model
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "lpr_pipeline"
        assert "feature_method" in data
        assert "char_classes" in data
        assert len(data["char_classes"]) == 31


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        assert "api_requests_total" in body or "model_predictions_total" in body


class TestPredictEndpoint:
    @staticmethod
    def _make_test_image():
        img = np.full((80, 400, 3), 220, dtype=np.uint8)
        for x in range(20, 360, 45):
            img[15:65, x : x + 30] = 40
        _, buf = cv2.imencode(".png", img)
        return io.BytesIO(buf.tobytes())

    @requires_model
    def test_predict_returns_200(self, client):
        img_file = self._make_test_image()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_file, "image/png")},
            data={"assume_plate_crop": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
        assert "latency_ms" in data
        assert "timestamp" in data

    def test_predict_invalid_file(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_empty_file(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert resp.status_code == 400


class TestPredictBase64Endpoint:
    def test_predict_base64_missing_image(self, client):
        resp = client.post("/predict/base64", json={})
        assert resp.status_code == 400

    def test_predict_base64_invalid(self, client):
        resp = client.post("/predict/base64", json={"image": "not_valid_base64!!!"})
        assert resp.status_code == 400
