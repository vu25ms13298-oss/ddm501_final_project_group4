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


# ---------------------------------------------------------------------------
# Behaviour added after the review: auth, limits, threading, metrics, sources
# ---------------------------------------------------------------------------
import base64  # noqa: E402

import api.main as api_main  # noqa: E402


def _png_bytes(width=400, height=80):
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class _FakePipeline:
    """Stands in for LPRPipeline so API logic is tested without real inference."""

    feature_method = "hog"
    classifier_name = "svm"
    char_classes = list("0123456789ABCDEFGHKLMNPRSTUVXYZ")
    scaler = None
    yolo_model = None
    svm_model = object()

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def recognize(self, image, assume_plate_crop=False, verbose=True):
        self.calls.append(
            {"shape": image.shape, "assume_plate_crop": assume_plate_crop}
        )
        if self.fail:
            raise RuntimeError("secret internal detail")
        return {
            "success": True,
            "plate_string": "51F12345",
            "raw_plate_string": "51F12345",
            "plate_type": "1line",
            "char_images": [None] * 8,
            "detection_confidence": 0.9,
            "format_score": 15.0,
        }


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake = _FakePipeline()
    monkeypatch.setattr(api_main, "pipeline", fake)
    return fake


def _request_count(**labels):
    return api_main.REQUEST_COUNT.labels(**labels)._value.get()


class TestPredictBehaviour:
    def test_assume_plate_crop_form_field_is_honoured(self, client, fake_pipeline):
        resp = client.post(
            "/predict",
            files={"file": ("p.png", io.BytesIO(_png_bytes()), "image/png")},
            data={"assume_plate_crop": "true"},
        )
        assert resp.status_code == 200
        assert fake_pipeline.calls[-1]["assume_plate_crop"] is True
        assert resp.json()["plate_text"] == "51F12345"
        assert resp.json()["char_count"] == 8

    def test_predict_counts_request_once(self, client, fake_pipeline):
        labels = {"method": "POST", "endpoint": "/predict", "status": 200}
        before = _request_count(**labels)
        client.post(
            "/predict",
            files={"file": ("p.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert _request_count(**labels) - before == 1

    def test_payload_too_large(self, client, fake_pipeline, monkeypatch):
        monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 100)
        resp = client.post(
            "/predict",
            files={"file": ("p.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert resp.status_code == 413
        assert fake_pipeline.calls == []

    def test_inference_error_hides_details(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "pipeline", _FakePipeline(fail=True))
        resp = client.post(
            "/predict",
            files={"file": ("p.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert resp.status_code == 500
        assert "secret" not in resp.text

    def test_model_not_loaded_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "pipeline", None)
        resp = client.post(
            "/predict",
            files={"file": ("p.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert resp.status_code == 503

    def test_base64_success(self, client, fake_pipeline):
        payload = {
            "image": base64.b64encode(_png_bytes()).decode(),
            "assume_plate_crop": True,
        }
        resp = client.post("/predict/base64", json=payload)
        assert resp.status_code == 200
        assert fake_pipeline.calls[-1]["assume_plate_crop"] is True

    def test_base64_valid_but_not_image(self, client, fake_pipeline):
        payload = {"image": base64.b64encode(b"hello world").decode()}
        assert client.post("/predict/base64", json=payload).status_code == 400

    def test_base64_too_large(self, client, fake_pipeline, monkeypatch):
        monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 10)
        payload = {"image": base64.b64encode(_png_bytes()).decode()}
        assert client.post("/predict/base64", json=payload).status_code == 413

    def test_drift_metrics_exposed(self, client, fake_pipeline):
        client.post(
            "/predict",
            files={"file": ("p.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        body = client.get("/metrics").text
        assert "input_image_brightness_count" in body
        assert 'plate_recognition_result_total{result="success"}' in body


class TestModelReload:
    def test_disabled_without_token(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "API_ADMIN_TOKEN", "")
        assert client.post("/model/reload").status_code == 403

    def test_rejects_wrong_token(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "API_ADMIN_TOKEN", "s3cret")
        resp = client.post("/model/reload", headers={"X-Admin-Token": "nope"})
        assert resp.status_code == 401

    def test_rejects_missing_header(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "API_ADMIN_TOKEN", "s3cret")
        assert client.post("/model/reload").status_code == 401

    @requires_model
    def test_reload_with_valid_token(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "API_ADMIN_TOKEN", "s3cret")
        resp = client.post("/model/reload", headers={"X-Admin-Token": "s3cret"})
        assert resp.status_code == 200
        assert resp.json()["model_source"] == "local"

    def test_reload_failure_returns_500(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "API_ADMIN_TOKEN", "s3cret")

        def boom():
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(api_main, "load_pipeline", boom)
        resp = client.post("/model/reload", headers={"X-Admin-Token": "s3cret"})
        assert resp.status_code == 500
        assert "disk" not in resp.text


class TestModelSource:
    @requires_model
    def test_mlflow_source_falls_back_to_local(self, monkeypatch):
        def unavailable(*args, **kwargs):
            raise ConnectionError("registry down")

        monkeypatch.setattr(api_main, "MODEL_SOURCE", "mlflow")
        monkeypatch.setattr(api_main, "load_from_registry", unavailable)
        p = api_main.load_pipeline()
        assert p.svm_model is not None
        assert api_main.model_source == "local"

    @requires_model
    def test_mlflow_source_uses_registry_model(self, monkeypatch):
        local = api_main.LPRPipeline(models_dir=MODELS_DIR)

        def from_registry(uri, tracking_uri=None):
            return local.svm_model, local.scaler, {"feature_method": "hog"}, 7

        monkeypatch.setattr(api_main, "MODEL_SOURCE", "mlflow")
        monkeypatch.setattr(api_main, "load_from_registry", from_registry)
        p = api_main.load_pipeline()
        assert p.svm_model is local.svm_model
        assert (api_main.model_source, api_main.model_version) == ("mlflow", "7")

    def test_missing_yolo_weights_logs_and_continues(self, monkeypatch, tmp_path):
        monkeypatch.setattr(api_main, "YOLO_MODEL_PATH", str(tmp_path / "none.pt"))
        p = api_main.load_pipeline()
        assert p.yolo_model is None

    def test_model_info_reports_source(self, client):
        resp = client.get("/model/info")
        if resp.status_code == 200:
            assert resp.json()["model_source"] in ("local", "mlflow", "none")
