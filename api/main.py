import base64
import binascii
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_registry import DEFAULT_MODEL_URI, load_from_registry
from src.pipeline import LPRPipeline

logger = logging.getLogger("lpr_api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_name", "plate_type"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency",
    ["model_name"],
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction errors",
    ["error_type"],
)
MODEL_INFO_GAUGE = Gauge(
    "model_info",
    "Currently loaded model information",
    ["model_name", "feature_method", "classifier", "model_source", "model_version"],
)
CHAR_COUNT_HISTOGRAM = Histogram(
    "plate_char_count",
    "Number of characters detected per plate",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
CONFIDENCE_HISTOGRAM = Histogram(
    "detection_confidence",
    "YOLO detection confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
# ---- Data / prediction drift signals ----
RECOGNITION_RESULT = Counter(
    "plate_recognition_result_total",
    "Recognitions by outcome (success = valid Vietnamese plate format)",
    ["result"],
)
FORMAT_SCORE_HISTOGRAM = Histogram(
    "plate_format_score",
    "Plate-format score of the recognised text",
    buckets=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
)
INPUT_BRIGHTNESS = Histogram(
    "input_image_brightness",
    "Mean grayscale brightness of input images (0-255)",
    buckets=[25, 50, 75, 100, 125, 150, 175, 200, 225, 255],
)
INPUT_WIDTH = Histogram(
    "input_image_width_pixels",
    "Width of input images in pixels",
    buckets=[100, 200, 400, 640, 800, 1280, 1920, 2560, 4096],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "lpr_pipeline"
MODELS_DIR = os.environ.get("MODELS_DIR", "models/ocr_hog_svm")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local").strip().lower()
MODEL_URI = os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
API_ADMIN_TOKEN = os.environ.get("API_ADMIN_TOKEN", "")
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(",")
    if o.strip()
]

pipeline: LPRPipeline | None = None
model_source: str = "none"
model_version: str | None = None
start_time: float = time.time()


def _load_ocr_into(p: LPRPipeline) -> tuple[str, str | None]:
    """Loads the OCR model into ``p``; returns (source, version).

    With MODEL_SOURCE=mlflow the registry is tried first and the baked-in
    MODELS_DIR is used as a fallback so the API still starts on a fresh stack.
    """
    if MODEL_SOURCE == "mlflow":
        try:
            clf, scaler, metadata, version = load_from_registry(
                MODEL_URI, MLFLOW_TRACKING_URI or None
            )
            p.set_ocr_model(clf, scaler, metadata)
            logger.info("Loaded OCR model %s (version %s)", MODEL_URI, version)
            return "mlflow", str(version) if version is not None else None
        except Exception:
            logger.warning(
                "Could not load %s from MLflow; falling back to %s",
                MODEL_URI,
                MODELS_DIR,
                exc_info=True,
            )
    p.load_svm_models(MODELS_DIR)
    return ("local", None) if p.svm_model is not None else ("none", None)


def load_pipeline():
    global model_source, model_version
    p = LPRPipeline()
    source, version = _load_ocr_into(p)
    if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
        p.load_yolo_model(YOLO_MODEL_PATH)
    elif YOLO_MODEL_PATH:
        logger.warning(
            "YOLO weights not found at %s; using contour fallback detection",
            YOLO_MODEL_PATH,
        )
    model_source, model_version = source, version
    MODEL_INFO_GAUGE.clear()
    MODEL_INFO_GAUGE.labels(
        model_name=MODEL_NAME,
        feature_method=p.feature_method,
        classifier=p.classifier_name,
        model_source=source,
        model_version=version or "",
    ).set(1)
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = await run_in_threadpool(load_pipeline)
    except Exception:
        logger.exception("Failed to load pipeline")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="License Plate Recognition API",
    description="Vietnamese License Plate Recognition — YOLOv8 detection + HOG/SVM OCR",
    version="1.1.0",
    root_path="/api/v1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


_client_request_times: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60.0
MAX_REQUESTS_PER_WINDOW = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))


@app.middleware("http")
async def track_requests(request, call_next):
    """Single place where every HTTP request is counted, timed, and rate-limited."""
    endpoint = request.url.path
    if endpoint.startswith("/predict"):
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        times = _client_request_times.setdefault(client_ip, [])
        times[:] = [t for t in times if now - t < RATE_LIMIT_WINDOW]
        if len(times) >= MAX_REQUESTS_PER_WINDOW:
            PREDICTION_ERRORS.labels(error_type="rate_limit_exceeded").inc()
            return Response(
                content='{"detail":"Rate limit exceeded (max 60 requests per minute)"}',
                status_code=429,
                media_type="application/json",
            )
        times.append(now)

    t0 = time.time()
    response = await call_next(request)
    if endpoint != "/metrics":
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(
            time.time() - t0
        )
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    success: bool
    plate_text: str | None = None
    raw_text: str | None = None
    plate_type: str | None = None
    char_count: int = 0
    detection_confidence: float | None = None
    format_score: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = ""


class Base64PredictRequest(BaseModel):
    image: str | None = None
    assume_plate_crop: bool = False


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    feature_method: str | None = None
    classifier: str | None = None
    uptime_seconds: float = 0.0


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    feature_method: str
    classifier: str
    char_classes: list[str]
    feature_dim: int | None = None
    yolo_loaded: bool
    model_source: str
    model_version: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(data: bytes) -> np.ndarray:
    """Decodes image bytes to RGB, raising HTTP 400 when invalid."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        PREDICTION_ERRORS.labels(error_type="invalid_image").inc()
        raise HTTPException(status_code=400, detail="Invalid image file")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _check_size(n_bytes: int):
    if n_bytes > MAX_UPLOAD_BYTES:
        PREDICTION_ERRORS.labels(error_type="payload_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )


def _observe_input(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    INPUT_BRIGHTNESS.observe(float(gray.mean()))
    INPUT_WIDTH.observe(image_rgb.shape[1])


async def _predict(image_rgb: np.ndarray, assume_plate_crop: bool):
    """Runs inference off the event loop and records model metrics."""
    if pipeline is None or pipeline.svm_model is None:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    _observe_input(image_rgb)
    t0 = time.time()
    try:
        # CPU-bound: keep the event loop free for /health and /metrics.
        result = await run_in_threadpool(
            pipeline.recognize,
            image_rgb,
            assume_plate_crop=assume_plate_crop,
            verbose=False,
        )
    except Exception:
        PREDICTION_ERRORS.labels(error_type="inference_error").inc()
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")
    latency = time.time() - t0

    plate_type = result.get("plate_type", "unknown")
    char_count = len(result.get("char_images") or [])
    det_conf = result.get("detection_confidence")
    format_score = float(result.get("format_score", 0.0))
    success = bool(result.get("success", False))

    PREDICTION_COUNT.labels(model_name=MODEL_NAME, plate_type=plate_type).inc()
    PREDICTION_LATENCY.labels(model_name=MODEL_NAME).observe(latency)
    CHAR_COUNT_HISTOGRAM.observe(char_count)
    FORMAT_SCORE_HISTOGRAM.observe(format_score)
    RECOGNITION_RESULT.labels(result="success" if success else "failure").inc()
    if det_conf is not None:
        CONFIDENCE_HISTOGRAM.observe(det_conf)

    return PredictionResponse(
        success=success,
        plate_text=result.get("plate_string"),
        raw_text=result.get("raw_plate_string"),
        plate_type=plate_type,
        char_count=char_count,
        detection_confidence=det_conf,
        format_score=format_score,
        latency_ms=round(latency * 1000, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    return {
        "service": "License Plate Recognition API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    loaded = pipeline is not None and pipeline.svm_model is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        feature_method=pipeline.feature_method if pipeline else None,
        classifier=pipeline.classifier_name if pipeline else None,
        uptime_seconds=round(time.time() - start_time, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        feature_method=pipeline.feature_method,
        classifier=pipeline.classifier_name,
        char_classes=pipeline.char_classes,
        feature_dim=getattr(pipeline.scaler, "n_features_in_", None),
        yolo_loaded=pipeline.yolo_model is not None,
        model_source=model_source,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    assume_plate_crop: bool = Form(False),
):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        PREDICTION_ERRORS.labels(error_type="empty_file").inc()
        raise HTTPException(status_code=400, detail="Empty file")
    _check_size(len(contents))
    return await _predict(_decode_image(contents), assume_plate_crop)


@app.post("/predict/base64", response_model=PredictionResponse, tags=["Prediction"])
async def predict_base64(payload: Base64PredictRequest):
    if not payload.image:
        raise HTTPException(status_code=400, detail="Missing 'image' field (base64)")
    # base64 inflates size by 4/3; reject before decoding.
    _check_size(len(payload.image) * 3 // 4)
    try:
        image_bytes = base64.b64decode(payload.image, validate=True)
    except (binascii.Error, ValueError):
        PREDICTION_ERRORS.labels(error_type="invalid_image").inc()
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    return await _predict(_decode_image(image_bytes), payload.assume_plate_crop)


@app.post("/model/reload", tags=["Model"])
async def reload_model(x_admin_token: str | None = Header(default=None)):
    if not API_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Model reload is disabled")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, API_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token")

    global pipeline
    try:
        new_pipeline = await run_in_threadpool(load_pipeline)
    except Exception:
        logger.exception("Model reload failed")
        raise HTTPException(status_code=500, detail="Reload failed")
    pipeline = new_pipeline
    return {
        "status": "reloaded",
        "feature_method": pipeline.feature_method,
        "model_source": model_source,
        "model_version": model_version,
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
