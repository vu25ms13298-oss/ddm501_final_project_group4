import base64
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
from starlette.responses import Response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import LPRPipeline

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
    ["model_name", "feature_method", "classifier"],
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

# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "models/ocr_hog_svm")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "")

pipeline: LPRPipeline | None = None
start_time: float = 0.0


def load_pipeline():
    global pipeline, start_time
    start_time = time.time()
    p = LPRPipeline(models_dir=MODELS_DIR)
    if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
        p.load_yolo_model(YOLO_MODEL_PATH)
    MODEL_INFO_GAUGE.labels(
        model_name="lpr_pipeline",
        feature_method=p.feature_method,
        classifier=p.classifier_name,
    ).set(1)
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = load_pipeline()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="License Plate Recognition API",
    description="Vietnamese License Plate Recognition — YOLOv8 detection + HOG/SVM OCR",
    version="1.0.0",
    root_path="/api/v1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request, call_next):
    """Track all HTTP requests for Prometheus metrics."""
    t0 = time.time()
    response = await call_next(request)
    latency = time.time() - t0
    endpoint = request.url.path
    if endpoint != "/metrics":
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(
            latency
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_method: str | None = None
    classifier: str | None = None
    uptime_seconds: float = 0.0


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_method: str
    classifier: str
    char_classes: list[str]
    feature_dim: int | None = None
    yolo_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    return {
        "service": "License Plate Recognition API",
        "version": "1.0.0",
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
    n_features = getattr(pipeline.scaler, "n_features_in_", None)
    return ModelInfoResponse(
        model_name="lpr_pipeline",
        feature_method=pipeline.feature_method,
        classifier=pipeline.classifier_name,
        char_classes=pipeline.char_classes,
        feature_dim=n_features,
        yolo_loaded=pipeline.yolo_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    assume_plate_crop: bool = False,
):
    t0 = time.time()
    endpoint = "/predict"

    if pipeline is None:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    if not contents:
        PREDICTION_ERRORS.labels(error_type="empty_file").inc()
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        PREDICTION_ERRORS.labels(error_type="invalid_image").inc()
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = pipeline.recognize(
            image_rgb,
            assume_plate_crop=assume_plate_crop,
            verbose=False,
        )
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    latency = (time.time() - t0) * 1000

    plate_type = result.get("plate_type", "unknown")
    PREDICTION_COUNT.labels(model_name="lpr_pipeline", plate_type=plate_type).inc()
    PREDICTION_LATENCY.labels(model_name="lpr_pipeline").observe(latency / 1000)
    CHAR_COUNT_HISTOGRAM.observe(result.get("char_count", 0))
    det_conf = result.get("detection_confidence")
    if det_conf is not None:
        CONFIDENCE_HISTOGRAM.observe(det_conf)

    REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=200).inc()
    REQUEST_LATENCY.labels(method="POST", endpoint=endpoint).observe(latency / 1000)

    return PredictionResponse(
        success=result.get("success", False),
        plate_text=result.get("plate_string"),
        raw_text=result.get("raw_plate_string"),
        plate_type=plate_type,
        char_count=len(result.get("char_images") or []),
        detection_confidence=det_conf,
        format_score=result.get("format_score", 0.0),
        latency_ms=round(latency, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/base64", response_model=PredictionResponse, tags=["Prediction"])
async def predict_base64(
    payload: dict,
):
    t0 = time.time()
    endpoint = "/predict/base64"

    if pipeline is None:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_b64 = payload.get("image")
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field (base64)")

    assume_plate_crop = payload.get("assume_plate_crop", False)

    try:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        PREDICTION_ERRORS.labels(error_type="invalid_image").inc()
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    try:
        result = pipeline.recognize(
            image_rgb,
            assume_plate_crop=assume_plate_crop,
            verbose=False,
        )
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    latency = (time.time() - t0) * 1000

    plate_type = result.get("plate_type", "unknown")
    PREDICTION_COUNT.labels(model_name="lpr_pipeline", plate_type=plate_type).inc()
    PREDICTION_LATENCY.labels(model_name="lpr_pipeline").observe(latency / 1000)

    REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=200).inc()
    REQUEST_LATENCY.labels(method="POST", endpoint=endpoint).observe(latency / 1000)

    return PredictionResponse(
        success=result.get("success", False),
        plate_text=result.get("plate_string"),
        raw_text=result.get("raw_plate_string"),
        plate_type=plate_type,
        char_count=len(result.get("char_images") or []),
        detection_confidence=result.get("detection_confidence"),
        format_score=result.get("format_score", 0.0),
        latency_ms=round(latency, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    global pipeline
    try:
        pipeline = load_pipeline()
        return {"status": "reloaded", "feature_method": pipeline.feature_method}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
