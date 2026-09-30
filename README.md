# License Plate Recognition — End-to-End ML System

[![CI Pipeline](https://github.com/vu25ms13298-oss/ddm501_final_project_group4/actions/workflows/ci.yml/badge.svg)](https://github.com/vu25ms13298-oss/ddm501_final_project_group4/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Vietnamese License Plate Recognition system built as a complete ML production pipeline — from data preprocessing to model serving, monitoring, and CI/CD.

## Problem Definition & Requirements

### Business Context

In Vietnam, manual license plate verification is used across parking systems, toll booths, traffic enforcement, and residential access control. This process is slow, error-prone, and labor-intensive. An automated License Plate Recognition (LPR) system can reduce human effort, speed up vehicle processing, and enable scalable traffic management.

### Problem Statement

Build an end-to-end ML system that automatically recognizes Vietnamese license plates from vehicle images, serving predictions through a production-ready REST API with monitoring, experiment tracking, and CI/CD infrastructure.

### User Requirements

**Functional Requirements:**
- FR1: Accept vehicle/plate images via REST API and return recognized plate text
- FR2: Support both full scene images (with plate detection) and pre-cropped plate images
- FR3: Provide model health status and metadata through API endpoints
- FR4: Track training experiments with parameters, metrics, and model artifacts
- FR5: Monitor system and model performance in real-time via dashboards

**Non-Functional Requirements:**
- NFR1: API response latency < 3 seconds for single image prediction
- NFR2: System availability monitored with automated alerting
- NFR3: Containerized deployment reproducible across environments
- NFR4: Automated CI/CD pipeline for code quality and testing
- NFR5: Model decisions explainable via SHAP/LIME

**Priority:** FR1 > FR5 > NFR1 > FR2 > FR3 > NFR3 > FR4 > NFR2 > NFR4 > NFR5

### Success Metrics

| Level    | Metric                        | Target       |
|----------|-------------------------------|--------------|
| Business | Plate recognition accuracy    | > 70% (full plate exact match) |
| Business | Processing throughput          | > 10 images/min |
| Model    | Character OCR accuracy         | > 90%        |
| Model    | Character OCR macro F1         | > 90%        |
| Model    | Digit/letter fairness gap      | < 10%        |
| System   | API p95 latency                | < 3 seconds  |
| System   | API availability               | > 99%        |
| System   | Test coverage                  | > 80%        |

### Scope & Constraints

**In scope:**
- Vietnamese license plates (1-line and 2-line formats)
- Single plate per image
- Static images (JPEG/PNG)
- Classical ML approach (HOG + SVM) for OCR
- Self-hosted deployment via Docker Compose

**Out of scope:**
- Real-time video stream processing
- Multi-plate detection in single image
- International plate formats
- Cloud-managed deployment (AWS/GCP/Azure)
- Mobile/edge deployment

## Overview

This project recognizes Vietnamese license plates using a two-stage ML pipeline:
1. **YOLOv8** — Detects and crops the license plate region from scene images
2. **HOG + SVM** — Extracts HOG features from segmented characters and classifies them using an SVM classifier

The system is deployed as a REST API with full MLOps infrastructure: experiment tracking (MLflow), monitoring (Prometheus + Grafana), containerization (Docker), CI/CD (GitHub Actions), and Responsible AI analysis (SHAP + LIME).

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────>│  FastAPI     │────>│  LPR Pipeline│
│   (Image)    │<────│  REST API    │<────│  (ML Model)  │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                    ┌───────┴────────┐
                    │  /metrics      │
                    └───────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       ┌──────┴──────┐           ┌────────┴────────┐
       │ Prometheus   │──────────│    Grafana       │
       │ (Metrics)    │          │  (Dashboards)    │
       └──────────────┘          └─────────────────┘

       ┌──────────────┐     ┌──────────────┐
       │   MLflow     │     │   Airflow    │
       │  (Tracking)  │<────│ (Pipeline    │
       │  + Registry  │     │  Orchestration)│
       └──────┬───────┘     └──────────────┘
              │
       ┌──────┴───────┐
       │  PostgreSQL  │
       │  (Backend)   │
       └──────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)
- Git

### 1. Clone and Start Services

```bash
git clone https://github.com/vu25ms13298-oss/ddm501_final_project_group4.git
cd license-plate-recognition
docker compose up -d --build
```

### 2. Verify Services

| Service    | URL                    | Credentials     |
|------------|------------------------|-----------------|
| API        | http://localhost:8000  | —               |
| API Docs   | http://localhost:8000/docs | —           |
| MLflow     | http://localhost:5000  | —               |
| Airflow    | http://localhost:8080  | admin / admin   |
| Prometheus | http://localhost:9090  | —               |
| Grafana    | http://localhost:3000  | admin / admin   |

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Recognize a license plate (file upload)
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "assume_plate_crop=false"

# Model info
curl http://localhost:8000/model/info
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | Service info                         |
| GET    | `/health`         | Health check with model status       |
| GET    | `/model/info`     | Loaded model details                 |
| POST   | `/predict`        | Recognize plate from uploaded image  |
| POST   | `/predict/base64` | Recognize plate from base64 image    |
| POST   | `/model/reload`   | Reload model from disk               |
| GET    | `/metrics`        | Prometheus metrics                   |

### Prediction Response

```json
{
  "success": true,
  "plate_text": "51F12345",
  "raw_text": "51F12345",
  "plate_type": "1line",
  "char_count": 8,
  "detection_confidence": 0.92,
  "format_score": 15.0,
  "latency_ms": 234.5,
  "timestamp": "2026-09-29T10:30:00Z"
}
```

## Project Structure

```
license-plate-recognition/
├── api/                          # FastAPI REST API
│   ├── main.py                   # API endpoints + Prometheus metrics
│   ├── Dockerfile                # API container
│   └── requirements.txt
├── src/                          # Core ML pipeline
│   ├── preprocessing.py          # CLAHE, bilateral filter, deskew
│   ├── detection.py              # YOLOv8 + contour fallback
│   ├── plate_classifier.py       # 1-line vs 2-line classification
│   ├── segmentation.py           # Character segmentation
│   ├── features.py               # HOG, ResNet18, Wavelet extractors
│   ├── classifier.py             # SVM training + synthetic data gen
│   └── pipeline.py               # End-to-end orchestration
├── models/                       # Trained model artifacts
│   ├── ocr_hog_svm/              # Primary OCR model (HOG+SVM)
│   └── yolo_runs/                # YOLO training outputs
├── mlflow/                       # MLflow tracking server
│   └── Dockerfile
├── airflow/                      # Airflow pipeline orchestration
│   └── Dockerfile
├── dags/                         # Airflow DAG definitions
│   └── lpr_training_pipeline.py  # OCR training pipeline (6 tasks)
├── config/                       # Monitoring configuration
│   ├── prometheus.yml
│   ├── prometheus/alert_rules.yml
│   └── grafana/                  # Dashboards + provisioning
├── tests/                        # Test suite
│   ├── test_unit_preprocessing.py
│   ├── test_unit_features.py
│   ├── test_unit_pipeline.py
│   ├── test_integration_api.py
│   ├── test_data_quality.py
│   └── test_model_validation.py
├── responsible_ai/               # Responsible AI
│   ├── explainability.py         # SHAP + LIME analysis
│   ├── fairness_analysis.py      # Bias detection
│   └── ETHICS.md                 # Ethical considerations
├── scripts/                      # Training & utility scripts
│   ├── train_ocr_model.py
│   ├── train_with_mlflow.py      # MLflow-integrated training
│   └── ...
├── .github/workflows/ci.yml     # GitHub Actions CI/CD
├── docker-compose.yml            # Multi-service orchestration
├── ARCHITECTURE.md               # System design documentation
├── CONTRIBUTING.md               # Team roles & responsibilities
└── requirements.txt              # Python dependencies
```

## ML Pipeline

### Data Flow
```
Scene Image → Preprocessing (CLAHE, bilateral filter)
  → YOLO Detection (plate crop)
  → Perspective Rectification + Deskew
  → Binary Thresholding (multi-candidate)
  → Character Segmentation (connected components)
  → HOG Feature Extraction (1764-d)
  → SVM Classification (31 classes)
  → Vietnamese Plate Format Correction
  → Result
```

### Model Details
- **Feature extractor**: HOG (pixels_per_cell=4x4, orientations=9) → 1764-d
- **Classifier**: SVM (RBF kernel, C=10)
- **Character set**: 31 classes — `0-9, A-H, K-N, P, R-V, X-Z` (excluding I, J, O, Q, W per Vietnamese plate rules)
- **Training data**: Synthetic characters (250/class) + plate-style synthetic (400/class)
- **Reported accuracy**: 92.73% (character-level), 72.73% (full plate)

## Monitoring

### Prometheus Metrics
- `api_requests_total` — Request count by method/endpoint/status
- `api_request_latency_seconds` — Request latency histogram
- `model_predictions_total` — Predictions by plate type
- `model_prediction_latency_seconds` — Inference latency
- `model_prediction_errors_total` — Error count by type
- `plate_char_count` — Characters detected per plate
- `detection_confidence` — YOLO detection confidence

### Grafana Dashboard
Pre-configured dashboard at http://localhost:3000 includes:
- Requests/sec, prediction latency (p50/p95/p99)
- Predictions by plate type, error rates
- Detection confidence distribution
- API uptime status

### Alert Rules
- `APIDown` — API unreachable for >1 minute
- `APIHighLatency` — p95 latency >5 seconds
- `APIHighErrorRate` — Error rate >0.1/sec
- `HighPredictionLatency` — Model inference >3 seconds
- `LowCharacterDetection` — Median chars <5

## Experiment Tracking (MLflow)

```bash
# Train with MLflow tracking
python scripts/train_with_mlflow.py \
  --feature hog \
  --classifier svm \
  --experiment-name lpr-ocr-training
```

MLflow UI at http://localhost:5000 tracks:
- Parameters: feature method, classifier type, hyperparameters
- Metrics: accuracy, F1, precision, recall, training time
- Artifacts: model files, classification report, confusion matrix

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run all tests
pytest tests/ -v

# Run specific test types
pytest tests/test_unit_*.py         # Unit tests
pytest tests/test_integration_*.py  # Integration tests
pytest tests/test_data_quality.py   # Data quality tests
pytest tests/test_model_validation.py  # Model validation

# With coverage
pytest tests/ --cov=src --cov=api --cov-report=term-missing
```

## Responsible AI

```bash
# Run SHAP + LIME explainability analysis
pip install shap lime
python responsible_ai/explainability.py --method both

# Run fairness analysis
python responsible_ai/fairness_analysis.py
```

Outputs in `responsible_ai/results/`:
- SHAP feature importance plots
- LIME individual prediction explanations
- Fairness report (digit vs letter accuracy, confusion analysis)
- Ethics documentation (`responsible_ai/ETHICS.md`)

## Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r api/requirements.txt
pip install pytest pytest-cov httpx shap lime

# Run API locally
uvicorn api.main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## Troubleshooting

**Docker build fails with memory error**: Increase Docker memory to 8GB+ (Settings → Resources).

**API returns 503**: The model files are not found. Ensure `models/ocr_hog_svm/` contains `classifier.joblib` and `scaler.joblib`.

**MLflow connection refused**: Wait for PostgreSQL to be healthy before MLflow starts. Run `docker compose logs mlflow` to check.

**Airflow DAG not visible**: Wait ~60 seconds after startup for Airflow to parse DAGs. Check `docker compose logs airflow` for errors.

**Grafana shows no data**: Verify Prometheus is scraping the API: visit http://localhost:9090/targets.
