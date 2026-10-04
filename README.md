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
- Git and **[Git LFS](https://git-lfs.com)** — model binaries (`models/**/*.pkl`,
  `*.joblib`, `*.pt`) are stored in LFS, see `.gitattributes`

### 1. Clone and Start Services

```bash
git lfs install        # once per machine, before cloning
git clone https://github.com/vu25ms13298-oss/ddm501_final_project_group4.git
cd ddm501_final_project_group4
cp .env.example .env   # optional: set API_ADMIN_TOKEN to enable /model/reload
docker compose up -d --build
```

### 2. Verify Services

| Service       | URL                         | Credentials     |
|---------------|-----------------------------|-----------------|
| API           | http://localhost:8000       | —               |
| API Docs      | http://localhost:8000/docs  | —               |
| MLflow        | http://localhost:5000       | —               |
| MinIO S3 API  | http://localhost:9000       | minio / minio123 |
| MinIO Console | http://localhost:9001       | minio / minio123 |
| Airflow       | http://localhost:8080       | admin / admin   |
| Prometheus    | http://localhost:9090       | —               |
| Grafana       | http://localhost:3000       | admin / admin   |

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
| POST   | `/model/reload`   | Reload model (registry or disk); requires `X-Admin-Token` |
| GET    | `/metrics`        | Prometheus metrics                   |

`/predict` accepts `assume_plate_crop` as a form field (`true` when the image is
already a tight plate crop). Uploads larger than `MAX_UPLOAD_MB` (default 10) get
HTTP 413. `/model/reload` is disabled unless `API_ADMIN_TOKEN` is set:

```bash
curl -X POST http://localhost:8000/model/reload -H "X-Admin-Token: $API_ADMIN_TOKEN"
```

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
- **Feature extractor**: HOG (pixels_per_cell=4x4, cells_per_block=2x2, orientations=9) → 1764-d
- **Classifier**: SVM (RBF kernel, C=5, gamma=auto — selected by 3-fold GridSearchCV)
- **Character set**: 31 classes — `0-9, A-H, K-N, P, R-V, X-Z` (excluding I, J, O, Q, W per Vietnamese plate rules)
- **Training data**: Synthetic characters (250/class) + plate-style synthetic (400/class), 16,120 train / 4,030 test
- **Environment**: scikit-learn 1.5.2 (pinned identically in API, Airflow, trainer and CI)
- Metrics below are reproducible with the commands in this README; all are on
  **synthetic** data and therefore an upper bound for real camera images:

| Model | Char accuracy | Char macro F1 | E2E exact plate (60 synthetic plates) | E2E char accuracy |
|---|---|---|---|---|
| **HOG-1764 + SVM (deployed, `@production` v2)** | 92.93% | 93.35% | **85.00%** | **96.41%** |
| HOG-324 (8x8 cells) + SVM | 96.90% | 96.90% | 81.67% | 93.61% |

HOG-324 scores higher on isolated characters, but HOG-1764 is better on the
end-to-end plate metric that matters for the business goal, so HOG-1764 is
deployed. Both runs are in the MLflow experiment `lpr-ocr-training`.

## Monitoring

### Prometheus Metrics
- `api_requests_total` — Request count by method/endpoint/status
- `api_request_latency_seconds` — Request latency histogram
- `model_predictions_total` — Predictions by plate type
- `model_prediction_latency_seconds` — Inference latency
- `model_prediction_errors_total` — Error count by type
- `plate_char_count` — Characters detected per plate
- `detection_confidence` — YOLO detection confidence
- `model_info` — Loaded model (feature method, classifier, source, registry version)
- Drift signals: `plate_recognition_result_total{result}` (valid-format rate),
  `plate_format_score`, `input_image_brightness`, `input_image_width_pixels`

### Grafana Dashboard
Pre-configured dashboard at http://localhost:3000 includes:
- Requests/sec, prediction latency (p50/p95/p99)
- Predictions by plate type, error rates
- Detection confidence distribution
- API uptime status
- Data & prediction drift: recognition success rate, input brightness and width
  (last hour vs 7-day baseline)

### Alert Rules
- `APIDown` — API unreachable for >1 minute
- `APIHighLatency` — p95 latency >5 seconds
- `APIHighErrorRate` — Error rate >0.1/sec
- `HighPredictionLatency` — Model inference >3 seconds
- `LowCharacterDetection` — Median chars <5
- `LowRecognitionSuccessRate` — <50% of predictions have a valid plate format (30 min)
- `InputBrightnessDrift` / `InputResolutionDrift` — 1h mean deviates from the 7-day
  baseline by >30% / >50%

## Experiment Tracking & Model Registry (MLflow)

Training runs in the `trainer` container, which pins the same library versions as
the API and Airflow images (a model pickled with one scikit-learn version may not
load correctly with another):

```bash
# Train, log to MLflow, register, and promote to @production if accuracy >= 0.90
docker compose run --rm trainer python scripts/train_with_mlflow.py \
  --feature hog --classifier svm --tune --cv-folds 3

# Make the running API pick up the new @production version
curl -X POST http://localhost:8000/model/reload -H "X-Admin-Token: $API_ADMIN_TOKEN"
```

MLflow UI at http://localhost:5000 tracks:
- Parameters: feature method, classifier type, hyperparameters
- Metrics: accuracy, F1, precision, recall, training time
- Artifacts: classification report, confusion matrix, metadata
- Registry: `lpr-ocr-classifier` (sklearn `Pipeline(scaler → SVM)` + metadata).
  The alias `@production` is what the API serves when `MODEL_SOURCE=mlflow`
  (default in docker-compose). Without a promoted version the API falls back to the
  baked-in `models/ocr_hog_svm`; `/model/info` reports `model_source` and
  `model_version`.

The system includes two automated Airflow pipelines:
- **`lpr_training_pipeline`** (@weekly): Automates the continuous training loop:
  generate synthetic data → validate quality → extract HOG features → train SVM with CV →
  evaluate → register to MLflow → promote to `@production` (Champion/Challenger) →
  hot-reload API.
- **`lpr_monitoring_pipeline`** (every 15 min): Automates health and drift oversight:
  check API health & model consistency → perform canary prediction → query sliding-window
  Prometheus metrics → evaluate against drift thresholds → write JSON report → optionally
  trigger retraining if drift is detected (`AUTO_RETRAIN_ON_DRIFT=true`).

## Traffic & Drift Simulations

A dedicated simulation toolkit (`simulations/`) allows testing the entire system under realistic traffic loads and induced environmental drift:

```bash
cd simulations
pip install -r requirements.txt

# Run a baseline simulation (60 requests @ 0.8 rps):
python run_simulation.py -n 60 -s normal

# Run specific scenarios (1 to 7):
python scenarios.py 2   # Nightfall (gradual brightness drop)
python scenarios.py 3   # Camera Swap (sudden resolution shift)
python scenarios.py 5   # Prediction Drift (non-plate images)
python scenarios.py 6   # Traffic Spike (triggers HTTP 429 rate limit)

# Run end-to-end stack smoke test:
# Windows PowerShell: .\quick_test.ps1
# Linux / macOS:      ./quick_test.sh
```

Observe live Grafana dashboards (`http://localhost:3000`) and Prometheus alert states (`http://localhost:9090/alerts`) as scenarios execute.

## Evaluation on Real Images

`dataset/` contains a 50-image sample (35 train + 15 val) of the real detection
dataset. Plate text was labelled manually in `dataset/plate_text_labels.csv`
(with a `confidence` column: 38 high, 4 medium). 8 images whose plates are too
small, blurry or ambiguous are excluded and listed in
`dataset/plate_text_unlabelled.txt`. Evaluated: **42 images** — 24 two-line car
plates, 15 motorbike plates (`59-S3 614.75` style), 3 one-line plates; 32 white,
8 yellow, 2 blue. Only the largest plate in each image is labelled.

```bash
# Mode 1 — full scene: YOLO detection + OCR (real usage)
python scripts/evaluate_lpr_end_to_end.py --manifest dataset/plate_text_labels.csv \
  --output results/real_eval_scene.csv
# Mode 2 — OCR only on ground-truth plate crops (from the YOLO polygon labels)
python scripts/crop_plates_from_labels.py
python scripts/evaluate_lpr_end_to_end.py --manifest results/real_plate_crops/manifest.csv \
  --output results/real_eval_crops.csv
```

| Mode | Exact plate | Char accuracy |
|---|---|---|
| Full scene (YOLO + OCR) | **15/42 (35.7%, 95% CI 23–51%)** | 63.2% |
| Ground-truth crops (OCR only) | 18/42 (42.9%, 95% CI 29–58%) | 81.0% |

Exact-plate accuracy by group (full scene):

| Group | Exact plate |
|---|---|
| 2-line car plates | 11/24 (45.8%) |
| Motorbike plates | **2/15 (13.3%)** |
| 1-line plates | 2/3 |
| Yellow / white / blue | 4/8 / 11/32 / **0/2** |
| Detection-dataset train / val split | 10/27 (37.0%) / 5/15 (33.3%) |

Findings:
- **The real-image result (35.7%) is far below the synthetic benchmark (85%) and
  the 70% business target.** Synthetic scores are an upper bound, as expected.
- **Motorbike plates are the main failure mode** (13% exact): their 4-character top
  line with a hyphen (`59-S3`) and small size break character segmentation.
- **Blue plates (white text on blue)** fail completely: segmentation and the
  synthetic training data assume dark text on a light plate.
- On ground-truth crops the character accuracy rises from 63% to 81%, so roughly
  half of the character errors come from the scene path (detection crop,
  rectification, deskew) rather than from the HOG+SVM classifier itself.
- Train vs val images score similarly, so YOLO having seen the train images does
  not noticeably inflate the result.
- Typical classifier errors: 1↔4/7, 5↔9, 6↔0, extra or missing characters.
- Next steps: inverted-polarity handling, motorbike-specific segmentation (split
  the top line on the hyphen), and training the OCR on real character crops taken
  from these labels.

## End-to-End Benchmark (synthetic)

Only a 50-image real sample is committed (the full Roboflow detection dataset is
referenced in `dataset/data.yaml`). For a reproducible end-to-end regression check,
the repo generates labelled synthetic plates (1-line and 2-line, crops and scenes):

```bash
python scripts/generate_synthetic_plates.py --out-dir data/benchmarks/synthetic_plates --n 60
python scripts/evaluate_lpr_end_to_end.py \
  --manifest data/benchmarks/synthetic_plates/manifest.csv \
  --min-char-accuracy 0.85 --min-exact-accuracy 0.50
```

CI runs this as a quality gate. Synthetic plates are cleaner than camera images, so
these scores are an **upper bound**; to evaluate on real data, pass a CSV manifest
with `image,label[,plate_crop]` columns to the same script.

## Testing

```bash
# Install test dependencies (Python 3.10+; Debian/Ubuntu also need fonts-dejavu-core)
pip install -r requirements-ci.txt

# Run all tests
pytest tests/ -v

# Run specific test types
pytest tests/test_unit_*.py         # Unit tests
pytest tests/test_integration_*.py  # Integration tests (API)
pytest tests/test_e2e_*.py          # End-to-end on synthetic plates
pytest tests/test_data_quality.py   # Data quality tests
pytest tests/test_model_validation.py  # Model validation

# With coverage (CI requires >= 80%)
pytest tests/ --cov=src --cov=api --cov-report=term-missing

# Or, without a local Python setup, inside the trainer image:
docker compose run --rm -v "$PWD:/app" trainer pytest tests/ --cov=src --cov=api
```

Current result: 110 tests, 80.5% line coverage (`src/` + `api/`). Uncovered code is
mostly the optional ResNet18/YOLO-training paths that need torch, which CI does
not install.

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

## Upgrading an Existing Stack (MinIO + multi-container Airflow)

Stacks created before MinIO / the split Airflow services need two one-off steps;
fresh clones do not.

1. **Airflow database**: `airflow-init` now creates the `airflow` database when it
   is missing (the postgres init script only runs on an empty volume). Remove the
   old container with `docker compose up -d --build --remove-orphans`.
2. **MLflow artifacts**: registered models created before MinIO still have their
   files in the old `mlflow_artifacts` volume. Copy them into the bucket, then
   reload the API — otherwise the API cannot load `@production` and silently falls
   back to the baked-in model (`/model/info` shows `model_source: local`):

```bash
docker run --rm --network ddm501_final_project_group4_lpr_network \
  -v ddm501_final_project_group4_mlflow_artifacts:/old:ro \
  --entrypoint sh cgr.dev/chainguard/minio-client:latest-dev \
  -c 'mc alias set local http://minio:9000 minio minio123 && mc mirror /old local/mlflow-artifacts'
curl -X POST http://localhost:8000/model/reload -H "X-Admin-Token: $API_ADMIN_TOKEN"
```

## Troubleshooting

**Docker build fails with memory error**: Increase Docker memory to 8GB+ (Settings → Resources).

**API returns 503**: The model files are not found. Ensure `models/ocr_hog_svm/` contains `svm_classifier.pkl`, `feature_scaler.pkl` and `metadata.json` (or promote a model to `@production` in MLflow).

**"… is a Git LFS pointer, not the model file"** (or `models/**/*.pkl` / `best.pt` are ~130-byte text files): the repo was cloned without Git LFS. Run `git lfs install && git lfs pull`, then `docker compose up -d --build api`.

**`yolo_loaded: false`**: `models/yolo_runs/license_plate_yolov8_bbox/weights/best.pt` is missing or still an LFS pointer (see above). The directory is mounted read-only into the API container; after fixing it run `docker compose restart api`. Without weights the API uses the contour-based fallback detector.

**Synthetic data looks wrong / RuntimeWarning about fonts**: the character generator needs TrueType fonts. Install `fonts-dejavu-core` (already included in the trainer and Airflow images).

**MLflow connection refused**: Wait for PostgreSQL to be healthy before MLflow starts. Run `docker compose logs mlflow` to check.

**Airflow DAG not visible**: Wait ~60 seconds after startup for Airflow to parse DAGs. Check `docker compose logs airflow` for errors.

**Grafana shows no data**: Verify Prometheus is scraping the API: visit http://localhost:9090/targets.
