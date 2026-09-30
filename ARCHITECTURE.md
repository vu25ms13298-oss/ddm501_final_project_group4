# System Architecture

## 1. High-Level Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              Docker Compose Network          │
                        │                                             │
  ┌──────────┐          │  ┌──────────┐    ┌──────────────────────┐  │
  │  Client   │─────────┼─>│ FastAPI  │───>│    LPR Pipeline      │  │
  │ (Browser/ │<────────┼──│  :8000   │<───│ (Preprocessing →     │  │
  │  curl)    │         │  │          │    │  Detection → OCR)    │  │
  └──────────┘          │  └────┬─────┘    └──────────────────────┘  │
                        │       │                                     │
                        │       │ /metrics                            │
                        │       │                                     │
                        │  ┌────┴──────┐    ┌──────────────────┐     │
                        │  │Prometheus │───>│     Grafana      │     │
                        │  │  :9090    │    │     :3000        │     │
                        │  └───────────┘    └──────────────────┘     │
                        │                                             │
                        │  ┌───────────┐    ┌──────────────────┐     │
                        │  │  MLflow   │    │   Airflow         │     │
                        │  │  :5000    │<───│   :8080           │     │
                        │  └─────┬─────┘    │ (Pipeline Orch.)  │     │
                        │        │          └──────────────────┘     │
                        │  ┌─────┴─────┐                              │
                        │  │PostgreSQL │                              │
                        │  │  :5432    │                              │
                        │  └───────────┘                              │
                        └─────────────────────────────────────────────┘
```

## 2. Component Design

### 2.1. API Service (FastAPI)
**Responsibility**: Serve the LPR model via REST endpoints.

- Receives image uploads, decodes them, and passes to the LPR pipeline
- Exposes Prometheus metrics for monitoring
- Supports model hot-reloading without restart
- Health check endpoint for container orchestration

### 2.2. LPR Pipeline (Core ML)
**Responsibility**: End-to-end license plate recognition.

```
Input Image (RGB)
    │
    ▼
┌──────────────────┐
│ Scene Preprocessing│  CLAHE contrast enhancement (LAB space)
│                    │  Bilateral filter denoising
│                    │  Resize to max 1280px
└────────┬───────────┘
         ▼
┌──────────────────┐
│ Plate Detection   │  YOLOv8n (primary)
│                    │  Contour-based fallback (AR 1.2-5.5)
└────────┬───────────┘
         ▼
┌──────────────────┐
│ Plate Processing  │  Perspective rectification
│                    │  Deskew (Hough line angle)
│                    │  Resize to 400px width
│                    │  CLAHE + unsharp masking
└────────┬───────────┘
         ▼
┌──────────────────┐
│ Char Segmentation │  Multi-candidate binarization (Otsu, adaptive, Sauvola)
│                    │  Connected components analysis
│                    │  2-line split via horizontal projection
│                    │  Normalize to 32x32
└────────┬───────────┘
         ▼
┌──────────────────┐
│ Feature Extraction│  HOG: pixels_per_cell=4x4, orientations=9
│                    │  Output: 1764-d feature vector per character
└────────┬───────────┘
         ▼
┌──────────────────┐
│ SVM Classification│  RBF kernel, C=10, gamma=scale
│                    │  31 Vietnamese plate character classes
│                    │  Position-based constraint (digit/letter)
└────────┬───────────┘
         ▼
┌──────────────────┐
│ Post-Processing   │  Vietnamese plate format correction
│                    │  Digit/letter coercion by position
│                    │  Duplicate character deduplication
└────────┬───────────┘
         ▼
    Plate Text (e.g., "51F12345")
```

### 2.3. MLflow (Experiment Tracking)
**Responsibility**: Track training experiments, model versioning, and artifact storage.

- Backend store: PostgreSQL (parameters, metrics, run metadata)
- Artifact store: local filesystem volume (model files, plots, reports)
- Model registry for versioning and stage promotion

### 2.6. Apache Airflow (Pipeline Orchestration)
**Responsibility**: Orchestrate and schedule ML training pipelines.

- Standalone mode with SQLite backend (lightweight, no extra infra)
- DAG: `lpr_training_pipeline` with 6 tasks in sequence:
  1. `generate_data` — Synthetic character generation (250 + 400 samples/class)
  2. `validate_data` — Data quality checks (NaN, value range, class count)
  3. `extract_features` — HOG feature extraction from images
  4. `train` — SVM training with 5-fold CV, logged to MLflow
  5. `evaluate` — Accuracy/F1 metrics, classification report
  6. `register_model` — Conditional model registration (accuracy > 85%)
- Scheduled `@weekly`, integrates with MLflow for experiment tracking
- Web UI at port 8080 for pipeline monitoring and manual triggers

### 2.4. Monitoring Stack
**Responsibility**: Real-time system and model performance monitoring.

- **Prometheus**: Scrapes `/metrics` from API every 10s; evaluates alert rules
- **Grafana**: Visualizes metrics via pre-provisioned dashboards; alert notifications
- Metrics tracked: request rate, latency, errors, detection confidence, character count

### 2.5. CI/CD (GitHub Actions)
**Responsibility**: Automated quality gates on every push/PR.

```
Push/PR → Lint (ruff, black) → Unit Tests → Data Quality Tests
  → Model Validation Tests → Integration Tests → Docker Build
```

## 3. Data Flow Diagram

```
┌─────────┐    POST /predict    ┌──────────┐
│  Client  │ ─── image file ──> │   API    │
└─────────┘                     └────┬─────┘
                                     │ decode image
                                     ▼
                               ┌───────────┐
                               │ Pipeline   │
                               │ .recognize()│
                               └─────┬─────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              preprocess_scene  detect_plate    segment_plate
                    │                │                │
                    ▼                ▼                ▼
              CLAHE + filter   YOLO/contour    binary + CC
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                                     ▼
                              extract_hog_features
                                     │
                                     ▼
                              SVM predict + format correct
                                     │
                                     ▼
                              JSON response → Client
```

## 4. Technology Stack Justification

| Component          | Technology      | Justification                                              |
|--------------------|-----------------|------------------------------------------------------------|
| ML Framework       | scikit-learn    | Mature, well-tested for classical ML (SVM, scaling)        |
| Object Detection   | YOLOv8 (ultralytics) | State-of-the-art real-time detection, easy fine-tuning |
| Feature Extraction | scikit-image HOG| Proven for character recognition, no GPU required          |
| API Framework      | FastAPI         | Async, auto-docs (Swagger), Pydantic validation            |
| Containerization   | Docker Compose  | Multi-service orchestration, reproducible deployments      |
| Experiment Tracking| MLflow          | Industry standard, model registry, artifact management     |
| Metrics            | Prometheus      | Pull-based, time-series DB, PromQL, AlertManager-ready     |
| Visualization      | Grafana         | Rich dashboards, Prometheus integration, alerting          |
| Pipeline Orchestration | Apache Airflow | DAG-based scheduling, web UI, MLflow integration       |
| CI/CD              | GitHub Actions  | Native GitHub integration, free for public repos           |
| Explainability     | SHAP + LIME     | Model-agnostic, complementary global/local explanations    |

## 5. Trade-offs Analysis

| Decision                          | Pros                                    | Cons                                     |
|-----------------------------------|-----------------------------------------|------------------------------------------|
| HOG+SVM over deep learning OCR    | Interpretable, fast, no GPU needed      | Lower accuracy than CNN/Transformer      |
| Synthetic training data           | Unlimited, no labeling cost             | Domain gap with real plates              |
| Docker Compose over Kubernetes    | Simple setup, single-machine            | No auto-scaling, no rolling updates      |
| MLflow over W&B/Neptune           | Open-source, self-hosted, no vendor lock| Less polished UI, manual setup           |
| Single API container              | Simple, low resource usage              | No horizontal scaling                    |
| Prometheus pull model             | Simple config, no agent needed          | Requires network access to targets       |
