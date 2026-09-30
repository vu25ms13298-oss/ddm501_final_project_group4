# Contributing — Team Roles & Responsibilities

## Team Members

| Member   | Role                        | Primary Responsibilities                              |
|----------|-----------------------------|-------------------------------------------------------|
| Member 1 | Data Engineer               | Dataset collection, data pipeline, data quality tests |
| Member 2 | ML Engineer — Preprocessing | Image preprocessing, plate rectification, deskew      |
| Member 3 | ML Engineer — Detection/OCR | YOLO detection, segmentation, HOG features, SVM       |
| Member 4 | MLOps Engineer              | API, Docker, MLflow, monitoring, CI/CD, pipeline      |

## Responsibilities Breakdown

### Member 1 — Data Engineer
- Dataset sourcing (Roboflow Vietnamese plate data, Kaggle)
- Data labeling and quality assurance
- Synthetic data generation for OCR training
- Data quality validation tests

### Member 2 — ML Engineer (Preprocessing)
- Scene image preprocessing (CLAHE, bilateral filter)
- Plate crop rectification (perspective transform)
- Deskew algorithm (Hough line angle estimation)
- Plate image enhancement for OCR

### Member 3 — ML Engineer (Detection & OCR)
- YOLOv8 fine-tuning for plate detection
- Character segmentation (binary thresholding, connected components)
- HOG feature extraction implementation
- SVM classifier training and evaluation
- Feature comparison experiments (HOG, Wavelet, ResNet18, Raw)

### Member 4 — MLOps Engineer
- FastAPI REST API development
- Docker containerization and Docker Compose setup
- MLflow experiment tracking integration
- Prometheus metrics and Grafana dashboards
- GitHub Actions CI/CD pipeline
- Responsible AI (SHAP, LIME, fairness analysis)
- System documentation (README, ARCHITECTURE.md)

## Git Workflow

### Branch Strategy
- `main` — Production-ready code
- `develop` — Integration branch
- `feature/<name>` — Individual feature branches
- `fix/<name>` — Bug fix branches

### Commit Convention
```
<type>: <short description>

Types: feat, fix, docs, test, ci, refactor, style
```

Examples:
```
feat: add /predict endpoint with file upload support
fix: handle empty plate crop in segmentation
test: add model validation tests for SVM accuracy
ci: add GitHub Actions lint and test pipeline
docs: update ARCHITECTURE.md with data flow diagram
```

### Pull Request Process
1. Create feature branch from `develop`
2. Implement changes with meaningful commits
3. Ensure all tests pass locally (`pytest tests/ -v`)
4. Create PR to `develop` with description
5. At least 1 team member reviews before merge
6. CI pipeline must pass (lint, test, Docker build)

## Development Setup

```bash
# Clone repository
git clone https://github.com/vu25ms13298-oss/ddm501_final_project_group4.git
cd license-plate-recognition

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt
pip install -r api/requirements.txt
pip install pytest pytest-cov httpx shap lime ruff black

# Run tests
pytest tests/ -v

# Start services
docker compose up -d --build
```
