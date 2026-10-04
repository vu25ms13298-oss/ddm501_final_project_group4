# Contributing — Team Roles & Responsibilities

## Team Members

| Member | GitHub account(s) | Role | Presents (slides) |
|---|---|---|---|
| **Nguyễn Đình Đán** | `dan25ms13302-spec` | DataOps & model lifecycle | Motivation, problem, metrics, lifecycle, architecture |
| **Hoàng Xuân Sơn** | `hung2004-web`, `xn-son-tech` | ModelOps & inference pipeline | ML pipeline, data, training, registry, orchestration |
| **Nguyễn Hoàng Thái** | `NHT`, `NHT03` | Infrastructure, monitoring & simulation | Serving, results, Responsible AI |
| **Nguyễn Văn Vũ** | `Nguyen Van Vu` | Platform lead & CI/CD | Monitoring, CI/CD, lessons, limitations, conclusion |

Contributions below are taken from the git history (`git log --author=<account>`).

### Nguyễn Đình Đán — DataOps & model lifecycle
- Real-image evaluation: plate-text labels for 42 images with confidence levels
  (`dataset/plate_text_labels.csv`), ground-truth plate cropping
  (`scripts/crop_plates_from_labels.py`), analysis by plate type
- Synthetic plate benchmark (`scripts/generate_synthetic_plates.py`), end-to-end
  tests and the CI quality gate; coverage raised to ≥ 80%
- Registry → serving loop: `src/model_registry.py` (Pipeline packaging, `@production`
  alias, shared champion/challenger rule), API loading from MLflow
- Reproducibility: pinned ML library versions, trainer image, model retrained with
  consistent metadata, model binaries in Git LFS
- API hardening (metrics, threadpool inference, auth, upload limit, warm-up),
  drift metrics and alerts, Airflow fixes (paused DAG, stale PID, DB init on upgrade)
- EDA and experiment notebooks; Responsible AI fixes (feature extractor, held-out seed)

### Hoàng Xuân Sơn — ModelOps & inference pipeline
- API rate limiting (`RATE_LIMIT_PER_MINUTE`, HTTP 429) — `f399fe2`
- Inference pipeline (`src/pipeline.py`): thread-safe recognition lock, two-line plate
  grammar for two-letter series (e.g. `92CA`), `1` vs `7` disambiguation, fewer
  residual rotations to cut latency — `f399fe2`, `22c1398`
- Moved Airflow and MLflow Dockerfiles to `docker/` and updated compose/CI — `f399fe2`
- Code formatting fixes for CI — `05126ae`

### Nguyễn Hoàng Thái — Infrastructure, monitoring & simulation
- MinIO artifact storage and multi-container Airflow (webserver + scheduler) on
  PostgreSQL — `958baff`
- Monitoring DAG `lpr_monitoring_pipeline` with drift evaluation utilities — `527dab3`
- Traffic and drift simulation toolkit with fast alerts and tests (`simulations/`) — `0feb848`
- `DEMO_GUIDE.md`, architecture and README updates; lint and CI fixes; merged PR #2 and #3

### Nguyễn Văn Vũ — Platform lead & CI/CD
- Initial end-to-end system: pipeline, FastAPI service, Docker Compose, MLflow,
  Prometheus/Grafana — `eeb94ac`
- GitHub Actions CI/CD (lint, test, build, deploy) — `bd8a27a`
- Apache Airflow training pipeline — `130baec`
- YOLOv8 detection in Docker and YOLO weights — `6151e16`, `d22127f`
- Real sample dataset (50 images with YOLO labels) — `de081b9`, `9b4fe9a`

## Git Workflow

### Branch strategy
- `main` — always deployable; CI must be green
- `feat/<name>`, `fix/<name>`, `data/<name>`, `chore/<name>` — short-lived branches
  merged into `main` through a pull request

### Commit convention
```
<type>: <short description>

Types: feat, fix, docs, test, ci, refactor, style, data, chore, build
```

Examples from this repo:
```
feat: close the MLflow registry -> serving loop
fix(api): metrics double counting, blocking inference and hardening
data: label real sample plates and evaluate on real images
```

### Pull request process
1. Branch from the latest `main`
2. Commit in small, meaningful steps
3. Run `ruff check` and `black --check` on `src/ api/ scripts/ tests/ responsible_ai/ simulations/ dags/`
   and `pytest tests/ --cov=src --cov=api` locally (coverage must stay ≥ 80%)
4. Open a PR to `main` with a summary and a test plan
5. At least one other member reviews
6. Merge only when CI is green (lint, tests, end-to-end benchmark, Docker build)

## Development Setup

```bash
git lfs install                       # model binaries are stored in Git LFS
git clone https://github.com/vu25ms13298-oss/ddm501_final_project_group4.git
cd ddm501_final_project_group4

python -m venv .venv
source .venv/bin/activate             # Linux/Mac
.venv\Scripts\activate                # Windows

pip install -r requirements-ci.txt    # pinned versions, same as CI
pytest tests/ -v

cp .env.example .env
docker compose up -d --build
```
