"""LPR Monitoring Pipeline — Airflow DAG

Periodically checks:
1. Serving API health and model consistency (MLflow @production vs loaded)
2. Canary inference with a known validation image
3. Prometheus metrics for performance and data/prediction drift
4. Evaluates thresholds and writes a JSON monitoring report
5. Optionally triggers retraining when drift is detected
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from dags.monitoring_utils import (
    PROMQL,
    Thresholds,
    evaluate,
    format_summary,
    parse_prom_value,
)

log = logging.getLogger(__name__)

LPR_API_URL = os.getenv("LPR_API_URL", "http://api:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MONITORING_SCHEDULE = os.getenv("MONITORING_SCHEDULE", "*/15 * * * *")
AUTO_RETRAIN_ON_DRIFT = os.getenv("AUTO_RETRAIN_ON_DRIFT", "false").lower() in ("true", "1", "yes")

REPORTS_DIR = PROJECT / "data" / "monitoring_reports"


@dag(
    dag_id="lpr_monitoring_pipeline",
    description="Periodically inspect API health, canary inference, Prometheus metrics & drift",
    schedule=MONITORING_SCHEDULE,
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
        "execution_timeout": timedelta(minutes=5),
    },
    tags=["ddm501", "lpr", "monitoring"],
)
def lpr_monitoring_pipeline():

    @task
    def check_api_health() -> dict:
        """Inspects /health and /model/info of the serving API."""
        try:
            resp = requests.get(f"{LPR_API_URL}/health", timeout=10)
            resp.raise_for_status()
            health_data = resp.json()
        except Exception as exc:
            raise AirflowFailException(f"API /health unreachable: {exc}") from exc

        if health_data.get("status") != "healthy" or not health_data.get("model_loaded"):
            raise AirflowFailException(f"API reported unhealthy status: {health_data}")

        try:
            info_resp = requests.get(f"{LPR_API_URL}/model/info", timeout=10)
            info_data = info_resp.json() if info_resp.status_code == 200 else {}
        except Exception as exc:
            log.warning("Could not reach /model/info: %s", exc)
            info_data = {}

        return {
            "status": health_data.get("status"),
            "model_loaded": health_data.get("model_loaded"),
            "uptime_seconds": health_data.get("uptime_seconds"),
            "model_source": info_data.get("model_source", "unknown"),
            "model_version": info_data.get("model_version"),
            "yolo_loaded": info_data.get("yolo_loaded", False),
        }

    @task
    def check_model_consistency(api_health: dict) -> dict:
        """Compares the model version served by the API against MLflow @production."""
        import mlflow

        serving_version = api_health.get("model_version")
        registry_version = None
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.MlflowClient()
            model_info = client.get_model_version_by_alias("lpr-ocr-classifier", "production")
            registry_version = str(model_info.version)
        except Exception as exc:
            log.info("No active @production model in MLflow registry: %s", exc)

        consistent = (
            serving_version is not None
            and registry_version is not None
            and str(serving_version) == str(registry_version)
        )
        return {
            "registry_version": registry_version,
            "serving_version": serving_version,
            "consistent": consistent,
        }

    @task
    def canary_prediction() -> dict:
        """Performs a live canary prediction on the API with a validation sample."""
        val_dir = PROJECT / "dataset" / "images" / "val"
        sample_images = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg"))

        image_bytes = None
        filename = "canary_sample.png"
        if sample_images:
            chosen = sample_images[0]
            filename = chosen.name
            image_bytes = chosen.read_bytes()
        else:
            # Fallback synthetic 1-line plate
            from scripts.generate_synthetic_plates import random_plate, render_plate
            import random
            import cv2
            _, lines = random_plate(random.Random(42), two_line=False)
            img_rgb = render_plate(lines)
            _, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            image_bytes = buf.tobytes()

        try:
            files = {"file": (filename, image_bytes, "image/png")}
            data = {"assume_plate_crop": "false"}
            resp = requests.post(f"{LPR_API_URL}/predict", files=files, data=data, timeout=15)
            if resp.status_code == 429:
                log.warning("Canary prediction skipped due to rate limit (429)")
                return {"status": "rate_limited", "detail": "HTTP 429"}
            resp.raise_for_status()
            res_json = resp.json()
            return {
                "status": "success",
                "latency_ms": res_json.get("latency_ms"),
                "plate_text": res_json.get("plate_text"),
                "success": res_json.get("success"),
            }
        except Exception as exc:
            log.error("Canary prediction failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    @task
    def query_prometheus_metrics() -> dict:
        """Fetches sliding-window metrics from Prometheus /api/v1/query."""
        results = {}
        for metric_name, query in PROMQL.items():
            try:
                r = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query},
                    timeout=8,
                )
                if r.status_code == 200:
                    results[metric_name] = parse_prom_value(r.json())
                else:
                    results[metric_name] = None
            except Exception as exc:
                log.warning("Could not query %s: %s", metric_name, exc)
                results[metric_name] = None
        return results

    @task
    def evaluate_health_and_drift(
        api_health: dict,
        canary: dict,
        prom_metrics: dict,
    ) -> dict:
        """Evaluates health and drift thresholds against collected telemetry."""
        thresholds = Thresholds.from_env()
        eval_result = evaluate(
            metrics=prom_metrics,
            thresholds=thresholds,
            api_health=api_health,
            canary=canary,
        )
        return eval_result

    @task
    def write_monitoring_report(
        api_health: dict,
        consistency: dict,
        canary: dict,
        prom_metrics: dict,
        eval_result: dict,
        ds: str = None,
        ts_nodash: str = None,
    ) -> str:
        """Persists monitoring snapshot to disk."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"report_{ds}_{ts_nodash}.json"

        report_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "api_health": api_health,
            "consistency": consistency,
            "canary": canary,
            "metrics": prom_metrics,
            "evaluation": eval_result,
        }
        report_path.write_text(json.dumps(report_payload, indent=2))
        log.info(format_summary(report_payload))
        return str(report_path)

    @task.branch
    def branch_on_drift(eval_result: dict) -> str:
        """Branches to trigger retraining if drift is detected and auto-retrain is active."""
        if eval_result.get("drift_detected") and AUTO_RETRAIN_ON_DRIFT:
            log.warning("Drift detected and AUTO_RETRAIN_ON_DRIFT=true -> trigger_retraining")
            return "trigger_retraining"
        return "no_action"

    trigger_retrain_task = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="lpr_training_pipeline",
        conf={"reason": "drift_detected"},
    )

    no_action_task = EmptyOperator(task_id="no_action")

    @task(trigger_rule="all_done")
    def fail_if_unhealthy(eval_result: dict):
        """Marks the DAG run as FAILED if any violation occurred."""
        if not eval_result.get("healthy", True):
            violations = eval_result.get("violations", [])
            msg = f"Monitoring detected violations: {violations}"
            log.error(msg)
            raise AirflowFailException(msg)
        log.info("System is healthy. No violations detected.")

    health_info = check_api_health()
    consistency_info = check_model_consistency(health_info)
    canary_info = canary_prediction()
    prom_info = query_prometheus_metrics()

    evaluation = evaluate_health_and_drift(health_info, canary_info, prom_info)
    report_file = write_monitoring_report(
        health_info, consistency_info, canary_info, prom_info, evaluation
    )

    branch = branch_on_drift(evaluation)
    branch >> [trigger_retrain_task, no_action_task]

    fail_task = fail_if_unhealthy(evaluation)
    [trigger_retrain_task, no_action_task, report_file] >> fail_task


lpr_monitoring_pipeline()
