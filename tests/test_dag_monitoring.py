"""Unit tests for the monitoring utils and DAG structure."""

import pytest
from dags.monitoring_utils import (
    Thresholds,
    evaluate,
    format_summary,
    parse_prom_value,
)


def test_parse_prom_value():
    valid = {"status": "success", "data": {"result": [{"value": [1600000000, "42.5"]}]}}
    assert parse_prom_value(valid) == 42.5

    empty = {"status": "success", "data": {"result": []}}
    assert parse_prom_value(empty) is None

    nan_val = {
        "status": "success",
        "data": {"result": [{"value": [1600000000, "NaN"]}]},
    }
    assert parse_prom_value(nan_val) is None

    err = {"status": "error"}
    assert parse_prom_value(err) is None


def test_evaluate_healthy():
    metrics = {
        "api_up": 1.0,
        "success_rate": 0.95,
        "p95_latency_s": 0.25,
        "error_rate": 0.0,
        "brightness_drift_ratio": 0.05,
        "width_drift_ratio": 0.02,
    }
    t = Thresholds()
    res = evaluate(metrics, t, api_health={"model_loaded": True})
    assert res["healthy"] is True
    assert res["drift_detected"] is False
    assert len(res["violations"]) == 0


def test_evaluate_drift_detected():
    metrics = {
        "api_up": 1.0,
        "success_rate": 0.40,  # Below 0.5 threshold
        "brightness_drift_ratio": 0.45,  # Above 0.3 threshold
        "predictions_15m": 120.0,  # Enough volume for the statistics to count
    }
    t = Thresholds()
    res = evaluate(metrics, t, api_health={"model_loaded": True})
    assert res["healthy"] is False
    assert res["drift_detected"] is True
    v_metrics = [v["metric"] for v in res["violations"]]
    assert "success_rate" in v_metrics
    assert "brightness_drift_ratio" in v_metrics


def test_evaluate_ignores_statistics_on_low_volume():
    # One cold-start request among three pushes p95 to the 10 s bucket.
    metrics = {
        "api_up": 1.0,
        "p95_latency_s": 10.0,
        "success_rate": 0.0,
        "predictions_15m": 3.0,
    }
    res = evaluate(metrics, Thresholds(), api_health={"model_loaded": True})
    assert res["healthy"] is True
    assert res["drift_detected"] is False


def test_evaluate_flags_latency_with_enough_volume():
    metrics = {"api_up": 1.0, "p95_latency_s": 4.2, "predictions_15m": 50.0}
    res = evaluate(metrics, Thresholds(), api_health={"model_loaded": True})
    assert [v["metric"] for v in res["violations"]] == ["p95_latency_s"]


def test_low_volume_still_reports_availability_problems():
    metrics = {"api_up": 0.0, "predictions_15m": 0.0}
    res = evaluate(metrics, Thresholds(), api_health={"model_loaded": False})
    assert {v["metric"] for v in res["violations"]} == {"api_up", "model_loaded"}


def test_evaluate_api_down():
    metrics = {"api_up": 0.0}
    t = Thresholds()
    res = evaluate(metrics, t, api_health={"model_loaded": False})
    assert res["healthy"] is False
    assert any(v["metric"] == "api_up" for v in res["violations"])


def test_format_summary():
    report = {
        "metrics": {"success_rate": 0.9},
        "evaluation": {"healthy": True, "drift_detected": False, "violations": []},
    }
    s = format_summary(report)
    assert "LPR monitoring summary" in s
    assert "healthy" in s


def test_airflow_dag_loads_cleanly():
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="dags", include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
    assert "lpr_monitoring_pipeline" in dagbag.dags
    dag = dagbag.dags["lpr_monitoring_pipeline"]
    assert len(dag.tasks) > 5
