"""Pure helpers for the lpr_monitoring_pipeline DAG.

Kept free of Airflow imports so the threshold logic can be unit tested in CI
(requirements-ci.txt does not install Airflow).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

# PromQL evaluated by the DAG. Windows are 15m to match the default schedule;
# the drift baseline is 24h so it is meaningful on a freshly started stack.
PROMQL: dict[str, str] = {
    "api_up": 'max(up{job="lpr-api"})',
    "success_rate": (
        'sum(rate(plate_recognition_result_total{result="success"}[15m]))'
        " / sum(rate(plate_recognition_result_total[15m]))"
    ),
    "p95_latency_s": (
        "histogram_quantile(0.95, sum by (le) "
        "(rate(model_prediction_latency_seconds_bucket[15m])))"
    ),
    "error_rate": (
        'sum(rate(model_prediction_errors_total{error_type!="rate_limit_exceeded"}[15m]))'
    ),
    "brightness_drift_ratio": (
        "abs(1 - ((sum(rate(input_image_brightness_sum[15m]))"
        " / sum(rate(input_image_brightness_count[15m])))"
        " / (sum(rate(input_image_brightness_sum[24h]))"
        " / sum(rate(input_image_brightness_count[24h])))))"
    ),
    "width_drift_ratio": (
        "abs(1 - ((sum(rate(input_image_width_pixels_sum[15m]))"
        " / sum(rate(input_image_width_pixels_count[15m])))"
        " / (sum(rate(input_image_width_pixels_sum[24h]))"
        " / sum(rate(input_image_width_pixels_count[24h])))))"
    ),
    "rate_limited_15m": (
        'sum(increase(model_prediction_errors_total{error_type="rate_limit_exceeded"}[15m]))'
    ),
    "predictions_15m": "sum(increase(model_predictions_total[15m]))",
}

# Metrics whose violation means the *input data* (or outputs) drifted.
DRIFT_METRICS = {"success_rate", "brightness_drift_ratio", "width_drift_ratio"}

# Statistics that are meaningless on very few predictions (see min_predictions).
VOLUME_SENSITIVE_METRICS = DRIFT_METRICS | {"p95_latency_s"}


@dataclass(frozen=True)
class Thresholds:
    min_success_rate: float = 0.5
    max_p95_latency_s: float = 3.0
    max_error_rate: float = 0.1
    max_brightness_drift: float = 0.3
    max_width_drift: float = 0.5
    # Percentiles and ratios over a handful of requests are noise (e.g. one cold
    # start makes p95 = 10 s with 3 samples); judge them only above this volume.
    min_predictions: float = 20

    @classmethod
    def from_env(cls) -> "Thresholds":
        def _f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except ValueError:
                return default

        d = cls()
        return cls(
            min_success_rate=_f("MIN_SUCCESS_RATE", d.min_success_rate),
            max_p95_latency_s=_f("MAX_P95_LATENCY_S", d.max_p95_latency_s),
            max_error_rate=_f("MAX_ERROR_RATE", d.max_error_rate),
            max_brightness_drift=_f("MAX_BRIGHTNESS_DRIFT", d.max_brightness_drift),
            max_width_drift=_f("MAX_WIDTH_DRIFT", d.max_width_drift),
            min_predictions=_f("MIN_PREDICTIONS", d.min_predictions),
        )


def parse_prom_value(payload: dict) -> float | None:
    """Extracts a scalar from a Prometheus /api/v1/query response.

    Returns None for empty results or NaN/Inf (e.g. 0/0 when there was no
    traffic), so "no data" is never mistaken for a violation.
    """
    if not payload or payload.get("status") != "success":
        return None
    result = payload.get("data", {}).get("result") or []
    if not result:
        return None
    try:
        value = float(result[0]["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def evaluate(
    metrics: dict,
    thresholds: Thresholds,
    api_health: dict | None = None,
    canary: dict | None = None,
) -> dict:
    """Compares observed metrics with thresholds.

    Returns ``{"violations": [...], "drift_detected": bool, "healthy": bool}``
    where each violation is ``{"metric", "value", "threshold", "kind"}``.
    """
    t = thresholds
    rules = [
        ("success_rate", lambda v: v < t.min_success_rate, t.min_success_rate),
        ("p95_latency_s", lambda v: v > t.max_p95_latency_s, t.max_p95_latency_s),
        ("error_rate", lambda v: v > t.max_error_rate, t.max_error_rate),
        (
            "brightness_drift_ratio",
            lambda v: v > t.max_brightness_drift,
            t.max_brightness_drift,
        ),
        ("width_drift_ratio", lambda v: v > t.max_width_drift, t.max_width_drift),
    ]

    low_volume = (metrics.get("predictions_15m") or 0) < t.min_predictions
    violations = []
    for name, breached, threshold in rules:
        value = metrics.get(name)
        if low_volume and name in VOLUME_SENSITIVE_METRICS:
            continue
        if value is not None and breached(value):
            violations.append(
                {
                    "metric": name,
                    "value": round(float(value), 4),
                    "threshold": threshold,
                    "kind": "drift" if name in DRIFT_METRICS else "performance",
                }
            )

    if metrics.get("api_up") == 0:
        violations.append(
            {"metric": "api_up", "value": 0, "threshold": 1, "kind": "availability"}
        )
    if api_health is not None and not api_health.get("model_loaded", False):
        violations.append(
            {
                "metric": "model_loaded",
                "value": False,
                "threshold": True,
                "kind": "availability",
            }
        )
    if canary is not None and canary.get("status") == "error":
        violations.append(
            {
                "metric": "canary_prediction",
                "value": canary.get("detail"),
                "threshold": "HTTP 200",
                "kind": "availability",
            }
        )

    drift = any(v["kind"] == "drift" for v in violations)
    return {
        "violations": violations,
        "drift_detected": drift,
        "healthy": not violations,
        "thresholds": asdict(t),
    }


def format_summary(report: dict) -> str:
    """Human-readable multi-line summary for task logs."""
    lines = ["LPR monitoring summary", "-" * 40]
    for key, value in (report.get("metrics") or {}).items():
        shown = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"{key:<24} {shown}")
    evaluation = report.get("evaluation") or {}
    lines.append("-" * 40)
    lines.append(f"healthy        : {evaluation.get('healthy')}")
    lines.append(f"drift_detected : {evaluation.get('drift_detected')}")
    for v in evaluation.get("violations", []):
        lines.append(
            f"  VIOLATION {v['metric']}={v['value']} (threshold {v['threshold']})"
        )
    return "\n".join(lines)
