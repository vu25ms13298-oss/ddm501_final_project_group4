"""
LPR OCR Training Pipeline — Airflow DAG

generate_data -> validate_data -> extract_features -> train -> evaluate
  -> register_model -> reload_api

Trains the HOG+SVM OCR classifier, logs everything to MLflow, registers the
model as a Pipeline(scaler -> SVM) and promotes it to @production only when it
beats both MIN_ACCURACY and the current production model (champion/challenger).
The API (MODEL_SOURCE=mlflow) is then asked to reload the production model.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT / "models" / "ocr_hog_svm"
DATA_DIR = PROJECT / "data" / "training_runs"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = "lpr-ocr-training"
API_URL = os.getenv("LPR_API_URL", "http://api:8000")
API_ADMIN_TOKEN = os.getenv("API_ADMIN_TOKEN", "")

CHAR_CLASSES_STR = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
MIN_ACCURACY = 0.90
SAMPLES_PER_CLASS = 100
PLATE_STYLE_SAMPLES = 150
TEST_SIZE = 0.2
SEED = 42


def run_dir(ds: str) -> Path:
    d = DATA_DIR / ds
    d.mkdir(parents=True, exist_ok=True)
    return d


@dag(
    dag_id="lpr_training_pipeline",
    description="LPR OCR: generate data, extract HOG, train SVM, evaluate, register",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    # Airflow pauses new DAGs by default, which silently disables the weekly
    # retraining schedule until someone unpauses it in the UI.
    is_paused_upon_creation=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["ddm501", "lpr", "ocr"],
)
def lpr_training_pipeline():

    @task
    def generate_data(ds: str = None) -> dict:
        """Generate synthetic character images for training."""
        from src.classifier import (
            CHAR_CLASSES,
            generate_plate_style_synthetic_chars,
            generate_synthetic_chars,
        )

        np.random.seed(SEED)
        out = run_dir(ds)

        x_syn, y_syn = generate_synthetic_chars(
            char_classes=CHAR_CLASSES,
            samples_per_class=SAMPLES_PER_CLASS,
            img_size=64,
        )
        log.info("Generated %d synthetic chars", len(x_syn))

        x_plate, y_plate = generate_plate_style_synthetic_chars(
            char_classes=CHAR_CLASSES,
            samples_per_class=PLATE_STYLE_SAMPLES,
            img_size=96,
        )
        log.info("Generated %d plate-style chars", len(x_plate))

        all_x = np.concatenate([x_syn, x_plate], axis=0)
        all_y = np.concatenate([y_syn, y_plate], axis=0)

        np.save(str(out / "images.npy"), all_x)
        np.save(str(out / "labels.npy"), all_y)

        return {
            "total_samples": len(all_x),
            "num_classes": len(CHAR_CLASSES),
            "synthetic": len(x_syn),
            "plate_style": len(x_plate),
            "path": str(out),
        }

    @task
    def validate_data(meta: dict, ds: str = None) -> dict:
        """Validate generated data quality."""
        out = run_dir(ds)
        images = np.load(str(out / "images.npy"))
        labels = np.load(str(out / "labels.npy"))

        n_classes = len(set(labels.tolist()))
        samples_per_class = {}
        for label in set(labels.tolist()):
            samples_per_class[str(label)] = int((labels == label).sum())

        min_count = min(samples_per_class.values())
        max_count = max(samples_per_class.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        has_nan = bool(np.isnan(images).any())
        all_valid_range = bool((images >= 0).all() and (images <= 255).all())

        report = {
            "n_samples": len(images),
            "n_classes": n_classes,
            "min_per_class": min_count,
            "max_per_class": max_count,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "has_nan": has_nan,
            "valid_range": all_valid_range,
        }

        (out / "validation_report.json").write_text(json.dumps(report, indent=2))

        if has_nan:
            raise AirflowFailException("Data contains NaN values")
        if not all_valid_range:
            raise AirflowFailException("Image values out of [0, 255] range")
        if n_classes < 31:
            raise AirflowFailException(f"Only {n_classes} classes, expected 31")

        log.info(
            "Validation passed: %d samples, %d classes, imbalance=%.2f",
            len(images),
            n_classes,
            imbalance_ratio,
        )
        return report

    @task
    def extract_features(validation: dict, ds: str = None) -> dict:
        """Extract HOG features from character images."""
        from src.features import extract_hog_features

        out = run_dir(ds)
        images = np.load(str(out / "images.npy"))

        log.info("Extracting HOG features from %d images...", len(images))
        features = extract_hog_features(images)

        np.save(str(out / "features.npy"), features)
        log.info("Feature shape: %s", features.shape)

        return {
            "n_samples": len(features),
            "feature_dim": features.shape[1],
        }

    @task
    def train(feature_meta: dict, ds: str = None, **context) -> dict:
        """Train SVM classifier and log to MLflow."""
        import mlflow
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        out = run_dir(ds)
        features = np.load(str(out / "features.npy"))
        labels = np.load(str(out / "labels.npy"))

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        dag_run = context.get("dag_run")
        trigger_reason = (dag_run.conf or {}).get("reason", "scheduled") if dag_run else "scheduled"

        with mlflow.start_run(run_name=f"lpr-ocr-{ds}") as run:
            mlflow.set_tag("trigger_reason", trigger_reason)
            mlflow.log_param("feature_method", "hog")
            mlflow.log_param("classifier", "svm")
            mlflow.log_param("samples_per_class", SAMPLES_PER_CLASS)
            mlflow.log_param("plate_style_samples", PLATE_STYLE_SAMPLES)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("feature_dim", feature_meta["feature_dim"])

            clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)

            cv_scores = cross_val_score(
                clf, X_train_s, y_train, cv=3, scoring="f1_macro"
            )
            mlflow.log_metric("cv_mean_f1", round(float(cv_scores.mean()), 4))
            mlflow.log_metric("cv_std_f1", round(float(cv_scores.std()), 4))
            log.info("CV F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)

            np.save(str(out / "y_test.npy"), y_test)
            np.save(str(out / "y_pred.npy"), y_pred)

            import joblib

            joblib.dump(clf, str(out / "svm_classifier.pkl"))
            joblib.dump(scaler, str(out / "feature_scaler.pkl"))

            return {
                "mlflow_run_id": run.info.run_id,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }

    @task
    def evaluate(train_meta: dict, ds: str = None) -> dict:
        """Evaluate trained model and log metrics to MLflow."""
        import mlflow
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        out = run_dir(ds)
        y_test = np.load(str(out / "y_test.npy"))
        y_pred = np.load(str(out / "y_pred.npy"))
        char_list = list(CHAR_CLASSES_STR)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        report = classification_report(y_test, y_pred, target_names=char_list)
        (out / "classification_report.txt").write_text(report)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        run_id = train_meta["mlflow_run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", round(accuracy, 4))
            mlflow.log_metric("macro_f1", round(macro_f1, 4))

        log.info("Accuracy: %.4f, Macro F1: %.4f", accuracy, macro_f1)

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "mlflow_run_id": run_id,
        }

    @task
    def register_model(eval_meta: dict, feature_meta: dict, ds: str = None) -> dict:
        """Register the model; promote to @production if it beats the champion."""
        import joblib
        import mlflow

        from src.classifier import CHAR_CLASSES
        from src.model_registry import (
            PRODUCTION_ALIAS,
            REGISTERED_MODEL_NAME,
            champion_accuracy,
            log_and_register,
            should_promote,
        )

        accuracy = eval_meta["accuracy"]
        run_id = eval_meta["mlflow_run_id"]
        out = run_dir(ds)

        if accuracy < MIN_ACCURACY:
            log.warning(
                "Accuracy %.4f below threshold %.2f. Model NOT registered.",
                accuracy,
                MIN_ACCURACY,
            )
            return {"registered": False, "promoted": False}

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        champion_acc = champion_accuracy(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        if champion_acc is None:
            log.info(
                "No current @%s model; this run becomes champion", PRODUCTION_ALIAS
            )
        promote = should_promote(accuracy, MIN_ACCURACY, champion_acc)

        metadata = {
            "char_classes": list(CHAR_CLASSES),
            "char_size": 32,
            "feature_method": "hog",
            "classifier": "svm",
            "feature_dim": feature_meta["feature_dim"],
            "metrics": {
                "accuracy": accuracy,
                "macro_f1": eval_meta["macro_f1"],
            },
        }
        clf = joblib.load(str(out / "svm_classifier.pkl"))
        scaler = joblib.load(str(out / "feature_scaler.pkl"))
        with mlflow.start_run(run_id=run_id):
            registered = log_and_register(clf, scaler, metadata, promote=promote)

        log.info(
            "Registered %s v%s (accuracy=%.4f, champion=%s, promoted=%s)",
            registered["model_name"],
            registered["model_version"],
            accuracy,
            champion_acc,
            promote,
        )
        summary = {
            "ds": ds,
            "accuracy": accuracy,
            "macro_f1": eval_meta["macro_f1"],
            "champion_accuracy": champion_acc,
            "mlflow_run_id": run_id,
            "registered": True,
            **registered,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    @task
    def reload_api(summary: dict) -> str:
        """Ask the serving API to pick up a newly promoted production model."""
        import urllib.request

        if not summary.get("promoted"):
            return "Nothing promoted; API reload skipped"
        if not API_ADMIN_TOKEN:
            return "API_ADMIN_TOKEN not set; reload the API manually"

        req = urllib.request.Request(
            f"{API_URL}/model/reload",
            method="POST",
            headers={"X-Admin-Token": API_ADMIN_TOKEN},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode()
        except Exception as exc:
            # The model is already promoted; a failed reload must not fail the run.
            log.warning("API reload failed: %s", exc)
            return f"API reload failed: {exc}"
        log.info("API reloaded: %s", body)
        return body

    data = generate_data()
    validated = validate_data(data)
    features = extract_features(validated)
    trained = train(features)
    evaluated = evaluate(trained)
    summary = register_model(evaluated, features)
    reload_api(summary)


lpr_training_pipeline()
