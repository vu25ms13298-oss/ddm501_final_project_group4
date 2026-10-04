#!/usr/bin/env python3
"""Train OCR model with MLflow experiment tracking.

Wraps the existing training pipeline and logs parameters, metrics,
and model artifacts to an MLflow tracking server.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (
    CHAR_CLASSES,
    generate_plate_style_synthetic_chars,
    generate_synthetic_chars,
    save_models,
)

from src.model_registry import (
    PRODUCTION_ALIAS,
    champion_accuracy,
    log_and_register,
    should_promote,
)
from scripts.train_ocr_model import (
    FEATURE_EXTRACTORS,
    build_classifier,
    collect_labeled_char_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train OCR model with MLflow tracking.")
    p.add_argument("--feature", choices=list(FEATURE_EXTRACTORS.keys()), default="hog")
    p.add_argument(
        "--classifier",
        choices=["svm", "logistic", "knn", "random_forest"],
        default="svm",
    )
    p.add_argument("--samples-per-class", type=int, default=250)
    p.add_argument("--plate-style-samples-per-class", type=int, default=400)
    p.add_argument("--no-emnist", action="store_true", default=True)
    p.add_argument(
        "--labeled-char-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "characters" / "labeled",
    )
    p.add_argument("--real-augmentations", type=int, default=60)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "models" / "ocr_hog_svm"
    )
    p.add_argument("--experiment-name", type=str, default="lpr-ocr-training")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning with cross-validation",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="Promote the registered version to @production only above this",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    char_list = list(CHAR_CLASSES)
    feature_fn = FEATURE_EXTRACTORS[args.feature]

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"{args.feature}_{args.classifier}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("feature_method", args.feature)
        mlflow.log_param("classifier", args.classifier)
        mlflow.log_param("samples_per_class", args.samples_per_class)
        mlflow.log_param(
            "plate_style_samples_per_class", args.plate_style_samples_per_class
        )
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("num_classes", len(char_list))
        mlflow.log_param("char_classes", "".join(char_list))

        # ---- Generate data ----
        print(f"Generating synthetic characters: {args.samples_per_class}/class")
        x_syn, y_syn = generate_synthetic_chars(
            char_classes=CHAR_CLASSES,
            samples_per_class=args.samples_per_class,
            img_size=64,
        )

        datasets_x = [x_syn]
        datasets_y = [y_syn]

        if args.plate_style_samples_per_class > 0:
            print(
                f"Generating plate-style synthetic: {args.plate_style_samples_per_class}/class"
            )
            x_plate, y_plate = generate_plate_style_synthetic_chars(
                char_classes=CHAR_CLASSES,
                samples_per_class=args.plate_style_samples_per_class,
                img_size=96,
            )
            datasets_x.append(x_plate)
            datasets_y.append(y_plate)

        x_real, y_real = collect_labeled_char_dir(
            args.labeled_char_dir,
            char_list,
            args.real_augmentations,
        )
        if len(x_real) > 0:
            datasets_x.append(x_real)
            datasets_y.append(y_real)
            mlflow.log_param("real_samples", len(x_real))

        all_x = np.concatenate(datasets_x, axis=0)
        all_y = np.concatenate(datasets_y, axis=0)
        mlflow.log_param("total_samples", len(all_x))

        # ---- Feature extraction ----
        print(f"Extracting {args.feature} features from {len(all_x)} images...")
        t0 = time.time()
        features = feature_fn(all_x)
        feat_time = time.time() - t0
        mlflow.log_metric("feature_extraction_time_sec", round(feat_time, 2))
        mlflow.log_param("feature_dim", features.shape[1])

        # ---- Train/test split ----
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            all_y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=all_y,
        )
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ---- Cross-validation & hyperparameter tuning ----
        if args.tune and args.classifier == "svm":
            mlflow.log_param("tuning", True)
            mlflow.log_param("cv_folds", args.cv_folds)
            param_grid = {
                "C": [1, 5, 10, 50],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf"],
            }
            print(f"Running GridSearchCV with {args.cv_folds}-fold CV...")
            from sklearn.svm import SVC

            grid = GridSearchCV(
                SVC(random_state=args.seed),
                param_grid,
                cv=args.cv_folds,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True,
            )
            t0 = time.time()
            grid.fit(X_train_s, y_train)
            train_time = time.time() - t0
            clf = grid.best_estimator_
            mlflow.log_param("best_C", grid.best_params_["C"])
            mlflow.log_param("best_gamma", grid.best_params_["gamma"])
            mlflow.log_metric("cv_best_f1_macro", round(grid.best_score_, 4))
            print(f"Best params: {grid.best_params_}, CV F1={grid.best_score_:.4f}")
        else:
            if args.tune:
                mlflow.log_param("tuning", True)
                mlflow.log_param("cv_folds", args.cv_folds)

            clf = build_classifier(args.classifier, args.seed)

            if args.tune:
                cv_scores = cross_val_score(
                    clf,
                    X_train_s,
                    y_train,
                    cv=args.cv_folds,
                    scoring="f1_macro",
                )
                mlflow.log_metric("cv_mean_f1_macro", round(cv_scores.mean(), 4))
                mlflow.log_metric("cv_std_f1_macro", round(cv_scores.std(), 4))
                print(f"CV F1 (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

            print(f"Training {args.classifier}...")
            t0 = time.time()
            clf.fit(X_train_s, y_train)
            train_time = time.time() - t0

        mlflow.log_metric("training_time_sec", round(train_time, 2))

        # ---- Evaluate ----
        y_pred = clf.predict(X_test_s)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        macro_precision = precision_score(y_test, y_pred, average="macro")
        macro_recall = recall_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("macro_f1", round(macro_f1, 4))
        mlflow.log_metric("macro_precision", round(macro_precision, 4))
        mlflow.log_metric("macro_recall", round(macro_recall, 4))

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        # Log classification report as artifact
        report = classification_report(y_test, y_pred, target_names=char_list)
        report_path = args.output_dir / "classification_report.txt"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = args.output_dir / "confusion_matrix.csv"
        np.savetxt(str(cm_path), cm, delimiter=",", fmt="%d")
        mlflow.log_artifact(str(cm_path))

        # ---- Save model locally ----
        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "synthetic_per_class": int(args.samples_per_class),
            "plate_style_synthetic_per_class": int(args.plate_style_samples_per_class),
        }
        save_models(
            clf,
            scaler,
            str(args.output_dir),
            feature_method=args.feature,
            classifier_name=args.classifier,
            metrics=metrics,
            feature_dim=features.shape[1],
        )
        meta_path = args.output_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        mlflow.log_artifact(str(meta_path))

        # ---- Register in MLflow (Pipeline: scaler -> classifier) ----
        champion_acc = champion_accuracy()
        promote = should_promote(accuracy, args.min_accuracy, champion_acc)
        registered = log_and_register(clf, scaler, metadata, promote=promote)
        if promote:
            print(
                f"Promoted {registered['model_name']} v{registered['model_version']} "
                f"to @{PRODUCTION_ALIAS}"
            )
        else:
            print(
                f"Registered v{registered['model_version']} but NOT promoted: "
                f"accuracy {accuracy:.4f}, gate {args.min_accuracy}, "
                f"champion {champion_acc}"
            )

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"MLflow experiment: {args.experiment_name}")
        print(f"Model registered as: {registered['model_name']}")


if __name__ == "__main__":
    main()
