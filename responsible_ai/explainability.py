#!/usr/bin/env python3
"""Model explainability analysis using SHAP and LIME.

Generates explanations for the HOG+SVM OCR classifier to understand
which HOG features contribute most to character classification decisions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import CHAR_CLASSES, generate_synthetic_chars
from src.features import extract_hog_features


def load_model(models_dir: str):
    clf_path = os.path.join(models_dir, "classifier.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if not os.path.exists(clf_path):
        clf_path = os.path.join(models_dir, "svm_classifier.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler


def generate_background_data(n_per_class: int = 10):
    """Generate a small background dataset for SHAP."""
    x_imgs, y_labels = generate_synthetic_chars(
        char_classes=CHAR_CLASSES,
        samples_per_class=n_per_class,
        img_size=64,
    )
    features = extract_hog_features(x_imgs)
    return features, y_labels, x_imgs


def run_shap_analysis(
    clf,
    scaler,
    output_dir: str,
    n_background: int = 10,
    n_explain: int = 50,
):
    """Run SHAP analysis on the OCR classifier."""
    import shap

    print("Generating background data for SHAP...")
    bg_features, bg_labels, _ = generate_background_data(n_per_class=n_background)
    bg_scaled = scaler.transform(bg_features)

    print("Generating explanation samples...")
    exp_features, exp_labels, exp_imgs = generate_background_data(n_per_class=2)
    exp_scaled = scaler.transform(exp_features)

    sample_idx = np.random.choice(
        len(exp_scaled), min(n_explain, len(exp_scaled)), replace=False
    )
    X_explain = exp_scaled[sample_idx]
    _ = exp_labels[sample_idx]

    print(f"Computing SHAP values for {len(X_explain)} samples...")
    bg_sample = shap.sample(bg_scaled, min(100, len(bg_scaled)))
    explainer = shap.KernelExplainer(clf.decision_function, bg_sample)
    shap_values = explainer.shap_values(X_explain, nsamples=100)

    os.makedirs(output_dir, exist_ok=True)

    # Global feature importance (mean |SHAP| across all classes)
    if isinstance(shap_values, list):
        all_shap = np.array(shap_values)  # (n_classes, n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(all_shap), axis=(0, 1))
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    top_k = 30
    top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]

    plt.figure(figsize=(12, 6))
    plt.barh(
        range(top_k),
        mean_abs_shap[top_indices][::-1],
        color="#4C72B0",
    )
    plt.yticks(range(top_k), [f"HOG_{i}" for i in top_indices[::-1]])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 30 Most Important HOG Features (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=150)
    plt.close()

    # Per-class importance for selected classes
    sample_classes = ["0", "A", "B", "5", "S"]
    class_indices = [
        list(CHAR_CLASSES).index(c) for c in sample_classes if c in CHAR_CLASSES
    ]

    if isinstance(shap_values, list) and len(class_indices) > 0:
        fig, axes = plt.subplots(
            1, len(class_indices), figsize=(5 * len(class_indices), 5)
        )
        if len(class_indices) == 1:
            axes = [axes]
        for ax, cls_idx in zip(axes, class_indices):
            cls_shap = np.mean(np.abs(shap_values[cls_idx]), axis=0)
            top_cls = np.argsort(cls_shap)[-15:][::-1]
            ax.barh(range(15), cls_shap[top_cls][::-1])
            ax.set_yticks(range(15))
            ax.set_yticklabels([f"HOG_{i}" for i in top_cls[::-1]], fontsize=8)
            ax.set_title(f"Class '{CHAR_CLASSES[cls_idx]}'")
        plt.suptitle("Per-Class SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_per_class.png"), dpi=150)
        plt.close()

    results = {
        "method": "SHAP (KernelExplainer)",
        "n_background": len(bg_sample),
        "n_explained": len(X_explain),
        "top_features": [
            {"feature": f"HOG_{int(idx)}", "importance": float(mean_abs_shap[idx])}
            for idx in top_indices
        ],
    }
    with open(os.path.join(output_dir, "shap_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"SHAP analysis saved to {output_dir}")
    return results


def run_lime_analysis(
    clf,
    scaler,
    output_dir: str,
    n_explain: int = 10,
):
    """Run LIME analysis on individual predictions."""
    from lime.lime_tabular import LimeTabularExplainer

    print("Generating data for LIME...")
    bg_features, bg_labels, _ = generate_background_data(n_per_class=15)
    bg_scaled = scaler.transform(bg_features)

    exp_features, exp_labels, exp_imgs = generate_background_data(n_per_class=2)
    exp_scaled = scaler.transform(exp_features)

    feature_names = [f"HOG_{i}" for i in range(bg_scaled.shape[1])]
    class_names = list(CHAR_CLASSES)

    explainer = LimeTabularExplainer(
        bg_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    sample_idx = np.random.choice(
        len(exp_scaled), min(n_explain, len(exp_scaled)), replace=False
    )
    all_explanations = []

    for i, idx in enumerate(sample_idx):
        instance = exp_scaled[idx]
        true_label = int(exp_labels[idx])
        pred_label = int(clf.predict(instance.reshape(1, -1))[0])

        exp = explainer.explain_instance(
            instance,
            (
                clf.predict_proba
                if hasattr(clf, "predict_proba")
                else clf.decision_function
            ),
            num_features=15,
            top_labels=3,
        )

        fig = exp.as_pyplot_figure(label=pred_label)
        fig.set_size_inches(10, 5)
        plt.title(
            f"LIME: True='{class_names[true_label]}', Pred='{class_names[pred_label]}'"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lime_sample_{i}.png"), dpi=150)
        plt.close()

        exp_list = exp.as_list(label=pred_label)
        all_explanations.append(
            {
                "sample_index": int(idx),
                "true_class": class_names[true_label],
                "predicted_class": class_names[pred_label],
                "correct": true_label == pred_label,
                "top_features": [
                    {"feature": f, "weight": float(w)} for f, w in exp_list
                ],
            }
        )

    results = {
        "method": "LIME (LimeTabularExplainer)",
        "n_explained": len(all_explanations),
        "explanations": all_explanations,
    }
    with open(os.path.join(output_dir, "lime_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"LIME analysis saved to {output_dir}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run model explainability analysis.")
    parser.add_argument(
        "--models-dir", type=str, default=str(PROJECT_ROOT / "models" / "ocr_hog_svm")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "responsible_ai" / "results"),
    )
    parser.add_argument("--method", choices=["shap", "lime", "both"], default="both")
    args = parser.parse_args()

    clf, scaler = load_model(args.models_dir)
    np.random.seed(42)

    if args.method in ("shap", "both"):
        run_shap_analysis(clf, scaler, os.path.join(args.output_dir, "shap"))

    if args.method in ("lime", "both"):
        run_lime_analysis(clf, scaler, os.path.join(args.output_dir, "lime"))


if __name__ == "__main__":
    main()
