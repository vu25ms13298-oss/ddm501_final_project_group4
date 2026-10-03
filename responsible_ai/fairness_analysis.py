#!/usr/bin/env python3
"""Fairness analysis for the OCR classifier.

Analyzes potential biases in character recognition across different
character groups (digits vs letters, visually similar characters).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (
    CHAR_CLASSES,
    generate_plate_style_synthetic_chars,
    generate_synthetic_chars,
)
from responsible_ai.explainability import load_model

EVAL_SEED = 20261003


VISUALLY_SIMILAR_GROUPS = {
    "0-O-D": ["0", "D"],
    "1-I-L": ["1", "L"],
    "5-S": ["5", "S"],
    "8-B": ["8", "B"],
    "6-G": ["6", "G"],
    "2-Z": ["2", "Z"],
}


def analyze_class_group_fairness(y_true, y_pred, char_list):
    """Analyze accuracy disparity between digits and letters."""
    digit_mask = np.array([char_list[int(y)].isdigit() for y in y_true])
    letter_mask = ~digit_mask

    results = {}

    if digit_mask.sum() > 0:
        digit_acc = accuracy_score(y_true[digit_mask], y_pred[digit_mask])
        results["digits"] = {
            "accuracy": float(digit_acc),
            "count": int(digit_mask.sum()),
        }

    if letter_mask.sum() > 0:
        letter_acc = accuracy_score(y_true[letter_mask], y_pred[letter_mask])
        results["letters"] = {
            "accuracy": float(letter_acc),
            "count": int(letter_mask.sum()),
        }

    if "digits" in results and "letters" in results:
        disparity = abs(results["digits"]["accuracy"] - results["letters"]["accuracy"])
        results["disparity"] = float(disparity)
        results["fair"] = disparity < 0.10

    return results


def analyze_similar_char_confusion(y_true, y_pred, char_list):
    """Analyze confusion rates among visually similar characters."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(char_list)))
    results = {}

    for group_name, chars in VISUALLY_SIMILAR_GROUPS.items():
        indices = [char_list.index(c) for c in chars if c in char_list]
        if len(indices) < 2:
            continue

        group_confusion = {}
        for i in indices:
            for j in indices:
                if i != j:
                    total_i = cm[i].sum()
                    if total_i > 0:
                        confusion_rate = float(cm[i, j]) / total_i
                        group_confusion[f"{char_list[i]}->{char_list[j]}"] = {
                            "confusion_rate": round(confusion_rate, 4),
                            "count": int(cm[i, j]),
                            "total": int(total_i),
                        }

        results[group_name] = group_confusion

    return results


def analyze_per_class_fairness(y_true, y_pred, char_list):
    """Identify under-performing classes that may indicate bias."""
    per_class_acc = {}
    for i, char in enumerate(char_list):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        per_class_acc[char] = {
            "accuracy": float(acc),
            "count": int(mask.sum()),
        }

    accuracies = [v["accuracy"] for v in per_class_acc.values()]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    underperforming = {
        k: v for k, v in per_class_acc.items() if v["accuracy"] < mean_acc - 2 * std_acc
    }

    return {
        "per_class": per_class_acc,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "underperforming_classes": underperforming,
        "equity_gap": float(max(accuracies) - min(accuracies)) if accuracies else 0.0,
    }


def generate_fairness_report(output_dir: str, models_dir: str):
    """Generate the complete fairness analysis report."""
    clf, scaler, extract_fn = load_model(models_dir)
    char_list = list(CHAR_CLASSES)

    print("Generating evaluation data...")
    # Training uses seed 42; a different seed avoids re-generating training
    # samples. Plate-style glyphs are harder and closer to real segmented chars.
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    x_clean, y_clean = generate_synthetic_chars(
        char_classes=CHAR_CLASSES,
        samples_per_class=25,
        img_size=64,
    )
    x_plate, y_plate = generate_plate_style_synthetic_chars(
        char_classes=CHAR_CLASSES,
        samples_per_class=50,
    )
    x_imgs = np.concatenate([x_clean, x_plate])
    y_true = np.concatenate([y_clean, y_plate])
    features = extract_fn(x_imgs)
    X_scaled = scaler.transform(features)
    y_pred = clf.predict(X_scaled)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Group fairness (digits vs letters)
    group_results = analyze_class_group_fairness(y_true, y_pred, char_list)

    # 2. Similar character confusion
    confusion_results = analyze_similar_char_confusion(y_true, y_pred, char_list)

    # 3. Per-class fairness
    class_results = analyze_per_class_fairness(y_true, y_pred, char_list)

    # Generate plots
    # Plot 1: Digit vs Letter accuracy
    if "digits" in group_results and "letters" in group_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = ["Digits", "Letters"]
        accs = [
            group_results["digits"]["accuracy"],
            group_results["letters"]["accuracy"],
        ]
        colors = ["#4C72B0", "#DD8452"]
        ax.bar(groups, accs, color=colors, width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Digit vs Letter Recognition Accuracy")
        for i, v in enumerate(accs):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")
        ax.axhline(
            y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "digit_vs_letter_accuracy.png"), dpi=150)
        plt.close()

    # Plot 2: Per-class accuracy distribution
    per_class = class_results["per_class"]
    chars = list(per_class.keys())
    accs = [per_class[c]["accuracy"] for c in chars]

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = ["#4C72B0" if c.isdigit() else "#DD8452" for c in chars]
    ax.bar(chars, accs, color=bar_colors, width=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Character Class")
    ax.set_title("Per-Class Recognition Accuracy")
    ax.axhline(
        y=class_results["mean_accuracy"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {class_results['mean_accuracy']:.2%}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=150)
    plt.close()

    # Plot 3: Confusion matrix heatmap for similar chars
    cm = confusion_matrix(y_true, y_pred, labels=range(len(char_list)))
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(char_list)))
    ax.set_yticks(range(len(char_list)))
    ax.set_xticklabels(char_list, fontsize=7)
    ax.set_yticklabels(char_list, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"), dpi=150)
    plt.close()

    report = {
        "group_fairness": group_results,
        "similar_char_confusion": confusion_results,
        "per_class_analysis": class_results,
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }

    with open(os.path.join(output_dir, "fairness_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nFairness Report Summary:")
    print(f"  Overall accuracy: {report['overall_accuracy']:.2%}")
    if "digits" in group_results:
        print(f"  Digit accuracy:  {group_results['digits']['accuracy']:.2%}")
    if "letters" in group_results:
        print(f"  Letter accuracy: {group_results['letters']['accuracy']:.2%}")
    if "disparity" in group_results:
        print(f"  Disparity:       {group_results['disparity']:.2%}")
        print(f"  Fair (< 10%):    {group_results['fair']}")
    print(f"  Equity gap:      {class_results['equity_gap']:.2%}")
    if class_results["underperforming_classes"]:
        print(
            f"  Underperforming: {list(class_results['underperforming_classes'].keys())}"
        )

    print(f"\nReport saved to {output_dir}")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run fairness analysis.")
    parser.add_argument(
        "--models-dir", type=str, default=str(PROJECT_ROOT / "models" / "ocr_hog_svm")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "responsible_ai" / "results" / "fairness"),
    )
    args = parser.parse_args()

    generate_fairness_report(args.output_dir, args.models_dir)


if __name__ == "__main__":
    main()
