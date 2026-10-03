#!/usr/bin/env python3
"""Evaluate end-to-end LPR predictions against plate-level labels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import LPRPipeline


def read_rgb_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype("float32")
        alpha = image[:, :, 3:4].astype("float32") / 255.0
        white = 255.0 * (1.0 - alpha)
        image = (bgr * alpha + white).astype("uint8")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_text(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def parse_bool(value, default=False):
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "plate", "crop"}


def parse_model_arg(value: str):
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
        return list(reader)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate exact plate accuracy for one or more OCR model directories."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "characters" / "plate_labels_eval.csv",
        help="CSV with image,label and optional plate_crop columns.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec as name=path. May be repeated.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "yolo_runs"
        / "license_plate_yolov8_bbox"
        / "weights"
        / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "ocr_eval_exact_plate_accuracy.csv",
    )
    parser.add_argument(
        "--min-exact-accuracy",
        type=float,
        default=None,
        help="Exit non-zero if any model's exact plate accuracy is below this",
    )
    parser.add_argument(
        "--min-char-accuracy",
        type=float,
        default=None,
        help="Exit non-zero if any model's normalized char accuracy is below this",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    model_specs = args.model or [
        "ocr_hog_svm=models/ocr_hog_svm",
    ]
    # Only use YOLO when weights exist; otherwise the contour fallback is used.
    yolo_path = str(args.yolo_model) if args.yolo_model.exists() else None
    failures = []

    output_rows = []
    for model_spec in model_specs:
        model_name, model_dir = parse_model_arg(model_spec)
        model_dir = resolve_path(str(model_dir))
        pipeline = LPRPipeline(models_dir=str(model_dir))

        exact = 0
        edit_total = 0
        norm_total = 0
        attempted = 0

        for row in rows:
            image_path = resolve_path(row["image"])
            label = normalize_text(row["label"])
            assume_plate_crop = parse_bool(row.get("plate_crop"), default=False)

            image_rgb = read_rgb_image(image_path)
            if image_rgb is None:
                prediction = ""
                success = False
                char_count = 0
            else:
                result = pipeline.recognize(
                    image_rgb,
                    yolo_model_path=None if assume_plate_crop else yolo_path,
                    assume_plate_crop=assume_plate_crop,
                    verbose=False,
                )
                prediction = normalize_text(result.get("plate_string", ""))
                success = bool(result.get("success", False))
                char_count = int(
                    result.get("char_count", len(result.get("char_images") or []))
                )

            dist = levenshtein(label, prediction)
            denom = max(len(label), len(prediction), 1)
            is_correct = label == prediction
            exact += int(is_correct)
            edit_total += dist
            norm_total += denom
            attempted += 1

            output_rows.append(
                {
                    "model": model_name,
                    "image": row["image"],
                    "label": label,
                    "prediction": prediction,
                    "correct": str(is_correct),
                    "success": str(success),
                    "char_count": str(char_count),
                    "plate_crop": str(assume_plate_crop),
                    "edit_distance": str(dist),
                    "normalized_char_accuracy": f"{1.0 - dist / denom:.4f}",
                }
            )

        exact_acc = exact / attempted if attempted else 0.0
        char_acc = 1.0 - edit_total / norm_total if norm_total else 0.0
        print(
            f"{model_name}: exact={exact}/{attempted} "
            f"({exact_acc:.2%}), normalized_char_accuracy={char_acc:.2%}"
        )
        if args.min_exact_accuracy is not None and exact_acc < args.min_exact_accuracy:
            failures.append(f"{model_name}: exact {exact_acc:.2%}")
        if args.min_char_accuracy is not None and char_acc < args.min_char_accuracy:
            failures.append(f"{model_name}: char {char_acc:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "image",
        "label",
        "prediction",
        "correct",
        "success",
        "char_count",
        "plate_crop",
        "edit_distance",
        "normalized_char_accuracy",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved detail CSV: {args.output}")
    if failures:
        print("Quality gate FAILED: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
