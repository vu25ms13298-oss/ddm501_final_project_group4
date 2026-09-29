#!/usr/bin/env python3
"""Pseudo-label OCR character crops from plate-detection data.

The plate dataset provides plate polygons/bboxes but not transcription text.
This script therefore builds a semi-supervised OCR dataset:

1. Warp each labeled plate polygon into a normalized plate crop.
2. Segment characters with the current LPR pipeline.
3. Use the current HOG-SVM OCR model as a teacher.
4. Save only high-confidence character predictions, capped per class.

The output folder is compatible with scripts/train_ocr_model.py:

    data/characters/labeled_plate_auto_20/A/*.png
    data/characters/labeled_plate_auto_20/0/*.png
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import CHAR_CLASSES
from src.pipeline import (
    LPRPipeline,
    PREFERRED_PLATE_RE,
    VALID_PLATE_RE,
    _resize_plate_for_ocr,
)
from src.preprocessing import deskew_plate_with_angle, rectify_plate_crop


def read_rgb_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype("float32")
        alpha = image[:, :, 3:4].astype("float32") / 255.0
        image = (bgr * alpha + 255.0 * (1.0 - alpha)).astype("uint8")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype="float32",
    )


def warp_quad(image: np.ndarray, points: np.ndarray):
    rect = order_quad_points(points)
    tl, tr, br, bl = rect
    dst_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    dst_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if dst_w < 45 or dst_h < 18:
        return None

    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def parse_plate_polygons(label_path: Path, image_shape):
    h, w = image_shape[:2]
    polygons = []
    if not label_path.exists():
        return polygons

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            values = [float(v) for v in parts[1:9]]
        except ValueError:
            continue
        pts = np.array(values, dtype=np.float32).reshape(4, 2)
        if np.max(pts) <= 1.5:
            pts[:, 0] *= w
            pts[:, 1] *= h
        polygons.append(pts)
    return polygons


def label_path_for_image(image_path: Path, images_root: Path, labels_root: Path):
    rel = image_path.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


def safe_stem(value: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def project_path(path: Path):
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def candidate_from_plate(pipeline: LPRPipeline, plate_crop):
    plate_crop = rectify_plate_crop(plate_crop)
    plate_crop = _resize_plate_for_ocr(plate_crop)
    deskewed_crop, deskew_angle = deskew_plate_with_angle(plate_crop)

    was_rotated, best = pipeline._evaluate_ocr_crop_variants(
        pipeline._build_primary_ocr_crop_variants(
            plate_crop,
            deskewed_crop,
            deskew_angle,
            allow_full_crop_rescue=True,
        )
    )
    if pipeline._needs_fallback(best):
        fallback_rotated, fallback_best = pipeline._evaluate_ocr_crop_variants(
            pipeline._build_fallback_ocr_crop_variants(plate_crop, deskewed_crop, deskew_angle)
        )
        if pipeline._candidate_key(fallback_best) > pipeline._candidate_key(best):
            was_rotated, best = fallback_rotated, fallback_best

    best["rotated_180"] = was_rotated
    return best


def raw_predictions_and_margins(pipeline: LPRPipeline, char_images):
    if not char_images:
        return [], []
    features = pipeline._extract_char_features(char_images)
    scaled = pipeline.scaler.transform(features)
    raw_preds = pipeline.svm_model.predict(scaled)
    raw_chars = [pipeline.char_classes[int(i)] for i in raw_preds]

    margins = []
    if hasattr(pipeline.svm_model, "decision_function"):
        scores = pipeline.svm_model.decision_function(scaled)
        for pred_idx, row in zip(raw_preds, scores):
            pred_idx = int(pred_idx)
            others = np.delete(row, pred_idx)
            margins.append(float(row[pred_idx] - np.max(others)))
    else:
        margins = [0.0] * len(raw_chars)
    return raw_chars, margins


def is_plate_like(text: str):
    return bool(PREFERRED_PLATE_RE.match(text) or VALID_PLATE_RE.match(text))


def save_contact_sheet(output_dir: Path, counts: Counter, sheet_path: Path):
    cell_w, cell_h = 86, 58
    header_h = 28
    cols = 10
    rows = int(np.ceil(len(CHAR_CLASSES) / cols))
    width = cols * cell_w
    height = rows * (header_h + cell_h)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, ch in enumerate(CHAR_CLASSES):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * (header_h + cell_h)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + header_h + cell_h - 1], outline=(210, 210, 210))
        draw.text((x0 + 4, y0 + 4), f"{ch}: {counts.get(ch, 0)}", fill=(0, 0, 0), font=font)
        samples = sorted((output_dir / ch).glob("*.png"))[:4]
        for sample_idx, sample_path in enumerate(samples):
            img = Image.open(sample_path).convert("L").resize((24, 24), Image.Resampling.NEAREST)
            rgb = Image.new("RGB", (24, 24), "black")
            rgb.paste(Image.merge("RGB", (img, img, img)))
            sx = x0 + 4 + sample_idx * 20
            sy = y0 + header_h + 6
            sheet.paste(rgb, (sx, sy))

    sheet.save(sheet_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build pseudo-labeled OCR chars from plate polygons.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "plates" / "licenseplates" / "images",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "plates" / "licenseplates" / "labels",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "ocr_eval_hog324_real_mix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "characters" / "labeled_plate_auto_20",
    )
    parser.add_argument("--target-per-class", type=int, default=20)
    parser.add_argument("--min-margin", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.images_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    random.shuffle(image_paths)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    pipeline = LPRPipeline(models_dir=str(args.models_dir))
    counts = Counter()
    metadata_rows = []
    skipped = Counter()
    total_candidates = 0

    for image_path in image_paths:
        if all(counts[ch] >= args.target_per_class for ch in CHAR_CLASSES):
            break

        image = read_rgb_image(image_path)
        if image is None:
            skipped["unreadable_image"] += 1
            continue

        label_path = label_path_for_image(image_path, args.images_root, args.labels_root)
        polygons = parse_plate_polygons(label_path, image.shape)
        if not polygons:
            skipped["missing_plate_polygon"] += 1
            continue

        for plate_idx, polygon in enumerate(polygons):
            if all(counts[ch] >= args.target_per_class for ch in CHAR_CLASSES):
                break

            plate_crop = warp_quad(image, polygon)
            if plate_crop is None:
                skipped["small_plate"] += 1
                continue

            candidate = candidate_from_plate(pipeline, plate_crop)
            if not candidate.get("success"):
                skipped["ocr_failed"] += 1
                continue

            text = candidate.get("plate_string") or ""
            char_images = candidate.get("char_images") or []
            predictions = candidate.get("predictions") or []
            if len(char_images) != len(predictions):
                skipped["prediction_count_mismatch"] += 1
                continue
            if not (7 <= len(predictions) <= 9):
                skipped["bad_char_count"] += 1
                continue
            if not is_plate_like(text):
                skipped["bad_plate_format"] += 1
                continue

            raw_chars, margins = raw_predictions_and_margins(pipeline, char_images)
            if len(raw_chars) != len(predictions):
                skipped["raw_count_mismatch"] += 1
                continue

            total_candidates += len(predictions)
            for char_idx, (char_img, pred, raw_pred, margin) in enumerate(
                zip(char_images, predictions, raw_chars, margins)
            ):
                if pred not in CHAR_CLASSES:
                    skipped["unsupported_char"] += 1
                    continue
                if counts[pred] >= args.target_per_class:
                    skipped["class_full"] += 1
                    continue
                if pred != raw_pred:
                    skipped["grammar_forced_char"] += 1
                    continue
                if margin < args.min_margin:
                    skipped["low_margin_char"] += 1
                    continue

                class_dir = args.output_dir / pred
                class_dir.mkdir(parents=True, exist_ok=True)
                stem = safe_stem(f"{image_path.parent.name}_{image_path.stem}_p{plate_idx}_c{char_idx}_{pred}")
                out_path = class_dir / f"{stem}.png"
                suffix = 1
                while out_path.exists():
                    out_path = class_dir / f"{stem}_{suffix}.png"
                    suffix += 1
                cv2.imwrite(str(out_path), char_img)
                counts[pred] += 1
                metadata_rows.append({
                    "path": project_path(out_path),
                    "label": pred,
                    "source_image": project_path(image_path),
                    "source_label": project_path(label_path),
                    "plate_idx": plate_idx,
                    "char_idx": char_idx,
                    "plate_text": text,
                    "raw_plate_text": candidate.get("raw_plate_string", ""),
                    "plate_type": candidate.get("plate_type", ""),
                    "margin": f"{margin:.4f}",
                })

    for ch in CHAR_CLASSES:
        (args.output_dir / ch).mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "path",
            "label",
            "source_image",
            "source_label",
            "plate_idx",
            "char_idx",
            "plate_text",
            "raw_plate_text",
            "plate_type",
            "margin",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    sheet_path = args.output_dir / "_contact_sheet.jpg"
    save_contact_sheet(args.output_dir, counts, sheet_path)

    print(f"Images scanned: {len(image_paths)}")
    print(f"Character candidates seen: {total_candidates}")
    print(f"Saved chars: {sum(counts.values())}")
    print("Counts:")
    for ch in CHAR_CLASSES:
        print(f"  {ch}: {counts.get(ch, 0)}")
    print("Skipped:")
    for key, value in skipped.most_common():
        print(f"  {key}: {value}")
    print(f"Output: {args.output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
