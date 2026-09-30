#!/usr/bin/env python3
"""Build a labeled OCR character dataset from labeled plate images.

The input label is the full plate text once per image. The script runs the
current LPR segmentation pipeline, then maps char_00 -> label[0],
char_01 -> label[1], ... when the counts match. Output follows the folder
layout already supported by train_ocr_model.py:

    data/characters/labeled/0/*.png
    data/characters/labeled/A/*.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import CHAR_CLASSES
from src.pipeline import LPRPipeline


def normalize_label(text: str) -> str:
    allowed = set(CHAR_CLASSES)
    return "".join(ch for ch in text.upper() if ch in allowed)


def parse_bool(value, default=False):
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "plate", "crop"}


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def iter_rows(args):
    if args.image:
        yield {
            "image": str(args.image),
            "label": args.label,
            "plate_crop": str(args.plate_crop),
        }
        return

    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            yield row


def save_char_samples(char_images, label: str, source_path: Path, output_dir: Path):
    saved = []
    stem = safe_stem(source_path)
    for idx, (char_img, ch) in enumerate(zip(char_images, label)):
        class_dir = output_dir / ch
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{stem}_char_{idx:02d}_{ch}.png"
        suffix = 1
        while out_path.exists():
            out_path = class_dir / f"{stem}_char_{idx:02d}_{ch}_{suffix}.png"
            suffix += 1
        cv2.imwrite(str(out_path), char_img)
        saved.append(out_path)
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create data/characters/labeled samples from labeled plate images."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV with columns: image,label, optional plate_crop.",
    )
    parser.add_argument("--image", type=Path, help="Single image mode.")
    parser.add_argument("--label", default="", help="Full plate text for --image mode.")
    parser.add_argument(
        "--plate-crop",
        action="store_true",
        help="Treat input images as already-cropped plates unless row overrides it.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "ocr_hog_svm",
    )
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "characters" / "labeled",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Save the aligned prefix when crop count and label length differ.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.manifest:
        raise SystemExit("Provide --manifest or --image with --label.")
    if args.image and not args.label:
        raise SystemExit("--image mode requires --label.")

    pipeline = LPRPipeline(models_dir=str(args.models_dir))
    total_saved = 0
    skipped = 0

    for row in iter_rows(args):
        image_path = (
            (PROJECT_ROOT / row["image"]).resolve()
            if not Path(row["image"]).is_absolute()
            else Path(row["image"])
        )
        label = normalize_label(row["label"])
        if not label:
            print(f"SKIP {image_path}: empty/unsupported label")
            skipped += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"SKIP {image_path}: cannot read image")
            skipped += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        assume_plate_crop = parse_bool(row.get("plate_crop"), default=args.plate_crop)
        result = pipeline.recognize(
            image_rgb,
            yolo_model_path=str(args.yolo_model) if args.yolo_model else None,
            assume_plate_crop=assume_plate_crop,
            verbose=False,
        )
        char_images = result.get("char_images") or []

        if len(char_images) != len(label):
            if not args.allow_mismatch:
                print(
                    f"SKIP {image_path}: {len(char_images)} crops but label "
                    f"has {len(label)} chars ({label})"
                )
                skipped += 1
                continue
            n = min(len(char_images), len(label))
            char_images = char_images[:n]
            label = label[:n]

        saved = save_char_samples(char_images, label, image_path, args.output_dir)
        total_saved += len(saved)
        print(
            f"OK {image_path}: saved {len(saved)} chars, "
            f"pipeline_text={result.get('plate_string', '')}, label={label}"
        )

    print(f"Done. saved={total_saved}, skipped={skipped}, output={args.output_dir}")


if __name__ == "__main__":
    main()
