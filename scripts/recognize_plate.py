#!/usr/bin/env python3
"""Run end-to-end license plate recognition on one image."""

from __future__ import annotations

import argparse
import json
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


def _json_safe_bbox(bbox):
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize license plate text from an image.")
    parser.add_argument("image", type=Path, help="Input scene image or cropped plate image.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "ocr_hog_svm",
        help="Directory containing classifier.joblib, scaler.joblib, metadata.json.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=None,
        help="Optional YOLOv8 weights for plate detection. If omitted, contour fallback is used.",
    )
    parser.add_argument(
        "--plate-crop",
        action="store_true",
        help="Treat the input image as an already-cropped license plate.",
    )
    parser.add_argument(
        "--save-debug",
        type=Path,
        default=None,
        help="Optional directory to save crop, binary image, and segmented characters.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON result.")
    return parser.parse_args()


def save_debug_outputs(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("char_*.png"):
        old_file.unlink()
    for old_name in ("plate_crop.png", "binary.png"):
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()

    if result.get("plate_crop") is not None:
        crop_bgr = cv2.cvtColor(result["plate_crop"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "plate_crop.png"), crop_bgr)

    if result.get("binary") is not None:
        cv2.imwrite(str(out_dir / "binary.png"), result["binary"])

    for idx, char_img in enumerate(result.get("char_images") or []):
        cv2.imwrite(str(out_dir / f"char_{idx:02d}.png"), char_img)


def main():
    args = parse_args()
    image_rgb = read_rgb_image(args.image)
    if image_rgb is None:
        raise SystemExit(f"Could not read image: {args.image}")

    pipeline = LPRPipeline(models_dir=str(args.models_dir))
    result = pipeline.recognize(
        image_rgb,
        yolo_model_path=str(args.yolo_model) if args.yolo_model else None,
        assume_plate_crop=args.plate_crop,
        verbose=not args.json,
    )

    if args.save_debug:
        save_debug_outputs(result, args.save_debug)

    text = result.get("plate_string", "")
    payload = {
        "success": bool(result.get("success", False)),
        "plate_text": text,
        "bbox": _json_safe_bbox(result.get("bbox")),
        "plate_type": result.get("plate_type"),
        "char_count": int(result.get("char_count", len(result.get("char_images") or []))),
        "deskew_angle": (
            None if result.get("deskew_angle") is None else float(result.get("deskew_angle"))
        ),
        "raw_plate_text": result.get("raw_plate_string"),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 40)
        print(f"Success   : {payload['success']}")
        print(f"Plate text: {payload['plate_text'] or 'N/A'}")
        print(f"Type      : {payload['plate_type'] or 'N/A'}")
        print(f"Chars     : {payload['char_count']}")
        if payload["deskew_angle"] is not None:
            print(f"Deskew    : {payload['deskew_angle']:.2f} deg")
        print(f"BBox      : {payload['bbox']}")


if __name__ == "__main__":
    main()
