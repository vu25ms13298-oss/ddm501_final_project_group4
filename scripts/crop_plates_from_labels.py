#!/usr/bin/env python3
"""Crop ground-truth plates using YOLO polygon labels (OCR-only evaluation).

Reads a plate-text manifest (``image,label,...``), finds each image's YOLO
label file (``images/`` -> ``labels/``, ``.png`` -> ``.txt``; one polygon per
line: ``cls x1 y1 x2 y2 x3 y3 x4 y4`` normalised), warps the largest polygon to
an upright rectangle and writes a new manifest with ``plate_crop=true``.

Evaluating on these crops isolates the OCR stage (segmentation + HOG/SVM +
format correction) from plate detection.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def label_path_for(image_path: Path) -> Path:
    parts = ["labels" if p == "images" else p for p in image_path.parts]
    return Path(*parts).with_suffix(".txt")


def largest_polygon(label_file: Path, width: int, height: int) -> np.ndarray:
    """Returns the 4x2 pixel polygon with the largest area in a label file."""
    polygons = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 9:
            continue
        pts = np.array(values[1:], dtype=np.float32).reshape(4, 2) * [width, height]
        polygons.append(pts.astype(np.float32))
    if not polygons:
        raise ValueError(f"No 4-point polygon in {label_file}")
    return max(polygons, key=lambda p: cv2.contourArea(p))


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Orders corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def warp_plate(image: np.ndarray, polygon: np.ndarray, margin: float = 0.06):
    """Perspective-warps the polygon (plus a small margin) to a rectangle."""
    tl, tr, br, bl = order_corners(polygon)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    mx, my = margin * w, margin * h
    dst = np.array(
        [[mx, my], [w + mx, my], [w + mx, h + my], [mx, h + my]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    size = (int(w + 2 * mx), int(h + 2 * my))
    return cv2.warpPerspective(
        image, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "plate_text_labels.csv",
    )
    p.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "results" / "real_plate_crops"
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        image_path = PROJECT_ROOT / row["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        h, w = image.shape[:2]
        polygon = largest_polygon(label_path_for(image_path), w, h)
        crop_path = args.out_dir / image_path.name
        cv2.imwrite(str(crop_path), warp_plate(image, polygon))
        out_rows.append(
            {"image": str(crop_path), "label": row["label"], "plate_crop": "true"}
        )

    manifest = args.out_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "plate_crop"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} crops and {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
