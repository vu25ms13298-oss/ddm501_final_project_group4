#!/usr/bin/env python3
"""Train a lightweight YOLOv8 plate detector on the local Kaggle dataset.

The original dataset labels are polygons. For the end-to-end LPR pipeline we
only need a bounding box to crop the plate before OCR, so this script converts
polygon labels to YOLO bbox labels and trains a smaller detection model by
default. This is much friendlier to 4GB GPUs and low-RAM laptops.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "plates" / "licenseplates"
BBOX_DATASET_ROOT = PROJECT_ROOT / "data" / "plates" / "licenseplates_bbox"


def _link_or_copy_images(src: Path, dst: Path) -> None:
    """Materialize images as hardlinks when possible, falling back safely."""
    if dst.is_symlink():
        dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for img_path in src.iterdir():
        if not img_path.is_file():
            continue
        target = dst / img_path.name
        if target.exists():
            continue
        try:
            os.link(img_path, target)
        except OSError:
            try:
                os.symlink(img_path, target)
            except OSError:
                shutil.copy2(img_path, target)


def _polygon_or_box_to_bbox(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    vals = [float(v) for v in parts]
    if len(vals) == 5:
        _, xc, yc, w, h = vals
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
    else:
        coords = vals[1:]
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    xc = x1 + w / 2
    yc = y1 + h / 2
    return xc, yc, w, h


def prepare_bbox_dataset(
    src_root: Path, dst_root: Path, single_class: bool = True
) -> Path:
    """Create a bbox-only YOLO dataset from polygon labels."""
    for split in ("train", "val"):
        _link_or_copy_images(src_root / "images" / split, dst_root / "images" / split)
        src_label_dir = src_root / "labels" / split
        dst_label_dir = dst_root / "labels" / split
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for label_path in src_label_dir.glob("*.txt"):
            out_lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls = 0 if single_class else int(float(line.split()[0]))
                bbox = _polygon_or_box_to_bbox(line)
                if bbox is None:
                    continue
                xc, yc, w, h = bbox
                out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (dst_label_dir / label_path.name).write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""),
                encoding="utf-8",
            )

    return dst_root


def write_local_yaml(dataset_root: Path, single_class: bool = True) -> Path:
    yaml_path = dataset_root / "local_dataset.yaml"
    names = "['plate']" if single_class else "['BSD', 'BSV']"
    nc = 1 if single_class else 2
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                f"nc: {nc}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 plate detector.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dataset-root", type=Path, default=BBOX_DATASET_ROOT)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", default="license_plate_yolov8")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--keep-classes",
        action="store_true",
        help="Keep BSD/BSV as two classes instead of training one generic plate class.",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Train from original polygon labels with a YOLO segmentation model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")

    single_class = not args.keep_classes
    if args.segment:
        data_root = args.dataset_root
        data_yaml = write_local_yaml(data_root, single_class=False)
        base_model = (
            args.base_model if args.base_model != "yolov8n.pt" else "yolov8n-seg.pt"
        )
    else:
        data_root = prepare_bbox_dataset(
            args.dataset_root, args.output_dataset_root, single_class
        )
        data_yaml = write_local_yaml(data_root, single_class=single_class)
        base_model = args.base_model

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Dataset YAML: {data_yaml}")
    print(f"Base model  : {base_model}")
    print(f"Device      : {device}")
    print(f"imgsz/batch : {args.imgsz}/{args.batch}")
    print(f"workers     : {args.workers}")

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        device=device,
        workers=args.workers,
        cache=False,
        plots=False,
        val=args.val,
        fraction=args.fraction,
        project=str(PROJECT_ROOT / "models" / "yolo_runs"),
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best YOLO weights: {best}")


if __name__ == "__main__":
    main()
