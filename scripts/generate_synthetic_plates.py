#!/usr/bin/env python3
"""Generate a labelled synthetic Vietnamese plate benchmark for end-to-end checks.

Renders 1-line (car, e.g. ``51F-123.45``) and 2-line (motorbike, e.g.
``59-X1 / 123.45``) plates with mild rotation, blur and noise, either as tight
plate crops or pasted into a larger scene. Writes the images plus a
``manifest.csv`` (image,label,plate_crop) consumable by
``scripts/evaluate_lpr_end_to_end.py``.

This is a regression/smoke benchmark, NOT a substitute for real-world data:
synthetic plates are cleaner than camera images, so scores here are an upper
bound on real performance.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLATE_LETTERS = "ABCDEFGHKLMNPSTUVXYZ"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError(
        "No TrueType font found; install fonts-dejavu-core to render plates"
    )


def random_plate(rng: random.Random, two_line: bool) -> tuple[str, list[str]]:
    """Returns (label, display_lines) for a random Vietnamese-format plate."""
    province = f"{rng.randint(11, 99)}"
    letter = rng.choice(PLATE_LETTERS)
    serial = f"{rng.randint(0, 99999):05d}"
    serial_display = f"{serial[:3]}.{serial[3:]}"
    if two_line:
        series_digit = str(rng.randint(1, 9))
        label = province + letter + series_digit + serial
        return label, [f"{province}-{letter}{series_digit}", serial_display]
    label = province + letter + serial
    return label, [f"{province}{letter}-{serial_display}"]


def render_plate(lines: list[str]) -> np.ndarray:
    """Renders black glyphs on a white plate with a dark border (RGB)."""
    two_line = len(lines) == 2
    width, height = (300, 220) if two_line else (560, 120)
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([3, 3, width - 4, height - 4], outline=(20, 20, 20), width=6)

    line_h = (height - 20) / len(lines)
    # Largest font whose glyphs fit inside the border with a margin.
    size = int(line_h)
    while size > 10:
        font = _load_font(size)
        boxes = [draw.textbbox((0, 0), t, font=font) for t in lines]
        if all(
            b[2] - b[0] <= width * 0.86 and b[3] - b[1] <= line_h * 0.78 for b in boxes
        ):
            break
        size -= 2
    for i, text in enumerate(lines):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - tw) / 2 - bbox[0]
        y = 10 + i * line_h + (line_h - th) / 2 - bbox[1]
        draw.text((x, y), text, fill=(15, 15, 15), font=font)
    return np.array(img)


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Applies mild camera-like rotation, blur and noise."""
    h, w = img.shape[:2]
    angle = rng.uniform(-3.0, 3.0)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(
        img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.9))
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
        0, rng.uniform(2, 8), img.shape
    )
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def place_in_scene(plate: np.ndarray, rng: random.Random) -> np.ndarray:
    """Pastes the plate onto a darker, textured 960x720 scene."""
    scene_h, scene_w = 720, 960
    base = rng.randint(50, 110)
    scene = np.random.default_rng(rng.randint(0, 2**31)).normal(
        base, 12, (scene_h, scene_w, 3)
    )
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    ph, pw = plate.shape[:2]
    y = rng.randint(80, scene_h - ph - 80)
    x = rng.randint(80, scene_w - pw - 80)
    scene[y : y + ph, x : x + pw] = plate
    return scene


def generate(out_dir: Path, n: int, seed: int, scene_fraction: float) -> Path:
    """Generates ``n`` plates under ``out_dir``; returns the manifest path."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        two_line = i % 3 == 2  # roughly one third motorbike plates
        label, lines = random_plate(rng, two_line)
        plate = degrade(render_plate(lines), rng)
        in_scene = rng.random() < scene_fraction
        image = place_in_scene(plate, rng) if in_scene else plate
        name = f"plate_{i:03d}.png"
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        rows.append(
            {"image": str(out_dir / name), "label": label, "plate_crop": not in_scene}
        )

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "plate_crop"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks" / "synthetic_plates",
    )
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--scene-fraction",
        type=float,
        default=0.3,
        help="Share of plates pasted into a scene (needs detection)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    manifest = generate(args.out_dir, args.n, args.seed, args.scene_fraction)
    print(f"Wrote {args.n} plates and {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
