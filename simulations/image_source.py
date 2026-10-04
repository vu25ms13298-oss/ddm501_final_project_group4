"""Image source manager for LPR simulations.

Supplies real dataset images, dynamically generated synthetic plates (in-memory),
and non-plate background noise to test prediction drift.
"""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("lpr_simulations.image_source")


@dataclass
class Sample:
    name: str
    image: np.ndarray  # RGB uint8
    label: str | None = None
    is_crop: bool = False


def load_dataset_images(dirs: list[str | Path]) -> list[Sample]:
    """Loads all JPEG/PNG files from the specified directories."""
    samples = []
    base_dir = Path(__file__).resolve().parent
    for d in dirs:
        p = Path(d)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            log.warning("Dataset directory not found: %s", p)
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            for f in sorted(p.glob(ext)):
                bgr = cv2.imread(str(f))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                samples.append(
                    Sample(name=f.name, image=rgb, label=None, is_crop=False)
                )
    log.info("Loaded %d dataset images", len(samples))
    return samples


def build_synthetic_pool(
    n: int = 40, seed: int = 42, scene_fraction: float = 0.3
) -> list[Sample]:
    """Builds an in-memory pool of rendered synthetic plates."""
    from scripts.generate_synthetic_plates import (
        degrade,
        place_in_scene,
        random_plate,
        render_plate,
    )

    rng = random.Random(seed)
    samples = []
    for i in range(n):
        two_line = i % 3 == 2
        label, lines = random_plate(rng, two_line=two_line)
        try:
            plate_rgb = render_plate(lines)
        except RuntimeError:
            log.warning("TrueType fonts not found; synthetic plate generation skipped")
            break
        plate_degraded = degrade(plate_rgb, rng)
        in_scene = rng.random() < scene_fraction
        if in_scene:
            final_img = place_in_scene(plate_degraded, rng)
            is_crop = False
        else:
            final_img = plate_degraded
            is_crop = True

        samples.append(
            Sample(
                name=f"syn_{label}_{i:03d}.png",
                image=final_img,
                label=label,
                is_crop=is_crop,
            )
        )
    log.info("Built synthetic pool of %d plates", len(samples))
    return samples


def make_non_plate_images(n: int = 20, seed: int = 123) -> list[Sample]:
    """Generates synthetic non-plate scenery/noise images to induce prediction drift."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        mode = i % 3
        if mode == 0:
            # Random natural-ish gradient
            x = np.linspace(50, 180, 480, dtype=np.uint8)
            y = np.linspace(80, 220, 640, dtype=np.uint8)
            xx, yy = np.meshgrid(y, x)
            img = np.stack([xx, yy, (xx + yy) // 2], axis=-1)
        elif mode == 1:
            # Geometric noise patterns
            img = rng.integers(60, 200, (480, 640, 3), dtype=np.uint8)
            for _ in range(5):
                pt1 = (int(rng.integers(0, 600)), int(rng.integers(0, 400)))
                pt2 = (int(rng.integers(0, 600)), int(rng.integers(0, 400)))
                cv2.rectangle(img, pt1, pt2, (int(rng.integers(0, 255)), 0, 0), -1)
        else:
            # Gaussian texture
            base = rng.normal(128, 30, (480, 640, 3))
            img = np.clip(base, 0, 255).astype(np.uint8)

        samples.append(
            Sample(name=f"non_plate_{i:03d}.png", image=img, label=None, is_crop=False)
        )
    return samples


class ImageSource:
    """Provides samples according to chosen source strategy."""

    def __init__(self, config: dict):
        img_cfg = config.get("images", {})
        dirs = img_cfg.get("dataset_dirs", [])
        self.dataset_samples = load_dataset_images(dirs)

        pool_size = img_cfg.get("synthetic_pool_size", 40)
        seed = img_cfg.get("synthetic_seed", 42)
        scene_frac = img_cfg.get("scene_fraction", 0.3)
        self.synthetic_samples = build_synthetic_pool(pool_size, seed, scene_frac)

        self.noise_samples = make_non_plate_images(20)

    def sample(self, source: str = "mixed", rng: random.Random = None) -> Sample:
        r = rng or random.Random()
        s = source.lower()

        if s == "noise_images":
            return r.choice(self.noise_samples)
        elif s == "dataset" and self.dataset_samples:
            return r.choice(self.dataset_samples)
        elif s == "synthetic" and self.synthetic_samples:
            return r.choice(self.synthetic_samples)

        # Default: mixed (or fallback when one is empty)
        candidates = []
        if self.dataset_samples:
            candidates.extend(self.dataset_samples)
        if self.synthetic_samples:
            candidates.extend(self.synthetic_samples)
        if not candidates:
            # Emergency fallback: generate 1 synthetic plate on the fly
            return build_synthetic_pool(1, 99)[0]
        return r.choice(candidates)
