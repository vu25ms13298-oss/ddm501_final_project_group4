"""Image drift transformations for simulating environmental and sensor variations.

Each function takes an RGB uint8 image and returns a transformed RGB uint8 image.
"""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


def brightness(img: np.ndarray, factor: float, **kwargs) -> np.ndarray:
    """Multiplies pixel brightness by factor (clipped to 0..255)."""
    return np.clip(img.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def gaussian_noise(
    img: np.ndarray, sigma: float = 10.0, rng: random.Random = None, **kwargs
) -> np.ndarray:
    """Adds zero-mean Gaussian sensor noise."""
    seed = rng.randint(0, 2**31 - 1) if rng else 42
    np_rng = np.random.default_rng(seed)
    noise = np_rng.normal(0, float(sigma), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, k: int = 7, **kwargs) -> np.ndarray:
    """Simulates vehicle or camera motion blur along a diagonal."""
    k = max(3, int(k))
    kernel = np.zeros((k, k), dtype=np.float32)
    np.fill_diagonal(kernel, 1.0 / k)
    return cv2.filter2D(img, -1, kernel)


def downscale(img: np.ndarray, factor: float = 0.5, **kwargs) -> np.ndarray:
    """Downscales the image and resizes it back, simulating low-res sensor."""
    factor = float(factor)
    h, w = img.shape[:2]
    new_w = max(16, int(w * factor))
    new_h = max(16, int(h * factor))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg(img: np.ndarray, quality: int = 30, **kwargs) -> np.ndarray:
    """Simulates lossy JPEG compression artifacts."""
    q = max(5, min(100, int(quality)))
    _, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q]
    )
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def rotate(
    img: np.ndarray, max_deg: float = 15.0, rng: random.Random = None, **kwargs
) -> np.ndarray:
    """Applies arbitrary in-plane rotation simulating tilted mounting."""
    angle = rng.uniform(-max_deg, max_deg) if rng else max_deg
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def occlusion(
    img: np.ndarray, frac: float = 0.3, rng: random.Random = None, **kwargs
) -> np.ndarray:
    """Paints a solid dark or noisy patch over part of the plate."""
    h, w = img.shape[:2]
    patch_w = max(4, int(w * frac))
    patch_h = max(4, int(h * frac))
    r = rng or random.Random(42)
    x = r.randint(0, max(0, w - patch_w))
    y = r.randint(0, max(0, h - patch_h))

    out = img.copy()
    out[y : y + patch_h, x : x + patch_w] = 20
    return out


TRANSFORMS = {
    "brightness": brightness,
    "noise": gaussian_noise,
    "motion_blur": motion_blur,
    "downscale": downscale,
    "jpeg": jpeg,
    "rotate": rotate,
    "occlusion": occlusion,
}


def apply_pipeline(
    img: np.ndarray, specs: list[dict[str, Any]], rng: random.Random = None
) -> np.ndarray:
    """Sequentially applies transform specifications."""
    cur = img
    for spec in specs:
        name = spec.get("name")
        if name not in TRANSFORMS:
            raise ValueError(f"Unknown transform: '{name}'. Available: {list(TRANSFORMS.keys())}")
        fn = TRANSFORMS[name]
        params = {k: v for k, v in spec.items() if k != "name"}
        cur = fn(cur, rng=rng, **params)
    return cur
