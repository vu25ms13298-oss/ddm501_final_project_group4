import random
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from simulations.drift_transforms import (
    TRANSFORMS,
    apply_pipeline,
    brightness,
    downscale,
    gaussian_noise,
    jpeg,
    motion_blur,
    occlusion,
    rotate,
)


@pytest.fixture
def base_image():
    # 64x64 solid gray RGB image
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def test_brightness_darken(base_image):
    dark = brightness(base_image, factor=0.5)
    assert dark.shape == base_image.shape
    assert dark.dtype == np.uint8
    assert dark.mean() < base_image.mean()


def test_brightness_brighten(base_image):
    bright = brightness(base_image, factor=1.5)
    assert bright.mean() > base_image.mean()


def test_gaussian_noise(base_image):
    noisy = gaussian_noise(base_image, sigma=20.0, rng=random.Random(42))
    assert noisy.shape == base_image.shape
    assert not np.array_equal(noisy, base_image)


def test_motion_blur(base_image):
    blurred = motion_blur(base_image, k=5)
    assert blurred.shape == base_image.shape


def test_downscale(base_image):
    small = downscale(base_image, factor=0.25)
    assert small.shape == base_image.shape  # resized back to original


def test_jpeg_compression(base_image):
    comp = jpeg(base_image, quality=20)
    assert comp.shape == base_image.shape


def test_rotate(base_image):
    rot = rotate(base_image, max_deg=10.0, rng=random.Random(42))
    assert rot.shape == base_image.shape


def test_occlusion(base_image):
    occ = occlusion(base_image, frac=0.3, rng=random.Random(42))
    assert occ.shape == base_image.shape
    assert occ.mean() < base_image.mean()


def test_apply_pipeline(base_image):
    specs = [
        {"name": "brightness", "factor": 0.5},
        {"name": "downscale", "factor": 0.5},
    ]
    res = apply_pipeline(base_image, specs)
    assert res.shape == base_image.shape
    assert res.mean() < base_image.mean()


def test_apply_pipeline_unknown_transform(base_image):
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_pipeline(base_image, [{"name": "non_existent_transform"}])


def test_transforms_registry():
    assert "brightness" in TRANSFORMS
    assert "noise" in TRANSFORMS
    assert "downscale" in TRANSFORMS
    assert "rotate" in TRANSFORMS
