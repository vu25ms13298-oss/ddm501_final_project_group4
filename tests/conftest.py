import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_rgb_image():
    """A simple 640x480 RGB image with some structure."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:380, 150:490] = 200
    img[200:280, 200:440] = 255
    return img


@pytest.fixture
def sample_plate_crop():
    """A synthetic 400x80 grayscale plate-like crop."""
    plate = np.full((80, 400), 220, dtype=np.uint8)
    for x in range(20, 360, 45):
        plate[15:65, x : x + 30] = 40
    return plate


@pytest.fixture
def sample_char_image():
    """A 32x32 grayscale character-like image."""
    img = np.full((32, 32), 240, dtype=np.uint8)
    img[4:28, 8:24] = 30
    return img


@pytest.fixture
def models_dir():
    """Path to the OCR models directory."""
    p = PROJECT_ROOT / "models" / "ocr_hog_svm"
    if not p.exists():
        pytest.skip("OCR model files not found")
    return str(p)
