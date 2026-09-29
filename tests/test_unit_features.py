"""Unit tests for feature extraction module."""

import numpy as np
import pytest

from src.features import (
    extract_hog_features,
    extract_hog_legacy_features,
    extract_raw_features,
    extract_wavelet_features,
)


@pytest.fixture
def char_batch():
    """Batch of 5 synthetic 32x32 grayscale character images."""
    batch = []
    for _ in range(5):
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        batch.append(img)
    return batch


class TestHOGFeatures:
    def test_output_shape(self, char_batch):
        features = extract_hog_features(char_batch)
        assert features.shape[0] == 5
        assert features.shape[1] == 1764  # 4x4 pixels_per_cell

    def test_deterministic(self, char_batch):
        f1 = extract_hog_features(char_batch)
        f2 = extract_hog_features(char_batch)
        np.testing.assert_array_equal(f1, f2)

    def test_single_image(self):
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        features = extract_hog_features([img])
        assert features.shape == (1, 1764)


class TestHOGLegacyFeatures:
    def test_output_shape(self, char_batch):
        features = extract_hog_legacy_features(char_batch)
        assert features.shape[0] == 5
        assert features.shape[1] == 324  # 8x8 pixels_per_cell


class TestRawFeatures:
    def test_output_shape(self, char_batch):
        features = extract_raw_features(char_batch)
        assert features.shape == (5, 1024)  # 32*32 = 1024

    def test_normalized(self, char_batch):
        features = extract_raw_features(char_batch)
        assert features.min() >= 0.0
        assert features.max() <= 1.0


class TestWaveletFeatures:
    def test_output_shape(self, char_batch):
        features = extract_wavelet_features(char_batch)
        assert features.shape[0] == 5
        assert features.ndim == 2
