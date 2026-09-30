"""Unit tests for preprocessing module."""

import numpy as np

from src.preprocessing import (
    preprocess_scene_image,
    rectify_plate_crop,
    deskew_plate_with_angle,
    enhance_plate_crop_for_ocr,
)


class TestPreprocessSceneImage:
    def test_output_shape_preserved(self, sample_rgb_image):
        result = preprocess_scene_image(sample_rgb_image)
        assert result.shape == sample_rgb_image.shape
        assert result.dtype == np.uint8

    def test_large_image_resized(self):
        big = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        result = preprocess_scene_image(big)
        assert max(result.shape[:2]) <= 1280

    def test_small_image_not_upscaled(self):
        small = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = preprocess_scene_image(small)
        assert result.shape[0] == 100
        assert result.shape[1] == 150

    def test_output_is_rgb(self, sample_rgb_image):
        result = preprocess_scene_image(sample_rgb_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3


class TestRectifyPlateCrop:
    def test_returns_valid_image(self, sample_plate_crop):
        rgb_plate = np.stack([sample_plate_crop] * 3, axis=-1)
        result = rectify_plate_crop(rgb_plate)
        assert result is not None
        assert result.size > 0
        assert result.dtype == np.uint8

    def test_small_crop_handled(self):
        tiny = np.full((10, 20, 3), 128, dtype=np.uint8)
        result = rectify_plate_crop(tiny)
        assert result is not None


class TestDeskewPlate:
    def test_returns_tuple(self, sample_plate_crop):
        rgb_plate = np.stack([sample_plate_crop] * 3, axis=-1)
        result, angle = deskew_plate_with_angle(rgb_plate)
        assert isinstance(angle, float)
        assert result is not None
        assert result.size > 0

    def test_angle_within_range(self, sample_plate_crop):
        rgb_plate = np.stack([sample_plate_crop] * 3, axis=-1)
        _, angle = deskew_plate_with_angle(rgb_plate)
        assert -45.0 <= angle <= 45.0


class TestEnhancePlateCrop:
    def test_output_valid(self, sample_plate_crop):
        rgb_plate = np.stack([sample_plate_crop] * 3, axis=-1)
        result = enhance_plate_crop_for_ocr(rgb_plate)
        assert result is not None
        assert result.dtype == np.uint8
