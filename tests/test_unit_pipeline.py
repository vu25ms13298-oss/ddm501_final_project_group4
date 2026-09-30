"""Unit tests for the LPR pipeline module."""

import numpy as np

from src.pipeline import LPRPipeline, correct_plate_format


class TestCorrectPlateFormat:
    def test_valid_plate_unchanged(self):
        assert correct_plate_format("51F12345") == "51F12345"

    def test_letter_digit_swap(self):
        result = correct_plate_format("5IF12345")
        assert result[1].isdigit()

    def test_strips_non_alnum(self):
        result = correct_plate_format("51-F-123.45")
        assert result.isalnum()

    def test_empty_string(self):
        assert correct_plate_format("") == ""

    def test_short_input(self):
        result = correct_plate_format("AB")
        assert isinstance(result, str)


class TestLPRPipelineInit:
    def test_init_without_models(self):
        p = LPRPipeline()
        assert p.svm_model is None
        assert p.scaler is None
        assert p.yolo_model is None

    def test_init_with_models(self, models_dir):
        p = LPRPipeline(models_dir=models_dir)
        assert p.svm_model is not None
        assert p.scaler is not None

    def test_feature_method_loaded(self, models_dir):
        p = LPRPipeline(models_dir=models_dir)
        assert p.feature_method in ("hog", "hog_legacy", "resnet", "wavelet", "raw")

    def test_char_classes_loaded(self, models_dir):
        p = LPRPipeline(models_dir=models_dir)
        assert len(p.char_classes) == 31
        assert "A" in p.char_classes
        assert "0" in p.char_classes


class TestLPRPipelineRecognize:
    def test_recognize_returns_dict(self, models_dir, sample_rgb_image):
        p = LPRPipeline(models_dir=models_dir)
        result = p.recognize(sample_rgb_image, verbose=False)
        assert isinstance(result, dict)
        assert "success" in result

    def test_recognize_plate_crop(self, models_dir):
        plate = np.full((80, 400, 3), 220, dtype=np.uint8)
        for x in range(20, 360, 45):
            plate[15:65, x : x + 30] = 40
        p = LPRPipeline(models_dir=models_dir)
        result = p.recognize(plate, assume_plate_crop=True, verbose=False)
        assert isinstance(result, dict)
