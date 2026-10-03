"""Unit tests for synthetic data generation, SVM training and model saving."""

import json

import numpy as np
import pytest

from src import classifier as clf_mod
from src.classifier import (
    generate_plate_style_synthetic_chars,
    generate_synthetic_chars,
    save_models,
    train_svm,
)
from src.features import extract_hog_features
from src.pipeline import LPRPipeline


class TestSyntheticGeneration:
    def test_synthetic_shapes_and_labels(self):
        x, y = generate_synthetic_chars(char_classes="0A", samples_per_class=3)
        assert x.shape == (6, 32, 32)
        assert x.dtype == np.uint8
        # Labels index into the requested char_classes string.
        assert sorted(set(y.tolist())) == [0, 1]

    def test_plate_style_shapes_and_labels(self):
        x, y = generate_plate_style_synthetic_chars(
            char_classes="58", samples_per_class=4
        )
        assert x.shape == (8, 32, 32)
        assert len(y) == 8
        assert x.max() > 0  # glyphs are actually drawn

    def test_font_fallback_warns(self, monkeypatch):
        monkeypatch.setattr(clf_mod.os.path, "exists", lambda _p: False)
        with pytest.warns(RuntimeWarning, match="TrueType"):
            fonts = clf_mod._available_fonts()
        assert len(fonts) == 1


class TestTrainAndSave:
    @pytest.fixture
    def trained(self):
        np.random.seed(0)
        x, y = generate_synthetic_chars(char_classes="01AB", samples_per_class=6)
        feats = extract_hog_features(x)
        model, scaler = train_svm(feats, y)
        return model, scaler, feats, y

    def test_train_svm_fits(self, trained):
        model, scaler, feats, y = trained
        acc = (model.predict(scaler.transform(feats)) == y).mean()
        assert acc > 0.9  # training accuracy on a tiny separable set

    def test_save_models_writes_consistent_metadata(self, trained, tmp_path):
        model, scaler, feats, _ = trained
        save_models(model, scaler, tmp_path, feature_method="hog", metrics={"a": 1})

        meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        assert meta["feature_dim"] == feats.shape[1] == 1764
        assert meta["feature_method"] == "hog"
        assert meta["sklearn_version"]
        assert (tmp_path / "svm_classifier.pkl").exists()
        assert (tmp_path / "feature_scaler.pkl").exists()

    def test_saved_model_loads_into_pipeline(self, trained, tmp_path):
        model, scaler, _, _ = trained
        save_models(model, scaler, tmp_path, feature_method="hog")
        p = LPRPipeline(models_dir=str(tmp_path))
        assert p.feature_method == "hog"
        assert p.svm_model is not None


class TestPipelineMetadataGuard:
    def test_feature_dim_mismatch_is_rejected(self, sample_char_image):
        feats = extract_hog_features([sample_char_image] * 4)
        model, scaler = train_svm(feats, np.array([0, 1, 0, 1]))
        with pytest.raises(ValueError, match="feature_dim"):
            LPRPipeline().set_ocr_model(
                model, scaler, {"feature_method": "hog", "feature_dim": 324}
            )

    def test_legacy_hog_detected_from_scaler(self, sample_char_image):
        from src.features import extract_hog_legacy_features

        feats = extract_hog_legacy_features([sample_char_image] * 4)
        model, scaler = train_svm(feats, np.array([0, 1, 0, 1]))
        p = LPRPipeline()
        p.set_ocr_model(model, scaler, {"feature_method": "hog"})
        assert p.feature_method == "hog_legacy"
