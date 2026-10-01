"""Data quality tests for model artifacts and training data."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import CHAR_CLASSES

MODELS_DIR = PROJECT_ROOT / "models" / "ocr_hog_svm"


@pytest.fixture
def metadata():
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        pytest.skip("metadata.json not found")
    with open(meta_path) as f:
        return json.load(f)


class TestModelArtifacts:
    def test_classifier_file_exists(self):
        clf_path = MODELS_DIR / "classifier.joblib"
        if not clf_path.exists():
            clf_path = MODELS_DIR / "svm_classifier.pkl"
        assert clf_path.exists(), "No classifier file found"

    def test_scaler_file_exists(self):
        scaler_path = MODELS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            scaler_path = MODELS_DIR / "feature_scaler.pkl"
        assert scaler_path.exists(), "No scaler file found"

    def test_metadata_file_exists(self):
        assert (MODELS_DIR / "metadata.json").exists() or (
            MODELS_DIR / "char_classes.json"
        ).exists()

    def test_classifier_loadable(self):
        clf_path = MODELS_DIR / "classifier.joblib"
        if not clf_path.exists():
            clf_path = MODELS_DIR / "svm_classifier.pkl"
        clf = joblib.load(clf_path)
        assert hasattr(clf, "predict")

    def test_scaler_loadable(self):
        scaler_path = MODELS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            scaler_path = MODELS_DIR / "feature_scaler.pkl"
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, "transform")


class TestMetadataQuality:
    def test_char_classes_count(self, metadata):
        classes = metadata.get("char_classes", [])
        assert len(classes) == 31, f"Expected 31 classes, got {len(classes)}"

    def test_char_classes_match_source(self, metadata):
        classes = metadata.get("char_classes", [])
        expected = list(CHAR_CLASSES)
        assert classes == expected

    def test_feature_method_valid(self, metadata):
        valid_methods = {"hog", "hog_legacy", "resnet", "wavelet", "raw"}
        assert metadata.get("feature_method") in valid_methods

    def test_metrics_present(self, metadata):
        metrics = metadata.get("metrics", {})
        assert "accuracy" in metrics
        assert "macro_f1" in metrics

    def test_accuracy_above_threshold(self, metadata):
        accuracy = metadata.get("metrics", {}).get("accuracy", 0)
        assert accuracy >= 0.85, f"Model accuracy {accuracy} is below 85% threshold"

    def test_feature_dim_consistent(self, metadata):
        dim = metadata.get("feature_dim") or metadata.get("metrics", {}).get(
            "feature_dim"
        )
        if dim is not None:
            assert dim > 0, "Feature dimension must be positive"

    def test_no_excluded_chars(self, metadata):
        classes = metadata.get("char_classes", [])
        excluded = {"I", "J", "O", "Q", "W"}
        present = excluded.intersection(set(classes))
        assert len(present) == 0, f"Excluded chars found in classes: {present}"


class TestScalerDataQuality:
    def test_scaler_feature_count(self):
        scaler_path = MODELS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if not scaler_path.exists():
            pytest.skip("Scaler file not found")
        scaler = joblib.load(scaler_path)
        n_features = getattr(scaler, "n_features_in_", None)
        if n_features is not None:
            assert n_features > 0

    def test_scaler_no_nan_means(self):
        scaler_path = MODELS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if not scaler_path.exists():
            pytest.skip("Scaler file not found")
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, "mean_"):
            assert not np.any(np.isnan(scaler.mean_)), "Scaler has NaN means"

    def test_scaler_low_zero_variance(self):
        scaler_path = MODELS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if not scaler_path.exists():
            pytest.skip("Scaler file not found")
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, "var_"):
            n_features = len(scaler.var_)
            zero_var = int(np.sum(scaler.var_ == 0))
            ratio = zero_var / n_features
            assert ratio < 0.05, (
                f"{zero_var}/{n_features} features ({ratio:.1%}) " f"have zero variance"
            )
