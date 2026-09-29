"""Model validation tests — verify the trained model meets quality gates."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import CHAR_CLASSES, generate_synthetic_chars
from src.features import extract_hog_features

MODELS_DIR = PROJECT_ROOT / "models" / "ocr_hog_svm"


@pytest.fixture(scope="module")
def model_and_scaler():
    clf_path = MODELS_DIR / "classifier.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    if not clf_path.exists():
        clf_path = MODELS_DIR / "svm_classifier.pkl"
    if not scaler_path.exists():
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
    if not clf_path.exists() or not scaler_path.exists():
        pytest.skip("Model files not found")
    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler


@pytest.fixture(scope="module")
def validation_data():
    """Generate a small validation set from synthetic chars."""
    np.random.seed(99)
    x_imgs, y_labels = generate_synthetic_chars(
        char_classes=CHAR_CLASSES,
        samples_per_class=20,
        img_size=64,
    )
    features = extract_hog_features(x_imgs)
    return features, y_labels


class TestModelPredictionQuality:
    def test_accuracy_above_threshold(self, model_and_scaler, validation_data):
        clf, scaler = model_and_scaler
        X, y_true = validation_data
        X_scaled = scaler.transform(X)
        y_pred = clf.predict(X_scaled)
        acc = accuracy_score(y_true, y_pred)
        assert acc >= 0.70, f"Validation accuracy {acc:.2%} is below 70% threshold"

    def test_f1_above_threshold(self, model_and_scaler, validation_data):
        clf, scaler = model_and_scaler
        X, y_true = validation_data
        X_scaled = scaler.transform(X)
        y_pred = clf.predict(X_scaled)
        f1 = f1_score(y_true, y_pred, average="macro")
        assert f1 >= 0.65, f"Validation macro F1 {f1:.2%} is below 65% threshold"


class TestModelBehavior:
    def test_predicts_all_classes(self, model_and_scaler, validation_data):
        clf, scaler = model_and_scaler
        X, _ = validation_data
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        unique_preds = set(preds)
        assert len(unique_preds) >= 15, (
            f"Model only predicts {len(unique_preds)} classes out of 31"
        )

    def test_prediction_shape(self, model_and_scaler, validation_data):
        clf, scaler = model_and_scaler
        X, _ = validation_data
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        assert preds.shape[0] == X.shape[0]

    def test_prediction_range(self, model_and_scaler, validation_data):
        clf, scaler = model_and_scaler
        X, _ = validation_data
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        assert preds.min() >= 0
        assert preds.max() < len(CHAR_CLASSES)

    def test_decision_function_available(self, model_and_scaler, validation_data):
        clf, scaler = model_and_scaler
        X, _ = validation_data
        X_scaled = scaler.transform(X[:5])
        assert hasattr(clf, "decision_function"), "SVM should have decision_function"
        scores = clf.decision_function(X_scaled)
        assert scores.ndim == 2
        assert scores.shape[1] == len(CHAR_CLASSES)


class TestModelRobustness:
    def test_handles_noisy_input(self, model_and_scaler):
        clf, scaler = model_and_scaler
        noisy = [np.random.randint(0, 255, (32, 32), dtype=np.uint8) for _ in range(10)]
        features = extract_hog_features(noisy)
        X_scaled = scaler.transform(features)
        preds = clf.predict(X_scaled)
        assert len(preds) == 10

    def test_handles_blank_input(self, model_and_scaler):
        clf, scaler = model_and_scaler
        blank = [np.zeros((32, 32), dtype=np.uint8) for _ in range(5)]
        features = extract_hog_features(blank)
        X_scaled = scaler.transform(features)
        preds = clf.predict(X_scaled)
        assert len(preds) == 5

    def test_handles_white_input(self, model_and_scaler):
        clf, scaler = model_and_scaler
        white = [np.full((32, 32), 255, dtype=np.uint8) for _ in range(5)]
        features = extract_hog_features(white)
        X_scaled = scaler.transform(features)
        preds = clf.predict(X_scaled)
        assert len(preds) == 5
