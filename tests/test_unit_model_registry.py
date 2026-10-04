"""Tests for MLflow model registry helpers (uses a local sqlite-backed MLflow)."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.model_registry import (
    PRODUCTION_ALIAS,
    build_sklearn_pipeline,
    champion_accuracy,
    load_from_registry,
    log_and_register,
    resolve_model_uri,
    should_promote,
    split_sklearn_pipeline,
)

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6))
    y = np.repeat([0, 1], 20)
    scaler = StandardScaler().fit(x)
    clf = SVC(kernel="linear").fit(scaler.transform(x), y)
    return clf, scaler, x


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """Points MLflow at a throwaway sqlite registry + local artifact store."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-registry")
    yield uri
    mlflow.set_tracking_uri(None)


class TestPipelineWrapping:
    def test_round_trip(self, fitted_model):
        clf, scaler, _ = fitted_model
        clf_out, scaler_out = split_sklearn_pipeline(
            build_sklearn_pipeline(scaler, clf)
        )
        assert clf_out is clf
        assert scaler_out is scaler

    def test_pipeline_predicts_like_parts(self, fitted_model):
        clf, scaler, x = fitted_model
        pipe = build_sklearn_pipeline(scaler, clf)
        np.testing.assert_array_equal(pipe.predict(x), clf.predict(scaler.transform(x)))

    def test_split_rejects_non_pipeline(self, fitted_model):
        clf, _, _ = fitted_model
        with pytest.raises(ValueError):
            split_sklearn_pipeline(clf)


class TestRegistry:
    def test_requires_active_run(self, fitted_model, local_mlflow):
        clf, scaler, _ = fitted_model
        with pytest.raises(RuntimeError):
            log_and_register(clf, scaler, {}, promote=False)

    def test_register_promote_and_load(self, fitted_model, local_mlflow):
        clf, scaler, x = fitted_model
        metadata = {"feature_method": "hog", "feature_dim": 6, "classifier": "svm"}

        with mlflow.start_run():
            first = log_and_register(clf, scaler, metadata, promote=True)
        with mlflow.start_run():
            second = log_and_register(clf, scaler, metadata, promote=False)

        assert first["model_version"] == 1
        assert second["model_version"] == 2
        client = mlflow.MlflowClient()
        champion = client.get_model_version_by_alias(
            first["model_name"], PRODUCTION_ALIAS
        )
        assert int(champion.version) == 1

        clf_l, scaler_l, meta_l, version = load_from_registry(
            f"models:/{first['model_name']}@{PRODUCTION_ALIAS}", local_mlflow
        )
        assert meta_l["feature_dim"] == 6
        assert version == "1"
        np.testing.assert_array_equal(
            clf_l.predict(scaler_l.transform(x)), clf.predict(scaler.transform(x))
        )


class TestResolveModelUri:
    def test_non_alias_uri_is_unchanged(self):
        assert resolve_model_uri("models:/m/3") == ("models:/m/3", None)
        assert resolve_model_uri("runs:/abc/model") == ("runs:/abc/model", None)


class TestPromotionRule:
    def test_below_gate_is_never_promoted(self):
        assert not should_promote(0.85, 0.90, champion_acc=None)

    def test_first_model_above_gate_becomes_champion(self):
        assert should_promote(0.91, 0.90, champion_acc=None)

    def test_challenger_worse_than_champion_is_rejected(self):
        assert not should_promote(0.9006, 0.90, champion_acc=0.9293)

    def test_challenger_matching_champion_is_promoted(self):
        assert should_promote(0.93, 0.90, champion_acc=0.93)

    def test_champion_accuracy_reads_run_metric(self, fitted_model, local_mlflow):
        clf, scaler, _ = fitted_model
        assert champion_accuracy() is None
        with mlflow.start_run():
            mlflow.log_metric("accuracy", 0.95)
            log_and_register(clf, scaler, {}, promote=True)
        assert champion_accuracy() == pytest.approx(0.95)
