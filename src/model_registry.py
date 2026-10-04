"""MLflow Model Registry helpers shared by training code and the serving API.

The OCR model is registered as a single sklearn ``Pipeline(scaler -> clf)`` so a
registered version is self-contained: the API can load it without any extra
files. Model metadata (feature method, classes, feature dim, metrics) travels
with the model via MLflow's ``metadata`` field.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

REGISTERED_MODEL_NAME = "lpr-ocr-classifier"
PRODUCTION_ALIAS = "production"
DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"

SCALER_STEP = "scaler"
CLASSIFIER_STEP = "clf"


def build_sklearn_pipeline(scaler, clf) -> Pipeline:
    """Wraps an already-fitted scaler and classifier into one Pipeline."""
    return Pipeline([(SCALER_STEP, scaler), (CLASSIFIER_STEP, clf)])


def split_sklearn_pipeline(model: Pipeline):
    """Returns ``(clf, scaler)`` from a Pipeline built by build_sklearn_pipeline."""
    try:
        return model.named_steps[CLASSIFIER_STEP], model.named_steps[SCALER_STEP]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "Registered model must be a Pipeline with "
            f"'{SCALER_STEP}' and '{CLASSIFIER_STEP}' steps"
        ) from exc


def log_and_register(
    clf,
    scaler,
    metadata: dict,
    promote: bool,
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> dict:
    """Logs the OCR model to the active MLflow run and registers it.

    Must be called inside ``mlflow.start_run()``. When ``promote`` is True the
    new version receives ``alias`` (default "production"), which is what the
    API loads when ``MODEL_SOURCE=mlflow``.
    """
    import mlflow
    import mlflow.sklearn

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_and_register() requires an active MLflow run")

    model_info = mlflow.sklearn.log_model(
        build_sklearn_pipeline(scaler, clf),
        artifact_path="ocr_model",
        registered_model_name=model_name,
        metadata=metadata,
    )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(
        f"name='{model_name}' and run_id='{run.info.run_id}'"
    )
    if not versions:
        raise RuntimeError(f"Model version for run {run.info.run_id} not found")
    version = max(int(v.version) for v in versions)

    if promote:
        client.set_registered_model_alias(model_name, alias, str(version))
        mlflow.set_tag("promoted_alias", alias)

    return {
        "model_name": model_name,
        "model_version": version,
        "model_uri": model_info.model_uri,
        "promoted": promote,
    }


def champion_accuracy(
    model_name: str = REGISTERED_MODEL_NAME, alias: str = PRODUCTION_ALIAS
) -> float | None:
    """Accuracy logged by the run behind ``@alias``, or None without a champion."""
    import mlflow
    from mlflow.exceptions import MlflowException

    client = mlflow.MlflowClient()
    try:
        champion = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None
    return client.get_run(champion.run_id).data.metrics.get("accuracy")


def should_promote(
    accuracy: float, min_accuracy: float, champion_acc: float | None
) -> bool:
    """Champion/challenger rule shared by the Airflow DAG and manual training:
    promote only above the quality gate and when not worse than the champion."""
    if accuracy < min_accuracy:
        return False
    return champion_acc is None or accuracy >= champion_acc


def resolve_model_uri(model_uri: str) -> tuple[str, str | None]:
    """Turns ``models:/name@alias`` into ``(models:/name/<version>, version)``.

    ``mlflow.models.get_model_info`` does not accept alias URIs in MLflow 2.x,
    and pinning the version also guarantees model and metadata come from the
    same version even if the alias moves mid-load.
    """
    import mlflow

    prefix = "models:/"
    if not model_uri.startswith(prefix) or "@" not in model_uri:
        return model_uri, None
    name, alias = model_uri[len(prefix) :].split("@", 1)
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
    return f"{prefix}{name}/{version}", str(version)


def load_from_registry(model_uri: str = DEFAULT_MODEL_URI, tracking_uri: str = None):
    """Loads ``(clf, scaler, metadata, version)`` from the MLflow registry."""
    import mlflow
    import mlflow.sklearn

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    resolved_uri, version = resolve_model_uri(model_uri)
    model = mlflow.sklearn.load_model(resolved_uri)
    info = mlflow.models.get_model_info(resolved_uri)
    clf, scaler = split_sklearn_pipeline(model)
    metadata = dict(info.metadata or {})
    return clf, scaler, metadata, version
