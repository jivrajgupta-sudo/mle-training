"""MLflow configuration and run helpers."""

from __future__ import annotations

from pathlib import Path

import mlflow


DEFAULT_EXPERIMENT = "housing-project"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> None:
    """Configure the MLflow tracking destination and experiment.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment that receives the run.
    tracking_uri : str, optional
        MLflow server URI or local file-backed URI.
    """
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def log_path_parameters(**paths: str | Path) -> None:
    """Log filesystem paths as MLflow parameters."""
    mlflow.log_params({name: str(path) for name, path in paths.items()})


def log_file_artifact(path: str | Path) -> None:
    """Log a file when it exists, without failing the workflow otherwise."""
    path = Path(path)
    if path.exists():
        mlflow.log_artifact(str(path))
