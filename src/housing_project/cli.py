"""Command-line entry points for the housing workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow

from .data_ingestion import ingest_data
from .logging_config import configure_logging
from .modeling import train_model
from .scoring import score_model
from .tracking import (
    configure_mlflow,
    log_file_artifact,
    log_path_parameters,
)

LOGGER = logging.getLogger(__name__)


def _logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-console-log", action="store_true")


def _mlflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="housing-project")
    parser.add_argument("--run-name", default=None)


def _start_run(args: argparse.Namespace, default_name: str):
    configure_mlflow(args.experiment_name, args.tracking_uri)
    return mlflow.start_run(run_name=args.run_name or default_name)


def _configure(args: argparse.Namespace) -> None:
    configure_logging(args.log_level, args.log_path, not args.no_console_log)


def ingest_main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare housing data.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed"))
    _logging_arguments(parser)
    _mlflow_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    with _start_run(args, "data-ingestion"):
        log_path_parameters(raw_path=args.raw_path, processed_path=args.output_path)
        train_path, validation_path = ingest_data(args.raw_path, args.output_path)
        mlflow.log_params({
            "train_rows": sum(1 for _ in train_path.open(encoding="utf-8")) - 1,
            "validation_rows": sum(1 for _ in validation_path.open(encoding="utf-8")) - 1,
        })
        log_file_artifact(train_path)
        log_file_artifact(validation_path)
    return 0


def train_main() -> int:
    parser = argparse.ArgumentParser(description="Train the housing price model.")
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/housing_model.pkl"))
    _logging_arguments(parser)
    _mlflow_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    with _start_run(args, "model-training"):
        log_path_parameters(dataset_path=args.input_path, model_path=args.output_path)
        artifact_path = train_model(args.input_path, args.output_path)
        training_rows = sum(1 for _ in args.input_path.open(encoding="utf-8")) - 1
        mlflow.log_param("training_rows", training_rows)
        log_file_artifact(artifact_path)
    return 0


def score_main() -> int:
    parser = argparse.ArgumentParser(description="Score the housing price model.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/housing_model.pkl"))
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/validation.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/scores.json"))
    _logging_arguments(parser)
    _mlflow_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    with _start_run(args, "model-scoring"):
        log_path_parameters(
            model_path=args.model_path,
            dataset_path=args.data_path,
            score_path=args.output_path,
        )
        metrics = score_model(args.model_path, args.data_path, args.output_path)
        mlflow.log_metrics(metrics)
        log_file_artifact(args.output_path)
    return 0


def workflow_main() -> int:
    """Run all workflow stages as nested MLflow child runs."""
    parser = argparse.ArgumentParser(description="Run the complete housing workflow.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/housing_model.pkl"))
    parser.add_argument("--scores-path", type=Path, default=Path("artifacts/scores.json"))
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="housing-project")
    parser.add_argument("--run-name", default="housing-workflow")
    _logging_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    configure_mlflow(args.experiment_name, args.tracking_uri)
    with mlflow.start_run(run_name=args.run_name) as parent_run:
        log_path_parameters(
            raw_path=args.raw_path,
            processed_path=args.processed_path,
            model_path=args.model_path,
            scores_path=args.scores_path,
        )
        LOGGER.info("Started parent MLflow run %s", parent_run.info.run_id)
        with mlflow.start_run(run_name="data-ingestion", nested=True):
            train_path, validation_path = ingest_data(args.raw_path, args.processed_path)
            mlflow.log_params({
                "train_rows": sum(1 for _ in train_path.open(encoding="utf-8")) - 1,
                "validation_rows": sum(1 for _ in validation_path.open(encoding="utf-8")) - 1,
            })
            log_file_artifact(train_path)
            log_file_artifact(validation_path)
        with mlflow.start_run(run_name="model-training", nested=True):
            artifact_path = train_model(train_path, args.model_path)
            training_rows = sum(1 for _ in train_path.open(encoding="utf-8")) - 1
            mlflow.log_param("training_rows", training_rows)
            log_file_artifact(artifact_path)
        with mlflow.start_run(run_name="model-scoring", nested=True):
            metrics = score_model(args.model_path, validation_path, args.scores_path)
            mlflow.log_metrics(metrics)
            log_file_artifact(args.scores_path)
        mlflow.log_metrics(metrics)
    return 0
