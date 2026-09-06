"""Command-line entry points for the housing workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_ingestion import ingest_data
from .logging_config import configure_logging
from .modeling import train_model
from .scoring import score_model


def _logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-console-log", action="store_true")


def _configure(args: argparse.Namespace) -> None:
    configure_logging(args.log_level, args.log_path, not args.no_console_log)


def ingest_main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare housing data.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed"))
    _logging_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    ingest_data(args.raw_path, args.output_path)
    return 0


def train_main() -> int:
    parser = argparse.ArgumentParser(description="Train the housing price model.")
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/housing_model.pkl"))
    _logging_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    train_model(args.input_path, args.output_path)
    return 0


def score_main() -> int:
    parser = argparse.ArgumentParser(description="Score the housing price model.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/housing_model.pkl"))
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/validation.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/scores.json"))
    _logging_arguments(parser)
    args = parser.parse_args()
    _configure(args)
    score_model(args.model_path, args.data_path, args.output_path)
    return 0
