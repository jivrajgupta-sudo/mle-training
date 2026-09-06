"""CLI for scoring the housing price model."""

from __future__ import annotations

import argparse
from pathlib import Path

from housing_project.logging_config import configure_logging
from housing_project.scoring import score_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/housing_model.pkl"))
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/validation.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/scores.json"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-console-log", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level, args.log_path, not args.no_console_log)
    score_model(args.model_path, args.data_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
