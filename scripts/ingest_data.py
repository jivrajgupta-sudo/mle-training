"""CLI for downloading and preparing housing data."""

from __future__ import annotations

import argparse
from pathlib import Path

from housing_project.data_ingestion import ingest_data
from housing_project.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-console-log", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level, args.log_path, not args.no_console_log)
    ingest_data(args.raw_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
