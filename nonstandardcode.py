"""Compatibility entry point for the refactored workflow.

Use the scripts in ``scripts/`` for the individual ingestion, training, and
scoring stages.
"""

from housing_project.cli import ingest_main


if __name__ == "__main__":
    raise SystemExit(ingest_main())