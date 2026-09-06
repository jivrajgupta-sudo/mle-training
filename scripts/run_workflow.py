"""Run ingestion, training, and scoring under nested MLflow runs."""

from housing_project.cli import workflow_main


if __name__ == "__main__":
    raise SystemExit(workflow_main())
