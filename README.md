# Housing Project

This project refactors the California housing example into an installable Python package with separate ingestion, training, scoring, tests, and documentation workflows.

## Setup

```bash
conda env create -f env.yaml
conda activate housing-project
pip install -e .
```

## Workflow

Run these commands from the project root:

```bash
python scripts/ingest_data.py
python scripts/train.py
python scripts/score.py
```

The defaults are:

- Raw data: `data/raw`
- Processed datasets: `data/processed`
- Models and scores: `artifacts`
- Optional logs: `logs`

Each command accepts `--log-level`, `--log-path`, and `--no-console-log`. Use `--help` to see all path arguments.

Installed command equivalents are `housing-ingest`, `housing-train`, and `housing-score`.

## Testing

```bash
pytest -v
pytest --cov=housing_project --cov-report=term-missing
```

## Documentation

Build the HTML documentation with:

```bash
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` after the build.

## Layout

The reusable package is under `src/housing_project`. Unit and functional tests are kept under `tests`, generated data and artifacts are ignored by Git, and deployment configurations are under `deploy`.
