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

## MLflow tracking

Install the updated environment, then run the complete workflow:

```bash
pip install -e .
python scripts/run_workflow.py
```

This creates one parent run named `housing-workflow` with nested child runs for data ingestion, model training, and model scoring. The default tracking store is a local `mlflow.db` SQLite database. You can use another store or an MLflow server with `--tracking-uri` and select an experiment with `--experiment-name`.

Launch the local MLflow UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, then run the workflow in another terminal. The walkthrough notebook is [02-mlflow-walkthrough.ipynb](notebooks/reference/02-mlflow-walkthrough.ipynb).

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
