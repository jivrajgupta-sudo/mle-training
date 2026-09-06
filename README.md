# Housing Project

This project refactors the California housing example into an installable Python package with separate ingestion, training, scoring, tests, and documentation workflows.

## Setup

```bash
conda env create -f env.yml
conda activate housing-project
pip install -e .
```

## Build distribution archives

Install the build tool and create both shareable Python distribution formats:

```bash
python -m pip install --upgrade build
python -m build
```

This creates a wheel and a source archive in `dist/`:

- `housing_project-0.3.0-py3-none-any.whl`
- `housing_project-0.3.0.tar.gz`

Install the wheel in a clean environment with:

```bash
python -m pip install dist/housing_project-0.3.0-py3-none-any.whl
housing-ingest --help
housing-train --help
housing-score --help
```

The source archive can be installed with `python -m pip install dist/housing_project-0.3.0.tar.gz`.

For the final deployment submission, include the wheel, source archive, `env.yml`, `README.md`, and the `dist/` installation instructions in a ZIP. Do not include local datasets, logs, MLflow databases, virtual environments, or generated model artifacts.

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
