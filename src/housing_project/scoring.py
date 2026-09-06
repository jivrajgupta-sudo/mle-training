"""Model evaluation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .modeling import load_model

LOGGER = logging.getLogger(__name__)


def score_model(model_path: str | Path, dataset_path: str | Path, output_path: str | Path) -> dict[str, float]:
    """Score a saved model and write regression metrics as JSON."""
    dataset = pd.read_csv(dataset_path)
    model = load_model(model_path)
    features = dataset.drop(columns="median_house_value")
    labels = dataset["median_house_value"]
    predictions = model.predict(features)
    metrics = {
        "rmse": float(mean_squared_error(labels, predictions) ** 0.5),
        "mae": float(mean_absolute_error(labels, predictions)),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote scores to %s", output_path)
    return metrics
