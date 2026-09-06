"""Model training and persistence."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LOGGER = logging.getLogger(__name__)


def build_model() -> Pipeline:
    """Build the preprocessing and random forest pipeline."""
    numeric_features = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
    ]
    categorical_features = ["ocean_proximity"]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)),
    ])


def train_model(dataset_path: str | Path, artifact_path: str | Path) -> Path:
    """Train a model and save it as a pickle artifact."""
    dataset = pd.read_csv(dataset_path)
    features = dataset.drop(columns="median_house_value")
    labels = dataset["median_house_value"]
    model = build_model()
    model.fit(features, labels)
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as file:
        pickle.dump(model, file)
    LOGGER.info("Saved model to %s", artifact_path)
    return artifact_path


def load_model(artifact_path: str | Path) -> Pipeline:
    """Load a persisted model artifact."""
    with Path(artifact_path).open("rb") as file:
        return pickle.load(file)
