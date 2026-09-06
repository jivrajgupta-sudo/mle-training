from pathlib import Path

import pandas as pd

from housing_project.data_ingestion import create_datasets
from housing_project.modeling import train_model
from housing_project.scoring import score_model


def sample_data(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [-122.0 + index * 0.01 for index in range(rows)],
            "latitude": [37.0 + index * 0.01 for index in range(rows)],
            "housing_median_age": [20 + index % 10 for index in range(rows)],
            "total_rooms": [1000 + index * 10 for index in range(rows)],
            "total_bedrooms": [200 + index for index in range(rows)],
            "population": [500 + index * 5 for index in range(rows)],
            "households": [100 + index for index in range(rows)],
            "median_income": [2.0 + index % 5 for index in range(rows)],
            "median_house_value": [100000 + index * 1000 for index in range(rows)],
            "ocean_proximity": ["<1H OCEAN", "INLAND"] * (rows // 2),
        }
    )


def test_create_datasets_writes_stratified_files(tmp_path: Path) -> None:
    raw_path = tmp_path / "housing.csv"
    sample_data().to_csv(raw_path, index=False)
    train_path, validation_path = create_datasets(raw_path, tmp_path / "processed")
    assert train_path.exists()
    assert validation_path.exists()
    assert len(pd.read_csv(train_path)) + len(pd.read_csv(validation_path)) == 20


def test_train_and_score(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.csv"
    sample_data().to_csv(dataset_path, index=False)
    model_path = train_model(dataset_path, tmp_path / "artifacts" / "model.pkl")
    metrics = score_model(model_path, dataset_path, tmp_path / "artifacts" / "scores.json")
    assert model_path.exists()
    assert metrics["rmse"] >= 0
    assert (tmp_path / "artifacts" / "scores.json").exists()
